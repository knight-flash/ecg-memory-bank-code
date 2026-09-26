#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CODE-15 Chagas prototype experiment.

Purpose
-------
Train the binary case-linked prototype head evaluated on CODE-15 Chagas cases.

Method
------
Use the frozen CODE-15 embeddings and binary labels, make seeded
train/validation/test splits, and compare traditional, signed-vote, and
linear-aggregation branches.

Inputs
------
CODE-15 embedding tensor, Chagas labels, metadata/exams CSV and external
output root.

Outputs
-------
External branch metrics, CI, checkpoint, prototype plan, and projected case
anchors.

Paper parameters
----------------
Formal runner fixes budget=3000, max 30 per class, seed=42 and bootstrap=1000;
the standalone file retains smaller historical defaults.

Shapes and meaning
------------------
Input targets are [N,1] and prediction scores [N,1]; the projected bank has at
most 30 anchors for the one target class.
"""

from __future__ import annotations

import os

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from confidence_weighted_experiment_utils import (
    build_binary_target_labels,
    run_three_branch_eval_only_ci,
    summarize_split_counts,
    train_three_branch_binary_experiment,
    write_label_manifest,
    write_prototype_plan,
)
from mimic18_fixed_helper import build_adaptive_prototype_count_plan


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BASE_DIR.parent


def load_dataset_base():
    module_path = BASE_DIR / "prototype_code15.py"
    spec = importlib.util.spec_from_file_location("prototype_all_code15_chagas_dataset_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_base():
    module_path = BASE_DIR / "prototype_weighted_retrieval.py"
    spec = importlib.util.spec_from_file_location("prototype_all_code15_chagas_training_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training base from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA_BASE = load_dataset_base()
BASE = load_training_base()
EXPERIMENT_NAME = "code15_chagas_confidence_weighted_binary"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / EXPERIMENT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype_all CODE15 Chagas confidence-weighted binary ProtoPNet.")
    parser.add_argument("--code15-ecg-path", default=DATA_BASE.CODE15_ECG_PATH)
    parser.add_argument("--code15-labels-path", default=DATA_BASE.CODE15_LABELS_PATH)
    parser.add_argument("--code15-meta-path", default=DATA_BASE.CODE15_META_PATH)
    parser.add_argument("--code15-exams-path", default="__ECG_DATA_ROOT_REQUIRED__/cinc/code-15/exams.csv")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=BASE.RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--prototype-budget", type=int, default=128)
    parser.add_argument("--min-prototypes-per-class", type=int, default=1)
    parser.add_argument("--max-prototypes-per-class", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--eval-only-ci", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_BASE.apply_runtime_config(
        argparse.Namespace(
            code15_ecg_path=args.code15_ecg_path,
            code15_labels_path=args.code15_labels_path,
            code15_meta_path=args.code15_meta_path,
            output_root=args.output_root,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=args.seed,
        )
    )
    BASE.set_seed(args.seed)
    device = BASE.resolve_device(args.device)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output_root / "checkpoint"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_root / "summary.json"
    label_manifest_path = output_root / "target_label_manifest.json"
    prototype_plan_csv_path = output_root / "prototype_count_plan.csv"

    x_all, y_all_multilabel, _metadata_df = DATA_BASE.load_dataset()
    original_labels = list(DATA_BASE.TARGET_LABELS)
    task_payload = build_binary_target_labels(BASE, "Chagas disease")
    chagas_idx = original_labels.index("Chagas disease")
    y_all = y_all_multilabel[:, chagas_idx:chagas_idx + 1].float()

    indices = np.arange(len(x_all))
    test_ratio = 1.0 - float(args.train_ratio) - float(args.val_ratio)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=float(args.val_ratio) + float(test_ratio),
        random_state=int(args.seed),
        stratify=y_all.view(-1).numpy(),
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=float(test_ratio) / float(args.val_ratio + test_ratio),
        random_state=int(args.seed),
        stratify=y_all[temp_idx].view(-1).numpy(),
    )

    x_train = x_all[train_idx].float()
    y_train = y_all[train_idx].float()
    x_val = x_all[val_idx].float()
    y_val = y_all[val_idx].float()
    x_test = x_all[test_idx].float()
    y_test = y_all[test_idx].float()

    split_counts = summarize_split_counts(
        total_code15_samples=len(x_all),
        train_count=len(x_train),
        val_count=len(x_val),
        test_count=len(x_test),
    )
    label_selection = {
        "strategy": "binary_single_label",
        "selected_label_count": 1,
        "selected_labels": ["Chagas disease"],
        "selected_original_indices": [0],
    }
    write_label_manifest(
        label_manifest_path,
        task_payload=task_payload,
        label_selection=label_selection,
        split_counts=split_counts,
        extra_payload={"data_source": "code15_chagas_binary"},
    )

    prototype_counts = build_adaptive_prototype_count_plan(
        positive_counts=y_train.sum(dim=0).cpu().numpy().astype(int).tolist(),
        total_budget=int(args.prototype_budget),
        min_protos=int(args.min_prototypes_per_class),
        max_protos=int(args.max_prototypes_per_class),
    )
    write_prototype_plan(
        prototype_plan_csv_path,
        manifest_rows=task_payload["manifest_rows"],
        prototype_counts=prototype_counts,
        y_train=y_train,
    )

    if args.eval_only_ci:
        run_three_branch_eval_only_ci(
            base_module=BASE,
            task_name="code15_chagas",
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            x_test=x_test,
            y_test=y_test,
            device=device,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
        return

    train_three_branch_binary_experiment(
        base_module=BASE,
        task_name="code15_chagas",
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_idx_original=np.asarray(train_idx, dtype=np.int64),
        device=device,
        max_epochs=int(args.max_epochs),
        label_selection=label_selection,
        split_protocol={
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(test_ratio),
        },
        manifest_rows=task_payload["manifest_rows"],
        prototype_counts=prototype_counts,
        summary_json_path=summary_json_path,
        extra_summary_payload={"data_source": "code15_chagas_binary"},
    )


if __name__ == "__main__":
    main()
