#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedicated MIMIC external evaluation for the APE-to-MIMIC 18-label experiment.

This script reuses the trained APE checkpoints and evaluates them on:

1. the full external MIMIC set
2. a MIMIC aligned-test subset generated with the same split logic as
   `mimic18_confidence_weighted_multilabel.py`

Both views export split-specific per-class CSVs and bootstrap CI files.
"""

from __future__ import annotations

import os

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ci_bootstrap_utils import (
    build_multilabel_ci_report,
    save_json,
    write_multilabel_ci_outputs,
)
from confidence_weighted_experiment_utils import (
    _build_full_bank_linear_from_checkpoint,
    _build_single_label_from_checkpoint,
    _build_traditional_from_checkpoint,
    evaluate_multilabel_benchmark_compatible,
)
from mimic18_fixed_helper import (
    build_label_selection_reporting,
    configure_base_target_labels,
)
from mimic18_split_utils import build_mimic18_split_indices


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BASE_DIR.parent
EXPERIMENT_NAME = "ape_mimic18_confidence_weighted_multilabel"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / EXPERIMENT_NAME
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_ROOT / "mimic_external_eval_summary.json"


def load_dataset_base():
    module_path = BASE_DIR / "prototype_heedb.py"
    spec = importlib.util.spec_from_file_location("prototype_all_ape_dataset_base_for_external_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_base():
    module_path = BASE_DIR / "prototype_weighted_retrieval.py"
    spec = importlib.util.spec_from_file_location("prototype_all_ape_training_base_for_external_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training base from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA_BASE = load_dataset_base()
BASE = load_training_base()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedicated external MIMIC evaluation for the APE-to-MIMIC experiment.")
    parser.add_argument("--data-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_ecg_embeddings.pt")
    parser.add_argument("--data-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_texts_with_labels_correct.csv")
    parser.add_argument("--data-label-col", default="labels_correct")
    parser.add_argument("--heedb-train-export-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_texts_with_labels_correct.csv")
    parser.add_argument("--val-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/val/heedb_ecg_embeddings.pt")
    parser.add_argument("--val-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/val/heedb_texts_with_labels_correct.csv")
    parser.add_argument("--val-label-col", default="labels_correct")
    parser.add_argument("--test-robust-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_ecg_embeddings.pt")
    parser.add_argument("--test-robust-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv")
    parser.add_argument("--test-robust-label-col", default="labels_correct")
    parser.add_argument("--heedb-test-export-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv")
    parser.add_argument("--mimic-ecg-path", default=DATA_BASE.TEST_MIMIC_ECG_PATH)
    parser.add_argument("--mimic-meta-path", default=DATA_BASE.TEST_MIMIC_META_PATH)
    parser.add_argument("--mimic-label-col", default=DATA_BASE.TEST_MIMIC_COL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=BASE.RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--branch",
        choices=["traditional_best", "single_label_best", "full_bank_linear_best", "all"],
        default="single_label_best",
        help="Which trained branch to evaluate.",
    )
    parser.add_argument("--save-summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--skip-ci", action="store_true")
    return parser.parse_args()


def evaluate_split(
    *,
    model,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    split_name: str,
    branch_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, torch.Tensor]:
    loader = BASE.make_dataloader(x_eval, y_eval, BASE.EVAL_BATCH_SIZE, shuffle=False)
    logits = BASE.collect_logits(model, loader, device)
    metrics, per_class_df = evaluate_multilabel_benchmark_compatible(
        BASE,
        y_eval,
        logits,
        BASE.TARGET_LABELS,
        f"{branch_name} on {split_name}",
    )
    return metrics, per_class_df, logits


def export_ci_for_split(
    *,
    ci_output_dir: Path,
    branch_name: str,
    split_name: str,
    y_true: torch.Tensor,
    logits: torch.Tensor,
    bootstrap: int,
    seed: int,
) -> dict[str, str]:
    report_payload = build_multilabel_ci_report(
        class_names=list(BASE.TARGET_LABELS),
        y_true=y_true.detach().cpu().numpy(),
        y_score=torch.sigmoid(logits).detach().cpu().numpy(),
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    report_payload["metric_key"] = branch_name
    report_payload["eval_split_name"] = split_name
    return write_multilabel_ci_outputs(
        output_dir=ci_output_dir,
        metric_key=f"{branch_name}_{split_name}",
        report_payload=report_payload,
    )


def main() -> None:
    args = parse_args()
    DATA_BASE.apply_runtime_config(args)
    configure_base_target_labels(DATA_BASE)
    task_payload = configure_base_target_labels(BASE)
    BASE.set_seed(args.seed)
    device = BASE.resolve_device(args.device)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output_root / "checkpoint"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_root / Path(args.save_summary_json).name
    ci_output_dir = output_root / "ci"
    ci_output_dir.mkdir(parents=True, exist_ok=True)

    x_train_all, raw_y_train, _metadata_train_df = DATA_BASE.load_and_standardize(
        args.data_ecg_path, args.data_meta_path, args.data_label_col
    )
    y_train = DATA_BASE.build_ground_truth_matrix(raw_y_train).float()

    x_val_raw, raw_y_val, _metadata_val_df = DATA_BASE.load_and_standardize(
        args.val_ecg_path, args.val_meta_path, args.val_label_col
    )
    y_val = DATA_BASE.build_ground_truth_matrix(raw_y_val).float()

    x_test_raw, raw_y_test, _metadata_test_df = DATA_BASE.load_and_standardize(
        args.test_robust_ecg_path, args.test_robust_meta_path, args.test_robust_label_col
    )
    y_test = DATA_BASE.build_ground_truth_matrix(raw_y_test).float()

    x_mimic_raw, raw_y_mimic, mimic_df = DATA_BASE.load_and_standardize(
        args.mimic_ecg_path, args.mimic_meta_path, args.mimic_label_col
    )
    y_mimic = DATA_BASE.build_ground_truth_matrix(raw_y_mimic).float()
    x_mimic = x_mimic_raw.float()

    split_indices = build_mimic18_split_indices(
        len(x_mimic),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    aligned_test_idx = split_indices.test_idx
    x_mimic_aligned_test = x_mimic[aligned_test_idx].float()
    y_mimic_aligned_test = y_mimic[aligned_test_idx].float()

    label_selection = build_label_selection_reporting(
        task_payload,
        {
            "train": y_train,
            "val": y_val,
            "heedb_internal_test": y_test,
            "mimic_external_full": y_mimic,
            "mimic_external_aligned_test": y_mimic_aligned_test,
        },
    )

    branch_specs = {
        "traditional_best": _build_traditional_from_checkpoint,
        "single_label_best": _build_single_label_from_checkpoint,
        "full_bank_linear_best": _build_full_bank_linear_from_checkpoint,
    }
    checkpoint_by_branch = {
        "traditional_best": checkpoint_root / "ape_mimic18_traditional_best.pt",
        "single_label_best": checkpoint_root / "ape_mimic18_single_label_best.pt",
        "full_bank_linear_best": checkpoint_root / "ape_mimic18_full_bank_linear_best.pt",
    }

    pd.DataFrame({"mimic_index": aligned_test_idx.tolist()}).to_csv(
        output_root / "mimic_external_aligned_test_indices.csv",
        index=False,
    )

    selected_branches = list(branch_specs.keys()) if args.branch == "all" else [args.branch]

    branch_results: dict[str, Any] = {}
    ci_reports: dict[str, Any] = {}
    for branch_name in selected_branches:
        builder = branch_specs[branch_name]
        checkpoint_path = checkpoint_by_branch[branch_name]
        if not checkpoint_path.exists():
            branch_results[branch_name] = {
                "status": "missing_checkpoint",
                "checkpoint": str(checkpoint_path),
            }
            continue

        model = builder(BASE, checkpoint_path, device)
        branch_results[branch_name] = {
            "checkpoint": str(checkpoint_path),
            "splits": {},
        }
        ci_reports[branch_name] = {"checkpoint": str(checkpoint_path), "splits": {}}

        for split_name, x_eval, y_eval in (
            ("mimic_external_full", x_mimic, y_mimic),
            ("mimic_external_aligned_test", x_mimic_aligned_test, y_mimic_aligned_test),
        ):
            metrics, per_class_df, logits = evaluate_split(
                model=model,
                x_eval=x_eval,
                y_eval=y_eval,
                split_name=split_name,
                branch_name=branch_name,
                device=device,
            )
            per_class_csv_path = output_root / f"per_class_auc_{split_name}_{branch_name}.csv"
            per_class_df.to_csv(per_class_csv_path, index=False)

            branch_results[branch_name]["splits"][split_name] = {
                "metrics": metrics,
                "per_class_csv": str(per_class_csv_path),
            }
            if not args.skip_ci:
                ci_reports[branch_name]["splits"][split_name] = export_ci_for_split(
                    ci_output_dir=ci_output_dir,
                    branch_name=branch_name,
                    split_name=split_name,
                    y_true=y_eval,
                    logits=logits,
                    bootstrap=int(args.bootstrap),
                    seed=int(args.seed),
                )

    if not args.skip_ci:
        save_json(
            ci_output_dir / "aggregate_summary_ci_external_eval.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "evaluation_type": "mimic_external_only",
                "bootstrap_iterations": int(args.bootstrap),
                "seed": int(args.seed),
                "reports": ci_reports,
            },
        )

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_type": "mimic_external_only",
        "data_source": "heedb_to_mimic",
        "selected_branch": args.branch,
        "split_protocol": {
            "mimic_self_reference": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
                "seed": int(args.seed),
            }
        },
        "label_count": len(BASE.TARGET_LABELS),
        "label_selection": label_selection,
        "splits": {
            "mimic_external_full": {
                "sample_count": int(len(x_mimic)),
            },
            "mimic_external_aligned_test": {
                "sample_count": int(len(x_mimic_aligned_test)),
                "aligned_to": "mimic18_confidence_weighted_multilabel test split logic",
                "aligned_test_index_csv": str(output_root / "mimic_external_aligned_test_indices.csv"),
            },
        },
        "metrics": branch_results,
    }
    if not args.skip_ci:
        payload["ci"] = {
            "output_dir": str(ci_output_dir),
            "aggregate_summary_json": str(ci_output_dir / "aggregate_summary_ci_external_eval.json"),
        }
    save_json(summary_json_path, payload)


if __name__ == "__main__":
    main()
