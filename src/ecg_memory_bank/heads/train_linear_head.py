#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified embedding-level linear baseline for Prototype_all experiments.

This script adds a plain `nn.Linear` baseline on top of ECG embeddings and
keeps dataset preparation as close as possible to the existing ProtoPNet
wrappers in this folder.
"""

from __future__ import annotations

import os

import argparse
import copy
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ci_bootstrap_utils import build_multilabel_ci_report, save_json, write_multilabel_ci_outputs
from confidence_weighted_experiment_utils import (
    build_binary_target_labels,
    evaluate_multilabel_benchmark_compatible,
    summarize_split_counts,
    write_label_manifest,
)
from mimic18_split_utils import build_mimic18_split_indices


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BASE_DIR.parent
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / "embedding_linear_baseline"
BRANCH_NAME = "embedding_linear_best"
MODULE_CACHE: dict[str, ModuleType] = {}


def load_local_module(cache_key: str, filename: str) -> ModuleType:
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]
    module_path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULE_CACHE[cache_key] = module
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified embedding-level linear baseline for Prototype_all.")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "mimic100",
            "ape120",
            "code15",
            "code15_chagas",
        ],
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau"], default="plateau")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-only-ci", action="store_true")
    parser.add_argument("--ci-only-split", default="", help="Optionally restrict eval-only CI export to a single split.")
    parser.add_argument("--external-eval", action="store_true")
    parser.add_argument("--skip-ci", action="store_true")

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
    parser.add_argument("--mimic-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/feature/mimic/ecg_embeddings.pt")
    parser.add_argument("--mimic-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/feature/mimic/metadata_cleaned_with_labels_correct.csv")
    parser.add_argument("--mimic-label-col", default="labels_correct")

    parser.add_argument("--code15-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/code15_ecgclip_embeddings.pt")
    parser.add_argument("--code15-labels-path", default="__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/code15_labels.pt")
    parser.add_argument("--code15-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/metadata_code15_ecgclip.csv")
    parser.add_argument("--code15-exams-path", default="__ECG_DATA_ROOT_REQUIRED__/cinc/code-15/exams.csv")

    parser.add_argument("--label-config-csv", default=str(Path(__file__).resolve().parents[3] / "configs" / "l3_full_label_curated.csv"))
    return parser.parse_args()


def compute_default_output_root(args: argparse.Namespace) -> Path:
    return DEFAULT_OUTPUT_ROOT / args.task


def resolve_output_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root) if args.output_root else compute_default_output_root(args)


def set_seed_on_module(base_module: ModuleType, seed: int) -> None:
    if hasattr(base_module, "set_seed"):
        base_module.set_seed(int(seed))
        return
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_device(base_module: ModuleType, device_arg: str) -> torch.device:
    if hasattr(base_module, "resolve_device"):
        return base_module.resolve_device(device_arg)
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def make_dataloader(base_module: ModuleType, x_tensor: torch.Tensor, y_tensor: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    if hasattr(base_module, "make_dataloader"):
        return base_module.make_dataloader(x_tensor, y_tensor, batch_size, shuffle=shuffle)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def print_prompt_plan_if_available(base_module: ModuleType) -> None:
    if hasattr(base_module, "print_classification_prompt_plan"):
        base_module.print_classification_prompt_plan()


def build_alias_sets(base_module: ModuleType, labels: list[str]) -> list[set[str]]:
    normalize = getattr(base_module, "normalize_label_text")
    return [{normalize(label)} for label in labels]


def configure_simple_target_labels(base_module: ModuleType, task_payload: dict[str, Any]) -> None:
    labels = list(task_payload["labels"])
    base_module.TARGET_LABELS = labels
    base_module.TARGET_PROMPTS = [f"An ECG recording showing {label}." for label in labels]
    base_module.TARGET_ALIAS_SETS = build_alias_sets(base_module, labels)
    base_module.TARGET_KEYWORDS = [sorted(alias_set) for alias_set in base_module.TARGET_ALIAS_SETS]
    normal_idx = task_payload.get("normal_idx")
    if normal_idx is None:
        base_module.NORMAL_LABEL = "Normal ECG"
        base_module.NORMAL_CLASS_IDX = -1
    else:
        base_module.NORMAL_LABEL = labels[int(normal_idx)]
        base_module.NORMAL_CLASS_IDX = int(normal_idx)


class EmbeddingLinearClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def collect_linear_logits(model: nn.Module, dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    logits_list: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in dataloader:
            logits_list.append(model(x_batch.to(device)).cpu())
    return torch.cat(logits_list, dim=0)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = int(x_batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def build_lr_scheduler(
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(max_epochs)),
            eta_min=float(args.min_lr),
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(args.plateau_factor),
        patience=int(args.plateau_patience),
        min_lr=float(args.min_lr),
    )


def print_eval_split_sizes(eval_splits: OrderedDict[str, dict[str, Any]]) -> None:
    print("\n[INFO] Evaluation split sizes:")
    for split_name, split_bundle in eval_splits.items():
        x_eval = split_bundle["x"]
        y_eval = split_bundle["y"]
        print(
            f"  - {split_name}: sample_count={len(x_eval)} "
            f"| feature_rows={int(x_eval.shape[0])} | label_rows={int(y_eval.shape[0])}"
        )


def evaluate_split(
    *,
    base_module: ModuleType,
    model: nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    split_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, torch.Tensor]:
    loader = make_dataloader(base_module, x_eval, y_eval, int(getattr(base_module, "EVAL_BATCH_SIZE", 8192)), shuffle=False)
    logits = collect_linear_logits(model, loader, device)
    metrics, per_class_df = evaluate_multilabel_benchmark_compatible(
        base_module,
        y_eval,
        logits,
        list(base_module.TARGET_LABELS),
        f"{BRANCH_NAME} on {split_name}",
    )
    return metrics, per_class_df, logits


def export_ci_for_split(
    *,
    ci_output_dir: Path,
    split_name: str,
    y_true: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    report_payload = build_multilabel_ci_report(
        class_names=class_names,
        y_true=y_true.detach().cpu().numpy(),
        y_score=torch.sigmoid(logits).detach().cpu().numpy(),
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    report_payload["metric_key"] = BRANCH_NAME
    report_payload["eval_split_name"] = split_name
    refs = write_multilabel_ci_outputs(
        output_dir=ci_output_dir,
        metric_key=f"{BRANCH_NAME}_{split_name}",
        report_payload=report_payload,
    )
    return refs, report_payload


def merge_ci_columns(per_class_df: pd.DataFrame, ci_csv_path: str | Path | None) -> pd.DataFrame:
    if not ci_csv_path:
        return per_class_df
    ci_df = pd.read_csv(ci_csv_path)
    merge_cols = ["class_name", "auroc", "ci95_lower", "ci95_upper", "valid_bootstraps"]
    available_cols = [col for col in merge_cols if col in ci_df.columns]
    if "class_name" not in available_cols:
        return per_class_df
    return per_class_df.merge(ci_df[available_cols], on="class_name", how="left", suffixes=("", "_ci"))


def save_best_model(path: Path, model: nn.Module, label_names: list[str]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_dim": int(model.classifier.in_features),
        "num_classes": int(model.classifier.out_features),
        "label_names": list(label_names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_best_model(path: Path, feature_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        feature_dim = int(payload.get("feature_dim", feature_dim))
        num_classes = int(payload.get("num_classes", num_classes))
    else:
        state_dict = payload
    model = EmbeddingLinearClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_runtime_args(args: argparse.Namespace, output_root: Path, checkpoint_root: Path) -> argparse.Namespace:
    runtime_args = copy.copy(args)
    runtime_args.output_root = str(output_root)
    runtime_args.checkpoint_dir = str(checkpoint_root)
    return runtime_args


def ensure_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = resolve_output_root(args)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else output_root / "checkpoint"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return output_root, checkpoint_root


def prepare_ape_mimic18_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_base = load_local_module("linear_ape_dataset_base", "prototype_heedb.py")
    base = load_local_module("linear_ape_training_base", "prototype_weighted_retrieval.py")
    helper = load_local_module("linear_mimic18_helper_for_ape", "mimic18_fixed_helper.py")

    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    data_base.apply_runtime_config(runtime_args)
    helper.configure_base_target_labels(data_base)
    task_payload = helper.configure_base_target_labels(base)
    set_seed_on_module(base, args.seed)

    x_train_all, raw_y_train, _ = data_base.load_and_standardize(args.data_ecg_path, args.data_meta_path, args.data_label_col)
    y_train = data_base.build_ground_truth_matrix(raw_y_train).float()
    x_train = x_train_all.float()
    x_val_raw, raw_y_val, _ = data_base.load_and_standardize(args.val_ecg_path, args.val_meta_path, args.val_label_col)
    y_val = data_base.build_ground_truth_matrix(raw_y_val).float()
    x_val = x_val_raw.float()
    x_test_raw, raw_y_test, _ = data_base.load_and_standardize(
        args.test_robust_ecg_path,
        args.test_robust_meta_path,
        args.test_robust_label_col,
    )
    y_test = data_base.build_ground_truth_matrix(raw_y_test).float()
    x_test = x_test_raw.float()
    x_mimic_raw, raw_y_mimic, _mimic_df = data_base.load_and_standardize(
        args.mimic_ecg_path,
        args.mimic_meta_path,
        args.mimic_label_col,
    )
    y_mimic = data_base.build_ground_truth_matrix(raw_y_mimic).float()
    x_mimic = x_mimic_raw.float()

    label_selection = helper.build_label_selection_reporting(
        task_payload,
        {
            "train": y_train,
            "val": y_val,
            "heedb_internal_test": y_test,
            "mimic_external_test": y_mimic,
        },
    )
    split_counts = summarize_split_counts(
        heedb_train_count=len(x_train),
        heedb_val_count=len(x_val),
        heedb_internal_test_count=len(x_test),
        mimic_external_test_count=len(x_mimic),
    )
    return {
        "task": args.task,
        "experiment_name": "ape_mimic18_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {"data_source": "heedb_to_mimic"},
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict(
            [
                ("heedb_internal_test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}}),
                ("mimic_external_test", {"x": x_mimic, "y": y_mimic, "summary": {"sample_count": int(len(x_mimic))}}),
            ]
        ),
        "primary_split_name": "mimic_external_test",
        "split_protocol": {"data_source": "heedb_to_mimic"},
        "data_source": "heedb_to_mimic",
    }


def prepare_ape_mimic18_external_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_base = load_local_module("linear_ape_dataset_base_ext", "prototype_heedb.py")
    base = load_local_module("linear_ape_training_base_ext", "prototype_weighted_retrieval.py")
    helper = load_local_module("linear_mimic18_helper_for_ape_ext", "mimic18_fixed_helper.py")

    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    data_base.apply_runtime_config(runtime_args)
    helper.configure_base_target_labels(data_base)
    task_payload = helper.configure_base_target_labels(base)
    set_seed_on_module(base, args.seed)

    x_train_all, raw_y_train, _ = data_base.load_and_standardize(args.data_ecg_path, args.data_meta_path, args.data_label_col)
    y_train = data_base.build_ground_truth_matrix(raw_y_train).float()
    x_val_raw, raw_y_val, _ = data_base.load_and_standardize(args.val_ecg_path, args.val_meta_path, args.val_label_col)
    y_val = data_base.build_ground_truth_matrix(raw_y_val).float()
    x_test_raw, raw_y_test, _ = data_base.load_and_standardize(
        args.test_robust_ecg_path,
        args.test_robust_meta_path,
        args.test_robust_label_col,
    )
    y_test = data_base.build_ground_truth_matrix(raw_y_test).float()
    x_mimic_raw, raw_y_mimic, _mimic_df = data_base.load_and_standardize(
        args.mimic_ecg_path,
        args.mimic_meta_path,
        args.mimic_label_col,
    )
    y_mimic = data_base.build_ground_truth_matrix(raw_y_mimic).float()
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
    pd.DataFrame({"mimic_index": aligned_test_idx.tolist()}).to_csv(
        output_root / "mimic_external_aligned_test_indices.csv",
        index=False,
    )

    label_selection = helper.build_label_selection_reporting(
        task_payload,
        {
            "train": y_train,
            "val": y_val,
            "heedb_internal_test": y_test,
            "mimic_external_full": y_mimic,
            "mimic_external_aligned_test": y_mimic_aligned_test,
        },
    )
    return {
        "task": args.task,
        "experiment_name": "ape_mimic18_embedding_linear_baseline",
        "summary_name": "mimic_external_eval_summary.json",
        "ci_summary_name": "aggregate_summary_ci_external_eval.json",
        "evaluation_type": "mimic_external_only",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": {},
        "label_manifest_extra": {"data_source": "heedb_to_mimic"},
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_mimic,
        "y_train": y_mimic,
        "x_val": x_mimic_aligned_test,
        "y_val": y_mimic_aligned_test,
        "eval_splits": OrderedDict(
            [
                ("mimic_external_full", {"x": x_mimic, "y": y_mimic, "summary": {"sample_count": int(len(x_mimic))}}),
                (
                    "mimic_external_aligned_test",
                    {
                        "x": x_mimic_aligned_test,
                        "y": y_mimic_aligned_test,
                        "summary": {
                            "sample_count": int(len(x_mimic_aligned_test)),
                            "aligned_to": "mimic18_confidence_weighted_multilabel test split logic",
                            "aligned_test_index_csv": str(output_root / "mimic_external_aligned_test_indices.csv"),
                        },
                    },
                ),
            ]
        ),
        "primary_split_name": "mimic_external_aligned_test",
        "load_checkpoint_only": True,
        "split_protocol": {
            "mimic_self_reference": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
                "seed": int(args.seed),
            }
        },
        "data_source": "heedb_to_mimic",
    }


def prepare_mimic18_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_base = load_local_module("linear_mimic18_dataset_base", "prototype_mimic.py")
    base = load_local_module("linear_mimic18_training_base", "prototype_weighted_retrieval.py")
    helper = load_local_module("linear_mimic18_helper", "mimic18_fixed_helper.py")

    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    data_base.apply_runtime_config(runtime_args)
    task_payload = helper.configure_base_target_labels(base)
    helper.configure_base_target_labels(data_base)
    set_seed_on_module(base, args.seed)

    x_all, raw_labels, _ = data_base.load_dataset(args.mimic_ecg_path, args.mimic_meta_path, args.mimic_label_col)
    y_all = data_base.build_ground_truth_matrix(raw_labels).float()
    split_indices = build_mimic18_split_indices(
        len(x_all),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    train_idx, val_idx, test_idx = split_indices.train_idx, split_indices.val_idx, split_indices.test_idx
    x_train = x_all[train_idx].float()
    y_train = y_all[train_idx].float()
    x_val = x_all[val_idx].float()
    y_val = y_all[val_idx].float()
    x_test = x_all[test_idx].float()
    y_test = y_all[test_idx].float()
    label_selection = helper.build_label_selection_reporting(task_payload, {"train": y_train, "val": y_val, "test": y_test})
    split_counts = summarize_split_counts(
        total_mimic_samples=len(x_all),
        train_count=len(x_train),
        val_count=len(x_val),
        test_count=len(x_test),
    )
    return {
        "task": args.task,
        "experiment_name": "mimic18_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {
            "data_source": "mimic_only",
            "split_protocol": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
            },
        },
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict([("test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}})]),
        "primary_split_name": "test",
        "split_protocol": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
        },
        "data_source": "mimic_only",
    }


def prepare_code15_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_base = load_local_module("linear_code15_dataset_base", "prototype_code15.py")
    base = load_local_module("linear_code15_training_base", "prototype_weighted_retrieval.py")
    task_module = load_local_module("linear_code15_task_module", "prototype_code15_multilabel.py")
    helper = load_local_module("linear_code15_label_helper", "mimic18_fixed_helper.py")

    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    data_base.apply_runtime_config(runtime_args)
    set_seed_on_module(base, args.seed)

    task_payload = task_module.build_task_payload()
    original_labels = list(data_base.TARGET_LABELS)
    configure_simple_target_labels(base, task_payload)

    x_all, y_all_full, _ = data_base.load_dataset()
    indices = np.arange(len(x_all))
    test_ratio = 1.0 - float(args.train_ratio) - float(args.val_ratio)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=float(args.val_ratio) + float(test_ratio),
        random_state=int(args.seed),
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=float(test_ratio) / float(args.val_ratio + test_ratio),
        random_state=int(args.seed),
    )
    x_train = x_all[train_idx].float()
    y_train = task_module.filter_label_matrix_to_task(y_all_full[train_idx].float(), original_labels, list(task_payload["labels"]))
    x_val = x_all[val_idx].float()
    y_val = task_module.filter_label_matrix_to_task(y_all_full[val_idx].float(), original_labels, list(task_payload["labels"]))
    x_test = x_all[test_idx].float()
    y_test = task_module.filter_label_matrix_to_task(y_all_full[test_idx].float(), original_labels, list(task_payload["labels"]))
    label_selection = helper.build_label_selection_reporting(task_payload, {"train": y_train, "val": y_val, "test": y_test})
    split_counts = summarize_split_counts(total_code15_samples=len(x_all), train_count=len(x_train), val_count=len(x_val), test_count=len(x_test))
    return {
        "task": args.task,
        "experiment_name": "code15_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {
            "data_source": "code15_only",
            "split_protocol": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(test_ratio),
            },
        },
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict([("test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}})]),
        "primary_split_name": "test",
        "split_protocol": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(test_ratio),
        },
        "data_source": "code15_only",
    }


def prepare_code15_chagas_bundle(args: argparse.Namespace) -> dict[str, Any]:
    data_base = load_local_module("linear_code15_chagas_dataset_base", "prototype_code15.py")
    base = load_local_module("linear_code15_chagas_training_base", "prototype_weighted_retrieval.py")

    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    data_base.apply_runtime_config(runtime_args)
    set_seed_on_module(base, args.seed)

    x_all, y_all_multilabel, _ = data_base.load_dataset()
    original_labels = list(data_base.TARGET_LABELS)
    task_payload = build_binary_target_labels(base, "Chagas disease")
    chagas_idx = original_labels.index("Chagas disease")
    y_all = y_all_multilabel[:, chagas_idx : chagas_idx + 1].float()
    indices = np.arange(len(x_all))
    test_ratio = 1.0 - float(args.train_ratio) - float(args.val_ratio)
    stratify_labels = y_all.view(-1).numpy()
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=float(args.val_ratio) + float(test_ratio),
        random_state=int(args.seed),
        stratify=stratify_labels,
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
    label_selection = {
        "strategy": "binary_single_label",
        "selected_label_count": 1,
        "selected_labels": ["Chagas disease"],
        "selected_original_indices": [0],
    }
    split_counts = summarize_split_counts(total_code15_samples=len(x_all), train_count=len(x_train), val_count=len(x_val), test_count=len(x_test))
    return {
        "task": args.task,
        "experiment_name": "code15_chagas_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {"data_source": "code15_chagas_binary"},
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict([("test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}})]),
        "primary_split_name": "test",
        "split_protocol": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(test_ratio),
        },
        "data_source": "code15_chagas_binary",
    }


def prepare_ape120_bundle(args: argparse.Namespace) -> dict[str, Any]:
    task_module = load_local_module("linear_ape120_task_module", "prototype_heedb_multilabel.py")
    output_root, checkpoint_root = ensure_output_dirs(args)
    runtime_args = build_runtime_args(args, output_root, checkpoint_root)
    base = task_module.BASE
    runtime_args.seed = int(args.seed)
    base.apply_runtime_config(runtime_args)
    set_seed_on_module(base, args.seed)

    full_label_config = task_module.load_curated_label_config(args.label_config_csv)
    task_module.configure_target_labels(full_label_config, 10)

    x_train_all, raw_y_train, _ = base.load_and_standardize(base.DATA_ECG_PATH, base.DATA_META_PATH, base.DATA_COL)
    y_train = base.build_ground_truth_matrix(raw_y_train).float()
    x_train = x_train_all.float()
    x_val_raw, raw_y_val, _ = base.load_and_standardize(base.VAL_ECG_PATH, base.VAL_META_PATH, base.VAL_COL)
    y_val = base.build_ground_truth_matrix(raw_y_val).float()
    x_val = x_val_raw.float()
    x_test_raw, raw_y_test, _ = base.load_and_standardize(base.TEST_ROBUST_ECG_PATH, base.TEST_ROBUST_META_PATH, base.TEST_ROBUST_COL)
    y_test = base.build_ground_truth_matrix(raw_y_test).float()
    x_test = x_test_raw.float()
    x_mimic_raw, raw_y_mimic, mimic_df = base.load_and_standardize(base.TEST_MIMIC_ECG_PATH, base.TEST_MIMIC_META_PATH, base.TEST_MIMIC_COL)
    y_mimic = base.build_ground_truth_matrix(raw_y_mimic).float()
    x_mimic = x_mimic_raw.float()
    mimic_split_indices = build_mimic18_split_indices(
        len(x_mimic),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=int(args.seed),
    )
    x_mimic_split = x_mimic[mimic_split_indices.test_idx].float()
    y_mimic_split = y_mimic[mimic_split_indices.test_idx].float()

    label_config, label_selection = task_module.build_positive_intersection_label_config(
        full_label_config,
        OrderedDict([("train", y_train), ("val", y_val), ("test", y_test)]),
    )
    keep_indices = list(label_selection["selected_original_indices"])
    task_module.configure_target_labels(label_config, 10)
    y_train = task_module.slice_label_matrix(y_train, keep_indices)
    y_val = task_module.slice_label_matrix(y_val, keep_indices)
    y_test = task_module.slice_label_matrix(y_test, keep_indices)
    y_mimic = task_module.slice_label_matrix(y_mimic, keep_indices)
    y_mimic_split = task_module.slice_label_matrix(y_mimic_split, keep_indices)

    task_payload = {
        "task_name": "ape120",
        "columns": list(label_config["labels"]),
        "labels": list(label_config["labels"]),
        "normal_idx": label_config.get("normal_idx"),
        "manifest_rows": list(label_config["manifest_rows"]),
    }
    split_counts = summarize_split_counts(
        heedb_train_count=len(x_train),
        heedb_val_count=len(x_val),
        heedb_internal_test_count=len(x_test),
        mimic_external_full_count=len(x_mimic),
        mimic_external_split_count=len(x_mimic_split),
    )
    return {
        "task": args.task,
        "experiment_name": "ape120_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {"label_config_csv": label_config["config_path"], "data_source": "heedb_to_mimic"},
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict(
            [
                ("heedb_internal_test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}}),
                ("mimic_external_full", {"x": x_mimic, "y": y_mimic, "summary": {"sample_count": int(len(x_mimic))}}),
                ("mimic_external_split", {"x": x_mimic_split, "y": y_mimic_split, "summary": {"sample_count": int(len(x_mimic_split))}}),
            ]
        ),
        "primary_split_name": "mimic_external_full",
        "split_protocol": {"data_source": "heedb_to_mimic", "mimic_comparable_split": True},
        "data_source": "heedb_to_mimic",
    }


def prepare_mimic100_bundle(args: argparse.Namespace) -> dict[str, Any]:
    task_module = load_local_module("linear_mimic100_task_module", "prototype_mimic_multilabel.py")
    output_root, checkpoint_root = ensure_output_dirs(args)
    base = task_module.BASE
    set_seed_on_module(base, args.seed)

    full_label_config = task_module.load_curated_label_config(args.label_config_csv)
    task_module.configure_target_labels(full_label_config, 10)
    x_all_raw, raw_labels, _ = base.load_and_standardize(args.mimic_ecg_path, args.mimic_meta_path, args.mimic_label_col)
    y_all = base.build_ground_truth_matrix(raw_labels).float()
    x_all = x_all_raw.float()
    indices = np.arange(len(x_all))
    train_idx, temp_idx = train_test_split(indices, test_size=(1.0 - args.train_ratio), random_state=int(args.seed))
    relative_val = args.val_ratio / (1.0 - args.train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - relative_val), random_state=int(args.seed))
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    label_config, label_selection = task_module.build_positive_intersection_label_config(
        full_label_config,
        {"train": y_train, "val": y_val, "test": y_test},
    )
    keep_indices = list(label_selection["selected_original_indices"])
    task_module.configure_target_labels(label_config, 10)
    y_train = task_module.slice_label_matrix(y_train, keep_indices)
    y_val = task_module.slice_label_matrix(y_val, keep_indices)
    y_test = task_module.slice_label_matrix(y_test, keep_indices)
    task_payload = {
        "task_name": "mimic100",
        "columns": list(label_config["labels"]),
        "labels": list(label_config["labels"]),
        "normal_idx": label_config.get("normal_idx"),
        "manifest_rows": list(label_config["manifest_rows"]),
    }
    split_counts = summarize_split_counts(total_mimic_samples=len(x_all), train_count=len(x_train), val_count=len(x_val), test_count=len(x_test))
    return {
        "task": args.task,
        "experiment_name": "mimic100_embedding_linear_baseline",
        "base_module": base,
        "task_payload": task_payload,
        "label_selection": label_selection,
        "split_counts": split_counts,
        "label_manifest_extra": {"label_config_csv": label_config["config_path"], "data_source": "mimic_only"},
        "output_root": output_root,
        "checkpoint_root": checkpoint_root,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "eval_splits": OrderedDict([("test", {"x": x_test, "y": y_test, "summary": {"sample_count": int(len(x_test))}})]),
        "primary_split_name": "test",
        "split_protocol": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
        },
        "data_source": "mimic_only",
    }


def prepare_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if args.external_eval:
        if args.task != "ape_mimic18":
            raise ValueError("--external-eval is only supported for --task ape_mimic18")
        return prepare_ape_mimic18_external_bundle(args)
    if args.task == "ape_mimic18":
        return prepare_ape_mimic18_bundle(args)
    if args.task == "mimic18":
        return prepare_mimic18_bundle(args)
    if args.task == "code15":
        return prepare_code15_bundle(args)
    if args.task == "code15_chagas":
        return prepare_code15_chagas_bundle(args)
    if args.task == "ape120":
        return prepare_ape120_bundle(args)
    if args.task == "mimic100":
        return prepare_mimic100_bundle(args)
    raise ValueError(f"Unsupported task: {args.task}")


def run_bundle(args: argparse.Namespace, bundle: dict[str, Any]) -> None:
    base = bundle["base_module"]
    device = resolve_device(base, args.device)
    output_root = bundle["output_root"]
    checkpoint_root = bundle["checkpoint_root"]
    label_manifest_path = output_root / "target_label_manifest.json"
    checkpoint_path = checkpoint_root / "best_model.pt"
    task_payload = bundle["task_payload"]
    eval_splits: OrderedDict[str, dict[str, Any]] = bundle["eval_splits"]
    primary_split_name = bundle["primary_split_name"]
    summary_json_path = output_root / bundle.get("summary_name", "summary.json")
    ci_output_dir = output_root / "ci"
    ci_summary_name = bundle.get("ci_summary_name", "aggregate_summary_ci.json")

    write_label_manifest(
        label_manifest_path,
        task_payload=task_payload,
        label_selection=bundle["label_selection"],
        split_counts=bundle["split_counts"],
        extra_payload=bundle.get("label_manifest_extra"),
    )

    num_classes = int(bundle["y_train"].shape[1])
    feature_dim = int(bundle["x_train"].shape[1])
    max_epochs = int(args.max_epochs if args.max_epochs is not None else getattr(base, "MAX_EPOCHS", 100))
    eval_interval = int(getattr(base, "EVAL_INTERVAL", 2))
    lr = float(args.lr if args.lr is not None else getattr(base, "GLOBAL_LR", 1e-3))

    if args.eval_only_ci or bundle.get("load_checkpoint_only"):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
        model = load_best_model(checkpoint_path, feature_dim, num_classes, device)
        best_val_auc = None
        best_epoch = None
        if args.ci_only_split:
            if args.ci_only_split not in eval_splits:
                raise ValueError(
                    f"Unknown --ci-only-split '{args.ci_only_split}'. Available: {list(eval_splits.keys())}"
                )
            eval_splits = OrderedDict([(args.ci_only_split, eval_splits[args.ci_only_split])])
        print_eval_split_sizes(eval_splits)
    else:
        model = EmbeddingLinearClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
        scheduler = build_lr_scheduler(args=args, optimizer=optimizer, max_epochs=max_epochs)
        best_val_auc = float("-inf")
        best_epoch = 0
        train_loader = make_dataloader(base, bundle["x_train"], bundle["y_train"], int(getattr(base, "TRAIN_BATCH_SIZE", 4096)), shuffle=True)
        print_prompt_plan_if_available(base)
        print(
            "[Linear] "
            f"feature_dim={feature_dim}, num_classes={num_classes}, max_epochs={max_epochs}, "
            f"lr={lr}, weight_decay={float(args.weight_decay)}, "
            f"lr_scheduler={args.lr_scheduler}, min_lr={float(args.min_lr)}"
        )
        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"[Linear] Epoch {epoch}: loss={train_loss:.4f}, lr={get_current_lr(optimizer):.6g}")
            if epoch % eval_interval != 0:
                if scheduler is not None and args.lr_scheduler == "cosine":
                    scheduler.step()
                continue
            val_metrics, _val_df, _val_logits = evaluate_split(
                base_module=base,
                model=model,
                x_eval=bundle["x_val"],
                y_eval=bundle["y_val"],
                split_name="val",
                device=device,
            )
            val_auc = float(val_metrics["macro_auroc"])
            if val_auc > best_val_auc:
                old_best = best_val_auc
                best_val_auc = val_auc
                best_epoch = epoch
                save_best_model(checkpoint_path, model, list(base.TARGET_LABELS))
                print(f"[Linear] New best val macro AUROC: {old_best:.4f} -> {best_val_auc:.4f}")
            if scheduler is not None:
                if args.lr_scheduler == "plateau":
                    scheduler.step(val_auc)
                else:
                    scheduler.step()
        if checkpoint_path.exists():
            model = load_best_model(checkpoint_path, feature_dim, num_classes, device)
        else:
            save_best_model(checkpoint_path, model, list(base.TARGET_LABELS))

    if not args.skip_ci:
        ci_output_dir.mkdir(parents=True, exist_ok=True)

    split_results: dict[str, Any] = {}
    ci_reports: dict[str, Any] = {}
    primary_csv_written = False
    root_primary_csv_path = output_root / "per_class_auc.csv"

    for split_name, split_bundle in eval_splits.items():
        metrics, per_class_df, logits = evaluate_split(
            base_module=base,
            model=model,
            x_eval=split_bundle["x"],
            y_eval=split_bundle["y"],
            split_name=split_name,
            device=device,
        )
        per_split_csv_path = output_root / f"per_class_auc_{split_name}_{BRANCH_NAME}.csv"
        ci_summary = None
        ci_refs = None
        export_df = per_class_df
        if not args.skip_ci:
            ci_refs, ci_payload = export_ci_for_split(
                ci_output_dir=ci_output_dir,
                split_name=split_name,
                y_true=split_bundle["y"],
                logits=logits,
                class_names=list(base.TARGET_LABELS),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            ci_summary = ci_payload["summary"]
            export_df = merge_ci_columns(per_class_df, ci_refs["csv_path"])
            ci_reports[split_name] = ci_refs
        export_df.to_csv(per_split_csv_path, index=False)
        if split_name == primary_split_name:
            export_df.to_csv(root_primary_csv_path, index=False)
            primary_csv_written = True
        split_results[split_name] = {
            "metrics": metrics,
            "per_class_csv": str(per_split_csv_path),
        }
        if ci_summary is not None:
            split_results[split_name]["ci_summary"] = ci_summary

    if not primary_csv_written:
        first_split_name = next(iter(eval_splits))
        primary_path = output_root / f"per_class_auc_{first_split_name}_{BRANCH_NAME}.csv"
        if primary_path.exists():
            pd.read_csv(primary_path).to_csv(root_primary_csv_path, index=False)

    if not args.skip_ci:
        save_json(
            ci_output_dir / ci_summary_name,
            {
                "experiment_name": bundle["experiment_name"],
                "branch_name": BRANCH_NAME,
                "bootstrap_iterations": int(args.bootstrap),
                "seed": int(args.seed),
                "reports": ci_reports,
            },
        )

    payload: dict[str, Any] = {
        "experiment_name": bundle["experiment_name"],
        "evaluation_type": bundle.get("evaluation_type", "standard"),
        "data_source": bundle.get("data_source"),
        "selected_branch": BRANCH_NAME,
        "label_count": len(base.TARGET_LABELS),
        "label_selection": bundle["label_selection"],
        "split_protocol": bundle["split_protocol"],
        "splits": {name: value["summary"] for name, value in eval_splits.items()},
        "metrics": {
            BRANCH_NAME: {
                "checkpoint": str(checkpoint_path),
                "best_val_macro_auroc": best_val_auc,
                "best_epoch": best_epoch,
                "splits": split_results,
            }
        },
    }
    if not args.skip_ci:
        payload["ci"] = {
            "output_dir": str(ci_output_dir),
            "aggregate_summary_json": str(ci_output_dir / ci_summary_name),
        }
    save_json(summary_json_path, payload)


def main() -> None:
    args = parse_args()
    if args.eval_only_ci and args.external_eval:
        raise ValueError("--eval-only-ci and --external-eval are mutually exclusive")
    bundle = prepare_bundle(args)
    run_bundle(args, bundle)


if __name__ == "__main__":
    main()
