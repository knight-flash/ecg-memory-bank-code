#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-label HEEDB prototype experiment.

Purpose
-------
Run the paper's 120-label HEEDB internal model and its MIMIC external transfer
evaluation.

Method
------
Load curated L3 aliases, allocate prototypes by class support, train and
project a ProtoPNet head, then evaluate the same head on two cohorts.

Inputs
------
HEEDB train/val/test tensors and metadata, external MIMIC tensors/metadata,
and curated label-config CSV.

Outputs
-------
External model checkpoints, HEEDB/MIMIC AUROC summaries, per-class files, and
projected case-link artifacts.

Paper parameters
----------------
The formal runner passes seed=42, 3000 total prototype budget, 1–30 per class,
and 1000 bootstrap resamples; standalone historical defaults differ.

Shapes and meaning
------------------
Input vectors are [N,D]; targets/scores [N,120] internally and matched mapped
labels externally; prototypes are [P,D] with real-case anchors.
"""

from __future__ import annotations

import os

import argparse
import csv
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mimic18_split_utils import build_mimic18_split_indices

CI_UTILS_PATH = Path(__file__).resolve().parent / "ci_bootstrap_utils.py"

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = BASE_DIR.parent
LOCAL_PROTOPNET_DIR = EXPERIMENT_ROOT / "protopnet"
LOCAL_COMMON_DIR = EXPERIMENT_ROOT / "common"

for candidate in (LOCAL_PROTOPNET_DIR, LOCAL_COMMON_DIR):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))


def load_base_module():
    module_path = BASE_DIR / "prototype_heedb.py"
    spec = importlib.util.spec_from_file_location("prototype_all_ape120_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
_ci_spec = importlib.util.spec_from_file_location("prototype_all_ci_utils_ape120", CI_UTILS_PATH)
if _ci_spec is None or _ci_spec.loader is None:
    raise RuntimeError(f"Unable to load CI utils from {CI_UTILS_PATH}")
CI_UTILS = importlib.util.module_from_spec(_ci_spec)
_ci_spec.loader.exec_module(CI_UTILS)

EXPERIMENT_NAME = "ape_full_label_fixed"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ECG_OUTPUT_ROOT", "__OUTPUT_DIR_REQUIRED__")) / "protopnet" / EXPERIMENT_NAME
DEFAULT_LABEL_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "l3_full_label_curated.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_ROOT / "summary.json"
DEFAULT_PER_CLASS_CSV = DEFAULT_OUTPUT_ROOT / "per_class_auc.csv"
TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


def resolve_label_config_path(csv_path: str | Path) -> Path:
    primary = Path(csv_path)
    candidates = [
        primary,
        Path(__file__).resolve().parents[3] / "configs" / "l3_full_label_curated.csv",
        EXPERIMENT_ROOT.parent / "Finalmore" / "Prototype" / "config" / "l3_full_label_curated.csv",
        Path("__ECG_DATA_ROOT_REQUIRED__/Finalmore/Prototype/config/l3_full_label_curated.csv"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate
    return primary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalmore L3 full-label fixed-prototype HEEDB training with MIMIC-compatible external evaluation.",
    )
    parser.add_argument("--data-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_ecg_embeddings.pt")
    parser.add_argument("--data-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_texts_with_labels_correct.csv")
    parser.add_argument("--data-label-col", default="labels_correct")
    parser.add_argument("--heedb-train-export-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_final/train_processed_update.csv")
    parser.add_argument("--val-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/val/heedb_ecg_embeddings.pt")
    parser.add_argument("--val-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/val/heedb_texts_with_labels_correct.csv")
    parser.add_argument("--val-label-col", default="labels_correct")
    parser.add_argument("--test-robust-ecg-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_ecg_embeddings.pt")
    parser.add_argument("--test-robust-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv")
    parser.add_argument("--test-robust-label-col", default="labels_correct")
    parser.add_argument("--heedb-test-export-meta-path", default="__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv")
    parser.add_argument("--mimic-ecg-path", default=BASE.TEST_MIMIC_ECG_PATH)
    parser.add_argument("--mimic-meta-path", default=BASE.TEST_MIMIC_META_PATH)
    parser.add_argument("--mimic-label-col", default=BASE.TEST_MIMIC_COL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=BASE.RANDOM_SEED)
    parser.add_argument("--label-config-csv", default=str(DEFAULT_LABEL_CONFIG))
    parser.add_argument("--prototype-budget", type=int, default=1000,
                        help="Total prototype budget for adaptive allocation.")
    parser.add_argument("--min-prototypes-per-class", type=int, default=1,
                        help="Minimum prototypes per class with positives.")
    parser.add_argument("--max-prototypes-per-class", type=int, default=10,
                        help="Maximum prototypes per class.")
    parser.add_argument("--save-summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--save-per-class-csv", default=str(DEFAULT_PER_CLASS_CSV))
    parser.add_argument("--max-epochs", type=int, default=BASE.MAX_EPOCHS)
    parser.add_argument("--eval-only-ci", action="store_true")
    parser.add_argument(
        "--ci-only-split",
        choices=["best_heedb_internal", "best_mimic_external_full", "best_mimic_external_split"],
        default="",
        help="Optionally restrict eval-only CI export to a single split.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def is_enabled(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_curated_label_config(csv_path: str | Path) -> dict[str, Any]:
    config_path = resolve_label_config_path(csv_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Label config CSV not found: {config_path}")

    with config_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_cols = {"canonical_label", "alias", "prompt", "group", "enabled"}
        missing_cols = required_cols.difference(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing required columns in label config CSV: {sorted(missing_cols)}")

        ordered_labels: OrderedDict[str, dict[str, Any]] = OrderedDict()
        alias_owner: dict[str, str] = {}

        for row_idx, row in enumerate(reader, start=2):
            if not is_enabled(row.get("enabled", "")):
                continue

            canonical_label = str(row["canonical_label"]).strip()
            alias = str(row["alias"]).strip() or canonical_label
            prompt = str(row["prompt"]).strip() or f"An ECG recording showing {canonical_label}."
            group = str(row["group"]).strip() or "Ungrouped"

            if not canonical_label:
                raise ValueError(f"Empty canonical_label at row {row_idx} in {config_path}")

            label_entry = ordered_labels.setdefault(
                canonical_label,
                {
                    "prompt": prompt,
                    "group": group,
                    "aliases_original": [],
                    "aliases_normalized": set(),
                },
            )

            for candidate in (canonical_label, alias):
                normalized = BASE.normalize_label_text(candidate)
                if not normalized:
                    continue
                owner = alias_owner.get(normalized)
                if owner is not None and owner != canonical_label:
                    raise ValueError(
                        f"Alias '{candidate}' (normalized='{normalized}') is mapped to both "
                        f"'{owner}' and '{canonical_label}' in {config_path}.",
                    )
                alias_owner[normalized] = canonical_label
                label_entry["aliases_normalized"].add(normalized)
                if candidate not in label_entry["aliases_original"]:
                    label_entry["aliases_original"].append(candidate)

    labels = list(ordered_labels.keys())
    if not labels:
        raise ValueError(f"No enabled labels found in {config_path}")

    normalized_normal = BASE.normalize_label_text("Normal ECG")
    normal_idx = None
    for idx, label_name in enumerate(labels):
        if BASE.normalize_label_text(label_name) == normalized_normal:
            normal_idx = idx
            break
    if normal_idx is None:
        raise ValueError("Curated label config must include a canonical 'Normal ECG' label.")

    prompts = [str(ordered_labels[label_name]["prompt"]) for label_name in labels]
    alias_sets = [set(ordered_labels[label_name]["aliases_normalized"]) for label_name in labels]
    manifest_rows = []
    for class_idx, label_name in enumerate(labels):
        manifest_rows.append(
            {
                "class_idx": class_idx,
                "canonical_label": label_name,
                "prompt": str(ordered_labels[label_name]["prompt"]),
                "group": str(ordered_labels[label_name]["group"]),
                "alias_count": len(ordered_labels[label_name]["aliases_original"]),
                "aliases": list(ordered_labels[label_name]["aliases_original"]),
            }
        )

    return {
        "config_path": str(config_path),
        "labels": labels,
        "prompts": prompts,
        "alias_sets": alias_sets,
        "normal_idx": normal_idx,
        "manifest_rows": manifest_rows,
    }


def configure_target_labels(config_payload: dict[str, Any], max_prototypes_per_class: int) -> None:
    BASE.TARGET_LABELS = list(config_payload["labels"])
    BASE.TARGET_PROMPTS = list(config_payload["prompts"])
    BASE.TARGET_ALIAS_SETS = list(config_payload["alias_sets"])
    BASE.TARGET_KEYWORDS = [sorted(alias_set) for alias_set in config_payload["alias_sets"]]
    normal_idx = config_payload.get("normal_idx")
    if normal_idx is None:
        BASE.NORMAL_LABEL = "Normal ECG"
        BASE.NORMAL_CLASS_IDX = -1
    else:
        BASE.NORMAL_LABEL = BASE.TARGET_LABELS[int(normal_idx)]
        BASE.NORMAL_CLASS_IDX = int(normal_idx)
    BASE.DEFAULT_PROTOTYPES_PER_CLASS = int(max_prototypes_per_class)
    BASE.PROTOTYPE_COUNT_OVERRIDES = {}
    BASE.CLASSIFICATION_REFERENCE_CSV = Path(config_payload["config_path"])


def build_positive_intersection_label_config(
    config_payload: dict[str, Any],
    split_targets: OrderedDict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    keep_indices: list[int] = []
    dropped_rows: list[dict[str, Any]] = []

    for class_idx, label_name in enumerate(config_payload["labels"]):
        positive_counts = {
            split_name: int(targets[:, class_idx].sum().item())
            for split_name, targets in split_targets.items()
        }
        if all(count > 0 for count in positive_counts.values()):
            keep_indices.append(class_idx)
        else:
            dropped_rows.append(
                {
                    "class_idx": class_idx,
                    "canonical_label": label_name,
                    "positive_counts": positive_counts,
                }
            )

    if not keep_indices:
        raise ValueError("No labels remain after applying the positive-label intersection across train/val/test.")

    normal_idx = config_payload.get("normal_idx")
    filtered_normal_idx = keep_indices.index(normal_idx) if normal_idx in keep_indices else None

    filtered_manifest_rows = []
    for new_class_idx, original_idx in enumerate(keep_indices):
        row = dict(config_payload["manifest_rows"][original_idx])
        row["class_idx"] = new_class_idx
        filtered_manifest_rows.append(row)

    filtered_config = {
        "config_path": config_payload["config_path"],
        "labels": [config_payload["labels"][idx] for idx in keep_indices],
        "prompts": [config_payload["prompts"][idx] for idx in keep_indices],
        "alias_sets": [config_payload["alias_sets"][idx] for idx in keep_indices],
        "normal_idx": filtered_normal_idx,
        "manifest_rows": filtered_manifest_rows,
    }
    selection_info = {
        "strategy": "positive_intersection_across_core_splits",
        "required_splits": list(split_targets.keys()),
        "original_label_count": len(config_payload["labels"]),
        "selected_label_count": len(keep_indices),
        "dropped_label_count": len(dropped_rows),
        "dropped_labels": dropped_rows,
        "selected_labels": filtered_config["labels"],
        "selected_original_indices": keep_indices,
    }
    return filtered_config, selection_info


def slice_label_matrix(y_true: torch.Tensor, keep_indices: list[int]) -> torch.Tensor:
    index_tensor = torch.tensor(keep_indices, dtype=torch.long)
    return y_true.index_select(dim=1, index=index_tensor)


def build_adaptive_prototype_count_plan(
    positive_counts: list[int],
    total_budget: int,
    min_protos: int = 1,
    max_protos: int = 10,
) -> list[int]:
    """Allocate prototype counts proportional to per-class positive sample count.

    Classes with 0 positive samples receive 0 prototypes.
    The remaining budget is distributed proportionally among classes with positives,
    clamped to [min_protos, max_protos].
    """
    n = len(positive_counts)
    counts = [0] * n

    active_indices = [i for i in range(n) if positive_counts[i] > 0]

    if not active_indices:
        return counts

    active_positives = [max(positive_counts[i], 1) for i in active_indices]
    total_positives = sum(active_positives)

    raw_alloc = {}
    for idx, pos in zip(active_indices, active_positives):
        proportion = pos / total_positives
        raw_alloc[idx] = max(min_protos, round(proportion * total_budget))

    for _ in range(20):
        allocated = {idx: min(raw_alloc[idx], max_protos) for idx in active_indices}
        current_total = sum(allocated.values())

        if current_total == total_budget:
            break

        if current_total > total_budget:
            scale = total_budget / current_total
            for idx in active_indices:
                allocated[idx] = max(min_protos, round(allocated[idx] * scale))
            diff = total_budget - sum(allocated.values())
            if diff != 0:
                sorted_idx = sorted(active_indices, key=lambda i: positive_counts[i], reverse=True)
                for idx in sorted_idx:
                    if diff == 0:
                        break
                    room = max_protos - allocated[idx]
                    if diff > 0 and room > 0:
                        add = min(room, diff)
                        allocated[idx] += add
                        diff -= add
                    elif diff < 0 and allocated[idx] > min_protos:
                        sub = min(allocated[idx] - min_protos, -diff)
                        allocated[idx] -= sub
                        diff += sub
            raw_alloc = allocated
        else:
            remaining = total_budget - current_total
            sorted_idx = sorted(active_indices, key=lambda i: positive_counts[i], reverse=True)
            for idx in sorted_idx:
                if remaining <= 0:
                    break
                room = max_protos - allocated[idx]
                if room > 0:
                    add = min(room, remaining)
                    allocated[idx] += add
                    remaining -= add
            raw_alloc = allocated
            break

    for idx in active_indices:
        counts[idx] = raw_alloc[idx]

    return counts


def summarize_support(y_true: torch.Tensor, label_names: list[str]) -> dict[str, Any]:
    positive_counts = y_true.sum(dim=0).cpu().numpy().astype(int).tolist()
    zero_positive_labels = [label for label, count in zip(label_names, positive_counts) if count == 0]
    low_support_labels = [label for label, count in zip(label_names, positive_counts) if count < 6]
    return {
        "label_count": len(label_names),
        "zero_positive_label_count": len(zero_positive_labels),
        "zero_positive_labels": zero_positive_labels,
        "positive_support_lt6_count": len(low_support_labels),
        "positive_support_lt6_labels": low_support_labels,
    }


def evaluate_performance_detailed(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    label_names: list[str],
    title: str,
    split_name: str,
) -> tuple[float, pd.DataFrame]:
    y_scores = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    print(f"\n=== {title} ===")
    print(f"{'Class':<45} | {'AUROC':<10} | {'Pos'}")
    print("-" * 78)

    aucs: list[float] = []
    rows: list[dict[str, Any]] = []
    num_samples = int(y_true_np.shape[0])

    for class_idx, label_name in enumerate(label_names):
        positive_count = int(np.sum(y_true_np[:, class_idx]))
        negative_count = int(num_samples - positive_count)
        eligible = positive_count > 5 and negative_count > 5
        auc_value: float | None = None

        if eligible:
            try:
                roc_auc = BASE.roc_auc_score(y_true_np[:, class_idx], y_scores[:, class_idx])
                auc_value = float(roc_auc)
                aucs.append(auc_value)
                status_text = f"{auc_value:.4f}"
            except Exception:
                status_text = "Error"
        else:
            status_text = "N/A"

        print(f"{label_name:<45} | {status_text:<10} | {positive_count}")
        rows.append(
            {
                "split": split_name,
                "class_idx": class_idx,
                "class_name": label_name,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "eligible_for_auc": eligible,
                "auroc": auc_value,
            }
        )

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print("-" * 78)
    print(f"[{title}] Mean AUROC: {mean_auc:.4f}")
    return mean_auc, pd.DataFrame(rows)


def evaluate_on_loader_detailed(model, dataloader, y_true, device, label_names, title, split_name):
    logits = BASE.collect_logits(model, dataloader, device)
    return evaluate_performance_detailed(
        y_true=y_true,
        y_logits=logits,
        label_names=label_names,
        title=title,
        split_name=split_name,
    )


def print_eval_split_sizes(split_map: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    print("\n[INFO] Evaluation split sizes:")
    for split_name, (x_eval, y_eval) in split_map.items():
        print(
            f"  - {split_name}: sample_count={len(x_eval)} "
            f"| feature_rows={int(x_eval.shape[0])} | label_rows={int(y_eval.shape[0])}"
        )


def build_prototype_count_rows(
    manifest_rows: list[dict[str, Any]],
    prototype_counts: list[int],
    y_train: torch.Tensor,
) -> list[dict[str, Any]]:
    positive_counts = y_train.sum(dim=0).cpu().numpy().astype(int).tolist()
    rows = []
    for item, proto_count, positive_count in zip(manifest_rows, prototype_counts, positive_counts):
        rows.append(
            {
                "class_idx": item["class_idx"],
                "canonical_label": item["canonical_label"],
                "group": item["group"],
                "alias_count": item["alias_count"],
                "fixed_prototype_count": int(proto_count),
                "train_positive_count": int(positive_count),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    BASE.apply_runtime_config(args)
    BASE.set_seed(BASE.RANDOM_SEED)
    device = BASE.resolve_device(args.device)

    output_root = Path(BASE.OUTPUT_ROOT_DIR)
    checkpoint_root = Path(BASE.CHECKPOINT_ROOT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    full_label_config = load_curated_label_config(args.label_config_csv)
    configure_target_labels(full_label_config, args.max_prototypes_per_class)

    summary_json_path = Path(args.save_summary_json)
    per_class_csv_path = Path(args.save_per_class_csv)
    label_manifest_path = output_root / "target_label_manifest.json"
    prototype_plan_csv_path = output_root / "prototype_count_plan.csv"

    x_train_all, raw_y_train, metadata_train_df = BASE.load_and_standardize(BASE.DATA_ECG_PATH, BASE.DATA_META_PATH, BASE.DATA_COL)
    y_train = BASE.build_ground_truth_matrix(raw_y_train).float()
    x_train = x_train_all.float()
    df_train = BASE.attach_export_metadata(
        metadata_train_df.reset_index(drop=True),
        BASE.HEEDB_TRAIN_EXPORT_META_PATH,
    )

    x_val_raw, raw_y_val, metadata_val_df = BASE.load_and_standardize(BASE.VAL_ECG_PATH, BASE.VAL_META_PATH, BASE.VAL_COL)
    y_val = BASE.build_ground_truth_matrix(raw_y_val).float()
    x_val = x_val_raw.float()

    x_test_raw, raw_y_test, metadata_test_df = BASE.load_and_standardize(
        BASE.TEST_ROBUST_ECG_PATH,
        BASE.TEST_ROBUST_META_PATH,
        BASE.TEST_ROBUST_COL,
    )
    y_test = BASE.build_ground_truth_matrix(raw_y_test).float()
    x_test = x_test_raw.float()
    df_test = metadata_test_df.reset_index(drop=True)

    x_mimic_raw, raw_y_mimic, mimic_df = BASE.load_and_standardize(BASE.TEST_MIMIC_ECG_PATH, BASE.TEST_MIMIC_META_PATH, BASE.TEST_MIMIC_COL)
    y_mimic = BASE.build_ground_truth_matrix(raw_y_mimic).float()
    x_mimic = x_mimic_raw.float()
    mimic_split_indices = build_mimic18_split_indices(
        len(x_mimic),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=int(args.seed),
    )
    x_mimic_split = x_mimic[mimic_split_indices.test_idx].float()
    y_mimic_split = y_mimic[mimic_split_indices.test_idx].float()
    print(
        "[INFO] MIMIC split preview: "
        f"full={len(x_mimic)}, comparable_test={len(x_mimic_split)}"
    )

    label_config, label_selection = build_positive_intersection_label_config(
        full_label_config,
        OrderedDict(
            [
                ("train", y_train),
                ("val", y_val),
                ("test", y_test),
            ]
        ),
    )
    keep_indices = list(label_selection["selected_original_indices"])
    configure_target_labels(label_config, args.max_prototypes_per_class)

    y_train = slice_label_matrix(y_train, keep_indices)
    y_val = slice_label_matrix(y_val, keep_indices)
    y_test = slice_label_matrix(y_test, keep_indices)
    y_mimic = slice_label_matrix(y_mimic, keep_indices)
    y_mimic_split = slice_label_matrix(y_mimic_split, keep_indices)

    print("=== Finalmore L3 Full-Label Adaptive-Prototype ProtoPNet ===")
    print(f"[INFO] Label config: {label_config['config_path']}")
    print(f"[INFO] Core-split label intersection: train ∩ val ∩ test")
    print(
        f"[INFO] Label count: {label_selection['selected_label_count']} "
        f"(from {label_selection['original_label_count']}, dropped {label_selection['dropped_label_count']})"
    )
    print(f"[INFO] Prototype budget: {args.prototype_budget}")
    print(f"[INFO] Min prototypes/class: {args.min_prototypes_per_class}")
    print(f"[INFO] Max prototypes/class: {args.max_prototypes_per_class}")
    print(f"[INFO] Max epochs: {args.max_epochs}")

    BASE.print_classification_prompt_plan()

    train_idx = np.arange(len(x_train))

    train_positive_counts = y_train.sum(dim=0).cpu().numpy().astype(int).tolist()
    prototype_counts = build_adaptive_prototype_count_plan(
        positive_counts=train_positive_counts,
        total_budget=args.prototype_budget,
        min_protos=args.min_prototypes_per_class,
        max_protos=args.max_prototypes_per_class,
    )
    print(f"\n[INFO] Adaptive prototype allocation (total={sum(prototype_counts)}):")
    active_count = sum(1 for c in prototype_counts if c > 0)
    zero_count = sum(1 for c in prototype_counts if c == 0)
    print(f"  Active classes: {active_count}, Skipped (0 positive): {zero_count}")
    BASE.print_prototype_count_plan(prototype_counts)

    support_summary = {
        "train": summarize_support(y_train, BASE.TARGET_LABELS),
        "val": summarize_support(y_val, BASE.TARGET_LABELS),
        "heedb_internal": summarize_support(y_test, BASE.TARGET_LABELS),
        "mimic_external_full": summarize_support(y_mimic, BASE.TARGET_LABELS),
        "mimic_external_split": summarize_support(y_mimic_split, BASE.TARGET_LABELS),
    }

    save_json(
        label_manifest_path,
        {
            "label_config_csv": label_config["config_path"],
            "label_count": len(BASE.TARGET_LABELS),
            "label_selection": label_selection,
            "prototype_budget": args.prototype_budget,
            "min_prototypes_per_class": args.min_prototypes_per_class,
            "max_prototypes_per_class": args.max_prototypes_per_class,
            "labels": label_config["manifest_rows"],
        },
    )
    pd.DataFrame(build_prototype_count_rows(label_config["manifest_rows"], prototype_counts, y_train)).to_csv(
        prototype_plan_csv_path,
        index=False,
    )

    if args.eval_only_ci:
        ci_output_dir = output_root / "ci"
        ci_output_dir.mkdir(parents=True, exist_ok=True)

        best_checkpoint = Path(BASE.BEST_MODEL_PATH)
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

        state = torch.load(best_checkpoint, map_location=device)
        prototypes = state["prototypes"].detach().cpu()
        prototype_labels = state["prototype_labels"].detach().cpu().long()
        model = BASE.TraditionalProtoPNet(prototypes, prototype_labels, len(BASE.TARGET_LABELS)).to(device)
        model.load_state_dict(state)

        eval_splits = {
            "best_heedb_internal": (x_test, y_test),
            "best_mimic_external_full": (x_mimic, y_mimic),
            "best_mimic_external_split": (x_mimic_split, y_mimic_split),
        }
        if args.ci_only_split:
            eval_splits = {args.ci_only_split: eval_splits[args.ci_only_split]}
        print_eval_split_sizes(eval_splits)
        generated_reports: dict[str, Any] = {}
        for split_name, (x_eval, y_eval) in eval_splits.items():
            eval_loader = BASE.make_dataloader(x_eval, y_eval, BASE.EVAL_BATCH_SIZE, shuffle=False)
            logits = BASE.collect_logits(model, eval_loader, device)
            y_true_np = y_eval.detach().cpu().numpy()
            y_score_np = torch.sigmoid(logits).detach().cpu().numpy()
            report_payload = CI_UTILS.build_multilabel_ci_report(
                class_names=list(BASE.TARGET_LABELS),
                y_true=y_true_np,
                y_score=y_score_np,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
            report_payload["metric_key"] = "traditional_best"
            report_payload["eval_split_name"] = split_name
            refs = CI_UTILS.write_multilabel_ci_outputs(
                output_dir=ci_output_dir,
                metric_key=f"traditional_best_{split_name}",
                report_payload=report_payload,
            )
            generated_reports[split_name] = refs

        CI_UTILS.save_json(
            ci_output_dir / "aggregate_summary_ci.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "bootstrap_iterations": int(args.bootstrap),
                "seed": int(args.seed),
                "best_checkpoint": str(best_checkpoint),
                "reports": generated_reports,
            },
        )
        return

    init_prototypes, prototype_labels = BASE.initialize_fixed_prototypes(
        x_train,
        y_train,
        prototype_counts,
        BASE.INIT_MAX_SAMPLES_PER_CLASS,
    )
    model = BASE.TraditionalProtoPNet(init_prototypes, prototype_labels, len(BASE.TARGET_LABELS)).to(device)

    train_loader = BASE.make_dataloader(x_train, y_train, BASE.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = BASE.make_dataloader(x_val, y_val, BASE.EVAL_BATCH_SIZE, shuffle=False)
    heedb_test_loader = BASE.make_dataloader(x_test, y_test, BASE.EVAL_BATCH_SIZE, shuffle=False)
    mimic_loader = BASE.make_dataloader(x_mimic, y_mimic, BASE.EVAL_BATCH_SIZE, shuffle=False)
    mimic_split_loader = BASE.make_dataloader(x_mimic_split, y_mimic_split, BASE.EVAL_BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 60)
    print(" >> Baseline: Fixed Multi-Prototype Initialization")
    print("=" * 60)
    with torch.no_grad():
        init_logits_internal = BASE.collect_logits(model, heedb_test_loader, device)
        baseline_auc_internal, _ = evaluate_performance_detailed(
            y_true=y_test,
            y_logits=init_logits_internal,
            label_names=BASE.TARGET_LABELS,
            title="Initial Fixed-Prototype Baseline on HEEDB Test (Internal)",
            split_name="baseline_heedb_internal",
        )
        init_logits_mimic = BASE.collect_logits(model, mimic_loader, device)
        baseline_auc_mimic_full, _ = evaluate_performance_detailed(
            y_true=y_mimic,
            y_logits=init_logits_mimic,
            label_names=BASE.TARGET_LABELS,
            title="Initial Fixed-Prototype Baseline on MIMIC Test (External Full)",
            split_name="baseline_mimic_external_full",
        )
        init_logits_mimic_split = BASE.collect_logits(model, mimic_split_loader, device)
        baseline_auc_mimic_split, _ = evaluate_performance_detailed(
            y_true=y_mimic_split,
            y_logits=init_logits_mimic_split,
            label_names=BASE.TARGET_LABELS,
            title="Initial Fixed-Prototype Baseline on MIMIC Test (External Comparable Split)",
            split_name="baseline_mimic_external_split",
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE.GLOBAL_LR)
    best_val_auc = 0.0
    best_epoch = 0

    print("\n" + "=" * 60)
    print(" >> Start Full-Label Fixed-Prototype Training")
    print("=" * 60)
    print(
        f"Training config: epochs={args.max_epochs}, projection_interval={BASE.PROJECTION_INTERVAL}, "
        f"last_layer_epochs={BASE.LAST_LAYER_FINE_TUNE_EPOCHS}"
    )

    for epoch in range(1, args.max_epochs + 1):
        average_loss, loss_stats = BASE.run_global_epoch(model, train_loader, optimizer, device)
        print(
            f"Epoch {epoch}: Loss={average_loss:.4f} | "
            f"cls={loss_stats['classification']:.4f} | "
            f"cluster={loss_stats['cluster']:.4f} | "
            f"sep={loss_stats['separation']:.4f}"
        )

        if epoch % BASE.EVAL_INTERVAL == 0:
            val_auc, _ = evaluate_on_loader_detailed(
                model,
                val_loader,
                y_val,
                device,
                BASE.TARGET_LABELS,
                title=f"Epoch {epoch} Val Monitor",
                split_name=f"epoch_{epoch}_val_monitor",
            )
            best_val_auc, best_epoch = BASE.maybe_save_best(
                model,
                val_auc,
                best_val_auc,
                best_epoch,
                epoch,
                BASE.BEST_MODEL_PATH,
            )

        if epoch % BASE.PROJECTION_INTERVAL == 0:
            BASE.project_prototypes(
                model,
                x_train,
                y_train,
                device,
                BASE.EVAL_BATCH_SIZE,
                update_model=True,
            )

            val_auc_after_projection, _ = evaluate_on_loader_detailed(
                model,
                val_loader,
                y_val,
                device,
                BASE.TARGET_LABELS,
                title=f"Epoch {epoch} Val After Projection",
                split_name=f"epoch_{epoch}_val_after_projection",
            )
            best_val_auc, best_epoch = BASE.maybe_save_best(
                model,
                val_auc_after_projection,
                best_val_auc,
                best_epoch,
                epoch,
                BASE.BEST_MODEL_PATH,
            )

            BASE.fine_tune_last_layer(
                model,
                train_loader,
                device,
                BASE.LAST_LAYER_FINE_TUNE_EPOCHS,
                BASE.LAST_LAYER_LR,
            )

            val_auc_after_convex, _ = evaluate_on_loader_detailed(
                model,
                val_loader,
                y_val,
                device,
                BASE.TARGET_LABELS,
                title=f"Epoch {epoch} Val After Convex Opt",
                split_name=f"epoch_{epoch}_val_after_convex",
            )
            best_val_auc, best_epoch = BASE.maybe_save_best(
                model,
                val_auc_after_convex,
                best_val_auc,
                best_epoch,
                epoch,
                BASE.BEST_MODEL_PATH,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=BASE.GLOBAL_LR)

    print("\n" + "=" * 60)
    print(" >> Final Evaluation (Internal & External)")
    print("=" * 60)

    with torch.no_grad():
        last_logits_internal = BASE.collect_logits(model, heedb_test_loader, device)
        last_auc_internal, last_df_internal = evaluate_performance_detailed(
            y_true=y_test,
            y_logits=last_logits_internal,
            label_names=BASE.TARGET_LABELS,
            title="Last Full-Label Fixed-Prototype on HEEDB Test (Internal)",
            split_name="last_heedb_internal",
        )
        last_logits_mimic = BASE.collect_logits(model, mimic_loader, device)
        last_auc_mimic_full, last_df_mimic_full = evaluate_performance_detailed(
            y_true=y_mimic,
            y_logits=last_logits_mimic,
            label_names=BASE.TARGET_LABELS,
            title="Last Full-Label Fixed-Prototype on MIMIC Test (External Full)",
            split_name="last_mimic_external_full",
        )
        last_logits_mimic_split = BASE.collect_logits(model, mimic_split_loader, device)
        last_auc_mimic_split, last_df_mimic_split = evaluate_performance_detailed(
            y_true=y_mimic_split,
            y_logits=last_logits_mimic_split,
            label_names=BASE.TARGET_LABELS,
            title="Last Full-Label Fixed-Prototype on MIMIC Test (External Comparable Split)",
            split_name="last_mimic_external_split",
        )
    torch.save(model.state_dict(), BASE.LAST_MODEL_PATH)

    best_auc_internal = last_auc_internal
    best_auc_mimic_full = last_auc_mimic_full
    best_auc_mimic_split = last_auc_mimic_split
    best_df_internal = last_df_internal
    best_df_mimic_full = last_df_mimic_full
    best_df_mimic_split = last_df_mimic_split
    if Path(BASE.BEST_MODEL_PATH).exists():
        print(f"\n[Eval] Loading best checkpoint from epoch {best_epoch} ...")
        best_state = torch.load(BASE.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(best_state)

        with torch.no_grad():
            best_logits_internal = BASE.collect_logits(model, heedb_test_loader, device)
            best_auc_internal, best_df_internal = evaluate_performance_detailed(
                y_true=y_test,
                y_logits=best_logits_internal,
                label_names=BASE.TARGET_LABELS,
                title="Best Full-Label Fixed-Prototype on HEEDB Test (Internal)",
                split_name="best_heedb_internal",
            )
            best_logits_mimic = BASE.collect_logits(model, mimic_loader, device)
            best_auc_mimic_full, best_df_mimic_full = evaluate_performance_detailed(
                y_true=y_mimic,
                y_logits=best_logits_mimic,
                label_names=BASE.TARGET_LABELS,
                title="Best Full-Label Fixed-Prototype on MIMIC Test (External Full)",
                split_name="best_mimic_external_full",
            )
            best_logits_mimic_split = BASE.collect_logits(model, mimic_split_loader, device)
            best_auc_mimic_split, best_df_mimic_split = evaluate_performance_detailed(
                y_true=y_mimic_split,
                y_logits=best_logits_mimic_split,
                label_names=BASE.TARGET_LABELS,
                title="Best Full-Label Fixed-Prototype on MIMIC Test (External Comparable Split)",
                split_name="best_mimic_external_split",
            )
        torch.save(model.state_dict(), BASE.BEST_SNAPSHOT_PATH)

    mapped_indices = BASE.project_prototypes(
        model,
        x_train,
        y_train,
        device,
        BASE.EVAL_BATCH_SIZE,
        update_model=False,
    )
    prototype_report_df = BASE.build_prototype_report(model, mapped_indices, train_idx)
    heedb_activation_df = BASE.analyze_prototype_activation(model, heedb_test_loader, device)
    mimic_activation_df = BASE.analyze_prototype_activation(model, mimic_loader, device)

    prototype_report_df.to_csv(BASE.PROTOTYPE_REPORT_CSV, index=False)
    heedb_activation_df.to_csv(BASE.HEEDB_ACTIVATION_CSV, index=False)
    mimic_activation_df.to_csv(BASE.MIMIC_ACTIVATION_CSV, index=False)

    BASE.print_prototype_report(prototype_report_df, title="Prototype Sample Mapping (HEEDB Train Anchors)")
    BASE.print_activation_report(
        heedb_activation_df,
        title="Prototype Activation on HEEDB Test",
        top_k=min(20, len(heedb_activation_df)),
    )
    BASE.print_activation_report(
        mimic_activation_df,
        title="Prototype Activation on MIMIC Test",
        top_k=min(20, len(mimic_activation_df)),
    )

    BASE.export_prototype_tsne(model, x_train, y_train, device, BASE.PROTOTYPE_TSNE_PNG, BASE.TSNE_SAMPLE_SIZE)

    prototype_bank_x, prototype_bank_df = BASE.build_prototype_candidate_bank(
        x_train=x_train,
        train_df=df_train,
        prototype_report_df=prototype_report_df,
        activation_report_df=heedb_activation_df,
    )

    if BASE.has_exportable_waveform_paths(df_test):
        heedb_json_summary = BASE.export_embedding_showcase(
            model=model,
            x_bank=prototype_bank_x,
            bank_df=prototype_bank_df,
            x_query=x_test,
            y_query=y_test,
            query_df=df_test,
            output_base=BASE.JSON_EXPORT_DIR_HEEDB,
            device=device,
            query_count=BASE.JSON_EXPORT_QUERY_COUNT,
            query_per_class=BASE.JSON_EXPORT_QUERY_PER_CLASS,
            top_k=BASE.JSON_EXPORT_TOP_K,
            chunk_size=BASE.EVAL_BATCH_SIZE,
        )
    else:
        heedb_json_summary = {
            "exported_queries": 0,
            "manifest_path": str(Path(BASE.JSON_EXPORT_DIR_HEEDB) / "manifest.json"),
            "skipped_reason": "HEEDB robust metadata still has no usable waveform path columns",
        }

    mimic_json_summary = BASE.export_embedding_showcase(
        model=model,
        x_bank=prototype_bank_x,
        bank_df=prototype_bank_df,
        x_query=x_mimic,
        y_query=y_mimic,
        query_df=mimic_df,
        output_base=BASE.JSON_EXPORT_DIR_MIMIC,
        device=device,
        query_count=BASE.JSON_EXPORT_QUERY_COUNT,
        query_per_class=BASE.JSON_EXPORT_QUERY_PER_CLASS,
        top_k=BASE.JSON_EXPORT_TOP_K,
        chunk_size=BASE.EVAL_BATCH_SIZE,
    )

    per_class_df = pd.concat(
        [
            best_df_internal,
            best_df_mimic_full,
            best_df_mimic_split,
        ],
        ignore_index=True,
    )
    per_class_csv_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(per_class_csv_path, index=False)

    summary_payload = {
        "experiment_name": EXPERIMENT_NAME,
        "label_config_csv": label_config["config_path"],
        "label_count": len(BASE.TARGET_LABELS),
        "label_selection": label_selection,
        "max_epochs": int(args.max_epochs),
        "prototype_budget": args.prototype_budget,
        "min_prototypes_per_class": args.min_prototypes_per_class,
        "max_prototypes_per_class": args.max_prototypes_per_class,
        "total_prototypes": int(sum(prototype_counts)),
        "best_epoch": int(best_epoch),
        "support_summary": support_summary,
        "metrics": {
            "baseline": {
                "heedb_internal_macro_auroc": baseline_auc_internal,
                "mimic_external_full_macro_auroc": baseline_auc_mimic_full,
                "mimic_external_split_macro_auroc": baseline_auc_mimic_split,
            },
            "last": {
                "heedb_internal_macro_auroc": last_auc_internal,
                "mimic_external_full_macro_auroc": last_auc_mimic_full,
                "mimic_external_split_macro_auroc": last_auc_mimic_split,
            },
            "best": {
                "heedb_internal_macro_auroc": best_auc_internal,
                "mimic_external_full_macro_auroc": best_auc_mimic_full,
                "mimic_external_split_macro_auroc": best_auc_mimic_split,
            },
        },
        "artifacts": {
            "summary_json": str(summary_json_path),
            "per_class_auc_csv": str(per_class_csv_path),
            "target_label_manifest_json": str(label_manifest_path),
            "prototype_count_plan_csv": str(prototype_plan_csv_path),
            "prototype_report_csv": str(BASE.PROTOTYPE_REPORT_CSV),
            "heedb_activation_csv": str(BASE.HEEDB_ACTIVATION_CSV),
            "mimic_activation_csv": str(BASE.MIMIC_ACTIVATION_CSV),
            "tsne_png": str(BASE.PROTOTYPE_TSNE_PNG),
            "best_checkpoint": str(BASE.BEST_MODEL_PATH),
            "best_snapshot_checkpoint": str(BASE.BEST_SNAPSHOT_PATH),
            "last_checkpoint": str(BASE.LAST_MODEL_PATH),
            "heedb_json_manifest": heedb_json_summary["manifest_path"],
            "mimic_json_manifest": mimic_json_summary["manifest_path"],
        },
    }
    save_json(summary_json_path, summary_payload)

    print("\n=== Summary ===")
    print("-" * 110)
    print(f"{'Dataset':<32} | {'Baseline':<10} | {'Last':<10} | {'Best':<10}")
    print("-" * 110)
    print(
        f"{'HEEDB (Internal)':<32} | {baseline_auc_internal:.4f}     | "
        f"{last_auc_internal:.4f}     | {best_auc_internal:.4f}"
    )
    print(
        f"{'MIMIC (External Full)':<32} | {baseline_auc_mimic_full:.4f}     | "
        f"{last_auc_mimic_full:.4f}     | {best_auc_mimic_full:.4f}"
    )
    print(
        f"{'MIMIC (External Comparable)':<32} | {baseline_auc_mimic_split:.4f}     | "
        f"{last_auc_mimic_split:.4f}     | {best_auc_mimic_split:.4f}"
    )
    print("-" * 110)
    print(f"Label count:                {len(BASE.TARGET_LABELS)}")
    print(f"Prototype budget:           {args.prototype_budget} (min={args.min_prototypes_per_class}, max={args.max_prototypes_per_class})")
    print(f"Total prototypes:           {sum(prototype_counts)}")
    print(f"Zero-positive train labels: {support_summary['train']['zero_positive_label_count']}")
    print(f"Train positive <6 labels:   {support_summary['train']['positive_support_lt6_count']}")
    print(f"Best validation epoch:      {best_epoch}")
    print(f"Summary JSON:               {summary_json_path}")
    print(f"Per-class CSV:              {per_class_csv_path}")
    print(f"Label manifest:             {label_manifest_path}")
    print(f"Prototype count plan:       {prototype_plan_csv_path}")


if __name__ == "__main__":
    main()
