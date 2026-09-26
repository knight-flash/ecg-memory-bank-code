#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helper for Finaltabel fixed-prototype MIMIC 18-label wrappers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


MIMIC18_LABELS = [
    "Atrial fibrillation/flutter",
    "Supraventricular tachycardia",
    "Sinus tachycardia",
    "Sinus bradycardia",
    "Premature ventricular complex",
    "Premature atrial complex",
    "Pacemaker rhythm",
    "Atrioventricular block",
    "1st degree atrioventricular block",
    "Right bundle branch block",
    "Left bundle branch block",
    "Left ventricular hypertrophy",
    "ST-T change",
    "Prolonged QT interval",
    "Low QRS voltage",
    "Borderline ECG",
    "Abnormal ECG",
    "Normal ECG",
]

MIMIC18_GROUPS = ["MIMIC 18-label Fixed Protocol"] * len(MIMIC18_LABELS)
NORMAL_LABEL = "Normal ECG"
MIMIC18_ALIASES = {
    "Atrial fibrillation/flutter": [
        "atrial fibrillation/flutter",
        "atrial fibrillation",
        "atrial flutter",
        "afib",
        "a-fib",
        "aflutter",
        "with rapid ventricular response",
        "with slow ventricular response",
        "with variable av block",
    ],
    "Supraventricular tachycardia": [
        "supraventricular tachycardia",
        "svt",
        "psvt",
        "atrial tachycardia",
        "av nodal reentrant tachycardia",
        "avnrt",
        "avrt",
        "ectopic atrial tachycardia",
        "junctional tachycardia",
    ],
    "Sinus tachycardia": ["sinus tachycardia", "sinus tach"],
    "Sinus bradycardia": ["sinus bradycardia", "sinus brady", "marked sinus bradycardia"],
    "Premature ventricular complex": [
        "premature ventricular complex",
        "premature ventricular complexes",
        "pvc",
        "ventricular premature",
        "vpc",
        "ventricular ectopics",
        "ventricular bigeminy",
        "ventricular trigeminy",
        "with premature ventricular or aberrantly conducted complexes",
        "fusion complexes",
        "in a pattern of bigeminy",
    ],
    "Premature atrial complex": [
        "premature atrial complex",
        "premature atrial complexes",
        "pac",
        "atrial premature",
        "apc",
        "atrial ectopics",
        "atrial bigeminy",
        "premature supraventricular complexes",
        "supraventricular complexes",
    ],
    "Pacemaker rhythm": [
        "pacemaker",
        "paced rhythm",
        "pacing",
        "atrial paced",
        "ventricular paced",
        "dual paced",
        "ventricular-paced rhythm",
        "ventricular-paced complexes",
        "atrial-paced rhythm",
        "atrial-paced complexes",
        "atrial-sensed ventricular-paced rhythm",
        "av dual-paced rhythm",
        "av dual-paced complexes",
        "av sequential or dual chamber electronic pacemaker",
        "electronic atrial pacemaker",
        "electronic ventricular pacemaker",
    ],
    "Atrioventricular block": [
        "atrioventricular block",
        "av block",
        "avb",
        "heart block",
        "second degree",
        "third degree",
        "complete heart block",
        "mobitz",
        "wenckebach",
        "with 2nd degree av block",
        "with 2nd degree av block (mobitz i)",
        "with 2nd degree av block (mobitz ii)",
        "with complete heart block",
        "with 2:1 av conduction",
        "with 3:1 av conduction",
        "with 4:1 av conduction",
        "with 5:1 av conduction",
        "with av dissociation",
        "with retrograde conduction",
    ],
    "1st degree atrioventricular block": [
        "1st degree atrioventricular block",
        "with 1st degree av block",
        "1st degree av block",
        "first degree av block",
        "1st degree heart block",
        "first degree heart block",
        "prolonged pr interval",
        "1-avb",
        "grade 1 av block",
        "prolonged av conduction",
        "with prolonged av conduction",
    ],
    "Right bundle branch block": [
        "right bundle branch block",
        "rbbb",
        "incomplete right bundle branch block",
        "incomplete rbbb",
        "complete rbbb",
        "rsr' or qr pattern in v1 suggests right ventricular conduction delay",
    ],
    "Left bundle branch block": [
        "left bundle branch block",
        "lbbb",
        "incomplete left bundle branch block",
        "incomplete lbbb",
        "complete lbbb",
    ],
    "Left ventricular hypertrophy": [
        "left ventricular hypertrophy",
        "lvh",
        "left ventricular enlargement",
        "minimal voltage criteria for lvh",
        "moderate voltage criteria for lvh",
        "voltage criteria for left ventricular hypertrophy",
        "cornell product",
        "r in avl",
        "sokolow-lyon",
        "romhilt-estes",
        "with strain pattern",
        "with repolarization abnormality",
    ],
    "ST-T change": [
        "st-t change",
        "st-t abnormality",
        "st-t wave changes",
        "nonspecific t wave abnormality",
        "nonspecific st and t wave abnormality",
        "nonspecific st abnormality",
        "t wave abnormality",
        "t wave inversion",
        "st depression",
        "st elevation",
        "repolarization abnormality",
        "early repolarization",
        "ischemia",
        "injury",
        "infarct",
        "acute mi",
        "stemi",
        "abnormal qrs-t angle",
        "acute pericarditis",
        "t wave abnormality, consider lateral ischemia",
        "t wave abnormality, consider inferior ischemia",
        "t wave abnormality, consider anterior ischemia",
        "t wave abnormality, consider anterolateral ischemia",
        "t wave abnormality, consider inferolateral ischemia",
    ],
    "Prolonged QT interval": [
        "prolonged qt interval",
        "prolonged qt",
        "long qt",
        "qt prolongation",
        "qt interval prolonged",
        "qtcb >=",
        "qtcf >=",
    ],
    "Low QRS voltage": [
        "low qrs voltage",
        "low voltage",
        "low amplitude qrs",
        "diminished qrs",
        "low voltage qrs",
    ],
    "Borderline ECG": [
        "borderline ecg",
        "borderline electrocardiogram",
    ],
    "Abnormal ECG": [
        "abnormal ecg",
        "abnormal electrocardiogram",
    ],
    "Normal ECG": [
        "normal ecg",
        "normal electrocardiogram",
        "within normal limits",
        "otherwise normal ecg",
        "normal sinus rhythm",
        "sinus rhythm",
    ],
}


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_base_target_labels(base_module, fixed_prototypes_per_class: int | None = None) -> dict[str, Any]:
    prompts = [f"An ECG recording showing {label}." for label in MIMIC18_LABELS]
    normalize = getattr(base_module, "normalize_label_text")
    alias_sets = [{normalize(alias) for alias in MIMIC18_ALIASES[label]} for label in MIMIC18_LABELS]
    normal_idx = MIMIC18_LABELS.index(NORMAL_LABEL)

    base_module.TARGET_LABELS = list(MIMIC18_LABELS)
    base_module.TARGET_PROMPTS = list(prompts)
    base_module.TARGET_ALIAS_SETS = list(alias_sets)
    base_module.TARGET_KEYWORDS = [sorted(alias_set) for alias_set in alias_sets]
    base_module.NORMAL_LABEL = NORMAL_LABEL
    base_module.NORMAL_CLASS_IDX = normal_idx
    if fixed_prototypes_per_class is not None:
        base_module.DEFAULT_PROTOTYPES_PER_CLASS = int(fixed_prototypes_per_class)
    return build_task_payload()


def build_task_payload() -> dict[str, Any]:
    normal_idx = MIMIC18_LABELS.index(NORMAL_LABEL)
    manifest_rows = []
    for class_idx, (label_name, group_name) in enumerate(zip(MIMIC18_LABELS, MIMIC18_GROUPS)):
        manifest_rows.append(
            {
                "class_idx": class_idx,
                "canonical_label": label_name,
                "source_column": label_name,
                "prompt": f"An ECG recording showing {label_name}.",
                "group": group_name,
                "alias_count": 1,
                "aliases": [label_name],
            }
        )
    return {
        "task_name": "mimic18_fixed",
        "columns": list(MIMIC18_LABELS),
        "labels": list(MIMIC18_LABELS),
        "normal_idx": normal_idx,
        "manifest_rows": manifest_rows,
    }


def build_label_selection_reporting(task_payload: dict[str, Any], split_targets: dict[str, torch.Tensor]) -> dict[str, Any]:
    low_support_rows: list[dict[str, Any]] = []
    for class_idx, label_name in enumerate(task_payload["labels"]):
        positive_counts = {
            split_name: int(targets[:, class_idx].sum().item())
            for split_name, targets in split_targets.items()
        }
        if not all(count > 0 for count in positive_counts.values()):
            low_support_rows.append(
                {
                    "class_idx": class_idx,
                    "canonical_label": label_name,
                    "source_column": task_payload["columns"][class_idx],
                    "positive_counts": positive_counts,
                }
            )

    return {
        "strategy": "fixed_protocol_no_label_filtering",
        "required_splits": list(split_targets.keys()),
        "original_label_count": len(task_payload["labels"]),
        "selected_label_count": len(task_payload["labels"]),
        "dropped_label_count": 0,
        "dropped_labels": [],
        "low_support_label_count": len(low_support_rows),
        "low_support_labels": low_support_rows,
        "selected_labels": list(task_payload["labels"]),
        "selected_original_indices": list(range(len(task_payload["labels"]))),
    }


def build_prototype_count_rows(manifest_rows: list[dict[str, Any]], prototype_counts: list[int], y_train: torch.Tensor) -> list[dict[str, Any]]:
    positive_counts = y_train.sum(dim=0).cpu().numpy().astype(int).tolist()
    rows = []
    for item, proto_count, positive_count in zip(manifest_rows, prototype_counts, positive_counts):
        rows.append(
            {
                "class_idx": item["class_idx"],
                "canonical_label": item["canonical_label"],
                "source_column": item["source_column"],
                "group": item["group"],
                "alias_count": item["alias_count"],
                "fixed_prototype_count": int(proto_count),
                "train_positive_count": int(positive_count),
            }
        )
    return rows


def build_adaptive_prototype_count_plan(
    positive_counts: list[int],
    total_budget: int,
    min_protos: int = 1,
    max_protos: int = 100,
) -> list[int]:
    counts = [0] * len(positive_counts)
    active_indices = [i for i, count in enumerate(positive_counts) if count > 0]
    if not active_indices:
        return counts

    active_positives = [max(positive_counts[i], 1) for i in active_indices]
    total_positives = sum(active_positives)
    raw_alloc: dict[int, int] = {}

    for idx, pos in zip(active_indices, active_positives):
        raw_alloc[idx] = max(min_protos, round((pos / total_positives) * total_budget))

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


def evaluate_performance_benchmark_compatible(
    base_module,
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    label_names: list[str],
    split_name: str,
    title: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_scores = torch.sigmoid(y_logits).detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    valid_aucs: list[float] = []
    flattened_true: list[np.ndarray] = []
    flattened_score: list[np.ndarray] = []

    print(f"\n=== {title} ===")
    print(f"{'Class':<45} | {'AUROC':<10} | {'Pos'}")
    print("-" * 78)

    for class_idx, label_name in enumerate(label_names):
        positive_count = int(np.sum(y_true_np[:, class_idx]))
        negative_count = int(len(y_true_np) - positive_count)
        eligible = positive_count > 0 and negative_count > 0
        auc_value: float | None = None
        status_text = "N/A"
        if eligible:
            try:
                auc_value = float(base_module.roc_auc_score(y_true_np[:, class_idx], y_scores[:, class_idx]))
                valid_aucs.append(auc_value)
                flattened_true.append(y_true_np[:, class_idx])
                flattened_score.append(y_scores[:, class_idx])
                status_text = f"{auc_value:.4f}"
            except Exception:
                status_text = "Error"

        print(f"{label_name:<45} | {status_text:<10} | {positive_count}")
        rows.append(
            {
                "split": split_name,
                "class_idx": class_idx,
                "class_name": label_name,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "eligible_for_auc": bool(eligible),
                "auroc": auc_value,
            }
        )

    macro_auroc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    if flattened_true:
        micro_auroc = float(
            base_module.roc_auc_score(
                np.concatenate(flattened_true, axis=0),
                np.concatenate(flattened_score, axis=0),
            )
        )
    else:
        micro_auroc = 0.0

    print("-" * 78)
    print(f"[{title}] Mean AUROC: {macro_auroc:.4f}")

    metrics = {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "valid_class_count": int(sum(row["eligible_for_auc"] for row in rows)),
        "skipped_class_count": int(sum(not row["eligible_for_auc"] for row in rows)),
    }
    return metrics, pd.DataFrame(rows)


def evaluate_on_loader_benchmark_compatible(
    base_module,
    model,
    dataloader,
    y_true: torch.Tensor,
    device: torch.device,
    label_names: list[str],
    title: str,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    logits = base_module.collect_logits(model, dataloader, device)
    return evaluate_performance_benchmark_compatible(
        base_module=base_module,
        y_true=y_true,
        y_logits=logits,
        label_names=label_names,
        split_name=split_name,
        title=title,
    )


def write_label_manifest(
    path: str | Path,
    *,
    task_payload: dict[str, Any],
    label_selection: dict[str, Any],
    split_counts: dict[str, int],
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload = {
        "task_name": task_payload["task_name"],
        "label_selection": label_selection,
        "labels": task_payload["manifest_rows"],
        "split_counts": split_counts,
    }
    if extra_payload:
        payload.update(extra_payload)
    save_json(path, payload)


def write_prototype_plan(path: str | Path, manifest_rows: list[dict[str, Any]], prototype_counts: list[int], y_train: torch.Tensor) -> None:
    pd.DataFrame(build_prototype_count_rows(manifest_rows, prototype_counts, y_train)).to_csv(path, index=False)


def summarize_split_counts(**counts: int) -> dict[str, int]:
    return {key: int(value) for key, value in counts.items()}
