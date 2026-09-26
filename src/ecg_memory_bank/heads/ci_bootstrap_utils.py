"""Prototype AUROC confidence intervals.

Purpose
-------
Compute dataset/task-level and per-class uncertainty for the trained prototype
and linear heads.

Method
------
Resample evaluation rows, skip invalid single-class replicates, and aggregate
macro/micro or binary AUROC.

Inputs
------
Case-level binary labels and model scores from a fixed evaluation split.

Outputs
-------
JSON/CSV CI reports written only when called by a task entry.

Paper parameters
----------------
Paper runs use 1000 bootstrap draws and seed=42; class denominator is
determined by the corresponding task.

Shapes and meaning
------------------
Label/score arrays are [Q,C], or [Q] for binary Chagas; each resample keeps
paired label/score rows together.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def bootstrap_index_matrix(n_samples: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_samples, size=(n_bootstrap, n_samples), dtype=np.int32)


def percentile_ci(point_estimate: float, scores: list[float]) -> tuple[float | None, float | None, int]:
    if not scores:
        return None, None, 0
    arr = np.asarray(scores, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(arr.size)


def bootstrap_binary_auc_ci(y_true: np.ndarray, y_score: np.ndarray, index_matrix: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None, None, None, 0
    point_estimate = float(roc_auc_score(y_true, y_score))
    sampled_scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices]
        sampled_score = y_score[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        sampled_scores.append(float(roc_auc_score(sampled_true, sampled_score)))
    lower, upper, valid = percentile_ci(point_estimate, sampled_scores)
    return point_estimate, lower, upper, valid


def bootstrap_macro_auc_ci(y_true: np.ndarray, y_score: np.ndarray, index_matrix: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    from sklearn.metrics import roc_auc_score

    valid_class_indices = [idx for idx in range(y_true.shape[1]) if len(np.unique(y_true[:, idx])) >= 2]
    if not valid_class_indices:
        return None, None, None, 0
    point_estimate = float(np.mean([roc_auc_score(y_true[:, idx], y_score[:, idx]) for idx in valid_class_indices]))
    sampled_scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices]
        sampled_score = y_score[indices]
        fold_scores: list[float] = []
        for class_idx in valid_class_indices:
            if len(np.unique(sampled_true[:, class_idx])) < 2:
                continue
            fold_scores.append(float(roc_auc_score(sampled_true[:, class_idx], sampled_score[:, class_idx])))
        if fold_scores:
            sampled_scores.append(float(np.mean(fold_scores)))
    lower, upper, valid = percentile_ci(point_estimate, sampled_scores)
    return point_estimate, lower, upper, valid


def bootstrap_micro_auc_ci(y_true: np.ndarray, y_score: np.ndarray, index_matrix: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    from sklearn.metrics import roc_auc_score

    flat_true = y_true.ravel()
    flat_score = y_score.ravel()
    if len(np.unique(flat_true)) < 2:
        return None, None, None, 0
    point_estimate = float(roc_auc_score(flat_true, flat_score))
    sampled_scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices].ravel()
        sampled_score = y_score[indices].ravel()
        if len(np.unique(sampled_true)) < 2:
            continue
        sampled_scores.append(float(roc_auc_score(sampled_true, sampled_score)))
    lower, upper, valid = percentile_ci(point_estimate, sampled_scores)
    return point_estimate, lower, upper, valid


def build_multilabel_ci_report(
    *,
    class_names: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    index_matrix = bootstrap_index_matrix(int(y_true.shape[0]), int(bootstrap), int(seed))
    macro_auc, macro_lower, macro_upper, macro_valid = bootstrap_macro_auc_ci(y_true, y_score, index_matrix)
    micro_auc, micro_lower, micro_upper, micro_valid = bootstrap_micro_auc_ci(y_true, y_score, index_matrix)

    per_class_rows: list[dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        class_true = y_true[:, class_idx]
        class_score = y_score[:, class_idx]
        point, lower, upper, valid = bootstrap_binary_auc_ci(class_true, class_score, index_matrix)
        positive_count = int(class_true.sum())
        negative_count = int(len(class_true) - positive_count)
        per_class_rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "eligible_for_ci": point is not None,
                "auroc": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "valid_bootstraps": valid,
            }
        )

    return {
        "bootstrap_iterations": int(bootstrap),
        "seed": int(seed),
        "summary": {
            "sample_count": int(y_true.shape[0]),
            "label_count": int(y_true.shape[1]),
            "macro_auc": macro_auc,
            "macro_ci95_lower": macro_lower,
            "macro_ci95_upper": macro_upper,
            "macro_valid_bootstraps": macro_valid,
            "micro_auc": micro_auc,
            "micro_ci95_lower": micro_lower,
            "micro_ci95_upper": micro_upper,
            "micro_valid_bootstraps": micro_valid,
        },
        "per_class": per_class_rows,
    }


def write_multilabel_ci_outputs(
    *,
    output_dir: str | Path,
    metric_key: str,
    report_payload: dict[str, Any],
) -> dict[str, str]:
    ci_dir = Path(output_dir)
    ci_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ci_dir / f"per_class_auc_{metric_key}_ci_report.csv"
    json_path = ci_dir / f"per_class_auc_{metric_key}_ci_report.json"
    rows = list(report_payload["per_class"])
    fieldnames = list(rows[0].keys()) if rows else [
        "class_idx",
        "class_name",
        "positive_count",
        "negative_count",
        "eligible_for_ci",
        "auroc",
        "ci95_lower",
        "ci95_upper",
        "valid_bootstraps",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(json_path, report_payload)
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
