#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Retrieval bootstrap utilities.

Purpose
-------
Compute class, micro, macro, and mean confidence intervals from saved
retrieval evaluation inputs.

Method
------
Reuse paired bootstrap index matrices and skip invalid class resamples that
lack both outcome classes.

Inputs
------
Per-query labels/scores, retrieval metric arrays, and evaluation manifest
paths.

Outputs
-------
Per-class and aggregate percentile CI artifacts under caller-selected output
directories.

Paper parameters
----------------
Default paper bootstrap is 1000 resamples with seed=42; worker count changes
speed only.

Shapes and meaning
------------------
Binary labels and scores align as [Q,C]; query-level metric vectors have
length Q.
"""


from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ecg_memory_bank.retrieval.metrics_utils import to_jsonable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast bootstrap CI for Finaltabel retrieval experiments from saved ci_artifacts."
    )
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario_name filter. Can be passed multiple times to run only selected scenarios.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def bootstrap_index_matrix(n_samples: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_samples, size=(n_bootstrap, n_samples), dtype=np.int32)


def percentile_ci(point_estimate: float, scores: list[float]) -> tuple[float, float, int]:
    if not scores:
        return float("nan"), float("nan"), 0
    score_arr = np.asarray(scores, dtype=np.float64)
    return float(np.quantile(score_arr, 0.025)), float(np.quantile(score_arr, 0.975)), int(score_arr.size)


def bootstrap_mean_ci(values: np.ndarray, index_matrix: np.ndarray) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    point_estimate = float(np.nanmean(values)) if finite_mask.any() else float("nan")
    if not finite_mask.any():
        return point_estimate, float("nan"), float("nan"), 0
    sampled = values[index_matrix]
    sampled_means = np.nanmean(sampled, axis=1)
    sampled_means = sampled_means[np.isfinite(sampled_means)]
    lower, upper, valid = percentile_ci(point_estimate, sampled_means.tolist())
    return point_estimate, lower, upper, valid


def bootstrap_micro_auc_ci(y_true: np.ndarray, y_score: np.ndarray, index_matrix: np.ndarray) -> tuple[float, float, float, int]:
    flat_true = y_true.ravel()
    flat_score = y_score.ravel()
    if len(np.unique(flat_true)) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    point_estimate = float(roc_auc_score(flat_true, flat_score))
    scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices].ravel()
        sampled_score = y_score[indices].ravel()
        if len(np.unique(sampled_true)) < 2:
            continue
        scores.append(float(roc_auc_score(sampled_true, sampled_score)))
    lower, upper, valid = percentile_ci(point_estimate, scores)
    return point_estimate, lower, upper, valid


def bootstrap_macro_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    valid_class_indices: list[int],
    index_matrix: np.ndarray,
) -> tuple[float, float, float, int]:
    if not valid_class_indices:
        return float("nan"), float("nan"), float("nan"), 0
    point_estimate = float(
        np.mean([roc_auc_score(y_true[:, idx], y_score[:, idx]) for idx in valid_class_indices])
    )
    scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices]
        sampled_score = y_score[indices]
        fold_scores = []
        for class_idx in valid_class_indices:
            if len(np.unique(sampled_true[:, class_idx])) < 2:
                continue
            fold_scores.append(float(roc_auc_score(sampled_true[:, class_idx], sampled_score[:, class_idx])))
        if fold_scores:
            scores.append(float(np.mean(fold_scores)))
    lower, upper, valid = percentile_ci(point_estimate, scores)
    return point_estimate, lower, upper, valid


def bootstrap_binary_auc_ci(y_true: np.ndarray, y_score: np.ndarray, index_matrix: np.ndarray) -> tuple[float, float, float, int]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    point_estimate = float(roc_auc_score(y_true, y_score))
    scores: list[float] = []
    for indices in index_matrix:
        sampled_true = y_true[indices]
        sampled_score = y_score[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        scores.append(float(roc_auc_score(sampled_true, sampled_score)))
    lower, upper, valid = percentile_ci(point_estimate, scores)
    return point_estimate, lower, upper, valid


def normalize_targets(input_root: Path, raw_targets: list[str]) -> list[str]:
    if "all" in raw_targets:
        families = []
        for child in input_root.iterdir():
            if child.is_dir() and list(child.glob("*/ci_artifacts/ci_manifest.json")):
                families.append(child.name)
        return sorted(families)
    return raw_targets


def family_manifest_paths(input_root: Path, family: str) -> list[Path]:
    family_dir = input_root / family
    if not family_dir.exists():
        return []
    return sorted(family_dir.glob("*/ci_artifacts/ci_manifest.json"))


def filter_manifest_paths(manifest_paths: list[Path], scenario_filters: list[str]) -> list[Path]:
    if not scenario_filters:
        return manifest_paths
    allowed = {value.strip() for value in scenario_filters if value and value.strip()}
    if not allowed:
        return manifest_paths
    filtered: list[Path] = []
    for manifest_path in manifest_paths:
        if manifest_path.parent.parent.name in allowed:
            filtered.append(manifest_path)
    return filtered


def build_metric_rows(manifest: dict[str, Any], query_metrics_npz: Any, index_matrix: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for metric_name in manifest.get("query_metric_names", []):
        if metric_name not in query_metrics_npz:
            continue
        point_estimate, lower, upper, valid = bootstrap_mean_ci(query_metrics_npz[metric_name], index_matrix)
        rows.append(
            {
                "scenario_name": manifest["scenario_name"],
                "family": manifest["family"],
                "metric_name": metric_name,
                "point_estimate": point_estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "valid_bootstraps": valid,
            }
        )
    return rows


def build_per_class_report(
    metric_key: str,
    payload_path: str,
    scenario_output_dir: Path,
    index_matrix: np.ndarray,
    jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = np.load(payload_path, allow_pickle=False)
    y_true = payload["y_true"].astype(np.uint8, copy=False)
    y_score = payload["y_score"].astype(np.float32, copy=False)
    class_names = [str(item) for item in payload["class_names"].tolist()]

    valid_class_indices = [idx for idx in range(y_true.shape[1]) if 0 < int(y_true[:, idx].sum()) < len(y_true)]
    macro_auc, macro_lower, macro_upper, macro_valid = bootstrap_macro_auc_ci(
        y_true, y_score, valid_class_indices, index_matrix
    )
    micro_auc, micro_lower, micro_upper, micro_valid = bootstrap_micro_auc_ci(y_true, y_score, index_matrix)

    def process_class(class_idx: int) -> dict[str, Any]:
        y_true_col = y_true[:, class_idx]
        y_score_col = y_score[:, class_idx]
        positive_count = int(y_true_col.sum())
        negative_count = int(len(y_true_col) - positive_count)
        eligible = positive_count > 0 and negative_count > 0
        if eligible:
            auc, lower, upper, valid = bootstrap_binary_auc_ci(y_true_col, y_score_col, index_matrix)
        else:
            auc, lower, upper, valid = float("nan"), float("nan"), float("nan"), 0
        return {
            "class_idx": class_idx,
            "class_name": class_names[class_idx],
            "positive_count": positive_count,
            "negative_count": negative_count,
            "eligible_for_ci": eligible,
            "auroc": auc,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "valid_bootstraps": valid,
        }

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        rows = list(executor.map(process_class, range(len(class_names))))

    csv_path = scenario_output_dir / f"per_class_auc_{metric_key}_ci_report.csv"
    json_path = scenario_output_dir / f"per_class_auc_{metric_key}_ci_report.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report_payload = {
        "metric_key": metric_key,
        "summary": {
            "macro_auc": macro_auc,
            "macro_ci95_lower": macro_lower,
            "macro_ci95_upper": macro_upper,
            "macro_valid_bootstraps": macro_valid,
            "micro_auc": micro_auc,
            "micro_ci95_lower": micro_lower,
            "micro_ci95_upper": micro_upper,
            "micro_valid_bootstraps": micro_valid,
            "valid_class_count": len(valid_class_indices),
            "sample_count": int(y_true.shape[0]),
        },
        "per_class": rows,
    }
    json_path.write_text(json.dumps(to_jsonable(report_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows = [
        {
            "metric_name": f"{metric_key}_macro_auc",
            "point_estimate": macro_auc,
            "ci95_lower": macro_lower,
            "ci95_upper": macro_upper,
            "valid_bootstraps": macro_valid,
        },
        {
            "metric_name": f"{metric_key}_micro_auc",
            "point_estimate": micro_auc,
            "ci95_lower": micro_lower,
            "ci95_upper": micro_upper,
            "valid_bootstraps": micro_valid,
        },
    ]
    report_refs = [{"metric_key": metric_key, "csv_path": str(csv_path), "json_path": str(json_path)}]
    return metric_rows, report_refs


def run_scenario_ci(manifest_path: Path, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_output_dir = output_root / manifest["family"] / manifest["scenario_name"]
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    query_metrics_npz = np.load(manifest["query_metrics_npz"], allow_pickle=False)
    sample_count = None
    for metric_name in manifest.get("query_metric_names", []):
        if metric_name in query_metrics_npz:
            sample_count = int(len(query_metrics_npz[metric_name]))
            break
    if sample_count is None:
        auc_inputs = manifest.get("auc_input_npz", {})
        if auc_inputs:
            first_payload = np.load(next(iter(auc_inputs.values())), allow_pickle=False)
            sample_count = int(first_payload["y_true"].shape[0])
    if sample_count is None:
        raise ValueError(f"Unable to infer sample count for CI manifest: {manifest_path}")

    index_matrix = bootstrap_index_matrix(sample_count, args.bootstrap, args.seed)
    metric_rows = build_metric_rows(manifest, query_metrics_npz, index_matrix)
    per_class_reports = []
    for metric_key, payload_path in manifest.get("auc_input_npz", {}).items():
        auc_rows, report_refs = build_per_class_report(
            metric_key=metric_key,
            payload_path=payload_path,
            scenario_output_dir=scenario_output_dir,
            index_matrix=index_matrix,
            jobs=args.jobs,
        )
        for row in auc_rows:
            metric_rows.append(
                {
                    "scenario_name": manifest["scenario_name"],
                    "family": manifest["family"],
                    **row,
                }
            )
        per_class_reports.extend(report_refs)

    summary_csv_path = scenario_output_dir / f"{manifest['scenario_name']}_ci_summary.csv"
    summary_json_path = scenario_output_dir / f"{manifest['scenario_name']}_ci_summary.json"
    pd.DataFrame(metric_rows).to_csv(summary_csv_path, index=False)
    summary_payload = {
        "family": manifest["family"],
        "scenario_name": manifest["scenario_name"],
        "bootstrap_iterations": int(args.bootstrap),
        "sample_count": sample_count,
        "metrics": metric_rows,
        "source_manifest": str(manifest_path),
        "per_class_reports": per_class_reports,
    }
    summary_json_path.write_text(json.dumps(to_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "family": manifest["family"],
        "scenario_name": manifest["scenario_name"],
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "per_class_reports": per_class_reports,
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    families = normalize_targets(input_root, args.targets)
    combined = {"targets": []}
    for family in families:
        results = []
        manifest_paths = family_manifest_paths(input_root, family)
        manifest_paths = filter_manifest_paths(manifest_paths, args.scenario)
        for manifest_path in manifest_paths:
            results.append(run_scenario_ci(manifest_path, output_root, args))
        combined["targets"].append({"family": family, "scenarios": results})

    combined_path = output_root / "combined_ci_summary.json"
    combined_path.write_text(json.dumps(to_jsonable(combined), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] combined CI summary -> {combined_path}")


if __name__ == "__main__":
    main()
