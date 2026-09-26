#!/usr/bin/env python3
"""Fixed-label macro AUROC bootstrap.

Purpose
-------
Estimate broad case-addition macro AUROC and endpoint differences on one
common label denominator.

Method
------
Intersect evaluable classes across all eight memories and reuse paired query
bootstrap indices for each state.

Inputs
------
Eight scale-benchmark summaries and per-scenario AUROC input arrays from the
same fixed query cohort.

Outputs
-------
External per-fraction and endpoint 95% intervals, per-class statistics, and
the common-label manifest.

Paper parameters
----------------
Expected common labels are 114 for HEEDB or 83 for MIMIC; 1000 bootstrap
resamples and seed=42.

Shapes and meaning
------------------
Each scenario supplies target/score arrays [Q,C]; a paired resample draws the
same query rows at every fraction.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ecg_memory_bank.retrieval.metrics_utils import to_jsonable
from ecg_memory_bank.protocol import external_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "mimic_scale_benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CI_mimic_scale_fixed_labels_b1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute query-bootstrap CIs for MIMIC density macro AUROC on the "
            "common L3 label set."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--summary-file", default="mimic_scale_benchmark_summary.csv")
    parser.add_argument("--analysis-name", default="MIMIC self-split broad case-addition fixed-label macro AUROC")
    parser.add_argument("--level", choices=("L1", "L2", "L3"), default="L3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-label-count", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def auc_from_histograms(histograms: np.ndarray, score_bins: int) -> np.ndarray:
    """Return per-class AUROCs from negative/positive score-bin histograms."""
    negatives = histograms[:, :score_bins].astype(np.float64, copy=False)
    positives = histograms[:, score_bins:].astype(np.float64, copy=False)
    negative_before = np.cumsum(negatives, axis=1) - negatives
    wins = (positives * (negative_before + 0.5 * negatives)).sum(axis=1)
    denominators = positives.sum(axis=1) * negatives.sum(axis=1)
    aucs = np.full(len(denominators), np.nan, dtype=np.float64)
    valid = denominators > 0
    aucs[valid] = wins[valid] / denominators[valid]
    return aucs


def macro_auc_from_histogram(
    histogram: np.ndarray,
    score_bins: int,
) -> float:
    values = auc_from_histograms(histogram, score_bins)
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))


def ci95(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def load_scenarios(
    input_root: Path,
    summary_file: str,
    level: str,
) -> tuple[pd.DataFrame, list[str], list[sparse.csr_matrix], int, int]:
    summary_path = input_root / summary_file
    summary_df = pd.read_csv(summary_path).sort_values("gallery_fraction").reset_index(drop=True)
    if summary_df.empty:
        raise ValueError(f"No scale-benchmark rows found in {summary_path}")

    payloads: list[dict[str, Any]] = []
    class_sets: list[set[str]] = []
    for row in summary_df.itertuples(index=False):
        scenario_name = str(row.scenario)
        payload_path = input_root / scenario_name / "ci_artifacts" / f"{scenario_name}_{level.lower()}_auc_inputs.npz"
        payload = np.load(payload_path, allow_pickle=False)
        class_names = [str(name) for name in payload["class_names"].tolist()]
        payloads.append({"path": payload_path, "payload": payload, "class_names": class_names})
        class_sets.append(set(class_names))

    fixed_labels = sorted(set.intersection(*class_sets))
    if not fixed_labels:
        raise ValueError("No common L3 labels were found across the density points.")

    score_bins: int | None = None
    reference_truth: np.ndarray | None = None
    histogram_designs: list[sparse.csr_matrix] = []
    for item in payloads:
        payload = item["payload"]
        class_to_index = {name: index for index, name in enumerate(item["class_names"])}
        selected_indices = [class_to_index[name] for name in fixed_labels]
        y_true = payload["y_true"][:, selected_indices].astype(np.uint8, copy=False)
        y_score = payload["y_score"][:, selected_indices].astype(np.float32, copy=False)

        if reference_truth is None:
            reference_truth = y_true
        elif not np.array_equal(reference_truth, y_true):
            raise ValueError("The saved scenarios do not share an identical query-label matrix.")

        # Retrieval voting scores are proportions of the fixed top-10 neighbors.
        inferred_top_k = int(round(1.0 / np.min(y_score[y_score > 0]))) if np.any(y_score > 0) else 10
        bins = inferred_top_k + 1
        score_bin = np.rint(y_score * inferred_top_k).astype(np.uint8)
        if not np.allclose(y_score, score_bin / float(inferred_top_k), atol=1e-6):
            raise ValueError(f"Scores in {item['path']} are not discrete retrieval-voting proportions.")
        if score_bins is None:
            score_bins = bins
        elif score_bins != bins:
            raise ValueError("Density scenarios do not share the same retrieval-voting score scale.")

        class_offsets = np.arange(len(fixed_labels), dtype=np.int32) * (score_bins * 2)
        group_ids = score_bin.astype(np.int32) + y_true.astype(np.int32) * score_bins + class_offsets
        query_indices = np.repeat(np.arange(y_true.shape[0], dtype=np.int32), y_true.shape[1])
        # Each sparse-matrix row is one (label, truth, score) stratum. A matrix-vector
        # product gives all weighted bootstrap histograms without materializing repeats.
        histogram_designs.append(
            sparse.csr_matrix(
                (
                    np.ones(group_ids.size, dtype=np.uint8),
                    (group_ids.reshape(-1), query_indices),
                ),
                shape=(len(fixed_labels) * score_bins * 2, y_true.shape[0]),
                dtype=np.uint8,
            )
        )

    assert reference_truth is not None and score_bins is not None
    return summary_df, fixed_labels, histogram_designs, score_bins, reference_truth.shape[0]


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    input_root = Path(args.input_root)
    output_dir = external_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, fixed_labels, histogram_designs, score_bins, sample_count = load_scenarios(
        input_root, args.summary_file, args.level
    )
    class_count = len(fixed_labels)
    if class_count != args.expected_label_count:
        raise ValueError(f"Fixed-label set has {class_count} labels; expected {args.expected_label_count}")
    histogram_shape = (class_count, score_bins * 2)

    point_estimates = np.asarray(
        [
            macro_auc_from_histogram(design.sum(axis=1).A1.reshape(histogram_shape), score_bins)
            for design in histogram_designs
        ],
        dtype=np.float64,
    )

    # One resample per iteration is shared across density points for paired curve estimates.
    rng = np.random.RandomState(args.seed)
    bootstrap_values = np.empty((args.bootstrap, len(histogram_designs)), dtype=np.float32)
    per_class_bootstrap = np.empty(
        (args.bootstrap, len(histogram_designs), class_count), dtype=np.float32
    )

    def bootstrap_scenario(design: sparse.csr_matrix, weights: np.ndarray) -> np.ndarray:
        histogram = np.asarray(design @ weights).reshape(histogram_shape)
        return auc_from_histograms(histogram, score_bins)

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(histogram_designs))) as executor:
        for iteration in range(args.bootstrap):
            indices = rng.randint(0, sample_count, size=sample_count, dtype=np.int32)
            weights = np.bincount(indices, minlength=sample_count)
            values = list(executor.map(lambda design: bootstrap_scenario(design, weights), histogram_designs))
            per_class_bootstrap[iteration] = np.asarray(values, dtype=np.float32)
            bootstrap_values[iteration] = np.nanmean(per_class_bootstrap[iteration], axis=1)
            if (iteration + 1) % max(1, args.progress_every) == 0 or iteration + 1 == args.bootstrap:
                print(f"[Bootstrap] {iteration + 1}/{args.bootstrap}", flush=True)

    records = []
    for scenario_idx, row in enumerate(summary_df.itertuples(index=False)):
        lower, upper = ci95(bootstrap_values[:, scenario_idx])
        records.append(
            {
                "scenario": str(row.scenario),
                "gallery_fraction": float(row.gallery_fraction),
                "gallery_fraction_label": str(row.gallery_fraction_label),
                "num_gallery": int(row.num_gallery),
                "fixed_label_macro_auc": float(point_estimates[scenario_idx]),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_iterations": int(args.bootstrap),
            }
        )
    results_df = pd.DataFrame(records)

    point_per_class = [
        auc_from_histograms(
            np.asarray(design.sum(axis=1)).reshape(histogram_shape), score_bins
        )
        for design in histogram_designs
    ]
    per_class_records = []
    for scenario_idx, row in enumerate(summary_df.itertuples(index=False)):
        for class_idx, label in enumerate(fixed_labels):
            values = per_class_bootstrap[:, scenario_idx, class_idx]
            finite = values[np.isfinite(values)]
            lower, upper = ci95(finite) if finite.size else (float("nan"), float("nan"))
            per_class_records.append(
                {
                    "scenario": str(row.scenario),
                    "gallery_fraction": float(row.gallery_fraction),
                    "label": label,
                    "auroc": float(point_per_class[scenario_idx][class_idx]),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "valid_bootstraps": int(finite.size),
                }
            )

    delta_values = bootstrap_values[:, -1] - bootstrap_values[:, 0]
    delta_lower, delta_upper = ci95(delta_values)
    change = {
        "from_gallery_fraction": float(summary_df.iloc[0]["gallery_fraction"]),
        "to_gallery_fraction": float(summary_df.iloc[-1]["gallery_fraction"]),
        "fixed_label_macro_auc_change": float(point_estimates[-1] - point_estimates[0]),
        "ci95_lower": delta_lower,
        "ci95_upper": delta_upper,
        "bootstrap_iterations": int(args.bootstrap),
    }

    level_key = args.level.lower()
    summary_csv_path = output_dir / f"{level_key}_fixed_label_macro_auc_ci_summary.csv"
    summary_json_path = output_dir / f"{level_key}_fixed_label_macro_auc_ci_summary.json"
    labels_path = output_dir / f"{level_key}_fixed_label_manifest.csv"
    per_class_path = output_dir / f"{level_key}_fixed_label_per_class_auc_ci_summary.csv"
    samples_path = output_dir / f"{level_key}_fixed_label_macro_auc_bootstrap_samples.npz"
    results_df.to_csv(summary_csv_path, index=False)
    pd.DataFrame(per_class_records).to_csv(per_class_path, index=False)
    pd.DataFrame({"label": fixed_labels}).to_csv(labels_path, index=False)
    np.savez_compressed(
        samples_path,
        gallery_fraction=summary_df["gallery_fraction"].to_numpy(dtype=np.float64),
        macro_auc=bootstrap_values,
        per_class_auc=per_class_bootstrap,
        class_names=np.asarray(fixed_labels, dtype=str),
    )
    summary_json_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    "analysis": args.analysis_name,
                    "hierarchy_level": args.level,
                    "label_set_policy": "intersection of L3 labels represented in every density scenario",
                    "query_resampling": "paired nonparametric bootstrap over the fixed query cohort",
                    "bootstrap_iterations": int(args.bootstrap),
                    "seed": int(args.seed),
                    "jobs": int(args.jobs),
                    "query_count": int(sample_count),
                    "source_input_root": str(input_root),
                    "summary_csv": str(summary_csv_path),
                    "fixed_label_manifest_csv": str(labels_path),
                    "per_class_summary_csv": str(per_class_path),
                    "bootstrap_samples_npz": str(samples_path),
                    "results": records,
                    "endpoint_change": change,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Saved] summary CSV -> {summary_csv_path}")
    print(f"[Saved] summary JSON -> {summary_json_path}")


if __name__ == "__main__":
    main()
