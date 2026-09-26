#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Complex-query retrieval characterization.

Purpose
-------
Summarize retrieval coverage for ECGs carrying multiple mapped diagnoses.

Method
------
Partition saved query artifacts by label burden and compare coverage/precision
summaries without changing the retrieved memory.

Inputs
------
Existing scale-benchmark per-query arrays and scenario summaries.

Outputs
-------
External cohort tables describing performance by query complexity.

Paper parameters
----------------
Uses the same K and memory state as its input artifacts; no retraining or
resampling of embeddings.

Shapes and meaning
------------------
Query label burden is a count per ECG; subset metrics aggregate only queries
satisfying the cohort rule.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_memory_bank.retrieval.artifact_utils import format_fraction_label, format_fraction_tag, save_ci_artifacts
from ecg_memory_bank.retrieval.metrics_utils import to_jsonable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze union recall on simple vs complex cases using existing scale_benchmark outputs."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--compare-fractions", nargs=2, type=float, default=[0.1, 1.0])
    return parser.parse_args()


def load_scale_per_query(output_root: Path, fraction: float) -> tuple[str, pd.DataFrame]:
    fraction_tag = format_fraction_tag(fraction)
    scenario_name = f"heedb_train_gallery_{fraction_tag}_finalgallery_test"
    per_query_path = output_root / "scale_benchmark" / scenario_name / f"{scenario_name}_per_query.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"Missing scale_benchmark per-query CSV: {per_query_path}")
    return scenario_name, pd.read_csv(per_query_path)


def subset_array(df: pd.DataFrame, top_k: int, selector: str) -> np.ndarray:
    metric_col = f"union_recall_l3@{top_k}"
    if selector == "single_label":
        mask = df["l3_positive_count"] == 1
    elif selector == "multi_label_ge3":
        mask = df["l3_positive_count"] >= 3
    else:
        raise ValueError(f"Unsupported selector: {selector}")
    return df.loc[mask, metric_col].to_numpy(dtype=np.float32)


def build_summary_payload(
    *,
    group_name: str,
    top_k: int,
    lower_label: str,
    higher_label: str,
    lower_values: np.ndarray,
    higher_values: np.ndarray,
) -> dict[str, object]:
    limit = min(len(lower_values), len(higher_values))
    if limit == 0:
        delta_values = np.asarray([], dtype=np.float32)
    else:
        delta_values = higher_values[:limit] - lower_values[:limit]
    return {
        "group": group_name,
        "top_k": top_k,
        "comparison": {
            lower_label: {
                "count": int(len(lower_values)),
                "mean_union_recall_l3_at_top_k": float(np.mean(lower_values)) if len(lower_values) else float("nan"),
            },
            higher_label: {
                "count": int(len(higher_values)),
                "mean_union_recall_l3_at_top_k": float(np.mean(higher_values)) if len(higher_values) else float("nan"),
            },
        },
        "delta_100pct_minus_10pct": float(np.mean(delta_values)) if len(delta_values) else float("nan"),
    }, delta_values


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    family_output_dir = output_root / "complex_case_analysis"
    family_output_dir.mkdir(parents=True, exist_ok=True)

    lower_fraction, higher_fraction = [float(item) for item in args.compare_fractions]
    lower_label = format_fraction_label(lower_fraction)
    higher_label = format_fraction_label(higher_fraction)
    _, lower_df = load_scale_per_query(output_root, lower_fraction)
    _, higher_df = load_scale_per_query(output_root, higher_fraction)

    summary_rows = []
    tidy_rows = []
    for group_name in ("single_label", "multi_label_ge3"):
        lower_values = subset_array(lower_df, args.top_k, group_name)
        higher_values = subset_array(higher_df, args.top_k, group_name)
        summary, delta_values = build_summary_payload(
            group_name=group_name,
            top_k=args.top_k,
            lower_label=lower_label,
            higher_label=higher_label,
            lower_values=lower_values,
            higher_values=higher_values,
        )
        summary_rows.append(summary)
        tidy_rows.extend(
            [
                {
                    "group": group_name,
                    "fraction": lower_label,
                    "count": int(len(lower_values)),
                    "mean_union_recall_l3_at_top_k": float(np.mean(lower_values)) if len(lower_values) else float("nan"),
                },
                {
                    "group": group_name,
                    "fraction": higher_label,
                    "count": int(len(higher_values)),
                    "mean_union_recall_l3_at_top_k": float(np.mean(higher_values)) if len(higher_values) else float("nan"),
                },
            ]
        )

        scenario_name = group_name
        scenario_output_dir = family_output_dir / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = scenario_output_dir / f"{scenario_name}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)

        query_metric_arrays = {
            f"{group_name}_{lower_label}_union_recall_l3_at_{args.top_k}": lower_values,
            f"{group_name}_{higher_label}_union_recall_l3_at_{args.top_k}": higher_values,
            f"{group_name}_delta_union_recall_l3_at_{args.top_k}": delta_values,
        }
        save_ci_artifacts(
            scenario_output_dir=scenario_output_dir,
            family="complex_case_analysis",
            scenario_name=scenario_name,
            summary_path=summary_path,
            query_metric_arrays=query_metric_arrays,
            auc_inputs=None,
            extra_manifest={
                "top_k": int(args.top_k),
                "compare_fractions": [lower_fraction, higher_fraction],
            },
        )

    summary_json_path = family_output_dir / "complex_case_analysis_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary_rows), handle, ensure_ascii=False, indent=2)
    summary_csv_path = family_output_dir / "complex_case_analysis_summary.csv"
    pd.DataFrame(tidy_rows).to_csv(summary_csv_path, index=False)

    print(f"[Saved] JSON -> {summary_json_path}")
    print(f"[Saved] CSV  -> {summary_csv_path}")


if __name__ == "__main__":
    main()
