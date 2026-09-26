#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_memory_bank.retrieval.artifact_utils import format_fraction_tag, save_ci_artifacts
from ecg_memory_bank.retrieval.metrics_utils import to_jsonable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the long-tail top10 L3 gain analysis from existing scale_benchmark outputs."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--fractions", nargs=3, type=float, default=[0.001, 0.1, 1.0])
    parser.add_argument("--tail-quantile", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_scale_artifacts(output_root: Path, fraction: float) -> tuple[str, pd.DataFrame, dict]:
    fraction_tag = format_fraction_tag(fraction)
    scenario_name = f"heedb_train_gallery_{fraction_tag}_finalgallery_test"
    scenario_dir = output_root / "scale_benchmark" / scenario_name
    per_class_path = scenario_dir / f"per_class_auc_{scenario_name}_L3.csv"
    summary_path = scenario_dir / f"{scenario_name}_summary.json"
    if not per_class_path.exists():
        raise FileNotFoundError(f"Missing L3 per-class AUC CSV: {per_class_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing scenario summary: {summary_path}")
    return scenario_name, pd.read_csv(per_class_path), json.loads(summary_path.read_text(encoding="utf-8"))


def select_top10_tail_classes(
    *,
    low_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    top_n: int,
    tail_quantile: float,
) -> pd.DataFrame:
    ref_nonzero = ref_df.loc[ref_df["Support"] > 0].copy()
    if ref_nonzero.empty:
        raise ValueError("Reference scenario has no non-zero L3 support.")
    support_threshold = float(ref_nonzero["Support"].quantile(tail_quantile))
    tail_pool = ref_nonzero.loc[ref_nonzero["Support"] <= support_threshold, ["Class", "Support"]].copy()
    merged = tail_pool.merge(low_df[["Class", "AUROC"]], on="Class", how="left").rename(columns={"AUROC": "low_auc"})
    merged = merged.sort_values(by=["low_auc", "Support", "Class"], ascending=[True, True, True]).reset_index(drop=True)
    return merged.head(top_n)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    family_output_dir = output_root / "longtail_gain_analysis"
    family_output_dir.mkdir(parents=True, exist_ok=True)

    low_fraction, mid_fraction, ref_fraction = [float(item) for item in args.fractions]
    low_name, low_df, _ = load_scale_artifacts(output_root, low_fraction)
    mid_name, mid_df, _ = load_scale_artifacts(output_root, mid_fraction)
    ref_name, ref_df, _ = load_scale_artifacts(output_root, ref_fraction)

    selected = select_top10_tail_classes(
        low_df=low_df,
        ref_df=ref_df,
        top_n=int(args.top_n),
        tail_quantile=float(args.tail_quantile),
    )

    low_col = f"{format_fraction_tag(low_fraction)}_auc"
    mid_col = f"{format_fraction_tag(mid_fraction)}_auc"
    ref_col = f"{format_fraction_tag(ref_fraction)}_auc"
    delta_col = f"{format_fraction_tag(mid_fraction)}_to_{format_fraction_tag(ref_fraction)}_delta"

    comparison_df = (
        selected[["Class", "Support"]]
        .merge(low_df[["Class", "AUROC"]], on="Class", how="left")
        .rename(columns={"AUROC": low_col})
        .merge(mid_df[["Class", "AUROC"]], on="Class", how="left")
        .rename(columns={"AUROC": mid_col})
        .merge(ref_df[["Class", "AUROC"]], on="Class", how="left")
        .rename(columns={"AUROC": ref_col})
    )
    comparison_df[delta_col] = comparison_df[ref_col] - comparison_df[mid_col]

    top10_csv_path = family_output_dir / "longtail_top10_comparison.csv"
    comparison_df.to_csv(top10_csv_path, index=False)
    summary_payload = {
        "low_fraction": low_fraction,
        "mid_fraction": mid_fraction,
        "reference_fraction": ref_fraction,
        "tail_quantile": float(args.tail_quantile),
        "top_n": int(args.top_n),
        "selected_classes": comparison_df.to_dict(orient="records"),
        "paths": {
            "top10_csv": str(top10_csv_path),
        },
    }
    summary_json_path = family_output_dir / "longtail_gain_analysis_summary.json"
    summary_json_path.write_text(json.dumps(to_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    for fraction, scenario_df in [(low_fraction, low_df), (mid_fraction, mid_df), (ref_fraction, ref_df)]:
        scenario_name = f"top10_tail_{format_fraction_tag(fraction)}"
        scenario_output_dir = family_output_dir / scenario_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        scenario_table = comparison_df[["Class", "Support"]].merge(
            scenario_df[["Class", "AUROC"]], on="Class", how="left"
        )
        scenario_table_path = scenario_output_dir / f"{scenario_name}_per_class_auc.csv"
        scenario_table.to_csv(scenario_table_path, index=False)

        summary = {
            "scenario": scenario_name,
            "gallery_fraction": fraction,
            "num_classes": int(len(scenario_table)),
            "macro_auc": float(scenario_table["AUROC"].dropna().mean()) if "AUROC" in scenario_table else float("nan"),
            "paths": {
                "per_class_csv": str(scenario_table_path),
            },
        }
        scenario_summary_path = scenario_output_dir / f"{scenario_name}_summary.json"
        scenario_summary_path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

        auc_input = output_root / "scale_benchmark" / (
            f"heedb_train_gallery_{format_fraction_tag(fraction)}_finalgallery_test"
        ) / "ci_artifacts" / (
            f"heedb_train_gallery_{format_fraction_tag(fraction)}_finalgallery_test_l3_auc_inputs.npz"
        )
        if not auc_input.exists():
            raise FileNotFoundError(f"Missing upstream L3 auc_input npz: {auc_input}")
        payload = np.load(auc_input, allow_pickle=False)
        class_names = [str(name) for name in payload["class_names"].tolist()]
        selected_indices = [class_names.index(class_name) for class_name in comparison_df["Class"].tolist()]
        auc_inputs = {
            "l3_top10": {
                "y_true": payload["y_true"][:, selected_indices],
                "y_score": payload["y_score"][:, selected_indices],
                "class_names": comparison_df["Class"].tolist(),
            }
        }
        save_ci_artifacts(
            scenario_output_dir=scenario_output_dir,
            family="longtail_gain_analysis",
            scenario_name=scenario_name,
            summary_path=scenario_summary_path,
            query_metric_arrays=None,
            auc_inputs=auc_inputs,
            extra_manifest={
                "gallery_fraction": fraction,
                "top_n": int(args.top_n),
            },
        )

    print(f"[Saved] JSON -> {summary_json_path}")
    print(f"[Saved] CSV  -> {top10_csv_path}")


if __name__ == "__main__":
    main()
