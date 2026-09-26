#!/usr/bin/env python3
"""HEEDB log-linear curve fitting.

Purpose
-------
Fit the final paper's AUROC response to logarithmic active-memory size.

Method
------
Load per-class AUROCs at eight nested fractions, restrict to common labels,
average overall and by fixed support tier, then fit ordinary least squares.

Inputs
------
Scale-benchmark summary and per-class tables plus a full-memory support-tier
manifest.

Outputs
-------
External curve points, slope/CI/R-squared/p-value tables, and common-label
records.

Paper parameters
----------------
Uses eight memory states; slope confidence intervals come from the eight-point
OLS fit, not the query bootstrap.

Shapes and meaning
------------------
Predictor is ln(number of active cases); response is fixed-label macro AUROC.
Tier curves average their own common labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from ecg_memory_bank.protocol import external_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--label-mapping", type=Path)
    parser.add_argument("--tier-manifest", type=Path)
    parser.add_argument("--per-class-summary", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_counts(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    frame["mapped_label"] = frame["mapped_label"].astype(str).str.strip().str.lower()
    counts = frame.groupby("mapped_label", sort=False)["count"].sum().astype(int)
    if len(counts) != 120:
        raise ValueError(f"expected 120 canonical L3 labels, found {len(counts)}")
    return counts


def assign_quartiles(counts: pd.Series) -> dict[str, str]:
    ordered = sorted(counts.index, key=lambda label: (int(counts[label]), label))
    tiers = ("Tail", "Low-mid", "Mid", "Head")
    return {label: tiers[index // 30] for index, label in enumerate(ordered)}


def fit_curve(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    fit = stats.linregress(x, y)
    critical = stats.t.ppf(0.975, df=len(x) - 2)
    return {
        "slope": float(fit.slope),
        "ci95_lower": float(fit.slope - critical * fit.stderr),
        "ci95_upper": float(fit.slope + critical * fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "p_value": float(fit.pvalue),
        "intercept": float(fit.intercept),
    }


def main() -> None:
    args = parse_args()
    if args.tier_manifest:
        tier_frame = pd.read_csv(args.tier_manifest)
        tier_frame["mapped_label"] = tier_frame["mapped_label"].astype(str).str.strip().str.lower()
        counts = tier_frame.set_index("mapped_label")["count"].astype(int)
        tier_map = tier_frame.set_index("mapped_label")["support_tier"].astype(str).to_dict()
    elif args.label_mapping:
        counts = canonical_counts(args.label_mapping)
        tier_map = assign_quartiles(counts)
    else:
        raise ValueError("provide --tier-manifest or --label-mapping")

    if args.summary_csv:
        summary = pd.read_csv(args.summary_csv).sort_values("gallery_fraction")
    elif args.input_root:
        summary = pd.read_csv(args.input_root / "scale_benchmark_summary.csv").sort_values("gallery_fraction")
    else:
        raise ValueError("provide --summary-csv or --input-root")

    tables: list[tuple[float, int, pd.DataFrame]] = []
    label_sets = []
    if args.per_class_summary:
        all_classes = pd.read_csv(args.per_class_summary)
        all_classes["label"] = all_classes["label"].astype(str).str.strip().str.lower()
        for row in summary.itertuples(index=False):
            frame = all_classes[np.isclose(all_classes["gallery_fraction"], float(row.gallery_fraction))][
                ["label", "auroc"]
            ].rename(columns={"label": "Class", "auroc": "AUROC"})
            frame = frame[np.isfinite(frame["AUROC"])].copy()
            tables.append((float(row.gallery_fraction), int(row.num_gallery), frame))
            label_sets.append(set(frame["Class"]))
    else:
        if not args.input_root:
            raise ValueError("--input-root is required without --per-class-summary")
        for row in summary.itertuples(index=False):
            scenario = str(row.scenario)
            path = args.input_root / scenario / f"per_class_auc_{scenario}_L3.csv"
            frame = pd.read_csv(path)
            frame["Class"] = frame["Class"].astype(str).str.strip().str.lower()
            frame = frame[np.isfinite(frame["AUROC"])].copy()
            tables.append((float(row.gallery_fraction), int(row.num_gallery), frame))
            label_sets.append(set(frame["Class"]))

    common = sorted(set.intersection(*label_sets))
    if not common:
        raise ValueError("no common evaluable L3 labels")
    missing_from_mapping = sorted(set(common) - set(tier_map))
    if missing_from_mapping:
        raise ValueError(f"common labels missing from mapping: {missing_from_mapping}")

    point_rows = []
    for fraction, memory_size, frame in tables:
        indexed = frame.set_index("Class")["AUROC"]
        row = {
            "gallery_fraction": fraction,
            "memory_size": memory_size,
            "Overall": float(indexed.loc[common].mean()),
        }
        for tier in ("Tail", "Low-mid", "Mid", "Head"):
            labels = [label for label in common if tier_map[label] == tier]
            row[tier] = float(indexed.loc[labels].mean())
        point_rows.append(row)
    points = pd.DataFrame(point_rows).sort_values("gallery_fraction")

    x = np.log(points["memory_size"].to_numpy(dtype=np.float64))
    fit_rows = []
    for curve in ("Overall", "Tail", "Low-mid", "Mid", "Head"):
        labels = common if curve == "Overall" else [label for label in common if tier_map[label] == curve]
        fit_rows.append({"curve": curve, "labels": len(labels), **fit_curve(x, points[curve].to_numpy())})
    fits = pd.DataFrame(fit_rows)

    args.output_dir = external_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / "heedb_nested_loglinear_points.csv"
    fits_path = args.output_dir / "heedb_nested_loglinear_fits.csv"
    labels_path = args.output_dir / "heedb_nested_common_labels.csv"
    summary_path = args.output_dir / "heedb_nested_loglinear_summary.json"
    points.to_csv(points_path, index=False)
    fits.to_csv(fits_path, index=False)
    pd.DataFrame(
        {
            "label": common,
            "full_gallery_count": [int(counts[label]) for label in common],
            "support_tier": [tier_map[label] for label in common],
        }
    ).to_csv(labels_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "common_label_count": len(common),
                "tier_counts": {
                    tier: sum(tier_map[label] == tier for label in common)
                    for tier in ("Tail", "Low-mid", "Mid", "Head")
                },
                "points_csv": str(points_path),
                "fits_csv": str(fits_path),
                "labels_csv": str(labels_path),
                "fits": fit_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(points.to_string(index=False))
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
