#!/usr/bin/env python3
"""Merge rerun low fractions with unchanged legacy anchor scenarios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.05)
    parser.add_argument("--summary-name", required=True)
    return parser.parse_args()


def load_rows(root: Path, summary_name: str) -> pd.DataFrame:
    csv_path = root / f"{summary_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    scenario_paths = sorted(root.glob("*/*_summary.json"))
    if scenario_paths:
        payload = [json.loads(path.read_text(encoding="utf-8")) for path in scenario_paths]
    else:
        json_path = root / f"{summary_name}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for item in payload:
        row = {
            "scenario": item["scenario"],
            "gallery_fraction": item["gallery_fraction"],
            "gallery_fraction_label": item["gallery_fraction_label"],
            "num_gallery": item["num_gallery"],
            "num_queries": item["num_queries"],
        }
        for key, value in item.items():
            if key.startswith("mean_union_recall") or key.startswith("mean_hierarchical_ndcg"):
                row[key] = value
        for level, metrics in item.get("voting_auc", {}).items():
            prefix = level.lower()
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    new_df = load_rows(args.new_root, args.summary_name)
    legacy_df = load_rows(args.legacy_root, args.summary_name)
    combined = pd.concat(
        [
            new_df[new_df["gallery_fraction"] < args.cutoff],
            legacy_df[legacy_df["gallery_fraction"] >= args.cutoff],
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("gallery_fraction").reset_index(drop=True)
    if combined["gallery_fraction"].duplicated().any():
        raise ValueError("duplicate gallery fractions in merged output")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in combined.itertuples(index=False):
        scenario = str(row.scenario)
        source_root = args.new_root if float(row.gallery_fraction) < args.cutoff else args.legacy_root
        source = (source_root / scenario).resolve()
        destination = args.output_root / scenario
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise FileExistsError(f"unexpected existing scenario link: {destination}")
        else:
            os.symlink(source, destination, target_is_directory=True)

    csv_path = args.output_root / f"{args.summary_name}.csv"
    json_path = args.output_root / f"{args.summary_name}.json"
    provenance_path = args.output_root / "merge_provenance.json"
    combined.to_csv(csv_path, index=False)
    json_path.write_text(combined.to_json(orient="records", indent=2), encoding="utf-8")
    provenance_path.write_text(
        json.dumps(
            {
                "new_root": str(args.new_root.resolve()),
                "legacy_root": str(args.legacy_root.resolve()),
                "cutoff": float(args.cutoff),
                "policy": "new fractions below cutoff; unchanged legacy anchors at or above cutoff",
                "fractions": combined["gallery_fraction"].astype(float).tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(combined[["scenario", "gallery_fraction", "num_gallery", "num_queries"]].to_string(index=False))
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
