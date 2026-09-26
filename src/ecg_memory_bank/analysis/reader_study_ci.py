"""Case-clustered reader inference.

Purpose
-------
Estimate uncertainty and direction of blinded paired reader-study effects.

Method
------
Bootstrap ECG case IDs while keeping all ratings/labels for each sampled case;
perform two-sided sign tests on one mean difference per case.

Inputs
------
Case-level and label-level paired-difference CSVs produced by reader_study.py.

Outputs
-------
External JSON/CSV with pooled paired means, 95% intervals, case
wins/losses/ties, and sign-test p values.

Paper parameters
----------------
Exactly 1000 case-clustered percentile resamples with seed=42 for the formal
paper analysis.

Shapes and meaning
------------------
Rows are rater-case or label records; the bootstrap unit is the ECG case, so
four readers from one case stay together.
"""

from __future__ import annotations

import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


BOOTSTRAPS = 1000
SEED = 42


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def cluster_summary(rows: list[dict[str, str]], rng: np.random.RandomState) -> dict[str, float | int | str]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[str(row["sample_id"])].append(float(row["diff"]))
    case_ids = sorted(by_case)
    sums = np.asarray([sum(by_case[case_id]) for case_id in case_ids], dtype=float)
    counts = np.asarray([len(by_case[case_id]) for case_id in case_ids], dtype=float)
    case_means = sums / counts

    sampled = rng.randint(0, len(case_ids), size=(BOOTSTRAPS, len(case_ids)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    wins = int(np.sum(case_means > 0))
    losses = int(np.sum(case_means < 0))
    ties = int(np.sum(case_means == 0))
    sign_p = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue) if wins + losses else 1.0
    return {
        "point_estimate": float(sums.sum() / counts.sum()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "cases": int(len(case_ids)),
        "pairs": int(counts.sum()),
        "case_wins": wins,
        "case_losses": losses,
        "case_ties": ties,
        "case_level_sign_p_two_sided": sign_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Case-clustered reader-study confidence intervals")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.RandomState(SEED)
    case_rows = read_rows(args.input_dir / "case_level_paired_diffs.csv")
    label_rows = read_rows(args.input_dir / "label_level_paired_diffs.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_metrics = {}
    for metric in sorted({row["metric"] for row in case_rows}):
        case_metrics[metric] = cluster_summary([row for row in case_rows if row["metric"] == metric], rng)

    label_by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        category = (row.get("matched_category") or "").strip() or "UNMAPPED"
        label_by_category[category].append(row)

    payload = {
        "bootstrap_iterations": BOOTSTRAPS,
        "seed": SEED,
        "ci_method": "case_cluster_percentile_bootstrap_retaining_all_reader_records",
        "sign_test_unit": "case_mean_difference",
        "case_level": case_metrics,
        "label_level_overall": cluster_summary(label_rows, rng),
        "label_level_by_category": {
            category: cluster_summary(rows, rng)
            for category, rows in sorted(label_by_category.items())
        },
    }
    output_json = args.output_dir / "case_clustered_reader_ci.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    flat_rows: list[dict[str, str | float | int]] = []
    for name, summary in payload["case_level"].items():
        flat_rows.append({"level": "case_dimension", "name": name, **summary})
    flat_rows.append({"level": "label_overall", "name": "ALL", **payload["label_level_overall"]})
    for name, summary in payload["label_level_by_category"].items():
        flat_rows.append({"level": "label_category", "name": name, **summary})
    output_csv = args.output_dir / "case_clustered_reader_ci.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
