"""Export paired percentile bootstrap intervals for the blinded reader study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "eval_analysis_all4"
BOOTSTRAPS = 1000
SEED = 42


def percentile_interval(values: np.ndarray, rng: np.random.RandomState) -> dict[str, float | int]:
    indices = rng.randint(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "pairs": int(len(values)),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rng = np.random.RandomState(SEED)
    case_rows = read_rows(ROOT / "case_level_paired_diffs.csv")
    label_rows = read_rows(ROOT / "label_level_paired_diffs.csv")

    case_metrics = {}
    for metric in sorted({row["metric"] for row in case_rows}):
        values = np.asarray([float(row["diff"]) for row in case_rows if row["metric"] == metric])
        case_metrics[metric] = percentile_interval(values, rng)

    label_values = np.asarray([float(row["diff"]) for row in label_rows])
    payload = {
        "bootstrap_iterations": BOOTSTRAPS,
        "seed": SEED,
        "ci_method": "paired_record_percentile_bootstrap",
        "case_level": case_metrics,
        "label_level_overall": percentile_interval(label_values, rng),
    }
    output = ROOT / "paired_bootstrap_ci.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
