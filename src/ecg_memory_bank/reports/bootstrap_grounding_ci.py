from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_VIEW = "merged_best_sparse_first"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI for finalized MIMIC grounding evaluation packages.")
    parser.add_argument("--run-dir", required=True, help="Best-of run directory that contains evaluation/per_case.csv.")
    parser.add_argument("--view", default=DEFAULT_VIEW, help="Selected best-of view name for metadata only.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit output json path. Defaults to evaluation/ci_selected_view_summary.json under run-dir.",
    )
    return parser.parse_args()


def percentile_ci(point_estimate: float, scores: list[float]) -> tuple[float | None, float | None, int]:
    if not scores:
        return None, None, 0
    score_arr = np.asarray(scores, dtype=np.float64)
    return float(np.quantile(score_arr, 0.025)), float(np.quantile(score_arr, 0.975)), int(score_arr.size)


def bootstrap_index_matrix(n_samples: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_samples, size=(n_bootstrap, n_samples), dtype=np.int32)


def bootstrap_mean_ci(values: np.ndarray, index_matrix: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    point_estimate = float(np.mean(values)) if len(values) else None
    if not len(values):
        return point_estimate, None, None, 0
    sampled = values[index_matrix]
    sampled_means = np.mean(sampled, axis=1)
    lower, upper, valid = percentile_ci(float(point_estimate), sampled_means.tolist())
    return float(point_estimate), lower, upper, valid


def load_per_case(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Grounding per-case file not found: {path}")
    return pd.read_csv(path)


def build_metric_payload(rows: list[dict[str, Any]], mode: str, metric_col: str, index_matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray([float(row.get(metric_col, 0) or 0) for row in rows], dtype=np.float64)
    point, lower, upper, valid = bootstrap_mean_ci(values, index_matrix)
    return {
        "mode": mode,
        "metric_name": metric_col,
        "point_estimate": point,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "valid_bootstraps": valid,
        "sample_count": int(len(values)),
    }


def parse_label_text(text: Any) -> list[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split("|") if item.strip()]


def compute_micro_f1(rows: list[dict[str, Any]], pred_key: str) -> float:
    tp = fp = fn = 0
    for row in rows:
        gt = set(parse_label_text(row["ground_truth_labels"]))
        pred = set(parse_label_text(row[pred_key]))
        tp += len(gt & pred)
        fp += len(pred - gt)
        fn += len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap_micro_f1(rows: list[dict[str, Any]], mode: str, index_matrix: np.ndarray) -> dict[str, Any]:
    pred_key = "rag_predicted_labels" if mode == "rag" else "zero_predicted_labels"
    point_estimate = compute_micro_f1(rows, pred_key)
    sampled_scores: list[float] = []
    for indices in index_matrix:
        sampled_rows = [rows[int(idx)] for idx in indices]
        sampled_scores.append(compute_micro_f1(sampled_rows, pred_key))
    lower, upper, valid = percentile_ci(point_estimate, sampled_scores)
    return {
        "mode": mode,
        "metric_name": "micro_f1",
        "point_estimate": point_estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "valid_bootstraps": valid,
        "sample_count": int(len(rows)),
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    evaluation_dir = run_dir / "evaluation"
    per_case_path = evaluation_dir / "per_case.csv"
    summary_path = evaluation_dir / "summary.json"
    output_json = Path(args.output_json) if args.output_json else evaluation_dir / "ci_selected_view_summary.json"

    per_case_rows = load_rows(per_case_path)
    if not per_case_rows:
        raise ValueError(f"No rows found in {per_case_path}")

    index_matrix = bootstrap_index_matrix(len(per_case_rows), int(args.bootstrap), int(args.seed))
    metrics = [
        bootstrap_micro_f1(per_case_rows, "rag", index_matrix),
        bootstrap_micro_f1(per_case_rows, "zero", index_matrix),
        build_metric_payload(per_case_rows, "rag", "rag_exact_match", index_matrix),
        build_metric_payload(per_case_rows, "zero", "zero_exact_match", index_matrix),
    ]

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    selected_view = summary_payload.get("views", {}).get(args.view, {})
    selected_mode_summaries = selected_view.get("mode_summaries", {})

    output_payload = {
        "run_dir": str(run_dir),
        "selected_view": args.view,
        "bootstrap_iterations": int(args.bootstrap),
        "seed": int(args.seed),
        "per_case_source": str(per_case_path),
        "summary_source": str(summary_path),
        "exported_point_estimates": {
            "rag_micro_f1": selected_mode_summaries.get("rag", {}).get("micro_f1"),
            "zero_micro_f1": selected_mode_summaries.get("zero", {}).get("micro_f1"),
            "rag_exact_match_rate": selected_mode_summaries.get("rag", {}).get("exact_match_rate"),
            "zero_exact_match_rate": selected_mode_summaries.get("zero", {}).get("exact_match_rate"),
        },
        "metrics": metrics,
        "note": "Per-case bootstrap is computed from evaluation/per_case.csv for both micro-F1 and exact-match using the selected finalized best-of view outputs.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
