from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def split_pipe_labels(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_generation_model(run_dir: Path) -> str:
    run_summary = run_dir / "logs" / "run_summary.json"
    if run_summary.exists():
        payload = load_json(run_summary)
        prompt_snapshot = payload.get("prompt_config_snapshot", {})
        if isinstance(prompt_snapshot, dict):
            model = prompt_snapshot.get("model")
            if model:
                return str(model)
        api_snapshot = payload.get("direct_api_config_snapshot", {})
        if isinstance(api_snapshot, dict):
            model = api_snapshot.get("model")
            if model:
                return str(model)
    return "unknown"


def load_label_to_category(run_dir: Path) -> dict[str, str]:
    run_summary = run_dir / "logs" / "run_summary.json"
    if run_summary.exists():
        payload = load_json(run_summary)
        raw = payload.get("candidate_labels_by_category", {})
        if isinstance(raw, dict):
            mapping: dict[str, str] = {}
            for category, labels in raw.items():
                if not isinstance(labels, list):
                    continue
                for label in labels:
                    label_str = str(label).strip()
                    if label_str:
                        mapping[label_str] = str(category)
            if mapping:
                return mapping
    return {}


def load_rag_retrieval_labels(run_dir: Path) -> dict[str, set[str]]:
    request_file = run_dir / "requests" / "rag_requests.jsonl"
    if not request_file.exists():
        return {}

    query_to_labels: dict[str, set[str]] = {}
    with request_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            message_content = row["body"]["messages"][1]["content"]
            payload = json.loads(message_content)
            query_id = str(payload["query_case"]["id"])
            labels: set[str] = set()
            for retrieved_case in payload.get("retrieved_cases", []):
                for label in retrieved_case.get("reference_labels", []) or []:
                    label_str = str(label).strip()
                    if label_str:
                        labels.add(label_str)
            query_to_labels[query_id] = labels
    return query_to_labels


@dataclass
class ModeStats:
    run_name: str
    model: str
    mode: str
    case_count: int
    avg_ground_truth_labels: float
    avg_predicted_labels: float
    avg_true_positive_labels: float
    avg_false_positive_labels: float
    avg_missing_labels: float
    avg_retrieved_reference_labels: float
    avg_predicted_overlap_with_retrieval: float
    avg_retrieval_induced_noise_labels: float
    avg_nonretrieval_noise_labels: float
    any_false_positive_case_rate: float
    any_retrieval_noise_case_rate: float
    false_positive_from_retrieval_ratio: float
    total_false_positive_labels: int
    total_retrieval_induced_noise_labels: int
    total_nonretrieval_noise_labels: int
    top5_false_positive_share: float
    top5_retrieval_induced_noise_share: float
    top5_nonretrieval_noise_share: float
    top_false_positive_labels: list[dict[str, Any]]
    top_retrieval_induced_noise_labels: list[dict[str, Any]]
    top_nonretrieval_noise_labels: list[dict[str, Any]]
    false_positive_category_breakdown: list[dict[str, Any]]
    retrieval_noise_category_breakdown: list[dict[str, Any]]
    nonretrieval_noise_category_breakdown: list[dict[str, Any]]


def analyze_mode(
    *,
    run_name: str,
    model: str,
    mode: str,
    rows: list[dict[str, str]],
    retrieval_labels_by_query: dict[str, set[str]],
    label_to_category: dict[str, str],
) -> ModeStats:
    gt_total = 0
    pred_total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    retrieved_total = 0
    predicted_retrieval_overlap_total = 0
    retrieval_noise_total = 0
    nonretrieval_noise_total = 0
    any_fp_cases = 0
    any_retrieval_noise_cases = 0
    fp_label_counter: Counter[str] = Counter()
    retrieval_noise_label_counter: Counter[str] = Counter()
    nonretrieval_noise_label_counter: Counter[str] = Counter()
    fp_category_counter: Counter[str] = Counter()
    retrieval_noise_category_counter: Counter[str] = Counter()
    nonretrieval_noise_category_counter: Counter[str] = Counter()

    for row in rows:
        query_id = row["query_case_id"]
        gt = set(split_pipe_labels(row["ground_truth_labels"]))
        pred = set(split_pipe_labels(row[f"{mode}_predicted_labels"]))
        retrieved = retrieval_labels_by_query.get(query_id, set()) if mode == "rag" else set()

        tp = pred & gt
        fp = pred - gt
        fn = gt - pred
        predicted_overlap = pred & retrieved
        retrieval_noise = fp & retrieved
        nonretrieval_noise = fp - retrieved

        gt_total += len(gt)
        pred_total += len(pred)
        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)
        retrieved_total += len(retrieved)
        predicted_retrieval_overlap_total += len(predicted_overlap)
        retrieval_noise_total += len(retrieval_noise)
        nonretrieval_noise_total += len(nonretrieval_noise)

        fp_label_counter.update(fp)
        retrieval_noise_label_counter.update(retrieval_noise)
        nonretrieval_noise_label_counter.update(nonretrieval_noise)
        fp_category_counter.update(label_to_category.get(label, "Unmapped") for label in fp)
        retrieval_noise_category_counter.update(
            label_to_category.get(label, "Unmapped") for label in retrieval_noise
        )
        nonretrieval_noise_category_counter.update(
            label_to_category.get(label, "Unmapped") for label in nonretrieval_noise
        )

        if fp:
            any_fp_cases += 1
        if retrieval_noise:
            any_retrieval_noise_cases += 1

    case_count = len(rows)
    fp_ratio = (retrieval_noise_total / fp_total) if fp_total else 0.0

    def top_counter_rows(counter: Counter[str], limit: int = 15) -> list[dict[str, Any]]:
        total = sum(counter.values())
        return [
            {
                "label": label,
                "count": count,
                "proportion": (count / total) if total else 0.0,
            }
            for label, count in counter.most_common(limit)
        ]

    def category_rows(counter: Counter[str]) -> list[dict[str, Any]]:
        total = sum(counter.values())
        rows: list[dict[str, Any]] = []
        for label, count in counter.most_common():
            proportion = (count / total) if total else 0.0
            rows.append({"category": label, "count": count, "proportion": proportion})
        return rows

    def top5_share(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if not total:
            return 0.0
        return sum(count for _, count in counter.most_common(5)) / total

    return ModeStats(
        run_name=run_name,
        model=model,
        mode=mode,
        case_count=case_count,
        avg_ground_truth_labels=gt_total / case_count if case_count else 0.0,
        avg_predicted_labels=pred_total / case_count if case_count else 0.0,
        avg_true_positive_labels=tp_total / case_count if case_count else 0.0,
        avg_false_positive_labels=fp_total / case_count if case_count else 0.0,
        avg_missing_labels=fn_total / case_count if case_count else 0.0,
        avg_retrieved_reference_labels=retrieved_total / case_count if case_count else 0.0,
        avg_predicted_overlap_with_retrieval=predicted_retrieval_overlap_total / case_count if case_count else 0.0,
        avg_retrieval_induced_noise_labels=retrieval_noise_total / case_count if case_count else 0.0,
        avg_nonretrieval_noise_labels=nonretrieval_noise_total / case_count if case_count else 0.0,
        any_false_positive_case_rate=any_fp_cases / case_count if case_count else 0.0,
        any_retrieval_noise_case_rate=any_retrieval_noise_cases / case_count if case_count else 0.0,
        false_positive_from_retrieval_ratio=fp_ratio,
        total_false_positive_labels=fp_total,
        total_retrieval_induced_noise_labels=retrieval_noise_total,
        total_nonretrieval_noise_labels=nonretrieval_noise_total,
        top5_false_positive_share=top5_share(fp_label_counter),
        top5_retrieval_induced_noise_share=top5_share(retrieval_noise_label_counter),
        top5_nonretrieval_noise_share=top5_share(nonretrieval_noise_label_counter),
        top_false_positive_labels=top_counter_rows(fp_label_counter),
        top_retrieval_induced_noise_labels=top_counter_rows(retrieval_noise_label_counter),
        top_nonretrieval_noise_labels=top_counter_rows(nonretrieval_noise_label_counter),
        false_positive_category_breakdown=category_rows(fp_category_counter),
        retrieval_noise_category_breakdown=category_rows(retrieval_noise_category_counter),
        nonretrieval_noise_category_breakdown=category_rows(nonretrieval_noise_category_counter),
    )


def load_per_case_rows(run_dir: Path) -> list[dict[str, str]]:
    per_case_csv = run_dir / "metrics" / "per_case.csv"
    with per_case_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def default_run_name(run_dir: Path) -> str:
    return run_dir.name


def compare_modes(stats: list[ModeStats]) -> dict[str, Any]:
    by_mode = {item.mode: item for item in stats}
    rag = by_mode.get("rag")
    zero = by_mode.get("zero")
    if rag is None or zero is None:
        return {}
    return {
        "avg_predicted_labels_delta_rag_minus_zero": rag.avg_predicted_labels - zero.avg_predicted_labels,
        "avg_false_positive_labels_delta_rag_minus_zero": rag.avg_false_positive_labels - zero.avg_false_positive_labels,
        "avg_missing_labels_delta_rag_minus_zero": rag.avg_missing_labels - zero.avg_missing_labels,
        "avg_true_positive_labels_delta_rag_minus_zero": rag.avg_true_positive_labels - zero.avg_true_positive_labels,
        "rag_false_positive_from_retrieval_ratio": rag.false_positive_from_retrieval_ratio,
        "rag_any_retrieval_noise_case_rate": rag.any_retrieval_noise_case_rate,
    }


def analyze_run(run_dir: Path, run_name: str | None = None) -> dict[str, Any]:
    rows = load_per_case_rows(run_dir)
    retrieval_labels_by_query = load_rag_retrieval_labels(run_dir)
    label_to_category = load_label_to_category(run_dir)
    name = run_name or default_run_name(run_dir)
    model = infer_generation_model(run_dir)

    stats = [
        analyze_mode(
            run_name=name,
            model=model,
            mode="rag",
            rows=rows,
            retrieval_labels_by_query=retrieval_labels_by_query,
            label_to_category=label_to_category,
        ),
        analyze_mode(
            run_name=name,
            model=model,
            mode="zero",
            rows=rows,
            retrieval_labels_by_query=retrieval_labels_by_query,
            label_to_category=label_to_category,
        ),
    ]

    return {
        "run_name": name,
        "run_dir": str(run_dir),
        "generation_model": model,
        "mode_stats": [asdict(item) for item in stats],
        "mode_comparison": compare_modes(stats),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result: dict[str, Any]) -> None:
    print(f"Run: {result['run_name']}")
    print(f"Dir: {result['run_dir']}")
    print(f"Generation model: {result['generation_model']}")
    print()
    for stats in result["mode_stats"]:
        print(
            f"[{stats['mode']}] "
            f"cases={stats['case_count']} "
            f"avg_gt={stats['avg_ground_truth_labels']:.3f} "
            f"avg_pred={stats['avg_predicted_labels']:.3f} "
            f"avg_tp={stats['avg_true_positive_labels']:.3f} "
            f"avg_fp={stats['avg_false_positive_labels']:.3f} "
            f"avg_fn={stats['avg_missing_labels']:.3f}"
        )
        print(
            f"    retrieved_avg={stats['avg_retrieved_reference_labels']:.3f} "
            f"pred∩retrieved_avg={stats['avg_predicted_overlap_with_retrieval']:.3f} "
            f"retrieval_noise_avg={stats['avg_retrieval_induced_noise_labels']:.3f} "
            f"nonretrieval_noise_avg={stats['avg_nonretrieval_noise_labels']:.3f}"
        )
        print(
            f"    fp_case_rate={stats['any_false_positive_case_rate']:.3f} "
            f"retrieval_noise_case_rate={stats['any_retrieval_noise_case_rate']:.3f} "
            f"fp_from_retrieval_ratio={stats['false_positive_from_retrieval_ratio']:.3f}"
        )
        top_fp = stats.get("top_false_positive_labels", [])[:5]
        if top_fp:
            print(
                "    top_fp_labels="
                + ", ".join(
                    f"{item['label']}({item['count']}, {item['proportion']:.3f})"
                    for item in top_fp
                )
            )
            print(f"    top5_fp_share={stats['top5_false_positive_share']:.3f}")
        top_retrieval_fp = stats.get("top_retrieval_induced_noise_labels", [])[:5]
        if top_retrieval_fp:
            print(
                "    top_retrieval_fp_labels="
                + ", ".join(
                    f"{item['label']}({item['count']}, {item['proportion']:.3f})"
                    for item in top_retrieval_fp
                )
            )
            print(
                f"    top5_retrieval_fp_share="
                f"{stats['top5_retrieval_induced_noise_share']:.3f}"
            )
        top_fp_categories = stats.get("false_positive_category_breakdown", [])[:5]
        if top_fp_categories:
            print(
                "    fp_categories="
                + ", ".join(
                    f"{item['category']}({item['count']}, {item['proportion']:.3f})"
                    for item in top_fp_categories
                )
            )
        top_retrieval_categories = stats.get("retrieval_noise_category_breakdown", [])[:5]
        if top_retrieval_categories:
            print(
                "    retrieval_fp_categories="
                + ", ".join(
                    f"{item['category']}({item['count']}, {item['proportion']:.3f})"
                    for item in top_retrieval_categories
                )
            )
    if result["mode_comparison"]:
        comp = result["mode_comparison"]
        print()
        print("RAG vs Zero deltas:")
        print(
            f"  avg_predicted_labels_delta={comp['avg_predicted_labels_delta_rag_minus_zero']:.3f} "
            f"avg_false_positive_labels_delta={comp['avg_false_positive_labels_delta_rag_minus_zero']:.3f} "
            f"avg_missing_labels_delta={comp['avg_missing_labels_delta_rag_minus_zero']:.3f} "
            f"avg_true_positive_labels_delta={comp['avg_true_positive_labels_delta_rag_minus_zero']:.3f}"
        )
        print(
            f"  rag_fp_from_retrieval_ratio={comp['rag_false_positive_from_retrieval_ratio']:.3f} "
            f"rag_retrieval_noise_case_rate={comp['rag_any_retrieval_noise_case_rate']:.3f}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-case label counts and retrieval-related noise for LLM grounding runs."
        )
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Run directory containing metrics/per_case.csv and requests/*.jsonl. Can be provided multiple times.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        help="Optional display name for the corresponding --run-dir. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to save the aggregated analysis JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Optional path to save flattened mode-level stats as CSV.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    run_names = args.run_names or []
    if run_names and len(run_names) != len(args.run_dirs):
        parser.error("--run-name count must match --run-dir count when provided.")

    results: list[dict[str, Any]] = []
    flattened_rows: list[dict[str, Any]] = []

    for idx, run_dir_str in enumerate(args.run_dirs):
        run_dir = Path(run_dir_str)
        run_name = run_names[idx] if idx < len(run_names) else None
        result = analyze_run(run_dir, run_name)
        results.append(result)
        for row in result["mode_stats"]:
            flattened_rows.append(row)
        print_summary(result)
        if idx != len(args.run_dirs) - 1:
            print()
            print("-" * 80)
            print()

    payload = {"runs": results}
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_csv, flattened_rows)


if __name__ == "__main__":
    main()
