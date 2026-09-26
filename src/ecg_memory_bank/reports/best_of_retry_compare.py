from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import ensure_parent, write_json
from ecg_memory_bank.reports.retry_failed_cases import (
    MAIN_METRIC_KEYS,
    _build_comparison_payload,
    _evaluate_records,
    _is_label_score_complete,
    _load_ground_truth,
    _load_parsed_records,
    _load_request_bundle,
    _summarize_mode,
)


ALL_MODES = ("rag", "zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare best-of retry merge priorities without re-running requests.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--retry-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def _default_output_dir(retry_dir: Path) -> Path:
    return retry_dir.parent / f"{retry_dir.name}_bestof"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_retry_parsed_records(retry_dir: Path, mode: str, strategy: str) -> dict[str, dict[str, Any]]:
    parsed_dir = retry_dir / "responses" / "parsed" / mode / strategy
    records: dict[str, dict[str, Any]] = {}
    if not parsed_dir.exists():
        return records
    for path in sorted(parsed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("query_case_id", path.stem))
        records[case_id] = payload
    return records


def _base_complete(record: dict[str, Any], candidate_label_set: list[str]) -> bool:
    return bool(record.get("strict_json_ok", False)) and _is_label_score_complete(record.get("label_scores", {}), candidate_label_set)


def _retry_complete(record: dict[str, Any], candidate_label_set: list[str]) -> bool:
    if not record:
        return False
    if not bool(record.get("strict_json_ok", False)):
        return False
    if bool(record.get("raw_empty", False)):
        return False
    if "label_score_complete_after_retry" in record:
        return bool(record.get("label_score_complete_after_retry", False))
    return _is_label_score_complete(record.get("label_scores", {}), candidate_label_set)


def _select_record(
    *,
    base_record: dict[str, Any],
    original_record: dict[str, Any],
    sparse_record: dict[str, Any],
    candidate_label_set: list[str],
    preference: str,
) -> tuple[dict[str, Any], str]:
    base_ok = _base_complete(base_record, candidate_label_set)
    original_ok = _retry_complete(original_record, candidate_label_set)
    sparse_ok = _retry_complete(sparse_record, candidate_label_set)

    if preference == "original_first":
        if original_ok:
            return original_record, "original"
        if sparse_ok:
            return sparse_record, "sparse"
        return base_record, "base"

    if preference == "sparse_first":
        if sparse_ok:
            return sparse_record, "sparse"
        if original_ok:
            return original_record, "original"
        return base_record, "base"

    raise ValueError(f"Unsupported preference: {preference}")


def _build_views_and_selection_rows(
    *,
    base_records: dict[str, dict[str, dict[str, Any]]],
    retry_records: dict[str, dict[str, dict[str, Any]]],
    candidate_sets: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, int]]]]:
    views: dict[str, dict[str, dict[str, Any]]] = {
        "base": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        },
        "merged_original_retry": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        },
        "merged_sparse_retry": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        },
        "merged_best_original_first": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        },
        "merged_best_sparse_first": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        },
    }

    for mode in ALL_MODES:
        for case_id, record in retry_records["original"][mode].items():
            candidate_label_set = candidate_sets[mode].get(case_id, [])
            if _retry_complete(record, candidate_label_set):
                views["merged_original_retry"][mode][case_id] = record
        for case_id, record in retry_records["sparse"][mode].items():
            candidate_label_set = candidate_sets[mode].get(case_id, [])
            if _retry_complete(record, candidate_label_set):
                views["merged_sparse_retry"][mode][case_id] = record

    selection_rows: dict[str, list[dict[str, Any]]] = {
        "original_first": [],
        "sparse_first": [],
    }
    selection_counts: dict[str, dict[str, dict[str, int]]] = {
        "original_first": {mode: {"base": 0, "original": 0, "sparse": 0} for mode in ALL_MODES},
        "sparse_first": {mode: {"base": 0, "original": 0, "sparse": 0} for mode in ALL_MODES},
    }

    all_case_ids = sorted(set(base_records["rag"]) | set(base_records["zero"]))
    for preference, view_name in (
        ("original_first", "merged_best_original_first"),
        ("sparse_first", "merged_best_sparse_first"),
    ):
        for mode in ALL_MODES:
            for case_id in all_case_ids:
                base_record = base_records[mode].get(case_id, {})
                original_record = retry_records["original"][mode].get(case_id, {})
                sparse_record = retry_records["sparse"][mode].get(case_id, {})
                candidate_label_set = candidate_sets[mode].get(case_id, [])
                selected_record, selected_from = _select_record(
                    base_record=base_record,
                    original_record=original_record,
                    sparse_record=sparse_record,
                    candidate_label_set=candidate_label_set,
                    preference=preference,
                )
                views[view_name][mode][case_id] = selected_record
                selection_counts[preference][mode][selected_from] += 1
                selection_rows[preference].append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "selected_from": selected_from,
                        "base_complete": int(_base_complete(base_record, candidate_label_set)),
                        "original_complete": int(_retry_complete(original_record, candidate_label_set)),
                        "sparse_complete": int(_retry_complete(sparse_record, candidate_label_set)),
                        "base_strict_json_ok": int(bool(base_record.get("strict_json_ok", False))),
                        "original_strict_json_ok": int(bool(original_record.get("strict_json_ok", False))),
                        "sparse_strict_json_ok": int(bool(sparse_record.get("strict_json_ok", False))),
                        "base_label_score_complete": int(_is_label_score_complete(base_record.get("label_scores", {}), candidate_label_set)),
                        "original_label_score_complete": int(
                            bool(original_record.get("label_score_complete_after_retry", False))
                            if original_record
                            else False
                        ),
                        "sparse_label_score_complete": int(
                            bool(sparse_record.get("label_score_complete_after_retry", False))
                            if sparse_record
                            else False
                        ),
                    }
                )
    return views, selection_rows, selection_counts


def _build_summary_payload(
    *,
    summary_by_view: dict[str, Any],
    selection_counts: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    def _overlap_case_count(view_payload: dict[str, Any]) -> int:
        auroc_payload = dict(view_payload.get("auroc", {}) or {})
        if "overlap_case_count" in auroc_payload:
            return int(auroc_payload.get("overlap_case_count", 0) or 0)
        rag_payload = dict(auroc_payload.get("rag", {}) or {})
        zero_payload = dict(auroc_payload.get("zero", {}) or {})
        if "overlap_case_count" in rag_payload and "overlap_case_count" in zero_payload:
            return min(
                int(rag_payload.get("overlap_case_count", 0) or 0),
                int(zero_payload.get("overlap_case_count", 0) or 0),
            )
        return int(rag_payload.get("overlap_case_count", 0) or zero_payload.get("overlap_case_count", 0) or 0)

    comparison = _build_comparison_payload(summary_by_view)
    payload: dict[str, Any] = {
        "views": summary_by_view,
        "comparison": comparison,
    }
    base = summary_by_view["base"]
    base_overlap = _overlap_case_count(base)
    for preference, view_name in (
        ("original_first", "merged_best_original_first"),
        ("sparse_first", "merged_best_sparse_first"),
    ):
        view = summary_by_view[view_name]
        view_overlap = _overlap_case_count(view)
        payload[f"best_{preference}_summary"] = {
            "selection_counts": selection_counts[preference],
            "overlap_case_count": view_overlap,
            "auroc_overlap_case_delta": view_overlap - base_overlap,
            "metric_deltas": comparison.get(view_name, {}),
        }
    best_original = payload["best_original_first_summary"]["overlap_case_count"]
    best_sparse = payload["best_sparse_first_summary"]["overlap_case_count"]
    if best_original > best_sparse:
        recommendation = "original_first"
    elif best_sparse > best_original:
        recommendation = "sparse_first"
    else:
        recommendation = "tie"
    payload["recommendation"] = {
        "preferred_order": recommendation,
        "reason": "higher_overlap_case_count"
        if recommendation != "tie"
        else "same_overlap_case_count_compare_secondary_metrics",
    }
    return payload


def _build_per_label_auc_rows(summary_payload: dict[str, Any], view_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    view = summary_payload["views"].get(view_name, {})
    auroc = view.get("auroc", {})
    for mode in ALL_MODES:
        mode_payload = auroc.get(mode, {})
        for item in mode_payload.get("per_label", []):
            rows.append(
                {
                    "view": view_name,
                    "mode": mode,
                    "label": item.get("label", ""),
                    "support_positive": item.get("support_positive", 0),
                    "support_negative": item.get("support_negative", 0),
                    "auroc": item.get("auroc", ""),
                }
            )
    return rows


def _evaluate_records_with_fallback(
    *,
    ground_truth: dict[str, dict[str, Any]],
    candidate_sets: dict[str, dict[str, list[str]]],
    record_views: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return _evaluate_records(
            ground_truth=ground_truth,
            candidate_sets=candidate_sets,
            record_views=record_views,
        )
    except Exception as exc:
        if "sklearn" not in str(exc).lower() and "pyarrow" not in str(exc).lower() and "numpy.core.multiarray failed to import" not in str(exc).lower():
            raise

    summary_by_view: dict[str, Any] = {}
    per_case_rows: list[dict[str, Any]] = []
    mode_comparison_rows: list[dict[str, Any]] = []
    all_case_ids = sorted(ground_truth.keys())

    for view_name, mode_records in record_views.items():
        mode_summaries: dict[str, dict[str, Any]] = {}
        complete_sets: dict[str, set[str]] = {}
        for mode in ALL_MODES:
            rows: list[dict[str, Any]] = []
            current_records = mode_records[mode]
            complete_ids: set[str] = set()
            for case_id in all_case_ids:
                info = ground_truth[case_id]
                record = current_records.get(case_id, {})
                candidate_label_set = candidate_sets[mode].get(case_id, [])
                if _is_label_score_complete(record.get("label_scores", {}), candidate_label_set):
                    complete_ids.add(case_id)
                rows.append(
                    {
                        "mode": mode,
                        "query_case_id": case_id,
                        "ground_truth_labels": info["ground_truth_labels"],
                        "retrieved_count": info["retrieved_count"],
                        "predicted_labels": list(record.get("predicted_labels", [])),
                        "candidate_label_set": candidate_label_set,
                        "label_scores": record.get("label_scores", {}),
                        "strict_json_ok": bool(record.get("strict_json_ok", False)),
                        "response_success": bool(record.get("response_success", False)),
                        "diagnoses_by_category": record.get("diagnoses_by_category", {}),
                    }
                )
            complete_sets[mode] = complete_ids
            summary = _summarize_mode(rows, len(candidate_sets[mode]))
            mode_summaries[mode] = summary
            mode_comparison_rows.append({"view": view_name, "mode": mode, **summary})

        overlap = sorted(complete_sets["rag"] & complete_sets["zero"])
        summary_by_view[view_name] = {
            "mode_summaries": mode_summaries,
            "auroc_status": "skipped",
            "auroc": {
                "status": "skipped",
                "reason": "sklearn unavailable in current environment; overlap_case_count still computed",
                "overlap_case_count": len(overlap),
            },
        }

    for case_id in all_case_ids:
        row = {
            "query_case_id": case_id,
            "ground_truth_labels": " | ".join(ground_truth[case_id]["ground_truth_labels"]),
        }
        for view_name, mode_records in record_views.items():
            for mode in ALL_MODES:
                record = mode_records[mode].get(case_id, {})
                pred = list(record.get("predicted_labels", []))
                gt = ground_truth[case_id]["ground_truth_labels"]
                row[f"{view_name}_{mode}_predicted_labels"] = " | ".join(pred)
                row[f"{view_name}_{mode}_strict_json_ok"] = int(bool(record.get("strict_json_ok", False)))
                row[f"{view_name}_{mode}_response_success"] = int(bool(record.get("response_success", False)))
                row[f"{view_name}_{mode}_label_score_complete"] = int(
                    _is_label_score_complete(record.get("label_scores", {}), candidate_sets[mode].get(case_id, []))
                )
                row[f"{view_name}_{mode}_exact_match"] = int(set(pred) == set(gt))
                row[f"{view_name}_{mode}_subset_match"] = int((not pred and not gt) or set(pred).issubset(set(gt)))
        per_case_rows.append(row)

    return summary_by_view, per_case_rows, mode_comparison_rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    retry_dir = Path(args.retry_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(retry_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not retry_dir.exists():
        raise FileNotFoundError(f"Retry directory not found: {retry_dir}")

    candidate_sets: dict[str, dict[str, list[str]]] = {}
    base_records: dict[str, dict[str, dict[str, Any]]] = {}
    retry_records: dict[str, dict[str, dict[str, Any]]] = {
        "original": {},
        "sparse": {},
    }

    for mode in ALL_MODES:
        _, candidate_map = _load_request_bundle(run_dir, mode)
        candidate_sets[mode] = candidate_map
        base_records[mode] = _load_parsed_records(run_dir, mode)
        retry_records["original"][mode] = _load_retry_parsed_records(retry_dir, mode, "original")
        retry_records["sparse"][mode] = _load_retry_parsed_records(retry_dir, mode, "sparse")

    ground_truth = _load_ground_truth(run_dir)
    record_views, selection_rows, selection_counts = _build_views_and_selection_rows(
        base_records=base_records,
        retry_records=retry_records,
        candidate_sets=candidate_sets,
    )
    summary_by_view, per_case_rows, mode_comparison_rows = _evaluate_records_with_fallback(
        ground_truth=ground_truth,
        candidate_sets=candidate_sets,
        record_views=record_views,
    )
    summary_payload = _build_summary_payload(
        summary_by_view=summary_by_view,
        selection_counts=selection_counts,
    )

    selection_fieldnames = [
        "case_id",
        "mode",
        "selected_from",
        "base_complete",
        "original_complete",
        "sparse_complete",
        "base_strict_json_ok",
        "original_strict_json_ok",
        "sparse_strict_json_ok",
        "base_label_score_complete",
        "original_label_score_complete",
        "sparse_label_score_complete",
    ]
    per_case_fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["query_case_id", "ground_truth_labels"]
    mode_comparison_fieldnames = list(mode_comparison_rows[0].keys()) if mode_comparison_rows else ["view", "mode"]
    per_label_fieldnames = ["view", "mode", "label", "support_positive", "support_negative", "auroc"]

    _write_csv(
        output_dir / "reports" / "best_retry_selection_original_first.csv",
        selection_rows["original_first"],
        selection_fieldnames,
    )
    _write_csv(
        output_dir / "reports" / "best_retry_selection_sparse_first.csv",
        selection_rows["sparse_first"],
        selection_fieldnames,
    )
    _write_csv(output_dir / "evaluation" / "per_case.csv", per_case_rows, per_case_fieldnames)
    _write_csv(output_dir / "evaluation" / "mode_comparison.csv", mode_comparison_rows, mode_comparison_fieldnames)
    _write_csv(
        output_dir / "reports" / "merged_best_original_first_per_label_auc.csv",
        _build_per_label_auc_rows(summary_payload, "merged_best_original_first"),
        per_label_fieldnames,
    )
    _write_csv(
        output_dir / "reports" / "merged_best_sparse_first_per_label_auc.csv",
        _build_per_label_auc_rows(summary_payload, "merged_best_sparse_first"),
        per_label_fieldnames,
    )
    write_json(output_dir / "evaluation" / "summary.json", summary_payload)
    write_json(
        output_dir / "reports" / "best_of_retry_summary.json",
        {
            "run_dir": str(run_dir),
            "retry_dir": str(retry_dir),
            "output_dir": str(output_dir),
            **summary_payload,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "run_dir": str(run_dir),
            "retry_dir": str(retry_dir),
            "output_dir": str(output_dir),
            "created_views": list(summary_by_view.keys()),
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "views": list(summary_by_view.keys()),
                "recommendation": summary_payload["recommendation"],
                "best_original_overlap": summary_payload["best_original_first_summary"]["overlap_case_count"],
                "best_sparse_overlap": summary_payload["best_sparse_first_summary"]["overlap_case_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
