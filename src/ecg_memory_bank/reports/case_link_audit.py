#!/usr/bin/env python3
"""Report-level historical case-link audit.

Purpose
-------
Measure whether final RAG reports cite valid retrieved case IDs and how
diagnoses overlap their labels.

Method
------
Compare used_evidence_ids against each query's Top-5 results, deduplicate
links, then summarize coverage, validity, and overlap.

Inputs
------
Final GLM-5 RAG parsed records, best-of selection CSV, retrieval rows, and
case labels.

Outputs
-------
External case-level and aggregate traceability tables.

Paper parameters
----------------
No bootstrap is used for the complete report set; evidence IDs must match
retrieved case IDs exactly.

Shapes and meaning
------------------
One report may cite 0–5 cases; overlap is report-level co-occurrence, not
diagnosis-specific causal attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("__ECG_DATA_ROOT_REQUIRED__/Final/LLM/outputs/mimic_eval_grounding_subset_glm5_c15")
DEFAULT_SELECTION_RELATIVE_PATH = Path(
    "retries/retry_v2_both_c15_bestof/reports/best_retry_selection_sparse_first.csv"
)
METADATA_FIELDS = ("query_case_id", "used_evidence_ids", "predicted_labels", "ground_truth_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(payload)
    return rows


def unique_strings(values: Any) -> tuple[list[str], str]:
    """Return ordered unique IDs/labels and a field-status category."""
    if values is None:
        return [], "null"
    if not isinstance(values, list):
        return [], "invalid_type"
    clean: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in clean:
            clean.append(text)
    return clean, "valid_list"


def quantile_summary(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "iqr": [None, None], "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.quantile(array, 0.5)),
        "iqr": [float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))],
        "mean": float(np.mean(array)),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def gini(values: list[int]) -> float | None:
    if not values:
        return None
    array = np.sort(np.asarray(values, dtype=np.float64))
    if float(array.sum()) == 0.0:
        return 0.0
    positions = np.arange(1, len(array) + 1, dtype=np.float64)
    return float((2.0 * np.sum(positions * array) / (len(array) * array.sum())) - (len(array) + 1.0) / len(array))


def load_retrieval_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    preview_path = run_dir / "previews" / "retrieval_preview.jsonl"
    rows = read_jsonl(preview_path)
    retrieval_by_query: dict[str, dict[str, Any]] = {}
    for row in rows:
        query = row.get("query", {})
        query_id = str(query.get("id", "")).strip()
        if not query_id:
            raise ValueError(f"Missing query.id in {preview_path}")
        if query_id in retrieval_by_query:
            raise ValueError(f"Duplicate query id {query_id} in {preview_path}")
        retrieved_cases = row.get("retrieved_cases", [])
        if not isinstance(retrieved_cases, list):
            raise ValueError(f"retrieved_cases for query {query_id} is not a list")
        case_map: dict[str, dict[str, Any]] = {}
        for rank, case in enumerate(retrieved_cases, start=1):
            case_id = str(case.get("id", "")).strip() if isinstance(case, dict) else ""
            if not case_id:
                raise ValueError(f"Query {query_id} has a retrieved case without id")
            if case_id in case_map:
                raise ValueError(f"Query {query_id} has duplicated retrieved case {case_id}")
            labels, labels_status = unique_strings(case.get("labels_clean", []) if isinstance(case, dict) else [])
            if labels_status != "valid_list":
                labels = []
            case_map[case_id] = {"rank": rank, "labels": labels}
        ground_truth, _ = unique_strings(row.get("ground_truth_labels", []))
        retrieval_by_query[query_id] = {
            "retrieved_cases": case_map,
            "retrieved_count": len(case_map),
            "ground_truth_labels": ground_truth,
        }
    return retrieval_by_query


def load_final_rag_records(run_dir: Path, selection_path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    selected: dict[str, str] = {}
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "mode", "selected_from"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Selection CSV lacks columns: {sorted(missing)}")
        for row in reader:
            if str(row.get("mode", "")).strip() != "rag":
                continue
            case_id = str(row.get("case_id", "")).strip()
            source = str(row.get("selected_from", "")).strip()
            if not case_id or source not in {"base", "original", "sparse"}:
                raise ValueError(f"Invalid RAG selection row: {row}")
            if case_id in selected:
                raise ValueError(f"Duplicate RAG selection for query {case_id}")
            selected[case_id] = source

    records: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for case_id, source in selected.items():
        if source == "base":
            record_path = run_dir / "responses" / "parsed" / "rag" / f"{case_id}.json"
        else:
            record_path = run_dir / "retries" / "retry_v2_both_c15" / "responses" / "parsed" / "rag" / source / f"{case_id}.json"
        if not record_path.exists():
            raise FileNotFoundError(f"Selected {source} RAG record is missing: {record_path}")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        observed_id = str(payload.get("query_case_id", record_path.stem)).strip()
        if observed_id != case_id:
            raise ValueError(f"Selected record {record_path} contains query id {observed_id}, expected {case_id}")
        records[case_id] = payload
        source_counts[source] += 1
    return records, source_counts


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = Path(args.selection_csv) if args.selection_csv else run_dir / DEFAULT_SELECTION_RELATIVE_PATH
    retrieval_by_query = load_retrieval_rows(run_dir)
    final_records, source_counts = load_final_rag_records(run_dir, selection_path)

    retrieval_ids = set(retrieval_by_query)
    record_ids = set(final_records)
    if retrieval_ids != record_ids:
        raise ValueError(
            "Retrieval and finalized RAG record populations differ: "
            f"retrieval_only={len(retrieval_ids - record_ids)}, record_only={len(record_ids - retrieval_ids)}"
        )

    used_field_status = Counter[str]()
    all_used_count = all_valid_count = all_invalid_count = duplicate_id_count = 0
    reports_with_valid = reports_with_invalid = reports_with_no_declared = 0
    reports_with_missing_or_invalid_field = 0
    linked_counts: list[int] = []
    used_counts: list[int] = []
    rank_counter: Counter[int] = Counter()
    case_counter: Counter[str] = Counter()
    predicted_total = predicted_overlap_total = 0
    tp_total = tp_overlap_total = fp_total = fp_overlap_total = 0
    reports_with_predicted = reports_with_any_diagnosis_overlap = 0
    per_label: dict[str, Counter[str]] = {}
    per_report_rows: list[dict[str, Any]] = []

    for query_id in sorted(retrieval_ids):
        retrieval = retrieval_by_query[query_id]
        record = final_records[query_id]
        used_present = "used_evidence_ids" in record
        used_ids, field_status = unique_strings(record.get("used_evidence_ids") if used_present else None)
        if not used_present:
            field_status = "missing"
        used_field_status[field_status] += 1
        if field_status != "valid_list":
            reports_with_missing_or_invalid_field += 1
        raw_used = record.get("used_evidence_ids", [])
        if isinstance(raw_used, list):
            raw_clean_count = sum(1 for value in raw_used if str(value).strip())
            duplicate_id_count += max(raw_clean_count - len(used_ids), 0)

        retrieved_cases = retrieval["retrieved_cases"]
        valid_ids = [case_id for case_id in used_ids if case_id in retrieved_cases]
        invalid_ids = [case_id for case_id in used_ids if case_id not in retrieved_cases]
        valid_labels: set[str] = set()
        valid_ranks: list[int] = []
        for case_id in valid_ids:
            support = retrieved_cases[case_id]
            valid_labels.update(support["labels"])
            valid_ranks.append(int(support["rank"]))
            rank_counter[int(support["rank"])] += 1
            case_counter[case_id] += 1

        predicted_labels, predicted_status = unique_strings(record.get("predicted_labels", []))
        if predicted_status != "valid_list":
            predicted_labels = []
        ground_truth = set(retrieval["ground_truth_labels"])
        predicted_set = set(predicted_labels)
        overlap_labels = predicted_set & valid_labels
        tp_labels = predicted_set & ground_truth
        fp_labels = predicted_set - ground_truth
        tp_overlap = tp_labels & valid_labels
        fp_overlap = fp_labels & valid_labels

        all_used_count += len(used_ids)
        all_valid_count += len(valid_ids)
        all_invalid_count += len(invalid_ids)
        used_counts.append(len(used_ids))
        linked_counts.append(len(valid_ids))
        reports_with_valid += int(bool(valid_ids))
        reports_with_invalid += int(bool(invalid_ids))
        reports_with_no_declared += int(not used_ids)
        predicted_total += len(predicted_set)
        predicted_overlap_total += len(overlap_labels)
        tp_total += len(tp_labels)
        tp_overlap_total += len(tp_overlap)
        fp_total += len(fp_labels)
        fp_overlap_total += len(fp_overlap)
        reports_with_predicted += int(bool(predicted_set))
        reports_with_any_diagnosis_overlap += int(bool(overlap_labels))

        for label in predicted_set:
            counter = per_label.setdefault(label, Counter())
            counter["predicted_count"] += 1
            counter["diagnosis_overlap_count"] += int(label in valid_labels)
            counter["true_positive_count"] += int(label in ground_truth)
            counter["true_positive_overlap_count"] += int(label in tp_overlap)
            counter["false_positive_count"] += int(label not in ground_truth)
            counter["false_positive_overlap_count"] += int(label in fp_overlap)

        per_report_rows.append(
            {
                "query_case_id": query_id,
                "selected_record_source": next(source for source, count in source_counts.items() if False) if False else "",
                "retrieved_case_count": retrieval["retrieved_count"],
                "used_evidence_field_status": field_status,
                "declared_used_case_count": len(used_ids),
                "valid_linked_case_count": len(valid_ids),
                "invalid_or_unverifiable_id_count": len(invalid_ids),
                "valid_link_ranks": " | ".join(str(rank) for rank in valid_ranks),
                "reported_predicted_label_count": len(predicted_set),
                "reported_diagnosis_overlap_count": len(overlap_labels),
                "true_positive_label_count": len(tp_labels),
                "true_positive_diagnosis_overlap_count": len(tp_overlap),
                "false_positive_label_count": len(fp_labels),
                "false_positive_diagnosis_overlap_count": len(fp_overlap),
                "used_evidence_ids": " | ".join(used_ids),
                "valid_evidence_ids": " | ".join(valid_ids),
                "invalid_evidence_ids": " | ".join(invalid_ids),
            }
        )

    # Recover the chosen source without reopening every response during the main loop.
    with selection_path.open("r", encoding="utf-8-sig", newline="") as handle:
        selected_source_by_query = {
            str(row["case_id"]): str(row["selected_from"])
            for row in csv.DictReader(handle)
            if str(row.get("mode", "")).strip() == "rag"
        }
    for row in per_report_rows:
        row["selected_record_source"] = selected_source_by_query[row["query_case_id"]]

    report_count = len(per_report_rows)
    linked_distribution = Counter(linked_counts)
    used_distribution = Counter(used_counts)
    support_rows: list[dict[str, Any]] = []
    total_case_links = sum(case_counter.values())
    for rank, (case_id, count) in enumerate(sorted(case_counter.items(), key=lambda item: (-item[1], item[0])), start=1):
        support_rows.append(
            {
                "rank": rank,
                "retrieved_case_id": case_id,
                "linked_report_count": int(count),
                "fraction_of_reports": ratio(int(count), report_count),
                "fraction_of_valid_case_links": ratio(int(count), total_case_links),
            }
        )
    top_10_reports = sum(row["linked_report_count"] for row in support_rows[:10])
    label_rows: list[dict[str, Any]] = []
    for label, counter in sorted(per_label.items()):
        label_rows.append(
            {
                "label": label,
                **counter,
                "reported_diagnosis_overlap_coverage": ratio(counter["diagnosis_overlap_count"], counter["predicted_count"]),
                "true_positive_overlap_coverage": ratio(counter["true_positive_overlap_count"], counter["true_positive_count"]),
                "false_positive_overlap_coverage": ratio(counter["false_positive_overlap_count"], counter["false_positive_count"]),
            }
        )

    summary = {
        "status": "completed",
        "audit_definition": {
            "evaluation_unit": "one finalized GLM-5 RAG report",
            "finalized_record_view": "merged_best_sparse_first, matched to the formal 2,381-case GLM-5 c15 CI view",
            "retrieved_cases": "the top-5 cases recorded for the same query in previews/retrieval_preview.jsonl",
            "reported_case_use": "the de-duplicated nonempty identifiers in model-output used_evidence_ids",
            "valid_case_link": "a reported evidence identifier that exactly matches an identifier in that query's recorded retrieved_cases",
            "report_case_link_coverage": "reports with >=1 valid case link / all finalized RAG reports",
            "evidence_id_validity": "valid reported case identifiers / all reported case identifiers",
            "reported_diagnosis_overlap": "reported predicted labels also present among labels of >=1 valid linked case; global used_evidence_ids are not attached to individual diagnoses",
            "causal_limit": "These are reported and verifiable links, not causal or faithful-use attribution. No intervention on retrieved cases is performed.",
        },
        "sources": {
            "run_dir": str(run_dir),
            "query_subset": str(run_dir / "subsets" / "grounding_query_subset_input.csv"),
            "retrieval_preview": str(run_dir / "previews" / "retrieval_preview.jsonl"),
            "record_selection": str(selection_path),
        },
        "population": {
            "report_count": report_count,
            "retrieval_preview_count": len(retrieval_by_query),
            "retrieval_top_k": sorted({row["retrieved_case_count"] for row in per_report_rows}),
            "selected_record_source_counts": dict(sorted(source_counts.items())),
        },
        "report_level_case_link_coverage": {
            "linked_report_count": reports_with_valid,
            "report_count": report_count,
            "coverage": ratio(reports_with_valid, report_count),
            "reports_without_valid_link": report_count - reports_with_valid,
        },
        "evidence_id_validity": {
            "declared_evidence_id_count": all_used_count,
            "valid_evidence_id_count": all_valid_count,
            "invalid_or_unverifiable_evidence_id_count": all_invalid_count,
            "validity": ratio(all_valid_count, all_used_count),
            "reports_with_at_least_one_invalid_or_unverifiable_id": reports_with_invalid,
            "reports_with_at_least_one_invalid_or_unverifiable_id_rate": ratio(reports_with_invalid, report_count),
            "reports_with_no_declared_evidence_id": reports_with_no_declared,
            "reports_with_no_declared_evidence_id_rate": ratio(reports_with_no_declared, report_count),
            "reports_with_missing_or_invalid_used_evidence_ids_field": reports_with_missing_or_invalid_field,
            "reports_with_missing_or_invalid_used_evidence_ids_field_rate": ratio(reports_with_missing_or_invalid_field, report_count),
            "removed_duplicate_declared_ids": duplicate_id_count,
            "used_evidence_ids_field_status_counts": dict(sorted(used_field_status.items())),
        },
        "reported_diagnosis_overlap": {
            "reported_positive_diagnosis_count": predicted_total,
            "diagnosis_overlap_count": predicted_overlap_total,
            "coverage": ratio(predicted_overlap_total, predicted_total),
            "reports_with_positive_diagnosis": reports_with_predicted,
            "reports_with_any_diagnosis_overlap": reports_with_any_diagnosis_overlap,
            "report_rate_among_all_reports": ratio(reports_with_any_diagnosis_overlap, report_count),
            "report_rate_among_reports_with_positive_diagnoses": ratio(reports_with_any_diagnosis_overlap, reports_with_predicted),
            "true_positive_diagnosis_count": tp_total,
            "true_positive_diagnosis_overlap_count": tp_overlap_total,
            "true_positive_overlap_coverage": ratio(tp_overlap_total, tp_total),
            "false_positive_diagnosis_count": fp_total,
            "false_positive_diagnosis_overlap_count": fp_overlap_total,
            "false_positive_overlap_coverage": ratio(fp_overlap_total, fp_total),
        },
        "linked_cases_per_report": {
            "declared_used_cases": quantile_summary(used_counts),
            "valid_linked_cases": quantile_summary(linked_counts),
            "declared_case_count_distribution": {str(key): int(value) for key, value in sorted(used_distribution.items())},
            "valid_linked_case_count_distribution": {str(key): int(value) for key, value in sorted(linked_distribution.items())},
        },
        "retrieval_rank_of_valid_linked_cases": {
            "valid_case_link_count": int(sum(rank_counter.values())),
            "rank_counts": {str(key): int(value) for key, value in sorted(rank_counter.items())},
            "rank_fractions": {str(key): ratio(int(value), int(sum(rank_counter.values()))) for key, value in sorted(rank_counter.items())},
        },
        "support_case_reuse": {
            "unique_valid_linked_cases": int(len(case_counter)),
            "valid_case_link_count": int(total_case_links),
            "top_10_case_links": int(top_10_reports),
            "top_10_fraction_of_valid_case_links": ratio(int(top_10_reports), int(total_case_links)),
            "gini_of_linked_report_frequency": gini(list(case_counter.values())),
        },
        "artifacts": {
            "summary_json": str(output_dir / "traceability_summary.json"),
            "per_report_csv": str(output_dir / "traceability_per_report.csv"),
            "support_cases_csv": str(output_dir / "traceability_support_cases.csv"),
            "per_label_csv": str(output_dir / "traceability_per_label.csv"),
        },
    }
    write_json(output_dir / "traceability_summary.json", summary)
    pd.DataFrame(per_report_rows).to_csv(output_dir / "traceability_per_report.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output_dir / "traceability_support_cases.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_dir / "traceability_per_label.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
