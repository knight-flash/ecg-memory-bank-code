from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import write_json
from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.report_io import REQUEST_FILE_MAP, ensure_eval_dirs, iter_mode_paths, load_jsonl


DEFAULT_DIAGNOSIS_CATEGORIES = default_report_config().labels.categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 1000-query MIMIC self-RAG direct inference outputs.")
    parser.add_argument("--run-name", type=str, default=default_report_config().grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--case-id", type=str, default="")
    return parser.parse_args()


def _resolve_run_dir(run_name: str, run_dir_arg: str) -> Path:
    return Path(run_dir_arg) if run_dir_arg else Path(run_name)


def _split_labels(values: list[str]) -> list[str]:
    clean: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in clean:
            clean.append(text)
    return clean


def _label_text(values: list[str]) -> str:
    return " | ".join(_split_labels(values))


def _load_ground_truth(run_dir: Path, case_id: str) -> dict[str, dict[str, Any]]:
    preview_rows = load_jsonl(run_dir / "previews" / "retrieval_preview.jsonl")
    mapping: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        query = row.get("query", {})
        query_id = str(query.get("id", ""))
        if case_id and query_id != case_id:
            continue
        mapping[query_id] = {
            "ground_truth_labels": _split_labels(list(row.get("ground_truth_labels", []))),
            "retrieved_count": len(row.get("retrieved_cases", [])),
        }
    return mapping


def _load_candidate_sets(run_dir: Path, mode: str, case_id: str) -> dict[str, list[str]]:
    request_file = iter_mode_paths(run_dir, mode)["request_file"]
    rows = load_jsonl(request_file)
    mapping: dict[str, list[str]] = {}
    for row in rows:
        custom_id = str(row.get("custom_id", ""))
        current_case_id = custom_id.partition("_")[2] or custom_id
        if case_id and current_case_id != case_id:
            continue
        try:
            user_payload = json.loads(row["body"]["messages"][1]["content"])
        except Exception:
            user_payload = {}
        mapping[current_case_id] = _split_labels(list(user_payload.get("candidate_label_set", [])))
    return mapping


def _load_mode_records(run_dir: Path, mode: str, case_id: str) -> dict[str, dict[str, Any]]:
    parsed_dir = iter_mode_paths(run_dir, mode)["parsed_dir"]
    raw_dir = iter_mode_paths(run_dir, mode)["raw_dir"]
    parsed_records: dict[str, dict[str, Any]] = {}
    for path in tqdm(sorted(parsed_dir.glob("*.json")), desc=f"Load parsed ({mode})", unit="file"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_case_id = str(payload.get("query_case_id", path.stem))
        if case_id and current_case_id != case_id:
            continue
        raw_path = raw_dir / f"{current_case_id}.json"
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
        parsed_records[current_case_id] = {
            **payload,
            "response_success": bool(raw_payload.get("result", {}).get("ok")),
        }
    return parsed_records


def _compute_binary_metrics(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    tp = fp = fn = 0
    support = 0
    for row in rows:
        gt = set(row["ground_truth_labels"])
        pred = set(row["predicted_labels"])
        gt_has = label in gt
        pred_has = label in pred
        if gt_has:
            support += 1
        if gt_has and pred_has:
            tp += 1
        elif pred_has and not gt_has:
            fp += 1
        elif gt_has and not pred_has:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label": label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _is_label_score_complete(row: dict[str, Any]) -> bool:
    return set(row.get("label_scores", {}).keys()) == set(row.get("candidate_label_set", []))


def _compute_auroc_for_mode(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    if not rows or not labels:
        return {
            "status": "skipped",
            "reason": "no eligible rows or labels",
        }

    y_true = np.asarray(
        [
            [1 if label in set(row["ground_truth_labels"]) else 0 for label in labels]
            for row in rows
        ],
        dtype=int,
    )
    y_score = np.asarray(
        [
            [float(row["label_scores"][label]) for label in labels]
            for row in rows
        ],
        dtype=float,
    )

    valid_label_indices = [
        idx
        for idx in range(len(labels))
        if len(set(y_true[:, idx].tolist())) > 1
    ]
    if not valid_label_indices:
        return {
            "status": "skipped",
            "reason": "no labels with both positive and negative samples in overlap subset",
        }

    valid_labels = [labels[idx] for idx in valid_label_indices]
    valid_true = y_true[:, valid_label_indices]
    valid_score = y_score[:, valid_label_indices]
    try:
        macro_auroc = float(roc_auc_score(valid_true, valid_score, average="macro"))
        micro_auroc = float(roc_auc_score(valid_true, valid_score, average="micro"))
    except ValueError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
        }

    per_label = []
    for idx, label in enumerate(valid_labels):
        per_label.append(
            {
                "label": label,
                "support_positive": int(valid_true[:, idx].sum()),
                "support_negative": int((valid_true[:, idx] == 0).sum()),
                "auroc": round(float(roc_auc_score(valid_true[:, idx], valid_score[:, idx])), 6),
            }
        )
    return {
        "status": "ok",
        "overlap_case_count": len(rows),
        "eligible_label_count": len(valid_labels),
        "eligible_labels": valid_labels,
        "macro_auroc": round(macro_auroc, 6),
        "micro_auroc": round(micro_auroc, 6),
        "per_label": per_label,
    }


def _compute_overlap_auroc(mode_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rag_complete = {row["query_case_id"]: row for row in mode_rows["rag"] if _is_label_score_complete(row)}
    zero_complete = {row["query_case_id"]: row for row in mode_rows["zero"] if _is_label_score_complete(row)}
    overlap_ids = sorted(set(rag_complete) & set(zero_complete))
    if not overlap_ids:
        return {
            "status": "skipped",
            "reason": "no overlapping cases with complete label_scores in both modes",
        }

    rag_rows = [rag_complete[case_id] for case_id in overlap_ids]
    zero_rows = [zero_complete[case_id] for case_id in overlap_ids]

    rag_label_sets = {tuple(row["candidate_label_set"]) for row in rag_rows}
    zero_label_sets = {tuple(row["candidate_label_set"]) for row in zero_rows}
    if len(rag_label_sets) != 1 or len(zero_label_sets) != 1 or rag_label_sets != zero_label_sets:
        return {
            "status": "skipped",
            "reason": "candidate_label_set is not aligned across the overlap subset",
            "overlap_case_count": len(overlap_ids),
        }

    labels = list(next(iter(rag_label_sets)))
    rag_result = _compute_auroc_for_mode(rag_rows, labels)
    zero_result = _compute_auroc_for_mode(zero_rows, labels)
    overall_status = "ok" if rag_result.get("status") == "ok" and zero_result.get("status") == "ok" else "partial"
    return {
        "status": overall_status,
        "overlap_case_count": len(overlap_ids),
        "rag": rag_result,
        "zero": zero_result,
    }


def _summarize_mode(rows: list[dict[str, Any]], request_total: int) -> dict[str, Any]:
    response_success = sum(1 for row in rows if row["response_success"])
    parse_success = len(rows)
    strict_json_success = sum(1 for row in rows if row.get("strict_json_ok"))
    predicted_label_success = sum(1 for row in rows if row.get("predicted_labels"))
    label_score_complete = sum(1 for row in rows if set(row.get("label_scores", {}).keys()) == set(row.get("candidate_label_set", [])))
    diagnoses_by_category_complete = sum(
        1
        for row in rows
        if all(key in row.get("diagnoses_by_category", {}) for key in DEFAULT_DIAGNOSIS_CATEGORIES)
    )
    category_nonempty_counts = {
        category: sum(1 for row in rows if row.get("diagnoses_by_category", {}).get(category))
        for category in DEFAULT_DIAGNOSIS_CATEGORIES
    }
    exact_match = sum(1 for row in rows if set(row["predicted_labels"]) == set(row["ground_truth_labels"]))
    subset_match = sum(
        1 for row in rows
        if (not row["predicted_labels"] and not row["ground_truth_labels"]) or set(row["predicted_labels"]).issubset(set(row["ground_truth_labels"]))
    )

    tp = fp = fn = 0
    label_universe: set[str] = set()
    for row in rows:
        gt = set(row["ground_truth_labels"])
        pred = set(row["predicted_labels"])
        label_universe.update(gt)
        label_universe.update(pred)
        tp += len(gt & pred)
        fp += len(pred - gt)
        fn += len(gt - pred)

    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    per_label_rows = [_compute_binary_metrics(rows, label) for label in sorted(label_universe)]
    macro_precision = sum(item["precision"] for item in per_label_rows) / len(per_label_rows) if per_label_rows else 0.0
    macro_recall = sum(item["recall"] for item in per_label_rows) / len(per_label_rows) if per_label_rows else 0.0
    macro_f1 = sum(item["f1"] for item in per_label_rows) / len(per_label_rows) if per_label_rows else 0.0

    return {
        "request_total": request_total,
        "response_success": response_success,
        "parse_success": parse_success,
        "strict_json_success": strict_json_success,
        "predicted_label_success": predicted_label_success,
        "label_score_complete": label_score_complete,
        "diagnoses_by_category_complete": diagnoses_by_category_complete,
        "exact_match": exact_match,
        "subset_match": subset_match,
        "strict_json_rate": round(strict_json_success / request_total, 6) if request_total else 0.0,
        "predicted_label_success_rate": round(predicted_label_success / request_total, 6) if request_total else 0.0,
        "label_score_complete_rate": round(label_score_complete / request_total, 6) if request_total else 0.0,
        "diagnoses_by_category_complete_rate": round(diagnoses_by_category_complete / request_total, 6) if request_total else 0.0,
        "category_nonempty_counts": category_nonempty_counts,
        "exact_match_rate": round(exact_match / request_total, 6) if request_total else 0.0,
        "subset_match_rate": round(subset_match / request_total, 6) if request_total else 0.0,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_label_rows": per_label_rows,
    }


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_name, args.run_dir)
    dirs = ensure_eval_dirs(run_dir)
    ground_truth = _load_ground_truth(run_dir, args.case_id)

    mode_rows: dict[str, list[dict[str, Any]]] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}
    per_label_frames: list[pd.DataFrame] = []

    for mode in ("rag", "zero"):
        candidate_sets = _load_candidate_sets(run_dir, mode, args.case_id)
        parsed_records = _load_mode_records(run_dir, mode, args.case_id)
        rows: list[dict[str, Any]] = []
        for query_case_id, info in tqdm(ground_truth.items(), desc=f"Align metrics ({mode})", unit="case"):
            record = parsed_records.get(query_case_id, {})
            rows.append(
                {
                    "mode": mode,
                    "query_case_id": query_case_id,
                    "ground_truth_labels": info["ground_truth_labels"],
                    "retrieved_count": info["retrieved_count"],
                    "predicted_labels": _split_labels(list(record.get("predicted_labels", []))),
                    "candidate_label_set": candidate_sets.get(query_case_id, []),
                    "label_scores": record.get("label_scores", {}),
                    "strict_json_ok": bool(record.get("strict_json_ok", False)),
                    "response_success": bool(record.get("response_success", False)),
                    "reasoning_process": record.get("reasoning_process", ""),
                    "diagnoses_by_category": record.get("diagnoses_by_category", {}),
                    "used_evidence_ids": record.get("used_evidence_ids", []),
                    "abstain": bool(record.get("abstain", False)),
                    "uncertainty_reason": record.get("uncertainty_reason", ""),
                }
            )

        request_total = len(candidate_sets)
        summary = _summarize_mode(rows, request_total)
        mode_rows[mode] = rows
        mode_summaries[mode] = {key: value for key, value in summary.items() if key != "per_label_rows"}
        label_df = pd.DataFrame(summary["per_label_rows"])
        if not label_df.empty:
            label_df.insert(0, "mode", mode)
            per_label_frames.append(label_df)

    comparison_df = pd.DataFrame(
        [{"mode": mode, **summary} for mode, summary in mode_summaries.items()]
    )

    all_case_ids = sorted(ground_truth.keys())
    per_case_rows: list[dict[str, Any]] = []
    for case_id in tqdm(all_case_ids, desc="Build per-case comparison", unit="case"):
        rag_row = next((row for row in mode_rows["rag"] if row["query_case_id"] == case_id), {})
        zero_row = next((row for row in mode_rows["zero"] if row["query_case_id"] == case_id), {})
        gt_labels = ground_truth[case_id]["ground_truth_labels"]
        rag_pred = rag_row.get("predicted_labels", [])
        zero_pred = zero_row.get("predicted_labels", [])
        per_case_rows.append(
            {
                "query_case_id": case_id,
                "ground_truth_labels": _label_text(gt_labels),
                "rag_predicted_labels": _label_text(rag_pred),
                "zero_predicted_labels": _label_text(zero_pred),
                "rag_exact_match": int(set(rag_pred) == set(gt_labels)),
                "zero_exact_match": int(set(zero_pred) == set(gt_labels)),
                "rag_subset_match": int((not rag_pred and not gt_labels) or set(rag_pred).issubset(set(gt_labels))),
                "zero_subset_match": int((not zero_pred and not gt_labels) or set(zero_pred).issubset(set(gt_labels))),
                "rag_strict_json_ok": int(bool(rag_row.get("strict_json_ok", False))),
                "zero_strict_json_ok": int(bool(zero_row.get("strict_json_ok", False))),
                "rag_response_success": int(bool(rag_row.get("response_success", False))),
                "zero_response_success": int(bool(zero_row.get("response_success", False))),
                "rag_retrieved_evidence_count": len(rag_row.get("used_evidence_ids", [])),
                "zero_retrieved_evidence_count": len(zero_row.get("used_evidence_ids", [])),
                "rag_diagnoses_by_category": json.dumps(rag_row.get("diagnoses_by_category", {}), ensure_ascii=False),
                "zero_diagnoses_by_category": json.dumps(zero_row.get("diagnoses_by_category", {}), ensure_ascii=False),
            }
        )

    per_case_df = pd.DataFrame(per_case_rows)
    per_label_df = pd.concat(per_label_frames, ignore_index=True) if per_label_frames else pd.DataFrame()
    auroc_payload = _compute_overlap_auroc(mode_rows)

    comparison_df.to_csv(dirs["metrics"] / "mode_comparison.csv", index=False)
    per_case_df.to_csv(dirs["metrics"] / "per_case.csv", index=False)
    per_label_df.to_csv(dirs["metrics"] / "per_label.csv", index=False)

    summary_payload = {
        "run_dir": str(run_dir),
        "request_files": REQUEST_FILE_MAP,
        "mode_summaries": mode_summaries,
        "auroc_status": auroc_payload.get("status"),
        "auroc": auroc_payload,
    }
    write_json(dirs["metrics"] / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
