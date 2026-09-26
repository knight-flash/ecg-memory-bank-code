"""Report retry repair.

Purpose
-------
Retry failed or incomplete GLM outputs under the original or sparse prompt
strategy.

Method
------
Identify JSON/score-incomplete cases, resend only needed requests, and save
parsed alternatives for best-of selection.

Inputs
------
Original RAG/No-RAG run directory, API key, model settings, and failure scope.

Outputs
-------
External retry requests, raw/parsed responses, and completeness reports.

Paper parameters
----------------
Formal chain uses both retry strategies and the same GLM backend with thinking
enabled and max_tokens=8192.

Shapes and meaning
------------------
Retry rows retain original case IDs; they are candidate replacements for the
same [Q,C] evaluation records.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in os.sys.path:
    os.sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import ensure_parent, write_json, write_jsonl
from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.report_io import ensure_eval_dirs, iter_mode_paths, load_jsonl
from ecg_memory_bank.reports.response_parsing import (
    extract_completion_message_content,
    extract_completion_reasoning_content,
    find_json_candidate,
    normalize_prediction_payload,
)


DEFAULT_DIAGNOSIS_CATEGORIES = default_report_config().labels.categories
MAIN_METRIC_KEYS = (
    "strict_json_rate",
    "predicted_label_success_rate",
    "label_score_complete_rate",
    "exact_match_rate",
    "subset_match_rate",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
SPARSE_RULE_RE = re.compile(r"Rule 4: GLOBAL SCORING\..*?(?=Rule 5:)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Retry failed MIMIC eval cases and compare repaired outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["rag", "zero", "all"], default="all")
    parser.add_argument("--retry-name", type=str, default="")
    parser.add_argument(
        "--failure-scope",
        choices=["json_or_score_incomplete", "json_only", "score_only"],
        default="json_or_score_incomplete",
    )
    parser.add_argument("--retry-strategy", choices=["original", "sparse", "both"], default="both")
    parser.add_argument("--sparse-threshold", type=float, default=0.05)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default=config.direct.model)
    parser.add_argument("--concurrency", type=int, default=config.direct.concurrency)
    parser.add_argument("--batch-size", type=int, default=config.direct.batch_size)
    parser.add_argument("--endpoint", type=str, default=config.direct.endpoint)
    parser.add_argument("--max-tokens", type=int, default=config.direct.max_tokens)
    parser.add_argument("--thinking-type", type=str, default=config.direct.thinking_type)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def _extract_case_id(custom_id: str) -> str:
    return str(custom_id).partition("_")[2] or str(custom_id)


def _resolve_api_key(cli_api_key: str, config) -> str:
    return cli_api_key or os.getenv("ZHIPU_API_KEY", "").strip() or config.batch.api_key


def _retry_name(args: argparse.Namespace) -> str:
    if args.retry_name:
        return args.retry_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"retry_{args.retry_strategy}_{args.failure_scope}_{stamp}"


def _retry_root(run_dir: Path, retry_name: str) -> Path:
    return run_dir / "retries" / retry_name


def _retry_paths(retry_root: Path, mode: str, strategy: str) -> dict[str, Path]:
    return {
        "request_file": retry_root / "requests" / f"{mode}_{strategy}.jsonl",
        "raw_dir": retry_root / "responses" / "raw" / mode / strategy,
        "parsed_dir": retry_root / "responses" / "parsed" / mode / strategy,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _split_labels(values: list[str]) -> list[str]:
    clean: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in clean:
            clean.append(text)
    return clean


def _label_text(values: list[str]) -> str:
    return " | ".join(_split_labels(values))


def _load_request_bundle(run_dir: Path, mode: str) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    request_map: dict[str, dict[str, Any]] = {}
    candidate_map: dict[str, list[str]] = {}
    request_file = iter_mode_paths(run_dir, mode)["request_file"]
    for row in load_jsonl(request_file):
        custom_id = str(row.get("custom_id", ""))
        case_id = _extract_case_id(custom_id)
        request_map[case_id] = row
        try:
            user_payload = json.loads(row["body"]["messages"][1]["content"])
        except Exception:
            user_payload = {}
        candidate_map[case_id] = _split_labels(list(user_payload.get("candidate_label_set", [])))
    return request_map, candidate_map


def _load_parsed_records(run_dir: Path, mode: str) -> dict[str, dict[str, Any]]:
    parsed_dir = iter_mode_paths(run_dir, mode)["parsed_dir"]
    raw_dir = iter_mode_paths(run_dir, mode)["raw_dir"]
    parsed_records: dict[str, dict[str, Any]] = {}
    for path in sorted(parsed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(payload.get("query_case_id", path.stem))
        raw_path = raw_dir / f"{case_id}.json"
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
        payload["response_success"] = bool(raw_payload.get("result", {}).get("ok"))
        payload["raw_status_code"] = raw_payload.get("result", {}).get("status_code")
        parsed_records[case_id] = payload
    return parsed_records


def _load_ground_truth(run_dir: Path) -> dict[str, dict[str, Any]]:
    preview_rows = load_jsonl(run_dir / "previews" / "retrieval_preview.jsonl")
    mapping: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        query = row.get("query", {})
        query_id = str(query.get("id", ""))
        mapping[query_id] = {
            "ground_truth_labels": _split_labels(list(row.get("ground_truth_labels", []))),
            "retrieved_count": len(row.get("retrieved_cases", [])),
        }
    return mapping


def _is_label_score_complete(label_scores: dict[str, Any], candidate_label_set: list[str]) -> bool:
    return set(label_scores.keys()) == set(candidate_label_set)


def _failure_reasons(
    record: dict[str, Any],
    candidate_label_set: list[str],
    *,
    failure_scope: str,
) -> list[str]:
    reasons: list[str] = []
    raw_content = str(record.get("raw_content", "") or "")
    strict_json_ok = bool(record.get("strict_json_ok", False))
    label_score_complete = _is_label_score_complete(record.get("label_scores", {}), candidate_label_set)

    if failure_scope in {"json_or_score_incomplete", "json_only"}:
        if not strict_json_ok:
            reasons.append("strict_json_failed")
        if not raw_content.strip():
            reasons.append("empty_raw_content")
    if failure_scope in {"json_or_score_incomplete", "score_only"} and not label_score_complete:
        reasons.append("label_score_incomplete")
    return sorted(set(reasons))


def _make_sparse_system_prompt(system_prompt: str, threshold: float) -> str:
    threshold_text = f"{threshold:.2f}".rstrip("0").rstrip(".")
    replacement = (
        "Rule 4: SPARSE SCORING. In `label_scores`, you ONLY need to score labels from "
        "`candidate_label_set` that you assign probability > "
        f"{threshold_text}. Omit the rest. Do NOT list all labels. Any omitted candidate label "
        "will be treated as 0.0 by the evaluator.\n"
    )
    updated, count = SPARSE_RULE_RE.subn(replacement, system_prompt, count=1)
    if count != 1:
        raise ValueError("Failed to rewrite Rule 4 for sparse retry strategy.")
    return updated


def _build_retry_entry(entry: dict[str, Any], strategy: str, sparse_threshold: float) -> dict[str, Any]:
    retry_entry = copy.deepcopy(entry)
    if strategy == "sparse":
        messages = retry_entry.get("body", {}).get("messages", [])
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Retry entry missing system prompt message for sparse strategy.")
        messages[0]["content"] = _make_sparse_system_prompt(str(messages[0].get("content", "")), sparse_threshold)
    return retry_entry


def _post_single(entry: dict[str, Any], config, api_key: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_body = dict(entry["body"])
    request_body["model"] = config.direct.model
    request_body["max_tokens"] = config.direct.max_tokens
    request_body["stream"] = config.direct.stream
    request_body["thinking"] = {"type": config.direct.thinking_type}
    attempts = len(config.direct.backoff_seconds) + 1
    started = time.time()
    for attempt in range(attempts):
        try:
            response = requests.post(
                config.direct.endpoint,
                headers=headers,
                json=request_body,
                timeout=config.direct.timeout_seconds,
            )
            elapsed = round(time.time() - started, 3)
            if response.status_code == 200:
                return {
                    "ok": True,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed,
                    "body": response.json(),
                }
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < len(config.direct.backoff_seconds):
                    time.sleep(config.direct.backoff_seconds[attempt])
                    continue
            return {
                "ok": False,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "text": response.text,
            }
        except requests.RequestException as exc:
            if attempt < len(config.direct.backoff_seconds):
                time.sleep(config.direct.backoff_seconds[attempt])
                continue
            return {
                "ok": False,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
    return {"ok": False, "elapsed_seconds": round(time.time() - started, 3), "error": "unknown request failure"}


def _normalize_retry_payload(
    *,
    custom_id: str,
    response_body: dict[str, Any],
    elapsed_seconds: float | None,
    candidate_label_set: list[str],
    strategy: str,
) -> dict[str, Any]:
    raw_content = extract_completion_message_content(response_body)
    parsed_json = find_json_candidate(raw_content)
    normalized = normalize_prediction_payload(
        custom_id=custom_id,
        raw_content=raw_content,
        parsed_json=parsed_json,
    )
    normalized["usage"] = response_body.get("usage", {})
    normalized["raw_reasoning_content"] = extract_completion_reasoning_content(response_body)
    normalized["elapsed_seconds"] = elapsed_seconds
    normalized["response_success"] = True
    normalized["candidate_label_set"] = list(candidate_label_set)
    normalized["raw_empty"] = not bool(raw_content.strip())
    normalized["retry_strategy"] = strategy

    original_scores = dict(normalized.get("label_scores", {}))
    normalized["label_score_complete_before_retry_fill"] = _is_label_score_complete(original_scores, candidate_label_set)
    normalized["out_of_candidate_labels"] = sorted([label for label in original_scores if label not in candidate_label_set])
    normalized["missing_label_scores_filled_with_zero"] = []

    if strategy == "sparse" and normalized.get("strict_json_ok", False):
        filtered_scores = {label: float(value) for label, value in original_scores.items() if label in candidate_label_set}
        missing = [label for label in candidate_label_set if label not in filtered_scores]
        for label in missing:
            filtered_scores[label] = 0.0
        normalized["missing_label_scores_filled_with_zero"] = missing
        normalized["label_scores"] = {label: filtered_scores[label] for label in candidate_label_set}

    normalized["label_score_complete_after_retry"] = _is_label_score_complete(
        normalized.get("label_scores", {}),
        candidate_label_set,
    )
    return normalized


def _process_strategy(
    *,
    retry_root: Path,
    mode: str,
    strategy: str,
    failures: dict[str, dict[str, Any]],
    request_map: dict[str, dict[str, Any]],
    candidate_map: dict[str, list[str]],
    config,
    api_key: str,
    concurrency: int,
    limit: int,
    sparse_threshold: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    paths = _retry_paths(retry_root, mode, strategy)
    selected_case_ids = sorted(failures.keys())
    if limit >= 0:
        selected_case_ids = selected_case_ids[:limit]

    retry_entries = []
    case_id_to_entry: dict[str, dict[str, Any]] = {}
    for case_id in selected_case_ids:
        entry = _build_retry_entry(request_map[case_id], strategy, sparse_threshold)
        retry_entries.append(entry)
        case_id_to_entry[case_id] = entry
    write_jsonl(paths["request_file"], retry_entries)

    parsed_results: dict[str, dict[str, Any]] = {}
    report_rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(selected_case_ids), desc=f"Retry {mode} ({strategy})", unit="case")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(_post_single, case_id_to_entry[case_id], config, api_key): case_id
            for case_id in selected_case_ids
        }
        for future in as_completed(future_map):
            case_id = future_map[future]
            request_entry = case_id_to_entry[case_id]
            candidate_label_set = candidate_map.get(case_id, [])
            result = future.result()
            raw_payload = {
                "custom_id": request_entry["custom_id"],
                "mode": mode,
                "retry_strategy": strategy,
                "query_case_id": case_id,
                "request_entry": request_entry,
                "result": result,
                "original_failure_reasons": failures[case_id]["failure_reasons"],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            raw_path = paths["raw_dir"] / f"{case_id}.json"
            write_json(raw_path, raw_payload)

            parsed_path = paths["parsed_dir"] / f"{case_id}.json"
            normalized: dict[str, Any] | None = None
            if result.get("ok"):
                normalized = _normalize_retry_payload(
                    custom_id=request_entry["custom_id"],
                    response_body=result.get("body", {}),
                    elapsed_seconds=result.get("elapsed_seconds"),
                    candidate_label_set=candidate_label_set,
                    strategy=strategy,
                )
                write_json(parsed_path, normalized)
                parsed_results[case_id] = normalized

            report_rows.append(
                {
                    "mode": mode,
                    "strategy": strategy,
                    "case_id": case_id,
                    "original_failure_reasons": "|".join(failures[case_id]["failure_reasons"]),
                    "original_strict_json_ok": int(failures[case_id]["original_strict_json_ok"]),
                    "original_raw_empty": int(failures[case_id]["original_raw_empty"]),
                    "original_label_score_complete": int(failures[case_id]["original_label_score_complete"]),
                    "retry_response_success": int(bool(result.get("ok"))),
                    "retry_strict_json_ok": int(bool(normalized and normalized.get("strict_json_ok", False))),
                    "retry_raw_empty": int(bool(normalized and normalized.get("raw_empty", False))),
                    "retry_label_score_complete": int(bool(normalized and normalized.get("label_score_complete_after_retry", False))),
                    "resolved_json_failure": int(
                        bool(normalized)
                        and not normalized.get("raw_empty", False)
                        and bool(normalized.get("strict_json_ok", False))
                        and any(reason in {"strict_json_failed", "empty_raw_content"} for reason in failures[case_id]["failure_reasons"])
                    ),
                    "resolved_score_failure": int(
                        bool(normalized)
                        and bool(normalized.get("label_score_complete_after_retry", False))
                        and "label_score_incomplete" in failures[case_id]["failure_reasons"]
                    ),
                    "out_of_candidate_labels_count": len(normalized.get("out_of_candidate_labels", []) if normalized else []),
                    "missing_filled_count": len(normalized.get("missing_label_scores_filled_with_zero", []) if normalized else []),
                    "raw_path": str(raw_path),
                    "parsed_path": str(parsed_path) if normalized else "",
                }
            )
            progress.update(1)
    progress.close()
    return parsed_results, report_rows


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


def _compute_auroc_for_mode(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if not rows or not labels:
        return {"status": "skipped", "reason": "no eligible rows or labels"}

    y_true = np.asarray(
        [[1 if label in set(row["ground_truth_labels"]) else 0 for label in labels] for row in rows],
        dtype=int,
    )
    y_score = np.asarray(
        [[float(row["label_scores"][label]) for label in labels] for row in rows],
        dtype=float,
    )
    valid_label_indices = [idx for idx in range(len(labels)) if len(set(y_true[:, idx].tolist())) > 1]
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
        return {"status": "skipped", "reason": str(exc)}

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
    rag_complete = {
        row["query_case_id"]: row
        for row in mode_rows["rag"]
        if _is_label_score_complete(row.get("label_scores", {}), row.get("candidate_label_set", []))
    }
    zero_complete = {
        row["query_case_id"]: row
        for row in mode_rows["zero"]
        if _is_label_score_complete(row.get("label_scores", {}), row.get("candidate_label_set", []))
    }
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
    label_score_complete = sum(
        1
        for row in rows
        if _is_label_score_complete(row.get("label_scores", {}), row.get("candidate_label_set", []))
    )
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
        1
        for row in rows
        if (not row["predicted_labels"] and not row["ground_truth_labels"])
        or set(row["predicted_labels"]).issubset(set(row["ground_truth_labels"]))
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
    }


def _evaluate_records(
    *,
    ground_truth: dict[str, dict[str, Any]],
    candidate_sets: dict[str, dict[str, list[str]]],
    record_views: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_by_view: dict[str, Any] = {}
    per_case_rows: list[dict[str, Any]] = []
    mode_comparison_rows: list[dict[str, Any]] = []
    all_case_ids = sorted(ground_truth.keys())

    for view_name, mode_records in record_views.items():
        mode_rows: dict[str, list[dict[str, Any]]] = {}
        mode_summaries: dict[str, dict[str, Any]] = {}
        for mode in ("rag", "zero"):
            rows: list[dict[str, Any]] = []
            current_records = mode_records[mode]
            for case_id in all_case_ids:
                info = ground_truth[case_id]
                record = current_records.get(case_id, {})
                rows.append(
                    {
                        "mode": mode,
                        "query_case_id": case_id,
                        "ground_truth_labels": info["ground_truth_labels"],
                        "retrieved_count": info["retrieved_count"],
                        "predicted_labels": _split_labels(list(record.get("predicted_labels", []))),
                        "candidate_label_set": candidate_sets[mode].get(case_id, []),
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
            summary = _summarize_mode(rows, len(candidate_sets[mode]))
            mode_rows[mode] = rows
            mode_summaries[mode] = summary
            mode_comparison_rows.append({"view": view_name, "mode": mode, **summary})

        auroc_payload = _compute_overlap_auroc(mode_rows)
        summary_by_view[view_name] = {
            "mode_summaries": mode_summaries,
            "auroc_status": auroc_payload.get("status"),
            "auroc": auroc_payload,
        }

    for case_id in all_case_ids:
        row = {
            "query_case_id": case_id,
            "ground_truth_labels": _label_text(ground_truth[case_id]["ground_truth_labels"]),
        }
        for view_name, mode_records in record_views.items():
            for mode in ("rag", "zero"):
                record = mode_records[mode].get(case_id, {})
                pred = _split_labels(list(record.get("predicted_labels", [])))
                gt = ground_truth[case_id]["ground_truth_labels"]
                row[f"{view_name}_{mode}_predicted_labels"] = _label_text(pred)
                row[f"{view_name}_{mode}_strict_json_ok"] = int(bool(record.get("strict_json_ok", False)))
                row[f"{view_name}_{mode}_response_success"] = int(bool(record.get("response_success", False)))
                row[f"{view_name}_{mode}_label_score_complete"] = int(
                    _is_label_score_complete(record.get("label_scores", {}), candidate_sets[mode].get(case_id, []))
                )
                row[f"{view_name}_{mode}_exact_match"] = int(set(pred) == set(gt))
                row[f"{view_name}_{mode}_subset_match"] = int(
                    (not pred and not gt) or set(pred).issubset(set(gt))
                )
        per_case_rows.append(row)

    return summary_by_view, per_case_rows, mode_comparison_rows


def _build_record_views(
    *,
    base_records: dict[str, dict[str, dict[str, Any]]],
    retry_records: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    views: dict[str, dict[str, dict[str, Any]]] = {
        "base": {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        }
    }
    if "original" in retry_records:
        merged_original = {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        }
        for mode in ("rag", "zero"):
            for case_id, record in retry_records["original"].get(mode, {}).items():
                if bool(record.get("strict_json_ok", False)) and not bool(record.get("raw_empty", False)):
                    merged_original[mode][case_id] = record
        views["merged_original_retry"] = merged_original
    if "sparse" in retry_records:
        merged_sparse = {
            "rag": dict(base_records["rag"]),
            "zero": dict(base_records["zero"]),
        }
        for mode in ("rag", "zero"):
            for case_id, record in retry_records["sparse"].get(mode, {}).items():
                if bool(record.get("strict_json_ok", False)) and not bool(record.get("raw_empty", False)):
                    merged_sparse[mode][case_id] = record
        views["merged_sparse_retry"] = merged_sparse
    return views


def _build_comparison_payload(summary_by_view: dict[str, Any]) -> dict[str, Any]:
    base = summary_by_view.get("base", {})
    comparison: dict[str, Any] = {}
    for view_name, payload in summary_by_view.items():
        if view_name == "base":
            continue
        view_result = {
            "auroc_overlap_case_delta": payload.get("auroc", {}).get("overlap_case_count", 0)
            - base.get("auroc", {}).get("overlap_case_count", 0)
        }
        for mode in ("rag", "zero"):
            base_mode = base.get("mode_summaries", {}).get(mode, {})
            current_mode = payload.get("mode_summaries", {}).get(mode, {})
            view_result[mode] = {
                key: round(float(current_mode.get(key, 0.0)) - float(base_mode.get(key, 0.0)), 6)
                for key in MAIN_METRIC_KEYS
            }
        comparison[view_name] = view_result
    return comparison


def main() -> None:
    args = parse_args()
    config = default_report_config()
    config.direct.model = args.model
    config.direct.endpoint = args.endpoint
    config.direct.max_tokens = args.max_tokens
    config.direct.thinking_type = args.thinking_type
    api_key = _resolve_api_key(args.api_key, config)
    if not api_key:
        raise RuntimeError("Missing ZHIPU_API_KEY or --api-key.")

    run_dir = Path(args.run_dir)
    ensure_eval_dirs(run_dir)
    retry_name = _retry_name(args)
    retry_root = _retry_root(run_dir, retry_name)

    all_modes = ["rag", "zero"]
    modes = ["rag", "zero"] if args.mode == "all" else [args.mode]
    strategies = ["original", "sparse"] if args.retry_strategy == "both" else [args.retry_strategy]

    request_maps: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_maps: dict[str, dict[str, list[str]]] = {}
    base_records: dict[str, dict[str, dict[str, Any]]] = {}
    failures_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    failure_counts_by_mode: dict[str, dict[str, int]] = {}

    for mode in all_modes:
        request_map, candidate_map = _load_request_bundle(run_dir, mode)
        parsed_map = _load_parsed_records(run_dir, mode)
        request_maps[mode] = request_map
        candidate_maps[mode] = candidate_map
        base_records[mode] = parsed_map
        failures: dict[str, dict[str, Any]] = {}
        reason_counts = {
            "strict_json_failed": 0,
            "empty_raw_content": 0,
            "label_score_incomplete": 0,
            "failure_union": 0,
        }
        for case_id, entry in request_map.items():
            record = parsed_map.get(case_id, {})
            candidate_label_set = candidate_map.get(case_id, [])
            reasons = _failure_reasons(record, candidate_label_set, failure_scope=args.failure_scope)
            if not reasons:
                continue
            for reason in reasons:
                reason_counts[reason] += 1
            reason_counts["failure_union"] += 1
            failures[case_id] = {
                "request_entry": entry,
                "candidate_label_set": candidate_label_set,
                "failure_reasons": reasons,
                "original_strict_json_ok": bool(record.get("strict_json_ok", False)),
                "original_raw_empty": not bool(str(record.get("raw_content", "") or "").strip()),
                "original_label_score_complete": _is_label_score_complete(record.get("label_scores", {}), candidate_label_set),
            }
        failures_by_mode[mode] = failures
        failure_counts_by_mode[mode] = reason_counts

    retry_records: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    strategy_summary: dict[str, Any] = {}

    for strategy in strategies:
        retry_records[strategy] = {}
        strategy_summary[strategy] = {}
        for mode in modes:
            parsed_results, report_rows = _process_strategy(
                retry_root=retry_root,
                mode=mode,
                strategy=strategy,
                failures=failures_by_mode[mode],
                request_map=request_maps[mode],
                candidate_map=candidate_maps[mode],
                config=config,
                api_key=api_key,
                concurrency=args.concurrency,
                limit=args.limit,
                sparse_threshold=args.sparse_threshold,
            )
            retry_records[strategy][mode] = parsed_results
            comparison_rows.extend(report_rows)
            strategy_summary[strategy][mode] = {
                "failure_union": failure_counts_by_mode[mode]["failure_union"],
                "strict_json_failed": failure_counts_by_mode[mode]["strict_json_failed"],
                "empty_raw_content": failure_counts_by_mode[mode]["empty_raw_content"],
                "label_score_incomplete": failure_counts_by_mode[mode]["label_score_incomplete"],
                "retry_submitted": len(report_rows),
                "retry_response_success": sum(row["retry_response_success"] for row in report_rows),
                "retry_strict_json_success": sum(row["retry_strict_json_ok"] for row in report_rows),
                "retry_label_score_complete": sum(row["retry_label_score_complete"] for row in report_rows),
                "resolved_json_failure": sum(row["resolved_json_failure"] for row in report_rows),
                "resolved_score_failure": sum(row["resolved_score_failure"] for row in report_rows),
                "merge_replacements": sum(
                    1
                    for row in report_rows
                    if row["retry_strict_json_ok"] == 1 and row["retry_raw_empty"] == 0
                ),
            }

    ground_truth = _load_ground_truth(run_dir)
    all_candidate_sets = {"rag": candidate_maps.get("rag", {}), "zero": candidate_maps.get("zero", {})}
    all_base_records = {
        "rag": base_records.get("rag", {}),
        "zero": base_records.get("zero", {}),
    }
    record_views = _build_record_views(
        base_records=all_base_records,
        retry_records={
            strategy: {
                "rag": retry_records.get(strategy, {}).get("rag", {}),
                "zero": retry_records.get(strategy, {}).get("zero", {}),
            }
            for strategy in strategies
        },
    )
    summary_by_view, per_case_rows, mode_comparison_rows = _evaluate_records(
        ground_truth=ground_truth,
        candidate_sets=all_candidate_sets,
        record_views=record_views,
    )
    comparison_payload = _build_comparison_payload(summary_by_view)

    for strategy in strategies:
        view_name = "merged_original_retry" if strategy == "original" else "merged_sparse_retry"
        strategy_summary[strategy]["auroc_overlap_case_delta"] = (
            summary_by_view.get(view_name, {}).get("auroc", {}).get("overlap_case_count", 0)
            - summary_by_view.get("base", {}).get("auroc", {}).get("overlap_case_count", 0)
        )

    report_summary = {
        "run_dir": str(run_dir),
        "retry_root": str(retry_root),
        "retry_name": retry_name,
        "args": {
            "mode": args.mode,
            "failure_scope": args.failure_scope,
            "retry_strategy": args.retry_strategy,
            "sparse_threshold": args.sparse_threshold,
            "concurrency": args.concurrency,
            "batch_size": args.batch_size,
            "limit": args.limit,
        },
        "failure_counts_by_mode": failure_counts_by_mode,
        "strategy_summary": strategy_summary,
        "evaluation_comparison": comparison_payload,
    }

    report_rows_sorted = sorted(comparison_rows, key=lambda row: (row["strategy"], row["mode"], row["case_id"]))
    report_fieldnames = [
        "mode",
        "strategy",
        "case_id",
        "original_failure_reasons",
        "original_strict_json_ok",
        "original_raw_empty",
        "original_label_score_complete",
        "retry_response_success",
        "retry_strict_json_ok",
        "retry_raw_empty",
        "retry_label_score_complete",
        "resolved_json_failure",
        "resolved_score_failure",
        "out_of_candidate_labels_count",
        "missing_filled_count",
        "raw_path",
        "parsed_path",
    ]
    _write_csv(retry_root / "reports" / "retry_case_comparison.csv", report_rows_sorted, report_fieldnames)
    write_json(retry_root / "reports" / "retry_summary.json", report_summary)

    per_case_fieldnames = list(per_case_rows[0].keys()) if per_case_rows else ["query_case_id", "ground_truth_labels"]
    mode_comparison_fieldnames = list(mode_comparison_rows[0].keys()) if mode_comparison_rows else ["view", "mode"]
    _write_csv(retry_root / "evaluation" / "per_case.csv", per_case_rows, per_case_fieldnames)
    _write_csv(retry_root / "evaluation" / "mode_comparison.csv", mode_comparison_rows, mode_comparison_fieldnames)
    write_json(
        retry_root / "evaluation" / "summary.json",
        {
            "run_dir": str(run_dir),
            "retry_root": str(retry_root),
            "views": summary_by_view,
            "comparison": comparison_payload,
        },
    )
    write_json(
        retry_root / "manifest.json",
        {
            "run_dir": str(run_dir),
            "retry_root": str(retry_root),
            "retry_name": retry_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modes": modes,
            "strategies": strategies,
            "failure_scope": args.failure_scope,
            "sparse_threshold": args.sparse_threshold,
        },
    )

    print(
        json.dumps(
            {
                "retry_root": str(retry_root),
                "modes": modes,
                "strategies": strategies,
                "failure_counts_by_mode": failure_counts_by_mode,
                "evaluation_views": list(summary_by_view.keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
