"""Rule-based incomplete-response salvage.

Purpose
-------
Repair missing structured fields from already generated response text without
new clinical generation.

Method
------
Extract explicit diagnosis labels and scores from available answer/reasoning
fragments and record every repaired field.

Inputs
------
Existing raw/parsed GLM responses, an incomplete-case manifest, and run
directory.

Outputs
-------
Updated external parsed records and salvage report.

Paper parameters
----------------
Salvage rules are deterministic; they do not call another model or modify the
retrieval set.

Shapes and meaning
------------------
One repaired case retains its original ID; recovered labels/scores occupy the
same C-column candidate space.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.response_parsing import extract_completion_message_content, extract_completion_reasoning_content


DEFAULT_DIAGNOSIS_CATEGORIES = default_report_config().labels.categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Salvage failed parsed prediction files by extracting labels and scores from raw_reasoning_content."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--report-json", type=str, default="")
    parser.add_argument("--min-default-score", type=float, default=0.6)
    parser.add_argument("--fallback-zero-scores", action="store_true")
    return parser.parse_args()


def _normalize_label(text: str) -> str:
    cleaned = str(text or "").strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"\s+", " ", cleaned.lower())
    return cleaned


def _score_patterns(label: str) -> list[re.Pattern[str]]:
    escaped = re.escape(label)
    return [
        re.compile(rf'(?im)(?:^|[\n\r]\s*[-*]?\s*){escaped}\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b'),
        re.compile(rf'(?im)"{escaped}"\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b'),
    ]


def _extract_scored_labels(text: str, candidate_labels: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in sorted(candidate_labels, key=len, reverse=True):
        for pattern in _score_patterns(label):
            match = pattern.search(text)
            if match:
                scores[label] = float(match.group(1))
                break
    return scores


def _extract_bracketed_labels(body: str, candidate_set: set[str]) -> list[str]:
    labels: list[str] = []
    for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', body):
        text = quoted[0] or quoted[1]
        normalized = _normalize_label(text)
        if normalized in candidate_set and normalized not in labels:
            labels.append(normalized)
    if labels:
        return labels

    parts = re.split(r"[,;|]", body)
    for part in parts:
        normalized = _normalize_label(part)
        if normalized in candidate_set and normalized not in labels:
            labels.append(normalized)
    return labels


def _extract_category_labels(text: str, candidate_labels: list[str]) -> dict[str, list[str]]:
    candidate_set = set(candidate_labels)
    categories = {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES}
    for category in DEFAULT_DIAGNOSIS_CATEGORIES:
        pattern = re.compile(rf'(?is){re.escape(category)}\s*:\s*\[(.*?)\]')
        match = pattern.search(text)
        if not match:
            continue
        categories[category] = _extract_bracketed_labels(match.group(1), candidate_set)
    return categories


def _extract_predicted_labels(text: str, candidate_labels: list[str]) -> list[str]:
    candidate_set = set(candidate_labels)
    patterns = [
        re.compile(r"(?is)predicted_labels?\s*:\s*\[(.*?)\]"),
        re.compile(r"(?is)final predicted_labels?\s*:\s*\[(.*?)\]"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            labels = _extract_bracketed_labels(match.group(1), candidate_set)
            if labels:
                return labels
    return []


def _extract_final_diagnosis_labels(text: str, candidate_labels: list[str]) -> list[str]:
    candidate_set = set(candidate_labels)
    final_markers = [
        "final diagnosis",
        "final diagnoses",
        "primary diagnoses",
        "final interpretation",
        "final assessment",
        "final synthesis",
    ]
    lower = text.lower()
    start = -1
    for marker in final_markers:
        idx = lower.rfind(marker)
        if idx > start:
            start = idx
    tail = text[start:] if start >= 0 else text[-2500:]
    found: list[str] = []
    for label in sorted(candidate_labels, key=len, reverse=True):
        if re.search(rf"(?i)\b{re.escape(label)}\b", tail) and label not in found:
            found.append(label)
    return found


def _is_reasonable_label_list(labels: list[str]) -> bool:
    return 0 < len(labels) <= 12


def _flatten_categories(categories: dict[str, list[str]]) -> list[str]:
    flat: list[str] = []
    for category in DEFAULT_DIAGNOSIS_CATEGORIES:
        for label in categories.get(category, []):
            if label not in flat:
                flat.append(label)
    return flat


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_payload(parsed_path: Path, raw_path: Path, *, mode: str, case_id: str) -> dict[str, Any]:
    if parsed_path.exists() and parsed_path.stat().st_size > 0:
        parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
        if raw_path.exists() and raw_path.stat().st_size > 0:
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            result = dict(raw_payload.get("result", {}) or {})
            response_body = dict(result.get("body", {}) or {})
            request_entry = dict(raw_payload.get("request_entry", {}) or {})
            if not parsed_payload.get("candidate_label_set"):
                try:
                    parsed_payload["candidate_label_set"] = list(
                        json.loads(request_entry["body"]["messages"][1]["content"]).get("candidate_label_set", [])
                    )
                except Exception:
                    parsed_payload["candidate_label_set"] = []
            if not parsed_payload.get("raw_content"):
                parsed_payload["raw_content"] = extract_completion_message_content(response_body)
            if not parsed_payload.get("raw_reasoning_content"):
                parsed_payload["raw_reasoning_content"] = extract_completion_reasoning_content(response_body)
            if not parsed_payload.get("usage"):
                parsed_payload["usage"] = response_body.get("usage", {})
            if "response_success" not in parsed_payload:
                parsed_payload["response_success"] = bool(result.get("ok", False))
            if "elapsed_seconds" not in parsed_payload:
                parsed_payload["elapsed_seconds"] = result.get("elapsed_seconds")
        return parsed_payload

    if raw_path.exists() and raw_path.stat().st_size > 0:
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        result = dict(raw_payload.get("result", {}) or {})
        response_body = dict(result.get("body", {}) or {})
        request_entry = dict(raw_payload.get("request_entry", {}) or {})
        custom_id = str(request_entry.get("custom_id", "") or f"{mode}_{case_id}")
        return {
            "custom_id": custom_id,
            "mode": mode,
            "query_case_id": case_id,
            "strict_json_ok": False,
            "answer_tag_ok": False,
            "reasoning_process": "",
            "diagnoses_by_category": {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES},
            "predicted_labels": [],
            "label_scores": {},
            "used_evidence_ids": [],
            "abstain": False,
            "uncertainty_reason": "",
            "raw_answer_text": "",
            "raw_content": extract_completion_message_content(response_body),
            "usage": response_body.get("usage", {}),
            "raw_reasoning_content": extract_completion_reasoning_content(response_body),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "response_success": bool(result.get("ok", False)),
            "candidate_label_set": list(
                json.loads(request_entry["body"]["messages"][1]["content"]).get("candidate_label_set", [])
            )
            if request_entry.get("body", {}).get("messages")
            else [],
        }

    return {
        "custom_id": f"{mode}_{case_id}",
        "mode": mode,
        "query_case_id": case_id,
        "strict_json_ok": False,
        "answer_tag_ok": False,
        "reasoning_process": "",
        "diagnoses_by_category": {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES},
        "predicted_labels": [],
        "label_scores": {},
        "used_evidence_ids": [],
        "abstain": False,
        "uncertainty_reason": "",
        "raw_answer_text": "",
        "raw_content": "",
        "usage": {},
        "raw_reasoning_content": "",
        "response_success": False,
        "candidate_label_set": [],
    }


def _salvage_record(
    payload: dict[str, Any],
    *,
    default_score: float,
    fallback_zero_scores: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_labels = [_normalize_label(label) for label in payload.get("candidate_label_set", [])]
    reasoning = str(payload.get("raw_reasoning_content", "") or "")
    categories = _extract_category_labels(reasoning, candidate_labels)
    flat_categories = _flatten_categories(categories)
    if not _is_reasonable_label_list(flat_categories):
        categories = {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES}
        flat_categories = []

    predicted_labels = _extract_predicted_labels(reasoning, candidate_labels)
    if not _is_reasonable_label_list(predicted_labels):
        predicted_labels = []
    if not predicted_labels and flat_categories:
        predicted_labels = flat_categories
    if not predicted_labels:
        predicted_labels = []

    extracted_scores = _extract_scored_labels(reasoning, candidate_labels)
    positive_score_labels = [label for label, score in extracted_scores.items() if score > 0.5]
    if positive_score_labels:
        if predicted_labels:
            predicted_labels = [label for label in predicted_labels if label in positive_score_labels] or positive_score_labels
        else:
            predicted_labels = positive_score_labels

    if not predicted_labels and not extracted_scores:
        fallback_labels = _extract_final_diagnosis_labels(reasoning, candidate_labels)
        if _is_reasonable_label_list(fallback_labels):
            predicted_labels = fallback_labels

    for label in predicted_labels:
        extracted_scores.setdefault(label, default_score)
    for label in flat_categories:
        extracted_scores.setdefault(label, default_score)

    if not extracted_scores and not fallback_zero_scores:
        return payload, {
            "salvaged": False,
            "reason": "no_labels_or_scores_extracted",
        }

    if not extracted_scores and fallback_zero_scores:
        extracted_scores = {label: 0.0 for label in candidate_labels}
        predicted_labels = []
        categories = {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES}

    full_scores = {label: float(extracted_scores.get(label, 0.0)) for label in candidate_labels}
    predicted_from_scores = [label for label in candidate_labels if full_scores[label] > 0.5]
    if not predicted_labels:
        predicted_labels = predicted_from_scores

    repaired = dict(payload)
    repaired["strict_json_ok"] = True
    repaired["reasoning_process"] = reasoning
    repaired["diagnoses_by_category"] = categories
    repaired["predicted_labels"] = predicted_labels if predicted_labels else predicted_from_scores
    repaired["label_scores"] = full_scores
    repaired["uncertainty_reason"] = repaired.get("uncertainty_reason", "") or (
        "Zero-filled from raw_reasoning_content because no reliable label extraction was possible."
        if fallback_zero_scores and not predicted_from_scores and not flat_categories
        else "Salvaged from raw_reasoning_content."
    )
    repaired["salvaged_from_reasoning"] = True
    repaired["answer_tag_ok"] = bool(repaired.get("answer_tag_ok", False))

    return repaired, {
        "salvaged": True,
        "zero_filled": bool(fallback_zero_scores and not predicted_from_scores and not flat_categories),
        "predicted_labels": repaired["predicted_labels"],
        "positive_score_labels": predicted_from_scores,
        "nonzero_score_count": sum(1 for value in full_scores.values() if value > 0.0),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.manifest_csv)
    report_path = Path(args.report_json) if args.report_json else run_dir / "salvage_reasoning_report.json"

    rows = _load_manifest(manifest_path)
    summary_rows: list[dict[str, Any]] = []
    salvaged = 0

    for row in rows:
        case_id = str(row["case_id"])
        mode = str(row["mode"])
        parsed_path = run_dir / "responses" / "parsed" / mode / f"{case_id}.json"
        raw_path = run_dir / "responses" / "raw" / mode / f"{case_id}.json"
        payload = _load_payload(parsed_path, raw_path, mode=mode, case_id=case_id)
        repaired, info = _salvage_record(
            payload,
            default_score=args.min_default_score,
            fallback_zero_scores=args.fallback_zero_scores,
        )
        if info.get("salvaged"):
            _write_json(parsed_path, repaired)
            salvaged += 1
        else:
            if not parsed_path.exists() or parsed_path.stat().st_size == 0:
                _write_json(parsed_path, payload)
        summary_rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                **info,
            }
        )

    report = {
        "run_dir": str(run_dir),
        "manifest_csv": str(manifest_path),
        "requested_case_count": len(rows),
        "salvaged_case_count": salvaged,
        "results": summary_rows,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
