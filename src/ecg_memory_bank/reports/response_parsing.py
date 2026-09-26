"""Fault-tolerant report JSON parsing.

Purpose
-------
Recover the required structured diagnosis object from LLM responses without
generating a new diagnosis.

Method
------
Accept strict JSON, fenced JSON, or recoverable embedded JSON; normalize
completion content and category/label fields.

Inputs
------
Raw chat-completion bodies or text for one ECG report.

Outputs
-------
Parsed report dictionaries and completeness indicators used by retry and
evaluation.

Paper parameters
----------------
Parsing is rule-based; no LLM calls or adjustable clinical thresholds occur
here.

Shapes and meaning
------------------
label_scores maps candidate labels to [0,1] values; used_evidence_ids are case
identifiers declared at report level.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ecg_memory_bank.reports.config import default_report_config


CODEBLOCK_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
DEFAULT_DIAGNOSIS_CATEGORIES = default_report_config().labels.categories


def extract_message_content(batch_line: dict[str, Any]) -> str:
    return (
        batch_line.get("response", {})
        .get("body", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def extract_completion_message_content(response_payload: dict[str, Any]) -> str:
    message = response_payload.get("choices", [{}])[0].get("message", {})
    return _flatten_content(message.get("content", ""))


def extract_completion_reasoning_content(response_payload: dict[str, Any]) -> str:
    message = response_payload.get("choices", [{}])[0].get("message", {})
    reasoning = message.get("reasoning_content", "")
    return _flatten_content(reasoning).strip()


def try_parse_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def find_json_candidate(text: str) -> dict[str, Any] | None:
    parsed = try_parse_json_blob(text)
    if parsed is not None:
        return parsed

    match = CODEBLOCK_JSON_RE.search(text)
    if match:
        parsed = try_parse_json_blob(match.group(1))
        if parsed is not None:
            return parsed

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        parsed = try_parse_json_blob(text[first : last + 1])
        if parsed is not None:
            return parsed
    return None


def extract_answer_text(text: str) -> str:
    matches = ANSWER_TAG_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


def split_answer_labels(answer_text: str) -> list[str]:
    if not answer_text:
        return []
    parts = re.split(r";|\n|, (?=[A-Z])", answer_text)
    labels: list[str] = []
    for part in parts:
        label = part.strip().strip(";")
        if label and label not in labels:
            labels.append(label)
    return labels


def normalize_prediction_payload(
    *,
    custom_id: str,
    raw_content: str,
    parsed_json: dict[str, Any] | None,
) -> dict[str, Any]:
    answer_text = extract_answer_text(raw_content)
    labels_from_answer = split_answer_labels(answer_text)

    predicted_labels = []
    label_scores: dict[str, float] = {}
    used_evidence_ids: list[str] = []
    diagnoses_by_category = {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES}
    abstain = False
    uncertainty_reason = ""
    reasoning_process = ""
    strict_json_ok = parsed_json is not None

    if parsed_json is not None:
        reasoning_process = str(parsed_json.get("reasoning_process", "") or "")
        raw_categories = parsed_json.get("diagnoses_by_category", {})
        if isinstance(raw_categories, dict):
            for key in diagnoses_by_category:
                values = raw_categories.get(key, [])
                if isinstance(values, list):
                    diagnoses_by_category[key] = [str(item) for item in values if str(item).strip()]
        predicted_labels = [str(item) for item in parsed_json.get("predicted_labels", []) if str(item).strip()]
        raw_scores = parsed_json.get("label_scores", {})
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                try:
                    label_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        raw_evidence = parsed_json.get("used_evidence_ids", [])
        if isinstance(raw_evidence, list):
            used_evidence_ids = [str(item) for item in raw_evidence if str(item).strip()]
        abstain = bool(parsed_json.get("abstain", False))
        uncertainty_reason = str(parsed_json.get("uncertainty_reason", "") or "")

    if not predicted_labels and labels_from_answer:
        predicted_labels = labels_from_answer
    if not predicted_labels and any(diagnoses_by_category.values()):
        flat_labels: list[str] = []
        for values in diagnoses_by_category.values():
            for label in values:
                if label not in flat_labels:
                    flat_labels.append(label)
        predicted_labels = flat_labels

    mode, _, case_id = custom_id.partition("_")
    return {
        "custom_id": custom_id,
        "mode": mode,
        "query_case_id": case_id if case_id else custom_id,
        "strict_json_ok": strict_json_ok,
        "answer_tag_ok": bool(answer_text),
        "reasoning_process": reasoning_process,
        "diagnoses_by_category": diagnoses_by_category,
        "predicted_labels": predicted_labels,
        "label_scores": label_scores,
        "used_evidence_ids": used_evidence_ids,
        "abstain": abstain,
        "uncertainty_reason": uncertainty_reason,
        "raw_answer_text": answer_text,
        "raw_content": raw_content,
    }
