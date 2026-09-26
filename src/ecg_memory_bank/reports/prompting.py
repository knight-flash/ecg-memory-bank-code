"""Grounded and ungrounded ECG prompts.

Purpose
-------
Build paired prompt messages with the same query data and candidate labels for
RAG and No-RAG modes.

Method
------
Serialize query features and nine diagnostic slots; add Top-5 historical cases
only for RAG and require ordered structured JSON output.

Inputs
------
Case features, retrieved-case labels/report snippets/similarities, candidate
taxonomy, and prompt configuration.

Outputs
-------
System/user chat messages for the GLM backend.

Paper parameters
----------------
Paper prompt v2 uses K=5 historical cases, temperature=0.1 downstream, and the
same label list for both modes.

Shapes and meaning
------------------
Each response is one JSON object with reasoning_process,
diagnoses_by_category, predicted_labels, label_scores, used_evidence_ids,
abstain, uncertainty_reason.
"""

from __future__ import annotations

import json
from typing import Any

from ecg_memory_bank.reports.config import default_report_config


DEFAULT_DIAGNOSIS_CATEGORIES = default_report_config().labels.categories


def _build_system_prompt_v1() -> str:
    return (
        "You are an expert clinical ECG assistant. Diagnose the patient in `query_case` using the provided ECG evidence.\n\n"
        "=== EVIDENCE USAGE ===\n"
        "You may or may not be provided with `retrieved_cases`.\n"
        "- IF `retrieved_cases` ARE PROVIDED: use them as historical comparative evidence only.\n"
        "- IF `retrieved_cases` IS EMPTY ([]): rely SOLELY on your internal medical knowledge and the evidence in `query_case`.\n"
        "Do not complain about missing references.\n\n"
        "=== ANTI-SYCOPHANCY ===\n"
        "Retrieved cases may be noisy, weakly matched, partially wrong, or clinically irrelevant.\n"
        "Do NOT copy their labels blindly. If the query evidence conflicts with retrieved labels or reports, trust the query evidence and explicitly say the retrieval is weak or contradictory in `reasoning_process`.\n\n"
        "=== DATA USE ===\n"
        "If structured ECG features are present, prioritize them for rhythm, conduction, axis, and morphology reasoning.\n"
        "If `retrieved_cases` are present, treat them as reference evidence only, not ground truth.\n"
        "If raw signal arrays are absent, rely on the structured ECG evidence and metadata in the payload.\n\n"
        "=== REASONING PROTOCOL ===\n"
        "In `reasoning_process`, reason in this order:\n"
        "1. Quantitative baseline: heart rate, regularity, and overall rhythm impression.\n"
        "2. Waveform analysis: P-wave behavior, QRS width/morphology, ST-T abnormalities, axis or polarity clues.\n"
        "3. Retrieval comparison: which retrieved cases support or contradict the query, if any.\n"
        "4. Differential diagnosis: explain why likely labels are favored and why close alternatives are down-weighted.\n"
        "5. Category synthesis: assign candidate labels into the required diagnostic categories.\n"
        "6. Final synthesis: map the reasoning to `predicted_labels` and `label_scores`.\n\n"
    )


def _build_system_prompt_v2() -> str:
    return (
        "You are an expert clinical ECG assistant. Diagnose the patient in `query_case` using the provided ECG evidence.\n\n"
        "=== EVIDENCE USAGE ===\n"
        "You may or may not be provided with `retrieved_cases`.\n"
        "- IF `retrieved_cases` ARE PROVIDED: use them as historical comparative evidence only.\n"
        "- IF `retrieved_cases` IS EMPTY ([]): rely SOLELY on your internal medical knowledge and the evidence in `query_case`.\n"
        "Do not complain about missing references.\n\n"
        "=== ANTI-SYCOPHANCY ===\n"
        "Retrieved cases may be noisy, weakly matched, partially wrong, or clinically irrelevant.\n"
        "Do NOT copy their labels blindly. If the query evidence conflicts with retrieved labels or reports, trust the query evidence and explicitly say the retrieval is weak or contradictory in `reasoning_process`.\n\n"
        "=== DATA USE ===\n"
        "If structured ECG features are present, prioritize stable, cross-lead-consistent findings over isolated extreme values.\n"
        "If some measurements are physiologically implausible or internally contradictory, explicitly treat them as possible extraction noise instead of over-calling disease.\n"
        "If `retrieved_cases` are present, use them to check whether unusual feature patterns repeat across similar cases, but do not treat them as ground truth.\n"
        "If raw signal arrays are absent, rely on the structured ECG evidence and metadata in the payload.\n\n"
        "=== REASONING PROTOCOL ===\n"
        "In `reasoning_process`, reason in this order:\n"
        "1. Quantitative baseline: heart rate, regularity, and overall rhythm impression.\n"
        "2. Consistency audit: identify measurements that are implausible, contradictory across leads, or likely extraction artifacts.\n"
        "3. Waveform analysis: P-wave behavior, QRS width/morphology, ST-T abnormalities, axis or polarity clues.\n"
        "4. Retrieval comparison: which retrieved cases support or contradict the query, if any.\n"
        "5. Differential diagnosis: explain why likely labels are favored and why close alternatives are down-weighted.\n"
        "6. Category synthesis: assign candidate labels into the required diagnostic categories.\n"
        "7. Final synthesis: map the reasoning to `predicted_labels` and `label_scores`.\n\n"
    )


def build_system_prompt(*, strict_json: bool, prompt_version: str = "v1") -> str:
    if prompt_version == "v1":
        base = _build_system_prompt_v1()
    elif prompt_version == "v2":
        base = _build_system_prompt_v2()
    else:
        raise ValueError(f"Unsupported prompt_version: {prompt_version}")
    rules = (
        "=== STRICT OUTPUT RULES ===\n"
        "Rule 1: NO ABSTENTION. You are NOT allowed to refuse answering. Always provide your best clinical estimate.\n"
        "Rule 2: JSON CHAIN OF THOUGHT. You MUST put your reasoning inside the `reasoning_process` field, and it MUST be the first key in the JSON object.\n"
        "Rule 3: CATEGORY SLOTS. You MUST output `diagnoses_by_category` with exactly these keys: "
        "`Rhythm`, `Conduction`, `Ischemia events`, `Hypertrophy`, `Pacemaker`, `General report`, `ST-T changes`, `Axis`, `Others`. Each key must be present. "
        "If no diagnosis belongs to that category, output an empty list [].\n"
        "Rule 4: GLOBAL SCORING. You MUST score EVERY label in `candidate_label_set`. `label_scores` must contain exactly those labels as keys, each mapped to a float probability between 0.0 and 1.0.\n"
        "Rule 5: CATEGORY VALIDITY. Every label placed inside `diagnoses_by_category` must come from `candidate_label_set`. Do not invent labels.\n"
        "Rule 6: PREDICTED LABELS. `predicted_labels` must be the flattened union of labels from `diagnoses_by_category`, restricted to labels from `candidate_label_set` with score > 0.5. If no label exceeds 0.5, return an empty list [].\n"
        "Rule 7: EVIDENCE IDS. `used_evidence_ids` may only contain real `retrieved_cases.id` values that you actually used. If `retrieved_cases` is empty, return [].\n"
        "Rule 8: UNCERTAINTY. `abstain` must always be false. Use `uncertainty_reason` to describe uncertainty instead of refusing.\n\n"
        "Return ONLY one valid JSON object with keys in this order: "
        "`reasoning_process`, `diagnoses_by_category`, `predicted_labels`, `label_scores`, `used_evidence_ids`, `abstain`, `uncertainty_reason`."
    )
    if strict_json:
        rules += " Do not include markdown, code fences, or any text outside the JSON object."
    else:
        rules += " Even in non-strict mode, the final output must still be a single valid JSON object."
    return base + rules


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text


def _case_signal_summary(case_payload: dict[str, Any]) -> dict[str, Any]:
    selected_leads = case_payload.get("selectedLeads", {})
    lead_names = [str(name) for name in selected_leads.keys()]
    sample_count = 0
    if selected_leads:
        first_values = next(iter(selected_leads.values()), [])
        if isinstance(first_values, list):
            sample_count = len(first_values)
    return {
        "ecg_available": bool(case_payload.get("ecgAvailable")),
        "sampling_rate": case_payload.get("samplingRate"),
        "lead_names": lead_names,
        "lead_count": len(lead_names),
        "samples_per_lead": sample_count,
        "serialization_error": case_payload.get("serializationError"),
    }


def _build_query_summary(case_payload: dict[str, Any]) -> str:
    signal_info = _case_signal_summary(case_payload)
    lead_names = signal_info["lead_names"]
    leads_text = ", ".join(lead_names) if lead_names else "no leads available"
    availability = "available" if signal_info["ecg_available"] else "unavailable"
    return (
        f"Query ECG summary: signal {availability}; sampling_rate={signal_info['sampling_rate']}; "
        f"lead_count={signal_info['lead_count']}; leads={leads_text}; "
        f"samples_per_lead={signal_info['samples_per_lead']}."
    )


def _build_retrieval_summary(case_payload: dict[str, Any]) -> str:
    signal_info = _case_signal_summary(case_payload)
    report = _clean_text(case_payload.get("report"))
    lead_names = signal_info["lead_names"]
    leads_text = ", ".join(lead_names) if lead_names else "no leads available"
    base = (
        f"Historical reference summary: signal_available={signal_info['ecg_available']}; "
        f"sampling_rate={signal_info['sampling_rate']}; leads={leads_text}."
    )
    if report:
        return f"{base} Report snippet: {report[:280]}"
    return base


def _build_query_structured_features(case_payload: dict[str, Any]) -> dict[str, Any]:
    if "structured_features" in case_payload:
        return dict(case_payload.get("structured_features", {}))
    signal_info = _case_signal_summary(case_payload)
    return {
        "demographics": {
            "age": case_payload.get("age"),
            "gender": case_payload.get("gender"),
            "time": case_payload.get("time"),
        },
        "signal_metadata": signal_info,
    }


def _build_retrieval_structured_features(case_payload: dict[str, Any]) -> dict[str, Any]:
    if "structured_features" in case_payload:
        return dict(case_payload.get("structured_features", {}))
    signal_info = _case_signal_summary(case_payload)
    report = _clean_text(case_payload.get("report"))
    return {
        "signal_metadata": signal_info,
        "report_available": bool(report),
        "report_preview": report[:280] if report else "",
    }


def build_user_payload(
    query_case: dict[str, Any],
    retrieved_cases: list[dict[str, Any]],
    *,
    global_label_set: list[str],
    candidate_labels_by_category: dict[str, list[str]] | None,
    include_raw_signals: bool,
    include_query_summary: bool,
    include_retrieved_labels: bool,
    include_retrieved_reports: bool,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    compact_query = {
        "id": query_case["id"],
        "samplingRate": query_case.get("samplingRate"),
        "age": query_case.get("age"),
        "gender": query_case.get("gender"),
        "time": query_case.get("time"),
        "feature_status": query_case.get("feature_status", "ok"),
        "missing_standard_leads": query_case.get("missing_standard_leads", []),
        "structured_features": _build_query_structured_features(query_case),
    }
    if include_query_summary and "ecg_summary" in query_case:
        compact_query["ecg_summary"] = query_case.get("ecg_summary", "")
    elif include_query_summary:
        compact_query["ecg_summary"] = _build_query_summary(query_case)
    if include_raw_signals:
        compact_query["selectedLeads"] = query_case.get("selectedLeads", {})

    compact_retrievals = []
    for case in retrieved_cases:
        item = {
            "id": case["id"],
            "similarity": case.get("similarity"),
            "samplingRate": case.get("samplingRate"),
            "feature_status": case.get("feature_status", "ok"),
            "missing_standard_leads": case.get("missing_standard_leads", []),
            "structured_features": _build_retrieval_structured_features(case),
        }
        if "ecg_summary" in case:
            item["ecg_summary"] = case.get("ecg_summary", "")
        else:
            item["ecg_summary"] = _build_retrieval_summary(case)
        if include_raw_signals:
            item["selectedLeads"] = case.get("selectedLeads", {})
        if include_retrieved_reports:
            item["report"] = case.get("report", "")
        if include_retrieved_labels:
            item["reference_labels"] = case.get("diagnosis", [])
        compact_retrievals.append(item)

    return {
        "task": (
            "Interpret the query ECG using the provided structured ECG evidence and return the required JSON."
            if prompt_version == "v1"
            else "Interpret the query ECG using the provided structured ECG evidence, explicitly auditing feature consistency before diagnosing, and return the required JSON."
        ),
        "query_case": compact_query,
        "retrieved_cases": compact_retrievals,
        "label_category_context": candidate_labels_by_category or {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES},
        "candidate_labels_by_category": candidate_labels_by_category or {category: [] for category in DEFAULT_DIAGNOSIS_CATEGORIES},
        "candidate_label_set": list(global_label_set),
    }


def build_chat_messages(
    query_case: dict[str, Any],
    retrieved_cases: list[dict[str, Any]],
    *,
    strict_json: bool,
    global_label_set: list[str],
    candidate_labels_by_category: dict[str, list[str]] | None = None,
    include_raw_signals: bool,
    include_query_summary: bool,
    include_retrieved_labels: bool,
    include_retrieved_reports: bool,
    prompt_version: str = "v1",
) -> list[dict[str, str]]:
    user_payload = build_user_payload(
        query_case,
        retrieved_cases,
        global_label_set=global_label_set,
        candidate_labels_by_category=candidate_labels_by_category,
        include_raw_signals=include_raw_signals,
        include_query_summary=include_query_summary,
        include_retrieved_labels=include_retrieved_labels,
        include_retrieved_reports=include_retrieved_reports,
        prompt_version=prompt_version,
    )
    return [
        {"role": "system", "content": build_system_prompt(strict_json=strict_json, prompt_version=prompt_version)},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
