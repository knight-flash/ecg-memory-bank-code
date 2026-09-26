from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SERVER_ROOT = "__ECG_DATA_ROOT_REQUIRED__/Final/LLM"
DEFAULT_SERVER_R1EVAL_DIR = f"{DEFAULT_SERVER_ROOT}/r1eval"
DEFAULT_SERVER_RUN_DIR = f"{DEFAULT_SERVER_ROOT}/outputs/ecg_grounding_full"
DEFAULT_SERVER_RETRY_DIR = (
    f"{DEFAULT_SERVER_RUN_DIR}/retries/retry_v2_both_c20"
)
DEFAULT_SERVER_BESTOF_DIR = (
    f"{DEFAULT_SERVER_RUN_DIR}/retries/retry_v2_both_c20_bestof"
)
DEFAULT_SERVER_GROUNDTRUTH_JSONL = f"{DEFAULT_SERVER_R1EVAL_DIR}/ecg-grounding-test-mimiciv_full.jsonl"
DEFAULT_SERVER_PROMPT_TEMPLATE = f"{DEFAULT_SERVER_R1EVAL_DIR}/prompts_evaluation_V2.txt"
DEFAULT_SERVER_EVAL_OUTPUT_ROOT = f"{DEFAULT_SERVER_ROOT}/outputs/evaluation_retrieval_neutral"
DEFAULT_SERVER_GENERATE_OUTPUT_DIR = f"{DEFAULT_SERVER_EVAL_OUTPUT_ROOT}/mimic_reasoning_eval_jobs"
DEFAULT_SERVER_SCORE_OUTPUT_DIR = f"{DEFAULT_SERVER_EVAL_OUTPUT_ROOT}/mimic_reasoning_eval_scores"
DEFAULT_SELECTION_RELATIVE_PATH = "reports/best_retry_selection_original_first.csv"

TARGET_CRITERIA = [
    "DiagnosisAccuracy",
    "AnalysisCompleteness",
    "AnalysisRelevance",
    "LeadEvidenceValidity",
    "RetrievalEvidenceUtilization",
    "GroundedECGUnderstanding",
    "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess",
]

CRITERION_ALIASES = {
    "DiagnosisAccuracy": "DiagnosisAccuracy",
    "AnalysisCompleteness": "AnalysisCompleteness",
    "AnalysisRelevance": "AnalysisRelevance",
    "LeadEvidenceValidity": "LeadEvidenceValidity",
    "RetrievalEvidenceUtilization": "RetrievalEvidenceUtilization",
    "GroundedECGUnderstanding": "GroundedECGUnderstanding",
    "ECGFeatureGrounding": "GroundedECGUnderstanding",
    "EvidenceBasedReasoning": "EvidenceBasedReasoning",
    "Evidence-BasedReasoning": "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess": "RealisticDiagnosticProcess",
    "ClinicalDiagnosticFidelity": "RealisticDiagnosticProcess",
}


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_selection_rows(path: str | Path, mode: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8")))
    if mode == "all":
        return rows
    return [row for row in rows if str(row.get("mode", "")).strip() == mode]


def load_ground_truth_map(path: str | Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path):
        case_id = str(row.get("id", "")).strip()
        if not case_id:
            continue
        messages = row.get("messages", [])
        clinician_text = ""
        if isinstance(messages, list) and messages:
            clinician_text = str(messages[-1].get("content", "") or "").strip()
        mapping[case_id] = {
            "groundtruth_text": clinician_text,
        }
    return mapping


def resolve_source_record_path(
    *,
    run_dir: str | Path,
    retry_dir: str | Path,
    mode: str,
    case_id: str,
    selected_from: str,
) -> Path:
    run_dir_path = Path(run_dir)
    retry_dir_path = Path(retry_dir)
    if selected_from == "base":
        return run_dir_path / "responses" / "parsed" / mode / f"{case_id}.json"
    if selected_from in {"original", "sparse"}:
        return retry_dir_path / "responses" / "parsed" / mode / selected_from / f"{case_id}.json"
    raise ValueError(f"Unsupported selected_from value: {selected_from}")


def render_prompt(template_text: str, generated_text: str, groundtruth_text: str) -> str:
    return (
        template_text.replace("{{generated}}", generated_text).replace("{{groundtruth}}", groundtruth_text)
    )


def format_generated_text(record: dict[str, Any]) -> tuple[str, str, bool]:
    reasoning_process = str(record.get("reasoning_process", "") or "").strip()
    raw_reasoning_content = str(record.get("raw_reasoning_content", "") or "").strip()
    predicted_labels = unique_preserve_order(list(record.get("predicted_labels", [])))

    used_fallback = False
    reasoning_body = raw_reasoning_content
    reasoning_source = "raw_reasoning_content"
    if not reasoning_body and reasoning_process:
        reasoning_body = reasoning_process
        reasoning_source = "reasoning_process"
        used_fallback = True

    labels_text = ", ".join(predicted_labels) if predicted_labels else "[]"
    generated_text = (
        f"Reasoning Process:\n{reasoning_body}\n\n"
        f"Final Predicted Labels: {labels_text}"
    )
    return generated_text, reasoning_source, used_fallback
