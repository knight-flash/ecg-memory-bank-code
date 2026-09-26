from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

try:
    import json_repair  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    json_repair = None

SCRIPT_DIR = Path(__file__).resolve().parent

from ecg_memory_bank.reports.judge_io import (
    CRITERION_ALIASES,
    DEFAULT_SERVER_SCORE_OUTPUT_DIR,
    TARGET_CRITERIA,
)


EXPECTED_KEYS = sorted(CRITERION_ALIASES.keys(), key=len, reverse=True)
CRITERION_PATTERN = re.compile(
    r'"(?P<key>{})"\s*:\s*(?P<content>\[.*?\])'.format("|".join(re.escape(key) for key in EXPECTED_KEYS)),
    re.DOTALL,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Volcengine batch outputs for MIMIC reasoning_process evaluation.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SERVER_SCORE_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--eval-llm", type=str, default="")
    parser.add_argument("--qualitative-threshold", type=float, default=20.0)
    return parser.parse_args()


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl_to_dict(file_path: Path) -> dict[str, Any]:
    data_dict: dict[str, Any] = {}
    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("error"):
                continue
            record_id = str(item.get("custom_id", "")).strip()
            if not record_id:
                continue
            content = (
                item.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            data_dict[record_id] = content
    return data_dict


def _load_input_records(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_to_dict(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported input payload type in {path}")


def _manual_fix_string(text: str) -> str:
    text = re.sub(r'\[\s*"Score"\s*:', '[{"Score":', text)
    text = re.sub(r'("Score"\s*:\s*[-+]?\d+(\.\d+)?)\s*"Explanation"', r'\1, "Explanation"', text)
    text = re.sub(r'\}\s*"Score"', '}, {"Score"', text)
    return text


def _fix_unterminated_string(content: str) -> str:
    quote_count = len(re.findall(r'(?<!\\)"', content))
    if quote_count % 2 == 1:
        content = re.sub(r'(\s*[}\]])', r'"\1', content, count=1)
    return content


def _escape_inner_quotes_in_explanation(content: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        explanation = match.group(1)
        fixed = re.sub(r'(?<!\\)"', r'\\"', explanation)
        return f'"Explanation": "{fixed}"'

    return re.sub(r'"Explanation":\s*"([^"]*?)"', replacer, content)


def _fix_missing_commas(content: str) -> str:
    content = re.sub(r'(\})(\s*\{)', r'\1,\2', content)
    return content


def _clean_criterion_blob(content: str) -> str:
    content = _manual_fix_string(content)
    content = content.replace("\"", '"').replace("“", '"').replace("”", '"')
    content = re.sub(r'//.*', '', content)
    content = re.sub(r',\s*([}\]])', r'\1', content)
    content = re.sub(r'"\s*"', ' ', content)
    content = re.sub(r'\+(\d)', r'\1', content)
    content = content.replace('");', '"')
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n|\r', ' ', content)
    content = re.sub(r'"\s*[\]\)]', '"', content)
    content = re.sub(r'\}\s*\{', '},{', content)
    content = re.sub(r'("Explanation":\s*".*?)(?<!\\)"\s*,?\s*\{', r'\1"}, {', content)
    content = _fix_unterminated_string(content)
    content = _escape_inner_quotes_in_explanation(content)
    content = _fix_missing_commas(content)
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces > close_braces:
        content += "}" * (open_braces - close_braces)
    open_brackets = content.count("[")
    close_brackets = content.count("]")
    if open_brackets > close_brackets:
        content += "]" * (open_brackets - close_brackets)
    return content


def _safe_json_loads(content: str) -> Any:
    if json_repair is not None:
        return json_repair.loads(content)
    return json.loads(content)


def _extract_json_object_from_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    try:
        payload = _safe_json_loads(stripped)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidate = stripped[first : last + 1]
        try:
            payload = _safe_json_loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
    return {}


def _extract_criterion_payload(content: Any) -> dict[str, dict[str, Any]]:
    if isinstance(content, dict):
        raw_payload = content
    else:
        text = str(content or "").strip()
        raw_payload = _extract_json_object_from_text(text)
        if not raw_payload:
            raw_payload = {}
            for match in CRITERION_PATTERN.finditer(text):
                raw_payload[match.group("key")] = match.group("content")

    parsed: dict[str, dict[str, Any]] = {
        criterion: {"Scores": [], "Explanations": []} for criterion in TARGET_CRITERIA
    }
    for raw_key, raw_value in raw_payload.items():
        canonical_key = CRITERION_ALIASES.get(raw_key)
        if canonical_key not in parsed:
            continue
        items: list[dict[str, Any]] = []
        if isinstance(raw_value, list):
            items = [item for item in raw_value if isinstance(item, dict)]
        elif isinstance(raw_value, str):
            cleaned = _clean_criterion_blob(raw_value)
            try:
                repaired = _safe_json_loads(cleaned)
            except json.JSONDecodeError:
                repaired = []
            if isinstance(repaired, list):
                items = [item for item in repaired if isinstance(item, dict)]
        elif isinstance(raw_value, dict):
            items = [raw_value]

        scores: list[float] = []
        explanations: list[str] = []
        for item in items:
            score = item.get("Score", item.get(" score", item.get(" Score")))
            if score is None:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                match = re.search(r"(\d+(\.\d+)?)", str(score))
                if not match:
                    continue
                numeric_score = float(match.group(1))
            explanation = str(item.get("Explanation", "") or "").strip()
            extra_fields = {k: v for k, v in item.items() if k not in {"Score", " score", " Score", "Explanation"}}
            if extra_fields:
                extra_json = json.dumps(extra_fields, ensure_ascii=False)
                explanation = f"{explanation} Additional details: {extra_json}".strip()
            scores.append(numeric_score)
            explanations.append(explanation)
        parsed[canonical_key] = {
            "Scores": scores,
            "Explanations": explanations,
        }
    return parsed


def _criterion_value(criterion: str, scores: list[float]) -> float:
    if not scores:
        return 0.0
    if criterion == "DiagnosisAccuracy":
        return round(100.0 * (len([score for score in scores if score > 0]) / len(scores)), 4)
    return round(sum(scores), 4)


def _split_custom_id(custom_id: str) -> tuple[str, str]:
    mode, _, case_id = str(custom_id).partition("_")
    return mode or "unknown", case_id or str(custom_id)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_name = args.name or input_path.stem
    eval_llm = args.eval_llm or "unknown"
    raw_records = _load_input_records(input_path)

    clean_payload: dict[str, Any] = {}
    per_case_rows: list[dict[str, Any]] = []
    grouped_rows: dict[str, list[dict[str, Any]]] = {"rag": [], "zero": [], "cot": []}

    for custom_id, raw_content in raw_records.items():
        criterion_payload = _extract_criterion_payload(raw_content)
        clean_payload[custom_id] = criterion_payload
        mode, case_id = _split_custom_id(custom_id)
        row: dict[str, Any] = {
            "custom_id": custom_id,
            "mode": mode,
            "case_id": case_id,
        }
        explanation_row: dict[str, Any] = {
            "custom_id": custom_id,
            "mode": mode,
            "case_id": case_id,
        }
        for criterion in TARGET_CRITERIA:
            scores = criterion_payload.get(criterion, {}).get("Scores", [])
            explanations = criterion_payload.get(criterion, {}).get("Explanations", [])
            if criterion == "RetrievalEvidenceUtilization" and mode == "zero":
                row[criterion] = None
            else:
                row[criterion] = _criterion_value(criterion, scores)
            explanation_row[f"{criterion}_explanations"] = " || ".join(explanations)
        row["Average"] = round(
            (
                row["GroundedECGUnderstanding"]
                + row["EvidenceBasedReasoning"]
                + row["RealisticDiagnosticProcess"]
            )
            / 3.0,
            4,
        )
        row.update(explanation_row)
        per_case_rows.append(row)
        if mode in grouped_rows:
            grouped_rows[mode].append(row)

    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_json_path = clean_dir / f"{input_name}.json"
    with clean_json_path.open("w", encoding="utf-8") as handle:
        json.dump(clean_payload, handle, ensure_ascii=False, indent=2)

    score_fields = ["custom_id", "mode", "case_id", *TARGET_CRITERIA, "Average"]
    explanation_fields = [f"{criterion}_explanations" for criterion in TARGET_CRITERIA]
    per_case_fields = score_fields + explanation_fields
    per_case_rows_sorted = sorted(per_case_rows, key=lambda row: (row["mode"], row["case_id"]))
    _write_csv(output_dir / "per_case_scores.csv", per_case_rows_sorted, per_case_fields)

    mode_summary_rows: list[dict[str, Any]] = []
    summary_payload: dict[str, Any] = {
        "name": input_name,
        "eval_llm": eval_llm,
        "input_file": str(input_path),
        "clean_json": str(clean_json_path),
        "score_scale": "legacy_compute_score_volengine_batch_with_retrieval_na_for_zero",
        "mode_summaries": {},
        "rag_zero_delta": {},
        "retrieval_utilization_summary": {},
    }
    for mode in ("rag", "zero", "cot"):
        rows = grouped_rows[mode]
        summary_row: dict[str, Any] = {
            "name": input_name,
            "eval_llm": eval_llm,
            "mode": mode,
            "case_count": len(rows),
        }
        mode_summary: dict[str, Any] = {
            "case_count": len(rows),
        }
        for criterion in TARGET_CRITERIA + ["Average"]:
            values = [
                float(row[criterion])
                for row in rows
                if row.get(criterion) not in (None, "")
            ]
            mean_value = round(sum(values) / len(values), 4) if values else None
            summary_row[criterion] = mean_value
            mode_summary[criterion] = mean_value
        mode_summary_rows.append(summary_row)
        summary_payload["mode_summaries"][mode] = mode_summary

    if summary_payload["mode_summaries"].get("rag") and summary_payload["mode_summaries"].get("zero"):
        for criterion in TARGET_CRITERIA + ["Average"]:
            if criterion == "RetrievalEvidenceUtilization":
                summary_payload["rag_zero_delta"][criterion] = None
                continue
            rag_raw = summary_payload["mode_summaries"]["rag"].get(criterion, 0.0)
            zero_raw = summary_payload["mode_summaries"]["zero"].get(criterion, 0.0)
            rag_value = float(rag_raw) if rag_raw is not None else 0.0
            zero_value = float(zero_raw) if zero_raw is not None else 0.0
            summary_payload["rag_zero_delta"][criterion] = round(rag_value - zero_value, 4)

    summary_payload["retrieval_utilization_summary"] = {
        "rag_mean": summary_payload["mode_summaries"].get("rag", {}).get("RetrievalEvidenceUtilization"),
        "rag_case_count": len([row for row in grouped_rows["rag"] if row.get("RetrievalEvidenceUtilization") is not None]),
        "zero_mean": None,
        "zero_case_count": 0,
        "note": "RetrievalEvidenceUtilization is reported for rag only; zero is treated as NA.",
    }

    _write_csv(
        output_dir / "mode_summary.csv",
        mode_summary_rows,
        ["name", "eval_llm", "mode", "case_count", *TARGET_CRITERIA, "Average"],
    )

    rag_by_case = {row["case_id"]: row for row in grouped_rows["rag"]}
    zero_by_case = {row["case_id"]: row for row in grouped_rows["zero"]}
    qualitative_rows: list[dict[str, Any]] = []
    overlap_case_ids = sorted(set(rag_by_case) & set(zero_by_case))
    for case_id in overlap_case_ids:
        rag_row = rag_by_case[case_id]
        zero_row = zero_by_case[case_id]
        grounded_delta = round(rag_row["GroundedECGUnderstanding"] - zero_row["GroundedECGUnderstanding"], 4)
        realistic_delta = round(rag_row["RealisticDiagnosticProcess"] - zero_row["RealisticDiagnosticProcess"], 4)
        if grounded_delta < args.qualitative_threshold and realistic_delta < args.qualitative_threshold:
            continue
        qualitative_rows.append(
            {
                "case_id": case_id,
                "GroundedECGUnderstanding_rag": rag_row["GroundedECGUnderstanding"],
                "GroundedECGUnderstanding_zero": zero_row["GroundedECGUnderstanding"],
                "GroundedECGUnderstanding_delta": grounded_delta,
                "RealisticDiagnosticProcess_rag": rag_row["RealisticDiagnosticProcess"],
                "RealisticDiagnosticProcess_zero": zero_row["RealisticDiagnosticProcess"],
                "RealisticDiagnosticProcess_delta": realistic_delta,
                "GroundedECGUnderstanding_rag_explanations": rag_row["GroundedECGUnderstanding_explanations"],
                "GroundedECGUnderstanding_zero_explanations": zero_row["GroundedECGUnderstanding_explanations"],
                "RealisticDiagnosticProcess_rag_explanations": rag_row["RealisticDiagnosticProcess_explanations"],
                "RealisticDiagnosticProcess_zero_explanations": zero_row["RealisticDiagnosticProcess_explanations"],
            }
        )
    qualitative_rows.sort(
        key=lambda row: (
            row["GroundedECGUnderstanding_delta"] + row["RealisticDiagnosticProcess_delta"],
            row["case_id"],
        ),
        reverse=True,
    )
    _write_csv(
        output_dir / "qualitative_cases.csv",
        qualitative_rows,
        [
            "case_id",
            "GroundedECGUnderstanding_rag",
            "GroundedECGUnderstanding_zero",
            "GroundedECGUnderstanding_delta",
            "RealisticDiagnosticProcess_rag",
            "RealisticDiagnosticProcess_zero",
            "RealisticDiagnosticProcess_delta",
            "GroundedECGUnderstanding_rag_explanations",
            "GroundedECGUnderstanding_zero_explanations",
            "RealisticDiagnosticProcess_rag_explanations",
            "RealisticDiagnosticProcess_zero_explanations",
        ],
    )

    summary_payload["qualitative_case_count"] = len(qualitative_rows)
    summary_payload["qualitative_threshold"] = args.qualitative_threshold
    summary_json_path = output_dir / "summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
