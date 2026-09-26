from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_BATCH_COUNT = 5
EXPECTED_PER_BATCH = 20
EXPECTED_TOTAL = 100
NEW_KEYS = ("glm5_rag", "glm5_zero")
LEGACY_TO_NEW = {
    "pulse": "glm5_rag",
    "gpt5": "glm5_zero",
}
MODEL_FIELDS = (
    "true_positive",
    "false_positive",
    "missed_diagnosis",
    "relevance",
    "accuracy",
    "completeness",
    "reasoning",
    "novelty",
    "clinical_value",
    "satisfaction",
    "comment",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_batch_id(path: Path) -> int | None:
    match = re.search(r"batch_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def normalize_case(case_key: str, case_payload: dict[str, Any], source_file: Path, batch_id: int | None) -> dict[str, Any]:
    sample_id = str(case_payload.get("id") or case_key)
    normalized = {
        "id": sample_id,
        "batch_id": batch_id,
        "source_file": str(source_file),
        "glm5_rag": {},
        "glm5_zero": {},
    }

    for raw_key, value in case_payload.items():
        if raw_key == "id":
            continue
        target_key = LEGACY_TO_NEW.get(raw_key, raw_key)
        if target_key in NEW_KEYS:
            normalized[target_key] = value or {}

    return normalized


def find_input_files(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in input_paths:
        if input_path.is_dir():
            files.extend(sorted(input_path.glob("*batch_*.json")))
        elif input_path.is_file():
            files.append(input_path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def aggregate(files: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    per_batch_counter: Counter[int] = Counter()
    source_files_used: list[str] = []

    for file_path in files:
        batch_id = detect_batch_id(file_path)
        payload = load_json(file_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path} should contain a top-level JSON object keyed by sample id")
        source_files_used.append(str(file_path))

        for case_key, case_payload in payload.items():
            if not isinstance(case_payload, dict):
                raise ValueError(f"{file_path} case {case_key} should be a JSON object")
            normalized = normalize_case(case_key, case_payload, file_path, batch_id)
            sample_id = normalized["id"]
            if sample_id in merged:
                duplicate_ids.append(sample_id)
                continue
            merged[sample_id] = normalized
            if batch_id is not None:
                per_batch_counter[batch_id] += 1

    summary = {
        "file_count": len(files),
        "source_files": source_files_used,
        "case_count": len(merged),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "per_batch_counts": {str(key): per_batch_counter[key] for key in sorted(per_batch_counter)},
    }
    return merged, summary


def validate(
    merged: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    duplicate_ids = summary.get("duplicate_ids", [])
    if duplicate_ids:
        raise ValueError(f"Duplicate sample ids detected: {', '.join(duplicate_ids)}")

    batch_counts = {int(key): value for key, value in summary.get("per_batch_counts", {}).items()}
    batch_ids_present = sorted(batch_counts)
    missing_batches = [idx for idx in range(EXPECTED_BATCH_COUNT) if idx not in batch_counts]

    if not allow_partial:
        if missing_batches:
            raise ValueError(f"Missing batch files or batch ids: {missing_batches}")
        wrong_batch_sizes = {key: value for key, value in batch_counts.items() if value != EXPECTED_PER_BATCH}
        if wrong_batch_sizes:
            raise ValueError(f"Unexpected batch sizes: {wrong_batch_sizes}; expected {EXPECTED_PER_BATCH} per batch")
        if len(merged) != EXPECTED_TOTAL:
            raise ValueError(f"Expected {EXPECTED_TOTAL} total cases, found {len(merged)}")

    model_completion = {key: 0 for key in NEW_KEYS}
    for case in merged.values():
        for model_key in NEW_KEYS:
            payload = case.get(model_key) or {}
            if isinstance(payload, dict) and payload:
                model_completion[model_key] += 1

    return {
        "batch_ids_present": batch_ids_present,
        "missing_batches": missing_batches,
        "model_completion_counts": model_completion,
        "is_complete": len(merged) == EXPECTED_TOTAL and not missing_batches and all(
            batch_counts.get(idx) == EXPECTED_PER_BATCH for idx in range(EXPECTED_BATCH_COUNT)
        ),
    }


def write_outputs(
    merged: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_payload = {
        "summary": {**summary, **validation},
        "cases": {sample_id: merged[sample_id] for sample_id in sorted(merged)},
    }
    (output_dir / "merged.json").write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        **summary,
        **validation,
        "expected_batch_count": EXPECTED_BATCH_COUNT,
        "expected_cases_per_batch": EXPECTED_PER_BATCH,
        "expected_total_cases": EXPECTED_TOTAL,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_columns = ["sample_id", "batch_id", "source_file"]
    for model_key in NEW_KEYS:
        for field in MODEL_FIELDS:
            csv_columns.append(f"{model_key}_{field}")

    with (output_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for sample_id in sorted(merged):
            case = merged[sample_id]
            row = {
                "sample_id": sample_id,
                "batch_id": case.get("batch_id"),
                "source_file": case.get("source_file"),
            }
            for model_key in NEW_KEYS:
                payload = case.get(model_key) or {}
                for field in MODEL_FIELDS:
                    row[f"{model_key}_{field}"] = payload.get(field, "") if isinstance(payload, dict) else ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate human GLM5 scoring JSON files with completeness checks.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parent / "evaluation_file_dixinxin"],
        help="Batch JSON files or a directory that contains them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_file_dixinxin" / "aggregated_glm5_scores",
        help="Directory to write merged.json, per_case.csv, and summary.json",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow incomplete inputs without enforcing the 5-batch / 100-case check.",
    )
    args = parser.parse_args()

    files = find_input_files(args.inputs)
    if not files:
        raise FileNotFoundError("No input JSON files found.")

    merged, summary = aggregate(files)
    validation = validate(merged, summary, allow_partial=args.allow_partial)
    write_outputs(merged, summary, validation, output_dir=args.output_dir)
    print(json.dumps({**summary, **validation, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
