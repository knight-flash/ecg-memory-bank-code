"""ECG-Grounding cohort matching.

Purpose
-------
Match released ECG-Grounding cases to MIMIC metadata and feature records
before report evaluation.

Method
------
Normalize case/study IDs, deduplicate benchmark entries, verify matching
metadata and structured features, then write a query selector.

Inputs
------
Released grounding JSON, corrected MIMIC metadata CSV, and external feature
root.

Outputs
-------
External query-subset CSV plus a selection/audit report.

Paper parameters
----------------
The formal runner requires all 2381 benchmark cases; silently dropping
unmatched cases is not accepted.

Shapes and meaning
------------------
Each subset row is one query ECG; subject_id later excludes all records from
the same subject from active memory.
"""

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

from ecg_memory_bank.reports.batch_io import write_json
from ecg_memory_bank.reports.config import default_report_config


DEFAULT_INPUT_JSON = Path(__file__).resolve().parent / "ecg-grounding-test-mimiciv.json"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "grounding_query_subset.csv"
DEFAULT_REPORT_JSON = Path(__file__).resolve().parent / "grounding_query_subset_report.json"


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(
        description=(
            "Build a query subset CSV for the standard MIMIC grounding set by matching JSON ids "
            "to current MIMIC metadata study_id values and checking feature availability."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--metadata-csv", type=Path, default=Path(config.dataset.corrected_label_metadata_path))
    parser.add_argument("--feature-root", type=Path, default=Path(config.dataset.feature_root))
    parser.add_argument("--key-column", type=str, default=config.dataset.feature_match_key)
    return parser.parse_args()


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if text.endswith(".0"):
        integer_part = text[:-2]
        if integer_part.isdigit():
            return integer_part
    return text


def _load_grounding_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Input JSON must be a list of objects: {path}")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"Input JSON row {idx} is not an object")
        rows.append(item)
    return rows


def _load_metadata_index(path: Path, key_column: str) -> tuple[dict[str, dict[str, str]], int]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    index: dict[str, dict[str, str]] = {}
    duplicate_keys = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain the column: {key_column}")
        for row in reader:
            key = _normalize_id(row.get(key_column))
            if not key:
                continue
            if key in index:
                duplicate_keys += 1
                continue
            index[key] = {str(col): str(value or "") for col, value in row.items()}
    return index, duplicate_keys


def _extract_unique_grounding_entries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {
        "input_rows": len(rows),
        "unique_ids": 0,
        "duplicate_ids": 0,
        "invalid_ids": 0,
    }
    for idx, row in enumerate(rows):
        study_id = _normalize_id(row.get("id"))
        if not study_id:
            stats["invalid_ids"] += 1
            continue
        if study_id in seen:
            stats["duplicate_ids"] += 1
            continue
        seen.add(study_id)
        unique_rows.append(
            {
                "study_id": study_id,
                "json_index": idx,
                "json_ecg": str(row.get("ecg", "") or ""),
                "json_image": str(row.get("image", "") or ""),
                "json_report": str(row.get("report", "") or ""),
                "source_dataset": str(row.get("source", "") or "MIMIC-IV-ECG"),
            }
        )
    stats["unique_ids"] = len(unique_rows)
    return unique_rows, stats


def _build_output_rows(
    unique_rows: list[dict[str, Any]],
    metadata_index: dict[str, dict[str, str]],
    *,
    input_json: Path,
    feature_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output_rows: list[dict[str, str]] = []
    missing_metadata_ids: list[str] = []
    missing_feature_ids: list[str] = []
    kept_ids: list[str] = []

    for item in unique_rows:
        study_id = item["study_id"]
        if study_id not in metadata_index:
            missing_metadata_ids.append(study_id)
            continue
        feature_path = feature_root / f"{study_id}.npy"
        if not feature_path.exists():
            missing_feature_ids.append(study_id)
            continue

        kept_ids.append(study_id)
        output_rows.append(
            {
                "study_id": study_id,
                "source_dataset": item["source_dataset"],
                "source_json_path": str(input_json),
                "json_index": str(item["json_index"]),
                "json_ecg": item["json_ecg"],
                "json_image": item["json_image"],
                "json_report": item["json_report"],
            }
        )

    report = {
        "usable_ids": kept_ids,
        "missing_metadata_ids": missing_metadata_ids,
        "missing_feature_ids": missing_feature_ids,
    }
    return output_rows, report


def _write_subset_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id",
        "source_dataset",
        "source_json_path",
        "json_index",
        "json_ecg",
        "json_image",
        "json_report",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.feature_root.exists():
        raise FileNotFoundError(f"Feature root not found: {args.feature_root}")

    grounding_rows = _load_grounding_rows(args.input_json)
    unique_rows, stats = _extract_unique_grounding_entries(grounding_rows)
    metadata_index, metadata_duplicate_keys = _load_metadata_index(args.metadata_csv, args.key_column)
    output_rows, report_lists = _build_output_rows(
        unique_rows,
        metadata_index,
        input_json=args.input_json,
        feature_root=args.feature_root,
    )
    _write_subset_csv(args.output_csv, output_rows)

    report = {
        "input_json": str(args.input_json),
        "output_csv": str(args.output_csv),
        "metadata_csv": str(args.metadata_csv),
        "feature_root": str(args.feature_root),
        "key_column": args.key_column,
        **stats,
        "metadata_unique_keys": len(metadata_index),
        "metadata_duplicate_keys_skipped": metadata_duplicate_keys,
        "missing_metadata_count": len(report_lists["missing_metadata_ids"]),
        "missing_feature_count": len(report_lists["missing_feature_ids"]),
        "usable_count": len(report_lists["usable_ids"]),
        "missing_metadata_ids": report_lists["missing_metadata_ids"],
        "missing_feature_ids": report_lists["missing_feature_ids"],
        "usable_ids": report_lists["usable_ids"],
    }
    write_json(args.report_json, report)

    print(
        json.dumps(
            {
                "input_rows": report["input_rows"],
                "unique_ids": report["unique_ids"],
                "duplicate_ids": report["duplicate_ids"],
                "invalid_ids": report["invalid_ids"],
                "missing_metadata_count": report["missing_metadata_count"],
                "missing_feature_count": report["missing_feature_count"],
                "usable_count": report["usable_count"],
                "output_csv": str(args.output_csv),
                "report_json": str(args.report_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
