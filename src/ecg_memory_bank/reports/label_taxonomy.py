"""Diagnostic candidate taxonomy.

Purpose
-------
Build the nine-category candidate vocabulary used in structured ECG report
generation and scoring.

Method
------
Normalize labels, count diagnoses from external metadata, combine manual
classification and overrides, then emit canonical candidate sets.

Inputs
------
Corrected metadata CSV, external label-classification spreadsheet, and static
override mapping.

Outputs
-------
Mapped/unmapped label tables and candidate/category dictionaries.

Paper parameters
----------------
The formal ECG-Grounding benchmark supplies 102 candidate labels; generated
frequency counts are recomputed rather than shipped.

Shapes and meaning
------------------
A case may carry multiple labels; category maps determine the slots in
diagnoses_by_category and columns of [Q,C] score matrices.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def split_label_text(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]

    return [text]


def normalize_label(label: str) -> str:
    text = str(label or "").strip().strip('"').strip("'")
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;:[]{}")
    return text.lower().strip()


def is_noise_label(normalized_label: str, stop_labels: tuple[str, ...]) -> bool:
    if not normalized_label:
        return True
    if normalized_label in {normalize_label(item) for item in stop_labels}:
        return True
    if re.fullmatch(r"[\W_]+", normalized_label):
        return True
    return False


def load_label_counts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise KeyError(f"Missing 'label' column in {path}")
    if "count" not in df.columns:
        df["count"] = 0
    df["label"] = df["label"].astype(str)
    df["normalized_label"] = df["label"].map(normalize_label)
    return df


def build_vocab_counts_from_metadata(path: str | Path, *, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise KeyError(f"Missing label column '{label_column}' in {path}")
    counts: dict[str, int] = {}
    for raw in df[label_column].tolist():
        for label in split_label_text(raw):
            counts[label] = counts.get(label, 0) + 1
    rows = [{"label": label, "count": count, "normalized_label": normalize_label(label)} for label, count in counts.items()]
    return pd.DataFrame(rows).sort_values(by=["count", "label"], ascending=[False, True]).reset_index(drop=True)


def load_reference_classification(path: str | Path) -> pd.DataFrame:
    excel = pd.ExcelFile(path)
    sheet_name = excel.sheet_names[0]
    df = excel.parse(sheet_name)
    rename_map = {col: str(col).strip() for col in df.columns}
    df = df.rename(columns=rename_map)
    for field in ("Classification", "Sub-group"):
        if field in df.columns:
            df[field] = df[field].ffill()
    labels_col = "Labels"
    if labels_col not in df.columns:
        raise KeyError(f"Missing 'Labels' column in {path}")
    df = df[df[labels_col].notna()].copy()
    df["reference_label"] = df[labels_col].astype(str).str.strip()
    df["normalized_label"] = df["reference_label"].map(normalize_label)
    return df[["Classification", "Sub-group", "reference_label", "normalized_label"]].reset_index(drop=True)


def load_label_overrides(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame(columns=["raw_label", "normalized_label", "Classification", "Sub-group", "mapped_label"])
    df = pd.read_csv(file_path)
    if "raw_label" not in df.columns:
        raise KeyError(f"Override file must contain 'raw_label': {path}")
    df["normalized_label"] = df["raw_label"].astype(str).map(normalize_label)
    if "mapped_label" not in df.columns:
        df["mapped_label"] = df["raw_label"].astype(str)
    if "Classification" not in df.columns:
        df["Classification"] = "Others"
    if "Sub-group" not in df.columns:
        df["Sub-group"] = ""
    return df[["raw_label", "normalized_label", "Classification", "Sub-group", "mapped_label"]]


def apply_label_mapping(
    vocab_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    *,
    stop_labels: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop_set = tuple(stop_labels)
    ref_map = {
        str(row["normalized_label"]): {
            "Classification": row["Classification"],
            "Sub-group": row["Sub-group"],
            "mapped_label": row["reference_label"],
            "mapping_source": "reference_exact",
        }
        for _, row in reference_df.iterrows()
        if str(row["normalized_label"]).strip()
    }
    override_map = {
        str(row["normalized_label"]): {
            "Classification": row["Classification"],
            "Sub-group": row["Sub-group"],
            "mapped_label": row["mapped_label"],
            "mapping_source": "override",
        }
        for _, row in overrides_df.iterrows()
        if str(row["normalized_label"]).strip()
    }

    mapped_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    for _, row in vocab_df.iterrows():
        raw_label = str(row["label"])
        normalized = str(row["normalized_label"])
        count = int(row.get("count", 0))
        if is_noise_label(normalized, stop_set):
            unmatched_rows.append(
                {
                    "label": raw_label,
                    "normalized_label": normalized,
                    "count": count,
                    "match_status": "filtered_noise",
                }
            )
            continue

        match = override_map.get(normalized) or ref_map.get(normalized)
        if match:
            mapped_rows.append(
                {
                    "label": raw_label,
                    "normalized_label": normalized,
                    "count": count,
                    **match,
                }
            )
        else:
            unmatched_rows.append(
                {
                    "label": raw_label,
                    "normalized_label": normalized,
                    "count": count,
                    "match_status": "unmatched",
                }
            )

    mapped_df = pd.DataFrame(mapped_rows).sort_values(by=["Classification", "Sub-group", "count", "label"], ascending=[True, True, False, True]).reset_index(drop=True) if mapped_rows else pd.DataFrame(columns=["label", "normalized_label", "count", "Classification", "Sub-group", "mapped_label", "mapping_source"])
    unmatched_df = pd.DataFrame(unmatched_rows).sort_values(by=["match_status", "count", "label"], ascending=[True, False, True]).reset_index(drop=True) if unmatched_rows else pd.DataFrame(columns=["label", "normalized_label", "count", "match_status"])
    return mapped_df, unmatched_df


def build_candidate_labels_by_category(mapped_df: pd.DataFrame, *, categories: tuple[str, ...]) -> dict[str, list[str]]:
    category_map = {category: [] for category in categories}
    if mapped_df.empty:
        return category_map
    for _, row in mapped_df.iterrows():
        category = str(row["Classification"]).strip() or "Others"
        if category not in category_map:
            category = "Others"
        mapped_label = str(row["mapped_label"]).strip()
        if mapped_label and mapped_label not in category_map[category]:
            category_map[category].append(mapped_label)
    for category in category_map:
        category_map[category] = sorted(category_map[category])
    return category_map


def build_label_to_category_map(mapped_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if mapped_df.empty:
        return mapping
    allowed_categories = {
        "Rhythm",
        "Conduction",
        "Ischemia events",
        "Hypertrophy",
        "Pacemaker",
        "General report",
        "ST-T changes",
        "Axis",
        "Others",
    }
    for _, row in mapped_df.iterrows():
        mapped_label = str(row["mapped_label"]).strip()
        if not mapped_label:
            continue
        category = str(row["Classification"]).strip() or "Others"
        if category not in allowed_categories:
            category = "Others"
        normalized = str(row["normalized_label"]).strip()
        if not normalized:
            continue
        mapping[normalized] = {
            "Classification": category,
            "Sub-group": str(row["Sub-group"]).strip(),
            "normalized_label": normalized,
            "mapped_label": mapped_label,
        }
    return mapping


def write_label_context(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
