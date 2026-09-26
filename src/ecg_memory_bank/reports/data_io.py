"""Report experiment data alignment.

Purpose
-------
Load ECG embeddings, MIMIC metadata, labels, reports, and export metadata for
report generation.

Method
------
Parse heterogeneous embedding payloads, join metadata by available
identifiers, and build record-level label/text fields.

Inputs
------
External .pt/.npy embeddings and CSV metadata with study/subject identifiers
and report-derived labels.

Outputs
-------
Aligned arrays and DataFrames passed to retrieval, prompt building, and
evaluation.

Paper parameters
----------------
Caller chooses label column and split key; no inference hyperparameters are
set here.

Shapes and meaning
------------------
Embedding matrices are [N,D], metadata has N rows, and each row is one ECG
record with optional multilabel diagnoses.
"""

from __future__ import annotations

import ast
import hashlib
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

DEFAULT_PATH_FIELDS = ("full_ecg_path", "ecg_path", "path", "record_path")


def load_embeddings(path: str | Path) -> torch.Tensor:
    start_time = time.time()
    print(f"[Load] embeddings <- {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        embeddings = payload.float()
    elif isinstance(payload, dict):
        for key in ("embeddings", "ecg_embeddings", "features", "x"):
            if key in payload and isinstance(payload[key], torch.Tensor):
                embeddings = payload[key].float()
                break
        else:
            raise TypeError(f"Unsupported embedding payload at {path}")
    else:
        raise TypeError(f"Unsupported embedding payload at {path}")

    print(
        f"[Load] embeddings ready: shape={tuple(embeddings.shape)} "
        f"elapsed={time.time() - start_time:.2f}s"
    )
    return embeddings


def load_metadata(path: str | Path) -> pd.DataFrame:
    start_time = time.time()
    print(f"[Load] metadata <- {path}")
    metadata = pd.read_csv(path).reset_index(drop=True)
    print(
        f"[Load] metadata ready: rows={len(metadata)} cols={len(metadata.columns)} "
        f"elapsed={time.time() - start_time:.2f}s"
    )
    return metadata


def attach_export_metadata(base_df: pd.DataFrame, export_meta_path: str | Path | None) -> pd.DataFrame:
    if not export_meta_path:
        print("[Merge] no export metadata path provided, skip path enrichment")
        return base_df.reset_index(drop=True)

    export_meta_path = Path(export_meta_path)
    if not export_meta_path.exists():
        print(f"[Merge] export metadata not found, skip path enrichment: {export_meta_path}")
        return base_df.reset_index(drop=True)

    print(f"[Merge] export metadata <- {export_meta_path}")
    export_df = pd.read_csv(export_meta_path)
    if "HashFileName" not in export_df.columns:
        print(f"[Merge] export metadata missing HashFileName, skip: {export_meta_path}")
        return base_df.reset_index(drop=True)

    if "HashFileName" not in base_df.columns:
        print("[Merge] base metadata missing HashFileName, skip export enrichment")
        return base_df.reset_index(drop=True)

    export_df = export_df.drop_duplicates(subset=["HashFileName"])
    merged_df = base_df.merge(
        export_df,
        on="HashFileName",
        how="left",
        suffixes=("", "_export"),
    )

    merged_fields = 0
    for field_name in DEFAULT_PATH_FIELDS:
        export_field_name = f"{field_name}_export"
        if export_field_name not in merged_df.columns:
            continue

        if field_name not in merged_df.columns:
            merged_df[field_name] = merged_df[export_field_name]
            merged_fields += 1
            continue

        primary_series = merged_df[field_name]
        fallback_series = merged_df[export_field_name]
        primary_missing = primary_series.isna() | primary_series.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        merged_df.loc[primary_missing, field_name] = fallback_series.loc[primary_missing]
        merged_fields += 1

    print(f"[Merge] export enrichment done: rows={len(merged_df)} path_fields_considered={merged_fields}")
    return merged_df.reset_index(drop=True)


def attach_corrected_labels(
    base_df: pd.DataFrame,
    corrected_meta_path: str | Path | None,
    *,
    label_column: str = "labels",
    key_column: str = "HashFileName",
) -> pd.DataFrame:
    if not corrected_meta_path:
        print("[Merge] no corrected label metadata path provided, skip label correction")
        return base_df.reset_index(drop=True)

    corrected_meta_path = Path(corrected_meta_path)
    if not corrected_meta_path.exists():
        print(f"[Merge] corrected label metadata not found, skip: {corrected_meta_path}")
        return base_df.reset_index(drop=True)

    if key_column not in base_df.columns:
        print(f"[Merge] base metadata missing {key_column}, skip corrected label merge")
        return base_df.reset_index(drop=True)

    print(f"[Merge] corrected labels <- {corrected_meta_path}")
    corrected_df = pd.read_csv(corrected_meta_path)
    if key_column not in corrected_df.columns or label_column not in corrected_df.columns:
        print(f"[Merge] corrected label metadata missing required columns: {key_column}, {label_column}")
        return base_df.reset_index(drop=True)

    corrected_df = corrected_df[[key_column, label_column]].drop_duplicates(subset=[key_column])
    merged_df = base_df.merge(
        corrected_df.rename(columns={label_column: f"{label_column}_corrected"}),
        on=key_column,
        how="left",
    )

    corrected_series = merged_df.get(f"{label_column}_corrected")
    if corrected_series is None:
        return merged_df.reset_index(drop=True)

    corrected_mask = ~(
        corrected_series.isna()
        | corrected_series.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
    )
    updated_rows = int(corrected_mask.sum())
    if label_column not in merged_df.columns:
        merged_df[label_column] = corrected_series
    else:
        merged_df.loc[corrected_mask, label_column] = corrected_series.loc[corrected_mask]

    print(f"[Merge] corrected labels applied: updated_rows={updated_rows}")
    return merged_df.drop(columns=[f"{label_column}_corrected"]).reset_index(drop=True)


def align_embeddings_and_metadata(embeddings: torch.Tensor, metadata: pd.DataFrame) -> tuple[torch.Tensor, pd.DataFrame]:
    if len(embeddings) == len(metadata):
        print(f"[Align] embeddings and metadata already aligned at {len(metadata)} rows")
        return embeddings, metadata.copy().reset_index(drop=True)
    aligned_len = min(len(embeddings), len(metadata))
    print(
        f"[Align] row mismatch: embeddings={len(embeddings)} metadata={len(metadata)} "
        f"-> truncating to {aligned_len}"
    )
    return embeddings[:aligned_len], metadata.iloc[:aligned_len].copy().reset_index(drop=True)


def _split_labels(raw_value: object) -> list[str]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []
    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    parsed: object | None = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
    if isinstance(parsed, list):
        values = parsed
    elif "|" in text:
        values = text.split("|")
    else:
        values = [text]

    clean: list[str] = []
    for item in values:
        label = str(item).strip().strip("'").strip('"')
        if label and label.lower() not in {"nan", "none"} and label not in clean:
            clean.append(label)
    return clean


def parse_labels_column(series: Iterable[object]) -> list[list[str]]:
    return [_split_labels(value) for value in series]


def first_non_empty_text(row: pd.Series, candidates: tuple[str, ...]) -> str:
    for field_name in candidates:
        if field_name not in row.index:
            continue
        value = row[field_name]
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def build_case_id(row: pd.Series, prefix: str, row_index: int) -> str:
    for field_name in ("case_id", "id", "HashFileName", "study_id", "exam_id"):
        if field_name in row.index:
            value = row[field_name]
            if pd.notna(value):
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none"}:
                    return text

    if "full_ecg_path" in row.index:
        path_text = str(row["full_ecg_path"]).strip()
        if path_text and path_text.lower() not in {"nan", "none"}:
            base_name = Path(path_text).stem
            if base_name:
                return base_name
            digest = hashlib.md5(path_text.encode("utf-8")).hexdigest()[:12]
            return f"{prefix}_{digest}"

    return f"{prefix}_{row_index:06d}"


def extract_patient_id(case_like: object) -> str:
    text = str(case_like or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    match = re.match(r"^de_([^_]+)_", text)
    if match:
        return match.group(1)
    return ""


def standardize_dataset(
    metadata: pd.DataFrame,
    *,
    split_name: str,
    label_column: str,
    path_column: str,
    text_candidates: tuple[str, ...],
) -> pd.DataFrame:
    start_time = time.time()
    print(f"[Dataset] standardizing split={split_name}")
    df = metadata.copy().reset_index(drop=True)
    if label_column not in df.columns:
        raise KeyError(f"Missing label column '{label_column}'")
    if path_column not in df.columns:
        raise KeyError(f"Missing path column '{path_column}'")

    labels_list = parse_labels_column(df[label_column].tolist())
    reports = [first_non_empty_text(df.iloc[idx], text_candidates) for idx in range(len(df))]
    case_ids = [build_case_id(df.iloc[idx], split_name, idx) for idx in range(len(df))]
    if "subject_id" in df.columns:
        patient_ids = [str(value).strip() if pd.notna(value) else "" for value in df["subject_id"].tolist()]
    else:
        patient_ids = [extract_patient_id(case_id) for case_id in case_ids]
    primary_labels = [labels[0] if labels else "Unlabeled" for labels in labels_list]

    df["case_id"] = case_ids
    df["patient_id"] = patient_ids
    df["source_split"] = split_name
    df["row_index"] = np.arange(len(df))
    df["labels_list"] = labels_list
    df["labels_text"] = [" | ".join(labels) for labels in labels_list]
    df["primary_label"] = primary_labels
    df["report_text"] = reports
    df["full_ecg_path"] = df[path_column].astype(str)
    print(
        f"[Dataset] split={split_name} ready: rows={len(df)} "
        f"unique_primary_labels={df['primary_label'].nunique()} "
        f"elapsed={time.time() - start_time:.2f}s"
    )
    return df


def balanced_subset_indices(df: pd.DataFrame, sample_size: int, seed: int) -> list[int]:
    if len(df) <= sample_size:
        print(f"[Sample] split size {len(df)} <= requested {sample_size}, using full set")
        return list(range(len(df)))

    rng = np.random.default_rng(seed)
    grouped = {}
    for idx, primary_label in enumerate(df["primary_label"].tolist()):
        grouped.setdefault(primary_label, []).append(idx)

    selected: list[int] = []
    labels = list(grouped.keys())
    rng.shuffle(labels)
    while len(selected) < sample_size:
        advanced = False
        for label in labels:
            candidates = grouped[label]
            if not candidates:
                continue
            selected.append(candidates.pop(0))
            advanced = True
            if len(selected) >= sample_size:
                break
        if not advanced:
            break

    if len(selected) < sample_size:
        selected_set = set(selected)
        remaining = [idx for idx in range(len(df)) if idx not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    selected = sorted(selected[:sample_size])
    print(
        f"[Sample] requested={sample_size} selected={len(selected)} "
        f"unique_primary_labels={df.iloc[selected]['primary_label'].nunique()}"
    )
    return selected


def subset_embeddings(embeddings: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return embeddings[indices].clone()


def build_standardized_subset_from_csv(
    subset_csv_path: str | Path,
    base_df: pd.DataFrame,
    *,
    split_name: str,
    label_column: str,
    path_column: str,
    text_candidates: tuple[str, ...],
) -> pd.DataFrame:
    subset_csv_path = Path(subset_csv_path)
    if not subset_csv_path.exists():
        raise FileNotFoundError(f"Missing subset csv: {subset_csv_path}")

    subset_df = pd.read_csv(subset_csv_path)
    required = {"HashFileName", path_column}
    missing = [column for column in required if column not in subset_df.columns]
    if missing:
        raise KeyError(f"Subset CSV missing required columns: {missing}")

    if "HashFileName" not in base_df.columns:
        raise KeyError("Base metadata missing HashFileName, cannot align external subset csv")

    subset_df = subset_df.copy()
    subset_df["HashFileName"] = subset_df["HashFileName"].astype(str).str.strip()
    base_df = base_df.copy()
    base_df["HashFileName"] = base_df["HashFileName"].astype(str).str.strip()

    merged = subset_df.merge(
        base_df,
        on="HashFileName",
        how="left",
        suffixes=("", "_base"),
    )

    if label_column not in merged.columns and f"{label_column}_base" in merged.columns:
        merged[label_column] = merged[f"{label_column}_base"]
    elif label_column in merged.columns and f"{label_column}_base" in merged.columns:
        subset_labels = merged[label_column]
        subset_non_missing = ~(
            subset_labels.isna()
            | subset_labels.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        )
        merged.loc[~subset_non_missing, label_column] = merged.loc[~subset_non_missing, f"{label_column}_base"]

    if label_column not in merged.columns:
        raise KeyError(
            f"Subset CSV and base metadata are both missing label column '{label_column}'"
        )

    if path_column in merged.columns and f"{path_column}_base" in merged.columns:
        subset_paths = merged[path_column]
        subset_non_missing = ~(
            subset_paths.isna()
            | subset_paths.astype(str).str.strip().str.lower().isin({"", "nan", "none"})
        )
        merged.loc[~subset_non_missing, path_column] = merged.loc[~subset_non_missing, f"{path_column}_base"]

    if "report_text" not in merged.columns and "report_text_base" in merged.columns:
        merged["report_text"] = merged["report_text_base"]

    standardized = standardize_dataset(
        merged,
        split_name=split_name,
        label_column=label_column,
        path_column=path_column,
        text_candidates=text_candidates,
    )
    standardized["subset_source"] = subset_df.get("subset_source", "")
    if "completeness_ratio" in subset_df.columns:
        standardized["completeness_ratio"] = subset_df["completeness_ratio"].values
    return standardized
