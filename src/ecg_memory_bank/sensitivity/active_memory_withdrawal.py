#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from ecg_memory_bank.protocol import external_output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from ecg_memory_bank.retrieval.artifact_utils import (  # noqa: E402
    build_hierarchical_ndcg_arrays,
    build_query_precision_hit_arrays,
    build_union_recall_arrays,
    save_ci_artifacts,
)
from ecg_memory_bank.retrieval.label_mapping_utils import load_label_hierarchy  # noqa: E402
from ecg_memory_bank.retrieval.metrics_utils import (  # noqa: E402
    align_metadata_and_vectors,
    build_faiss_index,
    build_multihot_from_df,
    build_query_level_artifacts,
    build_retrieval_voting_scores,
    build_support_tier_metrics,
    SUPPORT_TIER_DEFINITIONS,
    calculate_metrics_multihot,
    calculate_retrieval_voting_auc_from_scores,
    choose_label_column,
    load_dataset_vectors,
    load_metadata_df,
    faiss,
    safe_mean,
    to_jsonable,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LABEL_MAPPING = RETRIEVAL_DIR / "label_category_counts_9class.csv"
DEFAULT_FINALGALLERY_ROOT = Path("__ECG_DATA_ROOT_REQUIRED__/ecgfounder_heedb/aligned_current_labels")
DEFAULT_ENRICH_ROOT = Path("__ECG_DATA_ROOT_REQUIRED__/heedb_embedding_final")
DEFAULT_REPORT_KS = (1, 3, 5, 10)
DEFAULT_QUERY_YEARS = (2022,)
DEFAULT_HISTORICAL_YEAR_THRESHOLDS = (2010, 2015, 2018)
DEFAULT_PATIENT_WITHDRAWAL_FRACTIONS = (0.15, 0.30, 0.60, 0.90)
FAMILY_NAME = "forgetting_case_withdrawal"

YEAR_PATTERNS = (
    re.compile(r"[\\/](?:S\d{4})[\\/](\d{4})[\\/](\d{2})[\\/]"),
    re.compile(r"(?:^|[\\/])((?:19|20)\d{2})(?:[\\/])"),
)
YEAR_COLUMN_CANDIDATES = ("year", "Year", "ecg_year", "exam_year", "_parsed_year")
PATH_COLUMN_CANDIDATES = (
    "full_ecg_path",
    "ecg_path",
    "path",
    "filepath",
    "file_path",
    "HashFileName",
)
RECORD_ID_CANDIDATES = (
    "HashFileName",
    "full_ecg_path",
    "ecg_path",
    "path",
    "filepath",
    "file_path",
    "exam_id",
    "study_id",
)
PATIENT_ID_CANDIDATES = (
    "bdsp_id",
    "patient_id",
    "subject_id",
    "PatientID",
    "patient",
    "PID",
)
SOURCE_DEFAULTS = {
    "train": {
        "meta_name": "metadata_heedb_ecgfounder_current_labels.csv",
        "emb_name": "heedb_ecgfounder_embeddings_current_labels.pt",
        "enrich_name": "train_processed_update.csv",
    },
    "val": {
        "meta_name": "metadata_heedb_ecgfounder_current_labels.csv",
        "emb_name": "heedb_ecgfounder_embeddings_current_labels.pt",
        "enrich_name": "val_processed_update.csv",
    },
    "test": {
        "meta_name": "metadata_heedb_ecgfounder_current_labels.csv",
        "emb_name": "heedb_ecgfounder_embeddings_current_labels.pt",
        "enrich_name": "",
    },
}


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)


def format_percentage_tag(value: float) -> str:
    percentage = float(value) * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        return f"{int(round(percentage))}pct"
    return f"{percentage:g}pct".replace(".", "p")


def validate_strictly_increasing_ints(values: list[int], label: str) -> list[int]:
    if not values:
        raise ValueError(f"{label} cannot be empty.")
    cleaned = [int(value) for value in values]
    if any(value < 0 for value in cleaned):
        raise ValueError(f"{label} must be non-negative, got {cleaned}")
    for previous, current in zip(cleaned, cleaned[1:]):
        if current <= previous:
            raise ValueError(f"{label} must be strictly increasing, got {cleaned}")
    return cleaned


def validate_strictly_increasing_fractions(values: list[float], label: str) -> list[float]:
    if not values:
        raise ValueError(f"{label} cannot be empty.")
    cleaned = [float(value) for value in values]
    if any(not (0.0 < value < 1.0) for value in cleaned):
        raise ValueError(f"{label} must be in (0, 1), got {cleaned}")
    for previous, current in zip(cleaned, cleaned[1:]):
        if current <= previous:
            raise ValueError(f"{label} must be strictly increasing, got {cleaned}")
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HEEDB case-withdrawal / forgetting experiments: full baseline, "
            "protocol-driven historical withdrawal, and patient-driven mass withdrawal."
        )
    )
    parser.add_argument("--finalgallery-root", default=env_or("FINALGALLERY_ROOT", str(DEFAULT_FINALGALLERY_ROOT)))
    parser.add_argument("--train-meta-path", default=env_or("HEEDB_TRAIN_META_PATH", ""))
    parser.add_argument("--train-ecg-path", default=env_or("HEEDB_TRAIN_ECG_PATH", ""))
    parser.add_argument("--val-meta-path", default=env_or("HEEDB_VAL_META_PATH", ""))
    parser.add_argument("--val-ecg-path", default=env_or("HEEDB_VAL_ECG_PATH", ""))
    parser.add_argument("--test-meta-path", default=env_or("HEEDB_TEST_META_PATH", ""))
    parser.add_argument("--test-ecg-path", default=env_or("HEEDB_TEST_ECG_PATH", ""))
    parser.add_argument(
        "--train-enrich-meta-path",
        default=env_or("HEEDB_TRAIN_ENRICH_META_PATH", str(DEFAULT_ENRICH_ROOT / "train_processed_update.csv")),
    )
    parser.add_argument(
        "--val-enrich-meta-path",
        default=env_or("HEEDB_VAL_ENRICH_META_PATH", str(DEFAULT_ENRICH_ROOT / "val_processed_update.csv")),
    )
    parser.add_argument("--test-enrich-meta-path", default=env_or("HEEDB_TEST_ENRICH_META_PATH", ""))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-years", nargs="+", type=int, default=list(DEFAULT_QUERY_YEARS))
    parser.add_argument(
        "--include-val-query-years-in-gallery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "By default, validation records from query years are excluded from the gallery to avoid "
            "query/gallery leakage. Enable this only if you intentionally want the literal train+val gallery."
        ),
    )
    parser.add_argument(
        "--historical-year-thresholds",
        nargs="+",
        type=int,
        default=None,
        help="Historical withdrawal thresholds. Defaults to 2010 2015 2018 when omitted.",
    )
    parser.add_argument(
        "--historical-year-threshold",
        type=int,
        default=None,
        help="Legacy single-threshold fallback when --historical-year-thresholds is not provided.",
    )
    parser.add_argument(
        "--patient-withdrawal-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Patient withdrawal fractions. Defaults to 0.15 0.30 0.60 0.90 when omitted.",
    )
    parser.add_argument(
        "--patient-withdrawal-fraction",
        type=float,
        default=None,
        help="Legacy single-fraction fallback when --patient-withdrawal-fractions is not provided.",
    )
    parser.add_argument("--patient-withdrawal-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--retrieval-mode",
        choices=("gpu-shard", "cpu", "auto"),
        default=env_or("FORGETTING_RETRIEVAL_MODE", "gpu-shard"),
        help="Default gpu-shard avoids putting the full 8M+ gallery on GPU at once.",
    )
    parser.add_argument(
        "--gallery-shard-size",
        type=int,
        default=int(env_or("FORGETTING_GALLERY_SHARD_SIZE", "250000")),
        help="Gallery rows per GPU shard when --retrieval-mode=gpu-shard.",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--label-col", default=env_or("RETRIEVAL_LABEL_COL", "labels_correct"))
    parser.add_argument("--record-id-col", default=env_or("HEEDB_RECORD_ID_COL", ""))
    parser.add_argument("--patient-id-col", default=env_or("HEEDB_PATIENT_ID_COL", ""))
    parser.add_argument("--heedb-label-mapping-csv", default=env_or("HEEDB_LABEL_MAPPING_CSV", str(DEFAULT_LABEL_MAPPING)))
    return parser.parse_args()


def source_path(args: argparse.Namespace, split_name: str, kind: str) -> Path:
    explicit = getattr(args, f"{split_name}_{kind}_path")
    if explicit:
        return Path(explicit)
    default_name = SOURCE_DEFAULTS[split_name]["meta_name" if kind == "meta" else "emb_name"]
    return Path(args.finalgallery_root) / split_name / default_name


def source_enrich_path(args: argparse.Namespace, split_name: str) -> Path | None:
    explicit = getattr(args, f"{split_name}_enrich_meta_path")
    if explicit:
        path = Path(explicit)
        return path if str(path).strip() else None
    enrich_name = SOURCE_DEFAULTS[split_name].get("enrich_name", "")
    if not enrich_name:
        return None
    return DEFAULT_ENRICH_ROOT / enrich_name


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def missing_mask(series: pd.Series) -> pd.Series:
    return series.apply(is_missing_scalar)


def extract_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    for pattern in YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def infer_year_series(df: pd.DataFrame, split_name: str) -> pd.Series:
    for column_name in YEAR_COLUMN_CANDIDATES:
        if column_name not in df.columns:
            continue
        numeric = pd.to_numeric(df[column_name], errors="coerce")
        if int(numeric.notna().sum()) == len(df):
            return numeric.astype(np.int32)

    best_column = ""
    best_years: pd.Series | None = None
    best_count = -1
    for column_name in PATH_COLUMN_CANDIDATES:
        if column_name not in df.columns:
            continue
        parsed = df[column_name].apply(extract_year)
        parsed_count = int(parsed.notna().sum())
        if parsed_count > best_count:
            best_count = parsed_count
            best_column = column_name
            best_years = parsed

    if best_years is None or best_count <= 0:
        raise KeyError(
            f"Unable to infer ECG year for split '{split_name}'. Tried columns: "
            f"{list(YEAR_COLUMN_CANDIDATES + PATH_COLUMN_CANDIDATES)}"
        )
    missing_count = int(best_years.isna().sum())
    if missing_count > 0:
        examples = df.loc[best_years.isna(), best_column].astype(str).head(10).tolist()
        raise ValueError(
            f"Split '{split_name}' has {missing_count} rows with unparseable year in '{best_column}'. "
            f"Examples: {examples}"
        )
    return best_years.astype(np.int32)


def resolve_field(
    df: pd.DataFrame,
    explicit_col: str,
    candidates: tuple[str, ...],
    field_label: str,
    split_name: str,
    required: bool,
) -> str | None:
    if explicit_col:
        if explicit_col not in df.columns:
            raise KeyError(f"Requested {field_label} column '{explicit_col}' not found in split '{split_name}'.")
        return explicit_col
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(
            f"No {field_label} column found in split '{split_name}'. Tried: {', '.join(candidates)}"
        )
    return None


def build_stable_ids(series: pd.Series | None, split_name: str, prefix: str, row_count: int) -> pd.Series:
    values: list[str] = []
    if series is None:
        return pd.Series([f"{prefix}:{split_name}:row:{row_idx}" for row_idx in range(row_count)], dtype=str)
    for row_idx, raw_value in enumerate(series.tolist()):
        if is_missing_scalar(raw_value):
            values.append(f"{prefix}:{split_name}:row:{row_idx}")
        else:
            values.append(f"{prefix}:{split_name}:{str(raw_value).strip()}")
    return pd.Series(values, dtype=str)


def build_patient_ids(series: pd.Series | None, split_name: str, row_count: int) -> pd.Series:
    values: list[str] = []
    if series is None:
        return pd.Series([f"__missing_patient__:{split_name}:row:{row_idx}" for row_idx in range(row_count)], dtype=str)
    for row_idx, raw_value in enumerate(series.tolist()):
        if is_missing_scalar(raw_value):
            values.append(f"__missing_patient__:{split_name}:row:{row_idx}")
        else:
            values.append(str(raw_value).strip())
    return pd.Series(values, dtype=str)


def enrich_source_metadata(
    df: pd.DataFrame,
    split_name: str,
    enrich_meta_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_fields = ("full_ecg_path", "bdsp_id")
    missing_required = [field_name for field_name in required_fields if field_name not in df.columns]
    has_missing_values = {
        field_name: bool(field_name in df.columns and missing_mask(df[field_name]).any())
        for field_name in required_fields
    }
    needs_enrichment = bool(missing_required or any(has_missing_values.values()))
    enrich_info: dict[str, Any] = {
        "used": False,
        "path": str(enrich_meta_path) if enrich_meta_path else "",
        "filled_columns": [],
        "missing_required_columns_before": missing_required,
    }
    if not needs_enrichment:
        return df, enrich_info
    if enrich_meta_path is None or not enrich_meta_path.exists():
        return df, enrich_info
    if "HashFileName" not in df.columns:
        return df, enrich_info

    enrich_df = pd.read_csv(enrich_meta_path)
    if "HashFileName" not in enrich_df.columns:
        return df, enrich_info
    enrich_df = enrich_df.drop_duplicates(subset=["HashFileName"], keep="first").reset_index(drop=True)
    candidate_columns = [
        "HashFileName",
        "full_ecg_path",
        "bdsp_id",
        "diagnosis_current",
        "diagnosis_previous",
        "HashFileName_previous",
        "deid_t_diagnosis_original",
        "historical_diagnosis",
        "labels_current",
        "labels_historical",
    ]
    available_columns = [column_name for column_name in candidate_columns if column_name in enrich_df.columns]
    enrich_df = enrich_df[available_columns].copy()
    merged = df.merge(enrich_df, on="HashFileName", how="left", suffixes=("", "__enrich"))

    filled_columns: list[str] = []
    for column_name in available_columns:
        if column_name == "HashFileName":
            continue
        enrich_column = f"{column_name}__enrich"
        if enrich_column not in merged.columns:
            continue
        if column_name not in merged.columns:
            merged[column_name] = merged[enrich_column]
            filled_columns.append(column_name)
        else:
            original_missing = missing_mask(merged[column_name])
            if bool(original_missing.any()):
                merged.loc[original_missing, column_name] = merged.loc[original_missing, enrich_column]
                filled_columns.append(column_name)
        merged = merged.drop(columns=[enrich_column])

    enrich_info.update(
        {
            "used": True,
            "filled_columns": sorted(set(filled_columns)),
            "matched_rows": int(merged["HashFileName"].isin(enrich_df["HashFileName"]).sum()),
        }
    )
    return merged, enrich_info


def load_prepared_source(args: argparse.Namespace, split_name: str, require_patient: bool) -> dict[str, Any]:
    meta_path = source_path(args, split_name, "meta")
    emb_path = source_path(args, split_name, "ecg")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found for split '{split_name}': {meta_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found for split '{split_name}': {emb_path}")

    print(f"[Load] {split_name}: metadata={meta_path}")
    df = load_metadata_df(meta_path)
    vecs = load_dataset_vectors(emb_path)
    df, vecs = align_metadata_and_vectors(df, vecs)

    df, enrich_info = enrich_source_metadata(df, split_name, source_enrich_path(args, split_name))
    record_col = resolve_field(
        df,
        args.record_id_col,
        RECORD_ID_CANDIDATES,
        "record id",
        split_name,
        required=False,
    )
    patient_col = resolve_field(
        df,
        args.patient_id_col,
        PATIENT_ID_CANDIDATES,
        "patient id",
        split_name,
        required=require_patient,
    )

    prepared = df.copy()
    prepared["_source_split"] = split_name
    prepared["_source_row"] = np.arange(len(prepared), dtype=np.int64)
    prepared["_parsed_year"] = infer_year_series(prepared, split_name)
    prepared["_record_id"] = build_stable_ids(
        prepared[record_col] if record_col else None,
        split_name=split_name,
        prefix="record",
        row_count=len(prepared),
    )
    prepared["_patient_id"] = build_patient_ids(
        prepared[patient_col] if patient_col else None,
        split_name=split_name,
        row_count=len(prepared),
    )
    return {
        "name": split_name,
        "df": prepared.reset_index(drop=True),
        "vecs": vecs,
        "meta_path": str(meta_path),
        "emb_path": str(emb_path),
        "record_id_col": record_col,
        "patient_id_col": patient_col,
        "enrich_info": enrich_info,
    }


def concat_source_rows(parts: list[tuple[dict[str, Any], np.ndarray]]) -> tuple[pd.DataFrame, np.ndarray]:
    frames: list[pd.DataFrame] = []
    vectors: list[np.ndarray] = []
    for source, mask in parts:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(source["df"]):
            raise ValueError(f"Selection mask length mismatch for split '{source['name']}'.")
        selected_indices = np.flatnonzero(mask)
        if len(selected_indices) == 0:
            continue
        frames.append(source["df"].iloc[selected_indices].reset_index(drop=True))
        if len(selected_indices) == len(source["df"]) and bool(mask.all()):
            vectors.append(source["vecs"])
        else:
            vectors.append(source["vecs"][selected_indices])
    if not frames:
        raise ValueError("No rows selected for concatenated subset.")
    df = pd.concat(frames, ignore_index=True)
    vecs = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if len(df) != len(vecs):
        raise ValueError(f"Concatenated metadata rows={len(df)} but vectors={len(vecs)}.")
    return df, vecs


def maybe_limit_queries(
    query_df: pd.DataFrame,
    query_vecs: np.ndarray,
    max_queries: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    if max_queries <= 0 or max_queries >= len(query_df):
        return query_df.reset_index(drop=True), query_vecs, {"applied": False, "max_queries": int(max_queries)}
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(query_df), size=int(max_queries), replace=False))
    return (
        query_df.iloc[selected].reset_index(drop=True),
        query_vecs[selected],
        {"applied": True, "max_queries": int(max_queries), "seed": int(seed), "selected_count": int(len(selected))},
    )


def resolve_report_ks(top_k: int) -> list[int]:
    return [value for value in DEFAULT_REPORT_KS if value <= int(top_k)]


def search_gpu_sharded(
    gallery_vecs: np.ndarray,
    query_vecs: np.ndarray,
    top_k: int,
    batch_size: int,
    shard_size: int,
    scenario_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if faiss is None:
        raise RuntimeError("FAISS is required for GPU shard retrieval.")
    if shard_size <= 0:
        raise ValueError(f"gallery_shard_size must be positive, got {shard_size}")

    gallery_size = int(gallery_vecs.shape[0])
    query_size = int(query_vecs.shape[0])
    effective_top_k = min(int(top_k), gallery_size)
    global_scores = np.full((query_size, effective_top_k), -np.inf, dtype=np.float32)
    global_indices = np.full((query_size, effective_top_k), -1, dtype=np.int64)
    shard_ranges = list(range(0, gallery_size, shard_size))

    for offset in range(0, gallery_size, shard_size):
        shard_vecs = np.ascontiguousarray(
            gallery_vecs[offset : offset + shard_size].astype(np.float32, copy=False)
        )
        shard_top_k = min(effective_top_k, int(shard_vecs.shape[0]))
        dimension = int(shard_vecs.shape[1])
        index_flat = faiss.IndexFlatIP(dimension)
        resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index_flat)
        gpu_index.add(shard_vecs)
        shard_ordinal = offset // shard_size + 1
        desc = f"{scenario_name} shard {shard_ordinal}/{len(shard_ranges)}"
        for start in tqdm(range(0, query_size, batch_size), desc=desc, leave=False):
            end = min(start + batch_size, query_size)
            query_batch = np.ascontiguousarray(query_vecs[start:end].astype(np.float32, copy=False))
            shard_scores, shard_indices = gpu_index.search(query_batch, shard_top_k)
            shard_indices = shard_indices.astype(np.int64, copy=False) + offset
            merged_scores = np.concatenate(
                [global_scores[start:end], shard_scores.astype(np.float32, copy=False)],
                axis=1,
            )
            merged_indices = np.concatenate([global_indices[start:end], shard_indices], axis=1)
            order = np.argsort(-merged_scores, axis=1)[:, :effective_top_k]
            global_scores[start:end] = np.take_along_axis(merged_scores, order, axis=1)
            global_indices[start:end] = np.take_along_axis(merged_indices, order, axis=1)
        del gpu_index, resources, index_flat, shard_vecs
        gc.collect()
    return global_scores, global_indices


def build_query_metadata_rows(query_labels: list[dict[str, list[str]]]) -> dict[str, np.ndarray]:
    return {
        "l1_positive_count": np.asarray([len(item["l1"]) for item in query_labels], dtype=np.int32),
        "l2_positive_count": np.asarray([len(item["l2"]) for item in query_labels], dtype=np.int32),
        "l3_positive_count": np.asarray([len(item["l3"]) for item in query_labels], dtype=np.int32),
    }


def summarize_split_years(df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        df.groupby(["_source_split", "_parsed_year"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["_source_split", "_parsed_year"])
    )
    return [
        {
            "source_split": str(row["_source_split"]),
            "year": int(row["_parsed_year"]),
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


def summarize_subset(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": name,
        "row_count": int(len(df)),
        "unique_record_count": int(df["_record_id"].astype(str).nunique()),
        "unique_patient_count": int(df["_patient_id"].astype(str).nunique()),
        "by_split_year": summarize_split_years(df),
    }


def build_persistence_arrays(
    *,
    retrieved_indices: np.ndarray,
    gallery_df: pd.DataFrame,
    deleted_record_ids: pd.Series | None,
    deleted_patient_ids: set[str] | None,
    report_ks: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}

    record_flags = np.zeros(len(gallery_df), dtype=np.uint8)
    if deleted_record_ids is not None and len(deleted_record_ids) > 0:
        record_flags = gallery_df["_record_id"].astype(str).isin(deleted_record_ids.astype(str)).to_numpy(dtype=np.uint8)
    retrieved_record_flags = record_flags[retrieved_indices]

    patient_flags = np.zeros(len(gallery_df), dtype=np.uint8)
    if deleted_patient_ids:
        patient_flags = gallery_df["_patient_id"].astype(str).isin(deleted_patient_ids).to_numpy(dtype=np.uint8)
    retrieved_patient_flags = patient_flags[retrieved_indices]

    for report_k in report_ks:
        record_block = retrieved_record_flags[:, :report_k]
        patient_block = retrieved_patient_flags[:, :report_k]
        arrays[f"deleted_record_slot_rate_at_{report_k}"] = record_block.mean(axis=1).astype(np.float32)
        arrays[f"deleted_record_hit_at_{report_k}"] = record_block.any(axis=1).astype(np.uint8)
        arrays[f"deleted_patient_slot_rate_at_{report_k}"] = patient_block.mean(axis=1).astype(np.float32)
        arrays[f"deleted_patient_hit_at_{report_k}"] = patient_block.any(axis=1).astype(np.uint8)
        summary[f"deleted_record_slot_rate@{report_k}"] = safe_mean(arrays[f"deleted_record_slot_rate_at_{report_k}"])
        summary[f"deleted_record_query_rate@{report_k}"] = safe_mean(arrays[f"deleted_record_hit_at_{report_k}"])
        summary[f"deleted_patient_slot_rate@{report_k}"] = safe_mean(arrays[f"deleted_patient_slot_rate_at_{report_k}"])
        summary[f"deleted_patient_query_rate@{report_k}"] = safe_mean(arrays[f"deleted_patient_hit_at_{report_k}"])

    primary_basis = "none"
    if deleted_patient_ids:
        primary_basis = "patient"
    elif deleted_record_ids is not None and len(deleted_record_ids) > 0:
        primary_basis = "record"
    primary_key = (
        f"deleted_patient_query_rate@{max(report_ks)}"
        if primary_basis == "patient"
        else f"deleted_record_query_rate@{max(report_ks)}"
    )
    summary["primary_persistence_basis"] = primary_basis
    summary[f"primary_persistence_query_rate@{max(report_ks)}"] = (
        float(summary.get(primary_key, 0.0)) if primary_basis != "none" else None
    )
    summary["remaining_deleted_record_count_in_gallery"] = int(record_flags.sum())
    summary["remaining_deleted_patient_record_count_in_gallery"] = int(patient_flags.sum())
    if primary_basis != "none" and (
        summary["remaining_deleted_record_count_in_gallery"]
        or summary["remaining_deleted_patient_record_count_in_gallery"]
        or summary[f"primary_persistence_query_rate@{max(report_ks)}"] != 0.0
    ):
        raise ValueError("Withdrawal consistency check failed: removed cases remain active")
    return arrays, summary


def run_retrieval(
    *,
    gallery_vecs: np.ndarray,
    query_vecs: np.ndarray,
    top_k: int,
    batch_size: int,
    scenario_name: str,
    retrieval_mode: str,
    gallery_shard_size: int,
) -> np.ndarray:
    effective_top_k = min(int(top_k), int(gallery_vecs.shape[0]))
    if effective_top_k <= 0:
        raise ValueError(f"Scenario '{scenario_name}' has an empty gallery.")
    query_vecs = np.ascontiguousarray(query_vecs.astype(np.float32, copy=False))
    mode = str(retrieval_mode).lower().strip()
    if mode == "auto":
        mode = "gpu-shard" if faiss is not None else "cpu"

    if mode == "cpu":
        if faiss is None:
            index = build_faiss_index(gallery_vecs)
        else:
            index = faiss.IndexFlatIP(gallery_vecs.shape[1])
            index.add(np.ascontiguousarray(gallery_vecs.astype(np.float32, copy=False)))
        all_indices = []
        for start in tqdm(range(0, query_vecs.shape[0], batch_size), desc=f"Retrieving {scenario_name}"):
            _, indices = index.search(query_vecs[start : start + batch_size], effective_top_k)
            all_indices.append(indices)
        return np.concatenate(all_indices, axis=0)

    if mode != "gpu-shard":
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")
    if faiss is None:
        print("!! Warning: FAISS is unavailable; falling back to CPU retrieval.")
        return run_retrieval(
            gallery_vecs=gallery_vecs,
            query_vecs=query_vecs,
            top_k=effective_top_k,
            batch_size=batch_size,
            scenario_name=scenario_name,
            retrieval_mode="cpu",
            gallery_shard_size=gallery_shard_size,
        )
    if gallery_shard_size <= 0:
        raise ValueError(f"gallery_shard_size must be positive, got {gallery_shard_size}")
    _, indices = search_gpu_sharded(
        gallery_vecs=gallery_vecs,
        query_vecs=query_vecs,
        top_k=effective_top_k,
        batch_size=batch_size,
        shard_size=gallery_shard_size,
        scenario_name=scenario_name,
    )
    return indices


def evaluate_scenario(
    *,
    scenario_name: str,
    withdrawal_type: str,
    withdrawal_label: str,
    gallery_df: pd.DataFrame,
    gallery_vecs: np.ndarray,
    gallery_col: str,
    query_df: pd.DataFrame,
    query_vecs: np.ndarray,
    query_col: str,
    label_hierarchy: dict[str, tuple[str, str]],
    reference_classes: dict[str, list[str]] | None,
    top_k: int,
    batch_size: int,
    retrieval_mode: str,
    gallery_shard_size: int,
    family_output_dir: Path,
    baseline_gallery_size: int,
    deleted_df: pd.DataFrame | None = None,
    deleted_patient_ids: set[str] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    scenario_output_dir = family_output_dir / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    report_ks = resolve_report_ks(top_k)

    l1_db, l2_db, l3_db, effective_reference_classes, db_labels = build_multihot_from_df(
        gallery_df,
        gallery_col,
        label_hierarchy,
        reference_classes=reference_classes,
    )
    l1_query, l2_query, l3_query, _, query_labels = build_multihot_from_df(
        query_df,
        query_col,
        label_hierarchy,
        reference_classes=effective_reference_classes,
    )

    retrieved_indices = run_retrieval(
        gallery_vecs=gallery_vecs,
        query_vecs=query_vecs,
        top_k=top_k,
        batch_size=batch_size,
        scenario_name=scenario_name,
        retrieval_mode=retrieval_mode,
        gallery_shard_size=gallery_shard_size,
    )
    effective_top_k = int(retrieved_indices.shape[1])

    per_query_df, query_metric_arrays = build_query_level_artifacts(
        retrieved_indices=retrieved_indices,
        db_labels=db_labels,
        query_labels=query_labels,
        top_k=effective_top_k,
    )
    query_metric_arrays.update(build_union_recall_arrays(retrieved_indices, db_labels, query_labels, report_ks))
    query_metric_arrays.update(build_hierarchical_ndcg_arrays(retrieved_indices, db_labels, query_labels, report_ks))
    query_metric_arrays.update(build_query_metadata_rows(query_labels))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l1_db, l1_query, report_ks, "l1"))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l2_db, l2_query, report_ks, "l2"))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l3_db, l3_query, report_ks, "l3"))

    deleted_record_ids = None
    if deleted_df is not None and len(deleted_df) > 0:
        deleted_record_ids = deleted_df["_record_id"].astype(str)
    persistence_arrays, persistence_summary = build_persistence_arrays(
        retrieved_indices=retrieved_indices,
        gallery_df=gallery_df,
        deleted_record_ids=deleted_record_ids,
        deleted_patient_ids=deleted_patient_ids,
        report_ks=report_ks,
    )
    query_metric_arrays.update(persistence_arrays)

    for report_k in report_ks:
        per_query_df[f"union_recall_l3@{report_k}"] = query_metric_arrays[f"union_recall_l3_at_{report_k}"]
        per_query_df[f"hierarchical_ndcg@{report_k}"] = query_metric_arrays[f"hierarchical_ndcg_at_{report_k}"]
        per_query_df[f"l1_p@{report_k}"] = query_metric_arrays[f"l1_p_at_{report_k}"]
        per_query_df[f"l2_p@{report_k}"] = query_metric_arrays[f"l2_p_at_{report_k}"]
        per_query_df[f"l3_p@{report_k}"] = query_metric_arrays[f"l3_p_at_{report_k}"]
        per_query_df[f"l1_hit@{report_k}"] = query_metric_arrays[f"l1_hit_at_{report_k}"]
        per_query_df[f"l2_hit@{report_k}"] = query_metric_arrays[f"l2_hit_at_{report_k}"]
        per_query_df[f"l3_hit@{report_k}"] = query_metric_arrays[f"l3_hit_at_{report_k}"]
        per_query_df[f"deleted_record_hit@{report_k}"] = query_metric_arrays[f"deleted_record_hit_at_{report_k}"]
        per_query_df[f"deleted_patient_hit@{report_k}"] = query_metric_arrays[f"deleted_patient_hit_at_{report_k}"]
    for column_name in ("_source_split", "_source_row", "_parsed_year", "_record_id", "_patient_id"):
        if column_name in query_df.columns:
            per_query_df[column_name] = query_df[column_name].to_numpy()
    per_query_df["l1_positive_count"] = query_metric_arrays["l1_positive_count"]
    per_query_df["l2_positive_count"] = query_metric_arrays["l2_positive_count"]
    per_query_df["l3_positive_count"] = query_metric_arrays["l3_positive_count"]

    per_query_path = scenario_output_dir / f"{scenario_name}_per_query.csv"
    per_query_df.to_csv(per_query_path, index=False)

    retrieval_hits = {}
    retrieval_precision = {}
    voting_results = {}
    support_tier_results: dict[str, dict[str, Any]] = {}
    per_class_paths: dict[str, str] = {}
    auc_inputs = {}
    for level_name, db_mat, query_mat, class_names in [
        ("L1", l1_db, l1_query, effective_reference_classes["l1"]),
        ("L2", l2_db, l2_query, effective_reference_classes["l2"]),
        ("L3", l3_db, l3_query, effective_reference_classes["l3"]),
    ]:
        retrieval_metrics, _ = calculate_metrics_multihot(
            retrieved_indices=retrieved_indices,
            db_mat=db_mat,
            query_mat=query_mat,
            k_list=report_ks,
            class_names=class_names,
        )
        retrieval_hits[level_name] = {
            f"Hit@{report_k}": retrieval_metrics.get(f"Hit@{report_k}", float("nan")) for report_k in report_ks
        }
        retrieval_precision[level_name] = {
            f"P@{report_k}": retrieval_metrics.get(f"P@{report_k}", float("nan")) for report_k in report_ks
        }

        y_score = build_retrieval_voting_scores(retrieved_indices, db_mat, top_k=effective_top_k)
        auc_summary, auc_df = calculate_retrieval_voting_auc_from_scores(
            query_mat,
            y_score,
            class_names,
            effective_top_k,
        )
        voting_results[level_name] = auc_summary
        auc_inputs[level_name.lower()] = {
            "y_true": query_mat,
            "y_score": y_score,
            "class_names": class_names,
        }
        support_tier_results[level_name] = build_support_tier_metrics(auc_df)

        auc_path = scenario_output_dir / f"per_class_auc_{scenario_name}_{level_name}.csv"
        auc_df.to_csv(auc_path, index=False)
        per_class_paths[level_name] = str(auc_path)

    deleted_count = int(len(deleted_df)) if deleted_df is not None else 0
    deleted_patient_count = int(deleted_df["_patient_id"].astype(str).nunique()) if deleted_df is not None and len(deleted_df) else 0
    summary = {
        "scenario": scenario_name,
        "withdrawal_type": withdrawal_type,
        "withdrawal_label": withdrawal_label,
        "top_k_requested": int(top_k),
        "top_k_effective": int(effective_top_k),
        "num_gallery": int(len(gallery_df)),
        "num_gallery_baseline": int(baseline_gallery_size),
        "num_deleted_gallery_records": deleted_count,
        "num_deleted_gallery_patients": deleted_patient_count,
        "deleted_gallery_fraction_of_baseline": float(deleted_count / baseline_gallery_size) if baseline_gallery_size else 0.0,
        "num_queries": int(len(query_df)),
        "reference_class_counts": {key: int(len(value)) for key, value in effective_reference_classes.items()},
        "persistence": persistence_summary,
        "deleted_distribution": summarize_subset("deleted_records", deleted_df) if deleted_df is not None and len(deleted_df) else None,
        "retrieval_hits": retrieval_hits,
        "retrieval_precision": retrieval_precision,
        "voting_auc": voting_results,
        "paths": {
            "scenario_output_dir": str(scenario_output_dir),
            "per_query_csv": str(per_query_path),
        },
    }
    for report_k in report_ks:
        summary[f"mean_union_recall_l3@{report_k}"] = safe_mean(query_metric_arrays[f"union_recall_l3_at_{report_k}"])
        summary[f"mean_hierarchical_ndcg@{report_k}"] = safe_mean(query_metric_arrays[f"hierarchical_ndcg_at_{report_k}"])
    summary["support_tier_metrics"] = support_tier_results

    if extra_manifest:
        summary["withdrawal_manifest"] = extra_manifest

    summary_path = scenario_output_dir / f"{scenario_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)
    summary["paths"]["summary_json"] = str(summary_path)

    save_ci_artifacts(
        scenario_output_dir=scenario_output_dir,
        family=FAMILY_NAME,
        scenario_name=scenario_name,
        summary_path=summary_path,
        query_metric_arrays=query_metric_arrays,
        auc_inputs=auc_inputs,
        extra_manifest={
            "top_k_requested": int(top_k),
            "top_k_effective": int(effective_top_k),
            "withdrawal_type": withdrawal_type,
            "withdrawal_label": withdrawal_label,
            "per_query_csv": str(per_query_path),
            "per_class_auc_csv": per_class_paths,
            "persistence": persistence_summary,
        },
    )
    return summary, effective_reference_classes


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "scenario": summary["scenario"],
        "withdrawal_type": summary["withdrawal_type"],
        "withdrawal_label": summary["withdrawal_label"],
        "num_gallery": summary["num_gallery"],
        "num_gallery_baseline": summary["num_gallery_baseline"],
        "num_deleted_gallery_records": summary["num_deleted_gallery_records"],
        "num_deleted_gallery_patients": summary["num_deleted_gallery_patients"],
        "deleted_gallery_fraction_of_baseline": summary["deleted_gallery_fraction_of_baseline"],
        "num_queries": summary["num_queries"],
    }
    for report_k in resolve_report_ks(summary["top_k_effective"]):
        row[f"mean_union_recall_l3@{report_k}"] = summary.get(f"mean_union_recall_l3@{report_k}", float("nan"))
        row[f"mean_hierarchical_ndcg@{report_k}"] = summary.get(f"mean_hierarchical_ndcg@{report_k}", float("nan"))
        row[f"primary_persistence_query_rate@{report_k}"] = summary["persistence"].get(
            f"primary_persistence_query_rate@{report_k}",
            float("nan"),
        )
    for level_name, level_summary in summary["voting_auc"].items():
        prefix = level_name.lower()
        row[f"{prefix}_macro_auc"] = level_summary["macro_auc"]
        row[f"{prefix}_micro_auc"] = level_summary["micro_auc"]
        row[f"{prefix}_valid_class_count"] = level_summary["valid_class_count"]
        row[f"{prefix}_skipped_class_count"] = level_summary["skipped_class_count"]
    l3_support_tiers = summary.get("support_tier_metrics", {}).get("L3", {})
    for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS:
        tier_payload = l3_support_tiers.get(tier_name, {})
        row[f"l3_{tier_name}_support_min"] = tier_payload.get("support_min", min_support)
        row[f"l3_{tier_name}_support_max"] = tier_payload.get("support_max", max_support)
        row[f"l3_{tier_name}_class_count"] = tier_payload.get("class_count", float("nan"))
        row[f"l3_{tier_name}_mean_auc"] = tier_payload.get("mean_auc", float("nan"))
    return row


def build_patient_withdrawal_sets(
    gallery_df: pd.DataFrame,
    fractions: list[float],
    seed: int,
    explicit_count: int,
) -> list[tuple[float, set[str]]]:
    if explicit_count < 0:
        raise ValueError(f"patient_withdrawal_count must be >= 0, got {explicit_count}")
    if explicit_count > 0 and len(fractions) != 1:
        raise ValueError("patient_withdrawal_count can only be used with a single patient-withdrawal fraction.")

    patients = np.asarray(sorted(gallery_df["_patient_id"].astype(str).unique().tolist()), dtype=object)
    if len(patients) == 0:
        raise ValueError("Baseline gallery has no patient ids for patient-level withdrawal.")

    rng = np.random.default_rng(seed)
    shuffled_patients = patients.copy()
    rng.shuffle(shuffled_patients)

    selected_sets: list[tuple[float, set[str]]] = []
    if explicit_count > 0:
        target_count = min(int(explicit_count), len(shuffled_patients))
        selected_sets.append((float(fractions[0]), {str(value) for value in shuffled_patients[:target_count].tolist()}))
        return selected_sets

    total_patients = int(len(shuffled_patients))
    previous_count = 0
    for fraction in fractions:
        target_count = max(1, int(round(total_patients * float(fraction))))
        target_count = max(target_count, previous_count + 1)
        target_count = min(target_count, total_patients)
        selected_sets.append((float(fraction), {str(value) for value in shuffled_patients[:target_count].tolist()}))
        previous_count = target_count
    return selected_sets


def main() -> None:
    args = parse_args()
    if args.top_k < 10:
        raise ValueError("top_k must be >= 10 for the forgetting experiment.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    family_output_dir = external_output_dir(args.output_dir) / FAMILY_NAME
    family_output_dir.mkdir(parents=True, exist_ok=True)
    query_years = sorted({int(year) for year in args.query_years})
    historical_year_thresholds = validate_strictly_increasing_ints(
        list(args.historical_year_thresholds)
        if args.historical_year_thresholds is not None
        else ([int(args.historical_year_threshold)] if args.historical_year_threshold is not None else list(DEFAULT_HISTORICAL_YEAR_THRESHOLDS)),
        "historical_year_thresholds",
    )
    patient_withdrawal_fractions = validate_strictly_increasing_fractions(
        list(args.patient_withdrawal_fractions)
        if args.patient_withdrawal_fractions is not None
        else ([float(args.patient_withdrawal_fraction)] if args.patient_withdrawal_fraction is not None else list(DEFAULT_PATIENT_WITHDRAWAL_FRACTIONS)),
        "patient_withdrawal_fractions",
    )
    if args.patient_withdrawal_count > 0 and len(patient_withdrawal_fractions) != 1:
        raise ValueError("patient_withdrawal_count can only be used when a single patient-withdrawal fraction is provided.")

    label_hierarchy = load_label_hierarchy(args.heedb_label_mapping_csv)
    sources = {
        "train": load_prepared_source(args, "train", require_patient=True),
        "val": load_prepared_source(args, "val", require_patient=True),
        "test": load_prepared_source(args, "test", require_patient=False),
    }

    val_query_mask = sources["val"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    test_query_mask = sources["test"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    query_df, query_vecs = concat_source_rows(
        [
            (sources["val"], val_query_mask),
            (sources["test"], test_query_mask),
        ]
    )
    query_df, query_vecs, query_limit_manifest = maybe_limit_queries(
        query_df,
        query_vecs,
        max_queries=int(args.max_queries),
        seed=int(args.seed),
    )

    train_gallery_mask = np.ones(len(sources["train"]["df"]), dtype=bool)
    val_gallery_mask = np.ones(len(sources["val"]["df"]), dtype=bool)
    if not args.include_val_query_years_in_gallery:
        val_gallery_mask &= ~sources["val"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    baseline_gallery_df, baseline_gallery_vecs = concat_source_rows(
        [
            (sources["train"], train_gallery_mask),
            (sources["val"], val_gallery_mask),
        ]
    )
    baseline_gallery_size = int(len(baseline_gallery_df))
    for source in sources.values():
        source["vecs"] = np.empty((0, 0), dtype=np.float32)
    gc.collect()

    split_manifest = {
        "family": FAMILY_NAME,
        "query_years": query_years,
        "gallery_policy": {
            "sources": ["train", "val"],
            "include_val_query_years_in_gallery": bool(args.include_val_query_years_in_gallery),
            "note": (
                "Validation records from query years are excluded by default so val-2022 queries are not "
                "also present in the gallery."
            ),
        },
        "query_limit": query_limit_manifest,
        "sources": {
            split_name: {
                "meta_path": source["meta_path"],
                "emb_path": source["emb_path"],
                "record_id_col": source["record_id_col"],
                "patient_id_col": source["patient_id_col"],
                "enrich_info": source["enrich_info"],
                "row_count": int(len(source["df"])),
                "year_distribution": summarize_split_years(source["df"]),
            }
            for split_name, source in sources.items()
        },
        "subsets": {
            "baseline_gallery": summarize_subset("baseline_gallery", baseline_gallery_df),
            "query_holdout": summarize_subset("query_holdout", query_df),
        },
        "parameters": {
            "historical_year_thresholds": historical_year_thresholds,
            "patient_withdrawal_fractions": patient_withdrawal_fractions,
            "patient_withdrawal_count": int(args.patient_withdrawal_count),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "batch_size": int(args.batch_size),
            "retrieval_mode": str(args.retrieval_mode),
            "gallery_shard_size": int(args.gallery_shard_size),
        },
    }
    split_manifest_path = family_output_dir / "case_withdrawal_split_manifest.json"
    split_manifest_path.write_text(json.dumps(to_jsonable(split_manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] Split manifest -> {split_manifest_path}")

    gallery_col = choose_label_column(baseline_gallery_df, args.label_col)
    query_col = choose_label_column(query_df, args.label_col)

    summaries: list[dict[str, Any]] = []
    reference_classes: dict[str, list[str]] | None = None

    baseline_summary, reference_classes = evaluate_scenario(
        scenario_name="full_gallery_2022_holdout_baseline",
        withdrawal_type="baseline",
        withdrawal_label="Full gallery baseline",
        gallery_df=baseline_gallery_df,
        gallery_vecs=baseline_gallery_vecs,
        gallery_col=gallery_col,
        query_df=query_df,
        query_vecs=query_vecs,
        query_col=query_col,
        label_hierarchy=label_hierarchy,
        reference_classes=None,
        top_k=int(args.top_k),
        batch_size=int(args.batch_size),
        retrieval_mode=str(args.retrieval_mode),
        gallery_shard_size=int(args.gallery_shard_size),
        family_output_dir=family_output_dir,
        baseline_gallery_size=baseline_gallery_size,
        deleted_df=None,
        deleted_patient_ids=None,
        extra_manifest={"split_manifest": str(split_manifest_path)},
    )
    summaries.append(baseline_summary)

    for threshold in historical_year_thresholds:
        historical_delete_mask = baseline_gallery_df["_parsed_year"].to_numpy(dtype=np.int32) <= int(threshold)
        historical_deleted_df = baseline_gallery_df.loc[historical_delete_mask].reset_index(drop=True)
        historical_gallery_df = baseline_gallery_df.loc[~historical_delete_mask].reset_index(drop=True)
        historical_gallery_vecs = baseline_gallery_vecs[~historical_delete_mask]
        historical_summary, _ = evaluate_scenario(
            scenario_name=f"protocol_historical_y{threshold}_withdrawal",
            withdrawal_type="protocol_historical",
            withdrawal_label=f"Protocol-driven historical withdrawal (year <= {threshold})",
            gallery_df=historical_gallery_df,
            gallery_vecs=historical_gallery_vecs,
            gallery_col=gallery_col,
            query_df=query_df,
            query_vecs=query_vecs,
            query_col=query_col,
            label_hierarchy=label_hierarchy,
            reference_classes=reference_classes,
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            retrieval_mode=str(args.retrieval_mode),
            gallery_shard_size=int(args.gallery_shard_size),
            family_output_dir=family_output_dir,
            baseline_gallery_size=baseline_gallery_size,
            deleted_df=historical_deleted_df,
            deleted_patient_ids=None,
            extra_manifest={
                "split_manifest": str(split_manifest_path),
                "year_threshold": int(threshold),
                "deleted_by_split_year": summarize_split_years(historical_deleted_df),
            },
        )
        summaries.append(historical_summary)
        del historical_gallery_df, historical_gallery_vecs, historical_deleted_df, historical_delete_mask
        gc.collect()

    patient_withdrawal_sets = build_patient_withdrawal_sets(
        baseline_gallery_df,
        fractions=patient_withdrawal_fractions,
        seed=int(args.seed),
        explicit_count=int(args.patient_withdrawal_count),
    )
    for fraction, withdrawn_patients in patient_withdrawal_sets:
        patient_delete_mask = baseline_gallery_df["_patient_id"].astype(str).isin(withdrawn_patients).to_numpy(dtype=bool)
        patient_deleted_df = baseline_gallery_df.loc[patient_delete_mask].reset_index(drop=True)
        patient_gallery_df = baseline_gallery_df.loc[~patient_delete_mask].reset_index(drop=True)
        patient_gallery_vecs = baseline_gallery_vecs[~patient_delete_mask]
        patient_fraction_tag = (
            f"{len(withdrawn_patients)}patients"
            if int(args.patient_withdrawal_count) > 0
            else format_percentage_tag(float(fraction))
        )
        patient_summary, _ = evaluate_scenario(
            scenario_name=f"patient_mass_withdrawal_{patient_fraction_tag}",
            withdrawal_type="patient_mass",
            withdrawal_label=f"Patient-driven mass withdrawal ({len(withdrawn_patients)} gallery patients)",
            gallery_df=patient_gallery_df,
            gallery_vecs=patient_gallery_vecs,
            gallery_col=gallery_col,
            query_df=query_df,
            query_vecs=query_vecs,
            query_col=query_col,
            label_hierarchy=label_hierarchy,
            reference_classes=reference_classes,
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            retrieval_mode=str(args.retrieval_mode),
            gallery_shard_size=int(args.gallery_shard_size),
            family_output_dir=family_output_dir,
            baseline_gallery_size=baseline_gallery_size,
            deleted_df=patient_deleted_df,
            deleted_patient_ids=withdrawn_patients,
            extra_manifest={
                "split_manifest": str(split_manifest_path),
                "seed": int(args.seed),
                "requested_patient_fraction": float(fraction),
                "withdrawn_patient_count": int(len(withdrawn_patients)),
                "withdrawn_patient_fraction_of_baseline_unique_patients": float(
                    len(withdrawn_patients) / baseline_gallery_df["_patient_id"].nunique()
                ),
                "deletion_restricted_to_gallery_records": True,
                "test_queries_retained_regardless_of_patient_overlap": True,
            },
        )
        summaries.append(patient_summary)
        del patient_gallery_df, patient_gallery_vecs, patient_deleted_df, patient_delete_mask
        gc.collect()

    summary_json_path = family_output_dir / "case_withdrawal_summary.json"
    summary_json_path.write_text(json.dumps(to_jsonable(summaries), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_csv_path = family_output_dir / "case_withdrawal_summary.csv"
    pd.DataFrame([flatten_summary(summary) for summary in summaries]).to_csv(summary_csv_path, index=False)

    print(f"[Saved] Summary JSON -> {summary_json_path}")
    print(f"[Saved] Summary CSV  -> {summary_csv_path}")


if __name__ == "__main__":
    main()
