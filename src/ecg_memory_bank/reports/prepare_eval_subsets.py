"""Subject-disjoint report retrieval split.

Purpose
-------
Build the fixed query and active-gallery CSVs for ECG-Grounding report
generation.

Method
------
Select the supplied benchmark query rows, remove every query subject from the
gallery, and preserve row-to-embedding keys.

Inputs
------
MIMIC metadata with study_id and subject_id plus benchmark query-subset CSV.

Outputs
-------
External query_subset.csv, gallery_subset.csv, and split summary.

Paper parameters
----------------
Formal run uses all 2381 queries and seed=42; random self-split fallback is
outside the formal benchmark path.

Shapes and meaning
------------------
The query set has Q=2381 rows; the gallery has no subject_id present among
query rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import write_json
from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.data_io import (
    align_embeddings_and_metadata,
    attach_corrected_labels,
    load_embeddings,
    load_metadata,
    standardize_dataset,
)


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Prepare MIMIC self-RAG query/gallery subsets.")
    parser.add_argument("--run-name", type=str, default=config.grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--query-size", type=int, default=config.grounding.query_sample_size)
    parser.add_argument("--seed", type=int, default=config.sampling.random_seed)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--query-subset-csv", type=str, default=config.dataset.query_subset_csv)
    return parser.parse_args()


def _resolve_run_dir(config, run_name: str, run_dir_arg: str) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    return config.output.runs_root / run_name


def _load_mimic_df(config) -> pd.DataFrame:
    embeddings = load_embeddings(config.dataset.train_embeddings_path)
    metadata = load_metadata(config.dataset.train_metadata_path)
    _, metadata = align_embeddings_and_metadata(embeddings, metadata)
    metadata = attach_corrected_labels(
        metadata,
        config.dataset.corrected_label_metadata_path,
        label_column=config.dataset.label_column,
    )
    return standardize_dataset(
        metadata,
        split_name="mimic",
        label_column=config.dataset.label_column,
        path_column=config.dataset.path_column,
        text_candidates=config.dataset.report_text_candidates,
    )


def _build_query_subset_from_csv(
    subset_csv_path: str,
    base_df: pd.DataFrame,
    *,
    config,
    case_id: str,
) -> tuple[pd.DataFrame, dict]:
    subset_path = Path(subset_csv_path)
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing query subset csv: {subset_path}")

    subset_df = pd.read_csv(subset_path)
    key_column = config.dataset.feature_match_key
    if key_column not in subset_df.columns:
        raise KeyError(f"Subset CSV missing required key column '{key_column}'")

    subset_df = subset_df.copy()
    subset_df[key_column] = subset_df[key_column].astype(str).str.strip()
    base_tmp = base_df.copy()
    base_tmp[key_column] = base_tmp[key_column].astype(str).str.strip()

    merged = subset_df.merge(
        base_tmp,
        on=key_column,
        how="left",
        suffixes=("", "_base"),
    )

    def _merged_base_col(column: str) -> str | None:
        base_col = f"{column}_base"
        if base_col in merged.columns:
            return base_col
        if column in merged.columns:
            return column
        return None

    row_index_col = _merged_base_col("row_index")
    if row_index_col is None:
        raise KeyError(f"Base metadata did not provide row_index for key '{key_column}'")

    matched_mask = merged[row_index_col].notna()
    if not matched_mask.all():
        missing_keys = merged.loc[~matched_mask, key_column].astype(str).tolist()[:10]
        raise RuntimeError(
            f"Subset CSV contains keys not found in MIMIC metadata by '{key_column}'. Examples: {missing_keys}"
        )

    for column in ("case_id", "patient_id", "row_index", "source_split", "labels_list", "labels_text", "primary_label", "report_text"):
        source_col = _merged_base_col(column)
        if source_col is not None:
            merged[column] = merged[source_col]

    if config.dataset.label_column not in merged.columns and f"{config.dataset.label_column}_base" in merged.columns:
        merged[config.dataset.label_column] = merged[f"{config.dataset.label_column}_base"]
    elif config.dataset.label_column in merged.columns and f"{config.dataset.label_column}_base" in merged.columns:
        subset_labels = merged[config.dataset.label_column].astype(str).str.strip()
        missing_label_mask = subset_labels.eq("") | subset_labels.str.lower().isin({"nan", "none"})
        merged.loc[missing_label_mask, config.dataset.label_column] = merged.loc[missing_label_mask, f"{config.dataset.label_column}_base"]

    if "final_report" in merged.columns:
        final_report = merged["final_report"].astype(str).str.strip()
        report_missing_mask = merged.get("report_text", pd.Series([""] * len(merged))).astype(str).str.strip().eq("")
        merged.loc[report_missing_mask & final_report.ne(""), "report_text"] = final_report.loc[report_missing_mask & final_report.ne("")]

    merged["row_index"] = merged["row_index"].astype(int)
    merged["source_split"] = "mimic_query_subset"
    if case_id:
        merged = merged.loc[merged["case_id"].astype(str) == case_id].copy()
        if merged.empty:
            raise KeyError(f"case_id not found in external query subset: {case_id}")

    query_df = merged.reset_index(drop=True)
    meta = {
        "selection_mode": "external_query_subset",
        "query_subset_csv": str(subset_path),
        "query_rows": int(len(query_df)),
    }
    return query_df, meta


def _select_subject_query_pool(df: pd.DataFrame, *, split_key: str, query_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if split_key not in df.columns:
        raise KeyError(f"Missing split key '{split_key}' in metadata; cannot do subject split.")

    split_series = df[split_key].astype(str).fillna("").str.strip()
    valid_subjects = sorted({value for value in split_series.tolist() if value and value.lower() not in {"nan", "none"}})
    if not valid_subjects:
        raise RuntimeError(f"No valid values found in split key '{split_key}'.")

    rng = np.random.default_rng(seed)
    rng.shuffle(valid_subjects)

    selected_subjects: list[str] = []
    pooled_rows = 0
    subject_to_count = split_series.value_counts().to_dict()
    for subject in valid_subjects:
        selected_subjects.append(subject)
        pooled_rows += int(subject_to_count[subject])
        if pooled_rows >= query_size:
            break

    query_mask = split_series.isin(selected_subjects)
    query_pool = df[query_mask].copy()
    gallery_df = df[~query_mask].copy()
    if len(query_pool) < query_size:
        raise RuntimeError(
            f"Subject-based query pool has only {len(query_pool)} rows, smaller than requested query_size={query_size}."
        )

    query_indices = rng.choice(query_pool.index.to_numpy(), size=query_size, replace=False)
    query_df = query_pool.loc[sorted(query_indices)].copy().reset_index(drop=True)
    gallery_df = gallery_df.reset_index(drop=True)
    meta = {
        "split_by": split_key,
        "query_pool_subject_count": len(selected_subjects),
        "query_pool_rows": int(len(query_pool)),
        "query_rows": int(len(query_df)),
        "gallery_rows": int(len(gallery_df)),
    }
    return query_df, gallery_df, meta


def main() -> None:
    args = parse_args()
    total_start = time.time()
    config = default_report_config()
    run_dir = _resolve_run_dir(config, args.run_name, args.run_dir)
    subsets_dir = run_dir / "subsets"
    logs_dir = run_dir / "logs"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Run] prepare MIMIC subsets start -> {run_dir}")
    mimic_df = _load_mimic_df(config)

    if args.query_subset_csv:
        query_df, meta = _build_query_subset_from_csv(
            args.query_subset_csv,
            mimic_df,
            config=config,
            case_id=args.case_id,
        )
        split_key = config.dataset.split_key
        if split_key not in mimic_df.columns:
            raise KeyError(f"Missing split key '{split_key}' in metadata; cannot exclude same subject.")
        gallery_split_series = mimic_df[split_key].astype(str).fillna("").str.strip()
        query_subjects = set(query_df[split_key].astype(str).fillna("").str.strip().tolist())
        gallery_df = mimic_df.loc[~gallery_split_series.isin(query_subjects)].copy().reset_index(drop=True)
        meta["split_by"] = split_key
        meta["gallery_rows"] = int(len(gallery_df))
    elif args.case_id:
        query_df = mimic_df.loc[mimic_df["case_id"].astype(str) == args.case_id].copy()
        if query_df.empty:
            raise KeyError(f"case_id not found in MIMIC metadata: {args.case_id}")
        split_key = config.dataset.split_key
        if split_key not in mimic_df.columns:
            raise KeyError(f"Missing split key '{split_key}' in metadata; cannot exclude same subject.")
        gallery_split_series = mimic_df[split_key].astype(str).fillna("").str.strip()
        query_subjects = set(query_df[split_key].astype(str).fillna("").str.strip().tolist())
        gallery_df = mimic_df.loc[~gallery_split_series.isin(query_subjects)].copy().reset_index(drop=True)
        query_df = query_df.reset_index(drop=True)
        meta = {
            "selection_mode": "single_case",
            "split_by": split_key,
            "query_rows": int(len(query_df)),
            "gallery_rows": int(len(gallery_df)),
        }
    else:
        query_df, gallery_df, meta = _select_subject_query_pool(
            mimic_df,
            split_key=config.dataset.split_key,
            query_size=args.query_size,
            seed=args.seed,
        )
        meta["selection_mode"] = "subject_split_1000_query"

    query_df.to_csv(subsets_dir / "query_subset.csv", index=False)
    gallery_df.to_csv(subsets_dir / "gallery_subset.csv", index=False)
    write_json(
        subsets_dir / "gallery_source.json",
        {
            "mode": "mimic_self_rag_full_gallery",
            "metadata_path": config.dataset.train_metadata_path,
            "embedding_path": config.dataset.train_embeddings_path,
            "label_column": config.dataset.label_column,
            "feature_root": config.dataset.feature_root,
            **meta,
        },
    )

    summary = {
        "run_dir": str(run_dir),
        "label_column": config.dataset.label_column,
        "split_key": config.dataset.split_key,
        "feature_match_key": config.dataset.feature_match_key,
        "feature_root": config.dataset.feature_root,
        "elapsed_seconds": round(time.time() - total_start, 2),
        **meta,
    }
    write_json(logs_dir / "subset_prep_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
