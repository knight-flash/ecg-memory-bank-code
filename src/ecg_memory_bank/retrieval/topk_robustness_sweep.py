#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ecg_memory_bank.retrieval.artifact_utils import (
    build_query_precision_hit_arrays,
    format_fraction_label,
    format_fraction_tag,
    maybe_limit_queries,
    sample_gallery_subset,
    save_ci_artifacts,
)
from ecg_memory_bank.retrieval.label_mapping_utils import load_label_hierarchy
from ecg_memory_bank.retrieval.metrics_utils import (
    align_metadata_and_vectors,
    build_faiss_index,
    build_multihot_from_df,
    build_retrieval_voting_scores,
    calculate_retrieval_voting_auc_from_scores,
    choose_label_column,
    load_dataset_vectors,
    load_metadata_df,
    to_jsonable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LABEL_MAPPING = Path(__file__).resolve().parents[3] / "configs" / "label_category_counts_9class.csv"
DEFAULT_TOP_K_LIST = (1, 3, 5, 10, 20, 50)
DEFAULT_GALLERY_FRACTIONS = (0.001, 0.01, 0.1, 1.0)


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Finaltabel top-k robustness sweeps on finalgallery.")
    parser.add_argument("--top-k-list", nargs="+", type=int, default=list(DEFAULT_TOP_K_LIST))
    parser.add_argument("--gallery-fractions", nargs="+", type=float, default=list(DEFAULT_GALLERY_FRACTIONS))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--heedb-train-ecg-path", default=env_or("HEEDB_TRAIN_ECG_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_ecg_embeddings.pt"))
    parser.add_argument("--heedb-train-meta-path", default=env_or("HEEDB_TRAIN_META_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_texts_with_labels_correct.csv"))
    parser.add_argument("--heedb-test-ecg-path", default=env_or("HEEDB_TEST_ECG_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_ecg_embeddings.pt"))
    parser.add_argument("--heedb-test-meta-path", default=env_or("HEEDB_TEST_META_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv"))
    parser.add_argument("--label-col", default=env_or("RETRIEVAL_LABEL_COL", "labels_correct"))
    parser.add_argument("--heedb-label-mapping-csv", default=env_or("HEEDB_LABEL_MAPPING_CSV", str(DEFAULT_LABEL_MAPPING)))
    return parser.parse_args()


def validate_top_k_list(top_k_list: list[int]) -> list[int]:
    cleaned = []
    seen = set()
    for value in top_k_list:
        if value <= 0:
            raise ValueError(f"top-k must be positive, got {value}")
        if value not in seen:
            cleaned.append(int(value))
            seen.add(int(value))
    return cleaned


def main() -> None:
    args = parse_args()
    top_k_list = validate_top_k_list(args.top_k_list)
    output_dir = Path(args.output_dir) / "topk_robustness_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_hierarchy = load_label_hierarchy(args.heedb_label_mapping_csv)
    full_gallery_vecs = load_dataset_vectors(args.heedb_train_ecg_path)
    full_gallery_df = load_metadata_df(args.heedb_train_meta_path)
    full_gallery_df, full_gallery_vecs = align_metadata_and_vectors(full_gallery_df, full_gallery_vecs)
    gallery_col = choose_label_column(full_gallery_df, args.label_col)

    query_vecs = load_dataset_vectors(args.heedb_test_ecg_path)
    query_df = load_metadata_df(args.heedb_test_meta_path)
    query_df, query_vecs = align_metadata_and_vectors(query_df, query_vecs)
    query_col = choose_label_column(query_df, args.label_col)

    all_rows = []
    figure_rows = []
    for fraction in args.gallery_fractions:
        fraction = float(fraction)
        run_name = f"heedb_train_gallery_{format_fraction_tag(fraction)}_topk_sweep"
        scenario_output_dir = output_dir / run_name
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        gallery_vecs, gallery_df = sample_gallery_subset(full_gallery_vecs, full_gallery_df, fraction, args.seed)

        l1_db, l2_db, l3_db, ref_classes, _ = build_multihot_from_df(gallery_df, gallery_col, label_hierarchy)
        l1_query, l2_query, l3_query, _, query_labels = build_multihot_from_df(
            query_df, query_col, label_hierarchy, reference_classes=ref_classes
        )
        query_vecs_use, query_df_use, l1_query, l2_query, l3_query, query_labels = maybe_limit_queries(
            query_vecs, query_df, l1_query, l2_query, l3_query, query_labels, args.max_queries
        )

        max_top_k = max(top_k_list)
        index = build_faiss_index(gallery_vecs)
        all_indices = []
        for start in tqdm(range(0, query_vecs_use.shape[0], args.batch_size), desc=f"Retrieving {run_name}"):
            _, indices = index.search(query_vecs_use[start : start + args.batch_size], max_top_k)
            all_indices.append(indices)
        retrieved_indices = np.concatenate(all_indices, axis=0)

        query_metric_arrays = {}
        auc_inputs = {}
        for level_name, db_mat, query_mat, class_names in [
            ("L1", l1_db, l1_query, ref_classes["l1"]),
            ("L2", l2_db, l2_query, ref_classes["l2"]),
            ("L3", l3_db, l3_query, ref_classes["l3"]),
        ]:
            query_metric_arrays.update(
                build_query_precision_hit_arrays(retrieved_indices, db_mat, query_mat, top_k_list, level_name.lower())
            )
            for top_k in top_k_list:
                y_score = build_retrieval_voting_scores(retrieved_indices, db_mat, top_k=top_k)
                auc_summary, _ = calculate_retrieval_voting_auc_from_scores(query_mat, y_score, class_names, top_k)
                metric_key = f"{level_name.lower()}_top{top_k}"
                auc_inputs[metric_key] = {
                    "y_true": query_mat,
                    "y_score": y_score,
                    "class_names": class_names,
                }
                row = {
                    "scenario": run_name,
                    "gallery_fraction": fraction,
                    "gallery_fraction_label": format_fraction_label(fraction),
                    "top_k": int(top_k),
                    "level": level_name,
                    "num_gallery": int(len(gallery_df)),
                    "num_queries": int(len(query_df_use)),
                    f"{level_name.lower()}_P@K": float(np.mean(query_metric_arrays[f"{level_name.lower()}_p_at_{top_k}"])),
                    f"{level_name.lower()}_Hit@K": float(np.mean(query_metric_arrays[f"{level_name.lower()}_hit_at_{top_k}"])),
                    "macro_auc": auc_summary["macro_auc"],
                    "micro_auc": auc_summary["micro_auc"],
                    "valid_class_count": auc_summary["valid_class_count"],
                    "skipped_class_count": auc_summary["skipped_class_count"],
                }
                all_rows.append(row)
                if level_name == "L3":
                    figure_rows.append(
                        {
                            "scenario": run_name,
                            "gallery_fraction": fraction,
                            "gallery_fraction_label": format_fraction_label(fraction),
                            "top_k": int(top_k),
                            "l3_P@K": row["l3_P@K"],
                            "l3_macro_auc": auc_summary["macro_auc"],
                        }
                    )

        summary_rows = [row for row in all_rows if row["scenario"] == run_name]
        summary_csv_path = scenario_output_dir / f"{run_name}_topk_robustness.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
        summary_payload = {
            "scenario": run_name,
            "gallery_fraction": fraction,
            "gallery_fraction_label": format_fraction_label(fraction),
            "top_k_list": top_k_list,
            "num_gallery": int(len(gallery_df)),
            "num_queries": int(len(query_df_use)),
            "rows": summary_rows,
            "paths": {
                "summary_csv": str(summary_csv_path),
            },
        }
        summary_json_path = scenario_output_dir / f"{run_name}_summary.json"
        summary_json_path.write_text(json.dumps(to_jsonable(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")

        save_ci_artifacts(
            scenario_output_dir=scenario_output_dir,
            family="topk_robustness_sweep",
            scenario_name=run_name,
            summary_path=summary_json_path,
            query_metric_arrays=query_metric_arrays,
            auc_inputs=auc_inputs,
            extra_manifest={
                "gallery_fraction": fraction,
                "gallery_fraction_label": format_fraction_label(fraction),
                "top_k_list": top_k_list,
            },
        )

    summary_csv = output_dir / "topk_robustness_sweep_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
    figure_csv = output_dir / "topk_robustness_sweep_figure_ready.csv"
    pd.DataFrame(figure_rows).to_csv(figure_csv, index=False)
    summary_json = output_dir / "topk_robustness_sweep_summary.json"
    summary_json.write_text(json.dumps(to_jsonable(all_rows), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Saved] Summary CSV -> {summary_csv}")
    print(f"[Saved] Figure CSV  -> {figure_csv}")
    print(f"[Saved] Summary JSON -> {summary_json}")


if __name__ == "__main__":
    main()
