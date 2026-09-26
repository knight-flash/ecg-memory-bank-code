#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from ecg_memory_bank.protocol import external_output_dir, validate_broad_addition_protocol, validate_record_ids

from ecg_memory_bank.retrieval.artifact_utils import (
    build_anchored_nested_gallery_order,
    build_hierarchical_ndcg_arrays,
    build_query_precision_hit_arrays,
    build_union_recall_arrays,
    format_fraction_label,
    format_fraction_tag,
    maybe_limit_queries,
    select_gallery_indices,
    save_ci_artifacts,
)
from ecg_memory_bank.retrieval.label_mapping_utils import load_label_hierarchy
from ecg_memory_bank.retrieval.metrics_utils import (
    align_metadata_and_vectors,
    build_faiss_index,
    build_multihot_from_df,
    build_query_level_artifacts,
    build_retrieval_voting_scores,
    calculate_metrics_multihot,
    calculate_retrieval_voting_auc_from_scores,
    choose_label_column,
    load_dataset_vectors,
    load_metadata_df,
    safe_mean,
    to_jsonable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LABEL_MAPPING = Path(__file__).resolve().parents[3] / "configs" / "label_category_counts_9class.csv"
DEFAULT_GALLERY_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0)
DEFAULT_REPORT_KS = (1, 3, 5, 10)


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Finaltabel HEEDB scale benchmark on filtered finalgallery data."
    )
    parser.add_argument("--gallery-fractions", nargs="+", type=float, default=list(DEFAULT_GALLERY_FRACTIONS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        choices=("anchored_nested",),
        default="anchored_nested",
    )
    parser.add_argument(
        "--anchor-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.3, 0.5, 1.0],
        help="Audited anchor fractions; final-paper values are fixed.",
    )
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-id-col", required=True)
    parser.add_argument("--heedb-train-ecg-path", default=env_or("HEEDB_TRAIN_ECG_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_ecg_embeddings.pt"))
    parser.add_argument("--heedb-train-meta-path", default=env_or("HEEDB_TRAIN_META_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/train/heedb_texts_with_labels_correct.csv"))
    parser.add_argument("--heedb-test-ecg-path", default=env_or("HEEDB_TEST_ECG_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_ecg_embeddings.pt"))
    parser.add_argument("--heedb-test-meta-path", default=env_or("HEEDB_TEST_META_PATH", "__ECG_DATA_ROOT_REQUIRED__/finalgallery/test/heedb_metadata_with_labels_correct.csv"))
    parser.add_argument("--label-col", default=env_or("RETRIEVAL_LABEL_COL", "labels_correct"))
    parser.add_argument("--heedb-label-mapping-csv", default=env_or("HEEDB_LABEL_MAPPING_CSV", str(DEFAULT_LABEL_MAPPING)))
    args = parser.parse_args()
    validate_broad_addition_protocol(
        gallery_fractions=args.gallery_fractions,
        top_k=args.top_k,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        anchor_fractions=args.anchor_fractions,
        sampling_order_path="",
        max_queries=args.max_queries,
    )
    return args


def build_query_metadata_rows(query_labels: list[dict[str, list[str]]]) -> dict[str, np.ndarray]:
    return {
        "l1_positive_count": np.asarray([len(item["l1"]) for item in query_labels], dtype=np.int32),
        "l2_positive_count": np.asarray([len(item["l2"]) for item in query_labels], dtype=np.int32),
        "l3_positive_count": np.asarray([len(item["l3"]) for item in query_labels], dtype=np.int32),
    }


def resolve_report_ks(top_k: int) -> list[int]:
    return [value for value in DEFAULT_REPORT_KS if value <= int(top_k)]


def evaluate_fraction(
    *,
    fraction: float,
    args: argparse.Namespace,
    output_dir: Path,
    full_gallery_vecs: np.ndarray,
    full_gallery_df: pd.DataFrame,
    gallery_col: str,
    query_vecs: np.ndarray,
    query_df: pd.DataFrame,
    query_col: str,
    label_hierarchy: dict[str, tuple[str, str]],
    selection_order: np.ndarray,
    sampling_audit_path: str,
) -> dict[str, Any]:
    fraction_tag = format_fraction_tag(fraction)
    fraction_label = format_fraction_label(fraction)
    scenario_name = f"heedb_train_gallery_{fraction_tag}_finalgallery_test"
    scenario_output_dir = output_dir / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    report_ks = resolve_report_ks(args.top_k)

    selected_indices = select_gallery_indices(
        total=len(full_gallery_df),
        fraction=fraction,
        seed=args.seed,
        selection_order=selection_order,
    )
    gallery_vecs = full_gallery_vecs[selected_indices]
    gallery_df = full_gallery_df.iloc[selected_indices].reset_index(drop=True)
    selected_indices_path = scenario_output_dir / f"{scenario_name}_gallery_indices.npy"
    np.save(selected_indices_path, selected_indices)

    l1_db, l2_db, l3_db, ref_classes, db_labels = build_multihot_from_df(gallery_df, gallery_col, label_hierarchy)
    l1_query, l2_query, l3_query, _, query_labels = build_multihot_from_df(
        query_df, query_col, label_hierarchy, reference_classes=ref_classes
    )
    query_vecs, query_df, l1_query, l2_query, l3_query, query_labels = maybe_limit_queries(
        query_vecs,
        query_df,
        l1_query,
        l2_query,
        l3_query,
        query_labels,
        args.max_queries,
    )

    index = build_faiss_index(gallery_vecs)
    all_indices = []
    for start in tqdm(range(0, query_vecs.shape[0], args.batch_size), desc=f"Retrieving {fraction_tag}"):
        _, indices = index.search(query_vecs[start : start + args.batch_size], args.top_k)
        all_indices.append(indices)
    retrieved_indices = np.concatenate(all_indices, axis=0)

    per_query_df, query_metric_arrays = build_query_level_artifacts(
        retrieved_indices=retrieved_indices,
        db_labels=db_labels,
        query_labels=query_labels,
        top_k=args.top_k,
    )
    query_metric_arrays.update(build_union_recall_arrays(retrieved_indices, db_labels, query_labels, report_ks))
    query_metric_arrays.update(build_hierarchical_ndcg_arrays(retrieved_indices, db_labels, query_labels, report_ks))
    query_metric_arrays.update(build_query_metadata_rows(query_labels))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l1_db, l1_query, report_ks, "l1"))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l2_db, l2_query, report_ks, "l2"))
    query_metric_arrays.update(build_query_precision_hit_arrays(retrieved_indices, l3_db, l3_query, report_ks, "l3"))

    for report_k in report_ks:
        per_query_df[f"union_recall_l3@{report_k}"] = query_metric_arrays[f"union_recall_l3_at_{report_k}"]
        per_query_df[f"hierarchical_ndcg@{report_k}"] = query_metric_arrays[f"hierarchical_ndcg_at_{report_k}"]
        per_query_df[f"l1_p@{report_k}"] = query_metric_arrays[f"l1_p_at_{report_k}"]
        per_query_df[f"l2_p@{report_k}"] = query_metric_arrays[f"l2_p_at_{report_k}"]
        per_query_df[f"l3_p@{report_k}"] = query_metric_arrays[f"l3_p_at_{report_k}"]
        per_query_df[f"l1_hit@{report_k}"] = query_metric_arrays[f"l1_hit_at_{report_k}"]
        per_query_df[f"l2_hit@{report_k}"] = query_metric_arrays[f"l2_hit_at_{report_k}"]
        per_query_df[f"l3_hit@{report_k}"] = query_metric_arrays[f"l3_hit_at_{report_k}"]
    per_query_df["l1_positive_count"] = query_metric_arrays["l1_positive_count"]
    per_query_df["l2_positive_count"] = query_metric_arrays["l2_positive_count"]
    per_query_df["l3_positive_count"] = query_metric_arrays["l3_positive_count"]

    per_query_path = scenario_output_dir / f"{scenario_name}_per_query.csv"
    per_query_df.to_csv(per_query_path, index=False)

    voting_results = {}
    retrieval_hits = {}
    retrieval_precision = {}
    per_class_paths: dict[str, str] = {}
    auc_inputs = {}
    for level_name, db_mat, query_mat, class_names in [
        ("L1", l1_db, l1_query, ref_classes["l1"]),
        ("L2", l2_db, l2_query, ref_classes["l2"]),
        ("L3", l3_db, l3_query, ref_classes["l3"]),
    ]:
        retrieval_metrics, _ = calculate_metrics_multihot(
            retrieved_indices,
            db_mat,
            query_mat,
            k_list=report_ks,
            class_names=class_names,
        )
        retrieval_hits[level_name] = {f"Hit@{report_k}": retrieval_metrics.get(f"Hit@{report_k}", float("nan")) for report_k in report_ks}
        retrieval_precision[level_name] = {f"P@{report_k}": retrieval_metrics.get(f"P@{report_k}", float("nan")) for report_k in report_ks}

        y_score = build_retrieval_voting_scores(retrieved_indices, db_mat, top_k=args.top_k)
        auc_summary, auc_df = calculate_retrieval_voting_auc_from_scores(query_mat, y_score, class_names, args.top_k)
        voting_results[level_name] = auc_summary
        auc_inputs[level_name.lower()] = {
            "y_true": query_mat,
            "y_score": y_score,
            "class_names": class_names,
        }

        auc_path = scenario_output_dir / f"per_class_auc_{scenario_name}_{level_name}.csv"
        auc_df.to_csv(auc_path, index=False)
        per_class_paths[level_name] = str(auc_path)

    summary = {
        "scenario": scenario_name,
        "gallery_fraction": float(fraction),
        "gallery_fraction_label": fraction_label,
        "top_k": int(args.top_k),
        "num_gallery": int(len(gallery_df)),
        "num_queries": int(len(query_df)),
        "sampling_scheme": "anchored_nested_fixed_seed_prefix",
        "sampling_seed": int(args.seed),
        "sampling_audit_path": sampling_audit_path,
        f"mean_hierarchical_ndcg@{args.top_k}": safe_mean(query_metric_arrays["hierarchical_ndcg"]),
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

    summary_path = scenario_output_dir / f"{scenario_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)
    summary["paths"]["summary_json"] = str(summary_path)

    save_ci_artifacts(
        scenario_output_dir=scenario_output_dir,
        family="scale_benchmark",
        scenario_name=scenario_name,
        summary_path=summary_path,
        query_metric_arrays=query_metric_arrays,
        auc_inputs=auc_inputs,
        extra_manifest={
            "top_k": int(args.top_k),
            "gallery_fraction": float(fraction),
            "gallery_fraction_label": fraction_label,
            "sampling_scheme": "anchored_nested_fixed_seed_prefix",
            "sampling_seed": int(args.seed),
            "sampling_audit_path": sampling_audit_path,
            "gallery_indices_npy": str(selected_indices_path),
            "per_query_csv": str(per_query_path),
            "per_class_auc_csv": per_class_paths,
        },
    )
    return summary


def flatten_summary(summary: dict[str, Any], top_k: int) -> dict[str, Any]:
    report_ks = resolve_report_ks(top_k)
    row = {
        "scenario": summary["scenario"],
        "gallery_fraction": summary["gallery_fraction"],
        "gallery_fraction_label": summary["gallery_fraction_label"],
        "num_gallery": summary["num_gallery"],
        "num_queries": summary["num_queries"],
        "sampling_scheme": summary.get("sampling_scheme", ""),
        "sampling_seed": summary.get("sampling_seed"),
    }
    for report_k in report_ks:
        row[f"mean_union_recall_l3@{report_k}"] = summary.get(f"mean_union_recall_l3@{report_k}", float("nan"))
        row[f"mean_hierarchical_ndcg@{report_k}"] = summary.get(f"mean_hierarchical_ndcg@{report_k}", float("nan"))
    for level_name, level_summary in summary["voting_auc"].items():
        prefix = level_name.lower()
        row[f"{prefix}_macro_auc"] = level_summary["macro_auc"]
        row[f"{prefix}_micro_auc"] = level_summary["micro_auc"]
        row[f"{prefix}_valid_class_count"] = level_summary["valid_class_count"]
        row[f"{prefix}_skipped_class_count"] = level_summary["skipped_class_count"]
    for level_name, level_hits in summary["retrieval_hits"].items():
        prefix = level_name.lower()
        for report_k in report_ks:
            row[f"{prefix}_Hit@{report_k}"] = level_hits.get(f"Hit@{report_k}", float("nan"))
    for level_name, level_precision in summary["retrieval_precision"].items():
        prefix = level_name.lower()
        for report_k in report_ks:
            row[f"{prefix}_P@{report_k}"] = level_precision.get(f"P@{report_k}", float("nan"))
    return row


def main() -> None:
    args = parse_args()
    if args.top_k < 5:
        raise ValueError("top_k must be >= 5 for scale_benchmark.")

    output_dir = external_output_dir(args.output_dir) / "scale_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_hierarchy = load_label_hierarchy(args.heedb_label_mapping_csv)

    full_gallery_vecs = load_dataset_vectors(args.heedb_train_ecg_path)
    full_gallery_df = load_metadata_df(args.heedb_train_meta_path)
    full_gallery_df, full_gallery_vecs = align_metadata_and_vectors(full_gallery_df, full_gallery_vecs)
    validate_record_ids(full_gallery_df[args.record_id_col].tolist(), name="HEEDB memory")
    gallery_col = choose_label_column(full_gallery_df, args.label_col)

    selection_order, sampling_audit = build_anchored_nested_gallery_order(
        total=len(full_gallery_df),
        seed=args.seed,
        anchor_fractions=args.anchor_fractions,
    )
    order_path = output_dir / "heedb_anchored_nested_gallery_order.npy"
    audit_path = output_dir / "heedb_anchored_nested_sampling_audit.json"
    np.save(order_path, selection_order)
    sampling_audit["order_npy"] = str(order_path)
    audit_path.write_text(json.dumps(to_jsonable(sampling_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    sampling_audit_path = str(audit_path)

    query_vecs = load_dataset_vectors(args.heedb_test_ecg_path)
    query_df = load_metadata_df(args.heedb_test_meta_path)
    query_df, query_vecs = align_metadata_and_vectors(query_df, query_vecs)
    validate_record_ids(query_df[args.record_id_col].tolist(), name="HEEDB queries")
    query_col = choose_label_column(query_df, args.label_col)

    summaries = []
    for fraction in args.gallery_fractions:
        summaries.append(
            evaluate_fraction(
                fraction=float(fraction),
                args=args,
                output_dir=output_dir,
                full_gallery_vecs=full_gallery_vecs,
                full_gallery_df=full_gallery_df,
                gallery_col=gallery_col,
                query_vecs=query_vecs,
                query_df=query_df,
                query_col=query_col,
                label_hierarchy=label_hierarchy,
                selection_order=selection_order,
                sampling_audit_path=sampling_audit_path,
            )
        )

    summary_json_path = output_dir / "scale_benchmark_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summaries), handle, ensure_ascii=False, indent=2)

    summary_csv_path = output_dir / "scale_benchmark_summary.csv"
    pd.DataFrame([flatten_summary(summary, args.top_k) for summary in summaries]).to_csv(summary_csv_path, index=False)

    print(f"[Saved] JSON -> {summary_json_path}")
    print(f"[Saved] CSV  -> {summary_csv_path}")


if __name__ == "__main__":
    main()
