#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CODE-15 Chagas targeted addition.

Purpose
-------
Measure the diagnostic effect of adding Chagas-positive historical ECGs to a
memory initially containing no positive Chagas reference.

Method
------
Create one record-level seeded query split, add positive cases cumulatively,
rebuild the index, and compare with a matched-size negative-case control.

Inputs
------
CODE-15 embedding tensor and aligned metadata with record ID and binary chagas
indicator.

Outputs
-------
External split manifest, stage-wise retrieved neighbors, binary AUROC,
Recall@K, and paired bootstrap summaries.

Paper parameters
----------------
Query counts are 1000 positive and 4000 negative; additions are 0, 50, 100,
300, 800, 2000, and 5000; K=10, seed=42, bootstrap=1000.

Shapes and meaning
------------------
Embeddings are [N,D]; query labels are [5000] binary indicators. Each Chagas
score is the mean positive indicator among the retrieved K cases.
"""


from __future__ import annotations

import argparse
import json
import os
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

from ecg_memory_bank.retrieval.artifact_utils import save_ci_artifacts  # noqa: E402
from ecg_memory_bank.retrieval.eval_retrieval_ci_fast import (  # noqa: E402
    bootstrap_binary_auc_ci,
    bootstrap_index_matrix,
    bootstrap_mean_ci,
)
from ecg_memory_bank.retrieval.metrics_utils import (  # noqa: E402
    align_metadata_and_vectors,
    build_faiss_index,
    calculate_retrieval_voting_auc_from_scores,
    load_dataset_vectors,
    load_metadata_df,
    safe_mean,
    to_jsonable,
)


DEFAULT_OUTPUT_ROOT = Path("__ECG_DATA_ROOT_REQUIRED__/Finaltabel/Retrieval/outputs")
DEFAULT_STAGE_SIZES = (0, 50, 100, 300, 800, 2000, 5000)
DEFAULT_CODE15_ECG_PATH = "__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/code15_ecgclip_embeddings.pt"
DEFAULT_CODE15_META_PATH = "__ECG_DATA_ROOT_REQUIRED__/ecofounder2_ex/code15/metadata_code15_ecgclip.csv"
DEFAULT_REPORT_KS = (1, 3, 5, 10)


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight CODE15 Chagas cold-start retrieval experiment with case-level random splits."
    )
    parser.add_argument("--code15-ecg-path", default=env_or("CODE15_ECG_PATH", DEFAULT_CODE15_ECG_PATH))
    parser.add_argument("--code15-meta-path", default=env_or("CODE15_META_PATH", DEFAULT_CODE15_META_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-sizes", nargs="+", type=int, default=list(DEFAULT_STAGE_SIZES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-test-size", type=int, default=1000)
    parser.add_argument("--negative-test-size", type=int, default=4000)
    parser.add_argument("--chagas-col", default="chagas")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def validate_stage_sizes(stage_sizes: list[int]) -> list[int]:
    if not stage_sizes:
        raise ValueError("stage_sizes cannot be empty.")
    cleaned = [int(value) for value in stage_sizes]
    if any(value < 0 for value in cleaned):
        raise ValueError(f"stage_sizes must be non-negative, got {cleaned}")
    for previous, current in zip(cleaned, cleaned[1:]):
        if current <= previous:
            raise ValueError(f"stage_sizes must be strictly increasing, got {cleaned}")
    return cleaned


def resolve_report_ks(top_k: int) -> list[int]:
    return [value for value in DEFAULT_REPORT_KS if value <= int(top_k)]


def coerce_binary_series(series: pd.Series, column_name: str) -> np.ndarray:
    values = []
    true_tokens = {"1", "1.0", "true", "t", "yes", "y", "positive", "pos"}
    false_tokens = {"0", "0.0", "false", "f", "no", "n", "negative", "neg", ""}
    for raw_value in series.tolist():
        if pd.isna(raw_value):
            values.append(0)
            continue
        if isinstance(raw_value, (bool, np.bool_)):
            values.append(int(bool(raw_value)))
            continue
        if isinstance(raw_value, (int, np.integer)):
            values.append(1 if int(raw_value) != 0 else 0)
            continue
        if isinstance(raw_value, (float, np.floating)):
            values.append(1 if float(raw_value) != 0.0 else 0)
            continue
        token = str(raw_value).strip().lower()
        if token in true_tokens:
            values.append(1)
            continue
        if token in false_tokens:
            values.append(0)
            continue
        try:
            numeric = float(token)
            values.append(1 if numeric != 0.0 else 0)
            continue
        except ValueError as exc:
            raise ValueError(f"Unable to parse binary value '{raw_value}' from column '{column_name}'.") from exc
    return np.asarray(values, dtype=np.uint8)


def prepare_case_level_splits(
    *,
    code15_df: pd.DataFrame,
    code15_vecs: np.ndarray,
    chagas_col: str,
    positive_test_size: int,
    negative_test_size: int,
    stage_sizes: list[int],
    seed: int,
) -> dict[str, Any]:
    if chagas_col not in code15_df.columns:
        raise KeyError(f"Chagas column '{chagas_col}' was not found in CODE15 metadata.")

    chagas_flags = coerce_binary_series(code15_df[chagas_col], chagas_col)
    positive_indices = np.flatnonzero(chagas_flags == 1)
    negative_indices = np.flatnonzero(chagas_flags == 0)
    if len(positive_indices) == 0:
        raise ValueError(f"No positive rows found in CODE15 metadata using column '{chagas_col}'.")
    if len(negative_indices) == 0:
        raise ValueError(f"No negative rows found in CODE15 metadata using column '{chagas_col}'.")

    rng = np.random.default_rng(seed)
    positive_indices = positive_indices.copy()
    negative_indices = negative_indices.copy()
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    if len(positive_indices) < positive_test_size + stage_sizes[-1]:
        raise ValueError(
            f"Not enough Chagas-positive rows: need at least {positive_test_size + stage_sizes[-1]}, "
            f"but only {len(positive_indices)} are available."
        )
    if len(negative_indices) < negative_test_size + 1:
        raise ValueError(
            f"Not enough Chagas-negative rows: need at least {negative_test_size + 1}, "
            f"but only {len(negative_indices)} are available."
        )

    positive_holdout_idx = np.sort(positive_indices[:positive_test_size])
    injection_pool_idx = positive_indices[positive_test_size:]
    negative_holdout_idx = np.sort(negative_indices[:negative_test_size])
    base_gallery_idx = negative_indices[negative_test_size:]

    query_indices = np.concatenate([positive_holdout_idx, negative_holdout_idx], axis=0)
    rng.shuffle(query_indices)

    query_vecs = code15_vecs[query_indices]
    query_is_chagas = chagas_flags[query_indices].astype(np.uint8, copy=False)

    split_manifest = {
        "seed": int(seed),
        "split_type": "case_level_random",
        "chagas_col": chagas_col,
        "stage_sizes": [int(value) for value in stage_sizes],
        "targets": {
            "positive_test_size": int(positive_test_size),
            "negative_test_size": int(negative_test_size),
            "max_injection_size": int(stage_sizes[-1]),
        },
        "actual": {
            "total_code15_rows": int(len(code15_df)),
            "total_positive_rows": int(len(positive_indices)),
            "total_negative_rows": int(len(negative_indices)),
            "positive_holdout_count": int(len(positive_holdout_idx)),
            "negative_holdout_count": int(len(negative_holdout_idx)),
            "base_gallery_count": int(len(base_gallery_idx)),
            "injection_pool_count": int(len(injection_pool_idx)),
            "query_count": int(len(query_indices)),
        },
    }

    return {
        "query_vecs": query_vecs,
        "query_is_chagas": query_is_chagas,
        "base_gallery_idx": base_gallery_idx,
        "injection_pool_idx": injection_pool_idx,
        "split_manifest": split_manifest,
    }


def run_retrieval(
    *,
    index: Any,
    gallery_size: int,
    query_vecs: np.ndarray,
    top_k: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    all_indices = []
    effective_top_k = min(int(top_k), int(gallery_size))
    if effective_top_k <= 0:
        raise ValueError("Gallery is empty; unable to run retrieval.")
    print(f"[Retrieval] {desc}: gallery_size={gallery_size}, top_k={effective_top_k}, num_queries={len(query_vecs)}")
    for start in tqdm(range(0, query_vecs.shape[0], batch_size), desc=desc):
        _, indices = index.search(query_vecs[start : start + batch_size], effective_top_k)
        all_indices.append(indices)
    return np.concatenate(all_indices, axis=0)


def initialize_gallery_state(base_gallery_vecs: np.ndarray) -> dict[str, Any]:
    print(f"[Index] Building baseline CODE15 non-Chagas index with {len(base_gallery_vecs)} gallery vectors...")
    index = build_faiss_index(base_gallery_vecs)
    return {
        "index": index,
        "base_gallery_size": int(len(base_gallery_vecs)),
        "gallery_size": int(len(base_gallery_vecs)),
        "gallery_is_chagas": np.zeros(len(base_gallery_vecs), dtype=np.uint8),
        "injected_count": 0,
        "fallback_gallery_vecs": base_gallery_vecs,
    }


def extend_gallery_state(
    *,
    state: dict[str, Any],
    target_injected_count: int,
    injection_pool_idx: np.ndarray,
    code15_vecs: np.ndarray,
) -> None:
    current_injected_count = int(state["injected_count"])
    if target_injected_count < current_injected_count:
        raise ValueError(
            f"target_injected_count must be >= current_injected_count, got {target_injected_count} < {current_injected_count}."
        )
    if target_injected_count == current_injected_count:
        return

    delta_indices = np.asarray(injection_pool_idx[current_injected_count:target_injected_count], dtype=np.int64)
    delta_vecs = code15_vecs[delta_indices]
    delta_count = int(len(delta_vecs))
    print(
        f"[Index] Extending gallery from {current_injected_count} to {target_injected_count} injected Chagas rows "
        f"(adding {delta_count})."
    )

    if hasattr(state["index"], "add"):
        state["index"].add(np.ascontiguousarray(delta_vecs.astype(np.float32, copy=False)))
    else:
        fallback_gallery_vecs = np.concatenate([state["fallback_gallery_vecs"], delta_vecs], axis=0)
        state["fallback_gallery_vecs"] = fallback_gallery_vecs
        state["index"] = build_faiss_index(fallback_gallery_vecs)

    state["gallery_is_chagas"] = np.concatenate(
        [state["gallery_is_chagas"], np.ones(delta_count, dtype=np.uint8)],
        axis=0,
    )
    state["gallery_size"] = int(len(state["gallery_is_chagas"]))
    state["injected_count"] = int(target_injected_count)


def compute_positive_recall_at_k(
    neighbor_is_chagas: np.ndarray,
    positive_query_mask: np.ndarray,
    top_k: int,
) -> float:
    if not positive_query_mask.any():
        return float("nan")
    return safe_mean(
        neighbor_is_chagas[positive_query_mask, :top_k].any(axis=1).astype(np.float32)
    )


def build_ci_summary(
    *,
    query_is_chagas: np.ndarray,
    chagas_scores: np.ndarray,
    hit_arrays: dict[int, np.ndarray],
    report_ks: list[int],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    positive_query_mask = query_is_chagas.astype(bool, copy=False)
    positive_count = int(positive_query_mask.sum())
    ci_summary: dict[str, Any] = {}

    if positive_count > 0 and bootstrap > 0:
        positive_index_matrix = bootstrap_index_matrix(positive_count, int(bootstrap), int(seed))
        for report_k in report_ks:
            point_estimate, lower, upper, valid = bootstrap_mean_ci(
                hit_arrays[report_k][positive_query_mask].astype(np.float32, copy=False),
                positive_index_matrix,
            )
            ci_summary[f"R@{report_k}"] = {
                "point_estimate": point_estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "valid_bootstraps": valid,
            }
    else:
        for report_k in report_ks:
            ci_summary[f"R@{report_k}"] = {
                "point_estimate": float("nan"),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
                "valid_bootstraps": 0,
            }

    auc_point, auc_lower, auc_upper, auc_valid = bootstrap_binary_auc_ci(
        query_is_chagas.astype(np.uint8, copy=False),
        chagas_scores.astype(np.float32, copy=False),
        bootstrap_index_matrix(len(query_is_chagas), int(bootstrap), int(seed)),
    )
    ci_summary["AUC"] = {
        "point_estimate": auc_point,
        "ci95_lower": auc_lower,
        "ci95_upper": auc_upper,
        "valid_bootstraps": auc_valid,
    }
    return ci_summary


def build_stage_per_query_df(
    *,
    query_is_chagas: np.ndarray,
    chagas_scores: np.ndarray,
    hit_arrays: dict[int, np.ndarray],
    stage_name: str,
    stage_chagas_count: int,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "query_index": np.arange(len(query_is_chagas), dtype=np.int32),
        "query_is_chagas": query_is_chagas.astype(np.uint8, copy=False),
        "stage_name": stage_name,
        "stage_chagas_count": int(stage_chagas_count),
        "chagas_score@10": chagas_scores.astype(np.float32, copy=False),
    }
    for top_k, hit_values in hit_arrays.items():
        payload[f"hit@{top_k}"] = hit_values.astype(np.uint8, copy=False)
    return pd.DataFrame(payload)


def evaluate_stage(
    *,
    family_output_dir: Path,
    stage_index: int,
    stage_size: int,
    gallery_state: dict[str, Any],
    query_is_chagas: np.ndarray,
    query_vecs: np.ndarray,
    top_k: int,
    batch_size: int,
    report_ks: list[int],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    stage_name = f"t{stage_index}_chagas_{stage_size}"
    stage_output_dir = family_output_dir / stage_name
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    gallery_is_chagas = gallery_state["gallery_is_chagas"]
    retrieved_indices = run_retrieval(
        index=gallery_state["index"],
        gallery_size=gallery_state["gallery_size"],
        query_vecs=query_vecs,
        top_k=top_k,
        batch_size=batch_size,
        desc=f"Retrieving {stage_name}",
    )

    neighbor_is_chagas = gallery_is_chagas[retrieved_indices].astype(np.float32, copy=False)
    effective_top_k = int(neighbor_is_chagas.shape[1])
    chagas_scores = neighbor_is_chagas[:, :effective_top_k].mean(axis=1).astype(np.float32, copy=False)
    positive_query_mask = query_is_chagas.astype(bool, copy=False)
    negative_query_mask = ~positive_query_mask

    hit_arrays: dict[int, np.ndarray] = {}
    recall_summary: dict[str, float] = {}
    for report_k in report_ks:
        hit_values = neighbor_is_chagas[:, :report_k].any(axis=1).astype(np.uint8, copy=False)
        hit_arrays[report_k] = hit_values
        recall_summary[f"R@{report_k}"] = compute_positive_recall_at_k(neighbor_is_chagas, positive_query_mask, report_k)

    y_true = query_is_chagas.reshape(-1, 1).astype(np.uint8, copy=False)
    y_score = chagas_scores.reshape(-1, 1).astype(np.float32, copy=False)
    auc_summary, auc_df = calculate_retrieval_voting_auc_from_scores(y_true, y_score, ["chagas"], effective_top_k)
    ci_summary = build_ci_summary(
        query_is_chagas=query_is_chagas,
        chagas_scores=chagas_scores,
        hit_arrays=hit_arrays,
        report_ks=report_ks,
        bootstrap=bootstrap,
        seed=seed + stage_index,
    )

    per_query_df = build_stage_per_query_df(
        query_is_chagas=query_is_chagas,
        chagas_scores=chagas_scores,
        hit_arrays=hit_arrays,
        stage_name=stage_name,
        stage_chagas_count=stage_size,
    )
    per_query_path = stage_output_dir / f"{stage_name}_per_query.csv"
    per_query_df.to_csv(per_query_path, index=False)

    auc_path = stage_output_dir / f"per_class_auc_{stage_name}_binary.csv"
    auc_df.to_csv(auc_path, index=False)

    summary = {
        "scenario": stage_name,
        "stage_index": int(stage_index),
        "stage_chagas_count": int(stage_size),
        "top_k_requested": int(top_k),
        "top_k_effective": int(effective_top_k),
        "num_base_gallery": int(gallery_state["base_gallery_size"]),
        "num_injected_chagas": int(stage_size),
        "num_gallery": int(gallery_state["gallery_size"]),
        "num_queries": int(len(query_vecs)),
        "num_positive_queries": int(positive_query_mask.sum()),
        "num_negative_queries": int(negative_query_mask.sum()),
        "retrieval_voting_auroc": auc_summary["macro_auc"],
        "recall_at_k": recall_summary,
        "ci95": ci_summary,
        "mean_chagas_score_positive_queries": safe_mean(chagas_scores[positive_query_mask]),
        "mean_chagas_score_negative_queries": safe_mean(chagas_scores[negative_query_mask]),
        "voting_auc": auc_summary,
        "paths": {
            "scenario_output_dir": str(stage_output_dir),
            "per_query_csv": str(per_query_path),
            "per_class_auc_csv": str(auc_path),
        },
    }
    summary_path = stage_output_dir / f"{stage_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)
    summary["paths"]["summary_json"] = str(summary_path)

    query_metric_arrays: dict[str, np.ndarray] = {
        "query_index": np.arange(len(query_is_chagas), dtype=np.int32),
        "query_is_chagas": query_is_chagas.astype(np.uint8, copy=False),
        "chagas_score_at_k": chagas_scores.astype(np.float32, copy=False),
        "positive_query_mask": positive_query_mask.astype(np.uint8, copy=False),
    }
    for report_k, hit_values in hit_arrays.items():
        query_metric_arrays[f"hit_at_{report_k}"] = hit_values.astype(np.uint8, copy=False)

    save_ci_artifacts(
        scenario_output_dir=stage_output_dir,
        family="knowledge_editing_cold_start_chagas",
        scenario_name=stage_name,
        summary_path=summary_path,
        query_metric_arrays=query_metric_arrays,
        auc_inputs={
            "binary": {
                "y_true": y_true,
                "y_score": y_score,
                "class_names": ["chagas"],
            }
        },
        extra_manifest={
            "top_k_requested": int(top_k),
            "top_k_effective": int(effective_top_k),
            "stage_index": int(stage_index),
            "stage_chagas_count": int(stage_size),
            "report_ks": [int(value) for value in report_ks],
            "per_query_csv": str(per_query_path),
            "per_class_auc_csv": {"binary": str(auc_path)},
        },
    )
    return summary


def flatten_stage_summary(summary: dict[str, Any], report_ks: list[int]) -> dict[str, Any]:
    row = {
        "scenario": summary["scenario"],
        "stage_index": summary["stage_index"],
        "stage_chagas_count": summary["stage_chagas_count"],
        "num_base_gallery": summary["num_base_gallery"],
        "num_gallery": summary["num_gallery"],
        "num_queries": summary["num_queries"],
        "num_positive_queries": summary["num_positive_queries"],
        "num_negative_queries": summary["num_negative_queries"],
        "retrieval_voting_auroc": summary["retrieval_voting_auroc"],
        "top_k_requested": summary["top_k_requested"],
        "top_k_effective": summary["top_k_effective"],
        "binary_macro_auc": summary["voting_auc"]["macro_auc"],
        "binary_micro_auc": summary["voting_auc"]["micro_auc"],
        "binary_valid_class_count": summary["voting_auc"]["valid_class_count"],
        "binary_skipped_class_count": summary["voting_auc"]["skipped_class_count"],
        "auc_ci95_lower": summary["ci95"]["AUC"]["ci95_lower"],
        "auc_ci95_upper": summary["ci95"]["AUC"]["ci95_upper"],
        "auc_valid_bootstraps": summary["ci95"]["AUC"]["valid_bootstraps"],
    }
    for report_k in report_ks:
        row[f"R@{report_k}"] = summary["recall_at_k"].get(f"R@{report_k}", float("nan"))
        row[f"R@{report_k}_ci95_lower"] = summary["ci95"][f"R@{report_k}"]["ci95_lower"]
        row[f"R@{report_k}_ci95_upper"] = summary["ci95"][f"R@{report_k}"]["ci95_upper"]
        row[f"R@{report_k}_valid_bootstraps"] = summary["ci95"][f"R@{report_k}"]["valid_bootstraps"]
    return row


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.positive_test_size <= 0 or args.negative_test_size <= 0:
        raise ValueError("positive_test_size and negative_test_size must both be positive.")

    stage_sizes = validate_stage_sizes(args.stage_sizes)
    report_ks = resolve_report_ks(args.top_k)
    family_output_dir = external_output_dir(args.output_dir) / "knowledge_editing_cold_start_chagas"
    family_output_dir.mkdir(parents=True, exist_ok=True)

    print("[Load] Loading CODE15 embeddings and metadata...")
    code15_vecs = load_dataset_vectors(args.code15_ecg_path)
    code15_df = load_metadata_df(args.code15_meta_path)
    code15_df, code15_vecs = align_metadata_and_vectors(code15_df, code15_vecs)

    split_payload = prepare_case_level_splits(
        code15_df=code15_df,
        code15_vecs=code15_vecs,
        chagas_col=args.chagas_col,
        positive_test_size=args.positive_test_size,
        negative_test_size=args.negative_test_size,
        stage_sizes=stage_sizes,
        seed=args.seed,
    )

    split_manifest_path = family_output_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(split_payload["split_manifest"]), handle, ensure_ascii=False, indent=2)
    print(f"[Saved] Split manifest -> {split_manifest_path}")
    print(
        f"[Split] Base gallery negatives: {len(split_payload['base_gallery_idx'])}, "
        f"injection positives: {len(split_payload['injection_pool_idx'])}, "
        f"queries: {len(split_payload['query_vecs'])}"
    )
    print(
        f"[Metrics] Reporting simple case-level retrieval metrics: "
        + ", ".join([f"R@{value}" for value in report_ks])
        + ", AUC"
    )
    print(
        f"[CI] Computing inline 95% bootstrap CI with bootstrap={args.bootstrap} for stage-level R@k and AUC."
    )

    base_gallery_vecs = code15_vecs[np.asarray(split_payload["base_gallery_idx"], dtype=np.int64)]
    gallery_state = initialize_gallery_state(base_gallery_vecs)
    summaries = []
    for stage_index, stage_size in enumerate(stage_sizes):
        extend_gallery_state(
            state=gallery_state,
            target_injected_count=stage_size,
            injection_pool_idx=split_payload["injection_pool_idx"],
            code15_vecs=code15_vecs,
        )
        print(f"[Stage] Evaluating stage_index={stage_index}, stage_size={stage_size}")
        summaries.append(
            evaluate_stage(
                family_output_dir=family_output_dir,
                stage_index=stage_index,
                stage_size=stage_size,
                gallery_state=gallery_state,
                query_is_chagas=split_payload["query_is_chagas"],
                query_vecs=split_payload["query_vecs"],
                top_k=args.top_k,
                batch_size=args.batch_size,
                report_ks=report_ks,
                bootstrap=args.bootstrap,
                seed=args.seed,
            )
        )

    summary_json_path = family_output_dir / "knowledge_editing_cold_start_chagas_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summaries), handle, ensure_ascii=False, indent=2)

    summary_csv_path = family_output_dir / "knowledge_editing_cold_start_chagas_summary.csv"
    pd.DataFrame([flatten_stage_summary(summary, report_ks) for summary in summaries]).to_csv(summary_csv_path, index=False)

    print(f"[Saved] Summary JSON  -> {summary_json_path}")
    print(f"[Saved] Summary CSV   -> {summary_csv_path}")


if __name__ == "__main__":
    main()
