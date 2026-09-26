#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Retrieval metrics and indexing.

Purpose
-------
Provide shared embedding loading, strict row alignment, FAISS indexing,
multilabel voting, and classification metrics.

Method
------
Normalize finite nonzero vectors, search with inner-product similarity, map
labels into hierarchy levels, and calculate query/class aggregates.

Inputs
------
Embedding tensors, aligned metadata DataFrames, label hierarchy maps, and
top-K neighbor arrays.

Outputs
-------
NumPy arrays, DataFrames, index objects, per-query metrics, and AUROC
summaries; no files are written on import.

Paper parameters
----------------
K and class lists are supplied by callers; the normalizer rejects invalid
vectors or metadata length mismatch rather than truncating.

Shapes and meaning
------------------
Vectors are [N,D] float32; neighbor IDs [Q,K]; multilabel targets and vote
scores [Q,C]. AUROC treats each class one-vs-rest.
"""


from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ecg_memory_bank.retrieval.label_mapping_utils import load_label_hierarchy, parse_label_field

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class NumpyInnerProductIndex:
    """Minimal FAISS-like fallback for environments without faiss."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32, copy=False)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32, copy=False)
        scores = queries @ self.vectors.T
        top_k = min(top_k, self.vectors.shape[0])
        partition = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        indices = np.take_along_axis(partition, order, axis=1)
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return sorted_scores, indices


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def safe_mean(values: Iterable[float]) -> float:
    numeric = [float(value) for value in values if value == value]
    if not numeric:
        return float("nan")
    return float(np.mean(numeric))


def load_dataset_vectors(emb_path: str | Path) -> np.ndarray:
    ecg_emb = torch.load(emb_path, map_location="cpu")
    if ecg_emb.ndim != 2 or not torch.isfinite(ecg_emb).all():
        raise ValueError(f"Embedding tensor must be finite and two-dimensional: {emb_path}")
    if (torch.linalg.vector_norm(ecg_emb, dim=1) == 0).any():
        raise ValueError(f"Embedding tensor contains zero vectors: {emb_path}")
    ecg_emb = F.normalize(ecg_emb, p=2, dim=1)
    return ecg_emb.detach().cpu().numpy().astype(np.float32, copy=False)


def build_faiss_index(db_vectors: np.ndarray):
    db_vectors = db_vectors.astype(np.float32, copy=False)
    if faiss is None:
        return NumpyInnerProductIndex(db_vectors)

    dimension = db_vectors.shape[1]
    index_flat = faiss.IndexFlatIP(dimension)
    try:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index_flat)
        index.add(db_vectors)
        return index
    except Exception:
        print("!! Warning: FAISS GPU index build failed; falling back to CPU IndexFlatIP.")
        index = index_flat
    index.add(db_vectors)
    return index


def load_metadata_df(meta_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(meta_path).reset_index(drop=True)


def align_metadata_and_vectors(df: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if len(df) != len(vectors):
        raise ValueError(
            f"Metadata rows ({len(df)}) do not match embedding rows ({len(vectors)}); "
            "refusing positional truncation because it could change case links."
        )
    return df.reset_index(drop=True), vectors


def ensure_label_column(df: pd.DataFrame, preferred_col: str) -> str:
    if preferred_col in df.columns:
        return preferred_col
    fallback_candidates = ["labels_correct", "disease_labels", "labels"]
    for candidate in fallback_candidates:
        if candidate in df.columns:
            print(f"!! Warning: Column '{preferred_col}' not found. Using '{candidate}' instead.")
            return candidate
    raise KeyError(f"No valid label column found. Tried: {[preferred_col] + fallback_candidates}")


def choose_label_column(df: pd.DataFrame, preferred_col: str) -> str:
    return ensure_label_column(df, preferred_col)


def split_indices(
    df: pd.DataFrame,
    split_by: str,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    rng = np.random.default_rng(seed)

    if split_by == "subject" and "subject_id" in df.columns:
        subject_ids = df["subject_id"].astype(str).fillna("").to_numpy()
        unique_subjects = np.unique(subject_ids)
        rng.shuffle(unique_subjects)
        val_subject_count = max(1, int(round(len(unique_subjects) * val_ratio)))
        val_subjects = set(unique_subjects[:val_subject_count])
        val_mask = np.array([subject in val_subjects for subject in subject_ids], dtype=bool)
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        split_meta = {
            "split_by": "subject",
            "num_unique_subjects": int(len(unique_subjects)),
            "num_train_subjects": int(len(unique_subjects) - val_subject_count),
            "num_val_subjects": int(val_subject_count),
        }
        return train_idx, val_idx, split_meta

    if split_by == "subject":
        print("!! Warning: 'subject_id' not found. Falling back to row-wise split.")

    indices = np.arange(len(df))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_idx = np.sort(indices[:val_count])
    train_idx = np.sort(indices[val_count:])
    split_meta = {
        "split_by": "row",
        "num_train_rows": int(len(train_idx)),
        "num_val_rows": int(len(val_idx)),
    }
    return train_idx, val_idx, split_meta


def build_multihot_from_df(
    df: pd.DataFrame,
    label_col: str,
    hierarchy_map: dict[str, tuple[str, str]],
    reference_classes: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]], list[dict[str, list[str]]]]:
    raw_series = df[label_col].fillna("").astype(str)

    parsed_samples: list[dict[str, list[str]]] = []
    all_l1 = set()
    all_l2 = set()
    all_l3 = set()

    for raw_val in tqdm(raw_series, desc="Parsing Labels", leave=False):
        labels = parse_label_field(raw_val)
        sample_l1 = sorted({hierarchy_map[label][0] for label in labels if label in hierarchy_map})
        sample_l2 = sorted({hierarchy_map[label][1] for label in labels if label in hierarchy_map})
        sample_l3 = sorted({label for label in labels if label in hierarchy_map})

        if reference_classes is None:
            all_l1.update(sample_l1)
            all_l2.update(sample_l2)
            all_l3.update(sample_l3)

        parsed_samples.append({"l1": sample_l1, "l2": sample_l2, "l3": sample_l3})

    if reference_classes is None:
        class_lists = {
            "l1": sorted(all_l1),
            "l2": sorted(all_l2),
            "l3": sorted(all_l3),
        }
    else:
        class_lists = reference_classes

    l1_map = {name: idx for idx, name in enumerate(class_lists["l1"])}
    l2_map = {name: idx for idx, name in enumerate(class_lists["l2"])}
    l3_map = {name: idx for idx, name in enumerate(class_lists["l3"])}

    num_samples = len(parsed_samples)
    l1_mat = np.zeros((num_samples, len(class_lists["l1"])), dtype=bool)
    l2_mat = np.zeros((num_samples, len(class_lists["l2"])), dtype=bool)
    l3_mat = np.zeros((num_samples, len(class_lists["l3"])), dtype=bool)

    for row_idx, sample in enumerate(parsed_samples):
        for label in sample["l1"]:
            if label in l1_map:
                l1_mat[row_idx, l1_map[label]] = True
        for label in sample["l2"]:
            if label in l2_map:
                l2_mat[row_idx, l2_map[label]] = True
        for label in sample["l3"]:
            if label in l3_map:
                l3_mat[row_idx, l3_map[label]] = True

    return l1_mat, l2_mat, l3_mat, class_lists, parsed_samples


def calculate_metrics_multihot(
    retrieved_indices: np.ndarray,
    db_mat: np.ndarray,
    query_mat: np.ndarray,
    k_list: list[int] | tuple[int, ...] = (1, 5, 10),
    class_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if db_mat.shape[1] != query_mat.shape[1]:
        return {}, pd.DataFrame()

    max_k = retrieved_indices.shape[1]
    n_classes = db_mat.shape[1]
    retrieved_features = db_mat[retrieved_indices]
    query_features_exp = query_mat[:, None, :]
    intersection = retrieved_features & query_features_exp
    global_matches = intersection.any(axis=2)

    results: dict[str, float] = {}
    for k in k_list:
        if k > max_k:
            continue
        hits_k = global_matches[:, :k]
        results[f"P@{k}"] = float(hits_k.mean(axis=1).mean())
        results[f"Hit@{k}"] = float(hits_k.any(axis=1).astype(float).mean())

    hit_cumsum = np.cumsum(global_matches, axis=1)
    ranks = np.arange(1, global_matches.shape[1] + 1).reshape(1, -1)
    precision_matrix = hit_cumsum / ranks
    ap = (precision_matrix * global_matches).sum(axis=1) / (global_matches.sum(axis=1) + 1e-9)
    results["mAP"] = float(ap.mean())

    per_class_results = []
    retrieved_has_class = retrieved_features.any(axis=1)
    for class_idx in range(n_classes):
        class_name = class_names[class_idx] if class_names else str(class_idx)
        valid_queries = query_mat[:, class_idx]
        support = int(valid_queries.sum())
        if support == 0:
            continue
        hits = retrieved_has_class[valid_queries, class_idx]
        recall_at_k = float(hits.mean())
        specific_class_hits = retrieved_features[valid_queries, :, class_idx]
        precision_at_k = float(specific_class_hits.mean(axis=1).mean())
        per_class_results.append(
            {
                "Class": class_name,
                "Support": support,
                f"Hit@{max_k}": recall_at_k,
                f"P@{max_k}": precision_at_k,
            }
        )

    df_per_class = pd.DataFrame(per_class_results)
    if not df_per_class.empty:
        df_per_class = df_per_class.sort_values("Support", ascending=False).reset_index(drop=True)
    return results, df_per_class


def compute_global_hit_vectors(
    retrieved_indices: np.ndarray,
    db_mat: np.ndarray,
    query_mat: np.ndarray,
    top_ks: tuple[int, ...],
) -> dict[int, np.ndarray]:
    retrieved_features = db_mat[retrieved_indices]
    query_features_exp = query_mat[:, None, :]
    intersection = retrieved_features & query_features_exp
    global_matches = intersection.any(axis=2)
    result = {}
    for top_k in top_ks:
        result[top_k] = global_matches[:, :top_k].any(axis=1).astype(np.uint8)
    return result


def build_retrieval_voting_scores(
    retrieved_indices: np.ndarray,
    db_mat: np.ndarray,
    top_k: int | None = None,
) -> np.ndarray:
    retrieved_k = retrieved_indices.shape[1]
    effective_k = retrieved_k if top_k is None else min(int(top_k), retrieved_k)
    if effective_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    topk_indices = retrieved_indices[:, :effective_k]
    retrieved_features = db_mat[topk_indices].astype(np.float32, copy=False)
    return retrieved_features.sum(axis=1) / float(effective_k)


def calculate_retrieval_voting_auc_from_scores(
    query_mat: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str] | tuple[str, ...] | None,
    top_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = query_mat.astype(np.uint8, copy=False)
    n_queries, n_classes = y_true.shape
    effective_class_names = list(class_names) if class_names is not None else [str(index) for index in range(n_classes)]

    rows = []
    auc_values = []
    valid_class_count = 0
    skipped_class_count = 0
    for class_idx, class_name in enumerate(effective_class_names):
        y_true_col = y_true[:, class_idx]
        support = int(y_true_col.sum())
        if 0 < support < n_queries:
            auc = float(roc_auc_score(y_true_col, y_score[:, class_idx]))
            auc_values.append(auc)
            valid_class_count += 1
        else:
            auc = float("nan")
            skipped_class_count += 1
        rows.append({"Class": class_name, "Support": support, "AUROC": auc})

    per_class_df = pd.DataFrame(rows)
    if not per_class_df.empty:
        per_class_df = per_class_df.sort_values(by=["Support", "Class"], ascending=[False, True]).reset_index(drop=True)

    macro_auc = float(np.mean(auc_values)) if auc_values else float("nan")
    micro_true = y_true.ravel()
    if 0 < int(micro_true.sum()) < micro_true.size:
        micro_auc = float(roc_auc_score(micro_true, y_score.ravel()))
    else:
        micro_auc = float("nan")

    summary = {
        "top_k": int(top_k),
        "num_queries": int(n_queries),
        "num_classes": int(n_classes),
        "valid_class_count": int(valid_class_count),
        "skipped_class_count": int(skipped_class_count),
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
    }
    return summary, per_class_df


def calculate_retrieval_voting_auc(
    retrieved_indices: np.ndarray,
    db_mat: np.ndarray,
    query_mat: np.ndarray,
    class_names: list[str] | tuple[str, ...] | None = None,
    top_k: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    effective_k = retrieved_indices.shape[1] if top_k is None else min(int(top_k), retrieved_indices.shape[1])
    y_score = build_retrieval_voting_scores(retrieved_indices, db_mat, top_k=effective_k)
    return calculate_retrieval_voting_auc_from_scores(query_mat, y_score, class_names, effective_k)


def compute_union_recall(query_labels_l3: list[str], retrieved_l3_lists: list[list[str]]) -> float:
    query_set = set(query_labels_l3)
    if not query_set:
        return float("nan")
    retrieved_union = set()
    for labels in retrieved_l3_lists:
        retrieved_union.update(labels)
    return len(query_set.intersection(retrieved_union)) / len(query_set)


def gain_from_overlap(l1_hit: bool, l2_hit: bool, l3_hit: bool) -> int:
    if l3_hit:
        return 3
    if l2_hit:
        return 2
    if l1_hit:
        return 1
    return 0


def compute_hierarchical_ndcg(gains: list[int]) -> float:
    if not gains:
        return float("nan")
    if max(gains) <= 0:
        return 0.0

    dcg = 0.0
    for rank, gain in enumerate(gains, start=1):
        dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted(gains, reverse=True)
    idcg = 0.0
    for rank, gain in enumerate(ideal_gains, start=1):
        idcg += gain / math.log2(rank + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


SUPPORT_TIER_DEFINITIONS: tuple[tuple[str, int, int | None], ...] = (
    ("tail", 1, 1000),
    ("low_mid", 1001, 5000),
    ("mid", 5001, 10000),
    ("head", 10001, None),
)


def build_support_tier_metrics(
    per_class_df: pd.DataFrame,
    *,
    support_col: str = "Support",
    auc_col: str = "AUROC",
) -> dict[str, dict[str, Any]]:
    if per_class_df is None or per_class_df.empty:
        return {
            tier_name: {
                "support_min": int(min_support),
                "support_max": None if max_support is None else int(max_support),
                "class_count": 0,
                "mean_auc": float("nan"),
            }
            for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS
        }

    support_values = pd.to_numeric(per_class_df[support_col], errors="coerce")
    auc_values = pd.to_numeric(per_class_df[auc_col], errors="coerce")
    metrics: dict[str, dict[str, Any]] = {}
    for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS:
        mask = support_values >= min_support
        if max_support is not None:
            mask &= support_values <= max_support
        tier_auc = auc_values.loc[mask].dropna()
        metrics[tier_name] = {
            "support_min": int(min_support),
            "support_max": None if max_support is None else int(max_support),
            "class_count": int(mask.sum()),
            "mean_auc": float(tier_auc.mean()) if len(tier_auc) else float("nan"),
        }
    return metrics


def build_support_tier_class_lists(
    per_class_df: pd.DataFrame,
    *,
    support_col: str = "Support",
    class_col: str = "Class",
) -> dict[str, list[str]]:
    if per_class_df is None or per_class_df.empty:
        return {tier_name: [] for tier_name, _, _ in SUPPORT_TIER_DEFINITIONS}

    support_values = pd.to_numeric(per_class_df[support_col], errors="coerce")
    class_values = per_class_df[class_col].astype(str)
    class_lists: dict[str, list[str]] = {}
    for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS:
        mask = support_values >= min_support
        if max_support is not None:
            mask &= support_values <= max_support
        class_lists[tier_name] = class_values.loc[mask].tolist()
    return class_lists


def build_query_level_artifacts(
    retrieved_indices: np.ndarray,
    db_labels: list[dict[str, list[str]]],
    query_labels: list[dict[str, list[str]]],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    union_recalls = []
    ndcgs = []
    l1_hit_at_5 = []
    l1_hit_at_k = []
    l2_hit_at_5 = []
    l2_hit_at_k = []
    l3_hit_at_5 = []
    l3_hit_at_k = []
    rows = []

    for query_idx in tqdm(range(len(query_labels)), desc="Scoring queries", leave=False):
        query_label_dict = query_labels[query_idx]
        gains = []
        retrieved_l3_lists = []
        level_hits_5 = {"l1": False, "l2": False, "l3": False}
        level_hits_k = {"l1": False, "l2": False, "l3": False}

        for rank, gallery_idx in enumerate(retrieved_indices[query_idx], start=1):
            gallery_label_dict = db_labels[int(gallery_idx)]
            l1_hit = bool(set(query_label_dict["l1"]).intersection(gallery_label_dict["l1"]))
            l2_hit = bool(set(query_label_dict["l2"]).intersection(gallery_label_dict["l2"]))
            l3_hit = bool(set(query_label_dict["l3"]).intersection(gallery_label_dict["l3"]))
            if rank <= 5:
                level_hits_5["l1"] = level_hits_5["l1"] or l1_hit
                level_hits_5["l2"] = level_hits_5["l2"] or l2_hit
                level_hits_5["l3"] = level_hits_5["l3"] or l3_hit
            level_hits_k["l1"] = level_hits_k["l1"] or l1_hit
            level_hits_k["l2"] = level_hits_k["l2"] or l2_hit
            level_hits_k["l3"] = level_hits_k["l3"] or l3_hit
            gains.append(gain_from_overlap(l1_hit, l2_hit, l3_hit))
            retrieved_l3_lists.append(gallery_label_dict["l3"])

        union_recall = compute_union_recall(query_label_dict["l3"], retrieved_l3_lists)
        hierarchical_ndcg = compute_hierarchical_ndcg(gains)
        union_recalls.append(union_recall)
        ndcgs.append(hierarchical_ndcg)
        l1_hit_at_5.append(level_hits_5["l1"])
        l1_hit_at_k.append(level_hits_k["l1"])
        l2_hit_at_5.append(level_hits_5["l2"])
        l2_hit_at_k.append(level_hits_k["l2"])
        l3_hit_at_5.append(level_hits_5["l3"])
        l3_hit_at_k.append(level_hits_k["l3"])

        rows.append(
            {
                "query_index": query_idx,
                f"union_recall_l3@{top_k}": union_recall,
                f"hierarchical_ndcg@{top_k}": hierarchical_ndcg,
            }
        )

    artifact_arrays = {
        "query_index": np.arange(len(query_labels), dtype=np.int32),
        "union_recall_l3": np.asarray(union_recalls, dtype=np.float32),
        "hierarchical_ndcg": np.asarray(ndcgs, dtype=np.float32),
        "l1_hit_at_5": np.asarray(l1_hit_at_5, dtype=np.uint8),
        "l1_hit_at_k": np.asarray(l1_hit_at_k, dtype=np.uint8),
        "l2_hit_at_5": np.asarray(l2_hit_at_5, dtype=np.uint8),
        "l2_hit_at_k": np.asarray(l2_hit_at_k, dtype=np.uint8),
        "l3_hit_at_5": np.asarray(l3_hit_at_5, dtype=np.uint8),
        "l3_hit_at_k": np.asarray(l3_hit_at_k, dtype=np.uint8),
    }
    return pd.DataFrame(rows), artifact_arrays


def bundled_label_mapping_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "label_category_counts_9class.csv"


def load_bundled_label_hierarchy() -> dict[str, tuple[str, str]]:
    return load_label_hierarchy(bundled_label_mapping_path())
