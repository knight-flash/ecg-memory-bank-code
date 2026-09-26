#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_memory_bank.retrieval.metrics_utils import compute_hierarchical_ndcg, compute_union_recall, gain_from_overlap, to_jsonable


def format_fraction_label(fraction: float) -> str:
    percentage = float(fraction) * 100.0
    if percentage >= 1.0:
        if abs(percentage - round(percentage)) < 1e-9:
            return f"{int(round(percentage))}%"
        return f"{percentage:g}%"
    return f"{percentage:g}%"


def format_fraction_tag(fraction: float) -> str:
    percentage = float(fraction) * 100.0
    if abs(percentage - round(percentage)) < 1e-9:
        return f"{int(round(percentage))}pct"
    return f"{percentage:g}pct".replace(".", "p")


def sample_gallery_subset(
    gallery_vecs: np.ndarray,
    gallery_df: pd.DataFrame,
    fraction: float,
    seed: int,
    selection_order: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Sample a reproducible active memory for one case-addition fraction."""
    selected_indices = select_gallery_indices(
        total=len(gallery_df),
        fraction=fraction,
        seed=seed,
        selection_order=selection_order,
    )
    return gallery_vecs[selected_indices], gallery_df.iloc[selected_indices].reset_index(drop=True)


def select_gallery_indices(
    *,
    total: int,
    fraction: float,
    seed: int,
    selection_order: np.ndarray | None = None,
) -> np.ndarray:
    """Return gallery row indices under the requested sampling protocol."""
    if not (0 < fraction <= 1.0):
        raise ValueError(f"gallery fraction must be in (0, 1], got {fraction}")
    if total <= 0:
        raise ValueError(f"gallery size must be positive, got {total}")
    sample_size = max(1, int(round(total * float(fraction))))
    if selection_order is not None:
        order = np.asarray(selection_order, dtype=np.int64)
        if order.ndim != 1 or len(order) != total:
            raise ValueError("selection_order must contain exactly one index per gallery row")
        return np.sort(order[:sample_size])
    if float(fraction) >= 1.0:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=sample_size, replace=False))


def build_anchored_nested_gallery_order(
    *,
    total: int,
    seed: int,
    anchor_fractions: tuple[float, ...] | list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a nested order while preserving legacy fixed-seed anchor sets.

    The legacy ``Generator.choice`` samples at the supplied anchor fractions
    are required to be nested.  Their set differences form deterministic
    shuffled blocks.  Prefixes below the first anchor are therefore truly
    nested, while every anchor retains exactly the membership used by the
    original benchmark.
    """
    anchors = sorted({float(value) for value in anchor_fractions})
    if not anchors or anchors[-1] != 1.0:
        raise ValueError("anchor_fractions must include 1.0")
    if anchors[0] <= 0 or anchors[-1] > 1:
        raise ValueError("anchor fractions must lie in (0, 1]")

    previous_mask = np.zeros(total, dtype=bool)
    previous_count = 0
    blocks: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    for block_index, fraction in enumerate(anchors):
        selected = select_gallery_indices(total=total, fraction=fraction, seed=seed)
        retained_count = int(np.count_nonzero(previous_mask[selected]))
        if retained_count != previous_count:
            raise RuntimeError(
                f"legacy anchor sets are not nested at fraction={fraction:g}: "
                f"retained {retained_count} of {previous_count} previous rows"
            )
        is_new = ~previous_mask[selected]
        block = selected[is_new].copy()
        block_rng = np.random.default_rng(np.random.SeedSequence([seed, total, block_index]))
        block_rng.shuffle(block)
        blocks.append(block)
        previous_mask[selected] = True
        previous_count = int(len(selected))
        audit_rows.append(
            {
                "anchor_fraction": fraction,
                "anchor_size": previous_count,
                "new_block_size": int(len(block)),
                "previous_anchor_fully_retained": True,
            }
        )

    order = np.concatenate(blocks).astype(np.int64, copy=False)
    if len(order) != total:
        raise RuntimeError("anchored nested order is not a full gallery permutation")
    return order, {
        "sampling_scheme": "anchored_nested_fixed_seed_prefix",
        "numpy_version": np.__version__,
        "total": int(total),
        "seed": int(seed),
        "anchor_fractions": anchors,
        "anchors": audit_rows,
    }


def maybe_limit_queries(
    query_vecs: np.ndarray,
    query_df: pd.DataFrame,
    l1_query: np.ndarray,
    l2_query: np.ndarray,
    l3_query: np.ndarray,
    query_labels: list[dict[str, list[str]]],
    max_queries: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[dict[str, list[str]]]]:
    if max_queries <= 0 or max_queries >= len(query_df):
        return query_vecs, query_df, l1_query, l2_query, l3_query, query_labels
    selected = np.arange(max_queries)
    return (
        query_vecs[selected],
        query_df.iloc[selected].reset_index(drop=True),
        l1_query[selected],
        l2_query[selected],
        l3_query[selected],
        [query_labels[index] for index in selected],
    )


def build_query_precision_hit_arrays(
    retrieved_indices: np.ndarray,
    db_mat: np.ndarray,
    query_mat: np.ndarray,
    top_ks: list[int] | tuple[int, ...],
    prefix: str,
) -> dict[str, np.ndarray]:
    retrieved_features = db_mat[retrieved_indices]
    query_features_exp = query_mat[:, None, :]
    intersection = retrieved_features & query_features_exp
    global_matches = intersection.any(axis=2)
    arrays: dict[str, np.ndarray] = {}
    for top_k in top_ks:
        hit_block = global_matches[:, :top_k]
        arrays[f"{prefix}_p_at_{top_k}"] = hit_block.mean(axis=1).astype(np.float32)
        arrays[f"{prefix}_hit_at_{top_k}"] = hit_block.any(axis=1).astype(np.uint8)
    return arrays


def build_union_recall_arrays(
    retrieved_indices: np.ndarray,
    db_labels: list[dict[str, list[str]]],
    query_labels: list[dict[str, list[str]]],
    top_ks: list[int] | tuple[int, ...],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    sorted_top_ks = sorted({int(top_k) for top_k in top_ks})
    for top_k in sorted_top_ks:
        values = []
        for query_idx, query_label_dict in enumerate(query_labels):
            retrieved_l3_lists = [
                db_labels[int(gallery_idx)]["l3"]
                for gallery_idx in retrieved_indices[query_idx, :top_k]
            ]
            values.append(compute_union_recall(query_label_dict["l3"], retrieved_l3_lists))
        arrays[f"union_recall_l3_at_{top_k}"] = np.asarray(values, dtype=np.float32)
    return arrays


def build_hierarchical_ndcg_arrays(
    retrieved_indices: np.ndarray,
    db_labels: list[dict[str, list[str]]],
    query_labels: list[dict[str, list[str]]],
    top_ks: list[int] | tuple[int, ...],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    sorted_top_ks = sorted({int(top_k) for top_k in top_ks})
    for top_k in sorted_top_ks:
        values = []
        for query_idx, query_label_dict in enumerate(query_labels):
            gains = []
            for gallery_idx in retrieved_indices[query_idx, :top_k]:
                gallery_label_dict = db_labels[int(gallery_idx)]
                l1_hit = bool(set(query_label_dict["l1"]).intersection(gallery_label_dict["l1"]))
                l2_hit = bool(set(query_label_dict["l2"]).intersection(gallery_label_dict["l2"]))
                l3_hit = bool(set(query_label_dict["l3"]).intersection(gallery_label_dict["l3"]))
                gains.append(gain_from_overlap(l1_hit, l2_hit, l3_hit))
            values.append(compute_hierarchical_ndcg(gains))
        arrays[f"hierarchical_ndcg_at_{top_k}"] = np.asarray(values, dtype=np.float32)
    return arrays


def save_ci_artifacts(
    *,
    scenario_output_dir: Path,
    family: str,
    scenario_name: str,
    summary_path: Path,
    query_metric_arrays: dict[str, np.ndarray] | None = None,
    auc_inputs: dict[str, dict[str, Any]] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    artifact_dir = scenario_output_dir / "ci_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    query_metrics_path = artifact_dir / f"{scenario_name}_query_metrics.npz"
    query_metric_names: list[str] = []
    if query_metric_arrays:
        np.savez_compressed(query_metrics_path, **query_metric_arrays)
        query_metric_names = list(query_metric_arrays.keys())
    else:
        np.savez_compressed(query_metrics_path)

    auc_paths: dict[str, str] = {}
    if auc_inputs:
        for metric_key, payload in auc_inputs.items():
            auc_path = artifact_dir / f"{scenario_name}_{metric_key}_auc_inputs.npz"
            np.savez_compressed(
                auc_path,
                y_true=np.asarray(payload["y_true"]).astype(np.uint8, copy=False),
                y_score=np.asarray(payload["y_score"]).astype(np.float32, copy=False),
                class_names=np.asarray(payload["class_names"], dtype=str),
            )
            auc_paths[metric_key] = str(auc_path)

    manifest_payload = {
        "family": family,
        "scenario_name": scenario_name,
        "summary_json": str(summary_path),
        "query_metrics_npz": str(query_metrics_path),
        "query_metric_names": query_metric_names,
        "auc_input_npz": auc_paths,
    }
    if extra_manifest:
        manifest_payload.update(extra_manifest)

    manifest_path = artifact_dir / "ci_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(manifest_payload), handle, ensure_ascii=False, indent=2)
    return manifest_path
