from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

from json_export_utils import DEFAULT_ID_FIELDS, export_query_bundle, is_missing, row_get, save_json


def resolve_case_identifier(row: Mapping[str, object], fallback_index: Optional[int] = None) -> str:
    candidate = row_get(row, *DEFAULT_ID_FIELDS)
    if not is_missing(candidate):
        return str(candidate)

    if fallback_index is not None:
        return f"query_{fallback_index}"
    return "unknown_case"


def build_prototype_candidate_bank(
    *,
    x_train: torch.Tensor,
    train_df: pd.DataFrame,
    prototype_report_df: pd.DataFrame,
    activation_report_df: pd.DataFrame,
) -> tuple[torch.Tensor, pd.DataFrame]:
    merged_df = prototype_report_df.merge(
        activation_report_df[["prototype_id", "activation_count", "activation_rate"]],
        on="prototype_id",
        how="left",
    )

    bank_rows: list[dict] = []
    bank_embeddings: list[torch.Tensor] = []

    for row in merged_df.itertuples(index=False):
        local_idx = int(row.train_local_index)
        if local_idx < 0 or local_idx >= len(train_df):
            continue

        base_row = train_df.iloc[local_idx].to_dict()
        base_case_id = resolve_case_identifier(base_row, fallback_index=local_idx)
        export_case_id = f"{base_case_id}__p{int(row.prototype_id)}"

        enriched_row = dict(base_row)
        enriched_row.update(
            {
                "export_case_id": export_case_id,
                "prototype_id": int(row.prototype_id),
                "prototype_class_idx": int(row.class_idx),
                "prototype_class_name": str(row.class_name),
                "train_local_index": int(row.train_local_index),
                "train_global_index": int(row.train_global_index),
                "activation_count": int(getattr(row, "activation_count", 0) or 0),
                "activation_rate": float(getattr(row, "activation_rate", 0.0) or 0.0),
                "is_prototype_anchor": True,
            }
        )

        bank_rows.append(enriched_row)
        bank_embeddings.append(x_train[local_idx].detach().cpu().unsqueeze(0))

    if not bank_embeddings:
        return torch.empty((0, x_train.shape[1])), pd.DataFrame()

    return torch.cat(bank_embeddings, dim=0), pd.DataFrame(bank_rows)


def select_balanced_query_indices(
    y_query: torch.Tensor,
    max_queries: int = 12,
    max_per_class: int = 2,
) -> list[int]:
    selected: list[int] = []
    seen = set()
    num_classes = y_query.shape[1]

    for class_idx in range(num_classes):
        class_candidates = torch.nonzero(y_query[:, class_idx] == 1.0, as_tuple=False).squeeze(1).tolist()
        class_count = 0
        for idx in class_candidates:
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            class_count += 1
            if class_count >= max_per_class or len(selected) >= max_queries:
                break
        if len(selected) >= max_queries:
            break

    if len(selected) < max_queries:
        for idx in range(len(y_query)):
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            if len(selected) >= max_queries:
                break

    return selected


def compute_topk_indices_for_queries(
    model,
    x_queries: torch.Tensor,
    x_bank: torch.Tensor,
    top_k: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        query_encoded = model.encode(x_queries.to(device))

        num_queries = query_encoded.shape[0]
        best_scores = torch.full((num_queries, top_k), float("-inf"), device=device)
        best_indices = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)

        for start_idx in range(0, x_bank.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, x_bank.shape[0])
            bank_chunk = x_bank[start_idx:end_idx].to(device)
            bank_encoded = model.encode(bank_chunk)
            similarity = torch.matmul(query_encoded, bank_encoded.T)

            chunk_k = min(top_k, similarity.shape[1])
            chunk_scores, chunk_indices = torch.topk(similarity, k=chunk_k, dim=1)
            chunk_indices = chunk_indices + start_idx

            combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
            combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
            new_scores, top_pos = torch.topk(combined_scores, k=top_k, dim=1)
            new_indices = torch.gather(combined_indices, 1, top_pos)

            best_scores = new_scores
            best_indices = new_indices

    return best_indices.cpu(), best_scores.cpu()


def export_embedding_showcase(
    *,
    model,
    x_bank: torch.Tensor,
    bank_df: pd.DataFrame,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    query_df: pd.DataFrame,
    output_base: str,
    device: torch.device,
    query_count: int = 12,
    query_per_class: int = 2,
    top_k: int = 8,
    chunk_size: int = 8192,
    export_batch_size: int = 4,
    hierarchy_map: Optional[Mapping[str, Mapping[str, object]]] = None,
    full_signal: bool = False,
) -> dict:
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_query_indices = select_balanced_query_indices(
        y_query=y_query.cpu(),
        max_queries=query_count,
        max_per_class=query_per_class,
    )

    if not selected_query_indices:
        return {"exported_queries": 0, "manifest_path": str(output_path / "manifest.json")}

    x_selected = x_query[selected_query_indices]
    topk_indices, topk_scores = compute_topk_indices_for_queries(
        model=model,
        x_queries=x_selected,
        x_bank=x_bank,
        top_k=top_k,
        device=device,
        chunk_size=chunk_size,
    )

    manifest_items: list[dict] = []
    exported_queries = 0
    failed_queries: list[dict] = []
    batch_count = 0

    for local_query_rank, query_idx in enumerate(selected_query_indices):
        query_row = query_df.iloc[int(query_idx)].to_dict()
        query_id = resolve_case_identifier(query_row, fallback_index=int(query_idx))

        print(
            f"[JSON Export] Query {local_query_rank + 1}/{len(selected_query_indices)} "
            f"-> {query_id}"
        )

        retrieval_rows_with_scores = []
        for bank_idx, score in zip(topk_indices[local_query_rank].tolist(), topk_scores[local_query_rank].tolist()):
            if bank_idx < 0:
                continue
            bank_row = bank_df.iloc[int(bank_idx)].to_dict()
            retrieval_rows_with_scores.append((bank_row, float(score)))

        export_result = export_query_bundle(
            query_row=query_row,
            retrieval_rows_with_scores=retrieval_rows_with_scores,
            output_base=str(output_path),
            hierarchy_map=hierarchy_map,
            include_query_in_retrievals=True,
            manifest_items=manifest_items,
            full_signal=full_signal,
        )
        if export_result:
            exported_queries += 1
            print(f"[JSON Export]   Success: {query_id}")
        else:
            failed_queries.append(
                {
                    "query_index": int(query_idx),
                    "query_id": query_id,
                    "reason": "query export returned None",
                }
            )
            print(f"[JSON Export]   Failed: {query_id}")

        batch_count += 1
        if batch_count >= export_batch_size:
            manifest_path = output_path / "manifest.json"
            save_json(manifest_path, manifest_items)
            print(f"[JSON Export]   Flushed manifest batch to {manifest_path}")
            batch_count = 0

    manifest_path = output_path / "manifest.json"
    save_json(manifest_path, manifest_items)
    return {
        "exported_queries": exported_queries,
        "manifest_path": str(manifest_path),
        "failed_queries": failed_queries,
        "selected_queries": len(selected_query_indices),
    }
