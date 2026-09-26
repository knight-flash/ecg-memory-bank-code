from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.nan_to_num(embeddings.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(embeddings, p=2, dim=1)


def compute_topk_cosine(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int = 4096,
    device: str | None = None,
    show_progress: bool = False,
    progress_desc: str = "Retrieval chunks",
    query_patient_ids: list[str] | None = None,
    gallery_patient_ids: list[str] | None = None,
    exclude_same_patient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    query_embeddings = normalize_embeddings(query_embeddings)
    gallery_embeddings = normalize_embeddings(gallery_embeddings)

    query_device = query_embeddings.to(device)
    num_queries = query_device.shape[0]
    query_patient_ids_arr = np.asarray(query_patient_ids, dtype=object) if query_patient_ids is not None else None
    gallery_patient_ids_arr = np.asarray(gallery_patient_ids, dtype=object) if gallery_patient_ids is not None else None
    print(
        f"[Retrieval] device={device} queries={num_queries} gallery={gallery_embeddings.shape[0]} "
        f"top_k={top_k} chunk_size={chunk_size}"
    )
    best_scores = torch.full((num_queries, top_k), float("-inf"), device=device)
    best_indices = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)

    chunk_starts = range(0, gallery_embeddings.shape[0], chunk_size)
    iterator = tqdm(chunk_starts, desc=progress_desc) if show_progress else chunk_starts
    for start_idx in iterator:
        end_idx = min(start_idx + chunk_size, gallery_embeddings.shape[0])
        chunk = gallery_embeddings[start_idx:end_idx].to(device)
        similarity = torch.matmul(query_device, chunk.T)
        if exclude_same_patient and query_patient_ids_arr is not None and gallery_patient_ids_arr is not None:
            same_patient_mask = query_patient_ids_arr[:, None] == gallery_patient_ids_arr[start_idx:end_idx][None, :]
            similarity = similarity.masked_fill(torch.from_numpy(same_patient_mask).to(device), float("-inf"))
        chunk_k = min(top_k, similarity.shape[1])
        chunk_scores, chunk_indices = torch.topk(similarity, k=chunk_k, dim=1)
        chunk_indices = chunk_indices + start_idx

        combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
        combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
        new_scores, positions = torch.topk(combined_scores, k=top_k, dim=1)
        new_indices = torch.gather(combined_indices, 1, positions)
        best_scores = new_scores
        best_indices = new_indices

    print("[Retrieval] top-k search finished")
    return best_indices.cpu(), best_scores.cpu()
