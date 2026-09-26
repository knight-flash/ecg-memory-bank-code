from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import build_batch_entry, write_json, write_jsonl
from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.data_io import load_embeddings
from ecg_memory_bank.reports.report_io import ensure_eval_dirs, load_jsonl
from ecg_memory_bank.reports.label_taxonomy import normalize_label, split_label_text
from ecg_memory_bank.reports.npy_feature_utils import build_feature_summary, load_feature_payload
from ecg_memory_bank.reports.prompting import build_chat_messages
from ecg_memory_bank.reports.retrieval import compute_topk_cosine


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Prepare MIMIC feature-driven 1000-query evaluation inputs.")
    parser.add_argument("--run-name", type=str, default=config.grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--top-k", type=int, default=config.grounding.top_k)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--prompt-version", type=str, default=config.prompt.prompt_version)
    parser.add_argument("--model", type=str, default=config.direct.model)
    parser.add_argument("--thinking-type", type=str, default=config.direct.thinking_type)
    parser.add_argument("--max-tokens", type=int, default=config.direct.max_tokens)
    return parser.parse_args()


def _resolve_run_dir(config, run_name: str, run_dir_arg: str) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    return config.output.runs_root / run_name


def _load_subset_df(path: Path, *, case_id: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing subset file: {path}. Run prepare_eval_subsets.py first.")
    df = pd.read_csv(path)
    if case_id:
        df = df.loc[df["case_id"].astype(str) == case_id].copy()
        if df.empty:
            raise KeyError(f"case_id not found in subset file: {case_id}")
    return df.reset_index(drop=True)


def _load_label_context(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing label category context: {path}. Run prepare_label_taxonomy.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _split_row_labels(row: pd.Series, label_column: str) -> list[str]:
    labels_list = row.get("labels_list")
    if isinstance(labels_list, list):
        return [str(item).strip() for item in labels_list if str(item).strip()]
    return split_label_text(row.get(label_column))


def _clean_labels(labels: list[str], label_to_category: dict[str, dict[str, str]]) -> tuple[list[str], dict[str, list[str]]]:
    clean: list[str] = []
    category_map: dict[str, list[str]] = {}
    for raw in labels:
        text = str(raw).strip()
        normalized = normalize_label(text)
        if not text or normalized not in label_to_category:
            continue
        mapped_label = label_to_category[normalized]["mapped_label"]
        if mapped_label not in clean:
            clean.append(mapped_label)
        category = label_to_category[normalized]["Classification"]
        category_map.setdefault(category, [])
        if mapped_label not in category_map[category]:
            category_map[category].append(mapped_label)
    return clean, category_map


def _load_gallery_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        case_id = str(row.get("case_id", "")).strip()
        prompt_case = row.get("prompt_case")
        if case_id and isinstance(prompt_case, dict):
            cache[case_id] = prompt_case
    return cache


def _save_gallery_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    rows = [{"case_id": key, "prompt_case": value} for key, value in sorted(cache.items())]
    write_jsonl(path, rows)


def _build_prompt_case(
    row: pd.Series,
    feature_payload: dict[str, Any],
    *,
    include_reference_labels: bool,
    label_column: str,
    label_to_category: dict[str, dict[str, str]],
) -> dict[str, Any]:
    labels_clean, label_categories = _clean_labels(_split_row_labels(row, label_column), label_to_category)
    payload = {
        "id": str(row["case_id"]),
        "samplingRate": row.get("sampling_rate"),
        "age": row.get("Age", row.get("age")),
        "gender": row.get("Gender", row.get("gender")),
        "time": row.get("RecordingTime", row.get("time", "Unknown")),
        "feature_status": "ok",
        "missing_standard_leads": [],
        "structured_features": feature_payload.get("structured_features", {}),
        "ecg_summary": build_feature_summary(feature_payload),
    }
    if include_reference_labels:
        payload["report"] = str(row.get("report_text", "") or "")
        payload["diagnosis"] = labels_clean
    return payload


def main() -> None:
    args = parse_args()
    total_start = time.time()
    config = default_report_config()
    config.sampling.top_k = args.top_k
    config.prompt.prompt_version = args.prompt_version
    config.direct.model = args.model
    config.direct.thinking_type = args.thinking_type
    config.direct.max_tokens = args.max_tokens
    config.prompt.model = config.direct.model
    config.prompt.include_raw_signals = False
    config.prompt.include_query_summary = True
    config.prompt.include_retrieved_labels = True
    config.prompt.include_retrieved_reports = True

    run_dir = _resolve_run_dir(config, args.run_name, args.run_dir)
    dirs = ensure_eval_dirs(run_dir)
    print(f"[Run] prepare MIMIC eval start -> {dirs['root']}")

    query_subset_df = _load_subset_df(dirs["subsets"] / "query_subset.csv", case_id=args.case_id)
    gallery_subset_df = _load_subset_df(dirs["subsets"] / "gallery_subset.csv", case_id="")
    label_context = _load_label_context(run_dir / "labels" / "label_category_context.json")
    label_to_category = label_context.get("label_to_category", {})
    global_candidate_labels = list(label_context.get("candidate_label_set", []))
    candidate_labels_by_category = dict(label_context.get("candidate_labels_by_category", {}))
    if not global_candidate_labels:
        raise RuntimeError("Empty candidate_label_set in label_category_context.json")

    stage_start = time.time()
    embeddings = load_embeddings(config.dataset.train_embeddings_path)
    query_embeddings = embeddings[query_subset_df["row_index"].astype(int).tolist()].clone()
    gallery_embeddings = embeddings[gallery_subset_df["row_index"].astype(int).tolist()].clone()
    print(f"[Stage] embeddings loaded and subset in {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    topk_indices, topk_scores = compute_topk_cosine(
        query_embeddings,
        gallery_embeddings,
        top_k=args.top_k,
        show_progress=True,
        progress_desc="Top-k retrieval",
        query_patient_ids=query_subset_df["patient_id"].astype(str).tolist(),
        gallery_patient_ids=gallery_subset_df["patient_id"].astype(str).tolist(),
        exclude_same_patient=config.grounding.exclude_same_patient,
    )
    print(f"[Stage] retrieval done in {time.time() - stage_start:.2f}s")

    gallery_case_ids = gallery_subset_df["case_id"].astype(str).tolist()
    retrieval_preview_records: list[dict[str, Any]] = []
    rag_requests: list[dict[str, Any]] = []
    zero_requests: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    query_feature_failures: list[dict[str, Any]] = []
    gallery_feature_failures: list[dict[str, Any]] = []

    gallery_cache_path = run_dir / "features" / "retrieved_gallery_feature_cache.jsonl"
    gallery_cache = _load_gallery_cache(gallery_cache_path) if gallery_cache_path.exists() else {}
    gallery_cache_updated = False

    stage_start = time.time()
    for query_local_idx in tqdm(range(len(query_subset_df)), desc="Build MIMIC requests", unit="case"):
        query_row = query_subset_df.iloc[query_local_idx]
        query_case_id = str(query_row["case_id"])
        query_feature_id = query_row.get(config.dataset.feature_match_key)
        query_feature_payload = load_feature_payload(config.dataset.feature_root, query_feature_id)
        if not query_feature_payload:
            query_feature_failures.append(
                {
                    "case_id": query_case_id,
                    "feature_id": str(query_feature_id),
                    "reason": "missing_query_feature_file",
                }
            )
            status_rows.append(
                {
                    "query_case_id": query_case_id,
                    "status": "query_feature_missing",
                    "retrieved_count": 0,
                }
            )
            continue

        query_labels_clean, query_label_categories = _clean_labels(
            _split_row_labels(query_row, config.dataset.label_column),
            label_to_category,
        )
        query_prompt_case = _build_prompt_case(
            query_row,
            query_feature_payload,
            include_reference_labels=False,
            label_column=config.dataset.label_column,
            label_to_category=label_to_category,
        )

        retrieved_cases: list[dict[str, Any]] = []
        retrieved_preview_cases: list[dict[str, Any]] = []
        for gallery_local_idx, score in zip(topk_indices[query_local_idx].tolist(), topk_scores[query_local_idx].tolist()):
            if gallery_local_idx < 0:
                continue
            gallery_case_id = gallery_case_ids[gallery_local_idx]
            gallery_prompt_case = gallery_cache.get(gallery_case_id)
            gallery_row = gallery_subset_df.iloc[gallery_local_idx]
            if gallery_prompt_case is None:
                gallery_feature_id = gallery_row.get(config.dataset.feature_match_key)
                gallery_feature_payload = load_feature_payload(config.dataset.feature_root, gallery_feature_id)
                if not gallery_feature_payload:
                    gallery_feature_failures.append(
                        {
                            "case_id": gallery_case_id,
                            "feature_id": str(gallery_feature_id),
                            "reason": "missing_gallery_feature_file",
                        }
                    )
                    continue
                gallery_prompt_case = _build_prompt_case(
                    gallery_row,
                    gallery_feature_payload,
                    include_reference_labels=True,
                    label_column=config.dataset.label_column,
                    label_to_category=label_to_category,
                )
                gallery_cache[gallery_case_id] = gallery_prompt_case
                gallery_cache_updated = True

            prompt_case = dict(gallery_prompt_case)
            prompt_case["similarity"] = round(float(score), 4)
            retrieved_cases.append(prompt_case)
            retrieved_preview_cases.append(
                {
                    "id": gallery_case_id,
                    "similarity": round(float(score), 4),
                    "labels_clean": prompt_case.get("diagnosis", []),
                    "report_text": prompt_case.get("report", ""),
                    "feature_summary": prompt_case.get("ecg_summary", ""),
                }
            )

        retrieval_preview_records.append(
            {
                "query": {
                    "id": query_case_id,
                    "feature_status": "ok",
                    "labels_clean": query_labels_clean,
                    "label_categories": query_label_categories,
                },
                "ground_truth_labels": query_labels_clean,
                "retrieved_cases": retrieved_preview_cases,
                "retrieval_scores": [case["similarity"] for case in retrieved_preview_cases],
            }
        )

        rag_messages = build_chat_messages(
            query_prompt_case,
            retrieved_cases,
            strict_json=config.prompt.strict_json,
            global_label_set=global_candidate_labels,
            candidate_labels_by_category=candidate_labels_by_category,
            include_raw_signals=False,
            include_query_summary=True,
            include_retrieved_labels=True,
            include_retrieved_reports=True,
            prompt_version=config.prompt.prompt_version,
        )
        zero_messages = build_chat_messages(
            query_prompt_case,
            [],
            strict_json=config.prompt.strict_json,
            global_label_set=global_candidate_labels,
            candidate_labels_by_category=candidate_labels_by_category,
            include_raw_signals=False,
            include_query_summary=True,
            include_retrieved_labels=True,
            include_retrieved_reports=True,
            prompt_version=config.prompt.prompt_version,
        )

        rag_entry = build_batch_entry(
            custom_id=f"rag_{query_case_id}",
            model=config.direct.model,
            messages=rag_messages,
            temperature=config.direct.temperature,
        )
        rag_entry["body"]["thinking"] = {"type": config.direct.thinking_type}
        rag_entry["body"]["max_tokens"] = config.direct.max_tokens
        rag_entry["body"]["stream"] = config.direct.stream

        zero_entry = build_batch_entry(
            custom_id=f"zero_{query_case_id}",
            model=config.direct.model,
            messages=zero_messages,
            temperature=config.direct.temperature,
        )
        zero_entry["body"]["thinking"] = {"type": config.direct.thinking_type}
        zero_entry["body"]["max_tokens"] = config.direct.max_tokens
        zero_entry["body"]["stream"] = config.direct.stream

        rag_requests.append(rag_entry)
        zero_requests.append(zero_entry)
        status_rows.append(
            {
                "query_case_id": query_case_id,
                "status": "ok",
                "retrieved_count": len(retrieved_cases),
            }
        )

    print(f"[Stage] request assembly done in {time.time() - stage_start:.2f}s")

    write_jsonl(dirs["previews"] / "retrieval_preview.jsonl", retrieval_preview_records)
    write_json(dirs["previews"] / "retrieval_preview.json", retrieval_preview_records)
    write_jsonl(dirs["requests"] / "rag_requests.jsonl", rag_requests)
    write_jsonl(dirs["requests"] / "zero_requests.jsonl", zero_requests)
    write_jsonl(dirs["features"] / "query_feature_failures.jsonl", query_feature_failures)
    write_jsonl(dirs["features"] / "gallery_feature_failures.jsonl", gallery_feature_failures)
    if gallery_cache_updated:
        _save_gallery_cache(gallery_cache_path, gallery_cache)
    pd.DataFrame(status_rows).to_csv(dirs["logs"] / "query_status.csv", index=False)

    summary = {
        "run_dir": str(dirs["root"]),
        "query_rows": len(query_subset_df),
        "gallery_rows": len(gallery_subset_df),
        "query_exported": len(rag_requests),
        "rag_request_entries": len(rag_requests),
        "zero_request_entries": len(zero_requests),
        "top_k": args.top_k,
        "candidate_label_count": len(global_candidate_labels),
        "candidate_labels_by_category": candidate_labels_by_category,
        "label_column": config.dataset.label_column,
        "feature_root": config.dataset.feature_root,
        "feature_match_key": config.dataset.feature_match_key,
        "exclude_same_patient": config.grounding.exclude_same_patient,
        "prepare_mode": "mimic_npy_features",
        "prompt_version": config.prompt.prompt_version,
        "gallery_cache_entries": len(gallery_cache),
        "query_feature_missing": len(query_feature_failures),
        "gallery_feature_missing": len(gallery_feature_failures),
        "prompt_config_snapshot": asdict(config.prompt),
        "direct_api_config_snapshot": asdict(config.direct),
        "elapsed_seconds": round(time.time() - total_start, 2),
    }
    write_json(dirs["logs"] / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
