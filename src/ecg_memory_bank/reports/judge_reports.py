"""Hosted DeepSeek reasoning evaluation.

Purpose
-------
Execute prepared judge requests against the authorized Volcengine OpenAI-
compatible endpoint.

Method
------
Preserve request model/body settings, send nonstreaming calls with
retry/resume, and collect case-keyed responses.

Inputs
------
Prepared judge JSONL, external API credential, endpoint model/deployment name,
and output paths.

Outputs
-------
External judge response JSONL plus transport and retry summaries.

Paper parameters
----------------
Formal results label the model deepseek-ai/DeepSeek-V3.1-Terminus; service
alias is deepseek-v3.1, thinking off, max_tokens=8192.

Shapes and meaning
------------------
One response holds criterion scores for one report; judge values support
relative configuration comparison, not AUROC.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from ecg_memory_bank.reports.judge_io import DEFAULT_SERVER_GENERATE_OUTPUT_DIR


DEFAULT_BASE_URL = "https://operator.las.cn-beijing.volces.com/api/v1"
DEFAULT_REQUEST_FILE = f"{DEFAULT_SERVER_GENERATE_OUTPUT_DIR}/mimic_reasoning_eval_all.jsonl"
DEFAULT_OUTPUT_FILE = f"{DEFAULT_SERVER_GENERATE_OUTPUT_DIR}/judge_output_volcengine.jsonl"
DEFAULT_SUMMARY_FILE = f"{DEFAULT_SERVER_GENERATE_OUTPUT_DIR}/judge_output_volcengine_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MIMIC reasoning evaluation requests against a Volcengine OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_REQUEST_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--summary-json", type=str, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Volcengine model or endpoint ID, for example a DeepSeek 3.1 deployment name.",
    )
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible Volcengine base URL. Used unless --endpoint is provided.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="",
        help="Full chat completions endpoint override. If empty, uses <base-url>/chat/completions.",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking-type", type=str, default="disabled")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="Only resend request ids that currently have error entries in --output.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = str(payload.get("custom_id", "")).strip()
            if custom_id:
                completed.add(custom_id)
    return completed


def classify_existing_results(path: Path) -> tuple[set[str], set[str]]:
    success_ids: set[str] = set()
    error_ids: set[str] = set()
    if not path.exists():
        return success_ids, error_ids
    for row in load_jsonl(path):
        custom_id = str(row.get("custom_id", "")).strip()
        if not custom_id:
            continue
        if row.get("response"):
            success_ids.add(custom_id)
            if custom_id in error_ids:
                error_ids.remove(custom_id)
        elif row.get("error"):
            if custom_id not in success_ids:
                error_ids.add(custom_id)
    return success_ids, error_ids


def resolve_api_key(cli_api_key: str) -> str:
    return (
        cli_api_key
        or os.getenv("ARK_API_KEY", "").strip()
        or os.getenv("VOLCENGINE_API_KEY", "").strip()
    )


def resolve_endpoint(*, base_url: str, endpoint: str) -> str:
    if endpoint.strip():
        return endpoint.strip()
    return base_url.rstrip("/") + "/chat/completions"


def build_request_body(entry: dict[str, Any], *, model: str, max_tokens: int, thinking_type: str) -> dict[str, Any]:
    body = dict(entry.get("body", {}))
    body["model"] = body.get("model") or model
    body["max_tokens"] = int(body.get("max_tokens") or max_tokens)
    body["stream"] = bool(body.get("stream", False))
    body["thinking"] = body.get("thinking") or {"type": thinking_type}
    return body


def post_single(
    *,
    entry: dict[str, Any],
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
    max_tokens: int,
    thinking_type: str,
) -> dict[str, Any]:
    custom_id = str(entry.get("custom_id", "")).strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if custom_id:
        headers["X-Client-Request-Id"] = custom_id
    body = build_request_body(entry, model=model, max_tokens=max_tokens, thinking_type=thinking_type)
    started = time.time()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=timeout_seconds,
        )
        elapsed = round(time.time() - started, 3)
        if response.status_code == 200:
            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": response.status_code,
                    "body": response.json(),
                },
                "elapsed_seconds": elapsed,
            }
        return {
            "custom_id": custom_id,
            "error": {
                "status_code": response.status_code,
                "text": response.text,
            },
            "elapsed_seconds": elapsed,
        }
    except requests.RequestException as exc:
        return {
            "custom_id": custom_id,
            "error": {
                "error": str(exc),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.resume and args.retry_errors_only:
        raise ValueError("--resume and --retry-errors-only cannot be used together.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input request file not found: {input_path}")

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY, VOLCENGINE_API_KEY, or --api-key.")

    endpoint = resolve_endpoint(base_url=args.base_url, endpoint=args.endpoint)
    request_entries = load_jsonl(input_path)

    selected_mode = "all_requests"
    skipped_resume = 0
    if args.retry_errors_only:
        selected_mode = "retry_errors_only"
        _success_ids, error_ids = classify_existing_results(output_path)
        request_entries = [
            entry
            for entry in request_entries
            if str(entry.get("custom_id", "")).strip() in error_ids
        ]
    elif args.resume:
        selected_mode = "resume_skip_seen"
        completed_ids = load_completed_ids(output_path)
        filtered_entries = []
        for entry in request_entries:
            if str(entry.get("custom_id", "")).strip() in completed_ids:
                skipped_resume += 1
                continue
            filtered_entries.append(entry)
        request_entries = filtered_entries

    if args.limit >= 0:
        request_entries = request_entries[: args.limit]

    submitted = len(request_entries)
    success = 0
    failed = 0
    elapsed_samples: list[float] = []

    progress = tqdm(total=submitted, desc="Volcengine judge eval", unit="req")
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        future_map = {
            pool.submit(
                post_single,
                entry=entry,
                api_key=api_key,
                endpoint=endpoint,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                thinking_type=args.thinking_type,
            ): entry
            for entry in request_entries
        }
        for future in as_completed(future_map):
            result = future.result()
            append_jsonl(output_path, [result])
            elapsed_samples.append(float(result.get("elapsed_seconds", 0.0)))
            if result.get("error"):
                failed += 1
            else:
                success += 1
            progress.update(1)
            progress.set_postfix(
                success=success,
                failed=failed,
                avg_elapsed=f"{(sum(elapsed_samples) / len(elapsed_samples)):.1f}s" if elapsed_samples else "0.0s",
            )
    progress.close()

    final_success_ids, final_error_ids = classify_existing_results(output_path)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "endpoint": endpoint,
        "selection_mode": selected_mode,
        "submitted": submitted,
        "resume_skipped": skipped_resume,
        "success": success,
        "failed": failed,
        "final_success_count": len(final_success_ids),
        "remaining_error_count": len(final_error_ids),
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout_seconds,
        "max_tokens": args.max_tokens,
        "thinking_type": args.thinking_type,
        "average_elapsed_seconds": round(sum(elapsed_samples) / len(elapsed_samples), 4) if elapsed_samples else 0.0,
    }
    write_summary(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
