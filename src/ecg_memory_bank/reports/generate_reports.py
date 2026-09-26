"""Direct GLM report inference.

Purpose
-------
Send prepared ECG-Grounding requests to the Zhipu-compatible chat endpoint
with retry/resume support.

Method
------
Batch requests by mode, post them concurrently, parse recoverable responses,
and preserve case-level raw and structured outputs.

Inputs
------
Run-local request JSONL, ZHIPU_API_KEY, selected GLM model, and external
output directory.

Outputs
-------
External raw response JSONL, parsed JSON objects, failure metadata, and
progress summaries.

Paper parameters
----------------
Paper models are GLM-4.6 and GLM-5; temperature=0.1 is embedded in requests,
thinking enabled, max_tokens=8192.

Shapes and meaning
------------------
A response supplies one 0–1 score per candidate label plus text reasoning;
these are ranking scores, not calibrated probabilities.
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

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in os.sys.path:
    os.sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.batch_io import write_json, write_jsonl
from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.report_io import REQUEST_FILE_MAP, ensure_eval_dirs, iter_mode_paths, load_jsonl
from ecg_memory_bank.reports.response_parsing import (
    extract_completion_message_content,
    extract_completion_reasoning_content,
    find_json_candidate,
    normalize_prediction_payload,
)


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Run configurable direct requests for MIMIC self-RAG evaluation.")
    parser.add_argument("--run-name", type=str, default=config.grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--mode", choices=["rag", "zero", "all"], default="all")
    parser.add_argument("--model", type=str, default=config.direct.model)
    parser.add_argument("--batch-size", type=int, default=config.direct.batch_size)
    parser.add_argument("--concurrency", type=int, default=config.direct.concurrency)
    parser.add_argument("--endpoint", type=str, default=config.direct.endpoint)
    parser.add_argument("--max-tokens", type=int, default=config.direct.max_tokens)
    parser.add_argument("--thinking-type", type=str, default=config.direct.thinking_type)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key", type=str, default="")
    return parser.parse_args()


def _resolve_run_dir(config, run_name: str, run_dir_arg: str) -> Path:
    if run_dir_arg:
        return Path(run_dir_arg)
    return config.output.runs_root / run_name


def _resolve_api_key(cli_api_key: str, config) -> str:
    return cli_api_key or os.getenv("ZHIPU_API_KEY", "").strip() or config.batch.api_key


def _load_mode_requests(request_file: Path, *, case_id: str, limit: int | None) -> list[dict[str, Any]]:
    requests_data = load_jsonl(request_file)
    if case_id:
        requests_data = [item for item in requests_data if str(item.get("custom_id", "")).endswith(case_id)]
    if limit is not None and limit >= 0:
        requests_data = requests_data[:limit]
    return requests_data


def _should_skip(entry: dict[str, Any], raw_dir: Path, parsed_dir: Path) -> bool:
    case_id = str(entry.get("custom_id", "")).partition("_")[2] or str(entry.get("custom_id", ""))
    return (raw_dir / f"{case_id}.json").exists() and (parsed_dir / f"{case_id}.json").exists()


def _reparse_existing_raw(entry: dict[str, Any], raw_dir: Path, parsed_dir: Path) -> bool:
    custom_id = str(entry.get("custom_id", ""))
    case_id = custom_id.partition("_")[2] or custom_id
    raw_path = raw_dir / f"{case_id}.json"
    parsed_path = parsed_dir / f"{case_id}.json"
    if not raw_path.exists() or parsed_path.exists():
        return False

    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    result = raw_payload.get("result", {})
    if not result.get("ok"):
        return False

    response_body = result.get("body", {})
    raw_content = extract_completion_message_content(response_body)
    parsed_json = find_json_candidate(raw_content)
    normalized = normalize_prediction_payload(
        custom_id=custom_id,
        raw_content=raw_content,
        parsed_json=parsed_json,
    )
    normalized["usage"] = response_body.get("usage", {})
    normalized["raw_reasoning_content"] = extract_completion_reasoning_content(response_body)
    normalized["elapsed_seconds"] = result.get("elapsed_seconds")
    write_json(parsed_path, normalized)
    return True


def _split_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _post_single(entry: dict[str, Any], config, api_key: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_body = dict(entry["body"])
    request_body["model"] = config.direct.model
    request_body["max_tokens"] = config.direct.max_tokens
    request_body["stream"] = config.direct.stream
    request_body["thinking"] = {"type": config.direct.thinking_type}
    attempts = len(config.direct.backoff_seconds) + 1
    started = time.time()
    for attempt in range(attempts):
        try:
            response = requests.post(
                config.direct.endpoint,
                headers=headers,
                json=request_body,
                timeout=config.direct.timeout_seconds,
            )
            elapsed = round(time.time() - started, 3)
            if response.status_code == 200:
                return {
                    "ok": True,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed,
                    "body": response.json(),
                }
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < len(config.direct.backoff_seconds):
                    time.sleep(config.direct.backoff_seconds[attempt])
                    continue
            return {
                "ok": False,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "text": response.text,
            }
        except requests.RequestException as exc:
            if attempt < len(config.direct.backoff_seconds):
                time.sleep(config.direct.backoff_seconds[attempt])
                continue
            return {
                "ok": False,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
    return {"ok": False, "elapsed_seconds": round(time.time() - started, 3), "error": "unknown request failure"}


def _process_mode(run_dir: Path, mode: str, config, api_key: str, *, batch_size: int, concurrency: int, limit: int | None, case_id: str, resume: bool) -> dict[str, int]:
    mode_paths = iter_mode_paths(run_dir, mode)
    request_entries = _load_mode_requests(mode_paths["request_file"], case_id=case_id, limit=limit)
    total_candidates = len(request_entries)
    reparsed_from_raw = 0
    skipped_resume = 0
    if resume:
        pending_entries: list[dict[str, Any]] = []
        for item in request_entries:
            if _should_skip(item, mode_paths["raw_dir"], mode_paths["parsed_dir"]):
                skipped_resume += 1
                continue
            if _reparse_existing_raw(item, mode_paths["raw_dir"], mode_paths["parsed_dir"]):
                reparsed_from_raw += 1
                continue
            pending_entries.append(item)
        request_entries = pending_entries

    failures: list[dict[str, Any]] = []
    success = 0
    failed = 0
    elapsed_samples: list[float] = []
    submitted = len(request_entries)
    rolling_batch_index = 1
    rolling_slot = 0
    total_virtual_batches = (submitted + batch_size - 1) // batch_size if batch_size > 0 else 0

    progress = tqdm(
        total=submitted,
        desc=f"GLM-4.6 rolling requests ({mode})",
        unit="req",
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {
            pool.submit(_post_single, entry, config, api_key): entry
            for entry in request_entries
        }
        for future in as_completed(future_map):
            entry = future_map[future]
            custom_id = str(entry["custom_id"])
            case_part = custom_id.partition("_")[2] or custom_id
            result = future.result()
            elapsed_samples.append(float(result.get("elapsed_seconds", 0.0)))

            raw_payload = {
                "custom_id": custom_id,
                "mode": mode,
                "query_case_id": case_part,
                "request_entry": entry,
                "result": result,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(mode_paths["raw_dir"] / f"{case_part}.json", raw_payload)

            if result.get("ok"):
                response_body = result.get("body", {})
                raw_content = extract_completion_message_content(response_body)
                parsed_json = find_json_candidate(raw_content)
                normalized = normalize_prediction_payload(
                    custom_id=custom_id,
                    raw_content=raw_content,
                    parsed_json=parsed_json,
                )
                normalized["usage"] = response_body.get("usage", {})
                normalized["raw_reasoning_content"] = extract_completion_reasoning_content(response_body)
                normalized["elapsed_seconds"] = result.get("elapsed_seconds")
                write_json(mode_paths["parsed_dir"] / f"{case_part}.json", normalized)
                success += 1
            else:
                failure_row = {
                    "custom_id": custom_id,
                    "mode": mode,
                    "query_case_id": case_part,
                    "status_code": result.get("status_code"),
                    "error": result.get("error"),
                    "text": result.get("text"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                }
                failures.append(failure_row)
                failed += 1

            rolling_slot += 1
            if batch_size > 0 and rolling_slot > batch_size:
                rolling_batch_index += 1
                rolling_slot = 1
            progress.update(1)
            progress.set_postfix(
                virtual_batch=f"{min(rolling_batch_index, max(total_virtual_batches, 1))}/{max(total_virtual_batches, 1)}",
                success=success,
                failed=failed,
                avg_elapsed=f"{(sum(elapsed_samples) / len(elapsed_samples)):.1f}s" if elapsed_samples else "0.0s",
            )
    progress.close()

    write_jsonl(mode_paths["failure_file"], failures)
    summary = {
        "mode": mode,
        "request_total": total_candidates,
        "infer_submitted": submitted,
        "infer_skipped_resume": skipped_resume,
        "infer_reparsed_from_raw": reparsed_from_raw,
        "response_success": success,
        "response_failed": failed,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "mini_batch_count": total_virtual_batches,
        "dispatch_mode": "rolling_concurrency",
        "average_elapsed_seconds": round(sum(elapsed_samples) / len(elapsed_samples), 4) if elapsed_samples else 0.0,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(mode_paths["summary_file"], summary)
    return summary


def main() -> None:
    args = parse_args()
    config = default_report_config()
    if args.batch_size <= 0 or args.concurrency <= 0:
        raise ValueError("batch_size and concurrency must be positive integers.")
    config.direct.model = args.model
    config.direct.endpoint = args.endpoint
    config.direct.max_tokens = args.max_tokens
    config.direct.thinking_type = args.thinking_type

    api_key = _resolve_api_key(args.api_key, config)
    if not api_key:
        raise RuntimeError("Missing ZHIPU_API_KEY or --api-key.")

    run_dir = _resolve_run_dir(config, args.run_name, args.run_dir)
    ensure_eval_dirs(run_dir)
    limit = None if args.limit < 0 else args.limit
    modes = ["rag", "zero"] if args.mode == "all" else [args.mode]
    summaries = []
    for mode in modes:
        summaries.append(
            _process_mode(
                run_dir,
                mode,
                config,
                api_key,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                limit=limit,
                case_id=args.case_id,
                resume=args.resume,
            )
        )
    print(json.dumps({"run_dir": str(run_dir), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
