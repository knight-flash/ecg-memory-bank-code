"""Report experiment stage orchestrator.

Purpose
-------
Run preparation, GLM inference, and metric evaluation for a single ECG-
Grounding backend.

Method
------
Dispatch label-context, subset, request, inference, and evaluation scripts in
a fixed sequence; model and paths are passed through CLI args.

Inputs
------
External benchmark subset and run directory plus MIMIC embedding/feature
configuration and API access.

Outputs
-------
External requests, raw/parsed responses, metrics, and run summary.

Paper parameters
----------------
Top-level paper runner supplies all 2381 cases, K=5, GLM-4.6 or GLM-5,
thinking enabled, max_tokens=8192.

Shapes and meaning
------------------
One run produces paired RAG/No-RAG records for Q cases; per-label scores form
a [Q,C] matrix.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.config import default_report_config


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Orchestrate prepare/infer/evaluate for MIMIC self-RAG feasibility.")
    parser.add_argument("--stage", choices=["labels", "features", "prepare", "infer", "evaluate", "all"], default="all")
    parser.add_argument("--run-name", type=str, default=config.grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--query-size", type=int, default=config.grounding.query_sample_size)
    parser.add_argument("--gallery-size", type=int, default=config.grounding.gallery_sample_size)
    parser.add_argument("--top-k", type=int, default=config.grounding.top_k)
    parser.add_argument("--query-subset-csv", type=str, default=config.dataset.query_subset_csv)
    parser.add_argument("--prompt-version", type=str, default=config.prompt.prompt_version)
    parser.add_argument("--infer-model", type=str, default=config.direct.model)
    parser.add_argument("--infer-batch-size", type=int, default=config.direct.batch_size)
    parser.add_argument("--infer-concurrency", type=int, default=config.direct.concurrency)
    parser.add_argument("--infer-thinking-type", type=str, default=config.direct.thinking_type)
    parser.add_argument("--infer-max-tokens", type=int, default=config.direct.max_tokens)
    parser.add_argument("--mode", choices=["rag", "zero", "all"], default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--case-id", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key", type=str, default="")
    return parser.parse_args()


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def _run_command(args: list[str]) -> None:
    completed = subprocess.run(args, check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def main() -> None:
    args = parse_args()
    config = default_report_config()
    py = sys.executable
    run_dir_args = ["--run-dir", args.run_dir] if args.run_dir else []
    limit_args = ["--limit", str(args.limit)] if args.limit >= 0 else []
    case_args = ["--case-id", args.case_id] if args.case_id else []
    resume_args = ["--resume"] if args.resume else []
    api_key_args = ["--api-key", args.api_key] if args.api_key else []
    query_subset_args = ["--query-subset-csv", args.query_subset_csv] if args.query_subset_csv else []
    prompt_version_args = ["--prompt-version", args.prompt_version]
    if args.stage in {"prepare", "all"}:
        _run_command(
            [
                py,
                str(_script_path("prepare_label_taxonomy.py")),
                "--run-name",
                args.run_name,
                *run_dir_args,
            ]
        )
        _run_command(
            [
                py,
                str(_script_path("prepare_eval_subsets.py")),
                "--run-name",
                args.run_name,
                "--query-size",
                str(args.query_size),
                "--seed",
                str(config.sampling.random_seed),
                *query_subset_args,
                *run_dir_args,
                *case_args,
            ]
        )
        _run_command(
            [
                py,
                str(_script_path("prepare_report_requests.py")),
                "--run-name",
                args.run_name,
                "--top-k",
                str(args.top_k),
                *prompt_version_args,
                "--model",
                args.infer_model,
                "--thinking-type",
                args.infer_thinking_type,
                "--max-tokens",
                str(args.infer_max_tokens),
                *run_dir_args,
                *case_args,
            ]
        )
    elif args.stage == "labels":
        _run_command(
            [
                py,
                str(_script_path("prepare_label_taxonomy.py")),
                "--run-name",
                args.run_name,
                *run_dir_args,
            ]
        )
    elif args.stage == "features":
        _run_command(
            [
                py,
                str(_script_path("prepare_eval_subsets.py")),
                "--run-name",
                args.run_name,
                "--query-size",
                str(args.query_size),
                "--seed",
                str(config.sampling.random_seed),
                *query_subset_args,
                *run_dir_args,
                *case_args,
            ]
        )

    if args.stage in {"infer", "all"}:
        _run_command(
            [
                py,
                str(_script_path("generate_reports.py")),
                "--run-name",
                args.run_name,
                "--model",
                args.infer_model,
                "--mode",
                args.mode,
                "--batch-size",
                str(args.infer_batch_size),
                "--concurrency",
                str(args.infer_concurrency),
                "--thinking-type",
                args.infer_thinking_type,
                "--max-tokens",
                str(args.infer_max_tokens),
                *run_dir_args,
                *limit_args,
                *case_args,
                *resume_args,
                *api_key_args,
            ]
        )

    if args.stage in {"evaluate", "all"}:
        _run_command(
            [
                py,
                str(_script_path("evaluate_reports.py")),
                "--run-name",
                args.run_name,
                *run_dir_args,
                *case_args,
            ]
        )

    print(
        json.dumps(
            {
                "stage": args.stage,
                "run_name": args.run_name,
                "run_dir": args.run_dir or args.run_name,
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
