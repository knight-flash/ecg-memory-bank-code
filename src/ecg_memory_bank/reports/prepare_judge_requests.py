"""DeepSeek judge request preparation.

Purpose
-------
Create retrieval-neutral scoring requests for the finalized report
configurations.

Method
------
Pair report text with ground truth under the same rubric, assign stable custom
IDs, and write chat request bodies.

Inputs
------
Best-of/retry report directories, ECG-Grounding ground truth JSONL, and shared
judge prompt template.

Outputs
-------
External judge requests JSONL and preparation manifest.

Paper parameters
----------------
Paper protocol sets temperature=0, thinking disabled, max_tokens=8192
downstream, and one rubric for all four model/mode groups.

Shapes and meaning
------------------
A request is one generated report for one ECG; scores are per rubric
criterion, not the [Q,C] diagnostic label scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.judge_io import (
    DEFAULT_SELECTION_RELATIVE_PATH,
    DEFAULT_SERVER_BESTOF_DIR,
    DEFAULT_SERVER_GENERATE_OUTPUT_DIR,
    DEFAULT_SERVER_GROUNDTRUTH_JSONL,
    DEFAULT_SERVER_PROMPT_TEMPLATE,
    DEFAULT_SERVER_RETRY_DIR,
    DEFAULT_SERVER_RUN_DIR,
    format_generated_text,
    load_ground_truth_map,
    load_selection_rows,
    render_prompt,
    resolve_source_record_path,
)


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path: str | Path, payload: Any) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, items: list[dict[str, Any]]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Volcengine batch inputs for MIMIC reasoning_process evaluation.")
    parser.add_argument("--bestof-dir", type=str, default=DEFAULT_SERVER_BESTOF_DIR)
    parser.add_argument("--retry-dir", type=str, default=DEFAULT_SERVER_RETRY_DIR)
    parser.add_argument("--run-dir", type=str, default=DEFAULT_SERVER_RUN_DIR)
    parser.add_argument(
        "--selection-csv",
        type=str,
        default="",
        help="Optional override for the best-of selection CSV. Defaults to reports/best_retry_selection_original_first.csv under --bestof-dir.",
    )
    parser.add_argument(
        "--groundtruth-jsonl",
        type=str,
        default=DEFAULT_SERVER_GROUNDTRUTH_JSONL,
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_SERVER_PROMPT_TEMPLATE,
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_SERVER_GENERATE_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["rag", "zero", "all"], default="all")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-type", type=str, default="disabled")
    return parser.parse_args()


def _selection_csv_path(bestof_dir: str, selection_csv_arg: str) -> Path:
    if selection_csv_arg:
        return Path(selection_csv_arg)
    return Path(bestof_dir) / DEFAULT_SELECTION_RELATIVE_PATH


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bestof_dir = Path(args.bestof_dir)
    retry_dir = Path(args.retry_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    selection_csv = _selection_csv_path(args.bestof_dir, args.selection_csv)

    if not selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV not found: {selection_csv}")
    if not Path(args.groundtruth_jsonl).exists():
        raise FileNotFoundError(f"Ground truth file not found: {args.groundtruth_jsonl}")
    if not Path(args.prompt_template).exists():
        raise FileNotFoundError(f"Prompt template not found: {args.prompt_template}")

    selection_rows = load_selection_rows(selection_csv, args.mode)
    if args.limit >= 0:
        selection_rows = selection_rows[: args.limit]

    ground_truth_map = load_ground_truth_map(args.groundtruth_jsonl)
    prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")

    batch_items: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    missing_ground_truth_ids: list[str] = []

    for row in selection_rows:
        case_id = str(row["case_id"])
        mode = str(row["mode"])
        selected_from = str(row["selected_from"])
        source_path = resolve_source_record_path(
            run_dir=run_dir,
            retry_dir=retry_dir,
            mode=mode,
            case_id=case_id,
            selected_from=selected_from,
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source record for {mode}_{case_id}: {source_path}")
        if case_id not in ground_truth_map:
            missing_ground_truth_ids.append(case_id)
            continue

        record = json.loads(source_path.read_text(encoding="utf-8"))
        generated_text, reasoning_source, used_fallback = format_generated_text(record)
        groundtruth_text = ground_truth_map[case_id]["groundtruth_text"]
        prompt_text = render_prompt(prompt_template, generated_text, groundtruth_text)
        custom_id = f"{mode}_{case_id}"
        batch_items.append(
            {
                "custom_id": custom_id,
                "body": {
                    "messages": [{"role": "user", "content": prompt_text}],
                    "top_p": None,
                    "temperature": args.temperature,
                    "thinking": {"type": args.thinking_type},
                },
            }
        )
        alignment_rows.append(
            {
                "custom_id": custom_id,
                "case_id": case_id,
                "mode": mode,
                "selected_from": selected_from,
                "source_record_path": str(source_path),
                "reasoning_source": reasoning_source,
                "used_reasoning_fallback": int(used_fallback),
                "reasoning_process_length": len(str(record.get("reasoning_process", "") or "").strip()),
                "raw_reasoning_content_length": len(str(record.get("raw_reasoning_content", "") or "").strip()),
                "generated_text_length": len(generated_text),
                "groundtruth_text_length": len(groundtruth_text),
                "predicted_labels_count": len(list(record.get("predicted_labels", []))),
            }
        )

    if missing_ground_truth_ids:
        raise KeyError(
            "Missing ground truth entries for case ids: "
            + ", ".join(missing_ground_truth_ids[:20])
            + (" ..." if len(missing_ground_truth_ids) > 20 else "")
        )

    output_prefix = f"mimic_reasoning_eval_{args.mode}"
    batch_path = output_dir / f"{output_prefix}.jsonl"
    alignment_csv_path = output_dir / f"{output_prefix}_alignment.csv"
    alignment_json_path = output_dir / f"{output_prefix}_alignment.json"

    write_jsonl(batch_path, batch_items)
    _write_csv(
        alignment_csv_path,
        alignment_rows,
        [
            "custom_id",
            "case_id",
            "mode",
            "selected_from",
            "source_record_path",
            "reasoning_source",
            "used_reasoning_fallback",
            "reasoning_process_length",
            "raw_reasoning_content_length",
            "generated_text_length",
            "groundtruth_text_length",
            "predicted_labels_count",
        ],
    )
    write_json(
        alignment_json_path,
        {
            "bestof_dir": str(bestof_dir),
            "retry_dir": str(retry_dir),
            "run_dir": str(run_dir),
            "selection_csv": str(selection_csv),
            "groundtruth_jsonl": str(args.groundtruth_jsonl),
            "prompt_template": str(args.prompt_template),
            "output_batch_jsonl": str(batch_path),
            "output_alignment_csv": str(alignment_csv_path),
            "mode": args.mode,
            "request_count": len(batch_items),
            "reasoning_fallback_count": sum(int(row["used_reasoning_fallback"]) for row in alignment_rows),
            "selected_from_counts": {
                key: sum(1 for row in alignment_rows if row["selected_from"] == key)
                for key in ("base", "original", "sparse")
            },
        },
    )

    print(
        json.dumps(
            {
                "batch_jsonl": str(batch_path),
                "alignment_csv": str(alignment_csv_path),
                "request_count": len(batch_items),
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
