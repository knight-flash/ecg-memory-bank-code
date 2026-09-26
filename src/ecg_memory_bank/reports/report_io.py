"""Report run artifact layout.

Purpose
-------
Keep request, response, parsed output, and evaluation files in consistent run
directories.

Method
------
Resolve per-mode paths and load JSON/JSONL records with stable UTF-8 handling.

Inputs
------
External run directory, generation mode, and JSONL files.

Outputs
-------
Path maps and parsed record lists; directory creation occurs only when called.

Paper parameters
----------------
The two paired modes are rag and zero; run names distinguish GLM backends.

Shapes and meaning
------------------
Each record corresponds to one ECG case in one generation mode, keyed by its
custom ID.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUEST_FILE_MAP = {
    "rag": "rag_requests.jsonl",
    "zero": "zero_requests.jsonl",
}


def ensure_eval_dirs(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    dirs = {
        "root": root,
        "labels": root / "labels",
        "features": root / "features",
        "subsets": root / "subsets",
        "previews": root / "previews",
        "requests": root / "requests",
        "logs": root / "logs",
        "responses": root / "responses",
        "raw_rag": root / "responses" / "raw" / "rag",
        "raw_zero": root / "responses" / "raw" / "zero",
        "parsed_rag": root / "responses" / "parsed" / "rag",
        "parsed_zero": root / "responses" / "parsed" / "zero",
        "failures": root / "responses" / "failures",
        "metrics": root / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_json(path: str | Path, *, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_mode_paths(run_dir: str | Path, mode: str) -> dict[str, Path]:
    dirs = ensure_eval_dirs(run_dir)
    if mode not in REQUEST_FILE_MAP:
        raise KeyError(f"Unsupported mode: {mode}")
    return {
        "request_file": dirs["requests"] / REQUEST_FILE_MAP[mode],
        "raw_dir": dirs[f"raw_{mode}"],
        "parsed_dir": dirs[f"parsed_{mode}"],
        "failure_file": dirs["failures"] / f"{mode}.jsonl",
        "summary_file": dirs["logs"] / f"infer_summary_{mode}.json",
    }
