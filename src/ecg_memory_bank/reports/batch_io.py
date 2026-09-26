"""Report request and JSONL I/O.

Purpose
-------
Construct stable batch records and write or read run-local JSON/JSONL
artifacts.

Method
------
Attach custom case IDs, model parameters, temperature, and chat messages to
each API request entry.

Inputs
------
One prompt/message collection per case and an external destination path.

Outputs
-------
Request JSONL lines and small JSON summaries.

Paper parameters
----------------
Generation model and temperature are supplied by callers; this helper does not
choose the patient cohort.

Shapes and meaning
------------------
Each JSONL line identifies one query/mode; score arrays are supplied by
downstream model responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def build_batch_entry(
    *,
    custom_id: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
    }
