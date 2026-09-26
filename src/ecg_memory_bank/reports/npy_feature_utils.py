"""Structured ECG feature serialization.

Purpose
-------
Turn external per-case NumPy feature payloads into compact text-safe summaries
for LLM prompts.

Method
------
Load feature arrays/dictionaries, round numeric values, truncate oversized
content, and expose representative signal measurements.

Inputs
------
FeatureDB .npy or structured payload for one ECG and serialization settings.

Outputs
-------
Compact dictionaries or strings added to query/retrieved-case prompt context.

Paper parameters
----------------
Target length and decimal places come from configuration; no labels are
predicted here.

Shapes and meaning
------------------
Raw leads may be [12,T]; summarized values describe the ECG signal and are
separate from embedding vectors [N,D].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _safe_round(value: Any, digits: int = 6) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _compact_array(array: np.ndarray, *, max_values: int = 512) -> dict[str, Any]:
    flat = array.reshape(-1)
    payload: dict[str, Any] = {
        "kind": "ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(flat.size),
    }
    if flat.size == 0:
        payload["values"] = []
        return payload

    finite = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.number) else np.asarray([], dtype=float)
    if finite.size > 0:
        payload["stats"] = {
            "min": _safe_round(finite.min()),
            "max": _safe_round(finite.max()),
            "mean": _safe_round(finite.mean()),
            "std": _safe_round(finite.std()),
        }

    values = flat.tolist()
    if flat.size <= max_values:
        payload["values"] = [_safe_round(item) for item in values]
    else:
        payload["values_preview"] = [_safe_round(item) for item in values[:max_values]]
        payload["truncated"] = True
    return payload


def _compact_object(value: Any, *, max_values: int = 512) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_object(val, max_values=max_values) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_object(item, max_values=max_values) for item in value]
    if isinstance(value, np.ndarray):
        return _compact_array(value, max_values=max_values)
    if isinstance(value, (np.floating, float, np.integer, int)):
        return _safe_round(value)
    if value is None:
        return None
    return value


def load_feature_payload(
    feature_root: str | Path,
    feature_id: str | int,
    *,
    max_values: int = 512,
) -> dict[str, Any] | None:
    feature_name = str(feature_id).strip()
    if not feature_name:
        return None
    path = Path(feature_root) / f"{feature_name}.npy"
    if not path.exists():
        return None

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        loaded = loaded.item()

    if isinstance(loaded, np.ndarray):
        structured = _compact_array(loaded, max_values=max_values)
    else:
        structured = _compact_object(loaded, max_values=max_values)

    return {
        "feature_id": feature_name,
        "feature_file": str(path),
        "structured_features": structured,
    }


def build_feature_summary(feature_payload: dict[str, Any] | None) -> str:
    if not feature_payload:
        return "Feature file unavailable."
    structured = feature_payload.get("structured_features", {})
    if isinstance(structured, dict):
        kind = structured.get("kind", "object")
        shape = structured.get("shape")
        size = structured.get("size")
        return f"feature_id={feature_payload.get('feature_id')}; kind={kind}; shape={shape}; size={size}."
    return f"feature_id={feature_payload.get('feature_id')}; kind={type(structured).__name__}."
