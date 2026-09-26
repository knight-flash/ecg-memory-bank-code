"""Prototype case audit export.

Purpose
-------
Build inspectable ECG/prototype case records from model activations and
metadata.

Method
------
Resolve case identifiers and waveform paths, attach labels and predicted
contributions, and serialize selected query/prototype examples.

Inputs
------
Prototype projection IDs, per-query scores, metadata rows, optional waveform
path roots.

Outputs
-------
External structured JSON case bundles and audit metadata when called by a
task.

Paper parameters
----------------
Top-K or exemplar counts are set by callers; exports preserve provenance
rather than changing predictions.

Shapes and meaning
------------------
Each JSON case is one ECG record; contribution arrays map P prototype
responses to C diagnosis logits.
"""

import ast
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TIME_FIELDS = (
    "RecordingTime",
    "AcquisitionDate",
    "date_time",
    "time",
    "Date",
    "timestamp",
)
DEFAULT_DIAG_FIELDS = ("labels", "diagnosis", "diag")
DEFAULT_REPORT_FIELDS = ("text", "report")
DEFAULT_AGE_FIELDS = ("Age", "age")
DEFAULT_GENDER_FIELDS = ("Gender", "gender")
DEFAULT_PATH_FIELDS = ("full_ecg_path", "ecg_path", "path", "record_path")
DEFAULT_ID_FIELDS = ("export_case_id", "HashFileName", "id", "case_id", "study_id", "exam_id", "filename")

DEFAULT_RECORD_ROOTS = (
    "__EXTERNAL_DATA_ROOT_REQUIRED__/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    "__ECG_DATA_ROOT_REQUIRED__/mimic",
    "__ECG_DATA_ROOT_REQUIRED__/mimic_iv_ecg",
    "__ECG_DATA_ROOT_REQUIRED__/cinc/code-15/data",
    "__ECG_DATA_ROOT_REQUIRED__/cinc/code-15/holdout_data",
)


def require_wfdb():
    try:
        import wfdb  # type: ignore
    except Exception as exc:  # pragma: no cover - import depends on local env
        raise RuntimeError(
            "wfdb is required for export_case_from_row(), but it could not be imported. "
            "Use export_case_from_payload() when signals are already in memory."
        ) from exc
    return wfdb


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def row_get(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            value = row[key]
            if not is_missing(value):
                return value
    return default


def save_json(path: os.PathLike[str] | str, payload: Any, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def prepare_output_dirs(output_base: os.PathLike[str] | str) -> Tuple[Path, Path]:
    base = Path(output_base)
    cases_dir = base / "cases"
    retrievals_dir = base / "retrievals"
    cases_dir.mkdir(parents=True, exist_ok=True)
    retrievals_dir.mkdir(parents=True, exist_ok=True)
    return cases_dir, retrievals_dir


def load_hierarchy_map(hierarchy_path: os.PathLike[str] | str) -> Dict[str, Dict[str, Any]]:
    with Path(hierarchy_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key).lower().strip(): value for key, value in raw.items()}


def parse_diagnosis_tags(original_text: Any) -> list[str]:
    original = str(original_text).strip()
    raw_tags: list[str] = []

    if original.startswith("[") and original.endswith("]"):
        try:
            parsed = ast.literal_eval(original)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            raw_tags = [str(item).strip() for item in parsed if str(item).strip()]

    if not raw_tags:
        if "|" in original:
            raw_tags = original.split("|")
        elif "," in original:
            raw_tags = original.split(",")
        elif original:
            raw_tags = [original]

    tags: list[str] = []
    for item in raw_tags:
        clean = str(item).strip().strip("'").strip('"')
        if is_missing(clean):
            continue
        if clean not in tags:
            tags.append(clean)

    return tags or ["Unclassified"]


def get_category_priority(category: str) -> int:
    weights = {
        "Ischemia/Infarct": 100,
        "Rhythm": 200,
        "Conduction": 300,
        "Hypertrophy": 400,
        "Ectopic": 500,
        "Other": 600,
        "Global": 900,
        "Uncategorized": 2000,
    }
    return weights.get(category, 2000)


def get_medical_fields(
    diagnosis_tags: Sequence[str],
    hierarchy_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    fallback_name = diagnosis_tags[0] if diagnosis_tags else "Unknown"
    default_fields = {
        "medicalPriority": 5000,
        "medicalLevel": 3,
        "medicalCategory": "Uncategorized",
        "medicalGroup": "unknown",
        "cleanName": fallback_name,
    }
    if not hierarchy_map:
        return default_fields

    best = None
    best_priority = float("inf")
    for diagnosis in diagnosis_tags:
        info = hierarchy_map.get(str(diagnosis).lower().strip())
        if not info:
            continue
        level = int(info.get("level", 3))
        category = str(info.get("category", "Uncategorized"))
        priority = (4 - level) * 1000 + get_category_priority(category)
        if priority < best_priority:
            best_priority = priority
            best = {
                "medicalPriority": priority,
                "medicalLevel": level,
                "medicalCategory": category,
                "medicalGroup": str(info.get("group_id", "unknown")),
                "cleanName": str(info.get("clean_name", fallback_name)),
            }

    return best or default_fields


def normalize_record_base(path_like: os.PathLike[str] | str) -> Path:
    base_path = Path(str(path_like))
    if base_path.suffix in {".hea", ".dat", ".mat"}:
        base_path = base_path.with_suffix("")
    return base_path


def existing_record_base(base_path: os.PathLike[str] | str) -> Optional[Path]:
    base = normalize_record_base(base_path)
    if base.with_suffix(".hea").exists():
        return base
    return None


def iter_fallback_bases(raw_path: str) -> Iterable[Path]:
    yield normalize_record_base(raw_path)

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("files/") or not Path(raw_path).is_absolute():
        for root in DEFAULT_RECORD_ROOTS:
            yield normalize_record_base(Path(root) / normalized)

    if "/data/" in normalized:
        yield normalize_record_base(normalized.replace("/data/", "/holdout_data/"))
    if "/holdout_data/" in normalized:
        yield normalize_record_base(normalized.replace("/holdout_data/", "/data/"))


def iter_basename_search_bases(row: Mapping[str, Any]) -> Iterable[Path]:
    raw_path = row_get(row, *DEFAULT_PATH_FIELDS)
    candidate_names = []
    if not is_missing(raw_path):
        candidate_names.append(os.path.basename(str(raw_path)))
    for key in DEFAULT_ID_FIELDS:
        value = row_get(row, key)
        if is_missing(value):
            continue
        candidate_names.append(str(value))

    clean_names: list[str] = []
    for name in candidate_names:
        base_name = normalize_record_base(name).name
        if base_name and base_name not in clean_names:
            clean_names.append(base_name)

    for base_name in clean_names:
        for root in DEFAULT_RECORD_ROOTS:
            yield normalize_record_base(Path(root) / base_name)


def resolve_record_path(
    row: Mapping[str, Any],
    path_fields: Sequence[str] = DEFAULT_PATH_FIELDS,
) -> Optional[Path]:
    raw_path = row_get(row, *path_fields)
    if not is_missing(raw_path):
        for candidate_base in iter_fallback_bases(str(raw_path)):
            resolved = existing_record_base(candidate_base)
            if resolved:
                return resolved

    for candidate_base in iter_basename_search_bases(row):
        resolved = existing_record_base(candidate_base)
        if resolved:
            return resolved

    return None


def crop_or_pad_signals(
    signals: np.ndarray,
    target_samples: Optional[int],
    pad_to_target: bool,
) -> np.ndarray:
    if not target_samples:
        return signals

    if signals.shape[0] > target_samples:
        return signals[:target_samples, :]

    if pad_to_target and signals.shape[0] < target_samples:
        pad_len = target_samples - signals.shape[0]
        return np.pad(signals, ((0, pad_len), (0, 0)))

    return signals


def signals_to_leads(
    signals: np.ndarray,
    lead_names: Sequence[str],
    decimal_places: int = 3,
) -> Dict[str, list[float]]:
    leads: Dict[str, list[float]] = {}
    for index, lead in enumerate(lead_names):
        leads[str(lead)] = [round(float(value), decimal_places) for value in signals[:, index]]
    return leads


def build_preview_signal(
    leads: Mapping[str, Sequence[float]],
    downsample_ratio: int = 10,
    preferred_lead: str = "I",
) -> list[float]:
    if not leads:
        return []

    lead_name = preferred_lead if preferred_lead in leads else next(iter(leads.keys()))
    values = list(leads[lead_name])
    if downsample_ratio <= 1:
        return values
    return values[::downsample_ratio]


def format_demographics(age: Any, gender: Any) -> str:
    clean_gender = "Unknown" if is_missing(gender) else str(gender)
    clean_age = "??" if is_missing(age) else str(age)
    return f"{clean_gender}, {clean_age} yr"


def extract_export_extra_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    field_map = {
        "prototype_id": "prototypeId",
        "prototype_class_idx": "prototypeClassIdx",
        "prototype_class_name": "prototypeClassName",
        "train_local_index": "trainLocalIndex",
        "train_global_index": "trainGlobalIndex",
        "activation_count": "activationCount",
        "activation_rate": "activationRate",
        "is_prototype_anchor": "isPrototypeAnchor",
    }
    extra_fields: Dict[str, Any] = {}
    for source_key, target_key in field_map.items():
        if source_key not in row:
            continue
        value = row[source_key]
        if is_missing(value):
            continue
        extra_fields[target_key] = value
    return extra_fields


def build_case_detail(
    *,
    case_id: str,
    leads: Mapping[str, Sequence[float]],
    sampling_rate: Any,
    diagnosis: Sequence[str],
    report: str,
    meta: Mapping[str, Any],
    medical_fields: Mapping[str, Any],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    detail = {
        "id": case_id,
        "meta": {
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "time": meta.get("time"),
        },
        "report": report,
        "samplingRate": sampling_rate,
        "leads": dict(leads),
        "diagnosis": list(diagnosis),
        "medicalCategory": medical_fields.get("medicalCategory"),
        "medicalGroup": medical_fields.get("medicalGroup"),
        "cleanName": medical_fields.get("cleanName"),
    }
    if extra_fields:
        detail.update(extra_fields)
    return detail


def build_retrieval_item(
    *,
    case_id: str,
    diagnosis: Sequence[str],
    report: str,
    case_file_name: str,
    similarity: float,
    medical_fields: Mapping[str, Any],
    demographics: str,
    is_query_case: bool,
    preview_signal: Sequence[float],
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    item = {
        "id": case_id,
        "fileName": case_file_name,
        "medicalPriority": medical_fields.get("medicalPriority"),
        "medicalLevel": medical_fields.get("medicalLevel"),
        "medicalCategory": medical_fields.get("medicalCategory"),
        "medicalGroup": medical_fields.get("medicalGroup"),
        "cleanName": medical_fields.get("cleanName"),
        "isQueryCase": is_query_case,
        "diagnosis": list(diagnosis),
        "report": report,
        "demographics": demographics,
        "similarity": round(float(similarity), 4),
        "previewSignal": list(preview_signal),
    }
    if extra_fields:
        item.update(extra_fields)
    return item


def build_manifest_item(
    *,
    case_id: str,
    diagnosis: Sequence[str],
    clean_name: str,
    retrieval_file_name: str,
    timestamp: Any,
    case_file_name: Optional[str] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": case_id,
        "cleanName": clean_name,
        "diagnosis": list(diagnosis),
        "fileName": case_file_name or f"cases/{case_id}.json",
        "retrievalFile": retrieval_file_name,
        "timestamp": "Unknown" if is_missing(timestamp) else str(timestamp),
    }
    if extra_fields:
        item.update(extra_fields)
    return item


def export_case_from_payload(
    *,
    case_id: str,
    leads: Mapping[str, Sequence[float]],
    sampling_rate: Any,
    diagnosis: Sequence[str] | str,
    report: str = "No report available.",
    age: Any = None,
    gender: Any = None,
    timestamp: Any = None,
    output_cases_dir: os.PathLike[str] | str,
    similarity: float = 1.0,
    is_query_case: bool = False,
    hierarchy_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    downsample_ratio: int = 10,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    diagnosis_tags = (
        parse_diagnosis_tags(diagnosis) if not isinstance(diagnosis, (list, tuple)) else list(diagnosis)
    )
    medical_fields = get_medical_fields(diagnosis_tags, hierarchy_map)
    detail = build_case_detail(
        case_id=case_id,
        leads=leads,
        sampling_rate=sampling_rate,
        diagnosis=diagnosis_tags,
        report=report,
        meta={
            "age": 60 if is_missing(age) else age,
            "gender": "Unknown" if is_missing(gender) else gender,
            "time": "Unknown" if is_missing(timestamp) else str(timestamp),
        },
        medical_fields=medical_fields,
        extra_fields=extra_fields,
    )

    case_file_name = f"cases/{case_id}.json"
    case_output_path = Path(output_cases_dir) / f"{case_id}.json"
    save_json(case_output_path, detail)

    item = build_retrieval_item(
        case_id=case_id,
        diagnosis=diagnosis_tags,
        report=report,
        case_file_name=case_file_name,
        similarity=similarity,
        medical_fields=medical_fields,
        demographics=format_demographics(detail["meta"]["age"], detail["meta"]["gender"]),
        is_query_case=is_query_case,
        preview_signal=build_preview_signal(leads, downsample_ratio=downsample_ratio),
        extra_fields=extra_fields,
    )
    return {"detail": detail, "item": item, "output_path": str(case_output_path)}


def export_case_from_row(
    row: Mapping[str, Any],
    *,
    output_cases_dir: os.PathLike[str] | str,
    similarity: float,
    query_case_id: Optional[str] = None,
    hierarchy_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    target_samples: Optional[int] = 5000,
    pad_to_target: bool = True,
    decimal_places: int = 3,
    downsample_ratio: int = 10,
    full_signal: bool = False,
) -> Optional[Dict[str, Any]]:
    wfdb = require_wfdb()
    record_path = resolve_record_path(row)
    if not record_path:
        row_id = row_get(row, *DEFAULT_ID_FIELDS, default="unknown")
        raw_path = row_get(row, *DEFAULT_PATH_FIELDS, default="missing")
        print(f"[JSON Export] Skip row {row_id}: record path not resolved from {raw_path}")
        return None

    sampto = None if full_signal or not target_samples else target_samples
    try:
        if sampto is None:
            record = wfdb.rdrecord(str(record_path))
        else:
            try:
                record = wfdb.rdrecord(str(record_path), sampto=sampto)
            except ValueError as exc:
                if "sampto must be shorter than the signal length" in str(exc):
                    record = wfdb.rdrecord(str(record_path))
                else:
                    raise
    except Exception as exc:
        print(f"[JSON Export] Failed to read record {record_path}: {exc}")
        return None

    signals = np.nan_to_num(record.p_signal)
    if not full_signal:
        signals = crop_or_pad_signals(signals, target_samples=target_samples, pad_to_target=pad_to_target)

    case_id = str(row_get(row, *DEFAULT_ID_FIELDS))
    diagnosis_tags = parse_diagnosis_tags(row_get(row, *DEFAULT_DIAG_FIELDS, default="Unknown"))
    report = str(row_get(row, *DEFAULT_REPORT_FIELDS, default="No report available."))
    age = row_get(row, *DEFAULT_AGE_FIELDS)
    gender = row_get(row, *DEFAULT_GENDER_FIELDS, default="Unknown")
    timestamp = row_get(row, *DEFAULT_TIME_FIELDS, default="Unknown")

    if is_missing(age):
        age = random.randint(40, 90)
    else:
        age = int(float(age))

    if is_missing(gender):
        gender = random.choice(["Male", "Female"])

    leads = signals_to_leads(signals, record.sig_name, decimal_places=decimal_places)
    extra_fields = extract_export_extra_fields(row)
    return export_case_from_payload(
        case_id=case_id,
        leads=leads,
        sampling_rate=record.fs,
        diagnosis=diagnosis_tags,
        report=report,
        age=age,
        gender=gender,
        timestamp=timestamp,
        output_cases_dir=output_cases_dir,
        similarity=similarity,
        is_query_case=str(case_id) == str(query_case_id),
        hierarchy_map=hierarchy_map,
        downsample_ratio=downsample_ratio,
        extra_fields=extra_fields,
    )


def save_retrieval_list(
    retrievals_dir: os.PathLike[str] | str,
    query_case_id: str,
    retrieval_items: Sequence[Mapping[str, Any]],
) -> Path:
    file_name = f"retrieval_{query_case_id}.json"
    output_path = Path(retrievals_dir) / file_name
    save_json(output_path, list(retrieval_items))
    return output_path


def export_query_bundle(
    *,
    query_row: Mapping[str, Any],
    retrieval_rows_with_scores: Iterable[Tuple[Mapping[str, Any], float]],
    output_base: os.PathLike[str] | str,
    hierarchy_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    target_samples: Optional[int] = 5000,
    pad_to_target: bool = True,
    decimal_places: int = 3,
    downsample_ratio: int = 10,
    full_signal: bool = False,
    include_query_in_retrievals: bool = True,
    manifest_items: Optional[list[Dict[str, Any]]] = None,
    manifest_extra_fields: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cases_dir, retrievals_dir = prepare_output_dirs(output_base)
    query_case_id = str(row_get(query_row, *DEFAULT_ID_FIELDS))

    query_export = export_case_from_row(
        query_row,
        output_cases_dir=cases_dir,
        similarity=1.0,
        query_case_id=query_case_id,
        hierarchy_map=hierarchy_map,
        target_samples=target_samples,
        pad_to_target=pad_to_target,
        decimal_places=decimal_places,
        downsample_ratio=downsample_ratio,
        full_signal=full_signal,
    )
    if not query_export:
        return None

    retrieval_items = []
    if include_query_in_retrievals:
        retrieval_items.append(query_export["item"])

    for row, score in retrieval_rows_with_scores:
        exported = export_case_from_row(
            row,
            output_cases_dir=cases_dir,
            similarity=score,
            query_case_id=query_case_id,
            hierarchy_map=hierarchy_map,
            target_samples=target_samples,
            pad_to_target=pad_to_target,
            decimal_places=decimal_places,
            downsample_ratio=downsample_ratio,
            full_signal=full_signal,
        )
        if exported:
            retrieval_items.append(exported["item"])

    retrieval_path = save_retrieval_list(retrievals_dir, query_case_id, retrieval_items)
    manifest_item = build_manifest_item(
        case_id=query_case_id,
        diagnosis=query_export["detail"]["diagnosis"],
        clean_name=query_export["detail"]["cleanName"],
        retrieval_file_name=f"retrievals/{retrieval_path.name}",
        timestamp=query_export["detail"]["meta"]["time"],
        extra_fields=manifest_extra_fields,
    )
    if manifest_items is not None:
        manifest_items.append(manifest_item)

    return {
        "query": query_export,
        "retrievals": retrieval_items,
        "manifest_item": manifest_item,
        "retrieval_path": str(retrieval_path),
    }
