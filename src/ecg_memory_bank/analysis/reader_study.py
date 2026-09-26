from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


CASE_METRICS = (
    "relevance",
    "accuracy",
    "completeness",
    "reasoning",
    "novelty",
    "clinical_value",
    "satisfaction",
)
MODELS = ("GLM5_RAG", "GLM5_ZERO")
SLOTS = ("model_a", "model_b")
SUPPORT_TIER_DEFINITIONS: tuple[tuple[str, int, int | None], ...] = (
    ("tail", 1, 1000),
    ("low_mid", 1001, 5000),
    ("mid", 5001, 10000),
    ("head", 10001, None),
)
SUPPORT_TIER_BOUNDS = {
    tier_name: {"support_min": min_support, "support_max": max_support}
    for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_batch_key(path: Path) -> str | None:
    match = re.search(r"batch_(\d+)", path.stem)
    return f"batch_{match.group(1)}" if match else None


def parse_score(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        score = int(text)
    except ValueError:
        return None
    return score if 1 <= score <= 5 else None


def parse_reference_count(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_label_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def support_tier_for_count(count: int | None) -> str:
    if count is None:
        return ""
    for tier_name, min_support, max_support in SUPPORT_TIER_DEFINITIONS:
        if count >= min_support and (max_support is None or count <= max_support):
            return tier_name
    return ""


def safe_float(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((item - m) ** 2 for item in values) / (len(values) - 1))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def describe(values: list[int | float]) -> dict[str, Any]:
    clean = [float(item) for item in values if item is not None]
    return {
        "n": len(clean),
        "mean": mean(clean) if clean else None,
        "sd": sample_sd(clean),
        "median": median(clean) if clean else None,
        "q1": quantile(clean, 0.25),
        "q3": quantile(clean, 0.75),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def sign_test_p_two_sided(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    smaller = min(positive, negative)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_summary(diff_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [float(row["diff"]) for row in diff_rows]
    pos = sum(1 for item in diffs if item > 0)
    neg = sum(1 for item in diffs if item < 0)
    ties = sum(1 for item in diffs if item == 0)
    desc = describe(diffs)
    return {
        "n_pairs": len(diffs),
        "rag_wins": pos,
        "zero_wins": neg,
        "ties": ties,
        "mean_diff_rag_minus_zero": desc["mean"],
        "median_diff_rag_minus_zero": desc["median"],
        "sd_diff": desc["sd"],
        "sign_test_p_two_sided": sign_test_p_two_sided(pos, neg),
    }


def find_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("blinded_ab_scores_round2_batch_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def load_mapping(mapping_file: Path) -> dict[str, Any]:
    payload = load_json(mapping_file)
    by_case: dict[str, dict[str, Any]] = {}
    for batch_key, cases in payload.get("batches", {}).items():
        for sample_id, item in cases.items():
            by_case[str(sample_id)] = {"batch_key": batch_key, **item}
    return {"payload": payload, "by_case": by_case}


def load_label_map(label_map_file: Path | None) -> dict[str, dict[str, str]]:
    if not label_map_file or not label_map_file.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with label_map_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["web_label"]] = row
    return out


def load_label_counts(label_count_file: Path | None) -> dict[str, dict[str, Any]]:
    if not label_count_file or not label_count_file.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with label_count_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapped_label = row.get("mapped_label", "")
            key = normalize_label_key(mapped_label)
            if not key:
                continue
            reference_count = parse_reference_count(row.get("count"))
            out[key] = {
                "reference_category": row.get("category", ""),
                "reference_sub_group": row.get("sub_group", ""),
                "reference_raw_label": row.get("raw_label", ""),
                "reference_mapped_label": mapped_label,
                "reference_count": reference_count,
                "support_tier": support_tier_for_count(reference_count),
            }
    return out


def label_support_metadata(map_row: dict[str, str], label_counts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched_label = (map_row.get("matched_mapped_label") or "").strip()
    count_row = label_counts.get(normalize_label_key(matched_label), {}) if matched_label else {}
    reference_count = count_row.get("reference_count")
    if reference_count is None:
        reference_count = parse_reference_count(map_row.get("reference_count"))

    return {
        "reference_category": count_row.get("reference_category", ""),
        "reference_sub_group": count_row.get("reference_sub_group", ""),
        "reference_count": reference_count if reference_count is not None else "",
        "support_tier": count_row.get("support_tier") or support_tier_for_count(reference_count),
    }


def infer_rater_id(path: Path, explicit_rater_id: str | None) -> str:
    if explicit_rater_id:
        return explicit_rater_id
    parent_hint = path.parent.name
    return parent_hint if parent_hint and parent_hint != "evaluation_file_dixinxin" else path.stem


def decode_model(slot: str, mapping_item: dict[str, Any]) -> str:
    model = mapping_item.get(slot)
    if model not in MODELS:
        raise ValueError(f"Unexpected mapping for {slot}: {model}")
    return model


def row_model_prefix(model: str) -> str:
    return "glm5_rag" if model == "GLM5_RAG" else "glm5_zero"


def analyze(
    files: list[Path],
    *,
    mapping_file: Path,
    label_map_file: Path | None,
    label_count_file: Path | None,
    rater_id: str | None,
) -> dict[str, Any]:
    mapping = load_mapping(mapping_file)
    mapping_by_case = mapping["by_case"]
    label_map = load_label_map(label_map_file)
    label_counts = load_label_counts(label_count_file)

    per_case_rows: list[dict[str, Any]] = []
    case_long_rows: list[dict[str, Any]] = []
    case_diff_rows: list[dict[str, Any]] = []
    label_long_rows: list[dict[str, Any]] = []
    label_diff_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    missing_mapping: list[str] = []

    for file_path in files:
        payload = load_json(file_path)
        batch_key_from_file = detect_batch_key(file_path)
        source_file = str(file_path)
        source_rater_id = infer_rater_id(file_path, rater_id)
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path} should contain a top-level object keyed by sample id")

        for case_key, case_payload in payload.items():
            sample_id = str(case_payload.get("id") or case_key)
            mapping_item = mapping_by_case.get(sample_id)
            if not mapping_item:
                missing_mapping.append(sample_id)
                continue

            batch_key = batch_key_from_file or mapping_item.get("batch_key", "")
            shown_order = mapping_item.get("shown_order", "")
            model_for_slot = {slot: decode_model(slot, mapping_item) for slot in SLOTS}
            slot_for_model = {model: slot for slot, model in model_for_slot.items()}

            per_case_row: dict[str, Any] = {
                "rater_id": source_rater_id,
                "source_file": source_file,
                "batch_key": batch_key,
                "shown_order": shown_order,
                "sample_id": sample_id,
                "model_a_true_model": model_for_slot["model_a"],
                "model_b_true_model": model_for_slot["model_b"],
                "label_count": len(case_payload.get("labels", {}) or {}),
                "label_audit_valid": (case_payload.get("label_audit") or {}).get("valid", ""),
                "label_audit_comment": (case_payload.get("label_audit") or {}).get("comment", ""),
            }

            scores_by_model: dict[str, dict[str, int | None]] = {model: {} for model in MODELS}
            for slot in SLOTS:
                true_model = model_for_slot[slot]
                slot_payload = case_payload.get(slot) or {}
                case_scores = slot_payload.get("case_level") or {}
                for metric in CASE_METRICS:
                    score = parse_score(case_scores.get(metric))
                    scores_by_model[true_model][metric] = score
                    per_case_row[f"{slot}_{metric}"] = score if score is not None else ""
                    per_case_row[f"{row_model_prefix(true_model)}_{metric}"] = score if score is not None else ""
                    if score is not None:
                        case_long_rows.append(
                            {
                                "rater_id": source_rater_id,
                                "source_file": source_file,
                                "batch_key": batch_key,
                                "shown_order": shown_order,
                                "sample_id": sample_id,
                                "slot": slot,
                                "model": true_model,
                                "metric": metric,
                                "score": score,
                            }
                        )

            for metric in CASE_METRICS:
                rag_score = scores_by_model["GLM5_RAG"].get(metric)
                zero_score = scores_by_model["GLM5_ZERO"].get(metric)
                if rag_score is not None and zero_score is not None:
                    diff = rag_score - zero_score
                    per_case_row[f"diff_rag_minus_zero_{metric}"] = diff
                    case_diff_rows.append(
                        {
                            "rater_id": source_rater_id,
                            "source_file": source_file,
                            "batch_key": batch_key,
                            "shown_order": shown_order,
                            "sample_id": sample_id,
                            "metric": metric,
                            "glm5_rag": rag_score,
                            "glm5_zero": zero_score,
                            "diff": diff,
                        }
                    )
                else:
                    per_case_row[f"diff_rag_minus_zero_{metric}"] = ""

            labels = case_payload.get("labels") or {}
            label_scores_by_model: dict[tuple[str, str], int | None] = {}
            for slot in SLOTS:
                true_model = model_for_slot[slot]
                slot_payload = case_payload.get(slot) or {}
                label_level = slot_payload.get("label_level") or {}
                for label_index, label_item in label_level.items():
                    label = str(label_item.get("label") or labels.get(label_index) or "")
                    score = parse_score(label_item.get("clinical_value"))
                    map_row = label_map.get(label, {})
                    support_meta = label_support_metadata(map_row, label_counts)
                    label_scores_by_model[(true_model, str(label_index))] = score
                    if score is not None:
                        label_long_rows.append(
                            {
                                "rater_id": source_rater_id,
                                "source_file": source_file,
                                "batch_key": batch_key,
                                "shown_order": shown_order,
                                "sample_id": sample_id,
                                "label_index": label_index,
                                "web_label": label,
                                "match_status": map_row.get("match_status", ""),
                                "matched_category": map_row.get("matched_category", ""),
                                "matched_mapped_label": map_row.get("matched_mapped_label", ""),
                                "reference_category": support_meta["reference_category"],
                                "reference_sub_group": support_meta["reference_sub_group"],
                                "reference_count": support_meta["reference_count"],
                                "support_tier": support_meta["support_tier"],
                                "slot": slot,
                                "model": true_model,
                                "clinical_value": score,
                            }
                        )

            for label_index, label in labels.items():
                label = str(label)
                rag_score = label_scores_by_model.get(("GLM5_RAG", str(label_index)))
                zero_score = label_scores_by_model.get(("GLM5_ZERO", str(label_index)))
                if rag_score is None or zero_score is None:
                    continue
                map_row = label_map.get(label, {})
                support_meta = label_support_metadata(map_row, label_counts)
                label_diff_rows.append(
                    {
                        "rater_id": source_rater_id,
                        "source_file": source_file,
                        "batch_key": batch_key,
                        "shown_order": shown_order,
                        "sample_id": sample_id,
                        "label_index": label_index,
                        "web_label": label,
                        "match_status": map_row.get("match_status", ""),
                        "matched_category": map_row.get("matched_category", ""),
                        "matched_mapped_label": map_row.get("matched_mapped_label", ""),
                        "reference_category": support_meta["reference_category"],
                        "reference_sub_group": support_meta["reference_sub_group"],
                        "reference_count": support_meta["reference_count"],
                        "support_tier": support_meta["support_tier"],
                        "glm5_rag": rag_score,
                        "glm5_zero": zero_score,
                        "diff": rag_score - zero_score,
                    }
                )

            audit = case_payload.get("label_audit") or {}
            audit_rows.append(
                {
                    "rater_id": source_rater_id,
                    "source_file": source_file,
                    "batch_key": batch_key,
                    "shown_order": shown_order,
                    "sample_id": sample_id,
                    "valid": audit.get("valid", ""),
                    "comment": audit.get("comment", ""),
                    "label_count": len(labels),
                }
            )
            per_case_rows.append(per_case_row)

    if missing_mapping:
        raise ValueError(f"Missing A/B mapping for sample ids: {sorted(set(missing_mapping))}")

    return {
        "per_case_rows": per_case_rows,
        "case_long_rows": case_long_rows,
        "case_diff_rows": case_diff_rows,
        "label_long_rows": label_long_rows,
        "label_diff_rows": label_diff_rows,
        "audit_rows": audit_rows,
    }


def summarize_case_by_model(case_long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in case_long_rows:
        groups[(row["model"], row["metric"])].append(int(row["score"]))

    out: list[dict[str, Any]] = []
    for (model, metric), values in sorted(groups.items()):
        desc = describe(values)
        out.append(
            {
                "model": model,
                "metric": metric,
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in desc.items()},
            }
        )
    return out


def summarize_case_diffs(case_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_values: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in case_diff_rows:
        groups[row["metric"]].append(row)
        model_values[(row["metric"], "GLM5_RAG")].append(int(row["glm5_rag"]))
        model_values[(row["metric"], "GLM5_ZERO")].append(int(row["glm5_zero"]))

    out: list[dict[str, Any]] = []
    for metric, rows in sorted(groups.items()):
        summary = paired_summary(rows)
        rag_desc = describe(model_values[(metric, "GLM5_RAG")])
        zero_desc = describe(model_values[(metric, "GLM5_ZERO")])
        out.append(
            {
                "metric": metric,
                "n_pairs": summary["n_pairs"],
                "glm5_rag_mean": safe_float(rag_desc["mean"]),
                "glm5_rag_median": safe_float(rag_desc["median"]),
                "glm5_zero_mean": safe_float(zero_desc["mean"]),
                "glm5_zero_median": safe_float(zero_desc["median"]),
                "mean_diff_rag_minus_zero": safe_float(summary["mean_diff_rag_minus_zero"]),
                "median_diff_rag_minus_zero": safe_float(summary["median_diff_rag_minus_zero"]),
                "sd_diff": safe_float(summary["sd_diff"]),
                "rag_wins": summary["rag_wins"],
                "zero_wins": summary["zero_wins"],
                "ties": summary["ties"],
                "sign_test_p_two_sided": safe_float(summary["sign_test_p_two_sided"]),
            }
        )
    return out


def summarize_label_by_model(label_long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in label_long_rows:
        category = row.get("matched_category") or "UNMAPPED"
        groups[(category, row["model"], "clinical_value")].append(int(row["clinical_value"]))
        groups[("ALL", row["model"], "clinical_value")].append(int(row["clinical_value"]))

    out: list[dict[str, Any]] = []
    for (category, model, metric), values in sorted(groups.items()):
        desc = describe(values)
        out.append(
            {
                "group": category,
                "model": model,
                "metric": metric,
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in desc.items()},
            }
        )
    return out


def summarize_label_by_support_tier(label_long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in label_long_rows:
        tier = row.get("support_tier") or "UNMAPPED"
        groups[(tier, row["model"], "clinical_value")].append(int(row["clinical_value"]))
        if row.get("matched_category") not in ("Rhythm", "General report"):
            groups[(f"{tier}|diagnostic_non_rhythm_general", row["model"], "clinical_value")].append(
                int(row["clinical_value"])
            )

    out: list[dict[str, Any]] = []
    tier_order = {tier_name: index for index, (tier_name, _, _) in enumerate(SUPPORT_TIER_DEFINITIONS)}
    tier_order["UNMAPPED"] = len(tier_order)

    def sort_key(item: tuple[str, str, str]) -> tuple[int, str, str]:
        tier, model, metric = item
        base_tier = tier.split("|", 1)[0]
        return (tier_order.get(base_tier, 999), tier, model + metric)

    for (tier, model, metric), values in sorted(groups.items(), key=lambda item: sort_key(item[0])):
        desc = describe(values)
        base_tier = tier.split("|", 1)[0]
        bounds = SUPPORT_TIER_BOUNDS.get(base_tier, {})
        out.append(
            {
                "support_tier": tier,
                "support_min": bounds.get("support_min", ""),
                "support_max": bounds.get("support_max", ""),
                "model": model,
                "metric": metric,
                **{key: safe_float(value) if isinstance(value, float) else value for key, value in desc.items()},
            }
        )
    return out


def summarize_label_diffs(label_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groupings: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in label_diff_rows:
        groupings[("ALL", "ALL")].append(row)
        groupings[("category", row.get("matched_category") or "UNMAPPED")].append(row)
        groupings[("mapped_label", row.get("matched_mapped_label") or row.get("web_label") or "UNMAPPED")].append(row)

    out: list[dict[str, Any]] = []
    for (group_type, group_value), rows in sorted(groupings.items()):
        summary = paired_summary(rows)
        rag_desc = describe([int(row["glm5_rag"]) for row in rows])
        zero_desc = describe([int(row["glm5_zero"]) for row in rows])
        out.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "n_pairs": summary["n_pairs"],
                "glm5_rag_mean": safe_float(rag_desc["mean"]),
                "glm5_rag_median": safe_float(rag_desc["median"]),
                "glm5_zero_mean": safe_float(zero_desc["mean"]),
                "glm5_zero_median": safe_float(zero_desc["median"]),
                "mean_diff_rag_minus_zero": safe_float(summary["mean_diff_rag_minus_zero"]),
                "median_diff_rag_minus_zero": safe_float(summary["median_diff_rag_minus_zero"]),
                "sd_diff": safe_float(summary["sd_diff"]),
                "rag_wins": summary["rag_wins"],
                "zero_wins": summary["zero_wins"],
                "ties": summary["ties"],
                "sign_test_p_two_sided": safe_float(summary["sign_test_p_two_sided"]),
            }
        )
    return out


def reference_support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_to_count: dict[str, int] = {}
    for row in rows:
        label = str(row.get("matched_mapped_label") or row.get("web_label") or "").strip()
        reference_count = parse_reference_count(row.get("reference_count"))
        if label and reference_count is not None:
            label_to_count[label] = reference_count
    counts = list(label_to_count.values())
    return {
        "n_labels_with_reference_count": len(counts),
        "reference_support_unique_sum": sum(counts) if counts else "",
        "reference_support_min": min(counts) if counts else "",
        "reference_support_median": median(counts) if counts else "",
        "reference_support_max": max(counts) if counts else "",
    }


def summarize_label_support_tier_diffs(label_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groupings: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in label_diff_rows:
        tier = row.get("support_tier") or "UNMAPPED"
        category = row.get("matched_category") or "UNMAPPED"
        groupings[("support_tier", tier)].append(row)
        groupings[("support_tier_category", f"{tier}|{category}")].append(row)
        if category not in ("Rhythm", "General report"):
            groupings[("support_tier_diagnostic_non_rhythm_general", tier)].append(row)

    out: list[dict[str, Any]] = []
    tier_order = {tier_name: index for index, (tier_name, _, _) in enumerate(SUPPORT_TIER_DEFINITIONS)}
    tier_order["UNMAPPED"] = len(tier_order)

    def sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
        group_type, group_value = item
        tier = group_value.split("|", 1)[0]
        return (group_type, tier_order.get(tier, 999), group_value)

    for (group_type, group_value), rows in sorted(groupings.items(), key=lambda item: sort_key(item[0])):
        summary = paired_summary(rows)
        rag_desc = describe([int(row["glm5_rag"]) for row in rows])
        zero_desc = describe([int(row["glm5_zero"]) for row in rows])
        tier = group_value.split("|", 1)[0]
        bounds = SUPPORT_TIER_BOUNDS.get(tier, {})
        unique_labels = sorted({row.get("matched_mapped_label") or row.get("web_label") or "" for row in rows})
        support_summary = reference_support_summary(rows)
        out.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "support_min": bounds.get("support_min", ""),
                "support_max": bounds.get("support_max", ""),
                "n_pairs": summary["n_pairs"],
                "n_unique_labels": len([label for label in unique_labels if label]),
                **support_summary,
                "unique_labels": "; ".join(label for label in unique_labels if label),
                "glm5_rag_mean": safe_float(rag_desc["mean"]),
                "glm5_rag_median": safe_float(rag_desc["median"]),
                "glm5_zero_mean": safe_float(zero_desc["mean"]),
                "glm5_zero_median": safe_float(zero_desc["median"]),
                "mean_diff_rag_minus_zero": safe_float(summary["mean_diff_rag_minus_zero"]),
                "median_diff_rag_minus_zero": safe_float(summary["median_diff_rag_minus_zero"]),
                "sd_diff": safe_float(summary["sd_diff"]),
                "rag_wins": summary["rag_wins"],
                "zero_wins": summary["zero_wins"],
                "ties": summary["ties"],
                "sign_test_p_two_sided": safe_float(summary["sign_test_p_two_sided"]),
            }
        )
    return out


def label_hierarchy(row: dict[str, Any]) -> tuple[str, str]:
    l1 = str(row.get("reference_category") or row.get("matched_category") or "UNMAPPED").strip()
    if not l1:
        l1 = "UNMAPPED"
    l2 = str(row.get("reference_sub_group") or "").strip()
    if not l2:
        l2 = l1
    return l1, l2


def summarize_label_l1_l2_diffs(label_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groupings: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in label_diff_rows:
        l1, l2 = label_hierarchy(row)
        groupings[("L1", l1, "")].append(row)
        groupings[("L2", l1, l2)].append(row)

    out: list[dict[str, Any]] = []
    for (level, l1, l2), rows in sorted(groupings.items()):
        summary = paired_summary(rows)
        rag_desc = describe([int(row["glm5_rag"]) for row in rows])
        zero_desc = describe([int(row["glm5_zero"]) for row in rows])
        unique_labels = sorted({row.get("matched_mapped_label") or row.get("web_label") or "" for row in rows})
        support_summary = reference_support_summary(rows)
        out.append(
            {
                "level": level,
                "l1_category": l1,
                "l2_sub_group": l2,
                "group_value": l1 if level == "L1" else f"{l1}|{l2}",
                "n_pairs": summary["n_pairs"],
                "n_unique_labels": len([label for label in unique_labels if label]),
                **support_summary,
                "unique_labels": "; ".join(label for label in unique_labels if label),
                "glm5_rag_mean": safe_float(rag_desc["mean"]),
                "glm5_rag_median": safe_float(rag_desc["median"]),
                "glm5_zero_mean": safe_float(zero_desc["mean"]),
                "glm5_zero_median": safe_float(zero_desc["median"]),
                "mean_diff_rag_minus_zero": safe_float(summary["mean_diff_rag_minus_zero"]),
                "median_diff_rag_minus_zero": safe_float(summary["median_diff_rag_minus_zero"]),
                "sd_diff": safe_float(summary["sd_diff"]),
                "rag_wins": summary["rag_wins"],
                "zero_wins": summary["zero_wins"],
                "ties": summary["ties"],
                "sign_test_p_two_sided": safe_float(summary["sign_test_p_two_sided"]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict[str, Any], output_dir: Path, files: list[Path]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_by_model = summarize_case_by_model(results["case_long_rows"])
    case_paired = summarize_case_diffs(results["case_diff_rows"])
    label_by_model = summarize_label_by_model(results["label_long_rows"])
    label_by_support_tier = summarize_label_by_support_tier(results["label_long_rows"])
    label_paired = summarize_label_diffs(results["label_diff_rows"])
    label_support_tier_paired = summarize_label_support_tier_diffs(results["label_diff_rows"])
    label_l1_l2_paired = summarize_label_l1_l2_diffs(results["label_diff_rows"])

    write_csv(output_dir / "per_case_decoded.csv", results["per_case_rows"])
    write_csv(output_dir / "case_level_long.csv", results["case_long_rows"])
    write_csv(output_dir / "case_level_paired_diffs.csv", results["case_diff_rows"])
    write_csv(output_dir / "case_level_summary_by_model.csv", case_by_model)
    write_csv(output_dir / "case_level_paired_summary.csv", case_paired)
    write_csv(output_dir / "label_level_long.csv", results["label_long_rows"])
    write_csv(output_dir / "label_level_paired_diffs.csv", results["label_diff_rows"])
    write_csv(output_dir / "label_level_summary_by_model.csv", label_by_model)
    write_csv(output_dir / "label_level_summary_by_support_tier.csv", label_by_support_tier)
    write_csv(output_dir / "label_level_paired_summary.csv", label_paired)
    write_csv(output_dir / "label_level_support_tier_paired_summary.csv", label_support_tier_paired)
    write_csv(output_dir / "label_level_l1_l2_paired_summary.csv", label_l1_l2_paired)
    write_csv(output_dir / "label_audit.csv", results["audit_rows"])

    audit_counts = Counter(row.get("valid", "") for row in results["audit_rows"])
    summary = {
        "source_files": [str(path) for path in files],
        "case_count": len(results["per_case_rows"]),
        "case_level_model_scores": len(results["case_long_rows"]),
        "case_level_pairs": len(results["case_diff_rows"]),
        "label_level_model_scores": len(results["label_long_rows"]),
        "label_level_pairs": len(results["label_diff_rows"]),
        "audit_valid_counts": dict(audit_counts),
        "outputs": {
            "per_case_decoded": str(output_dir / "per_case_decoded.csv"),
            "case_level_paired_summary": str(output_dir / "case_level_paired_summary.csv"),
            "label_level_paired_summary": str(output_dir / "label_level_paired_summary.csv"),
            "label_level_support_tier_paired_summary": str(
                output_dir / "label_level_support_tier_paired_summary.csv"
            ),
            "label_level_l1_l2_paired_summary": str(output_dir / "label_level_l1_l2_paired_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and summarize blinded Model A/B human scoring exports.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Blinded JSON files or a directory containing blinded_ab_scores_round2_batch_*.json.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        required=True,
        help="Private A/B mapping JSON generated with the blinded pages.",
    )
    parser.add_argument(
        "--label-map-file",
        type=Path,
        default=None,
        help="Optional web-label to canonical 9-class map CSV.",
    )
    parser.add_argument(
        "--label-count-file",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs" / "label_category_counts_9class.csv",
        help="Optional canonical label count CSV used for support-tier stratification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for decoded CSV summaries.",
    )
    parser.add_argument("--rater-id", default=None, help="Optional rater id to attach to all input files.")
    args = parser.parse_args()

    files = find_input_files(args.inputs)
    if not files:
        raise FileNotFoundError("No blinded_ab_scores_round2_batch_*.json files found.")
    results = analyze(
        files,
        mapping_file=args.mapping_file,
        label_map_file=args.label_map_file,
        label_count_file=args.label_count_file,
        rater_id=args.rater_id,
    )
    summary = write_outputs(results, args.output_dir, files)
    print(json.dumps({**summary, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
