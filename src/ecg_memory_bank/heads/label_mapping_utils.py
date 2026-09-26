#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


CANONICAL_9_CATEGORIES = (
    "Axis",
    "Conduction",
    "General report",
    "Hypertrophy",
    "Ischemia events",
    "Others",
    "Pacemaker",
    "Rhythm",
    "ST-T changes",
)

LABEL_PARSE_SEPARATORS = ("|", ";", "\n")


def normalize_label_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("\r", " ").replace("\n", " ")
    text = text.replace("*", " ").replace('"', " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,")


def load_label_mapping(mapping_csv_path: str | Path) -> dict[str, Any]:
    mapping_path = Path(mapping_csv_path)
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_cols = {"category", "sub_group", "raw_label", "mapped_label"}
        missing_cols = required_cols.difference(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing required columns in mapping CSV: {sorted(missing_cols)}")

        exact_lookup: dict[str, dict[str, str]] = {}
        hierarchy_lookup: dict[str, tuple[str, str]] = {}
        categories: set[str] = set()

        for row in reader:
            category = str(row["category"]).strip()
            sub_group = str(row["sub_group"]).strip()
            raw_label = normalize_label_text(row["raw_label"])
            mapped_label = normalize_label_text(row["mapped_label"])
            normalized_category = normalize_label_text(category)
            normalized_sub_group = normalize_label_text(sub_group) or normalized_category

            if category:
                categories.add(category)

            entry = {
                "category": category,
                "sub_group": sub_group,
                "mapped_label": mapped_label,
            }

            if raw_label:
                exact_lookup.setdefault(raw_label, entry)
                hierarchy_lookup.setdefault(raw_label, (normalized_category, normalized_sub_group))
            if mapped_label:
                exact_lookup.setdefault(mapped_label, entry)
                hierarchy_lookup.setdefault(mapped_label, (normalized_category, normalized_sub_group))

    return {
        "mapping_path": str(mapping_path),
        "exact_lookup": exact_lookup,
        "hierarchy_lookup": hierarchy_lookup,
        "categories": tuple(sorted(categories)),
    }


def load_label_hierarchy(mapping_csv_path: str | Path) -> dict[str, tuple[str, str]]:
    return load_label_mapping(mapping_csv_path)["hierarchy_lookup"]


def _strip_known_prefixes(label: str) -> list[str]:
    prefixes = (
        "summary:",
        "probable ",
        "possible ",
        "consider ",
        "old ",
        "- ",
    )
    variants = [label]
    changed = True
    while changed:
        changed = False
        next_variants = list(variants)
        for variant in variants:
            for prefix in prefixes:
                if variant.startswith(prefix):
                    stripped = normalize_label_text(variant[len(prefix) :])
                    if stripped and stripped not in next_variants:
                        next_variants.append(stripped)
                        changed = True
        variants = next_variants
    return variants


def split_composite_mimic_label(label: str) -> list[str]:
    normalized = normalize_label_text(label)
    if not normalized:
        return []

    parts = [normalize_label_text(part) for part in re.split(r"\s+with\s+", normalized) if normalize_label_text(part)]
    if len(parts) <= 1:
        return [normalized]
    return parts


def parse_label_field(raw_value: Any) -> list[str]:
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() in {"nan", "none"}:
        return []

    candidates: list[str] | None = None
    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            candidates = [str(item) for item in parsed]

    if candidates is None:
        for separator in LABEL_PARSE_SEPARATORS:
            if separator in raw_text:
                candidates = raw_text.split(separator)
                break

    if candidates is None:
        candidates = [raw_text]

    expanded: list[str] = []
    seen = set()
    for candidate in candidates:
        for split_label in split_composite_mimic_label(candidate):
            normalized = normalize_label_text(split_label)
            if not normalized or normalized in seen:
                continue
            expanded.append(normalized)
            seen.add(normalized)
    return expanded


def _add_candidate(candidates: list[str], seen: set[str], value: str) -> None:
    normalized = normalize_label_text(value)
    if normalized and normalized not in seen:
        candidates.append(normalized)
        seen.add(normalized)


def generate_mimic_alias_candidates(label: str) -> list[str]:
    normalized = normalize_label_text(label)
    candidates: list[str] = []
    seen: set[str] = set()
    _add_candidate(candidates, seen, normalized)

    for variant in _strip_known_prefixes(normalized):
        _add_candidate(candidates, seen, variant)

    trimmed = normalize_label_text(re.sub(r"\s*-\s*age undetermined$", "", normalized))
    if trimmed != normalized:
        _add_candidate(candidates, seen, trimmed)
        for variant in _strip_known_prefixes(trimmed):
            _add_candidate(candidates, seen, variant)

    hyphen_normalized = normalized
    replacements = {
        "a-v": "av",
        "pvc(s)": "pvcs",
        "pac(s)": "pacs",
        "st elev": "st elevation",
    }
    for source_text, target_text in replacements.items():
        hyphen_normalized = hyphen_normalized.replace(source_text, target_text)
    _add_candidate(candidates, seen, hyphen_normalized)

    direct_aliases = {
        "leftward axis": "left axis deviation",
        "marked left axis deviation": "left axis deviation",
        "abnormal left axis deviation": "left axis deviation",
        "marked right axis deviation": "right axis deviation",
        "abnormal right axis deviation": "right axis deviation",
        "iv conduction defect": "nonspecific intraventricular conduction delay",
        "intraventricular conduction defect": "nonspecific intraventricular conduction delay",
        "extensive intraventricular conduction defect": "nonspecific intraventricular conduction delay",
        "borderline intraventricular conduction delay": "nonspecific intraventricular conduction delay",
        "intermittent conduction defect": "nonspecific intraventricular conduction delay",
        "incomplete rbbb": "incomplete right bundle branch block",
        "incomplete lbbb": "incomplete left bundle branch block",
        "rbbb with left anterior fascicular block": "bifascicular block",
        "rbbb and left anterior fascicular block": "bifascicular block",
        "rbbb and left posterior fascicular block": "bifascicular block",
        "ventricular pacing": "ventricular-paced rhythm",
        "regular ventricular pacing": "ventricular-paced rhythm",
        "demand ventricular pacing": "ventricular-paced rhythm",
        "demand pacing": "ventricular-paced rhythm",
        "electronic ventricular pacemaker": "ventricular-paced rhythm",
        "electronic demand pacing": "ventricular-paced rhythm",
        "pacemaker rhythm - no further analysis": "ventricular-paced rhythm",
        "pacemaker rhythm - no further analysis*": "ventricular-paced rhythm",
        "atrial pacing": "atrial-paced rhythm",
        "electronic atrial pacemaker": "atrial-paced rhythm",
        "a-v sequential and ventricular pacing": "av dual-paced rhythm",
        "demand a-v sequential and ventricular pacing": "av dual-paced rhythm",
        "av sequential or dual chamber electronic pacemaker": "av dual-paced rhythm",
        "low qrs voltages in precordial leads": "low voltage qrs",
        "low qrs voltages in limb leads": "low voltage qrs",
        "low qrs voltages in standard limb leads": "low voltage qrs",
        "generalized low qrs voltages": "low voltage qrs",
        "low voltage": "low voltage qrs",
        "prolonged qt interval": "prolonged qt",
        "rapid ventricular response": "with rapid ventricular response",
        "uncontrolled ventricular response": "with rapid ventricular response",
        "slow ventricular response": "with slow ventricular response",
        "variable av block": "with variable av block",
        "1st degree av block": "with 1st degree av block",
        "1st degree a-v block": "with 1st degree av block",
        "borderline 1st degree a-v block": "with 1st degree av block",
        "prolonged pr interval": "with 1st degree av block",
        "short pr interval": "with short pr",
        "sinus arrhythmia": "with sinus arrhythmia",
        "marked sinus arrhythmia": "with marked sinus arrhythmia",
        "left atrial abnormality": "left atrial enlargement",
        "possible left atrial abnormality": "left atrial enlargement",
        "probable left atrial enlargement": "left atrial enlargement",
        "probable left atrial abnormality": "left atrial enlargement",
        "consider left atrial abnormality": "left atrial enlargement",
        "possible right atrial abnormality": "right atrial enlargement",
        "ventricular premature complex": "premature ventricular complexes",
        "premature ventricular contractions": "premature ventricular complexes",
        "premature ventricular contraction": "premature ventricular complexes",
        "multifocal premature ventricular contractions": "premature ventricular complexes",
        "frequent multifocal premature ventricular contractions": "premature ventricular complexes",
        "multifocal interpolated premature ventricular contractions": "premature ventricular complexes",
        "premature atrial contractions": "premature atrial complexes",
        "aberrantly conducted supraventricular complexes": "aberrant conduction",
        "aberrant ventricular conduction": "aberrant conduction",
        "aberrant conduction of sv complex(es)": "aberrant conduction",
        "aberrant complex": "aberrant conduction",
        "unclassified aberrant complexes": "aberrant conduction",
        "possible left ventricular hypertrophy": "left ventricular hypertrophy",
        "lvh with secondary repolarization abnormality": "left ventricular hypertrophy",
        "consider acute infarct": "acute mi / stemi",
        "acute st elevation mi": "acute mi / stemi",
        "consider acute st elevation mi": "acute mi / stemi",
    }
    if normalized in direct_aliases:
        _add_candidate(candidates, seen, direct_aliases[normalized])

    if "pvc" in normalized and "aberrant ventricular conduction" in normalized:
        _add_candidate(candidates, seen, "with premature ventricular or aberrantly conducted complexes")

    if normalized.startswith("multifocal pvc") or normalized.startswith("frequent multifocal pvc"):
        _add_candidate(candidates, seen, "premature ventricular complexes")

    if "left ventricular hypertrophy" in normalized:
        _add_candidate(candidates, seen, "left ventricular hypertrophy")

    if "right ventricular hypertrophy" in normalized:
        _add_candidate(candidates, seen, "right ventricular hypertrophy")

    infarct_locations = (
        "inferior-posterior",
        "anteroseptal",
        "anterolateral",
        "inferior",
        "anterior",
        "septal",
        "lateral",
        "posterior",
    )
    for location in infarct_locations:
        infarct_label = f"{location} infarct"
        if infarct_label in normalized:
            _add_candidate(candidates, seen, infarct_label)

    if "acute mi" in normalized or "stemi" in normalized:
        _add_candidate(candidates, seen, "acute mi / stemi")

    if "injury pattern" in normalized:
        for location in ("inferolateral", "anterolateral", "inferior", "anterior", "lateral"):
            candidate = f"{location} injury pattern"
            if location in normalized:
                _add_candidate(candidates, seen, candidate)

    if "st elevation" in normalized and "acute infarct" in normalized:
        for location in ("inferolateral", "anterolateral", "inferior", "anterior", "lateral"):
            if location in normalized:
                _add_candidate(candidates, seen, f"st elevation, consider {location} injury or acute infarct")
        if "acute infarct" in normalized:
            _add_candidate(candidates, seen, "acute mi / stemi")

    if "st-t" in normalized or "t wave" in normalized or "st changes" in normalized or "st junctional depression" in normalized:
        if "ischemia" in normalized:
            location_candidates = (
                ("inferolateral", "t wave abnormality, consider inferolateral ischemia"),
                ("anterolateral", "t wave abnormality, consider anterolateral ischemia"),
                ("inferior", "t wave abnormality, consider inferior ischemia"),
                ("anterior", "t wave abnormality, consider anterior ischemia"),
                ("lateral", "t wave abnormality, consider lateral ischemia"),
            )
            for keyword, mapped in location_candidates:
                if keyword in normalized:
                    _add_candidate(candidates, seen, mapped)
                    break
            else:
                _add_candidate(candidates, seen, "nonspecific st and t wave abnormality")
        elif "repolarization abnormality" in normalized:
            _add_candidate(candidates, seen, "with repolarization abnormality")
        elif "early repol" in normalized:
            _add_candidate(candidates, seen, "early repolarization")
        elif "t wave" in normalized and "nonspecific" in normalized:
            _add_candidate(candidates, seen, "nonspecific t wave abnormality")
        elif ("st changes" in normalized or "st junctional depression" in normalized or "st depression" in normalized) and "nonspecific" in normalized:
            _add_candidate(candidates, seen, "nonspecific st abnormality")
        else:
            _add_candidate(candidates, seen, "nonspecific st and t wave abnormality")

    if not normalized.startswith("with "):
        for prefixed_candidate in list(candidates):
            if prefixed_candidate in {
                "rapid ventricular response",
                "slow ventricular response",
                "variable av block",
                "1st degree av block",
                "short pr",
                "sinus arrhythmia",
                "marked sinus arrhythmia",
            }:
                _add_candidate(candidates, seen, f"with {prefixed_candidate}")

    return candidates


def map_mimic_label(label: str, label_mapping: dict[str, Any]) -> dict[str, str] | None:
    exact_lookup = label_mapping["exact_lookup"]
    normalized = normalize_label_text(label)
    for index, candidate in enumerate(generate_mimic_alias_candidates(normalized)):
        if candidate in exact_lookup:
            entry = exact_lookup[candidate]
            source = "reference_exact" if index == 0 and candidate == normalized else "normalized_rule"
            return {
                "category": entry["category"],
                "sub_group": entry["sub_group"],
                "mapped_label": entry["mapped_label"],
                "mapping_source": source,
            }
    return None


def dumps_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)

