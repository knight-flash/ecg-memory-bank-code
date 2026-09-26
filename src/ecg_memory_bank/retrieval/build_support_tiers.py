"""Recompute HEEDB L3 support quartiles from the external full-memory metadata."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from ecg_memory_bank.protocol import external_output_dir
from ecg_memory_bank.retrieval.label_mapping_utils import load_label_mapping, parse_label_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--label-col", default="labels_correct")
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_output_dir(args.output_csv.parent)
    mapping = load_label_mapping(args.mapping_csv)
    lookup = mapping["exact_lookup"]
    labels = {entry["mapped_label"] for entry in lookup.values() if entry["mapped_label"]}
    if len(labels) != 120:
        raise ValueError(f"Expected 120 mapped L3 labels, found {len(labels)}")
    counts: Counter[str] = Counter({label: 0 for label in labels})
    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if args.label_col not in (reader.fieldnames or []):
            raise KeyError(args.label_col)
        for row in reader:
            mapped = {lookup[label]["mapped_label"] for label in parse_label_field(row[args.label_col]) if label in lookup}
            counts.update(mapped)
    ordered = sorted(labels, key=lambda label: (counts[label], label))
    tiers = ("Tail", "Low-mid", "Mid", "Head")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mapped_label", "count", "support_tier"])
        writer.writerows((label, counts[label], tiers[index // 30]) for index, label in enumerate(ordered))


if __name__ == "__main__":
    main()
