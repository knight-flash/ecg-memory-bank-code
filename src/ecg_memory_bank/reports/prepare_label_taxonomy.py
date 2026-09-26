"""Report label-context preparation.

Purpose
-------
Create the run-local label context consumed by RAG and No-RAG prompts.

Method
------
Compute label frequencies, merge reference classifications and overrides, and
save candidate labels grouped by category.

Inputs
------
External corrected MIMIC metadata, classification spreadsheet, and optional
count CSV.

Outputs
-------
External label-context JSON/CSV and mapping audit in the selected run
directory.

Paper parameters
----------------
Uses the shared nine-category ontology; the benchmark candidate list is fixed
for paired generation modes.

Shapes and meaning
------------------
Candidate order defines the C score columns and diagnostic slots; counts are
descriptive memory support, not model probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from ecg_memory_bank.reports.config import default_report_config
from ecg_memory_bank.reports.label_taxonomy import (
    apply_label_mapping,
    build_candidate_labels_by_category,
    build_label_to_category_map,
    build_vocab_counts_from_metadata,
    load_label_counts,
    load_label_overrides,
    load_reference_classification,
    write_label_context,
)


def parse_args() -> argparse.Namespace:
    config = default_report_config()
    parser = argparse.ArgumentParser(description="Prepare label vocabulary audit and classification mapping for ecg_memory_bank.reports.")
    parser.add_argument("--run-name", type=str, default=config.grounding.run_name)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--metadata-csv", type=str, default=config.dataset.corrected_label_metadata_path)
    parser.add_argument("--label-column", type=str, default=config.dataset.label_column)
    parser.add_argument("--label-counts-csv", type=str, default=str(config.labels.label_counts_csv))
    parser.add_argument("--classification-xlsx", type=str, default=str(config.labels.classification_xlsx))
    parser.add_argument("--overrides-csv", type=str, default=str(config.labels.overrides_csv))
    return parser.parse_args()


def _resolve_run_dir(config, run_name: str, run_dir_arg: str) -> Path:
    return Path(run_dir_arg) if run_dir_arg else (config.output.runs_root / run_name)


def main() -> None:
    args = parse_args()
    config = default_report_config()
    run_dir = _resolve_run_dir(config, args.run_name, args.run_dir)
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    metadata_vocab_df = build_vocab_counts_from_metadata(args.metadata_csv, label_column=args.label_column)
    try:
        count_audit_df = load_label_counts(args.label_counts_csv)
    except Exception:
        count_audit_df = metadata_vocab_df.copy()
    reference_df = load_reference_classification(args.classification_xlsx)
    overrides_df = load_label_overrides(args.overrides_csv)

    mapped_df, unmatched_df = apply_label_mapping(
        metadata_vocab_df,
        reference_df,
        overrides_df,
        stop_labels=config.labels.stop_labels,
    )

    metadata_vocab_df.to_csv(labels_dir / "label_vocab_counts.csv", index=False)
    count_audit_df.to_csv(labels_dir / "label_count_audit_reference.csv", index=False)
    mapped_df.to_csv(labels_dir / "label_classification_mapped.csv", index=False)
    unmatched_df.to_csv(labels_dir / "label_classification_unmatched.csv", index=False)

    candidate_labels_by_category = build_candidate_labels_by_category(mapped_df, categories=config.labels.categories)
    label_to_category = build_label_to_category_map(mapped_df)
    candidate_label_set = sorted(mapped_df["mapped_label"].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()) if not mapped_df.empty else []
    label_context = {
        "metadata_csv": args.metadata_csv,
        "label_column": args.label_column,
        "categories": list(config.labels.categories),
        "candidate_labels_by_category": candidate_labels_by_category,
        "candidate_label_set": candidate_label_set,
        "label_to_category": label_to_category,
        "mapped_count": len(mapped_df),
        "unmatched_count": len(unmatched_df),
    }
    write_label_context(labels_dir / "label_category_context.json", label_context)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "mapped_count": len(mapped_df),
                "unmatched_count": len(unmatched_df),
                "category_counts": {key: len(value) for key, value in candidate_labels_by_category.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
