#!/usr/bin/env python3
"""Export Q_aff masks for the existing HEEDB case-withdrawal scenarios.

The original withdrawal outputs retain post-withdrawal voting scores but not
the baseline top-K member identities.  This script performs one exact baseline
retrieval pass and records, for each scenario, which fixed holdout queries had
at least one subsequently withdrawn case in their pre-withdrawal top-K.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

import case_withdrawal as withdrawal  # noqa: E402


SERVER_RETRIEVAL_ROOT = Path("__ECG_DATA_ROOT_REQUIRED__/Finaltabel/Retrieval")
DEFAULT_OUTPUT_ROOT = SERVER_RETRIEVAL_ROOT / "outputs"
DEFAULT_FAMILY = withdrawal.FAMILY_NAME
DEFAULT_BASELINE = "full_gallery_2022_holdout_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export pre-withdrawal affected-query (Q_aff) masks for existing "
            "HEEDB case-withdrawal results. This runs one exact baseline retrieval, "
            "but does not encode or train a model."
        )
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--baseline-scenario", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Withdrawal scenario to export. Repeatable; defaults to all existing non-baseline scenarios.",
    )
    parser.add_argument("--membership-dir", default="", help="Defaults to <family>/audit_membership.")
    parser.add_argument("--finalgallery-root", default=str(withdrawal.DEFAULT_FINALGALLERY_ROOT))
    parser.add_argument("--train-meta-path", default="")
    parser.add_argument("--train-ecg-path", default="")
    parser.add_argument("--val-meta-path", default="")
    parser.add_argument("--val-ecg-path", default="")
    parser.add_argument("--test-meta-path", default="")
    parser.add_argument("--test-ecg-path", default="")
    parser.add_argument("--train-enrich-meta-path", default=str(withdrawal.DEFAULT_ENRICH_ROOT / "train_processed_update.csv"))
    parser.add_argument("--val-enrich-meta-path", default=str(withdrawal.DEFAULT_ENRICH_ROOT / "val_processed_update.csv"))
    parser.add_argument("--test-enrich-meta-path", default="")
    parser.add_argument("--record-id-col", default="")
    parser.add_argument("--patient-id-col", default="")
    parser.add_argument("--query-years", nargs="+", type=int, default=None)
    parser.add_argument(
        "--include-val-query-years-in-gallery",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--retrieval-mode", choices=("gpu-shard", "cpu", "auto"), default=None)
    parser.add_argument("--gallery-shard-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate saved scenario metadata and print the exact retrieval plan without loading embeddings.",
    )
    parser.add_argument(
        "--skip-query-id-check",
        action="store_true",
        help="Skip verification that reconstructed query IDs match the original baseline per-query artifact.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scenario_names(family_root: Path, baseline_scenario: str, requested: list[str]) -> list[str]:
    if requested:
        return requested
    names = []
    for summary_path in sorted(family_root.glob("*/*_summary.json")):
        if summary_path.name != f"{summary_path.parent.name}_summary.json":
            continue
        payload = load_json(summary_path)
        if payload.get("scenario") == baseline_scenario:
            continue
        if payload.get("withdrawal_type") in {"protocol_historical", "patient_mass"}:
            names.append(str(payload["scenario"]))
    if not names:
        raise FileNotFoundError(f"No supported withdrawal scenarios found under {family_root}")
    return names


def required_setting(
    explicit: Any,
    manifest_value: Any,
    label: str,
) -> Any:
    if explicit is None:
        return manifest_value
    if explicit != manifest_value:
        raise ValueError(
            f"{label}={explicit!r} does not match the original withdrawal manifest value {manifest_value!r}."
        )
    return explicit


def build_runtime_args(args: argparse.Namespace, split_manifest: dict[str, Any]) -> argparse.Namespace:
    params = split_manifest["parameters"]
    gallery_policy = split_manifest["gallery_policy"]
    query_years = required_setting(args.query_years, split_manifest["query_years"], "query_years")
    include_query_years = required_setting(
        args.include_val_query_years_in_gallery,
        gallery_policy["include_val_query_years_in_gallery"],
        "include_val_query_years_in_gallery",
    )
    seed = required_setting(args.seed, params["seed"], "seed")
    top_k = required_setting(args.top_k, params["top_k"], "top_k")
    batch_size = required_setting(args.batch_size, params["batch_size"], "batch_size")
    retrieval_mode = required_setting(args.retrieval_mode, params["retrieval_mode"], "retrieval_mode")
    shard_size = required_setting(args.gallery_shard_size, params["gallery_shard_size"], "gallery_shard_size")
    return argparse.Namespace(
        finalgallery_root=args.finalgallery_root,
        train_meta_path=args.train_meta_path,
        train_ecg_path=args.train_ecg_path,
        val_meta_path=args.val_meta_path,
        val_ecg_path=args.val_ecg_path,
        test_meta_path=args.test_meta_path,
        test_ecg_path=args.test_ecg_path,
        train_enrich_meta_path=args.train_enrich_meta_path,
        val_enrich_meta_path=args.val_enrich_meta_path,
        test_enrich_meta_path=args.test_enrich_meta_path,
        record_id_col=args.record_id_col,
        patient_id_col=args.patient_id_col,
        query_years=[int(value) for value in query_years],
        include_val_query_years_in_gallery=bool(include_query_years),
        seed=int(seed),
        top_k=int(top_k),
        batch_size=int(batch_size),
        retrieval_mode=str(retrieval_mode),
        gallery_shard_size=int(shard_size),
    )


def build_baseline_data(
    runtime_args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    sources = {
        "train": withdrawal.load_prepared_source(runtime_args, "train", require_patient=True),
        "val": withdrawal.load_prepared_source(runtime_args, "val", require_patient=True),
        "test": withdrawal.load_prepared_source(runtime_args, "test", require_patient=False),
    }
    query_years = set(runtime_args.query_years)
    val_query_mask = sources["val"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    test_query_mask = sources["test"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    query_df, query_vecs = withdrawal.concat_source_rows(
        [(sources["val"], val_query_mask), (sources["test"], test_query_mask)]
    )

    train_gallery_mask = np.ones(len(sources["train"]["df"]), dtype=bool)
    val_gallery_mask = np.ones(len(sources["val"]["df"]), dtype=bool)
    if not runtime_args.include_val_query_years_in_gallery:
        val_gallery_mask &= ~sources["val"]["df"]["_parsed_year"].isin(query_years).to_numpy(dtype=bool)
    gallery_df, gallery_vecs = withdrawal.concat_source_rows(
        [(sources["train"], train_gallery_mask), (sources["val"], val_gallery_mask)]
    )
    for source in sources.values():
        source["vecs"] = np.empty((0, 0), dtype=np.float32)
    gc.collect()
    return gallery_df, gallery_vecs, query_df, query_vecs


def deletion_mask_for_scenario(
    *,
    scenario_name: str,
    summary: dict[str, Any],
    family_manifest: dict[str, Any],
    gallery_df: pd.DataFrame,
) -> np.ndarray:
    withdrawal_type = str(summary.get("withdrawal_type", ""))
    scenario_manifest = summary.get("withdrawal_manifest", {})
    if withdrawal_type == "protocol_historical":
        threshold = scenario_manifest.get("year_threshold")
        if threshold is None:
            raise KeyError(f"{scenario_name} is missing withdrawal_manifest.year_threshold")
        return gallery_df["_parsed_year"].to_numpy(dtype=np.int32) <= int(threshold)

    if withdrawal_type == "patient_mass":
        fraction = scenario_manifest.get("requested_patient_fraction")
        if fraction is None:
            raise KeyError(f"{scenario_name} is missing withdrawal_manifest.requested_patient_fraction")
        patient_sets = withdrawal.build_patient_withdrawal_sets(
            gallery_df,
            fractions=[float(value) for value in family_manifest["parameters"]["patient_withdrawal_fractions"]],
            seed=int(family_manifest["parameters"]["seed"]),
            explicit_count=int(family_manifest["parameters"].get("patient_withdrawal_count", 0)),
        )
        matched = [patients for value, patients in patient_sets if np.isclose(float(value), float(fraction))]
        if len(matched) != 1:
            raise ValueError(f"Unable to reconstruct patient selection for {scenario_name} at fraction {fraction}.")
        return gallery_df["_patient_id"].astype(str).isin(matched[0]).to_numpy(dtype=bool)

    raise ValueError(
        f"Unsupported withdrawal_type={withdrawal_type!r} for {scenario_name}. "
        "This exporter currently supports the historical and patient-cohort scenarios only."
    )


def expected_query_count(family_root: Path, baseline_scenario: str) -> int:
    path = family_root / baseline_scenario / "ci_artifacts" / f"{baseline_scenario}_l3_auc_inputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline L3 AUC artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return int(payload["y_true"].shape[0])


def verify_query_alignment(
    family_root: Path,
    baseline_scenario: str,
    query_df: pd.DataFrame,
) -> None:
    per_query_path = family_root / baseline_scenario / f"{baseline_scenario}_per_query.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"Missing baseline per-query artifact required for query-ID validation: {per_query_path}")
    if "_record_id" not in query_df.columns:
        raise KeyError("Reconstructed query dataframe is missing _record_id.")
    original_ids = pd.read_csv(per_query_path, usecols=["_record_id"])["_record_id"].astype(str).to_numpy()
    reconstructed_ids = query_df["_record_id"].astype(str).to_numpy()
    if original_ids.shape != reconstructed_ids.shape or not np.array_equal(original_ids, reconstructed_ids):
        raise ValueError(
            "Reconstructed query order does not match the original baseline per-query artifact. "
            "Do not create Q_aff masks against misaligned voting-score arrays."
        )
    print(f"[Verified] Query identity and order match {per_query_path}", flush=True)


def main() -> None:
    args = parse_args()
    family_root = Path(args.output_dir) / args.family
    family_manifest = load_json(family_root / "case_withdrawal_split_manifest.json")
    runtime_args = build_runtime_args(args, family_manifest)
    names = scenario_names(family_root, args.baseline_scenario, args.scenario)
    expected_queries = expected_query_count(family_root, args.baseline_scenario)
    membership_dir = Path(args.membership_dir) if args.membership_dir else family_root / "audit_membership"
    membership_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[Dry run] The following scenarios will receive Q_aff masks:")
        for scenario_name in names:
            summary_path = family_root / scenario_name / f"{scenario_name}_summary.json"
            summary = load_json(summary_path)
            print(
                f"  - {scenario_name}: {summary['withdrawal_type']}, "
                f"deleted_records={summary['num_deleted_gallery_records']}",
                flush=True,
            )
        print(
            "[Dry run] One exact baseline top-K retrieval will be run after removing --dry-run; "
            "no encoder training or model inference is required.",
            flush=True,
        )
        return

    gallery_df, gallery_vecs, query_df, query_vecs = build_baseline_data(runtime_args)
    if len(query_vecs) != expected_queries:
        raise ValueError(
            f"Reconstructed query count {len(query_vecs)} does not match saved baseline artifact {expected_queries}."
        )
    if not args.skip_query_id_check:
        verify_query_alignment(family_root, args.baseline_scenario, query_df)
    print(
        f"[Retrieval] exact baseline membership: gallery={len(gallery_df)}, queries={len(query_vecs)}, "
        f"top_k={runtime_args.top_k}",
        flush=True,
    )
    retrieved_indices = withdrawal.run_retrieval(
        gallery_vecs=gallery_vecs,
        query_vecs=query_vecs,
        top_k=runtime_args.top_k,
        batch_size=runtime_args.batch_size,
        scenario_name="baseline_membership_export",
        retrieval_mode=runtime_args.retrieval_mode,
        gallery_shard_size=runtime_args.gallery_shard_size,
    )

    exported = []
    for scenario_name in names:
        summary_path = family_root / scenario_name / f"{scenario_name}_summary.json"
        summary = load_json(summary_path)
        deleted_mask = deletion_mask_for_scenario(
            scenario_name=scenario_name,
            summary=summary,
            family_manifest=family_manifest,
            gallery_df=gallery_df,
        )
        expected_deleted = int(summary["num_deleted_gallery_records"])
        if int(deleted_mask.sum()) != expected_deleted:
            raise ValueError(
                f"{scenario_name}: reconstructed deleted rows={int(deleted_mask.sum())}, "
                f"but summary records {expected_deleted}."
            )
        deleted_slots = deleted_mask[retrieved_indices].sum(axis=1).astype(np.uint8)
        q_aff = (deleted_slots > 0).astype(np.uint8)
        output_path = membership_dir / f"{scenario_name}_baseline_membership.npz"
        np.savez_compressed(
            output_path,
            q_aff=q_aff,
            deleted_slot_count=deleted_slots,
            top_k=np.asarray([retrieved_indices.shape[1]], dtype=np.int16),
        )
        row = {
            "scenario": scenario_name,
            "withdrawal_type": summary["withdrawal_type"],
            "num_queries": int(len(q_aff)),
            "num_deleted_gallery_records": expected_deleted,
            "q_aff_count": int(q_aff.sum()),
            "q_aff_rate": float(q_aff.mean()),
            "mean_deleted_slots_in_baseline_topk": float(deleted_slots.mean()),
            "max_deleted_slots_in_baseline_topk": int(deleted_slots.max()),
            "membership_npz": str(output_path),
        }
        exported.append(row)
        print(
            f"[Saved] {scenario_name}: Q_aff={row['q_aff_count']}/{row['num_queries']} "
            f"({row['q_aff_rate']:.6f}) -> {output_path}",
            flush=True,
        )

    manifest = {
        "family": args.family,
        "baseline_scenario": args.baseline_scenario,
        "membership_definition": "Q_aff is the set of queries with at least one withdrawn gallery record in baseline exact top-K.",
        "top_k": int(retrieved_indices.shape[1]),
        "gallery_size": int(len(gallery_df)),
        "query_count": int(len(query_vecs)),
        "scenarios": exported,
    }
    manifest_path = membership_dir / "baseline_membership_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] Membership manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
