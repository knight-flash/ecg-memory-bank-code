#!/usr/bin/env python3
"""Withdrawal influence metrics.

Purpose
-------
Quantify where removed support cases altered retrieval-voting outputs and
whether removed cases persisted.

Method
------
Compare pre/post score matrices and baseline top-K membership, then summarize
affected queries, score changes, Jaccard changes, and collateral effects.

Inputs
------
External withdrawal scenario artifacts, membership arrays, label-score
matrices, and optional collateral specification.

Outputs
-------
External influence summaries with optional 95% bootstrap intervals.

Paper parameters
----------------
Bootstrap defaults to 1000 resamples with seed=42; labels and queries remain
fixed across compared states.

Shapes and meaning
------------------
Score matrices are [Q,C] and membership is query-level; a changed label set
means at least one score crossed the positive-vote threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SERVER_RETRIEVAL_ROOT = Path("__ECG_DATA_ROOT_REQUIRED__/Finaltabel/Retrieval")
DEFAULT_OUTPUT_ROOT = SERVER_RETRIEVAL_ROOT / "outputs"
DEFAULT_FAMILY = "forgetting_case_withdrawal"
DEFAULT_BASELINE = "full_gallery_2022_holdout_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process case-withdrawal outputs into influence leakage and collateral-ratio metrics."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--family", default=DEFAULT_FAMILY)
    parser.add_argument("--baseline-scenario", default=DEFAULT_BASELINE)
    parser.add_argument("--membership-dir", default="", help="Defaults to <family>/audit_membership.")
    parser.add_argument("--audit-output-dir", default="", help="Defaults to <family>/audit_metrics.")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario to audit; repeatable.")
    parser.add_argument(
        "--collateral-spec",
        default="",
        help=(
            "JSON file that predeclares related and unrelated L3 labels per label-targeted withdrawal scenario. "
            "Without it, collateral ratios are intentionally marked not estimable."
        ),
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=1000, help="0 disables confidence intervals.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ci_mean(values: np.ndarray, bootstrap: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if bootstrap <= 0 or values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(bootstrap), dtype=np.float64)
    for index in range(int(bootstrap)):
        draws[index] = values[rng.integers(0, values.size, size=values.size)].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def ci_ratio(
    related_losses: np.ndarray,
    unrelated_losses: np.ndarray,
    bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    related_losses = np.asarray(related_losses, dtype=np.float64)
    unrelated_losses = np.asarray(unrelated_losses, dtype=np.float64)
    if bootstrap <= 0 or related_losses.size < 2 or unrelated_losses.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(bootstrap)):
        numerator = related_losses[rng.integers(0, related_losses.size, size=related_losses.size)].mean()
        denominator = unrelated_losses[rng.integers(0, unrelated_losses.size, size=unrelated_losses.size)].mean()
        if denominator > 0:
            draws.append(numerator / denominator)
    if not draws:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def read_l3_scores(family_root: Path, scenario: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = family_root / scenario / "ci_artifacts" / f"{scenario}_l3_auc_inputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing L3 voting-score artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return (
            payload["y_true"].astype(np.uint8, copy=False),
            payload["y_score"].astype(np.float32, copy=False),
            [str(item) for item in payload["class_names"].tolist()],
        )


def scenario_names(membership_dir: Path, requested: list[str]) -> list[str]:
    if requested:
        return requested
    manifest = load_json(membership_dir / "baseline_membership_manifest.json")
    return [str(item["scenario"]) for item in manifest["scenarios"]]


def load_membership(membership_dir: Path, scenario: str, expected_queries: int) -> np.ndarray:
    path = membership_dir / f"{scenario}_baseline_membership.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline membership export: {path}")
    with np.load(path, allow_pickle=False) as payload:
        q_aff = payload["q_aff"].astype(bool, copy=False)
    if q_aff.shape != (expected_queries,):
        raise ValueError(f"{scenario}: Q_aff shape {q_aff.shape} does not match query count {expected_queries}.")
    return q_aff


def influence_row(
    *,
    scenario: str,
    pre_scores: np.ndarray,
    post_scores: np.ndarray,
    q_aff: np.ndarray,
    threshold: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if not q_aff.any():
        return {
            "scenario": scenario,
            "q_aff_count": 0,
            "q_aff_rate": 0.0,
            "mean_label_set_jaccard_change": float("nan"),
            "mean_label_score_absolute_change": float("nan"),
            "changed_label_set_query_rate": float("nan"),
            "jaccard_change_ci95_lower": float("nan"),
            "jaccard_change_ci95_upper": float("nan"),
            "score_change_ci95_lower": float("nan"),
            "score_change_ci95_upper": float("nan"),
        }

    pre_active = pre_scores[q_aff] > threshold
    post_active = post_scores[q_aff] > threshold
    intersection = np.logical_and(pre_active, post_active).sum(axis=1)
    union = np.logical_or(pre_active, post_active).sum(axis=1)
    jaccard_change = np.divide(
        union - intersection,
        union,
        out=np.zeros(union.shape, dtype=np.float64),
        where=union > 0,
    )
    score_change = np.abs(pre_scores[q_aff] - post_scores[q_aff]).mean(axis=1, dtype=np.float64)
    j_low, j_high = ci_mean(jaccard_change, bootstrap, seed)
    s_low, s_high = ci_mean(score_change, bootstrap, seed + 1)
    return {
        "scenario": scenario,
        "q_aff_count": int(q_aff.sum()),
        "q_aff_rate": float(q_aff.mean()),
        "mean_label_set_jaccard_change": float(jaccard_change.mean()),
        "mean_label_score_absolute_change": float(score_change.mean()),
        "changed_label_set_query_rate": float((jaccard_change > 0).mean()),
        "jaccard_change_ci95_lower": j_low,
        "jaccard_change_ci95_upper": j_high,
        "score_change_ci95_lower": s_low,
        "score_change_ci95_upper": s_high,
    }


def load_collateral_spec(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    payload = load_json(Path(path_text))
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, dict):
        raise ValueError("collateral spec must contain a JSON object named 'scenarios'.")
    return scenarios


def collateral_row(
    *,
    family_root: Path,
    baseline_scenario: str,
    scenario: str,
    spec: dict[str, Any] | None,
    bootstrap: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    common = {"scenario": scenario}
    if spec is None:
        return common | {
            "status": "not_estimable_no_predeclared_label_groups",
            "collateral_ratio": float("nan"),
        }
    related_labels = [str(item) for item in spec.get("related_labels", [])]
    unrelated_labels = [str(item) for item in spec.get("unrelated_labels", [])]
    if not related_labels or not unrelated_labels:
        raise ValueError(f"{scenario}: collateral spec requires non-empty related_labels and unrelated_labels.")
    overlap = sorted(set(related_labels).intersection(unrelated_labels))
    if overlap:
        raise ValueError(f"{scenario}: labels cannot be both related and unrelated: {overlap}")

    baseline = pd.read_csv(
        family_root / baseline_scenario / f"per_class_auc_{baseline_scenario}_L3.csv"
    )[["Class", "AUROC"]].rename(columns={"AUROC": "baseline_auc"})
    post = pd.read_csv(
        family_root / scenario / f"per_class_auc_{scenario}_L3.csv"
    )[["Class", "AUROC"]].rename(columns={"AUROC": "post_auc"})
    table = baseline.merge(post, on="Class", how="inner")
    table["baseline_auc"] = pd.to_numeric(table["baseline_auc"], errors="coerce")
    table["post_auc"] = pd.to_numeric(table["post_auc"], errors="coerce")
    table = table.loc[np.isfinite(table["baseline_auc"]) & np.isfinite(table["post_auc"])].copy()
    available = set(table["Class"].astype(str))
    missing = sorted((set(related_labels) | set(unrelated_labels)).difference(available))
    if missing:
        raise KeyError(f"{scenario}: collateral spec labels not present among valid L3 AUROCs: {missing}")
    table["group"] = "excluded"
    table.loc[table["Class"].astype(str).isin(related_labels), "group"] = "related"
    table.loc[table["Class"].astype(str).isin(unrelated_labels), "group"] = "unrelated"
    table["signed_auc_delta"] = table["post_auc"] - table["baseline_auc"]
    table["auc_loss"] = np.maximum(0.0, -table["signed_auc_delta"])
    audit_table = output_dir / f"{scenario}_collateral_per_class.csv"
    table.to_csv(audit_table, index=False)

    related_losses = table.loc[table["group"] == "related", "auc_loss"].to_numpy(dtype=np.float64)
    unrelated_losses = table.loc[table["group"] == "unrelated", "auc_loss"].to_numpy(dtype=np.float64)
    related_mean = float(related_losses.mean())
    unrelated_mean = float(unrelated_losses.mean())
    ratio = float(related_mean / unrelated_mean) if unrelated_mean > 0 else float("nan")
    low, high = ci_ratio(related_losses, unrelated_losses, bootstrap, seed)
    return common | {
        "status": "ok" if unrelated_mean > 0 else "undefined_zero_unrelated_loss",
        "related_label_count": int(related_losses.size),
        "unrelated_label_count": int(unrelated_losses.size),
        "related_mean_auc_loss": related_mean,
        "unrelated_mean_auc_loss": unrelated_mean,
        "collateral_ratio": ratio,
        "collateral_ratio_ci95_lower": low,
        "collateral_ratio_ci95_upper": high,
        "per_class_csv": str(audit_table),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("bootstrap must be non-negative")
    family_root = Path(args.output_dir) / args.family
    membership_dir = Path(args.membership_dir) if args.membership_dir else family_root / "audit_membership"
    output_dir = Path(args.audit_output_dir) if args.audit_output_dir else family_root / "audit_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_names(membership_dir, args.scenario)
    _, baseline_scores, class_names = read_l3_scores(family_root, args.baseline_scenario)
    collateral_specs = load_collateral_spec(args.collateral_spec)

    influence_rows = []
    collateral_rows = []
    for index, scenario in enumerate(scenarios):
        _, post_scores, post_class_names = read_l3_scores(family_root, scenario)
        if class_names != post_class_names or baseline_scores.shape != post_scores.shape:
            raise ValueError(f"{scenario}: baseline and post-withdrawal L3 score artifacts are not aligned.")
        q_aff = load_membership(membership_dir, scenario, baseline_scores.shape[0])
        influence_rows.append(
            influence_row(
                scenario=scenario,
                pre_scores=baseline_scores,
                post_scores=post_scores,
                q_aff=q_aff,
                threshold=args.prediction_threshold,
                bootstrap=args.bootstrap,
                seed=args.seed + 10 * index,
            )
        )
        collateral_rows.append(
            collateral_row(
                family_root=family_root,
                baseline_scenario=args.baseline_scenario,
                scenario=scenario,
                spec=collateral_specs.get(scenario),
                bootstrap=args.bootstrap,
                seed=args.seed + 10 * index + 2,
                output_dir=output_dir,
            )
        )
        print(f"[Computed] {scenario}", flush=True)

    influence_path = output_dir / "withdrawal_influence_leakage_summary.csv"
    collateral_path = output_dir / "withdrawal_collateral_ratio_summary.csv"
    pd.DataFrame(influence_rows).to_csv(influence_path, index=False)
    pd.DataFrame(collateral_rows).to_csv(collateral_path, index=False)
    manifest = {
        "family": args.family,
        "baseline_scenario": args.baseline_scenario,
        "scenarios": scenarios,
        "prediction_set_definition": f"L3 labels with voting score > {args.prediction_threshold:g}",
        "influence_leakage_definition": (
            "On Q_aff, report label-set Jaccard change and mean absolute L3 voting-score change from baseline to withdrawal."
        ),
        "collateral_ratio_definition": (
            "mean non-negative AUROC loss for predeclared related labels divided by that for predeclared unrelated labels."
        ),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "outputs": {
            "influence_summary_csv": str(influence_path),
            "collateral_summary_csv": str(collateral_path),
        },
    }
    manifest_path = output_dir / "withdrawal_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] Influence summary -> {influence_path}", flush=True)
    print(f"[Saved] Collateral summary -> {collateral_path}", flush=True)
    print(f"[Saved] Audit manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
