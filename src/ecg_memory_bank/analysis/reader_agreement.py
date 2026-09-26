from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_blinded_ab_scores import CASE_METRICS, SLOTS, decode_model, detect_batch_key, load_mapping, parse_score


MODELS = ("GLM5_RAG", "GLM5_ZERO")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def discover_rater_files(paths: list[Path]) -> dict[str, list[Path]]:
    rater_files: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        if path.is_file():
            rater_files[path.parent.name].append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        direct_files = sorted(path.glob("blinded_ab_scores_round2_batch_*.json"))
        if direct_files:
            rater_files[path.name].extend(direct_files)

        for child in sorted(item for item in path.iterdir() if item.is_dir()):
            child_files = sorted(child.glob("blinded_ab_scores_round2_batch_*.json"))
            if child_files:
                rater_files[child.name].extend(child_files)

    return {rater: sorted(set(files)) for rater, files in sorted(rater_files.items())}


def decode_case_level_rows(
    rater_files: dict[str, list[Path]],
    *,
    mapping_file: Path,
) -> list[dict[str, Any]]:
    mapping_by_case = load_mapping(mapping_file)["by_case"]
    rows: list[dict[str, Any]] = []
    missing_mapping: set[str] = set()

    for rater_id, files in rater_files.items():
        seen_units: set[tuple[str, str, str, str]] = set()
        for file_path in files:
            payload = load_json(file_path)
            batch_key = detect_batch_key(file_path) or ""
            source_file = str(file_path)

            for case_key, case_payload in payload.items():
                sample_id = str(case_payload.get("id") or case_key)
                mapping_item = mapping_by_case.get(sample_id)
                if not mapping_item:
                    missing_mapping.add(sample_id)
                    continue

                model_for_slot = {slot: decode_model(slot, mapping_item) for slot in SLOTS}
                for slot in SLOTS:
                    true_model = model_for_slot[slot]
                    case_scores = ((case_payload.get(slot) or {}).get("case_level") or {})
                    for metric in CASE_METRICS:
                        score = parse_score(case_scores.get(metric))
                        if score is None:
                            continue
                        unit = (rater_id, sample_id, true_model, metric)
                        if unit in seen_units:
                            raise ValueError(
                                f"Duplicate score for rater={rater_id}, sample={sample_id}, "
                                f"model={true_model}, metric={metric}"
                            )
                        seen_units.add(unit)
                        rows.append(
                            {
                                "rater_id": rater_id,
                                "source_file": source_file,
                                "batch_key": batch_key,
                                "sample_id": sample_id,
                                "model": true_model,
                                "metric": metric,
                                "score": score,
                            }
                        )

    if missing_mapping:
        raise ValueError(f"Missing A/B mapping for sample ids: {sorted(missing_mapping)}")
    return rows


def pivot_scores(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, float]]]:
    raters = sorted({str(row["rater_id"]) for row in rows})
    by_target: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        target = f'{row["sample_id"]}|{row["model"]}|{row["metric"]}'
        by_target[target][str(row["rater_id"])] = float(row["score"])
    return raters, dict(by_target)


def ordinal_distance(a: float, b: float) -> float:
    # Squared rank distance for the 1-5 ordinal Likert scale. The scale factor
    # cancels in Krippendorff's alpha, but normalization keeps the value bounded.
    return ((a - b) / 4.0) ** 2


def krippendorff_alpha_ordinal(by_target: dict[str, dict[str, float]]) -> tuple[float | None, int, int, int]:
    observed_distances: list[float] = []
    all_values: list[float] = []
    targets_with_2plus = 0

    for ratings in by_target.values():
        values = list(ratings.values())
        all_values.extend(values)
        if len(values) < 2:
            continue
        targets_with_2plus += 1
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                observed_distances.append(ordinal_distance(values[i], values[j]))

    expected_distances: list[float] = []
    for i in range(len(all_values)):
        for j in range(i + 1, len(all_values)):
            expected_distances.append(ordinal_distance(all_values[i], all_values[j]))

    if not observed_distances or not expected_distances:
        return None, targets_with_2plus, len(all_values), len(observed_distances)

    do = sum(observed_distances) / len(observed_distances)
    de = sum(expected_distances) / len(expected_distances)
    if de == 0:
        return None, targets_with_2plus, len(all_values), len(observed_distances)
    return 1.0 - do / de, targets_with_2plus, len(all_values), len(observed_distances)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def icc_complete_case(
    raters: list[str],
    by_target: dict[str, dict[str, float]],
) -> dict[str, float | int | None]:
    complete_rows = [[ratings[rater] for rater in raters] for ratings in by_target.values() if all(r in ratings for r in raters)]
    n = len(complete_rows)
    k = len(raters)
    if k < 2 or n < 2:
        return {
            "icc_n_complete_targets": n,
            "icc_k_raters": k,
            "icc2_1_absolute_agreement": None,
            "icc2_k_absolute_agreement": None,
            "icc3_1_consistency": None,
            "icc3_k_consistency": None,
        }

    row_means = [mean(row) for row in complete_rows]
    col_means = [mean([row[j] for row in complete_rows]) for j in range(k)]
    grand_mean = mean([value for row in complete_rows for value in row])

    ss_rows = k * sum((item - grand_mean) ** 2 for item in row_means)
    ss_cols = n * sum((item - grand_mean) ** 2 for item in col_means)
    ss_error = 0.0
    for i, row in enumerate(complete_rows):
        for j, value in enumerate(row):
            ss_error += (value - row_means[i] - col_means[j] + grand_mean) ** 2

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    def safe_div(num: float, den: float) -> float | None:
        if den == 0:
            return None
        return num / den

    icc2_1 = safe_div(ms_rows - ms_error, ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icc2_k = safe_div(ms_rows - ms_error, ms_rows + (ms_cols - ms_error) / n)
    icc3_1 = safe_div(ms_rows - ms_error, ms_rows + (k - 1) * ms_error)
    icc3_k = safe_div(ms_rows - ms_error, ms_rows)

    return {
        "icc_n_complete_targets": n,
        "icc_k_raters": k,
        "icc_ms_target": ms_rows,
        "icc_ms_rater": ms_cols,
        "icc_ms_error": ms_error,
        "icc2_1_absolute_agreement": icc2_1,
        "icc2_k_absolute_agreement": icc2_k,
        "icc3_1_consistency": icc3_1,
        "icc3_k_consistency": icc3_k,
    }


def summarize_group(group_type: str, group_value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    raters, by_target = pivot_scores(rows)
    alpha, targets_with_2plus, n_ratings, observed_pairs = krippendorff_alpha_ordinal(by_target)
    icc = icc_complete_case(raters, by_target)
    status = "ok" if len(raters) >= 2 and targets_with_2plus >= 2 else "insufficient_raters_or_targets"
    return {
        "group_type": group_type,
        "group_value": group_value,
        "status": status,
        "n_raters": len(raters),
        "raters": "; ".join(raters),
        "n_targets_total": len(by_target),
        "n_targets_with_2plus_raters": targets_with_2plus,
        "n_ratings": n_ratings,
        "n_observed_rater_pairs": observed_pairs,
        "krippendorff_alpha_ordinal": alpha,
        **icc,
    }


def build_reliability_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    summaries.append(summarize_group("ALL", "ALL", rows))

    for model in MODELS:
        summaries.append(summarize_group("model", model, [row for row in rows if row["model"] == model]))

    for metric in CASE_METRICS:
        metric_rows = [row for row in rows if row["metric"] == metric]
        summaries.append(summarize_group("metric", metric, metric_rows))
        for model in MODELS:
            summaries.append(
                summarize_group(
                    "metric_model",
                    f"{metric}|{model}",
                    [row for row in metric_rows if row["model"] == model],
                )
            )

    return summaries


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
        for row in rows:
            formatted = {
                key: safe_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def write_report(path: Path, rater_files: dict[str, list[Path]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Inter-Rater Reliability Analysis",
        "",
        "This report computes case-level reliability for the 1-5 rating scale.",
        "",
        "Targets are `sample_id × model × metric`; metrics are the seven case-level dimensions.",
        "Krippendorff's alpha uses squared ordinal rank distance on the 1-5 scale.",
        "ICC is computed on complete cases for the raters present in each group.",
        "",
        "## Rater Inputs",
        "",
        "| Rater | Files |",
        "|---|---:|",
    ]
    for rater_id, files in sorted(rater_files.items()):
        lines.append(f"| {rater_id} | {len(files)} |")

    if len(rater_files) < 2:
        lines.extend(
            [
                "",
                "## Status",
                "",
                "Inter-rater reliability cannot be estimated from the current inputs because fewer than 2 raters were detected.",
                "Place each doctor's five `blinded_ab_scores_round2_batch_*.json` files in a separate subdirectory, then rerun this script.",
                "",
                "Expected layout example:",
                "",
                "```text",
                "result_all_raters/",
                "  rater_1/blinded_ab_scores_round2_batch_0.json ... batch_4.json",
                "  rater_2/blinded_ab_scores_round2_batch_0.json ... batch_4.json",
                "  rater_3/blinded_ab_scores_round2_batch_0.json ... batch_4.json",
                "  rater_4/blinded_ab_scores_round2_batch_0.json ... batch_4.json",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Main Reliability Summary",
                "",
                "| Group | n raters | targets with >=2 raters | alpha ordinal | ICC(2,1) | ICC(2,k) | ICC(3,1) | ICC(3,k) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            if row["group_type"] in ("ALL", "metric"):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["group_value"]),
                            str(row["n_raters"]),
                            str(row["n_targets_with_2plus_raters"]),
                            safe_float(row.get("krippendorff_alpha_ordinal")),
                            safe_float(row.get("icc2_1_absolute_agreement")),
                            safe_float(row.get("icc2_k_absolute_agreement")),
                            safe_float(row.get("icc3_1_consistency")),
                            safe_float(row.get("icc3_k_consistency")),
                        ]
                    )
                    + " |"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-rater reliability for blinded A/B case-level ratings.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Rater directories, JSON files, or a root directory with one subdirectory per rater.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        required=True,
        help="Private A/B mapping JSON generated with the blinded pages.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for reliability outputs.",
    )
    args = parser.parse_args()

    rater_files = discover_rater_files(args.inputs)
    if not rater_files:
        raise FileNotFoundError("No blinded_ab_scores_round2_batch_*.json files found.")

    rows = decode_case_level_rows(rater_files, mapping_file=args.mapping_file)
    summary_rows = build_reliability_summary(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "case_level_ratings_long.csv", rows)
    write_csv(args.output_dir / "case_level_irr_summary.csv", summary_rows)
    write_report(args.output_dir / "inter_rater_reliability_report.md", rater_files, summary_rows)

    payload = {
        "rater_count": len(rater_files),
        "raters": {rater: [str(path) for path in files] for rater, files in rater_files.items()},
        "rating_count": len(rows),
        "outputs": {
            "case_level_ratings_long": str(args.output_dir / "case_level_ratings_long.csv"),
            "case_level_irr_summary": str(args.output_dir / "case_level_irr_summary.csv"),
            "report": str(args.output_dir / "inter_rater_reliability_report.md"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
