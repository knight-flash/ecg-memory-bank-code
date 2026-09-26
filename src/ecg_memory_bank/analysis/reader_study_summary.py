from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Iterable


BASE = Path("FinalResult/Result&Code/LLM/human/eval_analysis_all4")
IRR_DIR = BASE / "inter_rater_reliability"
OUT = BASE / "analysis_report.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    text = text.replace("|", "\\|")
    return text if text else "--"


def fmt_float(value: object, ndigits: int = 4) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "--"
    try:
        return f"{float(text):.{ndigits}f}"
    except ValueError:
        return clean(text)


def fmt_int(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "--"
    try:
        return f"{int(float(text)):,}"
    except ValueError:
        return clean(text)


def wins(row: dict[str, str]) -> str:
    return f"{fmt_int(row.get('rag_wins'))}/{fmt_int(row.get('zero_wins'))}/{fmt_int(row.get('ties'))}"


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(clean(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def pair_table(rows: list[dict[str, str]], key_col: str, key_name: str = "Group") -> str:
    return md_table(
        [
            key_name,
            "n",
            "RAG mean",
            "RAG median",
            "ZERO mean",
            "ZERO median",
            "mean diff",
            "median diff",
            "sd diff",
            "RAG/ZERO/tie",
            "p",
        ],
        [
            [
                r.get(key_col, ""),
                fmt_int(r.get("n_pairs")),
                fmt_float(r.get("glm5_rag_mean")),
                fmt_float(r.get("glm5_rag_median")),
                fmt_float(r.get("glm5_zero_mean")),
                fmt_float(r.get("glm5_zero_median")),
                fmt_float(r.get("mean_diff_rag_minus_zero")),
                fmt_float(r.get("median_diff_rag_minus_zero")),
                fmt_float(r.get("sd_diff")),
                wins(r),
                fmt_float(r.get("sign_test_p_two_sided")),
            ]
            for r in rows
        ],
    )


def model_summary_table(rows: list[dict[str, str]], group_key: str) -> str:
    return md_table(
        ["Group", "Model", "Metric", "n", "mean", "sd", "median", "q1", "q3", "min", "max"],
        [
            [
                r.get(group_key, ""),
                r.get("model", ""),
                r.get("metric", ""),
                fmt_int(r.get("n")),
                fmt_float(r.get("mean")),
                fmt_float(r.get("sd")),
                fmt_float(r.get("median")),
                fmt_float(r.get("q1")),
                fmt_float(r.get("q3")),
                fmt_float(r.get("min")),
                fmt_float(r.get("max")),
            ]
            for r in rows
        ],
    )


def support_pair_table(rows: list[dict[str, str]]) -> str:
    return md_table(
        [
            "Type",
            "Group",
            "support min",
            "support median",
            "support max",
            "n pairs",
            "unique labels",
            "support sum",
            "RAG mean",
            "ZERO mean",
            "mean diff",
            "median diff",
            "sd diff",
            "RAG/ZERO/tie",
            "p",
            "labels",
        ],
        [
            [
                r.get("group_type", ""),
                r.get("group_value", ""),
                fmt_int(r.get("reference_support_min")),
                fmt_float(r.get("reference_support_median")),
                fmt_int(r.get("reference_support_max")),
                fmt_int(r.get("n_pairs")),
                fmt_int(r.get("n_unique_labels")),
                fmt_int(r.get("reference_support_unique_sum")),
                fmt_float(r.get("glm5_rag_mean")),
                fmt_float(r.get("glm5_zero_mean")),
                fmt_float(r.get("mean_diff_rag_minus_zero")),
                fmt_float(r.get("median_diff_rag_minus_zero")),
                fmt_float(r.get("sd_diff")),
                wins(r),
                fmt_float(r.get("sign_test_p_two_sided")),
                r.get("unique_labels", ""),
            ]
            for r in rows
        ],
    )


def l1_l2_table(rows: list[dict[str, str]]) -> str:
    return md_table(
        [
            "Level",
            "L1",
            "L2",
            "Group",
            "n pairs",
            "unique labels",
            "support sum",
            "support min",
            "support median",
            "support max",
            "RAG mean",
            "ZERO mean",
            "mean diff",
            "median diff",
            "sd diff",
            "RAG/ZERO/tie",
            "p",
            "labels",
        ],
        [
            [
                r.get("level", ""),
                r.get("l1_category", ""),
                r.get("l2_sub_group", ""),
                r.get("group_value", ""),
                fmt_int(r.get("n_pairs")),
                fmt_int(r.get("n_unique_labels")),
                fmt_int(r.get("reference_support_unique_sum")),
                fmt_int(r.get("reference_support_min")),
                fmt_float(r.get("reference_support_median")),
                fmt_int(r.get("reference_support_max")),
                fmt_float(r.get("glm5_rag_mean")),
                fmt_float(r.get("glm5_zero_mean")),
                fmt_float(r.get("mean_diff_rag_minus_zero")),
                fmt_float(r.get("median_diff_rag_minus_zero")),
                fmt_float(r.get("sd_diff")),
                wins(r),
                fmt_float(r.get("sign_test_p_two_sided")),
                r.get("unique_labels", ""),
            ]
            for r in rows
        ],
    )


def irr_table(rows: list[dict[str, str]]) -> str:
    return md_table(
        [
            "Type",
            "Group",
            "status",
            "raters",
            "targets",
            "targets >=2",
            "ratings",
            "alpha ordinal",
            "ICC(2,1)",
            "ICC(2,k)",
            "ICC(3,1)",
            "ICC(3,k)",
        ],
        [
            [
                r.get("group_type", ""),
                r.get("group_value", ""),
                r.get("status", ""),
                r.get("n_raters", ""),
                fmt_int(r.get("n_targets_total")),
                fmt_int(r.get("n_targets_with_2plus_raters")),
                fmt_int(r.get("n_ratings")),
                fmt_float(r.get("krippendorff_alpha_ordinal")),
                fmt_float(r.get("icc2_1_absolute_agreement")),
                fmt_float(r.get("icc2_k_absolute_agreement")),
                fmt_float(r.get("icc3_1_consistency")),
                fmt_float(r.get("icc3_k_consistency")),
            ]
            for r in rows
        ],
    )


def per_rater_tables(case_diff: list[dict[str, str]], label_diff: list[dict[str, str]]) -> tuple[str, str]:
    metrics = ["relevance", "accuracy", "completeness", "reasoning", "novelty", "clinical_value", "satisfaction"]
    by_rater_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in case_diff:
        try:
            by_rater_metric[r["rater_id"]][r["metric"]].append(float(r["diff"]))
        except (KeyError, ValueError):
            continue

    label_by_rater: dict[str, list[float]] = defaultdict(list)
    for r in label_diff:
        try:
            label_by_rater[r["rater_id"]].append(float(r["diff"]))
        except (KeyError, ValueError):
            continue

    rows = []
    for rater in sorted(set(by_rater_metric) | set(label_by_rater)):
        all_case = [v for vals in by_rater_metric[rater].values() for v in vals]
        rows.append(
            [
                rater,
                len(all_case),
                f"{mean(all_case):.4f}" if all_case else "--",
                *[
                    f"{mean(by_rater_metric[rater][metric]):.4f}"
                    if by_rater_metric[rater].get(metric)
                    else "--"
                    for metric in metrics
                ],
                len(label_by_rater[rater]),
                f"{mean(label_by_rater[rater]):.4f}" if label_by_rater[rater] else "--",
            ]
        )
    table1 = md_table(["Rater", "case pairs", "case mean diff", *metrics, "label pairs", "label mean diff"], rows)

    case_rater_metric_rows = []
    for rater in sorted(by_rater_metric):
        for metric in metrics:
            vals = by_rater_metric[rater].get(metric, [])
            if not vals:
                continue
            wins_n = sum(1 for v in vals if v > 0)
            zero_n = sum(1 for v in vals if v < 0)
            tie_n = sum(1 for v in vals if v == 0)
            case_rater_metric_rows.append(
                [rater, metric, len(vals), f"{mean(vals):.4f}", f"{wins_n}/{zero_n}/{tie_n}"]
            )
    table2 = md_table(["Rater", "metric", "n", "mean diff", "RAG/ZERO/tie"], case_rater_metric_rows)
    return table1, table2


def main() -> None:
    summary = read_json(BASE / "summary.json")
    irr_summary = read_json(IRR_DIR / "summary.json")

    case_model = read_csv(BASE / "case_level_summary_by_model.csv")
    case_pair = read_csv(BASE / "case_level_paired_summary.csv")
    case_diff = read_csv(BASE / "case_level_paired_diffs.csv")
    label_model = read_csv(BASE / "label_level_summary_by_model.csv")
    label_support_model = read_csv(BASE / "label_level_summary_by_support_tier.csv")
    label_pair = read_csv(BASE / "label_level_paired_summary.csv")
    label_support_pair = read_csv(BASE / "label_level_support_tier_paired_summary.csv")
    label_l1_l2 = read_csv(BASE / "label_level_l1_l2_paired_summary.csv")
    label_diff = read_csv(BASE / "label_level_paired_diffs.csv")
    label_audit = read_csv(BASE / "label_audit.csv")
    irr_rows = read_csv(IRR_DIR / "case_level_irr_summary.csv")

    audit_counts = Counter((r.get("valid") or "").strip() or "blank" for r in label_audit)
    audit_by_rater: dict[str, Counter[str]] = defaultdict(Counter)
    for r in label_audit:
        audit_by_rater[r.get("rater_id", "")][(r.get("valid") or "").strip() or "blank"] += 1
    comments = [r for r in label_audit if (r.get("comment") or "").strip()]

    per_rater_overall, per_rater_metric = per_rater_tables(case_diff, label_diff)

    label_pair_all = [r for r in label_pair if r["group_type"] == "ALL"]
    label_pair_category = [r for r in label_pair if r["group_type"] == "category"]
    label_pair_mapped = [r for r in label_pair if r["group_type"] == "mapped_label"]

    irr_main = [r for r in irr_rows if r["group_type"] in {"ALL", "model", "metric"}]
    irr_metric_model = [r for r in irr_rows if r["group_type"] == "metric_model"]

    lines: list[str] = []
    lines.append("# GLM-5 RAG vs GLM-5 ZERO Blinded A/B Reader Study: Complete All-4 Analysis")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")
    lines.append("This report is generated from the pooled four-rater analysis directory:")
    lines.append("")
    lines.append(f"- `{BASE.as_posix()}`")
    lines.append(f"- Inter-rater reliability: `{IRR_DIR.as_posix()}`")
    lines.append("")

    lines.append("## 1. Data Scope")
    lines.append("")
    lines.append(
        md_table(
            ["Item", "Value"],
            [
                ["Raters", ", ".join(sorted(irr_summary.get("raters", {}).keys()))],
                ["Source JSON files", len(summary.get("source_files", []))],
                ["Rater-case records", fmt_int(summary.get("case_count"))],
                ["Case-level model scores", fmt_int(summary.get("case_level_model_scores"))],
                ["Case-level paired comparisons", fmt_int(summary.get("case_level_pairs"))],
                ["Label-level model scores", fmt_int(summary.get("label_level_model_scores"))],
                ["Label-level paired comparisons", fmt_int(summary.get("label_level_pairs"))],
                ["Label audit yes", fmt_int(audit_counts.get("yes", 0))],
                ["Label audit no", fmt_int(audit_counts.get("no", 0))],
                ["Label audit blank", fmt_int(audit_counts.get("blank", 0))],
            ],
        )
    )
    lines.append("")
    lines.append(
        "Expected full case-level pairs are `4 raters x 100 cases x 7 metrics = 2,800`; "
        f"the current complete paired count is `{fmt_int(summary.get('case_level_pairs'))}`, "
        "so paired analyses automatically skip the few missing dropdown entries."
    )
    lines.append("")

    lines.append("## 2. Main Takeaways")
    lines.append("")
    lines.append("- Case-level scores favor `GLM5_RAG` over `GLM5_ZERO` on all seven dimensions.")
    lines.append("- Label-level clinical-value scores also favor `GLM5_RAG` overall and in most mapped L1 categories.")
    lines.append("- The largest label-level gains are in `Pacemaker`, `Conduction`, `Axis`, and `Ischemia events`.")
    lines.append("- `Rhythm` and `General report` contain many easy/common labels, so they include many ties.")
    lines.append("- Support-tier analysis is head-dominated; only 4 paired tail label instances are present, so tail claims should be avoided for this reader study.")
    lines.append("- Inter-rater reliability is acceptable for four-rater averaged comparisons, but not high at the single-rater level. `novelty` remains the weakest reliability dimension.")
    lines.append("")

    lines.append("## 3. Case-Level Ratings")
    lines.append("")
    lines.append("### 3.1 By Model")
    lines.append("")
    lines.append(model_summary_table(case_model, "metric"))
    lines.append("")
    lines.append("### 3.2 Paired RAG vs ZERO")
    lines.append("")
    lines.append(pair_table(case_pair, "metric", "Metric"))
    lines.append("")

    lines.append("## 4. Per-Rater Direction")
    lines.append("")
    lines.append("### 4.1 Overall Per Rater")
    lines.append("")
    lines.append(per_rater_overall)
    lines.append("")
    lines.append("### 4.2 Case-Level Per Rater x Metric")
    lines.append("")
    lines.append(per_rater_metric)
    lines.append("")

    lines.append("## 5. Label-Level Clinical-Value Ratings")
    lines.append("")
    lines.append("### 5.1 Overall Paired Result")
    lines.append("")
    lines.append(pair_table(label_pair_all, "group_value", "Group"))
    lines.append("")
    lines.append("### 5.2 L1 / 9-Class Category Paired Results")
    lines.append("")
    lines.append(pair_table(label_pair_category, "group_value", "Category"))
    lines.append("")
    lines.append("### 5.3 Mapped-Label / Raw-Label Paired Results")
    lines.append("")
    lines.append(pair_table(label_pair_mapped, "group_value", "Label"))
    lines.append("")

    lines.append("## 6. L1 / L2 Taxonomy Summary")
    lines.append("")
    lines.append("This table uses the retrieval label taxonomy columns `category` and `sub_group`.")
    lines.append("")
    lines.append(l1_l2_table(label_l1_l2))
    lines.append("")

    lines.append("## 7. Support-Tier Summary")
    lines.append("")
    lines.append("Support-tier definition inherited from the retrieval analysis: `tail=1-1000`, `low_mid=1001-5000`, `mid=5001-10000`, `head>=10001`; unmapped labels are reported separately.")
    lines.append("")
    lines.append(support_pair_table(label_support_pair))
    lines.append("")

    lines.append("## 8. Label-Level By-Model Descriptives")
    lines.append("")
    lines.append("### 8.1 By L1 / Category")
    lines.append("")
    lines.append(model_summary_table(label_model, "group"))
    lines.append("")
    lines.append("### 8.2 By Support Tier")
    lines.append("")
    lines.append(model_summary_table(label_support_model, "support_tier"))
    lines.append("")

    lines.append("## 9. Label Audit")
    lines.append("")
    lines.append("### 9.1 Audit Counts By Rater")
    lines.append("")
    audit_rater_rows = []
    for rater in sorted(audit_by_rater):
        c = audit_by_rater[rater]
        audit_rater_rows.append([rater, fmt_int(c.get("yes", 0)), fmt_int(c.get("no", 0)), fmt_int(c.get("blank", 0)), fmt_int(sum(c.values()))])
    lines.append(md_table(["Rater", "yes", "no", "blank", "total"], audit_rater_rows))
    lines.append("")
    lines.append("### 9.2 All Non-Empty Audit Comments")
    lines.append("")
    lines.append(
        md_table(
            ["Rater", "Sample", "valid", "label count", "comment"],
            [
                [
                    r.get("rater_id", ""),
                    r.get("sample_id", ""),
                    (r.get("valid") or "").strip() or "blank",
                    fmt_int(r.get("label_count")),
                    r.get("comment", ""),
                ]
                for r in comments
            ],
        )
    )
    lines.append("")

    lines.append("## 10. Inter-Rater Reliability")
    lines.append("")
    lines.append("Krippendorff's alpha uses the ordinal 1-5 scale. ICC rows are case-level reliability summaries over `sample_id x model x metric` targets.")
    lines.append("")
    lines.append("### 10.1 Main IRR Rows")
    lines.append("")
    lines.append(irr_table(irr_main))
    lines.append("")
    lines.append("### 10.2 Metric x Model IRR Rows")
    lines.append("")
    lines.append(irr_table(irr_metric_model))
    lines.append("")

    lines.append("## 11. Writing Guidance")
    lines.append("")
    lines.append("- Recommended main claim: four-reader blinded A/B scoring favors `GLM5_RAG` over `GLM5_ZERO` across all seven case-level dimensions and overall label-level clinical value.")
    lines.append("- Reliability wording: use `modest-to-moderate agreement`; do not write `high agreement` or `near consensus`.")
    lines.append("- `ICC(2,k)=0.669962` supports using the four-reader average for model comparison; `ICC(2,1)=0.336645` shows individual-reader scores are more variable.")
    lines.append("- `novelty` can be retained as a secondary/exploratory dimension because it is directionally positive, but its IRR is weakest (`alpha=0.056995`).")
    lines.append("- L1/L2 and support-tier tables are useful as exploratory decomposition. Because this 100-case reader set is head-label dominated, do not use it as strong evidence for tail-label reader benefits.")
    lines.append("")

    lines.append("## 12. Output Files Included In This Report")
    lines.append("")
    for path in [
        BASE / "summary.json",
        BASE / "case_level_summary_by_model.csv",
        BASE / "case_level_paired_summary.csv",
        BASE / "case_level_paired_diffs.csv",
        BASE / "label_level_summary_by_model.csv",
        BASE / "label_level_summary_by_support_tier.csv",
        BASE / "label_level_paired_summary.csv",
        BASE / "label_level_support_tier_paired_summary.csv",
        BASE / "label_level_l1_l2_paired_summary.csv",
        BASE / "label_level_paired_diffs.csv",
        BASE / "label_audit.csv",
        IRR_DIR / "summary.json",
        IRR_DIR / "case_level_irr_summary.csv",
        IRR_DIR / "inter_rater_reliability_report.md",
    ]:
        lines.append(f"- `{path.as_posix()}`")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
