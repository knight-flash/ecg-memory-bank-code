"""Supported paper experiment runner.

Purpose
-------
Expose one command interface for the manuscript's retrieval, prototype,
report, judge, and reader analyses.

Method
------
Load a JSON config, reject overrides of locked paper parameters, require
outputs outside the archive, and dispatch the appropriate source script and
postprocessing chain.

Inputs
------
Task name, JSON with external data paths and task flags, output directory, and
optional dry-run flag.

Outputs
-------
External run directories containing result artifacts; dry-run prints commands
without reading data or calling APIs.

Paper parameters
----------------
Locks fractions, seed, K, MIMIC subject split, Chagas cohort sizes, prototype
budget and report cohort/model settings by task.

Shapes and meaning
------------------
Inputs include vectors [N,D], labels [N,C], and JSONL reports; task-specific
outputs range from [Q,C] scores to CI and audit tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from ecg_memory_bank.protocol import (  # noqa: E402
    CHAGAS_ADDITION_STAGES,
    HEEDB_MEMORY_FRACTIONS,
    HISTORICAL_YEAR_THRESHOLDS,
    PATIENT_WITHDRAWAL_FRACTIONS,
    external_output_dir,
    require_input_file,
)

PKG = SRC / "ecg_memory_bank"
TASKS = {
    "broad-heedb": ("experiments/broad_case_addition_heedb.py", "output-dir"),
    "broad-mimic": ("experiments/broad_case_addition_mimic.py", "output-dir"),
    "chagas-addition": ("experiments/targeted_case_addition_chagas.py", "output-dir"),
    "withdrawal": ("experiments/active_memory_withdrawal.py", "output-dir"),
    "encoder-full": ("sensitivity/full_memory_retrieval.py", "output-dir"),
    "encoder-withdrawal": ("sensitivity/active_memory_withdrawal.py", "output-dir"),
    "encoder-chagas": ("sensitivity/targeted_case_addition_chagas.py", "output-dir"),
    "prototype-heedb": ("heads/prototype_heedb_multilabel.py", "output-root"),
    "prototype-mimic": ("heads/prototype_mimic_multilabel.py", "output-root"),
    "prototype-code15": ("heads/prototype_code15_multilabel.py", "output-root"),
    "prototype-chagas": ("heads/prototype_chagas.py", "output-root"),
    "linear": ("heads/train_linear_head.py", "output-root"),
    "reader-study": ("analysis/reader_study.py", "output-dir"),
}
REQUIRED_FLAGS = {
    "broad-heedb": {"heedb-train-ecg-path", "heedb-train-meta-path", "heedb-test-ecg-path", "heedb-test-meta-path", "record-id-col"},
    "broad-mimic": {"mimic-ecg-path", "mimic-meta-path", "record-id-col"},
    "chagas-addition": {"code15-ecg-path", "code15-meta-path"},
    "withdrawal": {"train-meta-path", "train-ecg-path", "val-meta-path", "val-ecg-path", "test-meta-path", "test-ecg-path", "train-enrich-meta-path", "val-enrich-meta-path", "record-id-col", "patient-id-col"},
    "encoder-full": {"heedb-train-ecg-path", "heedb-train-meta-path", "heedb-test-ecg-path", "heedb-test-meta-path"},
    "encoder-withdrawal": {"train-meta-path", "train-ecg-path", "val-meta-path", "val-ecg-path", "test-meta-path", "test-ecg-path", "train-enrich-meta-path", "val-enrich-meta-path", "record-id-col", "patient-id-col"},
    "encoder-chagas": {"code15-ecg-path", "code15-meta-path"},
    "linear": {"task"},
    "reader-study": {"mapping-file"},
}
FILE_SUFFIXES = ("-path", "-csv", "-xlsx", "-file")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[*TASKS, "report-glm46", "report-glm5", "judge"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def encode_flags(values: dict[str, object]) -> list[str]:
    encoded: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-").removeprefix("--")
        if isinstance(value, bool):
            if value:
                encoded.append(flag)
        elif isinstance(value, list):
            encoded.extend([flag, *map(str, value)])
        else:
            encoded.extend([flag, str(value)])
    return encoded


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = ["<API_KEY>" if "key" in part.lower() and len(part) > 30 else part for part in command]
    print("RUN", json.dumps(printable, ensure_ascii=False))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def prepare_environment(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["ECG_OUTPUT_ROOT"] = str(output_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def locked_flags(task: str) -> dict[str, object]:
    if task in {"broad-heedb", "broad-mimic"}:
        fixed = {"sampling-mode": "anchored_nested", "gallery-fractions": list(HEEDB_MEMORY_FRACTIONS), "top-k": 10, "seed": 42}
        if task == "broad-mimic":
            fixed.update({"split-by": "subject", "val-ratio": 0.2})
        return fixed
    if task in {"chagas-addition", "encoder-chagas"}:
        return {"stage-sizes": list(CHAGAS_ADDITION_STAGES), "top-k": 10, "seed": 42, "bootstrap": 1000,
                "positive-test-size": 1000, "negative-test-size": 4000}
    if task in {"withdrawal", "encoder-withdrawal"}:
        return {"query-years": [2022], "historical-year-thresholds": list(HISTORICAL_YEAR_THRESHOLDS), "patient-withdrawal-fractions": list(PATIENT_WITHDRAWAL_FRACTIONS), "top-k": 10, "seed": 42}
    if task == "encoder-full":
        return {"gallery-fractions": [1.0], "top-k": 10, "seed": 42}
    if task.startswith("prototype-"):
        return {"prototype-budget": 3000, "min-prototypes-per-class": 1, "max-prototypes-per-class": 30, "seed": 42, "bootstrap": 1000}
    if task == "linear":
        return {"seed": 42, "bootstrap": 1000}
    return {}


def run_script_task(task: str, config: dict, output_dir: Path, env: dict[str, str], dry_run: bool) -> None:
    script, output_flag = TASKS[task]
    supplied = dict(config.get("args", {}))
    forbidden = set(locked_flags(task)) | {
        output_flag, "checkpoint-dir", "include-val-query-years-in-gallery",
        "sampling-order-path", "anchor-fractions", "max-queries",
        "patient-withdrawal-count", "patient-withdrawal-fraction", "historical-year-threshold",
    }
    if forbidden.intersection(supplied):
        raise ValueError(f"Paper protocol or output flags cannot be overridden: {sorted(forbidden.intersection(supplied))}")
    missing = REQUIRED_FLAGS.get(task, set()) - supplied.keys()
    if missing:
        raise ValueError(f"Missing required flags for {task}: {sorted(missing)}")
    if task == "linear" and supplied["task"] not in {"ape120", "mimic100", "code15", "code15_chagas"}:
        raise ValueError("The paper linear control has four training tasks and five evaluation settings")
    for flag, value in supplied.items():
        if flag.endswith(FILE_SUFFIXES) and flag not in {"record-id-col", "patient-id-col", "label-col"} and isinstance(value, str):
            if flag not in {"finalgallery-root", "output-dir", "output-root"} and not dry_run:
                require_input_file(value, flag)
    for item in config.get("input_files", []):
        if not dry_run:
            require_input_file(item, "input_files entry")
    if (task.startswith("prototype-") or task == "linear") and not config.get("input_files"):
        raise ValueError(f"{task} requires input_files listing all external datasets")
    output = output_dir / task
    fixed = locked_flags(task)
    if task.startswith("broad-") or task in {"withdrawal", "encoder-full", "encoder-withdrawal"}:
        fixed["heedb-label-mapping-csv" if task != "broad-mimic" else "mimic-label-mapping-csv"] = str(ROOT / "configs" / "label_category_counts_9class.csv")
    if task in {"prototype-heedb", "prototype-mimic"}:
        fixed["label-config-csv"] = str(ROOT / "configs" / "l3_full_label_curated.csv")
    if task in {"prototype-heedb", "prototype-mimic", "prototype-code15"}:
        fixed["save-summary-json"] = str(output / "summary.json")
        fixed["save-per-class-csv"] = str(output / "per_class_auc.csv")
    command = [sys.executable, str(PKG / script), *encode_flags(supplied), *encode_flags(fixed), "--" + output_flag, str(output)]
    if task == "reader-study":
        inputs = config.get("reader_inputs", [])
        if not inputs:
            raise ValueError("reader-study requires reader_inputs")
        command.extend(map(str, inputs))
    run(command, env, dry_run)
    if task in {"broad-heedb", "broad-mimic"}:
        scale_name = "scale_benchmark" if task == "broad-heedb" else "mimic_scale_benchmark"
        scale_dir = output / scale_name
        label_count = 114 if task == "broad-heedb" else 83
        run([sys.executable, str(PKG / "retrieval" / "fixed_label_auc_ci.py"),
             "--input-root", str(scale_dir), "--summary-file", scale_name + "_summary.csv",
             "--expected-label-count", str(label_count), "--output-dir", str(output / "fixed_label_ci"),
             "--bootstrap", "1000", "--seed", "42"], env, dry_run)
        if task == "broad-heedb":
            tiers = output / "support_tiers.csv"
            run([sys.executable, str(PKG / "retrieval" / "build_support_tiers.py"),
                 "--metadata-csv", str(supplied["heedb-train-meta-path"]),
                 "--mapping-csv", str(ROOT / "configs" / "label_category_counts_9class.csv"),
                 "--output-csv", str(tiers)], env, dry_run)
            run([sys.executable, str(PKG / "retrieval" / "build_heedb_loglinear_case_addition.py"),
                 "--input-root", str(scale_dir), "--tier-manifest", str(tiers),
                 "--output-dir", str(output / "loglinear")], env, dry_run)
    if task == "reader-study":
        run([sys.executable, str(PKG / "analysis" / "reader_study_ci.py"), "--input-dir", str(output), "--output-dir", str(output)], env, dry_run)
        run([sys.executable, str(PKG / "analysis" / "reader_agreement.py"), *map(str, config["reader_inputs"]), "--mapping-file", str(supplied["mapping-file"]), "--output-dir", str(output / "agreement")], env, dry_run)


def run_report_task(task: str, config: dict, output_dir: Path, env: dict[str, str], dry_run: bool) -> None:
    paths = config.get("paths", {})
    required = {"grounding_json", "mimic_embeddings", "mimic_metadata", "mimic_feature_root", "classification_xlsx"}
    if required - paths.keys():
        raise ValueError(f"Missing report paths: {sorted(required - paths.keys())}")
    if not dry_run:
        for key in required - {"mimic_feature_root"}:
            require_input_file(paths[key], key)
        if not Path(paths["mimic_feature_root"]).is_dir():
            raise FileNotFoundError(paths["mimic_feature_root"])
        with Path(paths["grounding_json"]).open(encoding="utf-8") as handle:
            cases = json.load(handle)
        if not isinstance(cases, list) or len(cases) != 2381:
            raise ValueError("The paper report cohort must contain exactly 2381 cases")
        if not env.get("ZHIPU_API_KEY"):
            raise RuntimeError("ZHIPU_API_KEY is required for report generation")
    env.update({
        "ECG_MIMIC_EMBEDDINGS": str(paths["mimic_embeddings"]),
        "ECG_MIMIC_METADATA": str(paths["mimic_metadata"]),
        "ECG_MIMIC_FEATURE_ROOT": str(paths["mimic_feature_root"]),
        "ECG_LABEL_COUNTS_CSV": str(paths.get("label_counts_csv", "__REBUILD_FROM_METADATA__")),
        "ECG_CLASSIFICATION_XLSX": str(paths["classification_xlsx"]),
    })
    model = "glm-4.6" if task == "report-glm46" else "glm-5"
    run_dir = output_dir / task
    subset = run_dir / "subsets" / "grounding_query_subset_input.csv"
    prep = [sys.executable, str(PKG / "reports" / "prepare_grounding_subset.py"), "--input-json", str(paths["grounding_json"]), "--output-csv", str(subset), "--report-json", str(run_dir / "subset_report.json"), "--metadata-csv", str(paths["mimic_metadata"]), "--feature-root", str(paths["mimic_feature_root"])]
    pipeline = [sys.executable, str(PKG / "reports" / "generate_grounded_reports.py"), "--run-name", task, "--run-dir", str(run_dir), "--query-size", "2381", "--query-subset-csv", str(subset), "--top-k", "5", "--infer-model", model, "--infer-thinking-type", "enabled", "--infer-max-tokens", "8192", "--mode", "all"]
    run(prep, env, dry_run)
    if not dry_run:
        with subset.open(encoding="utf-8-sig", newline="") as handle:
            selected = sum(1 for _ in csv.DictReader(handle))
        if selected != 2381:
            raise ValueError(f"Grounding preparation retained {selected} cases; expected 2381")
    run(pipeline, env, dry_run)
    retry_dir = run_dir / "retries" / "retry_paper"
    bestof_dir = run_dir / "retries" / "retry_paper_bestof"
    run([sys.executable, str(PKG / "reports" / "retry_failed_cases.py"),
         "--run-dir", str(run_dir), "--mode", "all", "--retry-name", "retry_paper",
         "--retry-strategy", "both", "--failure-scope", "json_or_score_incomplete",
         "--model", model, "--thinking-type", "enabled", "--max-tokens", "8192"], env, dry_run)
    run([sys.executable, str(PKG / "reports" / "best_of_retry_compare.py"),
         "--run-dir", str(run_dir), "--retry-dir", str(retry_dir), "--output-dir", str(bestof_dir)], env, dry_run)
    run([sys.executable, str(PKG / "reports" / "bootstrap_grounding_ci.py"),
         "--run-dir", str(bestof_dir), "--bootstrap", "1000", "--seed", "42"], env, dry_run)
    if task == "report-glm5":
        run([sys.executable, str(PKG / "reports" / "case_link_audit.py"),
             "--run-dir", str(run_dir), "--selection-csv",
             str(bestof_dir / "reports" / "best_retry_selection_original_first.csv"),
             "--output-dir", str(run_dir / "case_link_audit")], env, dry_run)


def run_judge_task(config: dict, output_dir: Path, env: dict[str, str], dry_run: bool) -> None:
    requests = config.get("requests_jsonl")
    model = config.get("endpoint_model", "deepseek-v3.1")
    run_dir = output_dir / "judge"
    if not requests:
        needed = {"bestof_dir", "retry_dir", "run_dir", "groundtruth_jsonl"}
        if needed - config.keys():
            raise ValueError(f"judge requires requests_jsonl or {sorted(needed)}")
        requests = run_dir / "requests" / "mimic_reasoning_eval_all.jsonl"
        prep = [sys.executable, str(PKG / "reports" / "prepare_judge_requests.py"),
                "--bestof-dir", str(config["bestof_dir"]), "--retry-dir", str(config["retry_dir"]),
                "--run-dir", str(config["run_dir"]), "--groundtruth-jsonl", str(config["groundtruth_jsonl"]),
                "--prompt-template", str(ROOT / "configs" / "judge_prompt.txt"),
                "--output-dir", str(run_dir / "requests"), "--mode", "all"]
        run(prep, env, dry_run)
    if not dry_run:
        require_input_file(requests, "requests_jsonl")
        if not (env.get("ARK_API_KEY") or env.get("VOLCENGINE_API_KEY")):
            raise RuntimeError("ARK_API_KEY or VOLCENGINE_API_KEY is required")
    judged = run_dir / "judge_outputs.jsonl"
    command = [sys.executable, str(PKG / "reports" / "judge_reports.py"), "--input", str(requests), "--output", str(judged), "--summary-json", str(run_dir / "request_summary.json"), "--model", model, "--thinking-type", "disabled", "--max-tokens", "8192", "--resume"]
    if config.get("base_url"):
        command.extend(["--base-url", str(config["base_url"])])
    run(command, env, dry_run)
    run([sys.executable, str(PKG / "reports" / "summarize_judge_scores.py"), "--input", str(judged), "--output-dir", str(run_dir / "scores"), "--name", "paper_deepseek_judge", "--eval-llm", "deepseek-ai/DeepSeek-V3.1-Terminus"], env, dry_run)


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = external_output_dir(args.output_dir)
    env = prepare_environment(output_dir)
    if args.task in TASKS:
        run_script_task(args.task, config, output_dir, env, args.dry_run)
    elif args.task.startswith("report-"):
        run_report_task(args.task, config, output_dir, env, args.dry_run)
    else:
        run_judge_task(config, output_dir, env, args.dry_run)


if __name__ == "__main__":
    main()
