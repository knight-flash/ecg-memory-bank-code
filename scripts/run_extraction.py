"""Supported embedding extraction runner.

Purpose
-------
Dispatch the original HEEDB, MIMIC, or CODE-15 ECGCLIP extraction
implementation without storing data in Git.

Method
------
Validate external config/data/MedCPT paths and output location, set local
imports, and launch distributed HEEDB or single-device MIMIC/CODE-15
extraction.

Inputs
------
Dataset name, model YAML with external checkpoint, MedCPT directory, index
CSV, dataset-specific raw root, and external output directory.

Outputs
-------
External embedding tensors, aligned metadata, progress files and extraction
summaries; dry-run only prints the command.

Paper parameters
----------------
Default HEEDB worker count is six; batch/GPU flags affect throughput and do
not change the frozen checkpoint or target preprocessing.

Shapes and meaning
------------------
Input ECG records are prepared as [12,5000]; output ECGCLIP vectors are
[N,256] with row i associated with metadata row i.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from ecg_memory_bank.protocol import external_output_dir, require_input_file  # noqa: E402


SCRIPTS = {
    "heedb": "extract_heedb_embeddings.py",
    "mimic": "extract_mimic_embeddings.py",
    "code15": "extract_code15_embeddings.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SCRIPTS))
    parser.add_argument("--config", type=Path, required=True, help="External ECGCLIP model config")
    parser.add_argument("--medcpt-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Dataset index CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mimic-raw-data-dir", type=Path)
    parser.add_argument("--nproc-per-node", type=int, default=6)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = external_output_dir(args.output_dir)
    if not args.dry_run:
        require_input_file(args.config, "ECGCLIP config")
        if not args.medcpt_model.is_dir():
            raise FileNotFoundError(args.medcpt_model)
        require_input_file(args.input, "dataset index")
        if args.dataset == "mimic" and (not args.mimic_raw_data_dir or not args.mimic_raw_data_dir.is_dir()):
            raise ValueError("MIMIC needs --mimic-raw-data-dir")

    script = ROOT / "src" / "ecg_memory_bank" / "extraction" / SCRIPTS[args.dataset]
    command = [sys.executable]
    if args.dataset == "heedb":
        if args.nproc_per_node <= 0:
            raise ValueError("--nproc-per-node must be positive")
        command += ["-m", "torch.distributed.run", f"--nproc_per_node={args.nproc_per_node}"]
    command += [str(script), "--config_path", str(args.config), "--output_dir", str(output)]
    command += ["--main_csv_path", str(args.input)]
    if args.dataset == "mimic":
        command += ["--mimic_raw_data_dir", str(args.mimic_raw_data_dir)]
    if args.gpu_id is not None and args.dataset == "code15":
        command += ["--gpu_id", str(args.gpu_id)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ECG_MEDCPT_MODEL"] = str(args.medcpt_model)
    print("RUN", json.dumps(command, ensure_ascii=False))
    if not args.dry_run:
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
