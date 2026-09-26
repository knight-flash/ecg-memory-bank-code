#!/usr/bin/env python3
"""Query-mean bootstrap intervals.

Purpose
-------
Generate percentile confidence intervals for query-averaged retrieval coverage
measures.

Method
------
Read saved per-query arrays and resample query rows with a fixed RNG,
optionally processing in chunks.

Inputs
------
Existing retrieval CI arrays under an external result directory.

Outputs
-------
External point estimates and 95% interval summaries for requested
metrics/scenarios.

Paper parameters
----------------
Defaults to 1000 query resamples, seed=42; chunk size controls memory only.

Shapes and meaning
------------------
Input metric arrays have one scalar per query [Q]; bootstrap samples use query
indices, not memory cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "union_recall_l3_at_1",
    "union_recall_l3_at_3",
    "union_recall_l3_at_5",
    "union_recall_l3_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=20)
    return parser.parse_args()


def resolve_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    candidate = manifest_path.parent / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(raw)


def bootstrap_metrics(
    arrays: dict[str, np.ndarray], n_bootstrap: int, seed: int, chunk_size: int
) -> dict[str, np.ndarray]:
    first = next(iter(arrays.values()))
    sample_count = len(first)
    if any(len(values) != sample_count for values in arrays.values()):
        raise ValueError("query metric arrays do not share the same query count")
    outputs = {name: np.empty(n_bootstrap, dtype=np.float64) for name in arrays}
    rng = np.random.default_rng(seed)
    offset = 0
    while offset < n_bootstrap:
        count = min(chunk_size, n_bootstrap - offset)
        indices = rng.integers(0, sample_count, size=(count, sample_count), endpoint=False, dtype=np.int32)
        for name, values in arrays.items():
            outputs[name][offset : offset + count] = np.nanmean(values[indices], axis=1)
        offset += count
        if offset % 100 == 0 or offset == n_bootstrap:
            print(f"[Bootstrap] {offset}/{n_bootstrap}", flush=True)
    return outputs


def main() -> None:
    args = parse_args()
    metrics = tuple(args.metric or DEFAULT_METRICS)
    manifests = sorted(args.input_root.glob("*/ci_artifacts/ci_manifest.json"))
    allowed = set(args.scenario)
    if allowed:
        manifests = [path for path in manifests if path.parent.parent.name in allowed]
    if not manifests:
        raise FileNotFoundError("no matching CI manifests")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    sample_paths = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario = str(manifest["scenario_name"])
        query_path = resolve_path(str(manifest["query_metrics_npz"]), manifest_path)
        payload = np.load(query_path, allow_pickle=False)
        arrays = {
            name: payload[name].astype(np.float32, copy=False)
            for name in metrics
            if name in payload
        }
        boot = bootstrap_metrics(arrays, args.bootstrap, args.seed, args.chunk_size)
        for name, values in arrays.items():
            lower, upper = np.quantile(boot[name], [0.025, 0.975])
            records.append(
                {
                    "scenario": scenario,
                    "metric": name,
                    "point_estimate": float(np.nanmean(values)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "query_count": int(len(values)),
                    "bootstrap_iterations": int(args.bootstrap),
                    "seed": int(args.seed),
                }
            )
        sample_path = args.output_dir / f"{scenario}_query_mean_bootstrap_samples.npz"
        np.savez_compressed(sample_path, **boot)
        sample_paths[scenario] = str(sample_path)

    csv_path = args.output_dir / "query_mean_ci_summary.csv"
    json_path = args.output_dir / "query_mean_ci_summary.json"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "method": "query-level percentile bootstrap of mean metrics",
                "bootstrap_iterations": int(args.bootstrap),
                "seed": int(args.seed),
                "results": records,
                "bootstrap_samples": sample_paths,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
