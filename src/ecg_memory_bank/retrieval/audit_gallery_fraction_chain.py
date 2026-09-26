#!/usr/bin/env python3
"""Audit set inclusion across fixed-seed gallery-fraction samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_memory_bank.retrieval.artifact_utils import build_anchored_nested_gallery_order


DEFAULT_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-anchored", action="store_true")
    parser.add_argument("--anchor-fractions", nargs="+", type=float, default=[0.05, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--order-output", type=Path)
    return parser.parse_args()


def legacy_sample(total: int, fraction: float, seed: int) -> np.ndarray:
    size = max(1, int(round(total * float(fraction))))
    if fraction >= 1.0:
        return np.arange(total, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(total, size=size, replace=False))


def main() -> None:
    args = parse_args()
    if args.build_anchored:
        order, payload = build_anchored_nested_gallery_order(
            total=args.total,
            seed=args.seed,
            anchor_fractions=args.anchor_fractions,
        )
        prefix_rows = []
        previous = None
        for fraction in args.fractions:
            size = max(1, int(round(args.total * float(fraction))))
            current = np.sort(order[:size])
            retained_count = None if previous is None else int(
                np.intersect1d(previous, current, assume_unique=True).size
            )
            prefix_rows.append(
                {
                    "fraction": float(fraction),
                    "sample_size": int(size),
                    "previous_fully_retained": (
                        None if previous is None else retained_count == len(previous)
                    ),
                    "legacy_membership_preserved": (
                        bool(np.array_equal(current, legacy_sample(args.total, fraction, args.seed)))
                        if any(abs(fraction - anchor) < 1e-12 for anchor in args.anchor_fractions)
                        else None
                    ),
                }
            )
            previous = current
        payload["prefix_audit"] = prefix_rows
        if args.order_output:
            args.order_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.order_output, order)
            payload["order_npy"] = str(args.order_output)
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return

    previous = None
    rows = []
    for fraction in args.fractions:
        current = legacy_sample(args.total, fraction, args.seed)
        if previous is None:
            contained = None
            retained_count = None
            retained_fraction = None
        else:
            retained_count = int(np.intersect1d(previous, current, assume_unique=True).size)
            retained_fraction = retained_count / len(previous)
            contained = retained_count == len(previous)
        rows.append(
            {
                "fraction": float(fraction),
                "sample_size": int(len(current)),
                "previous_contained": contained,
                "previous_retained_count": retained_count,
                "previous_retained_fraction": retained_fraction,
            }
        )
        previous = current

    payload = {
        "numpy_version": np.__version__,
        "total": int(args.total),
        "seed": int(args.seed),
        "rows": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
