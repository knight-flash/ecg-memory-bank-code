from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class MimicSplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_mimic18_split_indices(
    sample_count: int,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> MimicSplitIndices:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = 1.0 - train_ratio - val_ratio

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")
    if test_ratio <= 0.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, got train={train_ratio}, val={val_ratio}."
        )

    indices = np.arange(int(sample_count), dtype=np.int64)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=int(seed),
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=int(seed),
    )
    return MimicSplitIndices(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )
