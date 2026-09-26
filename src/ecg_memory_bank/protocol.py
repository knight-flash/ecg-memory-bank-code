"""Final-paper protocol invariants.

Purpose
-------
Centralize the rules that distinguish the released experiments from historical
exploratory runs.

Method
------
Validate external output paths, strict row-to-ID alignment, nested memory
membership, subject disjointness, withdrawal zero persistence, prototype
anchors, and paired bootstrap inputs.

Inputs
------
Case IDs, fraction-to-membership maps, labels, score arrays, and output paths
supplied by experiment entries.

Outputs
-------
Validation results, unweighted vote dictionaries, or explicit errors before an
expensive run.

Paper parameters
----------------
Broad addition fixes eight fractions, five anchors, K=10 and seed=42; Chagas
stages and withdrawal fractions are recorded as constants.

Shapes and meaning
------------------
Embeddings align [N,D] with N metadata rows; neighbors are [Q,K]; each
unweighted label vote is a fraction in [0,1].
"""

from __future__ import annotations

from pathlib import Path
import random
import math
from typing import Iterable, Mapping, Sequence


HEEDB_MEMORY_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.10, 0.30, 0.50, 1.0)
MIMIC_MEMORY_FRACTIONS = HEEDB_MEMORY_FRACTIONS
HEEDB_FIXED_L3_LABELS = 114
MIMIC_FIXED_L3_LABELS = 83
CHAGAS_ADDITION_STAGES = (0, 50, 100, 300, 800, 2000, 5000)
PATIENT_WITHDRAWAL_FRACTIONS = (0.15, 0.30, 0.60, 0.90)
HISTORICAL_YEAR_THRESHOLDS = (2010, 2015, 2018)
ANCHORED_MEMORY_FRACTIONS = (0.05, 0.10, 0.30, 0.50, 1.0)


def archive_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_output_dir(value: str | Path) -> Path:
    """Reject paths inside the source archive, including nonexistent descendants."""
    path = Path(value).expanduser().resolve()
    root = archive_root()
    if path == root or root in path.parents:
        raise ValueError(f"Output directory must be outside the code archive: {path}")
    return path


def validate_broad_addition_protocol(
    *,
    gallery_fractions: Sequence[float],
    top_k: int,
    seed: int,
    sampling_mode: str,
    anchor_fractions: Sequence[float],
    sampling_order_path: str,
    max_queries: int,
    split_by: str | None = None,
    val_ratio: float | None = None,
) -> None:
    """Reject any direct-script invocation that departs from the final paper."""
    if tuple(gallery_fractions) != HEEDB_MEMORY_FRACTIONS:
        raise ValueError("The paper requires all eight fixed memory fractions")
    if top_k != 10 or seed != 42 or sampling_mode != "anchored_nested":
        raise ValueError("The paper requires top-K=10, seed=42, and anchored_nested sampling")
    if tuple(anchor_fractions) != ANCHORED_MEMORY_FRACTIONS:
        raise ValueError("The paper's five anchor fractions are fixed")
    if sampling_order_path or max_queries != 0:
        raise ValueError("Custom order files and query limits are not part of the paper protocol")
    if split_by is not None and (split_by != "subject" or val_ratio != 0.2):
        raise ValueError("The MIMIC paper replication requires a subject-level 80/20 split")


def require_input_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def validate_record_ids(ids: Sequence[object], *, name: str) -> None:
    values = [str(value).strip() for value in ids]
    if any(not value or value.lower() in {"nan", "none"} for value in values):
        raise ValueError(f"{name} contains an empty record ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate record IDs")


def validate_embedding_alignment(ids: Sequence[object], vector_rows: int) -> None:
    validate_record_ids(ids, name="embedding metadata")
    if len(ids) != vector_rows:
        raise ValueError(f"metadata rows ({len(ids)}) != embedding rows ({vector_rows})")


def validate_nested_memories(states: Mapping[float, Iterable[object]]) -> None:
    if tuple(sorted(states)) != HEEDB_MEMORY_FRACTIONS:
        raise ValueError("Broad case addition requires exactly the eight paper fractions")
    previous: set[object] = set()
    for fraction in HEEDB_MEMORY_FRACTIONS:
        current = set(states[fraction])
        if not current:
            raise ValueError(f"Memory at fraction {fraction} is empty")
        if not previous.issubset(current):
            raise ValueError(f"Memory at fraction {fraction} does not retain prior cases")
        previous = current


def assert_disjoint(query_ids: Iterable[object], memory_ids: Iterable[object], *, unit: str) -> None:
    overlap = set(query_ids).intersection(memory_ids)
    if overlap:
        raise ValueError(f"Query and memory {unit} sets overlap ({len(overlap)} units)")


def assert_withdrawn_absent(post_retrieval_ids: Iterable[object], withdrawn_ids: Iterable[object]) -> None:
    remaining = set(post_retrieval_ids).intersection(withdrawn_ids)
    if remaining:
        raise ValueError(f"Withdrawn cases remain in post-withdrawal retrieval ({len(remaining)})")


def unweighted_vote(neighbors: Sequence[Iterable[str]], labels: Sequence[str]) -> dict[str, float]:
    if not neighbors:
        raise ValueError("At least one neighbor is required")
    sets = [set(row) for row in neighbors]
    return {label: sum(label in row for row in sets) / len(sets) for label in labels}


def validate_prototype_anchors(anchor_ids: Iterable[object], training_ids: Iterable[object]) -> None:
    anchors = list(anchor_ids)
    training = set(training_ids)
    if not anchors or any(anchor not in training for anchor in anchors):
        raise ValueError("Every projected prototype must link to a training case")


def paired_bootstrap_difference(
    rag: Sequence[float], no_rag: Sequence[float], *, iterations: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Case-paired percentile interval for a mean difference."""
    if len(rag) != len(no_rag) or not rag or iterations <= 0:
        raise ValueError("Paired nonempty cohorts and positive iterations are required")
    differences = [float(a) - float(b) for a, b in zip(rag, no_rag)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("Non-finite paired outcome")
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    lower = samples[int(0.025 * (iterations - 1))]
    upper = samples[int(0.975 * (iterations - 1))]
    return sum(differences) / n, lower, upper


def unwrap_ecgclip_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Accept the original extractor's model_state_dict / DDP checkpoint format."""
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("ECGCLIP checkpoint must contain a state dictionary")
    normalized: dict[str, object] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError("ECGCLIP state keys must be strings")
        clean = key.removeprefix("module.")
        if clean in normalized:
            raise ValueError(f"Duplicate state key after DDP prefix removal: {clean}")
        normalized[clean] = value
    return normalized
