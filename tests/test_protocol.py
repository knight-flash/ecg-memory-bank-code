from __future__ import annotations

import sys
import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecg_memory_bank.protocol import (  # noqa: E402
    HEEDB_MEMORY_FRACTIONS,
    assert_disjoint,
    assert_withdrawn_absent,
    external_output_dir,
    paired_bootstrap_difference,
    unweighted_vote,
    unwrap_ecgclip_checkpoint,
    validate_embedding_alignment,
    validate_broad_addition_protocol,
    validate_nested_memories,
    validate_prototype_anchors,
)


class PaperProtocolTests(unittest.TestCase):
    def test_nested_case_addition_rejects_replaced_members(self) -> None:
        states = {fraction: range(index + 1) for index, fraction in enumerate(HEEDB_MEMORY_FRACTIONS)}
        validate_nested_memories(states)
        states[HEEDB_MEMORY_FRACTIONS[2]] = {3, 4, 5}
        with self.assertRaises(ValueError):
            validate_nested_memories(states)

    def test_direct_broad_addition_cannot_select_historical_protocol(self) -> None:
        base = dict(
            gallery_fractions=HEEDB_MEMORY_FRACTIONS,
            top_k=10,
            seed=42,
            sampling_mode="anchored_nested",
            anchor_fractions=(0.05, 0.1, 0.3, 0.5, 1.0),
            sampling_order_path="",
            max_queries=0,
        )
        validate_broad_addition_protocol(**base)
        validate_broad_addition_protocol(**base, split_by="subject", val_ratio=0.2)
        for change in (
            {"sampling_mode": "fixed_seed_fraction_sample"},
            {"gallery_fractions": (0.001, 1.0)},
            {"top_k": 5},
            {"seed": 7},
            {"max_queries": 100},
            {"sampling_order_path": "other.npy"},
        ):
            with self.assertRaises(ValueError):
                validate_broad_addition_protocol(**{**base, **change})
        with self.assertRaises(ValueError):
            validate_broad_addition_protocol(**base, split_by="row", val_ratio=0.2)

    def test_alignment_and_subject_isolation(self) -> None:
        validate_embedding_alignment(["a", "b"], 2)
        with self.assertRaises(ValueError):
            validate_embedding_alignment(["a", "b"], 1)
        with self.assertRaises(ValueError):
            validate_embedding_alignment(["a", "a"], 2)
        assert_disjoint({"p1"}, {"p2"}, unit="patient")
        with self.assertRaises(ValueError):
            assert_disjoint({"p1"}, {"p1"}, unit="patient")

    def test_withdrawal_retrieval_and_voting(self) -> None:
        assert_withdrawn_absent(["c2", "c3"], ["c1"])
        with self.assertRaises(ValueError):
            assert_withdrawn_absent(["c1", "c3"], ["c1"])
        self.assertEqual(unweighted_vote([{"A", "B"}, {"A"}], ["A", "B"]), {"A": 1.0, "B": 0.5})

    def test_case_link_and_paired_resampling(self) -> None:
        validate_prototype_anchors(["c1", "c2"], ["c1", "c2", "c3"])
        with self.assertRaises(ValueError):
            validate_prototype_anchors(["c4"], ["c1"])
        first = paired_bootstrap_difference([1, 0, 1], [0, 0, 0], iterations=1000)
        self.assertEqual(first, paired_bootstrap_difference([1, 0, 1], [0, 0, 0], iterations=1000))
        self.assertGreater(first[0], 0)

    def test_outputs_stay_outside_archive(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            external_output_dir(root / "outputs")

    def test_original_ecgclip_checkpoint_wrapper(self) -> None:
        state = unwrap_ecgclip_checkpoint({"model_state_dict": {"module.ecg_encoder.weight": 3}})
        self.assertEqual(state, {"ecg_encoder.weight": 3})
        with self.assertRaises(ValueError):
            unwrap_ecgclip_checkpoint({"model_state_dict": {"module.x": 1, "x": 2}})

    def test_support_tiers_are_recomputed_without_output_snapshot(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            metadata = work / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["labels_correct"])
                writer.writerow(["left axis deviation"])
                writer.writerow(["left axis deviation"])
            output = work / "support_tiers.csv"
            command = [
                sys.executable,
                str(root / "src" / "ecg_memory_bank" / "retrieval" / "build_support_tiers.py"),
                "--metadata-csv", str(metadata),
                "--mapping-csv", str(root / "configs" / "label_category_counts_9class.csv"),
                "--output-csv", str(output),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root / "src")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(command, check=True, env=env)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 120)
            self.assertEqual(sorted(set(row["support_tier"] for row in rows)), ["Head", "Low-mid", "Mid", "Tail"])
            self.assertEqual(sum(row["mapped_label"] == "left axis deviation" and row["count"] == "2" for row in rows), 1)

    def test_reader_ci_resamples_cases_and_keeps_reader_pairs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with (work / "case_level_paired_diffs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "metric", "diff"])
                for case, score in (("c1", 1), ("c2", -1)):
                    writer.writerows([[case, "accuracy", score]] * 4)
            with (work / "label_level_paired_diffs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "matched_category", "diff"])
                writer.writerow(["c1", "Rhythm", 1])
                writer.writerow(["c2", "Rhythm", -1])
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run([
                sys.executable,
                str(root / "src" / "ecg_memory_bank" / "analysis" / "reader_study_ci.py"),
                "--input-dir", str(work), "--output-dir", str(work),
            ], check=True, env=env, capture_output=True)
            import json
            summary = json.loads((work / "case_clustered_reader_ci.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["case_level"]["accuracy"]["pairs"], 8)
            self.assertEqual(summary["case_level"]["accuracy"]["cases"], 2)


if __name__ == "__main__":
    unittest.main()
