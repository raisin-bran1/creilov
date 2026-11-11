"""Check source precedence, combined sampling, and S2 benchmark isolation."""

from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from creilov.combined import prepare_combined_data
from creilov.config import TrainingConfig, load_config
from creilov.data import FIELDS, load_dataset, reconstruct_sequence
from creilov.s1 import S1_FIELDS
from creilov.splits import Split
from creilov.training import select_training_indices
from creilov.workflow import prepare

ROOT = Path(__file__).parents[1]


class CombinedTests(unittest.TestCase):
    def test_dedup_alignment_and_default_benchmark_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, fields, library, examples in (
                ("creilov.csv", FIELDS, "S2", [("wt", "4.02"), ("p.Thr7Ser", "4.12")]),
                (
                    "creilov_s1_aligned.csv",
                    S1_FIELDS,
                    "S1",
                    [("wt", "4.02"), ("p.Thr7Ser", "4.15"), ("p.Arg5Asp", "3.35")],
                ),
            ):
                with (root / name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for mutation, label in examples:
                        sequence, count, _ = reconstruct_sequence(mutation)
                        row = dict(
                            sequence=sequence,
                            mutations=mutation,
                            n_mutations=count,
                            library=library,
                            is_stop=0,
                            fitness=label,
                        )
                        if library == "S1":
                            from decimal import Decimal

                            row.update(
                                fitness_raw=str(Decimal(label) + Decimal("0.15")),
                                log10_alignment_offset="-0.15",
                                overlaps_s2=int(mutation != "p.Arg5Asp"),
                            )
                        writer.writerow(row)
            paths = [root / "creilov.csv", root / "creilov_s1_aligned.csv"]
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            output = prepare_combined_data(root)
            self.assertEqual(
                before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(Counter(r["library"] for r in rows), {"S2": 2, "S1": 1})
            self.assertEqual(
                rows[1]["fitness"], "4.12"
            )  # S2 wins even when aligned S1 differs.
            self.assertEqual(len({r["sequence"] for r in rows}), 3)
            self.assertEqual(
                len(load_dataset(output, allowed_libraries=("S1", "S2"))), 3
            )
            with self.assertRaisesRegex(ValueError, "S2-only"):
                load_dataset(output)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["s1_overlaps_removed"], 2)
            # The real benchmark entrypoint rejects S1 even if pointed at this pool.
            cfg = replace(
                load_config(ROOT / "configs/benchmark.toml"),
                csv_path=output,
                output_dir=root / "benchmark",
            )
            with self.assertRaisesRegex(ValueError, "S2-only"):
                prepare(cfg)

    def test_uniform_selection_uses_combined_pool_without_library_quota(self):
        split = Split(train=list(range(100)), validation=[100], test=[101])
        expected = list(split.train)
        random.Random(42).shuffle(expected)
        selected = select_training_indices(split, TrainingConfig(max_train=20), 42)
        self.assertEqual(selected, expected[:20])
        self.assertEqual(len(set(selected)), 20)
        self.assertTrue(set(selected).isdisjoint(split.validation + split.test))


if __name__ == "__main__":
    unittest.main()
