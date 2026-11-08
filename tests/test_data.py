"""Checks for biological numbering, stop handling, and dataset label selection."""

import csv
import tempfile
import unittest
from pathlib import Path

from creilov.data import CREILOV_WT, FIELDS, load_dataset, reconstruct_sequence


class MutationTests(unittest.TestCase):
    def test_wild_type_and_numbering(self) -> None:
        self.assertEqual(len(CREILOV_WT), 119)
        self.assertEqual(reconstruct_sequence("wt"), (CREILOV_WT, 0, False))
        sequence, count, is_stop = reconstruct_sequence("p.Thr7Ser")
        self.assertEqual(sequence, CREILOV_WT[:6] + "S" + CREILOV_WT[7:])
        self.assertEqual((count, is_stop), (1, False))

    def test_combination(self) -> None:
        sequence, count, is_stop = reconstruct_sequence("p.Arg5Asp, p.Thr7Ser")
        self.assertEqual(sequence[4], "D")
        self.assertEqual(sequence[6], "S")
        self.assertEqual((count, is_stop), (2, False))

    def test_stop_excludes_stop_residue(self) -> None:
        self.assertEqual(
            reconstruct_sequence("p.Ala119Ter"), (CREILOV_WT[:118], 1, True)
        )

    def test_bad_reference_or_duplicate_position_rejected(self) -> None:
        for mutation in ("p.Ala7Ser", "p.Thr7Ser, p.Thr7His", "p.Ala120Ser"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reconstruct_sequence(mutation)


class DatasetTests(unittest.TestCase):
    def test_s2_labels_and_batch_validation(self) -> None:
        import torch
        from proteingen.data import ProteinDataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assay.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for mutation, library, fitness in (
                    ("wt", "S2", 4.02),
                    ("p.Thr7Ser", "S2", 4.14),
                ):
                    sequence, count, is_stop = reconstruct_sequence(mutation)
                    writer.writerow(
                        {
                            "sequence": sequence,
                            "mutations": mutation,
                            "n_mutations": count,
                            "library": library,
                            "is_stop": int(is_stop),
                            "fitness": fitness,
                        }
                    )
            dataset = load_dataset(path)
            self.assertIsInstance(dataset, ProteinDataset)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.labels.dtype, torch.float32)
            self.assertEqual(dataset[1]["sequence"][6], "S")
            self.assertAlmostEqual(dataset[1]["labels"].item(), 4.14, places=5)
            self.assertEqual(dataset.observations, {})
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("S2", "S1"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "S2-only"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
