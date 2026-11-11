"""Check WT alignment, provenance, and isolation of the S2 benchmark data."""

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from creilov.data import CREILOV_WT, FIELDS, reconstruct_sequence
from creilov.s1 import S1_FILENAME, prepare_s1_data


class S1Tests(unittest.TestCase):
    def test_alignment_preserves_raw_values_and_s2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "raw" / S1_FILENAME).write_bytes(b"test workbook")
            s2_path = root / "creilov.csv"
            shared, _, _ = reconstruct_sequence("p.Thr7Ser")
            with s2_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(dict(sequence=CREILOV_WT, library="S2", fitness="4.02"))
                writer.writerow(dict(sequence=shared, library="S2", fitness="4.12"))
            before = hashlib.sha256(s2_path.read_bytes()).hexdigest()
            header = [
                "",
                "rep1",
                "rep2",
                "rep3",
                "mean",
                "log_rep1",
                "log_rep2",
                "log_rep3",
                "log_mean",
            ]
            source_rows = [header] + [
                [mutation, "100", "100", "100", "100", "2", "2", "2", label]
                for mutation, label in (
                    ("wt", "4.17"),
                    ("p.Thr7Ser", "4.30"),
                    ("p.Arg5Asp", "3.50"),
                    ("p.Ala119Ter", "1.00"),
                )
            ]
            with patch("creilov.s1._assay_rows", return_value=iter(source_rows)):
                output = prepare_s1_data(root)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(Decimal(rows[0]["fitness"]), Decimal("4.02"))
            self.assertEqual(Decimal(rows[1]["fitness"]), Decimal("4.15"))
            self.assertEqual(rows[1]["fitness_raw"], "4.30")
            self.assertEqual(rows[1]["rep1"], "100")
            self.assertEqual(rows[1]["source_row"], "3")
            self.assertEqual([r["overlaps_s2"] for r in rows], ["1", "1", "0"])
            self.assertTrue(
                all(r["library"] == "S1" and len(r["sequence"]) == 119 for r in rows)
            )
            self.assertEqual(hashlib.sha256(s2_path.read_bytes()).hexdigest(), before)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["excluded_stop_records"], 1)
            self.assertEqual(metadata["new_sequences_relative_to_s2"], 1)
            self.assertEqual(metadata["sha256"]["creilov.csv"], before)
            self.assertFalse(metadata["benchmark_uses_s1"])


if __name__ == "__main__":
    unittest.main()
