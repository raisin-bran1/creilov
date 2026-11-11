"""Export full-length S1 variants on S2's wild-type fluorescence scale.

Run ``uv run python -m creilov.s1 data``. This never modifies the S2 CSV or
connects S1 to training/benchmark commands.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import median

from .data import CREILOV_WT, FIELDS, _assay_rows, reconstruct_sequence

S1_FILENAME = "sb2c00662_si_001.xlsx"
S1_FIELDS = (*FIELDS, "fitness_raw", "log10_alignment_offset", "overlaps_s2")
METHOD_URL = "https://arxiv.org/html/2505.15093#A2"


def prepare_s1_data(data_dir: Path) -> Path:
    """Align S1 log10 labels by the WT difference, retaining raw measurements.

    Stop variants are omitted to retain the fixed-length model's input domain.
    WT and sequences shared with S2 remain in this standalone export, flagged
    for deduplication before any later use. S2 labels win when merging libraries.
    """
    s2_path = data_dir / "creilov.csv"
    source_path = data_dir / "raw" / S1_FILENAME
    with s2_path.open(newline="", encoding="utf-8") as stream:
        s2_labels = {}
        for row in csv.DictReader(stream):
            if row["library"] != "S2":
                raise ValueError("Alignment reference must be an S2-only CSV")
            label = Decimal(row["fitness"])
            if not label.is_finite() or row["sequence"] in s2_labels:
                raise ValueError("Expected finite labels and unique S2 sequences")
            s2_labels[row["sequence"]] = label
    if CREILOV_WT not in s2_labels:
        raise ValueError("S2 alignment reference requires a wild-type measurement")

    source_rows = list(_assay_rows(source_path))
    expected_header = [
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
    if source_rows[0] != expected_header:
        raise ValueError(f"Unexpected S1 headers: {source_rows[0]}")
    wt_rows = [row for row in source_rows[1:] if row[0] == "wt"]
    if len(wt_rows) != 1:
        raise ValueError("S1 requires exactly one wild-type measurement")
    s1_wt = Decimal(wt_rows[0][8])
    if not s1_wt.is_finite():
        raise ValueError("Non-finite S1 wild-type fitness")
    offset = s2_labels[CREILOV_WT] - s1_wt

    output = data_dir / "creilov_s1_aligned.csv"
    temporary = output.with_suffix(".csv.tmp")
    seen = set()
    stopped = 0
    shared_residuals = []
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=S1_FIELDS)
        writer.writeheader()
        for source_row, values in enumerate(source_rows[1:], start=2):
            sequence, count, is_stop = reconstruct_sequence(values[0])
            numbers = [Decimal(value) for value in values[1:]]
            if not all(value.is_finite() for value in numbers):
                raise ValueError(f"Non-finite S1 measurement at row {source_row}")
            if any(value <= 0 for value in numbers[:4]) or count > 1:
                raise ValueError(f"Invalid S1 measurement at row {source_row}")
            if is_stop:
                stopped += 1
                continue
            if len(sequence) != len(CREILOV_WT) or sequence in seen:
                raise ValueError("Expected unique, full-length S1 sequences")
            seen.add(sequence)
            aligned = numbers[-1] + offset
            overlaps = sequence in s2_labels
            if overlaps:
                shared_residuals.append(float(s2_labels[sequence] - aligned))
            row = dict(
                zip(
                    FIELDS,
                    (
                        sequence,
                        values[0],
                        count,
                        "S1",
                        0,
                        *values[1:8],
                        str(aligned),
                        S1_FILENAME,
                        source_row,
                    ),
                    strict=True,
                )
            )
            row.update(
                fitness_raw=values[8],
                log10_alignment_offset=str(offset),
                overlaps_s2=int(overlaps),
            )
            writer.writerow(row)
    temporary.replace(output)
    metadata = {
        "method": "Add S2 WT minus S1 WT to each S1 log10-of-mean fitness",
        "method_reference": METHOD_URL,
        "s1_wt_fitness_raw": str(s1_wt),
        "s2_wt_fitness": str(s2_labels[CREILOV_WT]),
        "log10_alignment_offset": str(offset),
        "source_records": len(source_rows) - 1,
        "exported_full_length_records": len(seen),
        "excluded_stop_records": stopped,
        "overlap_with_s2": len(shared_residuals),
        "new_sequences_relative_to_s2": len(seen) - len(shared_residuals),
        "raw_measurement_columns": [
            "rep1",
            "rep2",
            "rep3",
            "mean",
            "log_rep1",
            "log_rep2",
            "log_rep3",
            "fitness_raw",
        ],
        "shared_s2_minus_aligned_s1": {
            "n": len(shared_residuals),
            "min": min(shared_residuals),
            "median": median(shared_residuals),
            "max": max(shared_residuals),
            "rmse": (sum(x * x for x in shared_residuals) / len(shared_residuals))
            ** 0.5,
        },
        "sha256": {
            S1_FILENAME: hashlib.sha256(source_path.read_bytes()).hexdigest(),
            s2_path.name: hashlib.sha256(s2_path.read_bytes()).hexdigest(),
            output.name: hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        "benchmark_uses_s1": False,
        "note": "WT alignment does not remove all batch effects. No libraries have been merged.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    print(prepare_s1_data(args.data_dir))


if __name__ == "__main__":
    main()
