"""Build a unique S2 + WT-aligned S1 predictor pool, preserving S2 labels.

Run ``uv run python -m creilov.combined data`` after preparing both input CSVs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .data import CREILOV_WT, reconstruct_sequence
from .s1 import S1_FIELDS


def prepare_combined_data(data_dir: Path) -> Path:
    s2_path = data_dir / "creilov.csv"
    s1_path = data_dir / "creilov_s1_aligned.csv"
    with s1_path.open(newline="", encoding="utf-8") as stream:
        s1_rows = list(csv.DictReader(stream))
    s1_wt = [r for r in s1_rows if r["sequence"] == CREILOV_WT]
    if len(s1_wt) != 1:
        raise ValueError("Aligned S1 CSV must contain exactly one WT")
    counts = Counter()
    seen = set()
    s1_seen = set()
    shared = 0
    s2_wt = None
    output = data_dir / "creilov_combined.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=S1_FIELDS)
        writer.writeheader()
        for library, source in (("S2", s2_path), ("S1", s1_rows)):
            if library == "S2":
                with source.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
            else:
                if s2_wt is None:
                    raise ValueError(
                        "S2 CSV must contain WT for alignment verification"
                    )
                offset = s2_wt - Decimal(s1_wt[0]["fitness_raw"])
                rows = source
            for original in rows:
                row = dict(original)
                sequence, count, stop = reconstruct_sequence(row["mutations"])
                fitness = Decimal(row["fitness"])
                if (
                    row["library"] != library
                    or sequence != row["sequence"]
                    or len(sequence) != len(CREILOV_WT)
                    or stop
                    or row["is_stop"] != "0"
                    or not fitness.is_finite()
                ):
                    raise ValueError(f"Invalid {library} record: {row['mutations']}")
                if library == "S2":
                    if sequence in seen:
                        raise ValueError("Duplicate S2 sequence")
                    if sequence == CREILOV_WT:
                        s2_wt = fitness
                    row.update(
                        fitness_raw=row["fitness"],
                        log10_alignment_offset="0",
                        overlaps_s2="1",
                    )
                else:
                    if sequence in s1_seen or count > 1:
                        raise ValueError("Expected unique S1 WT/single substitutions")
                    s1_seen.add(sequence)
                    if (
                        Decimal(row["log10_alignment_offset"]) != offset
                        or Decimal(row["fitness_raw"]) + offset != fitness
                    ):
                        raise ValueError(
                            "S1 labels must be WT-aligned to the current S2 CSV"
                        )
                    if sequence in seen:
                        shared += 1
                        continue  # Always retain the measured S2 label for shared sequences.
                seen.add(sequence)
                counts[library] += 1
                writer.writerow(row)
    temporary.replace(output)
    metadata = {
        "sampling": "Uniform over unique combined rows, without an S1/S2 quota",
        "deduplication": "S2 takes precedence over aligned S1 for shared sequences",
        "n": len(seen),
        "library_counts": dict(counts),
        "s1_overlaps_removed": shared,
        "expected_s1_in_uniform_2000": 2000 * counts["S1"] / len(seen),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (s2_path, s1_path, output)
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    print(prepare_combined_data(args.data_dir))


if __name__ == "__main__":
    main()
