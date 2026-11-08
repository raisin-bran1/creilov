"""Convert the Chen CreiLOV assays and load them into ProteinGen.

Run from the project root: ``uv run python -m creilov.data data``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET
from zipfile import ZipFile

if TYPE_CHECKING:
    from proteingen.data import ProteinDataset

# ProteinGuide pdg/fitness/constants.py, CREILOV_WT:
# https://github.com/junhaobearxiong/proteinguide/blob/main/pdg/fitness/constants.py
# Includes the initiating methionine. Assay numbering runs from 2 to 119.
CREILOV_WT = (
    "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKK"
    "GEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
)
AA3_TO_AA1 = dict(
    zip(
        "Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)
ASSAYS = (("S2", "sb2c00662_si_002.xlsx", "Rep1", "Rep1_log", "mean_log"),)
FIELDS = (
    "sequence",
    "mutations",
    "n_mutations",
    "library",
    "is_stop",
    "rep1",
    "rep2",
    "rep3",
    "mean",
    "log_rep1",
    "log_rep2",
    "log_rep3",
    "fitness",
    "source_file",
    "source_row",
)
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def reconstruct_sequence(mutations: str) -> tuple[str, int, bool]:
    """Apply 1-based assay substitutions; Ter terminates before that residue.

    Validate the original residue against the reference, including for stops.
    ``n_mutations`` counts listed events, not Hamming distance for truncations.
    """
    if mutations == "wt":
        return CREILOV_WT, 0, False
    sequence = list(CREILOV_WT)
    positions: set[int] = set()
    stop = len(sequence)
    for mutation in mutations.split(", "):
        match = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", mutation)
        if match is None:
            raise ValueError(f"Unsupported mutation: {mutation!r}")
        original, position_text, replacement = match.groups()
        position = int(position_text) - 1
        if not 0 <= position < len(sequence) or position in positions:
            raise ValueError(f"Invalid or repeated position in {mutations!r}")
        if original not in AA3_TO_AA1 or CREILOV_WT[position] != AA3_TO_AA1[original]:
            raise ValueError(f"Reference residue mismatch: {mutation}")
        if replacement == original:
            raise ValueError(f"Substitution does not change residue: {mutation}")
        positions.add(position)
        if replacement == "Ter":
            stop = min(stop, position)
        elif replacement in AA3_TO_AA1:
            sequence[position] = AA3_TO_AA1[replacement]
        else:
            raise ValueError(f"Unsupported replacement: {mutation}")
    return "".join(sequence[:stop]), len(positions), stop < len(sequence)


def _assay_rows(path: Path) -> Iterator[list[str]]:
    """Read the nine source columns without an optional Excel dependency.

    This reader targets the supplied single-sheet XLSX files, not general Excel
    workbooks. Numeric values are read directly without rounding or recomputing.
    """
    with ZipFile(path) as archive:
        strings_xml = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(node.itertext()) for node in strings_xml.findall(f"{NS}si")]
        with archive.open("xl/worksheets/sheet1.xml") as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{NS}row":
                    continue
                values = [""] * 9
                for cell in row:
                    column = re.sub(r"\d", "", cell.attrib["r"])
                    if len(column) != 1 or not "A" <= column <= "I":
                        raise ValueError(f"Unexpected source column: {column}")
                    value_node = cell.find(f"{NS}v")
                    value = value_node.text if value_node is not None else ""
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    values[ord(column) - ord("A")] = value
                yield values
                row.clear()


def prepare_data(data_dir: Path) -> Path:
    """Write S2 records to creilov.csv; S1 is not read or combined with S2."""
    output = data_dir / "creilov.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for library, filename, rep_header, log_header, fitness_header in ASSAYS:
            rows = _assay_rows(data_dir / "raw" / filename)
            header = next(rows)
            if (
                header[1] != rep_header
                or header[5] != log_header
                or header[8] != fitness_header
            ):
                raise ValueError(f"Unexpected headers in {filename}: {header}")
            seen: set[str] = set()
            for source_row, values in enumerate(rows, start=2):
                mutations = values[0]
                if mutations in seen:
                    raise ValueError(f"Duplicate variant in {library}: {mutations}")
                seen.add(mutations)
                sequence, count, is_stop = reconstruct_sequence(mutations)
                numeric = [float(value) for value in values[1:]]
                if not all(math.isfinite(value) for value in numeric):
                    raise ValueError(f"Non-finite measurement: {filename}:{source_row}")
                if any(value <= 0 for value in numeric[:4]):
                    raise ValueError(
                        f"Nonpositive fluorescence: {filename}:{source_row}"
                    )
                writer.writerow(
                    dict(
                        zip(
                            FIELDS,
                            (
                                sequence,
                                mutations,
                                count,
                                library,
                                int(is_stop),
                                *values[1:8],
                                values[8],
                                filename,
                                source_row,
                            ),
                            strict=True,
                        )
                    )
                )
    temporary.replace(output)
    return output


def load_dataset(csv_path: Path) -> ProteinDataset:
    """Return a standard ProteinDataset with log-mean fluorescence labels.

    Expects the S2-only table produced by prepare_data. Metadata remains in the
    CSV; it is not model conditioning. No normalization, tokenization, masking,
    or splitting is performed.
    """
    import torch
    from proteingen.data import ProteinDataset

    sequences: list[str] = []
    fitness: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["library"] != "S2":
                raise ValueError("Expected an S2-only CSV")
            sequence, _, is_stop = reconstruct_sequence(row["mutations"])
            if sequence != row["sequence"] or int(is_stop) != int(row["is_stop"]):
                raise ValueError(f"Inconsistent sequence in CSV: {row['mutations']}")
            if is_stop or len(sequence) != len(CREILOV_WT):
                raise ValueError("S2 requires full-length sequences without stops")
            label = float(row["fitness"])
            if not math.isfinite(label):
                raise ValueError(f"Non-finite fitness: {row['mutations']}")
            sequences.append(sequence)
            fitness.append(label)
    if not sequences:
        raise ValueError("No records selected")
    labels = torch.tensor(fitness, dtype=torch.float32)  # [N] - log10 mean fluorescence
    return ProteinDataset(sequences=sequences, labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing raw/")
    args = parser.parse_args()
    print(prepare_data(args.data_dir))


if __name__ == "__main__":
    main()
