"""Reproducible extrapolation splits, with explicit disjoint validation sets."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from .config import SplitConfig
from .data import CREILOV_WT


@dataclass
class Split:
    train: list[int]
    validation: list[int]
    test: list[int]

    def validate(self, size: int) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(not group for group in groups):
            raise ValueError(
                "A split partition is empty; change cutoffs or held-out sites"
            )
        if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Split leakage: overlapping indices")
        if set.union(*groups) != set(range(size)):
            raise ValueError("Split must partition all rows")


def make_splits(
    sequences: list[str], labels: list[float], cfg: SplitConfig, seed: int
) -> dict[str, Split]:
    if len(sequences) != len(labels) or len(set(sequences)) != len(sequences):
        raise ValueError("Expected unique sequences with aligned labels")
    if any(len(s) != len(CREILOV_WT) for s in sequences):
        raise ValueError("Expected full-length CreiLOV sequences")
    n = len(sequences)
    order = np.random.default_rng(seed).permutation(n).tolist()
    a, b = int(n * cfg.train_fraction), int(
        n * (cfg.train_fraction + cfg.validation_fraction)
    )
    result = {"random": Split(order[:a], order[a:b], order[b:])}
    for name in ("distance", "activity", "position"):
        groups: list[list[int]] = [[], [], []]
        low, high = np.quantile(
            labels, [cfg.activity_train_quantile, cfg.activity_validation_quantile]
        )
        observed = {
            i + 1
            for s in sequences
            for i, (a, b) in enumerate(zip(s, CREILOV_WT))
            if a != b
        }
        if (
            name == "position"
            and not set(cfg.validation_positions + cfg.test_positions) <= observed
        ):
            raise ValueError("Position holdouts must vary in the dataset")
        for i, (sequence, label) in enumerate(zip(sequences, labels, strict=True)):
            changed = {
                p + 1 for p, (a, b) in enumerate(zip(sequence, CREILOV_WT)) if a != b
            }
            if name == "distance":
                part = (
                    0
                    if len(changed) <= cfg.distance_train_max
                    else 1 if len(changed) <= cfg.distance_validation_max else 2
                )
            elif name == "activity":
                part = 0 if label <= low else 1 if label <= high else 2
            else:
                part = (
                    2
                    if changed & set(cfg.test_positions)
                    else 1 if changed & set(cfg.validation_positions) else 0
                )
            groups[part].append(i)
        result[name] = Split(*groups)
    for split in result.values():
        split.validate(n)
    return result


def save_splits(path: Path, splits: dict[str, Split], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "csv_sha256": fingerprint,
                "splits": {k: asdict(v) for k, v in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
