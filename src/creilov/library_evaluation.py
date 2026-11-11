"""Score complete libraries using the existing ProteinGen predictor interface."""

import csv
import hashlib
import json
from pathlib import Path

from Bio import SeqIO
import numpy as np
from proteingen.data import ProteinDataset
from scipy.stats import pearsonr, spearmanr

from .config import Config, serialize_config
from .data import CREILOV_WT
from .predictors import load_checkpoint
from .training import predict, seed_all

DIVERSITY_SOURCE = (
    "https://github.com/junhaobearxiong/proteinguide/blob/main/pdg/fitness/metrics.py"
)


def read_library(path: Path) -> tuple[list[str], list[str]]:
    """Keep duplicate draws, but reject incomplete or malformed proteins."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        sequences = [row["sequence"] for row in rows]
        ids = [row.get("id") or f"sample_{i+1:06d}" for i, row in enumerate(rows)]
    elif path.suffix.lower() in {".fa", ".fasta", ".faa"}:
        with path.open(encoding="utf-8") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
        sequences, ids = [str(r.seq) for r in records], [r.id for r in records]
    else:
        raise ValueError("library_path must be a FASTA or CSV with a sequence column")
    if not sequences:
        raise ValueError("Library is empty")
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Library sample IDs must be unique (duplicate sequences are allowed)"
        )
    for identifier, sequence in zip(ids, sequences, strict=True):
        if len(sequence) != len(CREILOV_WT) or set(sequence) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            raise ValueError(
                f"{identifier}: expected 119 canonical, unmasked amino acids"
            )
    return ids, sequences


def mean_pairwise_hamming(sequences: list[str]) -> float:
    """ProteinGuide metric: unnormalized distance between unique pairs.

    Count matches per column to compute the exact pair mean in O(N*L),
    avoiding an N-by-N distance matrix or pair subsampling.
    """
    unique = list(dict.fromkeys(sequences))
    n = len(unique)
    if n < 2:
        return 0.0
    pairs = n * (n - 1) // 2
    mismatches = 0
    for column in zip(*unique, strict=True):
        _, counts = np.unique(column, return_counts=True)
        mismatches += pairs - int((counts * (counts - 1) // 2).sum())
    return mismatches / pairs


def library_statistics(sequences, oracle, predictor) -> tuple[dict, np.ndarray]:
    oracle, predictor = np.asarray(oracle), np.asarray(predictor)
    if oracle.shape != (len(sequences),) or predictor.shape != oracle.shape:
        raise ValueError("Expected one oracle and predictor score per sequence")
    if not np.isfinite(oracle).all() or not np.isfinite(predictor).all():
        raise ValueError("Non-finite library scores")
    mutations = np.array(
        [sum(a != b for a, b in zip(s, CREILOV_WT, strict=True)) for s in sequences]
    )
    valid_correlation = (
        len(sequences) > 1 and np.ptp(oracle) > 0 and np.ptp(predictor) > 0
    )
    stats = {
        "num_sequences": len(sequences),
        "num_unique": len(set(sequences)),
        "oracle_score_mean": float(oracle.mean()),
        "oracle_score_variance": float(oracle.var()),
        "predictor_oracle_pearson": (
            float(pearsonr(predictor, oracle).statistic) if valid_correlation else None
        ),
        "predictor_oracle_spearman": (
            float(spearmanr(predictor, oracle).statistic) if valid_correlation else None
        ),
        "diversity_mean_pairwise_hamming_unique": mean_pairwise_hamming(sequences),
        "mutations_mean": float(mutations.mean()),
        "mutations_variance": float(mutations.var()),
        "definitions": {
            "scores": "ensemble mean in log10 fluorescence units; complete unmasked sequences",
            "variance": "population variance across library draws (ddof=0), not ensemble uncertainty",
            "correlation": "clean predictor versus oracle on the generated library; null if undefined",
            "diversity": "mean Hamming distance over unordered pairs of unique sequences; zero if fewer than two",
            "diversity_source": DIVERSITY_SOURCE,
            "mutations": "Hamming distance from the complete 119-residue wild type",
            "duplicates": "kept for score, correlation and mutation statistics; removed only for diversity",
        },
    }
    return stats, mutations


def evaluate_library(cfg: Config) -> dict:
    """Load saved ensembles; run clean inference only, without generation/training."""
    if cfg.stage != "evaluate":
        raise ValueError("evaluate_library requires an evaluate config")
    directory = cfg.output_dir
    # Evaluation has no run folder of its own: its scores belong with the
    # library, so they are written into the run that generated it. The config
    # is saved under a separate name to preserve that run's own config.
    if directory.resolve() != cfg.library_path.parent.resolve():
        raise ValueError(
            f"Evaluation writes into the library's own run folder "
            f"({cfg.library_path.parent}), not {directory}"
        )
    ids, sequences = read_library(cfg.library_path)
    # Check inputs before loading models or writing outputs.
    for path in (cfg.oracle_checkpoint, cfg.predictor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation checkpoint: {path}")
    seed_all(cfg.seed)
    dataset = ProteinDataset(sequences)
    scores = {}
    provenance = {
        "library_sha256": hashlib.sha256(cfg.library_path.read_bytes()).hexdigest()
    }
    for role, path in (
        ("oracle", cfg.oracle_checkpoint),
        ("predictor", cfg.predictor_checkpoint),
    ):
        model = load_checkpoint(path, cfg.device)
        metadata = model.checkpoint_metadata
        if metadata.get("role") != role or metadata.get("noisy") != (
            role == "predictor"
        ):
            raise ValueError(
                f"Expected a {'noisy' if role == 'predictor' else 'clean'} {role} checkpoint: {path}"
            )
        scores[role] = predict(
            model, dataset, cfg.scoring_batch_size, mask_fraction=0.0
        )[:, 0]
        provenance[f"{role}_checkpoint_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        del model
    stats, mutations = library_statistics(
        sequences, scores["oracle"], scores["predictor"]
    )
    stats["provenance"] = provenance
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluate_config.json").write_text(
        serialize_config(cfg), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (directory / "per_sequence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "sequence", "oracle_score", "predictor_score", "n_mutations"]
        )
        writer.writerows(
            zip(
                ids,
                sequences,
                scores["oracle"],
                scores["predictor"],
                mutations,
                strict=True,
            )
        )
    # Figures are rendered locally by creilov.reports, titled by guidance run.
    print(json.dumps(stats, indent=2, allow_nan=False))
    return stats
