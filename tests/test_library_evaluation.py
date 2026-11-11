"""Exact library metrics and clean checkpoint inference, without downloads."""

from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from creilov.config import load_config, ModelConfig
from creilov.data import CREILOV_WT
from creilov.library_evaluation import (
    evaluate_library,
    library_statistics,
    mean_pairwise_hamming,
    read_library,
)
from creilov.predictors import FluorescenceEnsemble, save_checkpoint

ROOT = Path(__file__).parents[1]


class LibraryEvaluationTests(unittest.TestCase):
    def test_metrics_match_explicit_unique_pairs_and_keep_duplicate_draws(self):
        variants = [CREILOV_WT, "C" + CREILOV_WT[1:], "CC" + CREILOV_WT[2:]]
        sequences = variants + [variants[0]]
        expected = np.mean(
            [sum(a != b for a, b in zip(s, t)) for s, t in combinations(variants, 2)]
        )
        stats, mutations = library_statistics(sequences, [1, 2, 3, 1], [2, 4, 6, 2])
        self.assertAlmostEqual(mean_pairwise_hamming(sequences), expected)
        self.assertEqual(stats["num_unique"], 3)
        self.assertEqual(stats["oracle_score_mean"], 1.75)
        self.assertEqual(stats["oracle_score_variance"], 0.6875)
        np.testing.assert_array_equal(mutations, [0, 1, 2, 0])
        self.assertEqual(stats["mutations_variance"], 0.6875)
        self.assertAlmostEqual(stats["predictor_oracle_pearson"], 1.0)
        self.assertAlmostEqual(stats["predictor_oracle_spearman"], 1.0)
        singleton, _ = library_statistics([CREILOV_WT], [1], [2])
        self.assertIsNone(singleton["predictor_oracle_pearson"])
        self.assertEqual(singleton["diversity_mean_pairwise_hamming_unique"], 0)
        constant, _ = library_statistics(sequences, [1] * 4, [2] * 4)
        self.assertIsNone(constant["predictor_oracle_spearman"])

    def test_clean_native_inference_and_report_artifacts(self):
        torch.set_num_threads(1)
        cfg = load_config(ROOT / "configs/evaluate.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library.fasta"
            library.write_text(
                f">wt\n{CREILOV_WT}\n>mut\nC{CREILOV_WT[1:]}\n>dup\n{CREILOV_WT}\n"
            )
            for role, noisy in (("oracle", False), ("predictor", True)):
                model = FluorescenceEnsemble(
                    ModelConfig(ensemble_size=1, hidden_dim=4), 121, 0.05
                )
                save_checkpoint(
                    model, root / f"{role}.pt", {"role": role, "noisy": noisy}
                )
            cfg = replace(
                cfg,
                library_path=library,
                oracle_checkpoint=root / "oracle.pt",
                predictor_checkpoint=root / "predictor.pt",
                output_dir=root,
                scoring_batch_size=2,
                device="cpu",
            )
            with patch("builtins.print"):
                stats = evaluate_library(cfg)
            self.assertEqual(stats["num_sequences"], 3)
            self.assertEqual(stats["num_unique"], 2)
            self.assertAlmostEqual(stats["mutations_mean"], 1 / 3)
            self.assertEqual(stats["diversity_mean_pairwise_hamming_unique"], 1)
            # Scores land in the library's own run folder, under a config
            # name that cannot overwrite the generating run's config.
            for name in (
                "metrics.json",
                "per_sequence.csv",
                "evaluate_config.json",
            ):
                self.assertTrue((cfg.output_dir / name).is_file(), name)
            self.assertEqual(
                json.loads((cfg.output_dir / "metrics.json").read_text()), stats
            )
            library.write_text(">masked\n" + CREILOV_WT[:-1] + "X\n")
            with self.assertRaisesRegex(ValueError, "canonical"):
                read_library(library)
            library.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                read_library(library)


if __name__ == "__main__":
    unittest.main()
