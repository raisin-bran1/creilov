"""End-to-end checks without downloading a foundation model."""

from dataclasses import replace
from contextlib import nullcontext
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from proteingen.data import ProteinDataset
from proteingen.modeling.generative_modeling import (
    GenerativeModel,
    MaskedModelLogitFormatter,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from creilov.config import (
    ModelConfig,
    TrainingConfig,
    SplitConfig,
    EvaluationConfig,
    load_config,
)
from creilov.data import CREILOV_WT
from creilov.predictors import FluorescenceEnsemble, save_checkpoint, load_checkpoint
from creilov.splits import make_splits
from creilov.training import fit, metrics, predict, subset
from creilov.workflow import build_guidance, train, benchmark


def fixture():
    sequences, labels = [], []
    for flags in product((0, 1), repeat=7):
        sequence = list(CREILOV_WT)
        for flag, position in zip(flags, (3, 4, 5, 7, 29, 47, 60)):
            if flag:
                sequence[position - 1] = "A" if sequence[position - 1] != "A" else "C"
        sequences.append("".join(sequence))
        labels.append(3.0 + sum(flags) * 0.1 + flags[0] * flags[1] * 0.07)
    return ProteinDataset(sequences, labels=torch.tensor(labels))


class TinyBackbone(nn.Module):
    """Nonlinear contextual differentiable test double, not an ESM substitute."""

    EMB_DIM = 4

    def __init__(self):
        super().__init__()
        self.tokenizer = EsmSequenceTokenizer()
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.EMB_DIM)

    def differentiable_embedding(self, ohe):
        embedded = ohe @ self.embedding.weight  # [B, P, D]
        return torch.tanh(embedded + embedded.mean(1, keepdim=True))


class ConstantLogits(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-0.5, 0.5, vocab_size))

    def forward(self, ids):
        return self.logits.expand(*ids.shape, -1)


def generator():
    tokenizer = EsmSequenceTokenizer()
    tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
    return GenerativeModel(
        ConstantLogits(tokenizer.vocab_size),
        tokenizer,
        MaskedModelLogitFormatter(tokenizer, tokenizer.vocab_size),
    )


class WorkflowTests(unittest.TestCase):
    def test_train_and_benchmark_use_role_configs_without_architecture_sweep(self):
        torch.set_num_threads(1)
        data = fixture()
        cfg = load_config(Path(__file__).parents[1] / "configs/workflow.toml")
        cfg = replace(
            cfg,
            splits=SplitConfig(distance_train_max=2, distance_validation_max=3),
            oracle=ModelConfig(kind="additive", ensemble_size=2),
            predictor=ModelConfig(ensemble_size=3, hidden_dim=4, dropout=0),
            oracle_training=TrainingConfig(epochs=1, batch_size=32, max_train=24),
            predictor_training=TrainingConfig(epochs=1, batch_size=32, max_train=16),
            evaluation=EvaluationConfig(mask_fractions=[0.0, 0.5], max_rows=5),
        )
        splits = make_splits(data.sequences, data.labels.tolist(), cfg.splits, cfg.seed)
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(cfg, output_dir=Path(directory))
            with patch(
                "creilov.workflow.prepare", return_value=(data, splits, "fixture")
            ), patch("creilov.workflow.fit", wraps=fit) as fits:
                train(cfg)
                self.assertEqual(fits.call_count, 2)
                self.assertEqual(
                    [call.args[2].ensemble_size for call in fits.call_args_list], [2, 3]
                )
                self.assertEqual(
                    [call.args[3] for call in fits.call_args_list],
                    [cfg.oracle_training, cfg.predictor_training],
                )
                summary = json.loads(
                    (cfg.output_dir / "validation_summary.json").read_text()
                )
                self.assertEqual(len(summary["oracle"]), 1)
                self.assertEqual(len(summary["predictor"]), 2)
                selected = json.loads(
                    (cfg.output_dir / "validation_indices.json").read_text()
                )["indices"]
                self.assertTrue(
                    set(selected) <= set(splits[cfg.train_split].validation)
                )
                self.assertTrue(set(selected).isdisjoint(splits[cfg.train_split].test))
                self.assertFalse((cfg.output_dir / "benchmark").exists())
                fits.reset_mock()
                benchmark(cfg)
                self.assertEqual(fits.call_count, 8)
                for i, (name, split) in enumerate(splits.items()):
                    for offset, role in enumerate(("oracle", "predictor")):
                        call = fits.call_args_list[2 * i + offset]
                        self.assertEqual(call.args[1], split)
                        self.assertEqual(
                            call.args[2], replace(getattr(cfg, role), ensemble_size=1)
                        )
                        self.assertEqual(call.args[3], getattr(cfg, role + "_training"))
                        metadata = json.loads(
                            (
                                cfg.output_dir
                                / "benchmark"
                                / name
                                / role
                                / "training.json"
                            ).read_text()
                        )
                        self.assertTrue(
                            set(metadata["test_indices"]) <= set(split.test)
                        )
                        self.assertTrue(
                            set(metadata["train_indices"]).isdisjoint(
                                metadata["test_indices"]
                            )
                        )
                reports = json.loads(
                    (cfg.output_dir / "benchmark/summary.json").read_text()
                )
                self.assertEqual(
                    len(reports), 12
                )  # 4 x (one clean oracle + two predictor mask levels)
                self.assertTrue(
                    all(
                        report["ensemble_size"] == 1 and report["partition"] == "test"
                        for report in reports
                    )
                )
                self.assertEqual(cfg.oracle.ensemble_size, 2)
                self.assertEqual(cfg.predictor.ensemble_size, 3)

    def test_split_invariants_and_reproducibility(self):
        data = fixture()
        cfg = SplitConfig(distance_train_max=2, distance_validation_max=3)
        splits = make_splits(data.sequences, data.labels.tolist(), cfg, 42)
        self.assertEqual(
            splits, make_splits(data.sequences, data.labels.tolist(), cfg, 42)
        )
        for split in splits.values():
            split.validate(len(data))
        for i in splits["distance"].train:
            self.assertLessEqual(
                sum(a != b for a, b in zip(data.sequences[i], CREILOV_WT)), 2
            )
        for i in splits["position"].train:
            for position in (7, 29, 47):
                self.assertEqual(
                    data.sequences[i][position - 1], CREILOV_WT[position - 1]
                )
        activity = splits["activity"]
        self.assertLess(
            max(data.labels[activity.train]), min(data.labels[activity.validation])
        )
        self.assertLess(
            max(data.labels[activity.validation]), min(data.labels[activity.test])
        )

    def test_training_checkpoint_and_native_guidance(self):
        torch.set_num_threads(1)
        data = fixture()
        split = make_splits(
            data.sequences,
            data.labels.tolist(),
            SplitConfig(distance_train_max=2, distance_validation_max=3),
            42,
        )["random"]
        for kind in ("one_hot_mlp", "linear_probe", "additive"):
            with self.subTest(kind=kind), patch(
                "creilov.predictors.load_backbone", side_effect=lambda _: TinyBackbone()
            ):
                model, metadata = fit(
                    data,
                    split,
                    ModelConfig(kind=kind, ensemble_size=2, hidden_dim=8, dropout=0),
                    TrainingConfig(epochs=2, batch_size=32, max_train=64),
                    noisy=True,
                    seed=42,
                    device="cpu",
                    min_sigma=0.05,
                )
                self.assertTrue(set(metadata["train_indices"]).isdisjoint(split.test))
                model.set_target_(3.5)
                ids = model.tokenizer([CREILOV_WT], return_tensors="pt")["input_ids"]
                ids[:, 7] = model.tokenizer.vocab["<mask>"]
                gradient = model.grad_log_prob(ids)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
                outputs = predict(model, subset(data, split.test), 32)
                self.assertTrue(np.isfinite(outputs).all())
                self.assertIsNotNone(
                    metrics(data.labels[split.test].numpy(), outputs, 3.5, 5)[
                        "spearman"
                    ]
                )
                # Keep backbone initialization identical on reload without ESM weights.
                backbone_state = (
                    model.members[0].embed_model.state_dict()
                    if kind == "linear_probe"
                    else None
                )

                def matching_backbone(_):
                    backbone = TinyBackbone()
                    if backbone_state is not None:
                        backbone.load_state_dict(backbone_state)
                    return backbone

                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    checkpoint = root / "predictor/model.pt"
                    save_checkpoint(
                        model, checkpoint, {**metadata, "role": "predictor"}
                    )
                    with patch(
                        "creilov.predictors.load_backbone",
                        side_effect=matching_backbone,
                    ):
                        restored = load_checkpoint(checkpoint, "cpu")
                        np.testing.assert_allclose(
                            outputs,
                            predict(restored, subset(data, split.test), 32),
                            rtol=1e-6,
                        )
                        cfg = load_config(
                            Path(__file__).parents[1] / "configs/workflow.toml"
                        )
                        cfg = replace(
                            cfg,
                            output_dir=root,
                            guidance=replace(cfg.guidance, threshold=3.5),
                        )
                        for method in ("tag", "deg"):
                            cfg.guidance.method = method
                            guided = build_guidance(cfg, generator=generator())
                            context = (
                                guided.at_position([7])
                                if method == "deg"
                                else nullcontext()
                            )
                            with context:
                                result = guided.get_log_probs(ids)
                            choices = [
                                guided.tokenizer.vocab[a]
                                for a in "ACDEFGHIKLMNPQRSTVWY"
                            ]
                            self.assertTrue(torch.isfinite(result[0, 7, choices]).all())
                            if method == "deg":
                                candidates = ids.repeat(guided.tokenizer.vocab_size, 1)
                                candidates[:, 7] = torch.arange(
                                    guided.tokenizer.vocab_size
                                )
                                exact = guided.gen_model.get_log_probs(ids)[
                                    0, 7
                                ] + guided.pred_model.get_log_probs(candidates)
                                expected = exact.log_softmax(-1)
                                torch.testing.assert_close(
                                    result[0, 7, choices], expected[choices]
                                )

    def test_single_member_variance_and_finite_tail(self):
        model = FluorescenceEnsemble(
            ModelConfig(ensemble_size=1, hidden_dim=4), 121, 0.05
        ).eval()
        model.set_target_(20.0)
        ids = model.tokenizer([CREILOV_WT], return_tensors="pt")["input_ids"]
        self.assertTrue(torch.isfinite(model.get_log_probs(ids)).all())
        raw = model.predict(ids)
        self.assertGreaterEqual(raw[0, 1].exp().item(), 0.05**2 - 1e-8)


if __name__ == "__main__":
    unittest.main()
