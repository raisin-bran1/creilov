"""End-to-end checks without downloading a foundation model."""

from dataclasses import replace
from contextlib import nullcontext
from itertools import product
import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from proteingen.data import ProteinDataset
from proteingen.sampling import sample as native_sample
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
    guidance_overrides,
)
from creilov.data import CREILOV_WT
from creilov.predictors import FluorescenceEnsemble, save_checkpoint, load_checkpoint
from creilov.splits import make_splits
from creilov.training import fit, metrics, predict, subset
from creilov.workflow import (
    build_guidance,
    train_oracle,
    train_predictor,
    benchmark,
    generate_library,
    masked_wild_type,
)


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
    def test_generate_complete_libraries_with_native_tag_and_deg(self):
        torch.set_num_threads(1)
        cfg = load_config(Path(__file__).parents[1] / "configs/guidance.toml")
        cfg = replace(cfg, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "predictor.pt"
            predictor = FluorescenceEnsemble(
                ModelConfig(ensemble_size=1, hidden_dim=4, dropout=0), 121, 0.05
            )
            save_checkpoint(predictor, checkpoint, {"role": "predictor", "noisy": True})
            for method in ("tag", "deg"):
                run = replace(
                    cfg,
                    output_dir=root / method,
                    predictor_checkpoint=checkpoint,
                    guidance=replace(
                        cfg.guidance, method=method, num_samples=3, batch_size=2
                    ),
                )
                with self.subTest(method=method), patch(
                    "creilov.workflow.load_backbone", side_effect=lambda _: generator()
                ), patch("creilov.workflow.sample", wraps=native_sample) as sampler:
                    generate_library(run)
                    self.assertEqual(
                        [call.args[1].shape[0] for call in sampler.call_args_list],
                        [2, 1],
                    )
                    self.assertTrue(
                        all(
                            call.kwargs["n_parallel"] == 1
                            for call in sampler.call_args_list
                        )
                    )
                with (run.output_dir / "library.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                self.assertEqual(len({row["id"] for row in rows}), 3)
                self.assertTrue(
                    all(
                        len(row["sequence"]) == 119
                        and set(row["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY")
                        for row in rows
                    )
                )
                self.assertTrue(
                    all(np.isfinite(float(row["log_prob_given_order"])) for row in rows)
                )
                fasta = (run.output_dir / "library.fasta").read_text().splitlines()
                self.assertEqual(fasta[1::2], [row["sequence"] for row in rows])
                self.assertFalse((run.output_dir / "library_summary.json").exists())
                self.assertEqual(len(rows), 3)
                resolved = json.loads((run.output_dir / "config.json").read_text())
                self.assertEqual(resolved["guidance"]["num_samples"], 3)
                self.assertEqual(resolved["guidance"]["method"], method)

    def test_mask_positions_confine_the_library_to_the_named_sites(self):
        torch.set_num_threads(1)
        cfg = load_config(Path(__file__).parents[1] / "configs/guidance.toml")
        sites = [7, 36, 94]
        cfg = replace(
            cfg,
            device="cpu",
            guidance=replace(cfg.guidance, mask_positions=sites),
        )
        tokenizer = EsmSequenceTokenizer()
        tokenizer.mask_token_id = tokenizer.vocab["<mask>"]

        # The template masks exactly the named sites and nothing else.
        ids = masked_wild_type(cfg, tokenizer, 2, "cpu")  # [2, 121]
        wild = tokenizer([CREILOV_WT], return_tensors="pt")["input_ids"]  # [1, 121]
        masked = (ids == tokenizer.mask_token_id).nonzero()[:, 1].unique().tolist()
        # The BOS offset makes a biological position equal its token index.
        self.assertEqual(masked, sites)
        kept = [i for i in range(ids.shape[1]) if i not in sites]
        self.assertTrue(torch.equal(ids[:, kept], wild[:, kept].expand(2, -1)))

        # Dropping to mask_fraction restores independent per-draw site sets.
        every = masked_wild_type(replace(cfg, guidance=replace(cfg.guidance,
            mask_positions=None, mask_fraction=1.0)), tokenizer, 2, "cpu")
        self.assertTrue(
            torch.all(every[:, 1:-1] == tokenizer.mask_token_id)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "predictor.pt"
            predictor = FluorescenceEnsemble(
                ModelConfig(ensemble_size=1, hidden_dim=4, dropout=0), 121, 0.05
            )
            save_checkpoint(predictor, checkpoint, {"role": "predictor", "noisy": True})
            run = replace(
                cfg,
                output_dir=root / "library",
                predictor_checkpoint=checkpoint,
                guidance=replace(cfg.guidance, num_samples=4, batch_size=2),
            )
            with patch(
                "creilov.workflow.load_backbone", side_effect=lambda _: generator()
            ), patch("creilov.workflow.sample", wraps=native_sample):
                generate_library(run)
            with (run.output_dir / "library.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            for row in rows:
                differing = {
                    i + 1
                    for i, (a, b) in enumerate(zip(row["sequence"], CREILOV_WT))
                    if a != b
                }
                self.assertTrue(differing <= set(sites), differing)
                self.assertLessEqual(int(row["n_mutations"]), len(sites))
            resolved = json.loads((run.output_dir / "config.json").read_text())
            self.assertEqual(resolved["guidance"]["mask_positions"], sites)

    def test_nested_ensembles_reuse_members_and_calibrate_separately(self):
        torch.set_num_threads(1)
        data = fixture()
        cfg = load_config(Path(__file__).parents[1] / "configs/benchmark.toml")
        cfg = replace(cfg, device="cpu")
        cfg = replace(
            cfg,
            splits=SplitConfig(distance_train_max=2, distance_validation_max=3),
            oracle=ModelConfig(kind="additive", ensemble_size=1),
            predictor=ModelConfig(ensemble_size=3, hidden_dim=4, dropout=0),
            predictor_ensemble_sizes=[1, 2, 3],
            oracle_training=TrainingConfig(epochs=1, batch_size=32, max_train=24),
            predictor_training=TrainingConfig(epochs=1, batch_size=32, max_train=16),
            evaluation=EvaluationConfig(mask_fractions=[0.0, 0.5]),
        )
        split = make_splits(data.sequences, data.labels.tolist(), cfg.splits, cfg.seed)[
            "random"
        ]
        with tempfile.TemporaryDirectory() as directory:
            cfg = replace(cfg, output_dir=Path(directory))
            with patch(
                "creilov.workflow.prepare",
                return_value=(data, {"random": split}, "fixture"),
            ), patch("creilov.workflow.fit", wraps=fit) as fits:
                train_predictor(cfg)
                self.assertEqual(
                    fits.call_count, 1
                )  # Never retrain the small prefixes.
            self.assertFalse((cfg.output_dir / "oracle").exists())
            # The resume cache is cleaned up once the stage completes.
            self.assertFalse((cfg.output_dir / "predictor/members").exists())
            parent = load_checkpoint(cfg.output_dir / "predictor/model.pt", "cpu")
            summary = json.loads(
                (cfg.output_dir / "predictor_ensemble_summary.json").read_text()
            )
            ids = parent.tokenizer(data.sequences, return_tensors="pt")["input_ids"]
            ohe = torch.nn.functional.one_hot(
                ids, num_classes=parent.tokenizer.vocab_size
            ).float()  # [B, P, V] - fixed clean inputs
            values = parent.member_predictions(ohe)  # [B, M]
            for size in (1, 2, 3):
                entry = summary[str(size)]
                child = load_checkpoint(cfg.output_dir / entry["checkpoint"], "cpu")
                self.assertEqual(child.config.ensemble_size, size)
                self.assertEqual(len(child.checkpoint_metadata["history"]), size)
                self.assertEqual(
                    child.checkpoint_metadata["train_indices"],
                    parent.checkpoint_metadata["train_indices"],
                )
                actual = child.predict(ids)[:, 0]  # [B] - physical mean
                torch.testing.assert_close(actual, values[:, :size].mean(-1))
                residual_mse = (
                    (actual[split.validation] - data.labels[split.validation]) ** 2
                ).mean()  # scalar - child-specific validation residual
                self.assertAlmostEqual(
                    child.residual_variance.item(),
                    max(residual_mse.item(), cfg.guidance.min_sigma**2),
                    places=5,
                )
                self.assertTrue(
                    all(r["partition"] == "validation" for r in entry["reports"])
                )
                self.assertTrue(
                    set(child.checkpoint_metadata["train_indices"]).isdisjoint(
                        split.test
                    )
                )
            prefix = parent.prefix(1)
            self.assertIs(prefix.members[0], parent.members[0])
            before = parent.residual_variance.item()
            prefix.residual_variance.fill_(123.0)
            self.assertEqual(parent.residual_variance.item(), before)

    def test_train_and_benchmark_use_role_configs_without_architecture_sweep(self):
        torch.set_num_threads(1)
        data = fixture()
        cfg = load_config(Path(__file__).parents[1] / "configs/benchmark.toml")
        cfg = replace(cfg, device="cpu")
        cfg = replace(
            cfg,
            splits=SplitConfig(distance_train_max=2, distance_validation_max=3),
            oracle=ModelConfig(kind="additive", ensemble_size=2),
            predictor=ModelConfig(ensemble_size=3, hidden_dim=4, dropout=0),
            predictor_ensemble_sizes=[],
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
                train_oracle(cfg)
                self.assertFalse((cfg.output_dir / "predictor").exists())
                self.assertEqual(fits.call_count, 1)
                train_predictor(cfg)
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
                self.assertEqual(set(summary), {"predictor"})
                oracle_reports = json.loads(
                    (cfg.output_dir / "oracle/validation/metrics.json").read_text()
                )
                self.assertEqual(len(oracle_reports), 1)
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

    def test_interrupted_fit_resumes_only_matching_members(self):
        torch.set_num_threads(1)
        data = fixture()
        split = make_splits(
            data.sequences,
            data.labels.tolist(),
            SplitConfig(distance_train_max=2, distance_validation_max=3),
            42,
        )["random"]
        model_cfg = ModelConfig(
            kind="one_hot_mlp", ensemble_size=2, hidden_dim=8, dropout=0
        )
        train_cfg = TrainingConfig(epochs=2, batch_size=32, max_train=64)
        kwargs = dict(noisy=True, seed=42, device="cpu", min_sigma=0.05)
        with tempfile.TemporaryDirectory() as directory:
            resume_dir = Path(directory) / "members"
            reference, _ = fit(
                data, split, model_cfg, train_cfg, resume_dir=resume_dir, **kwargs
            )
            self.assertEqual(
                sorted(p.name for p in resume_dir.glob("*.pt")),
                ["member_000.pt", "member_001.pt"],
            )
            # A re-run reuses the cache, so weights match bit for bit.
            resumed, metadata = fit(
                data, split, model_cfg, train_cfg, resume_dir=resume_dir, **kwargs
            )
            for first, second in zip(
                reference.state_dict().values(), resumed.state_dict().values()
            ):
                self.assertTrue(torch.equal(first, second))
            self.assertEqual(len(metadata["history"]), 2)
            # A growing ensemble keeps the trained prefix and fits the rest.
            grown, _ = fit(
                data,
                split,
                replace(model_cfg, ensemble_size=3),
                train_cfg,
                resume_dir=resume_dir,
                **kwargs,
            )
            self.assertTrue(
                torch.equal(
                    reference.members[0].state_dict()["layers.0.weight"],
                    grown.members[0].state_dict()["layers.0.weight"],
                )
            )
            # Any other change to the fit invalidates the cache.
            changed, _ = fit(
                data,
                split,
                model_cfg,
                replace(train_cfg, learning_rate=train_cfg.learning_rate * 3),
                resume_dir=resume_dir,
                **kwargs,
            )
            self.assertFalse(
                torch.equal(
                    reference.members[0].state_dict()["layers.0.weight"],
                    changed.members[0].state_dict()["layers.0.weight"],
                )
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
                            Path(__file__).parents[1] / "configs/benchmark.toml"
                        )
                        cfg = replace(
                            cfg,
                            output_dir=root / "guidance",
                            device="cpu",
                            predictor_checkpoint=checkpoint,
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

    def test_control_guidance_leaves_the_generator_distribution_unchanged(self):
        torch.set_num_threads(1)
        cfg = load_config(Path(__file__).parents[1] / "configs/guidance.toml")
        cfg = guidance_overrides(cfg, None, None, control=True)
        cfg = replace(cfg, device="cpu")
        self.assertTrue(cfg.guidance.control)
        self.assertIsNone(cfg.predictor_checkpoint)
        choices = [EsmSequenceTokenizer().vocab[a] for a in "ACDEFGHIKLMNPQRSTVWY"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("tag", "deg"):
                run = replace(
                    cfg,
                    output_dir=root / method,
                    guidance=replace(
                        cfg.guidance, method=method, num_samples=2, batch_size=2
                    ),
                )
                with self.subTest(method=method), patch(
                    "creilov.workflow.load_backbone", side_effect=lambda _: generator()
                ):
                    guided = build_guidance(run)
                    ids = guided.tokenizer([CREILOV_WT], return_tensors="pt")[
                        "input_ids"
                    ]
                    ids[:, 7] = guided.tokenizer.mask_token_id
                    base = guided.gen_model.get_log_probs(ids)  # [1, P, V]
                    context = (
                        guided.at_position([7]) if method == "deg" else nullcontext()
                    )
                    with context:
                        result = guided.get_log_probs(ids)  # [1, P, V]
                    # A control may only shift every candidate equally, which
                    # the normalization then removes.
                    torch.testing.assert_close(
                        result[0, 7, choices], base[0, 7, choices]
                    )
                    generate_library(run)
                resolved = json.loads((run.output_dir / "config.json").read_text())
                self.assertTrue(resolved["guidance"]["control"])
                self.assertIsNone(resolved["predictor_checkpoint"])
                with (run.output_dir / "library.csv").open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)

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
