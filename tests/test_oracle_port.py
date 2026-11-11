"""Weight remapping for the ported ProteinGuide CreiLOV oracle, without downloads.

The real port is verified against their published weights by
``scripts/port_proteinguide_oracle.py``'s own gates. These tests cover the
mapping logic with synthetic weights, so a regression is caught without
re-downloading a 427 MB bundle.
"""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from proteingen.data import ProteinDataset

from creilov.config import ModelConfig, load_config
from creilov.data import CREILOV_WT
from creilov.predictors import FluorescenceEnsemble, save_checkpoint
from creilov.training import predict
from creilov.workflow import build_guidance

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from port_proteinguide_oracle import (  # noqa: E402
    CREILOV_ORACLE_ALPHABET,
    MASK_LETTER,
    WT_LENGTH,
    alphabet_for,
    describe_layers,
    reference_scores,
    remap_first_layer,
)

ROOT = Path(__file__).parents[1]
LENGTH = WT_LENGTH + 2  # BOS + residues + EOS
HIDDEN = 16  # small enough to stay fast; the mapping is width-agnostic
RESIDUES = CREILOV_ORACLE_ALPHABET.replace(MASK_LETTER, "")


def tokenizer_with_mask() -> EsmSequenceTokenizer:
    tokenizer = EsmSequenceTokenizer()
    tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
    return tokenizer


def synthetic_state(alphabet_size: int, hidden: int = HIDDEN, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "fc1.weight": torch.randn(
            hidden, WT_LENGTH * alphabet_size, generator=generator
        ),
        "fc1.bias": torch.randn(hidden, generator=generator),
        "fc2.weight": torch.randn(1, hidden, generator=generator),
        "fc2.bias": torch.randn(1, generator=generator),
    }


def remap(state: dict, alphabet_size: int) -> torch.Tensor:
    """Remapped first-layer weights, reshaped to [hidden, position, vocab]."""
    tokenizer = tokenizer_with_mask()
    flat = remap_first_layer(
        state["fc1.weight"], alphabet_for(alphabet_size), tokenizer, LENGTH
    )
    return flat.reshape(-1, LENGTH, tokenizer.vocab_size)


def build_ported(state: dict, alphabet_size: int) -> FluorescenceEnsemble:
    layout = describe_layers(state)
    model = FluorescenceEnsemble(
        ModelConfig(
            kind="one_hot_mlp",
            ensemble_size=1,
            hidden_dim=layout["hidden_dim"],
            hidden_layers=1,
            dropout=0.1,
        ),
        LENGTH,
        0.05,
    )
    member = model.members[0]
    with torch.no_grad():
        member.layers[0].weight.copy_(
            remap_first_layer(
                state["fc1.weight"], alphabet_for(alphabet_size), tokenizer_with_mask(), LENGTH
            )
        )
        member.layers[0].bias.copy_(state["fc1.bias"])
        member.layers[3].weight.copy_(state["fc2.weight"])
        member.layers[3].bias.copy_(state["fc2.bias"])
        model.label_mean.copy_(torch.tensor(0.0))
        model.label_scale.copy_(torch.tensor(1.0))
    return model.eval()


def probe_sequences(count: int = 8, seed: int = 1) -> list[str]:
    rng = np.random.default_rng(seed)
    sequences = [CREILOV_WT]
    for _ in range(count):
        mutant = list(CREILOV_WT)
        for position in rng.choice(WT_LENGTH, 6, replace=False):
            mutant[position] = RESIDUES[rng.integers(len(RESIDUES))]
        sequences.append("".join(mutant))
    return sequences


class RemapTests(unittest.TestCase):
    def test_ported_matches_their_forward_for_both_alphabet_widths(self):
        """The remap must reproduce their function, not merely a similar one."""
        for alphabet_size in (len(RESIDUES), len(CREILOV_ORACLE_ALPHABET)):
            with self.subTest(alphabet_size=alphabet_size):
                state = synthetic_state(alphabet_size)
                layout = describe_layers(state)
                self.assertEqual(layout["alphabet_size"], alphabet_size)
                model = build_ported(state, alphabet_size)
                sequences = probe_sequences()
                ours = predict(model, ProteinDataset(sequences), 8, mask_fraction=0.0)
                theirs = reference_scores(
                    state, layout, alphabet_for(alphabet_size), sequences
                )
                self.assertLess(float(np.abs(ours[:, 0] - theirs).max()), 1e-4)

    def test_remap_relocates_every_weight_without_loss(self):
        state = synthetic_state(len(RESIDUES))
        source = state["fc1.weight"]
        remapped = remap(state, len(RESIDUES))
        self.assertEqual(
            remapped.shape, (source.shape[0], LENGTH, tokenizer_with_mask().vocab_size)
        )
        # Same multiset of nonzero values, just relocated.
        kept = remapped.flatten()
        kept = kept[kept != 0].sort().values
        expected = source.flatten()
        expected = expected[expected != 0].sort().values
        self.assertTrue(torch.allclose(kept, expected))

    def test_only_residue_columns_carry_weight(self):
        """Zeroing always-on BOS/EOS is exact, so no bias correction is needed."""
        tokenizer = tokenizer_with_mask()
        remapped = remap(synthetic_state(len(RESIDUES)), len(RESIDUES))
        self.assertTrue(torch.all(remapped[:, 0, :] == 0))  # BOS position
        self.assertTrue(torch.all(remapped[:, -1, :] == 0))  # EOS position
        # Derive the non-residue set from the vocab rather than listing tokens.
        for token, column in tokenizer.vocab.items():
            if token not in set(RESIDUES):
                self.assertTrue(
                    torch.all(remapped[:, :, column] == 0),
                    f"{token} column should carry no weight for a "
                    f"{len(RESIDUES)}-letter checkpoint",
                )

    def test_mask_letter_maps_onto_mask_column_only_when_present(self):
        tokenizer = tokenizer_with_mask()
        wide = remap(
            synthetic_state(len(CREILOV_ORACLE_ALPHABET)),
            len(CREILOV_ORACLE_ALPHABET),
        )
        self.assertTrue(torch.any(wide[:, :, tokenizer.vocab["<mask>"]] != 0))
        # Their mask letter is a slot, not the residue of the same name.
        self.assertTrue(torch.all(wide[:, :, tokenizer.vocab[MASK_LETTER]] == 0))
        narrow = remap(synthetic_state(len(RESIDUES)), len(RESIDUES))
        self.assertTrue(torch.all(narrow[:, :, tokenizer.vocab["<mask>"]] == 0))


class LayoutDetectionTests(unittest.TestCase):
    def test_detects_layers_by_shape_not_name(self):
        state = synthetic_state(len(RESIDUES))
        renamed = {
            "backbone.input_proj.weight": state["fc1.weight"],
            "backbone.input_proj.bias": state["fc1.bias"],
            "head.out.weight": state["fc2.weight"],
            "head.out.bias": state["fc2.bias"],
        }
        layout = describe_layers(renamed)
        self.assertEqual(layout["first_key"], "backbone.input_proj.weight")
        self.assertEqual(layout["last_key"], "head.out.weight")
        self.assertEqual(layout["hidden_dim"], state["fc1.weight"].shape[0])
        self.assertEqual(layout["input_dim"], state["fc1.weight"].shape[1])

    def test_rejects_deeper_networks(self):
        state = synthetic_state(len(RESIDUES))
        state["fc_mid.weight"] = torch.zeros(HIDDEN, HIDDEN)
        state["fc_mid.bias"] = torch.zeros(HIDDEN)
        with self.assertRaisesRegex(ValueError, "hidden_layers=1"):
            describe_layers(state)

    def test_rejects_heteroscedastic_two_output_head(self):
        state = synthetic_state(len(RESIDUES))
        state["fc2.weight"] = torch.zeros(2, HIDDEN)
        state["fc2.bias"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "scalar-output"):
            describe_layers(state)

    def test_rejects_unexpected_alphabet_width(self):
        with self.assertRaisesRegex(ValueError, "expected 20 or 21"):
            alphabet_for(len(CREILOV_ORACLE_ALPHABET) + 4)

    def test_narrow_alphabet_drops_only_the_mask_slot(self):
        self.assertEqual(alphabet_for(len(CREILOV_ORACLE_ALPHABET)), CREILOV_ORACLE_ALPHABET)
        self.assertEqual(alphabet_for(len(RESIDUES)), RESIDUES)
        self.assertEqual(CREILOV_ORACLE_ALPHABET[-1], MASK_LETTER)


class GuidanceRefusalTests(unittest.TestCase):
    def test_guidance_refuses_an_oracle_checkpoint(self):
        """The oracle's variance channel is uncalibrated, so guidance must refuse it."""
        model = build_ported(synthetic_state(len(RESIDUES)), len(RESIDUES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path, {"role": "oracle", "noisy": False})
            cfg = load_config(ROOT / "configs/guidance.toml")
            # Pin the guided arm: a control run ignores the checkpoint entirely,
            # so it could never surface the role mismatch this test is about.
            cfg = replace(
                cfg,
                predictor_checkpoint=path,
                device="cpu",
                guidance=replace(cfg.guidance, control=False),
            )
            with self.assertRaisesRegex(ValueError, "noisy predictor"):
                build_guidance(cfg)


if __name__ == "__main__":
    unittest.main()
