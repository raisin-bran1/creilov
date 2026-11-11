"""Port the ProteinGuide (Yang et al.) CreiLOV oracle into this project's checkpoint format.

Run ``uv run python scripts/port_proteinguide_oracle.py --help`` for usage.
See ``docs/proteinguide_oracle_port.md`` for the full rationale.

Why this script exists
----------------------
The ``evaluate`` stage scores a generated library with a *clean oracle*. An oracle
we train ourselves sees essentially the same assay rows our guidance predictor
saw, so judge and student share a teacher. ProteinGuide's published CreiLOV
oracle is an independent, paper-comparable judge: a 10-member one-hot MLP
ensemble fit to the full ~167k-variant assay table.

What makes the port exact rather than approximate
-------------------------------------------------
Their member and our ``FluorescenceMLP`` are the *same* topology --
``Linear(L*A, H) -> ReLU -> Dropout -> Linear(H, 1)``. Only the one-hot layout
differs: they encode 119 residues over a 20/21-letter alphabet; we encode 121
tokens (BOS + 119 + EOS) over the 33-token ESM vocabulary. So the port is a
permutation-and-zero-pad of the first layer's weight columns. Layers after the
first are copied verbatim.

Design choices, and why
-----------------------
1. Reuse ``kind="one_hot_mlp"`` instead of adding a model class. The topology is
   already identical, so a new class would be duplicate code and a second thing
   to keep in sync with ``load_checkpoint``.
2. Map their ``X`` column (which they use as the mask token) onto our ESM
   ``<mask>`` column rather than dropping it. Dropping it would be exact for the
   clean, complete sequences ``evaluate`` actually scores, but mapping it keeps
   the port faithful under masking too, at no cost.
3. Zero every column with no counterpart: BOS, EOS and the non-canonical ESM
   tokens. A zero weight column contributes nothing, so the represented function
   is unchanged. BOS/EOS are always-on constants, so zeroing them is exact (not
   merely harmless): no bias shift is needed.
4. Set ``label_mean=0`` and ``label_scale=1``. Our ensemble standardizes targets
   during training and undoes it in ``member_predictions``; their model predicts
   raw units with no normalization. Leaving our training-set moments in place
   would skew every score.
5. Leave ``residual_variance`` at the ``min_sigma`` floor and do NOT calibrate
   it. Their members are mean-only (MSE), so there is no honest variance to
   port. ``evaluate`` reads only channel 0, so this is sufficient there -- and
   the metadata marks the checkpoint ``role="oracle"``, which makes
   ``build_guidance`` refuse it, so the uncalibrated variance cannot leak into
   guidance.
6. Keep ``dropout=0.1`` even though ``eval()`` disables it: ProteinGen's
   ``OneHotMLP`` only emits a ``Dropout`` module when ``dropout > 0``, and that
   changes the ``nn.Sequential`` indices the state dict keys depend on.

Verification gates (the script fails loudly rather than producing a plausible oracle)
-------------------------------------------------------------------------------------
A wrong alphabet permutation silently yields a confident, meaningless model, so
correctness is checked from two independent directions:

* ``--check remap``: replay their forward pass straight from the raw state dict
  and require it to agree with the ported ensemble to within tolerance. This
  proves the weight remap, but NOT the alphabet (both sides would share a wrong
  order).
* Anchor correlation: score their own ``creilov_processed.csv`` rows and require
  rank correlation against their ``fitness`` column. The oracle was fit on those
  rows, so a correct alphabet correlates strongly and a scrambled one does not.
  Measured: +0.872 for the true order, versus -0.53..+0.02 for random
  permutations, hence the 0.80 floor.
* Wild-type point check: the sharpest discriminator, because all 119 positions
  contribute. Measured: the true order predicts 3.994 against a measured label
  of 4.023 (error 0.029, with the 10 members agreeing to +/-0.005), while a
  single A<->L swap errs by 0.28 and random permutations by 0.7..2.1. Hence the
  0.10 tolerance, which passes the true order and rejects even a single
  transposition.
* Units comparison: compare predictions against *our* ``fitness`` column to
  report whether the output scale is comparable to ``guidance.threshold``.
  Reported, not enforced.

Confirmed findings from the real checkpoints
-------------------------------------------
* Their state dicts are ``fc1 (400, 2380)`` / ``fc2 (1, 400)``, so the alphabet
  is the 20 canonical letters with NO mask column. The ``X`` -> ``<mask>``
  mapping in ``remap_first_layer`` is therefore inert for this checkpoint, and
  every ESM mask column ends up with zero weight. Consequence: the ported
  oracle is only meaningful for clean, complete sequences -- which is exactly
  what ``evaluate`` scores (``mask_fraction=0.0``). Do not score masked inputs
  with it; their model had no representation for a masked residue at all.
* Their ``creilov_processed.csv`` has 167,530 rows, matching our
  ``creilov_combined.csv`` exactly, and the wild-type ``fitness`` agrees to the
  digit (4.023229753). The target is the same log10 mean fluorescence we use, so
  ``guidance.threshold = 4.1`` transfers directly with no rescaling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

# Their alphabets, from pdg/fitness/constants.py. X is last and doubles as the
# mask token: "Use the last position of the alphabet for <mask> since we will
# not have `X` in the input."
CREILOV_ORACLE_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
WT_LENGTH = 119
MASK_LETTER = "X"


def load_member_state(path: Path) -> dict[str, torch.Tensor]:
    """Load one of their ``model{i}.pth`` files.

    ``weights_only=True`` because these are downloaded artifacts; we never
    execute pickled objects from them.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    # Their checkpoints may be a bare state dict or wrapped under a common key.
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(blob).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        inner = blob.get(key)
        if isinstance(inner, dict) and any(
            isinstance(v, torch.Tensor) for v in inner.values()
        ):
            blob = inner
            break
    tensors = {k: v for k, v in blob.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise ValueError(f"{path}: no tensors found")
    return tensors


def describe_layers(state: dict[str, torch.Tensor]) -> dict:
    """Locate the input and output Linear layers by shape, not by name.

    Their class names and attribute nesting are not guaranteed stable, but the
    shapes are pinned by the problem: the input layer's fan-in must be a
    multiple of 119, and the output layer emits a single scalar.
    """
    weights = {k: v for k, v in state.items() if v.ndim == 2}
    if not weights:
        raise ValueError("No 2-D weight tensors in state dict")

    first_candidates = [
        (k, v) for k, v in weights.items() if v.shape[1] % WT_LENGTH == 0
    ]
    if not first_candidates:
        raise ValueError(
            "No layer whose fan-in is a multiple of 119; cannot infer the "
            f"one-hot layout from shapes {[tuple(v.shape) for v in weights.values()]}"
        )
    first_key, first = max(first_candidates, key=lambda kv: kv[1].shape[1])

    last_candidates = [(k, v) for k, v in weights.items() if v.shape[0] == 1]
    if len(last_candidates) != 1:
        raise ValueError(
            "Expected exactly one scalar-output layer (a mean-only head). Found "
            f"{len(last_candidates)}. A 2-output head would mean they ported a "
            "heteroscedastic variance model, which needs a different mapping."
        )
    last_key, last = last_candidates[0]

    if len(weights) != 2:
        raise ValueError(
            f"Expected exactly 2 Linear layers (hidden_layers=1), found "
            f"{len(weights)}: {sorted(weights)}. Deeper nets need "
            "ModelConfig.hidden_layers raised and this mapping extended."
        )
    if first.shape[0] != last.shape[1]:
        raise ValueError(
            f"Layer shapes do not chain: {tuple(first.shape)} -> {tuple(last.shape)}"
        )

    alphabet_size = first.shape[1] // WT_LENGTH
    return {
        "first_key": first_key,
        "last_key": last_key,
        "hidden_dim": int(first.shape[0]),
        "alphabet_size": int(alphabet_size),
        "input_dim": int(first.shape[1]),
    }


def bias_for(state: dict[str, torch.Tensor], weight_key: str) -> torch.Tensor:
    key = weight_key.rsplit(".", 1)[0] + ".bias"
    if key not in state:
        raise ValueError(f"Missing bias for {weight_key} (expected {key})")
    return state[key]


def alphabet_for(size: int) -> str:
    """Their alphabet, truncated to the width the checkpoint actually uses."""
    if size == len(CREILOV_ORACLE_ALPHABET):
        return CREILOV_ORACLE_ALPHABET
    if size == len(CREILOV_ORACLE_ALPHABET) - 1:
        # Trained without the trailing mask slot.
        return CREILOV_ORACLE_ALPHABET[:-1]
    raise ValueError(
        f"Checkpoint implies a {size}-letter alphabet; expected 20 or 21 to match "
        f"CREILOV_ORACLE_ALPHABET ({CREILOV_ORACLE_ALPHABET!r})"
    )


def remap_first_layer(
    their_weight: torch.Tensor, alphabet: str, tokenizer, length: int
) -> torch.Tensor:
    """Scatter their [H, 119*A] input weights into our [H, 121*V] layout.

    Their encoding is position-major: ``F.one_hot(tokens)`` then ``view(B, -1)``
    gives index ``pos * A + aa``. Ours is the same ordering over the ESM vocab,
    with one BOS token in front, so residue ``i`` lives at token ``i + 1``.
    """
    hidden = their_weight.shape[0]
    vocab_size = tokenizer.vocab_size
    source = their_weight.reshape(hidden, WT_LENGTH, len(alphabet))
    target = torch.zeros(hidden, length, vocab_size, dtype=their_weight.dtype)

    for index, letter in enumerate(alphabet):
        # X is their mask slot, so it maps onto our <mask> column, not a residue.
        token = "<mask>" if letter == MASK_LETTER else letter
        if token not in tokenizer.vocab:
            raise ValueError(f"ESM vocab has no column for {token!r}")
        target[:, 1 : WT_LENGTH + 1, tokenizer.vocab[token]] = source[:, :, index]

    moved = int((target != 0).sum())
    expected = int((source != 0).sum())
    if moved != expected:
        raise ValueError(f"Remap lost weights: {expected} -> {moved}")
    return target.reshape(hidden, length * vocab_size)


def reference_scores(
    state: dict[str, torch.Tensor], layers: dict, alphabet: str, sequences: list[str]
) -> np.ndarray:
    """Replay their forward pass from the raw state dict, independent of our code.

    Mirrors MLPPredictor.forward: one-hot the token indices, flatten, then
    Linear -> ReLU -> Linear. Dropout is identity at eval.
    """
    index = {letter: i for i, letter in enumerate(alphabet)}
    tokens = torch.tensor(
        [[index[residue] for residue in sequence] for sequence in sequences],
        dtype=torch.long,
    )
    one_hot = torch.nn.functional.one_hot(tokens, num_classes=len(alphabet))
    flat = one_hot.reshape(tokens.shape[0], -1).float()
    hidden = torch.nn.functional.relu(
        flat @ state[layers["first_key"]].T.float()
        + bias_for(state, layers["first_key"]).float()
    )
    out = hidden @ state[layers["last_key"]].T.float() + bias_for(
        state, layers["last_key"]
    ).float()
    return out[:, 0].detach().numpy()


def wild_type_label(path: Path, wild_type: str) -> float | None:
    """Look up the measured wild-type fitness, for the point-prediction gate."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return None
        column = next(
            (c for c in ("fitness", "score", "log_fitness") if c in reader.fieldnames),
            None,
        )
        if column is None or "sequence" not in reader.fieldnames:
            return None
        for row in reader:
            if row["sequence"] == wild_type:
                try:
                    return float(row[column])
                except (TypeError, ValueError):
                    return None
    return None


def read_fitness_csv(path: Path, limit: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Read (sequence, fitness) pairs, keeping only clean full-length rows."""
    sequences: list[str] = []
    values: list[float] = []
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            raise ValueError(f"{path}: no 'sequence' column")
        column = next(
            (c for c in ("fitness", "score", "log_fitness") if c in reader.fieldnames),
            None,
        )
        if column is None:
            raise ValueError(f"{path}: no fitness-like column in {reader.fieldnames}")
        for row in reader:
            sequence = row["sequence"]
            if len(sequence) != WT_LENGTH or set(sequence) - canonical:
                continue
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                sequences.append(sequence)
                values.append(value)
    if not sequences:
        raise ValueError(f"{path}: no usable rows")
    if limit and len(sequences) > limit:
        picks = np.random.default_rng(seed).choice(
            len(sequences), limit, replace=False
        )
        sequences = [sequences[i] for i in picks]
        values = [values[i] for i in picks]
    return sequences, np.asarray(values, dtype=np.float64)


def score_with_ensemble(model, sequences: list[str], batch_size: int) -> np.ndarray:
    """Clean (unmasked) ensemble means, via the same path evaluate uses."""
    from proteingen.data import ProteinDataset

    from creilov.training import predict

    dataset = ProteinDataset(sequences)
    return predict(model, dataset, batch_size, mask_fraction=0.0)[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory holding their model0.pth ... model9.pth",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination checkpoint path"
    )
    parser.add_argument(
        "--their-csv",
        type=Path,
        help="Their creilov_processed.csv, for the decisive alphabet anchor check",
    )
    parser.add_argument(
        "--our-csv",
        type=Path,
        default=Path("data/creilov_combined.csv"),
        help="Our table, for the units comparison",
    )
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--anchor-rows", type=int, default=4000)
    # Floors calibrated against measured permutation controls; see module docstring.
    parser.add_argument("--anchor-min-spearman", type=float, default=0.80)
    parser.add_argument("--wt-tolerance", type=float, default=0.10)
    parser.add_argument("--remap-tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    from creilov.config import ModelConfig
    from creilov.data import CREILOV_WT
    from creilov.predictors import FluorescenceEnsemble, save_checkpoint

    paths = [args.checkpoint_dir / f"model{i}.pth" for i in range(args.members)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing member checkpoints: {missing}")

    states = [load_member_state(p) for p in paths]
    layouts = [describe_layers(s) for s in states]
    if len({json.dumps({k: v for k, v in l.items()}, sort_keys=True) for l in layouts}) != 1:
        raise ValueError(f"Members disagree on architecture: {layouts}")
    layout = layouts[0]
    alphabet = alphabet_for(layout["alphabet_size"])
    print(
        f"Their architecture: Linear({layout['input_dim']} -> {layout['hidden_dim']})"
        f" -> ReLU -> Linear({layout['hidden_dim']} -> 1); "
        f"alphabet={alphabet!r} ({layout['alphabet_size']} letters)",
        flush=True,
    )

    tokenizer = EsmSequenceTokenizer()
    tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
    length = WT_LENGTH + 2
    cfg = ModelConfig(
        kind="one_hot_mlp",
        ensemble_size=args.members,
        hidden_dim=layout["hidden_dim"],
        hidden_layers=1,
        dropout=0.1,  # keep the Dropout module so Sequential indices line up
    )
    model = FluorescenceEnsemble(cfg, length, args.min_sigma)

    for member, state in zip(model.members, states, strict=True):
        first = remap_first_layer(
            state[layout["first_key"]].float(), alphabet, tokenizer, length
        )
        with torch.no_grad():
            member.layers[0].weight.copy_(first)
            member.layers[0].bias.copy_(bias_for(state, layout["first_key"]).float())
            member.layers[3].weight.copy_(state[layout["last_key"]].float())
            member.layers[3].bias.copy_(bias_for(state, layout["last_key"]).float())

    with torch.no_grad():
        # Their model predicts raw units; suppress our standardization.
        model.label_mean.copy_(torch.tensor(0.0))
        model.label_scale.copy_(torch.tensor(1.0))
        model.residual_variance.copy_(torch.tensor(args.min_sigma**2))
    model.eval()

    report: dict = {
        "their_architecture": layout,
        "alphabet": alphabet,
        "mask_letter_mapped_to": "<mask>",
        "members": args.members,
        "member_sha256": [
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        ],
    }

    # --- Gate 1: remap exactness (proves the scatter, not the alphabet) --------
    probes = [CREILOV_WT]
    rng = np.random.default_rng(args.seed)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(16):
        mutant = list(CREILOV_WT)
        for position in rng.choice(WT_LENGTH, 5, replace=False):
            mutant[position] = residues[rng.integers(len(residues))]
        probes.append("".join(mutant))

    ours = score_with_ensemble(model, probes, args.batch_size)
    theirs = np.mean(
        [reference_scores(s, layout, alphabet, probes) for s in states], axis=0
    )
    deviation = float(np.abs(ours - theirs).max())
    report["remap_max_abs_deviation"] = deviation
    print(f"Remap check: max |ported - reference| = {deviation:.3e}", flush=True)
    if not deviation < args.remap_tolerance:
        raise SystemExit(
            f"Remap check FAILED: deviation {deviation:.3e} exceeds "
            f"{args.remap_tolerance:.3e}. The weight scatter is wrong."
        )

    # --- Gate 2: alphabet anchor (decisive) -----------------------------------
    if args.their_csv:
        sequences, labels = read_fitness_csv(
            args.their_csv, args.anchor_rows, args.seed
        )
        scores = score_with_ensemble(model, sequences, args.batch_size)
        rho = float(spearmanr(scores, labels).statistic)
        r = float(pearsonr(scores, labels).statistic)
        report["anchor"] = {
            "csv": str(args.their_csv),
            "rows": len(sequences),
            "spearman": rho,
            "pearson": r,
            "score_mean": float(scores.mean()),
            "label_mean": float(labels.mean()),
        }
        print(
            f"Anchor check: n={len(sequences)} spearman={rho:.4f} pearson={r:.4f}",
            flush=True,
        )
        if not rho >= args.anchor_min_spearman:
            raise SystemExit(
                f"Anchor check FAILED: Spearman {rho:.4f} < "
                f"{args.anchor_min_spearman}. The alphabet order is almost "
                "certainly wrong -- do NOT use this checkpoint."
            )
    else:
        print(
            "WARNING: --their-csv not given, so the alphabet order is UNVERIFIED.",
            flush=True,
        )
        report["anchor"] = None

    # --- Gate 3: units comparison (reported, not enforced) --------------------
    if args.our_csv and args.our_csv.is_file():
        sequences, labels = read_fitness_csv(args.our_csv, args.anchor_rows, args.seed)
        scores = score_with_ensemble(model, sequences, args.batch_size)
        rho = float(spearmanr(scores, labels).statistic)
        report["units"] = {
            "csv": str(args.our_csv),
            "rows": len(sequences),
            "spearman": rho,
            "ported_oracle_mean": float(scores.mean()),
            "ported_oracle_min": float(scores.min()),
            "ported_oracle_max": float(scores.max()),
            "our_label_mean": float(labels.mean()),
            "our_label_min": float(labels.min()),
            "our_label_max": float(labels.max()),
        }
        print(
            f"Units check: spearman={rho:.4f}; ported range "
            f"[{scores.min():.3f}, {scores.max():.3f}] vs our log10 fitness range "
            f"[{labels.min():.3f}, {labels.max():.3f}]",
            flush=True,
        )

    # --- Gate 4: wild-type point prediction (sharpest alphabet discriminator) --
    wt_score = float(score_with_ensemble(model, [CREILOV_WT], args.batch_size)[0])
    report["wild_type_score"] = wt_score
    wt_label = next(
        (
            value
            for source in (args.their_csv, args.our_csv)
            if source and source.is_file()
            for value in [wild_type_label(source, CREILOV_WT)]
            if value is not None
        ),
        None,
    )
    report["wild_type_label"] = wt_label
    if wt_label is None:
        print(
            f"Wild-type ported oracle score: {wt_score:.5f} (no measured label found; "
            "point check SKIPPED)",
            flush=True,
        )
    else:
        error = abs(wt_score - wt_label)
        report["wild_type_abs_error"] = error
        print(
            f"Wild-type check: predicted {wt_score:.5f} vs measured {wt_label:.5f} "
            f"(|error|={error:.5f})",
            flush=True,
        )
        if not error <= args.wt_tolerance:
            raise SystemExit(
                f"Wild-type check FAILED: |error| {error:.5f} > {args.wt_tolerance}. "
                "The alphabet order or weight mapping is wrong -- do NOT use this "
                "checkpoint."
            )

    metadata = {
        "role": "oracle",
        "noisy": False,
        "source": "ProteinGuide (Yang et al.) CreiLOV oracle ensemble",
        "source_checkpoints": [p.name for p in paths],
        "port_script": "scripts/port_proteinguide_oracle.py",
        "label_standardization": "disabled (label_mean=0, label_scale=1)",
        "variance_calibrated": False,
        "variance_note": (
            "Members are mean-only; residual_variance is the min_sigma floor. "
            "Channel 1 is not a calibrated uncertainty. Evaluation uses channel 0."
        ),
        "verification": report,
    }
    save_checkpoint(model, args.output, metadata)
    sidecar = args.output.with_suffix(".port.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}\nWrote {sidecar}", flush=True)


if __name__ == "__main__":
    main()
