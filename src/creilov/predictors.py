"""ProteinGen regressors and a differentiable Gaussian-threshold ensemble.

Index legend: B batch, P token position (includes BOS/EOS), V vocabulary,
D contextual embedding dimension, M ensemble member, O output.
"""

from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from proteingen.modeling.predictive_modeling import (
    PredictiveModel,
    OneHotMLP,
    LinearProbe,
    gaussian_binary_logits,
    point_estimate_binary_logits,
)

from .config import ModelConfig


def load_backbone(name: str):
    from proteingen.modeling.models.esm import ESMC, ESM3

    if name in {"esmc_300m", "esmc_600m"}:
        return ESMC(name)
    if name == "esm3-open":
        return ESM3(name)
    raise ValueError(f"Unsupported ESM checkpoint: {name}")


class ScalarEvent:
    def format_raw_to_logits(self, raw, seq, **kwargs):
        return point_estimate_binary_logits(raw[:, 0], float(self.target), k=1.0)


class FluorescenceMLP(ScalarEvent, OneHotMLP):
    """ProteinGen MLP; hidden_layers counts hidden, not total, linear layers."""


class FluorescenceProbe(ScalarEvent, LinearProbe):
    """Frozen ESM features with a trainable linear readout."""

    def forward(self, ohe_seq_SPT):
        self.embed_model.eval()  # freeze stochastic backbone behavior too
        return super().forward(ohe_seq_SPT)


class AdditiveRegressor(ScalarEvent, PredictiveModel):
    def __init__(self, tokenizer, length: int):
        super().__init__(tokenizer)
        self.linear = nn.Linear(length * tokenizer.vocab_size, 1)

    def forward(self, ohe):
        flat = ohe.flatten(1)  # [B, P*V] - positional indicators
        return self.linear(flat)


def residue_mean(embeddings, token_ids, tokenizer):
    # Preserve mask positions; exclude BOS/EOS/PAD/UNK from pooling.
    excluded = [tokenizer.vocab[t] for t in ("<cls>", "<eos>", "<pad>", "<unk>")]
    keep = ~torch.isin(token_ids, token_ids.new_tensor(excluded))  # [B, P]
    weights = keep.unsqueeze(-1).to(embeddings.dtype)  # [B, P, 1]
    return (embeddings * weights).sum(1) / weights.sum(1).clamp_min(1)


def make_member(cfg: ModelConfig, tokenizer, length: int, backbone=None):
    if cfg.kind == "one_hot_mlp":
        return FluorescenceMLP(
            tokenizer, length, cfg.hidden_dim, cfg.hidden_layers, 1, cfg.dropout
        )
    if cfg.kind == "additive":
        return AdditiveRegressor(tokenizer, length)
    if cfg.kind == "linear_probe":
        if backbone is None:
            backbone = load_backbone(cfg.backbone)
        return FluorescenceProbe(
            backbone,
            1,
            pooling_fn=lambda emb, ids: residue_mean(emb, ids, tokenizer),
            freeze_embed_model=True,
        )
    raise ValueError(f"Unknown model kind: {cfg.kind}")


class FluorescenceEnsemble(PredictiveModel):
    """Returns physical-unit [mean, log variance], then Gaussian event logits.

    Residual variance is estimated on validation data. It supplements empirical
    ensemble variance and permits one-member models; it is not an OOD guarantee.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        length: int,
        min_sigma: float,
        *,
        members: list[PredictiveModel] | None = None,
    ):
        tokenizer = EsmSequenceTokenizer()
        tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
        super().__init__(tokenizer)
        self.config = cfg
        self.length = length
        self.min_sigma = min_sigma
        if members is None:
            backbone = (
                load_backbone(cfg.backbone) if cfg.kind == "linear_probe" else None
            )
            members = [
                make_member(cfg, tokenizer, length, backbone)
                for _ in range(cfg.ensemble_size)
            ]
        if len(members) != cfg.ensemble_size:
            raise ValueError("Member count must match ensemble_size")
        self.members = nn.ModuleList(members)
        self.register_buffer("label_mean", torch.tensor(0.0))  # scalar
        self.register_buffer("label_scale", torch.tensor(1.0))  # scalar
        self.register_buffer("residual_variance", torch.tensor(min_sigma**2))  # scalar

    def prefix(self, size: int) -> "FluorescenceEnsemble":
        """Share the first trained members, with independent calibration buffers."""
        if not 1 <= size <= len(self.members):
            raise ValueError("Prefix size must be within the trained ensemble")
        prefix = FluorescenceEnsemble(
            replace(self.config, ensemble_size=size),
            self.length,
            self.min_sigma,
            members=list(self.members[:size]),
        ).to(self.device)
        prefix.label_mean.copy_(self.label_mean)
        prefix.label_scale.copy_(self.label_scale)
        return prefix.eval()

    def member_predictions(self, ohe):
        if self.config.kind == "linear_probe":
            first = self.members[0]
            first.embed_model.eval()
            emb = first.embed_model.differentiable_embedding(ohe)  # [B, P, D]
            ids = ohe.argmax(-1)  # [B, P] - used only for pooling mask
            pooled = first.pooling_fn(emb, ids)  # [B, D] - shared frozen representation
            values = torch.stack(
                [m.w(pooled)[:, 0] for m in self.members], -1
            )  # [B, M]
        else:
            values = torch.stack([m(ohe)[:, 0] for m in self.members], -1)  # [B, M]
        return values * self.label_scale + self.label_mean

    def forward(self, ohe):
        values = self.member_predictions(ohe)  # [B, M] - physical fluorescence units
        mean = values.mean(-1)  # [B]
        variance = values.var(-1, unbiased=False) + self.residual_variance  # [B]
        return torch.stack([mean, variance.clamp_min(self.min_sigma**2).log()], -1)

    def format_raw_to_logits(self, raw, seq, **kwargs):
        return gaussian_binary_logits(raw[:, 0], raw[:, 1], float(self.target))


class ControlPredictor(PredictiveModel):
    """Uninformative predictor for an unguided control arm.

    Returns equal binary logits for every sequence, so log p(target | x) is
    log 0.5 everywhere and its input gradient is exactly zero. TAG then adds a
    zero delta and DEG adds a constant across the scored position's tokens,
    both of which leave the generator's distribution untouched. Guidance still
    runs through the same sampler, so a control library shares the generator's
    reveal order and token draws with a guided run at the same seed.
    """

    def __init__(self):
        tokenizer = EsmSequenceTokenizer()
        tokenizer.mask_token_id = tokenizer.vocab["<mask>"]
        super().__init__(tokenizer)
        # ProbabilityModel.device reads a parameter, and the zero weight keeps
        # the output attached to the autograd graph so grad_log_prob can run.
        self.zero_weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, ohe):
        attached = ohe.flatten(1).sum(-1) * self.zero_weight  # [B] - always zero
        return torch.stack([attached, attached], -1)  # [B, 2] - binary logits

    def format_raw_to_logits(self, raw, seq, **kwargs):
        return raw  # [false, true] are already equal


def save_checkpoint(model: FluorescenceEnsemble, path: Path, metadata: dict) -> None:
    # Frozen ESM is reconstructed from its named checkpoint; avoid serializing it.
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".embed_model." not in k
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "length": model.length,
            "min_sigma": model.min_sigma,
            "state": state,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: str) -> FluorescenceEnsemble:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FluorescenceEnsemble(
        ModelConfig(**checkpoint["model_config"]),
        checkpoint["length"],
        checkpoint["min_sigma"],
    )
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    if unexpected or any(".embed_model." not in key for key in missing):
        raise ValueError(f"Checkpoint mismatch: {missing}, {unexpected}")
    model.checkpoint_metadata = checkpoint["metadata"]
    return model.to(device).eval()
