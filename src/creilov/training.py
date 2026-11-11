"""Training, validation-only selection/calibration, and held-out metrics.

Index legend: B batch, P token position, V vocabulary, N dataset rows.
"""

import hashlib
import json
import math
from pathlib import Path
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from proteingen.data import (
    ProteinDataset,
    no_noise,
    fully_unmasked,
    uniform_mask_noise,
    uniform_time,
)

from .config import ModelConfig, TrainingConfig
from .predictors import FluorescenceEnsemble
from .splits import Split


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(dataset: ProteinDataset, indices: list[int]) -> ProteinDataset:
    return ProteinDataset(
        [dataset.sequences[i] for i in indices], labels=dataset.labels[indices]
    )


def make_loader(
    dataset, model, batch_size: int, *, noisy: bool, shuffle: bool, mask_fraction=None
):
    if mask_fraction is not None:
        noise_fn = uniform_mask_noise(model.tokenizer)
        time_sampler = lambda: 1.0 - mask_fraction
    elif noisy:
        noise_fn, time_sampler = uniform_mask_noise(model.tokenizer), uniform_time
    else:
        noise_fn, time_sampler = no_noise, fully_unmasked
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset.collator(model, noise_fn, time_sampler),
    )


@torch.no_grad()
def predict(
    model: FluorescenceEnsemble, dataset, batch_size: int, mask_fraction: float = 0.0
):
    model.eval()
    loader = make_loader(
        dataset,
        model,
        batch_size,
        noisy=False,
        shuffle=False,
        mask_fraction=mask_fraction,
    )
    outputs = []
    for batch in loader:
        ids = batch["input_ids"].to(model.device)  # [B, P] - clean or masked tokens
        raw = model.predict(ids)  # [B, 2] - physical mean and log variance
        outputs.append(raw.cpu())
    return torch.cat(outputs).numpy()


def select_training_indices(
    split: Split, train_cfg: TrainingConfig, seed: int
) -> list[int]:
    """Uniform, fixed-seed sample without replacement from the training pool."""
    indices = list(split.train)
    random.Random(seed).shuffle(indices)
    return indices[: train_cfg.max_train] if train_cfg.max_train else indices


def member_signature(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_indices: list[int],
    validation_indices: list[int],
    *,
    noisy: bool,
    seed: int,
    length: int,
    data_key: str,
) -> str:
    """Hash everything a single member's fit depends on, except member count.

    Excluding ensemble_size lets a larger re-run reuse members already trained
    under identical settings; any other change invalidates the cache.
    """
    model = {k: v for k, v in asdict(model_cfg).items() if k != "ensemble_size"}
    payload = json.dumps(
        {
            "model": model,
            "training": asdict(train_cfg),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "noisy": noisy,
            "seed": seed,
            "length": length,
            "data_key": data_key,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def member_path(resume_dir: Path, member_index: int) -> Path:
    return resume_dir / f"member_{member_index:03d}.pt"


def load_member(resume_dir: Path, member_index: int, signature: str) -> dict | None:
    """Return a cached member only when it was fit under the same signature."""
    path = member_path(resume_dir, member_index)
    if not path.is_file():
        return None
    try:
        cached = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # truncated by an interrupted write, or stale format
        print(f"member={member_index} ignoring unreadable cache: {error}", flush=True)
        return None
    if cached.get("signature") != signature:
        print(f"member={member_index} ignoring cache from different settings", flush=True)
        return None
    return cached


def save_member(
    resume_dir: Path,
    member_index: int,
    signature: str,
    state: dict,
    history: list[dict],
) -> None:
    """Write atomically so an interrupted save cannot leave a half file behind."""
    resume_dir.mkdir(parents=True, exist_ok=True)
    final = member_path(resume_dir, member_index)
    staging = final.with_suffix(".pt.tmp")
    torch.save({"signature": signature, "state": state, "history": history}, staging)
    staging.replace(final)


def clear_members(resume_dir: Path) -> int:
    """Drop the per-member cache once a stage no longer needs to resume."""
    if not resume_dir.is_dir():
        return 0
    removed = 0
    for path in sorted(resume_dir.glob("member_*.pt*")):
        path.unlink()
        removed += 1
    if not any(resume_dir.iterdir()):
        resume_dir.rmdir()
    return removed


def fit(
    dataset,
    split: Split,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    *,
    noisy: bool,
    seed: int,
    device: str,
    min_sigma: float,
    resume_dir: Path | None = None,
    data_key: str = "",
):
    seed_all(seed)
    indices = select_training_indices(split, train_cfg, seed)
    train, validation = subset(dataset, indices), subset(dataset, split.validation)
    length = len(dataset.sequences[0]) + 2
    model = FluorescenceEnsemble(model_cfg, length, min_sigma).to(device)
    model.label_mean.copy_(train.labels.mean())
    model.label_scale.copy_(train.labels.std(unbiased=False).clamp_min(1e-6))
    signature = member_signature(
        model_cfg,
        train_cfg,
        indices,
        split.validation,
        noisy=noisy,
        seed=seed,
        length=length,
        data_key=data_key,
    )
    histories = []
    for member_index, member in enumerate(model.members):
        seed_all(seed + member_index)
        cached = (
            load_member(resume_dir, member_index, signature) if resume_dir else None
        )
        if cached is not None:
            missing, unexpected = member.load_state_dict(cached["state"], strict=False)
            assert not unexpected and all(
                k.startswith("embed_model.") for k in missing
            )
            histories.append(cached["history"])
            print(f"member={member_index} restored from cache", flush=True)
            continue
        train_loader = make_loader(
            train, member, train_cfg.batch_size, noisy=noisy, shuffle=True
        )
        val_loader = make_loader(
            validation, member, train_cfg.batch_size, noisy=False, shuffle=False
        )
        optimizer = torch.optim.AdamW(
            [p for p in member.parameters() if p.requires_grad],
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        best, best_state, stale = math.inf, None, 0
        history = []
        for epoch in range(train_cfg.epochs):
            member.train()
            total, count = 0.0, 0
            for batch in train_loader:
                ids = batch["input_ids"].to(
                    device
                )  # [B, P] - stochastic masks for noisy training
                target = batch["labels"].to(device)  # [B] - measured log fluorescence
                target = (
                    target - model.label_mean
                ) / model.label_scale  # [B] - train-only standardization
                optimizer.zero_grad(set_to_none=True)
                output = member.predict(ids)[
                    :, 0
                ]  # [B] - standardized regression output
                loss = F.mse_loss(output, target)  # scalar - labels never masked
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), train_cfg.grad_clip)
                optimizer.step()
                total += loss.item() * len(ids)
                count += len(ids)
            member.eval()
            val_error, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(
                        device
                    )  # [B, P] - clean checkpoint-selection inputs
                    target = batch["labels"].to(device)  # [B]
                    output = (
                        member.predict(ids)[:, 0] * model.label_scale + model.label_mean
                    )  # [B]
                    val_error += F.mse_loss(output, target, reduction="sum").item()
                    val_count += len(ids)
            val_mse = val_error / val_count
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_standardized_mse": total / count,
                    "validation_mse": val_mse,
                }
            )
            print(
                f"member={member_index} epoch={epoch+1} train_mse={total/count:.5f} val_mse={val_mse:.5f}",
                flush=True,
            )
            if val_mse < best:
                best, stale = val_mse, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in member.state_dict().items()
                    if not k.startswith("embed_model.")
                }
            else:
                stale += 1
            if stale >= train_cfg.patience:
                break
        assert best_state is not None
        missing, unexpected = member.load_state_dict(best_state, strict=False)
        assert not unexpected and all(k.startswith("embed_model.") for k in missing)
        histories.append(history)
        if resume_dir is not None:
            save_member(resume_dir, member_index, signature, best_state, history)
    model.eval()
    residual_mse = calibrate_variance(model, validation, train_cfg.batch_size)
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # input gradients remain enabled for TAG
    metadata = {
        "noisy": noisy,
        "seed": seed,
        "training": asdict(train_cfg),
        "train_indices": indices,
        "validation_indices": split.validation,
        "history": histories,
        "validation_residual_variance": residual_mse,
    }
    return model, metadata


def calibrate_variance(
    model: FluorescenceEnsemble, validation, batch_size: int
) -> float:
    """Estimate each ensemble's own residual floor using clean validation only."""
    val_predictions = predict(model, validation, batch_size)
    residual_mse = float(
        np.mean((val_predictions[:, 0] - validation.labels.numpy()) ** 2)
    )
    model.residual_variance.fill_(max(residual_mse, model.min_sigma**2))
    return residual_mse


def correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def metrics(labels, predictions, threshold: float, top_k: int) -> dict:
    y, mean = np.asarray(labels), predictions[:, 0]
    sigma = np.exp(predictions[:, 1] / 2)
    probabilities = torch.special.ndtr(torch.tensor((mean - threshold) / sigma)).numpy()
    positives = y > threshold
    selected = np.argsort(-mean, kind="stable")[: min(top_k, len(y))]
    precision = float(positives[selected].mean())
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
        "mae": float(np.mean(np.abs(mean - y))),
        "spearman": correlation(y, mean),
        "gaussian_nll": float(
            np.mean(
                0.5
                * (
                    ((y - mean) / sigma) ** 2
                    + 2 * np.log(sigma)
                    + math.log(2 * math.pi)
                )
            )
        ),
        "brier": float(np.mean((probabilities - positives) ** 2)),
        "coverage_95": float(np.mean(np.abs(mean - y) <= 1.96 * sigma)),
        "top_k": len(selected),
        "top_k_precision": precision,
        "top_k_enrichment": (
            precision / float(positives.mean()) if positives.any() else None
        ),
        "top_k_recall": (
            float(positives[selected].sum() / positives.sum())
            if positives.any()
            else None
        ),
    }
