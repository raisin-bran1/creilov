"""CLI for training with validation diagnostics and ProteinGuide guidance."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from proteingen.modeling.guide import TAG, DEG

from .config import Config, load_config
from .data import CREILOV_WT, load_dataset
from .predictors import load_backbone, load_checkpoint, save_checkpoint
from .splits import make_splits, save_splits
from .training import seed_all, fit, subset, predict, metrics, correlation


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def prepare(cfg: Config):
    dataset = load_dataset(cfg.csv_path)
    fingerprint = hashlib.sha256(cfg.csv_path.read_bytes()).hexdigest()
    splits = make_splits(
        dataset.sequences, dataset.labels.tolist(), cfg.splits, cfg.seed
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = asdict(cfg)
    resolved["csv_path"], resolved["output_dir"] = str(cfg.csv_path), str(
        cfg.output_dir
    )
    write_json(cfg.output_dir / "config.json", resolved)
    save_splits(cfg.output_dir / "splits.json", splits, fingerprint)
    write_json(
        cfg.output_dir / "split_summary.json",
        {
            name: {
                part: len(getattr(split, part))
                for part in ("train", "validation", "test")
            }
            for name, split in splits.items()
        },
    )
    return dataset, splits, fingerprint


def evaluate(
    model, dataset, cfg: Config, directory: Path, *, noisy: bool, partition: str
) -> tuple[list[dict], np.ndarray]:
    """Score each masking level once and reuse predictions for distance groups."""
    reports = []
    distances = np.array(
        [sum(a != b for a, b in zip(s, CREILOV_WT)) for s in dataset.sequences]
    )
    labels = dataset.labels.numpy()
    clean_means = None
    fractions = cfg.evaluation.mask_fractions if noisy else [0.0]
    for fraction in fractions:
        seed_all(cfg.seed)
        predictions = predict(
            model, dataset, cfg.predictor_training.batch_size, fraction
        )
        if fraction == 0.0:
            clean_means = predictions[:, 0]

        def score(keep):
            if not np.any(keep):
                return {"n": 0}
            return metrics(
                labels[keep],
                predictions[keep],
                cfg.guidance.threshold,
                cfg.evaluation.top_k,
            )

        report = {
            "partition": partition,
            "mask_fraction": fraction,
            "used_for_checkpoint_selection_and_calibration": partition == "validation",
            **score(np.ones(len(dataset), dtype=bool)),
            "by_distance": {
                str(int(d)): score(distances == d)
                for d in sorted(set(distances.tolist()))
            },
        }
        reports.append(report)
        print(
            f"{partition} mask={fraction:g} n={report['n']} rmse={report['rmse']:.5f} spearman={report['spearman']}",
            flush=True,
        )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / f"predictions_mask_{fraction:g}.npz",
            sequences=np.array(dataset.sequences),
            labels=dataset.labels.numpy(),
            mean=predictions[:, 0],
            log_variance=predictions[:, 1],
        )
    write_json(directory / "metrics.json", reports)
    assert clean_means is not None
    return reports, clean_means


def evaluation_indices(indices: list[int], cfg: Config) -> list[int]:
    """Use a shared reproducible evaluation subset only if explicitly capped."""
    if cfg.evaluation.max_rows and len(indices) > cfg.evaluation.max_rows:
        return sorted(
            np.random.default_rng(cfg.seed)
            .choice(indices, cfg.evaluation.max_rows, replace=False)
            .tolist()
        )
    return list(indices)


def train(cfg: Config) -> None:
    dataset, splits, fingerprint = prepare(cfg)
    split = splits[cfg.train_split]
    validation_indices = evaluation_indices(split.validation, cfg)
    heldout = subset(dataset, validation_indices)
    summary, clean_scores = {}, {}
    write_json(
        cfg.output_dir / "validation_indices.json",
        {
            "csv_sha256": fingerprint,
            "indices": validation_indices,
            "test_evaluated": False,
        },
    )
    for role, noisy in (("oracle", False), ("predictor", True)):
        model, metadata = fit(
            dataset,
            split,
            getattr(cfg, role),
            getattr(cfg, role + "_training"),
            noisy=noisy,
            seed=cfg.seed,
            device=cfg.device,
            min_sigma=cfg.guidance.min_sigma,
        )
        metadata.update(csv_sha256=fingerprint, split=cfg.train_split, role=role)
        save_checkpoint(model, cfg.output_dir / role / "model.pt", metadata)
        write_json(cfg.output_dir / role / "training.json", metadata)
        reports, clean_scores[role] = evaluate(
            model,
            heldout,
            cfg,
            cfg.output_dir / role / "validation",
            noisy=noisy,
            partition="validation",
        )
        summary[role] = reports
        del model
    write_json(cfg.output_dir / "validation_summary.json", summary)
    oracle_scores, predictor_scores = clean_scores["oracle"], clean_scores["predictor"]
    write_json(
        cfg.output_dir / "agreement.json",
        {
            "n": len(heldout),
            "partition": "validation",
            "spearman": correlation(oracle_scores, predictor_scores),
            "rmse": float(np.sqrt(np.mean((oracle_scores - predictor_scores) ** 2))),
        },
    )


def benchmark(cfg: Config) -> None:
    """Eight single-member fits: configured oracle/predictor on each split.

    Architectures and training settings are copied from the two configured
    roles. Only ensemble_size changes; validation selects checkpoints and the
    disjoint test partition supplies benchmark metrics.
    """
    dataset, splits, fingerprint = prepare(cfg)
    summary = []
    for split_name, split in splits.items():
        test_indices = evaluation_indices(split.test, cfg)
        heldout = subset(dataset, test_indices)
        for role, noisy in (("oracle", False), ("predictor", True)):
            directory = cfg.output_dir / "benchmark" / split_name / role
            model_cfg = replace(getattr(cfg, role), ensemble_size=1)
            model, metadata = fit(
                dataset,
                split,
                model_cfg,
                getattr(cfg, role + "_training"),
                noisy=noisy,
                seed=cfg.seed,
                device=cfg.device,
                min_sigma=cfg.guidance.min_sigma,
            )
            metadata.update(
                csv_sha256=fingerprint,
                split=split_name,
                role=role,
                benchmark_ensemble_size=1,
                test_indices=test_indices,
            )
            save_checkpoint(model, directory / "model.pt", metadata)
            write_json(directory / "training.json", metadata)
            reports, _ = evaluate(
                model, heldout, cfg, directory / "test", noisy=noisy, partition="test"
            )
            summary.extend(
                {
                    "split": split_name,
                    "role": role,
                    "kind": model_cfg.kind,
                    "ensemble_size": 1,
                    **report,
                }
                for report in reports
            )
            write_json(cfg.output_dir / "benchmark/summary.json", summary)
            del model


def build_guidance(cfg: Config, generator=None):
    """Return native ProteinGen TAG/DEG; no family term and no fixed residues."""
    predictor = load_checkpoint(cfg.output_dir / "predictor/model.pt", cfg.device)
    if predictor.checkpoint_metadata.get(
        "role"
    ) != "predictor" or not predictor.checkpoint_metadata.get("noisy"):
        raise ValueError(
            "Guidance requires a noisy predictor checkpoint, not an oracle"
        )
    fingerprint = predictor.checkpoint_metadata.get("csv_sha256")
    if (
        fingerprint
        and hashlib.sha256(cfg.csv_path.read_bytes()).hexdigest() != fingerprint
    ):
        raise ValueError(
            "Predictor was trained on a different CSV; use its saved config"
        )
    predictor.set_target_(cfg.guidance.threshold)
    predictor.set_temp_(cfg.guidance.predictor_temperature)
    if generator is None:
        generator = load_backbone(cfg.guidance.generator)
    generator.tokenizer.mask_token_id = generator.tokenizer.vocab["<mask>"]
    generator.to(cfg.device).eval().set_temp_(cfg.guidance.generator_temperature)
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    if generator.tokenizer.vocab != predictor.tokenizer.vocab:
        raise ValueError("DEG and this workflow require matching ESM tokenizers")
    if cfg.guidance.method == "tag":
        return TAG(generator, predictor, use_clean_classifier=False).eval()
    return DEG(generator, predictor, argmax_masked_positions=False).eval()


def guidance_check(cfg: Config) -> None:
    seed_all(cfg.seed)
    guided = build_guidance(cfg)
    tokenizer = guided.tokenizer
    ids = tokenizer([CREILOV_WT], return_tensors="pt")["input_ids"].to(
        cfg.device
    )  # [1, 121]
    maskable = torch.zeros_like(ids, dtype=torch.bool)  # [1, 121] - residues only
    maskable[:, 1:-1] = True
    masked = maskable & (
        torch.rand(ids.shape, device=ids.device) < cfg.guidance.mask_fraction
    )  # [1, 121]
    ids[masked] = tokenizer.vocab["<mask>"]
    position = (
        cfg.guidance.position
    )  # BOS offset makes biological position equal token index
    ids[:, position] = tokenizer.vocab["<mask>"]
    base = guided.gen_model.get_log_probs(ids)  # [1, 121, V] - generator distribution
    context = (
        guided.at_position([position]) if isinstance(guided, DEG) else nullcontext()
    )
    with context:
        log_probs = guided.get_log_probs(
            ids
        )  # [1, 121, V] - TAG enables its own input gradients
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    chosen = [tokenizer.vocab[aa] for aa in amino_acids]
    if not torch.isfinite(log_probs[0, position, chosen]).all():
        raise ValueError("Non-finite guided amino-acid scores")
    write_json(
        cfg.output_dir / "guidance_check.json",
        {
            "method": cfg.guidance.method,
            "threshold": cfg.guidance.threshold,
            "position": position,
            "n_masked": int((ids == tokenizer.vocab["<mask>"]).sum()),
            "amino_acids": amino_acids,
            "base_probabilities": base[0, position, chosen]
            .exp()
            .detach()
            .cpu()
            .tolist(),
            "guided_probabilities": log_probs[0, position, chosen]
            .exp()
            .detach()
            .cpu()
            .tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["splits", "benchmark", "train", "guidance", "all"]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/workflow.toml"))
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    if args.command == "splits":
        prepare(cfg)
    elif args.command == "benchmark":
        benchmark(cfg)
    elif args.command == "train":
        train(cfg)
    elif args.command == "guidance":
        guidance_check(cfg)
    else:
        train(cfg)
        guidance_check(cfg)


if __name__ == "__main__":
    main()
