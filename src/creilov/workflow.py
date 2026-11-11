"""CLI for training with validation diagnostics and ProteinGuide guidance."""

import argparse
import csv
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from proteingen.modeling.guide import TAG, DEG
from proteingen.sampling import sample

from .config import (
    Config,
    DEFAULT_CONFIGS,
    load_config,
    serialize_config,
    validate_stage_config,
    guidance_overrides,
)
from .data import CREILOV_WT, load_dataset
from .library_evaluation import evaluate_library
from .predictors import (
    ControlPredictor,
    load_backbone,
    load_checkpoint,
    save_checkpoint,
)
from .splits import make_splits, save_splits
from .training import (
    seed_all,
    clear_members,
    fit,
    subset,
    predict,
    metrics,
    calibrate_variance,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def prepare(cfg: Config):
    libraries = ("S2", "S1") if cfg.stage == "train-predictor" else ("S2",)
    dataset = load_dataset(cfg.csv_path, allowed_libraries=libraries)
    fingerprint = hashlib.sha256(cfg.csv_path.read_bytes()).hexdigest()
    splits = make_splits(
        dataset.sequences, dataset.labels.tolist(), cfg.splits, cfg.seed
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = asdict(cfg)
    for key in ("csv_path", "output_dir", "predictor_checkpoint"):
        if resolved[key] is not None:
            resolved[key] = str(resolved[key])
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
    training_cfg = cfg.predictor_training if noisy else cfg.oracle_training
    if training_cfg is None:
        raise ValueError("Evaluation requires the role training configuration")
    fractions = cfg.evaluation.mask_fractions if noisy else [0.0]
    for fraction in fractions:
        seed_all(cfg.seed)
        predictions = predict(model, dataset, training_cfg.batch_size, fraction)
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


def train_oracle(cfg: Config) -> None:
    """Train only the clean oracle ensemble."""
    _train_role(cfg, "oracle")


def train_predictor(cfg: Config) -> None:
    """Train only the noisy predictor and its configured nested ensembles."""
    _train_role(cfg, "predictor")


def _train_role(cfg: Config, role: str) -> None:
    if getattr(cfg, role) is None or getattr(cfg, role + "_training") is None:
        raise ValueError(f"Missing {role} model/training configuration")
    noisy = role == "predictor"
    dataset, splits, fingerprint = prepare(cfg)
    split = splits[cfg.train_split]
    validation_indices = evaluation_indices(split.validation, cfg)
    heldout = subset(dataset, validation_indices)
    write_json(
        cfg.output_dir / "validation_indices.json",
        {
            "csv_sha256": fingerprint,
            "indices": validation_indices,
            "test_evaluated": False,
        },
    )
    print(
        f"Training {role}: split={cfg.train_split}, "
        f"members={getattr(cfg, role).ensemble_size}",
        flush=True,
    )
    model, metadata = fit(
        dataset,
        split,
        getattr(cfg, role),
        getattr(cfg, role + "_training"),
        noisy=noisy,
        seed=cfg.seed,
        device=cfg.device,
        min_sigma=cfg.guidance.min_sigma,
        resume_dir=cfg.output_dir / role / "members",
        data_key=f"{fingerprint}:{cfg.train_split}",
    )
    metadata.update(csv_sha256=fingerprint, split=cfg.train_split, role=role)
    save_checkpoint(model, cfg.output_dir / role / "model.pt", metadata)
    write_json(cfg.output_dir / role / "training.json", metadata)
    reports, _ = evaluate(
        model,
        heldout,
        cfg,
        cfg.output_dir / role / "validation",
        noisy=noisy,
        partition="validation",
    )
    if role == "predictor" and cfg.predictor_ensemble_sizes:
        ensemble_reports = {}
        full_validation = subset(dataset, split.validation)
        for size in cfg.predictor_ensemble_sizes:
            if size == cfg.predictor.ensemble_size:
                checkpoint = Path("predictor/model.pt")
                size_reports = reports
            else:
                print(f"Evaluating predictor prefix: members={size}", flush=True)
                prefix = model.prefix(size)
                residual_mse = calibrate_variance(
                    prefix, full_validation, cfg.predictor_training.batch_size
                )
                directory = Path("predictor") / f"ensemble_{size}"
                checkpoint = directory / "model.pt"
                prefix_metadata = {
                    **metadata,
                    "history": metadata["history"][:size],
                    "validation_residual_variance": residual_mse,
                    "parent_ensemble_size": cfg.predictor.ensemble_size,
                    "member_indices": list(range(size)),
                    "ensemble_selection": "nested_prefix",
                }
                save_checkpoint(prefix, cfg.output_dir / checkpoint, prefix_metadata)
                write_json(
                    cfg.output_dir / directory / "training.json", prefix_metadata
                )
                size_reports, _ = evaluate(
                    prefix,
                    heldout,
                    cfg,
                    cfg.output_dir / directory / "validation",
                    noisy=True,
                    partition="validation",
                )
                del prefix
            ensemble_reports[str(size)] = {
                "checkpoint": checkpoint.as_posix(),
                "member_indices": list(range(size)),
                "ensemble_selection": "nested_prefix",
                "reports": size_reports,
            }
            write_json(
                cfg.output_dir / "predictor_ensemble_summary.json", ensemble_reports
            )
    write_json(cfg.output_dir / "validation_summary.json", {role: reports})
    # Saved checkpoints now cover every member, so the resume cache is redundant.
    removed = clear_members(cfg.output_dir / role / "members")
    print(f"Removed {removed} cached member files", flush=True)


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
    if cfg.guidance.control:
        predictor = ControlPredictor().to(cfg.device).eval()
    else:
        predictor = load_predictor(cfg)
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


def load_predictor(cfg: Config):
    """Load the noisy predictor and confirm it matches this run's training data."""
    checkpoint = cfg.predictor_checkpoint or cfg.output_dir / "predictor/model.pt"
    predictor = load_checkpoint(checkpoint, cfg.device)
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
    return predictor


def masked_wild_type(cfg: Config, tokenizer, count: int, device) -> torch.Tensor:
    """Tokenized wild type with the designable residues replaced by <mask>.

    Every unmasked position keeps its wild-type residue, so the sampler infills
    only the chosen sites. With ``mask_positions`` the site set is identical for
    every draw; with ``mask_fraction`` each draw gets a fresh random set.
    """
    ids = tokenizer([CREILOV_WT] * count, return_tensors="pt")["input_ids"].to(
        device
    )  # [B, P] - includes protected BOS/EOS
    if cfg.guidance.mask_positions is None:
        masked = (
            torch.rand((count, len(CREILOV_WT)), device=ids.device)
            < cfg.guidance.mask_fraction
        )  # [B, L] - residue mask decisions
        ids[:, 1:-1].masked_fill_(masked, tokenizer.mask_token_id)
    else:
        # The BOS offset makes a biological position equal its token index.
        index = torch.tensor(cfg.guidance.mask_positions, device=ids.device)
        ids[:, index] = tokenizer.mask_token_id
    return ids


def generate_library(cfg: Config) -> None:
    """Sample full sequences with native ProteinGen random-order TAG or DEG."""
    seed_all(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.output_dir / "config.json", json.loads(serialize_config(cfg)))
    guided = build_guidance(cfg)
    # The saved config records the method, masking, seed and predictor, and
    # library.csv holds one row per draw, so no separate summary file is kept.
    # A run that stops early leaves fewer rows than guidance.num_samples, which
    # is how reporting detects an incomplete library.
    generated = 0
    unique = set()
    fields = ["id", "sequence", "n_mutations", "log_prob_given_order"]
    with (cfg.output_dir / "library.fasta").open("w", encoding="utf-8") as fasta, (
        cfg.output_dir / "library.csv"
    ).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while generated < cfg.guidance.num_samples:
            count = min(cfg.guidance.batch_size, cfg.guidance.num_samples - generated)
            ids = masked_wild_type(cfg, guided.tokenizer, count, cfg.device)
            trajectory = sample(
                guided, ids, n_parallel=1, in_order=None, live_preview=False
            )
            log_probs = trajectory["step_log_probs"].sum(
                -1
            )  # [B] - conditional on the sampled reveal order
            sequences = trajectory["sequences"]
            if len(sequences) != count or not torch.isfinite(log_probs).all():
                raise ValueError("Invalid sampler output")
            for sequence in sequences:
                if len(sequence) != len(CREILOV_WT) or set(sequence) - set(
                    "ACDEFGHIKLMNPQRSTVWY"
                ):
                    raise ValueError(
                        "Sampler must return complete 119-residue amino-acid sequences"
                    )
            for i, sequence in enumerate(sequences):
                name = f"sample_{generated + i + 1:06d}"
                fasta.write(f">{name}\n{sequence}\n")
                writer.writerow(
                    dict(
                        id=name,
                        sequence=sequence,
                        n_mutations=sum(a != b for a, b in zip(sequence, CREILOV_WT)),
                        log_prob_given_order=float(log_probs[i]),
                    )
                )
                unique.add(sequence)
            fasta.flush()
            stream.flush()
            generated += count
            print(
                f"Generated {generated}/{cfg.guidance.num_samples}; "
                f"unique={len(unique)}",
                flush=True,
            )


def guidance_check(cfg: Config) -> None:
    seed_all(cfg.seed)
    write_json(cfg.output_dir / "config.json", json.loads(serialize_config(cfg)))
    guided = build_guidance(cfg)
    tokenizer = guided.tokenizer
    ids = masked_wild_type(cfg, tokenizer, 1, cfg.device)  # [1, 121]
    position = (
        cfg.guidance.position
    )  # BOS offset makes biological position equal token index
    # The probe position must be masked even when it is not a designable site.
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
    parser.add_argument("command", choices=list(DEFAULT_CONFIGS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--method", choices=["tag", "deg"])
    parser.add_argument(
        "--control",
        action="store_true",
        help="unguided control: replace the predictor with an uninformative one",
    )
    args = parser.parse_args()
    config_path = args.config or Path(DEFAULT_CONFIGS[args.command])
    cfg = load_config(config_path.resolve())
    validate_stage_config(args.command, cfg)
    cfg = guidance_overrides(cfg, args.num_samples, args.method, args.control)
    commands = {
        "splits": prepare,
        "benchmark": benchmark,
        "train-predictor": train_predictor,
        "train-oracle": train_oracle,
        "guidance": generate_library,
        "evaluate": evaluate_library,
    }
    commands[args.command](cfg)


if __name__ == "__main__":
    main()
