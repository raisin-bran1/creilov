"""Typed TOML configuration; relative paths resolve from the project root."""

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from pathlib import PurePosixPath
import tomllib

DEFAULT_CONFIGS = {
    "benchmark": "configs/benchmark.toml",
    "splits": "configs/benchmark.toml",
    "train-predictor": "configs/train_predictor.toml",
    "train-oracle": "configs/train_oracle.toml",
    "guidance": "configs/guidance.toml",
    "evaluate": "configs/evaluate.toml",
}


@dataclass
class ModelConfig:
    kind: str = "one_hot_mlp"
    ensemble_size: int = 5
    hidden_dim: int = 400
    hidden_layers: int = 1
    dropout: float = 0.1
    backbone: str = "esmc_300m"

    def __post_init__(self) -> None:
        if self.kind not in {"one_hot_mlp", "linear_probe", "additive"}:
            raise ValueError(f"Unknown predictor: {self.kind}")
        if min(self.ensemble_size, self.hidden_dim, self.hidden_layers) < 1:
            raise ValueError("Model sizes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 5
    max_train: int = 2000  # 0 means all training rows
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch_size, patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.max_train < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid training budget or gradient clip")


@dataclass
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    distance_train_max: int = 4
    distance_validation_max: int = 5
    activity_train_quantile: float = 0.8
    activity_validation_quantile: float = 0.9
    validation_positions: list[int] = field(default_factory=lambda: [29])
    test_positions: list[int] = field(default_factory=lambda: [7, 47])

    def __post_init__(self) -> None:
        if not (
            0 < self.train_fraction < self.train_fraction + self.validation_fraction < 1
        ):
            raise ValueError("Invalid random split fractions")
        if not 0 <= self.distance_train_max < self.distance_validation_max:
            raise ValueError("Distance cutoffs must be ordered")
        if not 0 < self.activity_train_quantile < self.activity_validation_quantile < 1:
            raise ValueError("Activity quantiles must be ordered")
        positions = self.validation_positions + self.test_positions
        if not self.validation_positions or not self.test_positions:
            raise ValueError("Position splits need validation and test sites")
        if len(set(positions)) != len(positions) or any(
            p < 1 or p > 119 for p in positions
        ):
            raise ValueError("Held-out positions must be distinct and in 1..119")


@dataclass
class GuidanceConfig:
    method: str = "tag"
    num_samples: int = 32
    batch_size: int = 4
    generator: str = "esm3-open"
    threshold: float = 4.1  # original log10 fluorescence units
    generator_temperature: float = 0.7
    predictor_temperature: float = 1.0
    min_sigma: float = 0.05
    mask_fraction: float = 1.0
    # Designable sites in biological numbering (1..119). Omit the key to mask
    # every residue; TOML has no null, so absence is how "mask everything" is
    # written. Positions not listed keep their wild-type residue.
    mask_positions: list[int] | None = None
    position: int = 7  # biological numbering, excludes BOS
    control: bool = False  # unguided arm: uninformative predictor, no checkpoint

    def __post_init__(self) -> None:
        if self.num_samples < 1 or self.batch_size < 1:
            raise ValueError("Guidance num_samples and batch_size must be positive")
        if self.method not in {"tag", "deg"}:
            raise ValueError("guidance.method must be tag or deg")
        if (
            min(self.generator_temperature, self.predictor_temperature, self.min_sigma)
            <= 0
        ):
            raise ValueError("Temperatures and min_sigma must be positive")
        if not 0 <= self.mask_fraction <= 1 or not 1 <= self.position <= 119:
            raise ValueError("Invalid guidance masking fraction or position")
        if self.mask_positions is not None:
            positions = self.mask_positions
            if not positions:
                raise ValueError(
                    "guidance.mask_positions must name at least one site; omit the "
                    "key entirely to mask every residue"
                )
            if any(isinstance(p, bool) or not isinstance(p, int) for p in positions):
                raise ValueError("guidance.mask_positions must be integers")
            if len(set(positions)) != len(positions):
                raise ValueError("guidance.mask_positions must not repeat a site")
            if not all(1 <= p <= 119 for p in positions):
                raise ValueError(
                    "guidance.mask_positions must lie in 1..119 (biological "
                    "numbering, excludes BOS)"
                )
            # Canonical order, so the saved config and library records compare.
            self.mask_positions = sorted(positions)


@dataclass
class EvaluationConfig:
    mask_fractions: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9])
    top_k: int = 100
    max_rows: int = 0  # 0 means all evaluation rows

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.max_rows < 0:
            raise ValueError(
                "Evaluation top_k must be positive and max_rows nonnegative"
            )
        if not self.mask_fractions or self.mask_fractions[0] != 0.0:
            raise ValueError("Evaluation mask_fractions must start with 0.0")
        if len(set(self.mask_fractions)) != len(self.mask_fractions) or any(
            not 0 <= fraction <= 1 for fraction in self.mask_fractions
        ):
            raise ValueError("Evaluation mask fractions must be distinct and in [0, 1]")


@dataclass
class Config:
    csv_path: Path
    output_dir: Path
    seed: int
    device: str
    train_split: str
    oracle: ModelConfig | None
    predictor: ModelConfig | None
    oracle_training: TrainingConfig | None
    predictor_training: TrainingConfig | None
    splits: SplitConfig
    guidance: GuidanceConfig
    evaluation: EvaluationConfig
    predictor_ensemble_sizes: list[int] = field(default_factory=list)
    stage: str = "benchmark"
    predictor_checkpoint: Path | None = None
    oracle_checkpoint: Path | None = None
    library_path: Path | None = None
    scoring_batch_size: int = 128
    histogram_bins: int = 30

    def __post_init__(self) -> None:
        if self.stage not in set(DEFAULT_CONFIGS) - {"splits"}:
            raise ValueError(f"Unknown config stage: {self.stage}")
        roles = {
            "benchmark": ("oracle", "predictor"),
            "train-oracle": ("oracle",),
            "train-predictor": ("predictor",),
            "guidance": (),
            "evaluate": (),
        }[self.stage]
        for role in roles:
            if getattr(self, role) is None or getattr(self, role + "_training") is None:
                raise ValueError(
                    f"{self.stage} requires {role} model and training settings"
                )
        if self.guidance.control and self.stage != "guidance":
            raise ValueError("guidance.control only applies to the guidance stage")
        if self.stage == "guidance":
            if self.guidance.control:
                # Drop the checkpoint a control run would ignore, so the saved
                # config never implies a predictor influenced the library.
                self.predictor_checkpoint = None
            elif self.predictor_checkpoint is None:
                raise ValueError(
                    "Guidance requires run.predictor_checkpoint, or guidance.control"
                )
        if self.stage == "evaluate" and any(
            path is None
            for path in (
                self.library_path,
                self.oracle_checkpoint,
                self.predictor_checkpoint,
            )
        ):
            raise ValueError(
                "Evaluate requires run.library_path, oracle_checkpoint and predictor_checkpoint"
            )
        if self.scoring_batch_size < 1 or self.histogram_bins < 1:
            raise ValueError("scoring_batch_size and histogram_bins must be positive")
        sizes = self.predictor_ensemble_sizes
        if sizes and (
            self.predictor is None
            or sizes != sorted(set(sizes))
            or sizes[0] < 1
            or sizes[-1] != self.predictor.ensemble_size
        ):
            raise ValueError(
                "predictor_ensemble_sizes must be increasing positive sizes ending "
                "at predictor.ensemble_size"
            )


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.pop("run")
    # Config files conventionally live under configs/; root can be explicit.
    root = (path.parent / run.pop("root", "..")).resolve()

    def storage_path(value: str) -> Path:
        # Preserve Linux Volume paths even when resolving configs on Windows.
        return Path(value) if PurePosixPath(value).is_absolute() else root / value

    stage = run.pop("stage")
    references = raw.pop("training_configs", {})
    if references and stage != "benchmark":
        raise ValueError("Only benchmark configs may reference training_configs")
    for role, reference in references.items():
        if role not in {"oracle", "predictor"}:
            raise ValueError(f"Unknown training config role: {role}")
        if role in raw or role + "_training" in raw:
            raise ValueError(f"Define {role} inline or by reference, not both")
        reference_path = path.parent / reference
        reference_stage = tomllib.loads(reference_path.read_text(encoding="utf-8"))[
            "run"
        ]["stage"]
        if reference_stage != f"train-{role}":
            raise ValueError(f"{role} reference must point to a train-{role} config")
        source = load_config(reference_path)
        raw[role] = asdict(getattr(source, role))
        raw[role + "_training"] = asdict(getattr(source, role + "_training"))
    guidance = raw.pop("guidance", {})
    # Silently preferring one would be a footgun: the two describe different
    # experiments (fixed design sites versus a fresh random set per draw).
    if "mask_positions" in guidance and "mask_fraction" in guidance:
        raise ValueError(
            "Set guidance.mask_positions or guidance.mask_fraction, not both"
        )
    checkpoint = run.pop("predictor_checkpoint", None)
    oracle_checkpoint = run.pop("oracle_checkpoint", None)
    library_path = run.pop("library_path", None)
    config = Config(
        csv_path=storage_path(run.pop("csv_path")),
        output_dir=storage_path(run.pop("output_dir")),
        seed=run.pop("seed", 42),
        device=run.pop("device", "cpu"),
        train_split=run.pop("train_split", "random"),
        oracle=ModelConfig(**raw.pop("oracle")) if "oracle" in raw else None,
        predictor=ModelConfig(**raw.pop("predictor")) if "predictor" in raw else None,
        oracle_training=(
            TrainingConfig(**raw.pop("oracle_training"))
            if "oracle_training" in raw
            else None
        ),
        predictor_training=(
            TrainingConfig(**raw.pop("predictor_training"))
            if "predictor_training" in raw
            else None
        ),
        splits=SplitConfig(**raw.pop("splits", {})),
        guidance=GuidanceConfig(**guidance),
        evaluation=EvaluationConfig(**raw.pop("evaluation", {})),
        predictor_ensemble_sizes=run.pop("predictor_ensemble_sizes", []),
        stage=stage,
        predictor_checkpoint=(
            storage_path(checkpoint) if checkpoint is not None else None
        ),
        oracle_checkpoint=(
            storage_path(oracle_checkpoint) if oracle_checkpoint is not None else None
        ),
        library_path=storage_path(library_path) if library_path is not None else None,
        scoring_batch_size=run.pop("scoring_batch_size", 128),
        histogram_bins=run.pop("histogram_bins", 30),
    )
    if raw or run:
        raise ValueError(
            f"Unknown config keys: {raw.keys()}, {run.keys()}. Use [evaluation], not the removed [benchmark] section."
        )
    if config.train_split not in {"random", "distance", "activity", "position"}:
        raise ValueError("Unknown train_split")
    return config


def validate_stage_config(stage: str, cfg: Config) -> None:
    if stage != "splits" and stage != cfg.stage:
        raise ValueError(f"Stage {stage!r} cannot use a {cfg.stage!r} config")


def guidance_overrides(
    cfg: Config, num_samples: int | None, method: str | None, control: bool = False
) -> Config:
    """Apply CLI overrides once so the saved config records the actual request."""
    if num_samples is None and method is None and not control:
        return cfg
    if cfg.stage != "guidance":
        raise ValueError(
            "--num-samples, --method and --control are only used by guidance"
        )
    # Config.__post_init__ drops the now-unused predictor checkpoint.
    return replace(
        cfg,
        guidance=replace(
            cfg.guidance,
            num_samples=(
                cfg.guidance.num_samples if num_samples is None else num_samples
            ),
            method=cfg.guidance.method if method is None else method,
            control=cfg.guidance.control or control,
        ),
    )


def serialize_config(cfg: Config) -> str:
    """Freeze resolved references before sending a config to Modal."""
    return json.dumps(asdict(cfg), default=lambda path: path.as_posix())


def deserialize_config(text: str) -> Config:
    raw = json.loads(text)
    for key in (
        "csv_path",
        "output_dir",
        "predictor_checkpoint",
        "oracle_checkpoint",
        "library_path",
    ):
        if raw[key] is not None:
            raw[key] = Path(raw[key])
    for key, cls in (
        ("oracle", ModelConfig),
        ("predictor", ModelConfig),
        ("oracle_training", TrainingConfig),
        ("predictor_training", TrainingConfig),
        ("splits", SplitConfig),
        ("guidance", GuidanceConfig),
        ("evaluation", EvaluationConfig),
    ):
        if raw[key] is not None:
            raw[key] = cls(**raw[key])
    return Config(**raw)


def default_run_id(cfg: Config, library_path: str = "") -> str:
    """Run ID a config asks for, so the Volume layout follows its TOML.

    A run ID is the output directory below ``/runs``, which lets related runs
    nest (``guidance/15-positions/tag-v1``) and keeps ``creilov-runs`` laid out
    exactly like ``outputs/``. Evaluation has no folder of its own: it joins the
    guidance run holding the library it scores. A config aimed somewhere other
    than ``/runs`` names no Volume run, so such a launch needs an explicit
    ``--run-id``.
    """
    from .reports import run_id_of, volume_run_id

    if cfg.stage == "evaluate":
        return volume_run_id(library_path or cfg.library_path.as_posix())
    return run_id_of(cfg.output_dir.as_posix())


def stage_folder(cfg: Config, library_path: str = "") -> str:
    """First segment of the config's run ID: the stage folder below ``/runs``.

    Taken from the config's own ``output_dir`` (or, for evaluate, the library it
    scores) rather than from ``cfg.stage``, because the two differ: ``splits``
    shares the benchmark config and so writes under ``benchmark/``, and
    ``evaluate`` has no folder of its own and joins a guidance run. Empty when
    the config is aimed outside ``/runs``, which names no stage folder.
    """
    try:
        return default_run_id(cfg, library_path).split("/", 1)[0]
    except (ValueError, IndexError):
        return ""


def qualify_run_id(cfg: Config, run_id: str, library_path: str = "") -> str:
    """Resolve a ``--run-id`` against its stage folder.

    ``/runs`` mirrors ``outputs/``, where every run sits under a stage folder, so
    a run ID that already starts with that folder is returned unchanged and fully
    qualified IDs keep working. One that does not is read as relative to it, so
    ``--run-id 15-positions/tag-v1`` on a guidance config means
    ``guidance/15-positions/tag-v1`` instead of a stray top-level folder. A
    config with no stage folder to borrow leaves its run ID alone.
    """
    prefix = stage_folder(cfg, library_path)
    if not run_id or not prefix:
        return run_id
    if run_id == prefix or run_id.startswith(f"{prefix}/"):
        return run_id
    return f"{prefix}/{run_id}"


def modal_config(
    cfg: Config,
    run_id: str,
    predictor_checkpoint: str,
    oracle_checkpoint: str = "",
    library_path: str = "",
) -> Config:
    """Give each stage its own output and an explicit Volume checkpoint input."""
    from pathlib import PurePosixPath

    if cfg.stage == "guidance" and cfg.guidance.control:
        if predictor_checkpoint:
            raise ValueError("--predictor-checkpoint is unused by a control run")
        checkpoint = None
    elif cfg.stage in {"guidance", "evaluate"}:
        checkpoint = PurePosixPath(
            predictor_checkpoint or cfg.predictor_checkpoint.as_posix()
        )
        if not checkpoint.is_relative_to("/runs") or ".." in checkpoint.parts:
            raise ValueError(
                "Modal guidance/evaluate requires --predictor-checkpoint /runs/<predictor-run>/predictor/model.pt"
            )
        checkpoint = Path(str(checkpoint))
    else:
        if predictor_checkpoint:
            raise ValueError("--predictor-checkpoint is only used by guidance/evaluate")
        checkpoint = None
    evaluation_paths = {}
    if cfg.stage == "evaluate":
        for name, value in (
            (
                "oracle_checkpoint",
                oracle_checkpoint or cfg.oracle_checkpoint.as_posix(),
            ),
            ("library_path", library_path or cfg.library_path.as_posix()),
        ):
            remote = PurePosixPath(value)
            if not remote.is_relative_to("/runs") or ".." in remote.parts:
                raise ValueError(
                    f"Modal evaluate requires --{name.replace('_', '-')} /runs/..."
                )
            evaluation_paths[name] = Path(str(remote))
    elif oracle_checkpoint or library_path:
        raise ValueError(
            "--oracle-checkpoint and --library-path are only used by evaluate"
        )
    return replace(
        cfg,
        csv_path=Path("/data") / cfg.csv_path.name,
        output_dir=Path("/runs") / run_id,
        device="cuda",
        predictor_checkpoint=checkpoint,
        **evaluation_paths,
    )
