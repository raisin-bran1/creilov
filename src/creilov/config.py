"""Typed TOML configuration; relative paths resolve from the project root."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


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
    generator: str = "esm3-open"
    threshold: float = 4.1  # original log10 fluorescence units
    generator_temperature: float = 0.7
    predictor_temperature: float = 1.0
    min_sigma: float = 0.05
    mask_fraction: float = 1.0
    position: int = 7  # biological numbering, excludes BOS

    def __post_init__(self) -> None:
        if self.method not in {"tag", "deg"}:
            raise ValueError("guidance.method must be tag or deg")
        if (
            min(self.generator_temperature, self.predictor_temperature, self.min_sigma)
            <= 0
        ):
            raise ValueError("Temperatures and min_sigma must be positive")
        if not 0 <= self.mask_fraction <= 1 or not 1 <= self.position <= 119:
            raise ValueError("Invalid guidance masking fraction or position")


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
    oracle: ModelConfig
    predictor: ModelConfig
    oracle_training: TrainingConfig
    predictor_training: TrainingConfig
    splits: SplitConfig
    guidance: GuidanceConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.pop("run")
    # Config files conventionally live under configs/; root can be explicit.
    root = (path.parent / run.pop("root", "..")).resolve()
    config = Config(
        csv_path=root / run.pop("csv_path"),
        output_dir=root / run.pop("output_dir"),
        seed=run.pop("seed", 42),
        device=run.pop("device", "cpu"),
        train_split=run.pop("train_split", "random"),
        oracle=ModelConfig(**raw.pop("oracle")),
        predictor=ModelConfig(**raw.pop("predictor")),
        oracle_training=TrainingConfig(**raw.pop("oracle_training")),
        predictor_training=TrainingConfig(**raw.pop("predictor_training")),
        splits=SplitConfig(**raw.pop("splits", {})),
        guidance=GuidanceConfig(**raw.pop("guidance", {})),
        evaluation=EvaluationConfig(**raw.pop("evaluation", {})),
    )
    if raw or run:
        raise ValueError(
            f"Unknown config keys: {raw.keys()}, {run.keys()}. Use [evaluation], not the removed [benchmark] section."
        )
    if config.train_split not in {"random", "distance", "activity", "position"}:
        raise ValueError("Unknown train_split")
    return config
