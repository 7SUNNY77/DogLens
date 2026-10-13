"""Project limits and strictly validated training configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from doglens.errors import ConfigError

NUM_CLASSES = 120
MIB = 1024 * 1024


@dataclass(frozen=True)
class ImageLimits:
    """Upload limits from CLAUDE.md section 6."""

    max_bytes: int = 10 * MIB
    max_pixels: int = 20_000_000
    min_roi_side: int = 64
    # Allowed overshoot (in displayed pixels) when a preview selection touches the border.
    display_tolerance_px: float = 1.0


DEFAULT_LIMITS = ImageLimits()

ALLOWED_DEVICES = ("auto", "cpu", "cuda", "mps")
ALLOWED_PRECISIONS = ("float32",)
ALLOWED_OPTIMIZERS = ("AdamW",)
ALLOWED_BN_POLICIES = ("frozen", "train_unfrozen")


@dataclass(frozen=True)
class DataConfig:
    manifest: str = "data/prepared/manifest.csv"
    image_root: str = "data/images/Images"
    val_fraction: float = 0.2
    bbox_context: float = 0.1


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "efficientnet_b0"
    pretrained_weights: str = "IMAGENET1K_V1"
    num_classes: int = NUM_CLASSES


@dataclass(frozen=True)
class PretrainedConfig:
    """Local ImageNet weights used only to initialise training (never by the app)."""

    path: str = "data/weights/efficientnet_b0_rwightman-7f5810bc.pth"
    sha256: str = "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934"
    weights_enum: str = "EfficientNet_B0_Weights.IMAGENET1K_V1"


@dataclass(frozen=True)
class AugmentConfig:
    """Train-only, applied to the PIL crop before the shared deterministic preprocessing."""

    horizontal_flip: bool = True


@dataclass(frozen=True)
class SmokeConfig:
    per_class: int = 2
    train_batches: int = 15
    overfit_batch_size: int = 16
    overfit_steps: int = 40
    overfit_lr: float = 0.01
    overfit_max_loss_ratio: float = 0.5
    finetune_steps: int = 2


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "AdamW"
    weight_decay: float = 0.0001


@dataclass(frozen=True)
class StageAConfig:
    epochs: int = 3
    lr: float = 0.001


@dataclass(frozen=True)
class StageBConfig:
    epochs: int = 5
    lr: float = 0.0001
    unfreeze_blocks: tuple[str, ...] = ("features.6", "features.7", "features.8")
    batchnorm_policy: str = "frozen"


@dataclass(frozen=True)
class OutputConfig:
    run_dir: str = "artifacts/runs"
    bundle_dir: str = "artifacts/model_bundle"
    smoke_dir: str = "artifacts/smoke"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    precision: str = "float32"
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    early_stopping_patience: int = 3
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage_b"]["unfreeze_blocks"] = list(self.stage_b.unfreeze_blocks)
        return d


def _take(section: dict[str, Any], cls: type, name: str) -> Any:
    if not isinstance(section, dict):
        raise ConfigError(f"section '{name}' must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {sorted(unknown)}")
    return cls(**section)


def _check_int(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{name}' must be an integer >= {minimum}, got {value!r}")


def _check_positive_float(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{name}' must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"'{name}' must be finite and > 0, got {value!r}")


def validate_config(cfg: TrainingConfig) -> TrainingConfig:
    """Raise ConfigError on any out-of-range or unsupported value."""
    _check_int(cfg.seed, "seed", 0)
    if cfg.device not in ALLOWED_DEVICES:
        raise ConfigError(f"'device' must be one of {ALLOWED_DEVICES}, got {cfg.device!r}")
    if cfg.precision not in ALLOWED_PRECISIONS:
        raise ConfigError(f"'precision' must be one of {ALLOWED_PRECISIONS}")
    if not 0.0 < float(cfg.data.val_fraction) < 1.0:
        raise ConfigError("'data.val_fraction' must be in (0, 1)")
    if not 0.0 <= float(cfg.data.bbox_context) <= 1.0:
        raise ConfigError("'data.bbox_context' must be in [0, 1]")
    _check_int(cfg.loader.batch_size, "loader.batch_size", 1)
    _check_int(cfg.loader.num_workers, "loader.num_workers", 0)
    if cfg.model.architecture != "efficientnet_b0":
        raise ConfigError("only 'efficientnet_b0' is supported")
    if cfg.model.num_classes != NUM_CLASSES:
        raise ConfigError(f"'model.num_classes' must be {NUM_CLASSES}")
    if cfg.optimizer.name not in ALLOWED_OPTIMIZERS:
        raise ConfigError(f"'optimizer.name' must be one of {ALLOWED_OPTIMIZERS}")
    if isinstance(cfg.optimizer.weight_decay, bool) or cfg.optimizer.weight_decay < 0:
        raise ConfigError("'optimizer.weight_decay' must be >= 0")
    _check_int(cfg.stage_a.epochs, "stage_a.epochs", 0)
    _check_positive_float(cfg.stage_a.lr, "stage_a.lr")
    _check_int(cfg.stage_b.epochs, "stage_b.epochs", 0)
    _check_positive_float(cfg.stage_b.lr, "stage_b.lr")
    if cfg.stage_b.batchnorm_policy not in ALLOWED_BN_POLICIES:
        raise ConfigError(f"'stage_b.batchnorm_policy' must be one of {ALLOWED_BN_POLICIES}")
    blocks = cfg.stage_b.unfreeze_blocks
    if cfg.stage_b.epochs > 0 and not blocks:
        raise ConfigError("'stage_b.unfreeze_blocks' must list blocks when stage_b.epochs > 0")
    if len(set(blocks)) != len(blocks) or not all(isinstance(b, str) and b for b in blocks):
        raise ConfigError("'stage_b.unfreeze_blocks' must be unique non-empty strings")
    _check_int(cfg.early_stopping_patience, "early_stopping_patience", 1)
    sha = cfg.pretrained.sha256
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        raise ConfigError("'pretrained.sha256' must be a lower-case hex SHA-256")
    if not isinstance(cfg.augment.horizontal_flip, bool):
        raise ConfigError("'augment.horizontal_flip' must be a boolean")
    sm = cfg.smoke
    for name in ("per_class", "train_batches", "overfit_batch_size", "overfit_steps"):
        _check_int(getattr(sm, name), f"smoke.{name}", 1)
    _check_int(sm.finetune_steps, "smoke.finetune_steps", 0)
    _check_positive_float(sm.overfit_lr, "smoke.overfit_lr")
    if not 0 < sm.overfit_max_loss_ratio < 1:
        raise ConfigError("'smoke.overfit_max_loss_ratio' must be in (0, 1)")
    return cfg


def config_from_dict(raw: dict[str, Any]) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    sections = {
        "data": DataConfig,
        "loader": LoaderConfig,
        "model": ModelConfig,
        "optimizer": OptimizerConfig,
        "stage_a": StageAConfig,
        "stage_b": StageBConfig,
        "output": OutputConfig,
        "pretrained": PretrainedConfig,
        "augment": AugmentConfig,
        "smoke": SmokeConfig,
    }
    allowed = set(TrainingConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in sections:
            if key == "stage_b" and isinstance(value, dict) and "unfreeze_blocks" in value:
                blocks = value["unfreeze_blocks"]
                if not isinstance(blocks, list):
                    raise ConfigError("'stage_b.unfreeze_blocks' must be a list")
                value = {**value, "unfreeze_blocks": tuple(blocks)}
            kwargs[key] = _take(value, sections[key], key)
        else:
            kwargs[key] = value
    return validate_config(TrainingConfig(**kwargs))


def load_training_config(path: str | Path) -> TrainingConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return config_from_dict(raw or {})
