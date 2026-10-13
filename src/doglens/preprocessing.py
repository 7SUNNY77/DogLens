"""The single deterministic preprocessing used for training, validation, test and the app.

Parameters are read once from the torchvision weights metadata and then stored explicitly
(``assets/preprocessing.json``, checkpoint and bundle metadata), so a package upgrade that
changes defaults cannot silently change the model input. Random augmentation is *not* part of
this pipeline; train-only augmentation is applied to the PIL crop before it (see dataset.py).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F  # noqa: N812 - torchvision convention
from PIL import Image
from torchvision.transforms import InterpolationMode

from doglens.errors import ConfigError

PREPROCESSING_SCHEMA_VERSION = 1
# Exactly the order of torchvision.transforms._presets.ImageClassification.forward.
OPERATION_ORDER = (
    "resize_shorter_side(resize_size, interpolation, antialias) on the PIL image",
    "center_crop(crop_size)",
    "pil_to_tensor (uint8, CHW)",
    "convert_image_dtype(float32) -> [0, 1]",
    "normalize(mean, std)",
)
_INTERPOLATIONS = {m.value: m for m in InterpolationMode}


@dataclass(frozen=True)
class PreprocessingSpec:
    resize_size: int
    crop_size: int
    interpolation: str
    antialias: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str
    order: tuple[str, ...] = OPERATION_ORDER
    schema_version: int = PREPROCESSING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREPROCESSING_SCHEMA_VERSION:
            raise ConfigError(f"unsupported preprocessing schema {self.schema_version}")
        if tuple(self.order) != OPERATION_ORDER:
            raise ConfigError("preprocessing operation order differs from the implemented one")
        if self.interpolation not in _INTERPOLATIONS:
            raise ConfigError(f"unknown interpolation {self.interpolation!r}")
        if not (0 < self.crop_size <= self.resize_size):
            raise ConfigError("need 0 < crop_size <= resize_size")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigError("mean/std must have 3 channels")
        if not all(math.isfinite(v) for v in (*self.mean, *self.std)) or min(self.std) <= 0:
            raise ConfigError("mean/std must be finite and std > 0")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"], d["std"], d["order"] = list(self.mean), list(self.std), list(self.order)
        return d


def spec_from_dict(raw: dict[str, Any]) -> PreprocessingSpec:
    try:
        return PreprocessingSpec(
            resize_size=int(raw["resize_size"]),
            crop_size=int(raw["crop_size"]),
            interpolation=str(raw["interpolation"]),
            antialias=bool(raw["antialias"]),
            mean=tuple(float(v) for v in raw["mean"]),  # type: ignore[arg-type]
            std=tuple(float(v) for v in raw["std"]),  # type: ignore[arg-type]
            source=str(raw["source"]),
            order=tuple(raw["order"]),
            schema_version=int(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"malformed preprocessing spec: {exc}") from exc


def spec_from_weights(weights_enum_name: str = "EfficientNet_B0_Weights.IMAGENET1K_V1"):
    """Read the parameters from torchvision weights metadata (no download involved)."""
    import torchvision.models as tvm

    enum_name, member = weights_enum_name.split(".")
    weights = getattr(getattr(tvm, enum_name), member)
    t = weights.transforms()

    def one(value: Any) -> int:
        values = list(value) if isinstance(value, list | tuple) else [value]
        if len(values) != 1:
            raise ConfigError(f"expected a single size, got {value!r}")
        return int(values[0])

    return PreprocessingSpec(
        resize_size=one(t.resize_size),
        crop_size=one(t.crop_size),
        interpolation=t.interpolation.value,
        antialias=bool(t.antialias),
        mean=tuple(float(v) for v in t.mean),  # type: ignore[arg-type]
        std=tuple(float(v) for v in t.std),  # type: ignore[arg-type]
        source=f"torchvision.models.{weights_enum_name}.transforms()",
    )


def load_spec(path: str | Path) -> PreprocessingSpec:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"preprocessing spec not found: {path}")
    return spec_from_dict(json.loads(path.read_text(encoding="utf-8")))


def preprocess(image: Image.Image, spec: PreprocessingSpec) -> torch.Tensor:
    """RGB PIL image -> float32 tensor [3, crop, crop]. Deterministic; no randomness."""
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("preprocess expects an RGB PIL image")
    img = F.resize(
        image,
        [spec.resize_size],
        interpolation=_INTERPOLATIONS[spec.interpolation],
        antialias=spec.antialias,
    )
    img = F.center_crop(img, [spec.crop_size])
    tensor = F.pil_to_tensor(img)
    tensor = F.convert_image_dtype(tensor, torch.float)
    return F.normalize(tensor, mean=list(spec.mean), std=list(spec.std))
