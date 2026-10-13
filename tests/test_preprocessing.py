"""Shared deterministic preprocessing: parameters, equivalence with torchvision, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from doglens.errors import ConfigError
from doglens.preprocessing import (
    PreprocessingSpec,
    load_spec,
    preprocess,
    spec_from_dict,
    spec_from_weights,
)

REPO = Path(__file__).resolve().parents[1]


def _noise_image(size=(317, 211), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def test_spec_from_weights_values() -> None:
    spec = spec_from_weights()
    assert (spec.resize_size, spec.crop_size) == (256, 224)
    assert spec.interpolation == "bicubic" and spec.antialias is True
    assert spec.mean == (0.485, 0.456, 0.406) and spec.std == (0.229, 0.224, 0.225)


def test_repository_artifact_matches_installed_torchvision() -> None:
    assert load_spec(REPO / "assets" / "preprocessing.json") == spec_from_weights()


def test_output_shape_dtype_and_equivalence_with_torchvision() -> None:
    spec = spec_from_weights()
    for size in [(317, 211), (224, 224), (100, 400), (1000, 64)]:
        img = _noise_image(size)
        ours = preprocess(img, spec)
        assert ours.shape == (3, 224, 224) and ours.dtype == torch.float32
        reference = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()(img)
        assert torch.equal(ours, reference)


def test_preprocess_is_deterministic() -> None:
    spec = spec_from_weights()
    img = _noise_image()
    torch.manual_seed(0)
    a = preprocess(img, spec)
    torch.manual_seed(123)
    b = preprocess(img, spec)
    assert torch.equal(a, b)


def test_preprocess_rejects_non_rgb() -> None:
    with pytest.raises(ValueError):
        preprocess(Image.new("L", (50, 50)), spec_from_weights())


def test_spec_round_trip_and_validation(tmp_path: Path) -> None:
    spec = spec_from_weights()
    assert spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    bad = spec.to_dict() | {"interpolation": "magic"}
    with pytest.raises(ConfigError):
        spec_from_dict(bad)
    with pytest.raises(ConfigError):
        spec_from_dict(spec.to_dict() | {"order": ["normalize"]})
    with pytest.raises(ConfigError):
        spec_from_dict(spec.to_dict() | {"crop_size": 300})
    with pytest.raises(ConfigError):
        spec_from_dict({"resize_size": 256})
    with pytest.raises(ConfigError):
        load_spec(tmp_path / "absent.json")
    with pytest.raises(ConfigError):
        PreprocessingSpec(256, 224, "bicubic", True, (0, 0, 0), (1, 0, 1), "x")
