"""Tests for training configuration loading and shared schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from doglens.config import (
    DEFAULT_LIMITS,
    TrainingConfig,
    config_from_dict,
    load_training_config,
)
from doglens.errors import ConfigError
from doglens.schemas import MANIFEST_COLUMNS, Prediction, PredictionResult, Verdict

REPO = Path(__file__).resolve().parents[1]


def test_repository_config_matches_documented_start_values() -> None:
    cfg = load_training_config(REPO / "configs" / "training.yaml")
    assert cfg.seed == 42
    assert cfg.loader.batch_size == 16 and cfg.loader.num_workers == 0
    assert cfg.precision == "float32"
    assert (cfg.stage_a.epochs, cfg.stage_a.lr) == (3, 0.001)
    assert (cfg.stage_b.epochs, cfg.stage_b.lr) == (5, 0.0001)
    assert cfg.optimizer.name == "AdamW" and cfg.optimizer.weight_decay == 0.0001
    assert cfg.early_stopping_patience == 3
    assert cfg.data.val_fraction == 0.2 and cfg.data.bbox_context == 0.1
    assert cfg.model.num_classes == 120
    assert cfg.stage_b.unfreeze_blocks == ("features.6", "features.7", "features.8")


def test_to_dict_round_trip() -> None:
    cfg = load_training_config(REPO / "configs" / "training.yaml")
    assert config_from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict()))) == cfg


def test_empty_config_uses_defaults() -> None:
    assert config_from_dict({}) == TrainingConfig()


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"loader": {"batch_size": 0}},
        {"loader": {"batch_size": True}},
        {"loader": {"bogus": 1}},
        {"device": "xpu"},
        {"precision": "float16"},
        {"model": {"num_classes": 10}},
        {"model": {"architecture": "resnet50"}},
        {"stage_a": {"lr": 0}},
        {"stage_a": {"lr": float("nan")}},
        {"stage_b": {"unfreeze_blocks": "features.8"}},
        {"stage_b": {"unfreeze_blocks": []}},
        {"stage_b": {"unfreeze_blocks": ["features.8", "features.8"]}},
        {"stage_b": {"batchnorm_policy": "whatever"}},
        {"data": {"val_fraction": 1.0}},
        {"early_stopping_patience": 0},
        {"optimizer": {"name": "SGD"}},
        [],
    ],
)
def test_invalid_config_is_rejected(raw) -> None:
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_training_config(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_training_config(bad)


def test_image_limits_match_project_requirements() -> None:
    assert DEFAULT_LIMITS.max_bytes == 10 * 1024 * 1024
    assert DEFAULT_LIMITS.max_pixels == 20_000_000
    assert DEFAULT_LIMITS.min_roi_side == 64


def test_verdict_enum_is_exact() -> None:
    assert {v.value for v in Verdict} == {
        "looks_correct",
        "looks_wrong",
        "unknown",
        "mixed_breed",
        "breed_not_supported",
    }


def test_manifest_columns_order() -> None:
    assert MANIFEST_COLUMNS == (
        "image_id",
        "relative_path",
        "class_id",
        "breed_key",
        "split",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "dup_with_test",
        "label_conflict",
        "n_boxes",
    )


def test_prediction_result_to_dict() -> None:
    result = PredictionResult(
        request_id="r1",
        model_version="v0",
        predictions=(Prediction(3, "beagle", "Бигль", "Beagle", 0.5),),
        timings_ms={"total": 1.0},
        warnings=("w",),
    )
    d = result.to_dict()
    assert d["predictions"][0] == {
        "class_id": 3,
        "breed_key": "beagle",
        "name_ru": "Бигль",
        "name_en": "Beagle",
        "score": 0.5,
    }
    assert d["warnings"] == ["w"]
