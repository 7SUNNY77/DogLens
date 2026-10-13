"""Checks of the real exported bundle. Run with: uv run pytest -m integration

Skipped (not passed) when artifacts/model_bundle has not been exported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from doglens.bundle import load_bundle
from doglens.catalog import load_catalog, load_class_mapping, mapping_sha256
from doglens.preprocessing import load_spec

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "model_bundle"


@pytest.fixture(scope="module")
def bundle():
    if not (BUNDLE / "weights.pt").is_file():
        pytest.skip("artifacts/model_bundle not exported")
    return load_bundle(BUNDLE, "cpu")


def test_bundle_consistent_with_repository(bundle) -> None:
    assert bundle.metadata["num_classes"] == 120
    repo_classes = load_class_mapping(REPO / "data" / "prepared" / "class_mapping.json")
    assert mapping_sha256(bundle.classes) == mapping_sha256(repo_classes)
    digest, _ = load_catalog(REPO / "assets" / "breeds.json")
    assert digest == bundle.metadata["class_mapping_sha256"]
    assert load_spec(REPO / "assets" / "preprocessing.json") == bundle.spec
    assert bundle.metadata["source_checkpoint"]["selection_rule"].startswith("max validation")


def test_bundle_cpu_inference_probabilities(bundle) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 3, bundle.spec.crop_size, bundle.spec.crop_size)
    with torch.inference_mode():
        probs = torch.softmax(bundle.model(x), dim=1).numpy()
    assert probs.shape == (3, 120) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_bundle_metrics_status(bundle) -> None:
    metrics = json.loads((BUNDLE / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_version"] == bundle.metadata["model_version"]
    assert 0.0 <= metrics["validation"]["top1"] <= 1.0
    assert metrics["test"]["status"] in {"not_evaluated", "evaluated"}
