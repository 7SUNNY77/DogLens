"""Predictor on a tiny real bundle (random-init weights, exported via the release path)."""

from __future__ import annotations

import json
import shutil
import socket
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from doglens.bundle import BundleError, load_bundle
from doglens.errors import RoiError
from doglens.predictor import PredictionError, Predictor, get_predictor, rank_scores
from doglens.preprocessing import preprocess
from doglens.roi import crop_model_region, expand_box

REPO = Path(__file__).resolve().parents[1]
ROI = (40, 30, 200, 180)


@pytest.fixture(scope="module")
def predictor(tiny_bundle) -> Predictor:
    return Predictor.from_bundle_dir(tiny_bundle["bundle"])


def _image(size=(320, 240)) -> Image.Image:
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def test_rank_scores_ties_by_class_id() -> None:
    assert rank_scores(np.array([0.1, 0.3, 0.3, 0.3]), 3) == [1, 2, 3]
    assert rank_scores(np.array([0.5, 0.2, 0.3]), 3) == [0, 2, 1]


def test_top_k_order_finiteness_and_full_distribution(predictor: Predictor) -> None:
    result = predictor.predict(_image(), roi=ROI, top_k=3)
    scores = [p.score for p in result.predictions]
    assert len(scores) == 3 and all(np.isfinite(scores))
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) < 1.0  # top-3 is not renormalised
    everything = predictor.predict(_image(), roi=ROI, top_k=120)
    assert sum(p.score for p in everything.predictions) == pytest.approx(1.0, abs=1e-6)
    assert len({p.class_id for p in everything.predictions}) == 120
    assert [p.score for p in everything.predictions[:3]] == pytest.approx(scores, abs=1e-12)
    top = result.predictions[0]
    assert top.name_en and top.name_ru == top.name_en  # no verified Russian names -> fallback
    assert result.model_version == predictor.model_version
    assert set(result.timings_ms) == {"preprocessing", "model", "postprocessing", "total"}


@pytest.mark.parametrize("top_k", [0, 121, -1, True, 2.0, "3"])
def test_invalid_top_k(predictor: Predictor, top_k) -> None:
    with pytest.raises(PredictionError):
        predictor.predict(_image(), top_k=top_k)


def test_matches_direct_forward_on_the_same_input(predictor: Predictor, tiny_bundle) -> None:
    img = _image()
    result = predictor.predict(img, roi=ROI, top_k=5)
    crop, region = crop_model_region(img, ROI, 0.1)
    bundle = load_bundle(tiny_bundle["bundle"])
    with torch.inference_mode():
        logits = bundle.model(preprocess(crop, bundle.spec).unsqueeze(0))[0].double()
    probs = torch.softmax(logits, 0).numpy()
    for p in result.predictions:
        assert p.score == pytest.approx(probs[p.class_id], abs=1e-6)
    assert result.crop_box == region.crop_box


def test_context_added_exactly_once_and_preview_matches(predictor: Predictor) -> None:
    img = _image()
    result = predictor.predict(img, roi=ROI)
    once = expand_box(ROI, img.size, 0.1)
    twice = expand_box(once, img.size, 0.1)
    assert result.user_roi == ROI and result.crop_box == once != twice
    preview_crop, preview_region = predictor.prepare(img, ROI)  # what the UI shows
    assert preview_region.crop_box == result.crop_box
    assert preview_crop.size == (once[2] - once[0], once[3] - once[1])
    assert predictor.context == 0.1


def test_whole_image_is_not_expanded(predictor: Predictor) -> None:
    result = predictor.predict(_image(), roi=None)
    assert result.user_roi is None and result.crop_box == (0, 0, 320, 240)
    assert any("всё изображение" in w for w in result.warnings)


def test_invalid_inputs(predictor: Predictor) -> None:
    with pytest.raises(PredictionError):
        predictor.predict(Image.new("L", (100, 100)))
    with pytest.raises(RoiError):
        predictor.predict(_image(), roi=(0, 0, 30, 30))
    with pytest.raises(RoiError):
        predictor.predict(_image(), roi=(0, 0, 400, 100))


def test_non_finite_output_is_an_error(tiny_bundle) -> None:
    broken = Predictor.from_bundle_dir(tiny_bundle["bundle"])
    with torch.no_grad():
        broken._bundle.model.net.classifier[1].bias.fill_(float("nan"))
    with pytest.raises(PredictionError) as info:
        broken.predict(_image())
    assert info.value.code == "invalid_output"


def test_missing_and_corrupted_bundles(tiny_bundle, tmp_path) -> None:
    with pytest.raises(BundleError) as info:
        Predictor.from_bundle_dir(tmp_path / "absent")
    assert "artifacts/model_bundle" in info.value.user_message
    damaged = tmp_path / "damaged"
    shutil.copytree(tiny_bundle["bundle"], damaged)
    with open(damaged / "weights.pt", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(BundleError, match="checksum"):
        Predictor.from_bundle_dir(damaged)


def test_catalog_must_match_bundle(tiny_bundle, tmp_path) -> None:
    catalog = json.loads((REPO / "assets" / "breeds.json").read_text(encoding="utf-8"))
    catalog["mapping_sha256"] = "0" * 64
    bad = tmp_path / "breeds.json"
    bad.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(BundleError, match="catalog"):
        Predictor.from_bundle_dir(tiny_bundle["bundle"], bad)


def test_no_network_during_load_and_predict(tiny_bundle, monkeypatch) -> None:
    def no_network(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(torch.hub, "download_url_to_file", no_network)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_network)
    fresh = Predictor.from_bundle_dir(tiny_bundle["bundle"])
    assert fresh.predict(_image()).predictions


def test_concurrent_predictions_are_consistent_and_stateless(predictor: Predictor) -> None:
    img = _image()
    before = set(vars(predictor))
    expected = [p.score for p in predictor.predict(img, roi=ROI).predictions]
    results, errors = [], []

    def worker() -> None:
        try:
            results.append([p.score for p in predictor.predict(img, roi=ROI).predictions])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 6
    assert all(r == pytest.approx(expected, abs=1e-6) for r in results)
    assert set(vars(predictor)) == before  # no per-request state is attached


def test_get_predictor_caches_one_instance(tiny_bundle) -> None:
    first = get_predictor(tiny_bundle["bundle"])
    assert get_predictor(tiny_bundle["bundle"]) is first


def test_logs_contain_no_image_data(predictor: Predictor, caplog) -> None:
    with caplog.at_level("DEBUG"):
        predictor.predict(_image(), roi=ROI)
    assert "prediction request=" in caplog.text
    assert all(len(r.getMessage()) < 400 for r in caplog.records)
