"""Real bundle: predictor equivalence with the evaluation pipeline and the app end to end.

Run with: uv run pytest -m integration (skipped when bundle/dataset/evaluation are absent).
Stanford Dogs images are used only to check pipeline equivalence, never as external validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from doglens.image_io import load_image
from doglens.predictor import Predictor

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "artifacts" / "model_bundle"
IMAGES = REPO / "data" / "images" / "Images"
PREDICTIONS = REPO / "reports" / "evaluation" / "test" / "predictions.csv"


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    if not (BUNDLE / "weights.pt").is_file():
        pytest.skip("artifacts/model_bundle not exported")
    return Predictor.from_bundle_dir(BUNDLE)


def test_predictor_reproduces_evaluation_predictions(predictor: Predictor) -> None:
    if not (IMAGES.is_dir() and PREDICTIONS.is_file()):
        pytest.skip("dataset images or saved test predictions absent")
    manifest = pd.read_csv(REPO / "data" / "prepared" / "manifest.csv")
    saved = pd.read_csv(PREDICTIONS).set_index("image_id")
    test = manifest[manifest["split"] == "test"].sort_values("image_id")
    big = test[
        (test.bbox_xmax - test.bbox_xmin >= 64) & (test.bbox_ymax - test.bbox_ymin >= 64)
    ].iloc[::400]
    assert len(big) >= 15
    for row in big.itertuples():
        image = load_image((IMAGES / row.relative_path).read_bytes()).image
        roi = (row.bbox_xmin, row.bbox_ymin, row.bbox_xmax, row.bbox_ymax)
        result = predictor.predict(image, roi=roi, top_k=3)
        expected = saved.loc[row.image_id]
        got = [p.class_id for p in result.predictions]
        assert got == [expected[f"top{i}_class_id"] for i in (1, 2, 3)], row.image_id
        assert result.predictions[0].score == pytest.approx(expected["top1_score"], abs=1e-5)


def test_whole_image_and_metadata(predictor: Predictor) -> None:
    if not IMAGES.is_dir():
        pytest.skip("dataset images absent")
    path = next(iter(sorted((IMAGES / "n02085620-Chihuahua").glob("*.jpg"))))
    image = load_image(path.read_bytes()).image
    result = predictor.predict(image)
    assert result.crop_box == (0, 0, *image.size) and result.user_roi is None
    assert predictor.context == 0.1
    assert predictor.metadata["preprocessing"]["crop_size"] == 224


def test_app_end_to_end_with_real_bundle(predictor: Predictor, tmp_path, monkeypatch) -> None:
    if not IMAGES.is_dir():
        pytest.skip("dataset images absent")
    monkeypatch.setenv("DOGLENS_BUNDLE_DIR", str(BUNDLE))
    monkeypatch.setenv("DOGLENS_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    data = (IMAGES / "n02085620-Chihuahua" / "n02085620_10074.jpg").read_bytes()
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.run()
    at.file_uploader[0].upload("dog.jpg", data, "image/jpeg").run()
    key = at.session_state["upload_digest"][:12]
    at.toggle(key=f"whole_{key}").set_value(False).run()
    # Preview of a 333x500 photo is 333x500 (<= 640): select the annotated dog box.
    at.slider(key=f"x_{key}").set_value((25, 277)).run()
    at.slider(key=f"y_{key}").set_value((10, 499)).run()
    at.button(key="predict").click().run()
    assert not at.exception
    result = at.session_state["result"]
    assert result.user_roi == (25, 10, 277, 499)
    assert result.predictions[0].breed_key == "chihuahua"
