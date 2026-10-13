"""Streamlit app through streamlit.testing.v1.AppTest.

Uses the real app code with a tiny random-init bundle exported via the release path (no mock
predictor) and a temporary feedback DB. Interactive drag-selection is not part of the app (the
ROI is chosen with standard sliders), so everything below runs headless.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from doglens.app_texts import INTRO, SCORE_CAPTION
from doglens.schemas import Verdict

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _png(color=(120, 80, 40), size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.paste((240, 240, 240), (100, 80, 300, 220))
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tiny_bundle, tmp_path, monkeypatch) -> Path:
    db = tmp_path / "feedback.sqlite"
    monkeypatch.setenv("DOGLENS_BUNDLE_DIR", str(tiny_bundle["bundle"]))
    monkeypatch.setenv("DOGLENS_FEEDBACK_DB", str(db))
    return db


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += (
        [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    )
    parts += [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def _upload(at: AppTest, data: bytes, name: str = "photo.png") -> AppTest:
    at.file_uploader[0].upload(name, data, "image/png").run()
    assert not at.exception, at.exception
    return at


def test_empty_state_and_mandatory_text(env) -> None:
    at = _app()
    assert INTRO in _texts(at)
    assert at.session_state["phase"] == "EMPTY"
    assert "Загрузите фотографию" in _texts(at)
    assert not [b for b in at.button if b.key == "predict"]


def test_catalog_is_searchable(env) -> None:
    at = _app()
    assert len(at.dataframe[0].value) == 120
    at.text_input(key="catalog_query").set_value("terrier").run()
    rows = at.dataframe[0].value
    assert 0 < len(rows) < 120
    assert all("terrier" in r.lower() for r in rows["English"])


def test_full_flow_whole_image_prediction_and_feedback(env: Path) -> None:
    at = _upload(_app(), _png())
    assert at.session_state["phase"] == "READY"
    at.button(key="predict").click().run()
    assert not at.exception
    result = at.session_state["result"]
    assert at.session_state["phase"] == "RESULT" and len(result.predictions) == 3
    assert result.user_roi is None and result.crop_box == (0, 0, 400, 300)
    text = _texts(at)
    assert SCORE_CAPTION in text and "оценка модели" in text
    sid = at.session_state["submission_id"]
    at.radio(key=f"verdict_{sid}").set_value(Verdict.LOOKS_WRONG).run()
    at.button(key="send_feedback").click().run()
    assert "Отзыв сохранён" in _texts(at)
    at.button(key="send_feedback").click().run()
    assert "уже сохранён" in _texts(at)
    with sqlite3.connect(env) as conn:
        rows = conn.execute(
            "SELECT submission_id, user_verdict, top_predictions_json FROM feedback"
        )
        rows = rows.fetchall()
    assert len(rows) == 1 and rows[0][0] == sid and rows[0][1] == "looks_wrong"
    assert "photo.png" not in rows[0][2]
    raw = env.read_bytes()
    assert b"photo.png" not in raw and b"\x89PNG" not in raw


def test_box_roi_changes_crop_resets_result_and_drives_prediction(env) -> None:
    at = _upload(_app(), _png())
    at.button(key="predict").click().run()
    assert at.session_state["result"] is not None
    key = at.session_state["upload_digest"][:12]
    at.toggle(key=f"whole_{key}").set_value(False).run()
    assert at.session_state["result"] is None  # mode change invalidates the old result
    # Preview is 400x300 (no downscale): select x 100..300, y 80..220.
    at.slider(key=f"x_{key}").set_value((100, 300)).run()
    at.slider(key=f"y_{key}").set_value((80, 220)).run()
    assert at.session_state["roi"] == (100, 80, 300, 220)
    at.button(key="predict").click().run()
    result = at.session_state["result"]
    assert result.user_roi == (100, 80, 300, 220)
    assert result.crop_box == (80, 66, 320, 234)  # +10% once, as shown in the crop preview
    captions = [c for img in at.image for c in img.captions]
    assert any("240×168" in c for c in captions)  # preview crop size == model crop
    at.slider(key=f"x_{key}").set_value((90, 300)).run()
    assert at.session_state["result"] is None


def test_too_small_roi_shows_warning_and_disables_prediction(env) -> None:
    at = _upload(_app(), _png())
    key = at.session_state["upload_digest"][:12]
    at.toggle(key=f"whole_{key}").set_value(False).run()
    at.slider(key=f"x_{key}").set_value((100, 130)).run()
    assert "слишком маленькая" in _texts(at)
    assert at.button(key="predict").disabled


def test_new_file_resets_result_and_roi(env) -> None:
    at = _upload(_app(), _png())
    at.button(key="predict").click().run()
    assert at.session_state["result"] is not None
    _upload(at, _png(color=(10, 200, 10)), "other.png")
    assert at.session_state["result"] is None and at.session_state["roi"] is None
    assert at.session_state["phase"] == "READY"


def test_invalid_file_shows_russian_error_without_traceback(env) -> None:
    at = _app()
    at.file_uploader[0].upload("fake.jpg", b"definitely not an image", "image/jpeg").run()
    assert not at.exception
    assert at.session_state["phase"] == "ERROR"
    assert any("JPEG" in e.value for e in at.error)
    assert "Traceback" not in _texts(at)


def test_clear_removes_session_data(env) -> None:
    at = _upload(_app(), _png())
    at.button(key="predict").click().run()
    at.button(key="clear").click().run()
    assert not at.exception
    state = at.session_state
    assert state["phase"] == "EMPTY"
    for key in ("image", "result", "upload_digest"):
        assert key not in state or state[key] is None


def test_sessions_are_isolated(env) -> None:
    first = _upload(_app(), _png())
    first.button(key="predict").click().run()
    second = _app()
    assert second.session_state["phase"] == "EMPTY"
    assert "result" not in second.session_state or second.session_state["result"] is None


def test_missing_bundle_is_reported_and_prediction_disabled(env, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOGLENS_BUNDLE_DIR", str(tmp_path / "no_bundle"))
    at = _upload(_app(), _png())
    assert any("Модель не найдена" in e.value for e in at.error)
    assert at.button(key="predict").disabled
    assert len(at.dataframe[0].value) == 120  # catalog still available


def test_feedback_failure_keeps_the_prediction(env, tmp_path, monkeypatch) -> None:
    broken = tmp_path / "db_dir"
    broken.mkdir()
    monkeypatch.setenv("DOGLENS_FEEDBACK_DB", str(broken))  # a directory cannot be opened
    at = _upload(_app(), _png())
    at.button(key="predict").click().run()
    sid = at.session_state["submission_id"]
    at.radio(key=f"verdict_{sid}").set_value(Verdict.UNKNOWN).run()
    at.button(key="send_feedback").click().run()
    assert not at.exception
    assert any("Не удалось сохранить отзыв" in e.value for e in at.error)
    assert at.session_state["result"] is not None and SCORE_CAPTION in _texts(at)
