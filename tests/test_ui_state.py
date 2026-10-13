"""UI state machine (no Streamlit). Fake predict functions are test doubles only."""

from __future__ import annotations

import io

from PIL import Image

from doglens import ui_state as ui
from doglens.errors import DogLensError
from doglens.schemas import Prediction, PredictionResult


def _png(color=(10, 20, 30), size=(200, 150)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _fake_predict(calls: list):
    def predict(image, roi=None, top_k=3):
        calls.append(roi)
        return PredictionResult("r", "v", (Prediction(0, "a", "A", "A", 0.9),))

    return predict


def _ready_state() -> dict:
    state: dict = {}
    ui.init(state)
    ui.handle_upload(state, _png())
    return state


def test_upload_flow_and_states() -> None:
    state: dict = {}
    ui.init(state)
    assert state["phase"] == ui.Phase.EMPTY
    ui.handle_upload(state, _png())
    assert state["phase"] == ui.Phase.READY and state["image"].size == (200, 150)
    assert ui.can_predict(state)  # whole image by default
    calls: list = []
    assert ui.run_prediction(state, _fake_predict(calls))
    assert state["phase"] == ui.Phase.RESULT and calls == [None]
    assert len(state["submission_id"]) == 32


def test_same_file_on_rerun_keeps_result_new_file_resets_everything() -> None:
    state = _ready_state()
    ui.run_prediction(state, _fake_predict([]))
    sid = state["submission_id"]
    ui.handle_upload(state, _png())  # Streamlit rerun with the same bytes
    assert state["result"] is not None and state["submission_id"] == sid
    ui.set_region(state, False, (10, 10, 100, 100))
    ui.handle_upload(state, _png(color=(200, 0, 0)))  # different photo
    assert state["result"] is None and state["submission_id"] is None
    assert state["roi"] is None and state["whole_image"] is True
    assert state["feedback_status"] is None and state["phase"] == ui.Phase.READY


def test_region_change_resets_result() -> None:
    state = _ready_state()
    ui.run_prediction(state, _fake_predict([]))
    ui.set_region(state, True, None)  # unchanged -> result kept
    assert state["result"] is not None
    ui.set_region(state, False, (0, 0, 100, 100))
    assert state["result"] is None and state["phase"] == ui.Phase.READY
    calls: list = []
    ui.run_prediction(state, _fake_predict(calls))
    assert calls == [(0, 0, 100, 100)]
    ui.set_region(state, False, (0, 0, 120, 100))
    assert state["result"] is None
    ui.run_prediction(state, _fake_predict([]))
    ui.set_region(state, True, None)  # whole-image mode toggled back
    assert state["result"] is None


def test_box_mode_without_valid_roi_cannot_predict() -> None:
    state = _ready_state()
    ui.set_region(state, False, None)
    assert not ui.can_predict(state)
    assert not ui.run_prediction(state, _fake_predict([]))


def test_second_run_while_predicting_is_blocked() -> None:
    state = _ready_state()
    state["phase"] = ui.Phase.PREDICTING
    calls: list = []
    assert not ui.run_prediction(state, _fake_predict(calls))
    assert calls == []


def test_errors_never_leave_predicting() -> None:
    state = _ready_state()

    def failing(image, roi=None, top_k=3):
        raise DogLensError("x", "Понятная ошибка")

    assert not ui.run_prediction(state, failing)
    assert state["phase"] == ui.Phase.ERROR and state["error"] == "Понятная ошибка"
    ui.recover_from_error(state)
    assert state["phase"] == ui.Phase.READY

    def crashing(image, roi=None, top_k=3):
        raise RuntimeError("boom")

    ui.run_prediction(state, crashing)
    assert state["phase"] == ui.Phase.ERROR and "boom" not in state["error"]


def test_invalid_upload_goes_to_error_and_next_upload_recovers() -> None:
    state: dict = {}
    ui.init(state)
    ui.handle_upload(state, b"not an image")
    assert state["phase"] == ui.Phase.ERROR and state["image"] is None
    assert "JPEG" in state["error"]
    ui.handle_upload(state, _png())
    assert state["phase"] == ui.Phase.READY


def test_clear_removes_user_data_and_resets_uploader() -> None:
    state = _ready_state()
    ui.run_prediction(state, _fake_predict([]))
    nonce = state["uploader_nonce"]
    ui.clear(state)
    assert state["phase"] == ui.Phase.EMPTY and state["uploader_nonce"] == nonce + 1
    for key in ("image", "result", "roi", "upload_digest", "submission_id"):
        assert state.get(key) is None


def test_removing_the_file_clears_the_session() -> None:
    state = _ready_state()
    ui.handle_upload(state, None)
    assert state.get("image") is None and state["phase"] == ui.Phase.EMPTY
