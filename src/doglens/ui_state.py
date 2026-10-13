"""Session state machine for the Streamlit UI, independent of Streamlit (unit-testable).

States: EMPTY -> VALIDATING -> READY -> PREDICTING -> RESULT, and ERROR from any step.
All user data (image, ROI, result, feedback state) lives only in the per-session mapping
passed in; nothing here is global.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, MutableMapping
from enum import StrEnum
from typing import Any

from doglens.errors import DogLensError
from doglens.image_io import load_image
from doglens.schemas import PredictionResult, Roi

GENERIC_ERROR = "Что-то пошло не так. Попробуйте ещё раз или загрузите другое фото."


class Phase(StrEnum):
    EMPTY = "EMPTY"
    VALIDATING = "VALIDATING"
    READY = "READY"
    PREDICTING = "PREDICTING"
    RESULT = "RESULT"
    ERROR = "ERROR"


# Keys holding user data; clear() removes all of them.
USER_KEYS = (
    "phase",
    "upload_digest",
    "image",
    "image_format",
    "roi",
    "whole_image",
    "result",
    "submission_id",
    "feedback_status",
    "error",
)


def init(state: MutableMapping[str, Any]) -> None:
    state.setdefault("phase", Phase.EMPTY)
    state.setdefault("whole_image", True)
    state.setdefault("uploader_nonce", 0)


def _reset_result(state: MutableMapping[str, Any]) -> None:
    state["result"] = None
    state["submission_id"] = None
    state["feedback_status"] = None


def handle_upload(state: MutableMapping[str, Any], data: bytes | None) -> None:
    """Process an uploader value. A *new* file resets ROI, result and feedback state."""
    init(state)
    if data is None:
        if state.get("upload_digest") is not None:
            clear(state, reset_uploader=False)
        return
    digest = hashlib.sha256(data).hexdigest()
    if digest == state.get("upload_digest"):
        return  # same file on a rerun: keep ROI and result
    state["upload_digest"] = digest
    state["image"] = None
    state["roi"] = None
    state["whole_image"] = True
    state["error"] = None
    _reset_result(state)
    state["phase"] = Phase.VALIDATING
    try:
        loaded = load_image(data)
    except DogLensError as exc:
        state["phase"], state["error"] = Phase.ERROR, exc.user_message
        return
    except Exception:  # noqa: BLE001 - never leave the UI in VALIDATING
        state["phase"], state["error"] = Phase.ERROR, GENERIC_ERROR
        return
    state["image"] = loaded.image
    state["image_format"] = loaded.source_format
    state["phase"] = Phase.READY


def set_region(state: MutableMapping[str, Any], whole_image: bool, roi: Roi | None) -> None:
    """Changing the region (or the whole-image mode) makes an old result stale -> reset it."""
    new_roi = None if whole_image else roi
    if bool(state.get("whole_image")) == bool(whole_image) and state.get("roi") == new_roi:
        return
    state["whole_image"] = bool(whole_image)
    state["roi"] = new_roi
    if state.get("result") is not None:
        _reset_result(state)
    if state.get("phase") in (Phase.RESULT,):
        state["phase"] = Phase.READY


def can_predict(state: MutableMapping[str, Any]) -> bool:
    if state.get("phase") not in (Phase.READY, Phase.RESULT) or state.get("image") is None:
        return False
    return bool(state.get("whole_image")) or state.get("roi") is not None


def run_prediction(
    state: MutableMapping[str, Any],
    predict: Callable[..., PredictionResult],
) -> bool:
    """Run inference once; a second call while PREDICTING is refused. Never stays PREDICTING."""
    if state.get("phase") == Phase.PREDICTING or not can_predict(state):
        return False
    state["phase"] = Phase.PREDICTING
    roi = None if state.get("whole_image") else state.get("roi")
    try:
        result = predict(state["image"], roi=roi, top_k=3)
    except DogLensError as exc:
        state["phase"], state["error"] = Phase.ERROR, exc.user_message
        _reset_result(state)
        return False
    except Exception:  # noqa: BLE001
        state["phase"], state["error"] = Phase.ERROR, GENERIC_ERROR
        _reset_result(state)
        return False
    state["result"] = result
    state["submission_id"] = uuid.uuid4().hex  # one feedback id per shown result
    state["feedback_status"] = None
    state["error"] = None
    state["phase"] = Phase.RESULT
    return True


def recover_from_error(state: MutableMapping[str, Any]) -> None:
    """After a prediction error the validated image stays usable."""
    if state.get("phase") == Phase.ERROR:
        state["phase"] = Phase.READY if state.get("image") is not None else Phase.EMPTY


def clear(state: MutableMapping[str, Any], reset_uploader: bool = True) -> None:
    for key in USER_KEYS:
        if key in state:
            del state[key]
    if reset_uploader:
        state["uploader_nonce"] = int(state.get("uploader_nonce", 0)) + 1
    init(state)
