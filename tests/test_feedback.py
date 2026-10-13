"""SQLite feedback: schema, enums, idempotency, parameterized SQL, no image data."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

from doglens.feedback import FEEDBACK_COLUMNS, FeedbackError, FeedbackStore
from doglens.schemas import Prediction, Verdict

PREDS = (
    Prediction(0, "chihuahua", "Chihuahua", "Chihuahua", 0.7),
    Prediction(1, "japanese_spaniel", "Japanese spaniel", "Japanese spaniel", 0.2),
    Prediction(2, "maltese_dog", "Maltese dog", "Maltese dog", 0.05),
)
KEYS = {"chihuahua", "japanese_spaniel", "maltese_dog"}


def _sid() -> str:
    return uuid.uuid4().hex


def test_schema_has_only_the_specified_columns(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    assert store.columns() == list(FEEDBACK_COLUMNS)
    forbidden = {"image", "photo", "exif", "gps", "filename", "file_name", "path"}
    assert not forbidden & {c.lower() for c in store.columns()}


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_is_accepted(tmp_path: Path, verdict: Verdict) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    assert store.submit(_sid(), "v1", PREDS, verdict, known_breed_keys=KEYS) == "created"
    assert store.rows()[0]["user_verdict"] == verdict.value


def test_duplicate_submission_is_idempotent(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    sid = _sid()
    assert store.submit(sid, "v1", PREDS, "looks_correct", "chihuahua", KEYS) == "created"
    assert store.submit(sid, "v1", PREDS, "looks_wrong", None, KEYS) == "duplicate"
    rows = store.rows()
    assert len(rows) == 1 and rows[0]["user_verdict"] == "looks_correct"  # first write kept
    assert rows[0]["user_breed_key"] == "chihuahua"
    assert rows[0]["created_at"].endswith("Z")


def test_stored_predictions_contain_no_image_data(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    store.submit(_sid(), "v1", PREDS, "unknown")
    payload = json.loads(store.rows()[0]["top_predictions_json"])
    assert payload == [
        {"class_id": 0, "breed_key": "chihuahua", "score": 0.7},
        {"class_id": 1, "breed_key": "japanese_spaniel", "score": 0.2},
        {"class_id": 2, "breed_key": "maltese_dog", "score": 0.05},
    ]
    raw = (tmp_path / "fb.sqlite").read_bytes()
    assert b"\xff\xd8\xff" not in raw and b"\x89PNG" not in raw


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"submission_id": "not-hex"}, "invalid_submission_id"),
        ({"submission_id": "A" * 32}, "invalid_submission_id"),
        ({"verdict": "definitely"}, "invalid_verdict"),
        ({"user_breed_key": "unicorn"}, "invalid_breed_key"),
        ({"model_version": ""}, "invalid_model_version"),
        ({"predictions": []}, "invalid_predictions"),
        ({"predictions": [{"class_id": 0, "breed_key": "x", "score": float("nan")}]},
         "invalid_predictions"),
        ({"predictions": [{"class_id": "0", "breed_key": "x", "score": 0.1}]},
         "invalid_predictions"),
    ],
)  # fmt: skip
def test_invalid_submissions_are_rejected(tmp_path: Path, kwargs: dict, code: str) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    args = {"submission_id": _sid(), "model_version": "v1", "predictions": PREDS,
            "verdict": "unknown", "user_breed_key": None, "known_breed_keys": KEYS}  # fmt: skip
    args.update(kwargs)
    with pytest.raises(FeedbackError) as info:
        store.submit(**args)
    assert info.value.code == code
    assert store.rows() == []


def test_sql_is_parameterized(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    tricky = "v1'); DROP TABLE feedback; --"
    store.submit(_sid(), tricky, PREDS, "unknown")
    assert store.rows()[0]["model_version"] == tricky


def test_storage_failure_raises_user_friendly_error(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    store = FeedbackStore(blocker / "fb.sqlite")  # parent is a file -> cannot create the DB
    with pytest.raises((FeedbackError, OSError)):
        store.submit(_sid(), "v1", PREDS, "unknown")
    directory_db = tmp_path / "db_is_dir"
    directory_db.mkdir()
    with pytest.raises(FeedbackError) as info:
        FeedbackStore(directory_db).submit(_sid(), "v1", PREDS, "unknown")
    assert "Результат распознавания" in info.value.user_message


def test_threads_use_separate_connections(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.sqlite")
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(5):
                store.submit(_sid(), "v1", PREDS, "unknown")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(store.rows()) == 20
    with sqlite3.connect(tmp_path / "fb.sqlite") as conn:
        assert (
            conn.execute("SELECT COUNT(DISTINCT submission_id) FROM feedback").fetchone()[0] == 20
        )
