"""SQLite storage for user feedback on predictions.

Stored: submission_id (unique), created_at (UTC), model_version, top predictions (class_id,
breed_key, score), verdict and an optional breed_key. Never stored: images, EXIF/GPS, file
names or any personal data. Feedback is not a verified training label and is never used for
automatic retraining.

Each operation opens its own short-lived connection (no connection is shared across threads)
and uses parameterized SQL in a short transaction. A repeated submission_id is idempotent.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doglens.errors import DogLensError
from doglens.schemas import Prediction, Verdict

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.environ.get("DOGLENS_FEEDBACK_DB", "data/feedback/feedback.sqlite"))
SUBMISSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_PREDICTIONS = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    top_predictions_json TEXT NOT NULL,
    user_verdict TEXT NOT NULL CHECK (user_verdict IN (
        'looks_correct', 'looks_wrong', 'unknown', 'mixed_breed', 'breed_not_supported'
    )),
    user_breed_key TEXT
)
"""
FEEDBACK_COLUMNS = (
    "id",
    "submission_id",
    "created_at",
    "model_version",
    "top_predictions_json",
    "user_verdict",
    "user_breed_key",
)


class FeedbackError(DogLensError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(
            code,
            "Не удалось сохранить отзыв. Результат распознавания при этом не изменился.",
            detail,
        )


def predictions_payload(predictions: Iterable[Prediction | dict[str, Any]]) -> list[dict]:
    """Only class_id, breed_key and score are kept - nothing about the image."""
    payload = []
    for p in predictions:
        item = p if isinstance(p, dict) else {"class_id": p.class_id, "breed_key": p.breed_key,
                                              "score": p.score}  # fmt: skip
        class_id, breed_key, score = item.get("class_id"), item.get("breed_key"), item.get("score")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise FeedbackError("invalid_predictions", "class_id must be int")
        if not isinstance(breed_key, str) or not breed_key:
            raise FeedbackError("invalid_predictions", "breed_key must be a string")
        if not isinstance(score, int | float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise FeedbackError("invalid_predictions", "score must be finite in [0, 1]")
        payload.append({"class_id": class_id, "breed_key": breed_key, "score": float(score)})
    if not 1 <= len(payload) <= MAX_PREDICTIONS:
        raise FeedbackError("invalid_predictions", f"need 1..{MAX_PREDICTIONS} predictions")
    return payload


class FeedbackStore:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH, timeout_s: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout_s)
        conn.execute(SCHEMA)
        return conn

    def submit(
        self,
        submission_id: str,
        model_version: str,
        predictions: Sequence[Prediction | dict[str, Any]],
        verdict: str | Verdict,
        user_breed_key: str | None = None,
        known_breed_keys: set[str] | None = None,
    ) -> str:
        """Store feedback; returns 'created' or 'duplicate' (same submission_id, kept as is)."""
        if not isinstance(submission_id, str) or not SUBMISSION_ID_RE.match(submission_id):
            raise FeedbackError("invalid_submission_id", "submission_id must be 32 hex chars")
        if not isinstance(model_version, str) or not model_version:
            raise FeedbackError("invalid_model_version", "empty model_version")
        try:
            verdict_value = Verdict(verdict).value
        except ValueError:
            raise FeedbackError("invalid_verdict", f"unknown verdict {verdict!r}") from None
        bad_type = user_breed_key is not None and not isinstance(user_breed_key, str)
        unknown = (
            user_breed_key is not None
            and known_breed_keys is not None
            and user_breed_key not in known_breed_keys
        )
        if bad_type or unknown:
            raise FeedbackError("invalid_breed_key", "unknown breed_key")
        payload = json.dumps(predictions_payload(predictions), ensure_ascii=False)
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        try:
            with closing(self._connect()) as conn, conn:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO feedback (submission_id, created_at, model_version, "
                    "top_predictions_json, user_verdict, user_breed_key) VALUES (?, ?, ?, ?, ?, ?)",
                    (submission_id, created_at, model_version, payload, verdict_value,
                     user_breed_key),
                )  # fmt: skip
                status = "created" if cursor.rowcount == 1 else "duplicate"
        except sqlite3.Error as exc:
            logger.warning("feedback write failed: %s", type(exc).__name__)
            raise FeedbackError("storage_error", f"{type(exc).__name__}: {exc}") from exc
        logger.info("feedback %s submission=%s verdict=%s", status, submission_id, verdict_value)
        return status

    def rows(self) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                cur = conn.execute(
                    f"SELECT {', '.join(FEEDBACK_COLUMNS)} FROM feedback ORDER BY id"
                )
                return [dict(zip(FEEDBACK_COLUMNS, row, strict=True)) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            raise FeedbackError("storage_error", f"{type(exc).__name__}: {exc}") from exc

    def columns(self) -> list[str]:
        with closing(self._connect()) as conn:
            return [r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()]
