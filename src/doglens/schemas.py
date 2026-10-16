"""Shared data contracts between dataset preparation, inference, UI and feedback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# (xmin, ymin, xmax, ymax) in pixels of the EXIF-transposed image; xmax/ymax are exclusive.
Roi = tuple[int, int, int, int]

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "relative_path",
    "class_id",
    "breed_key",
    "split",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    # 1 for an official test image whose exact copy existed in official train (removed there).
    "dup_with_test",
    # 1 when the same file content also occurs with a different breed label.
    "label_conflict",
    # Number of valid <object> boxes in the source XML (multi-dog frames have > 1).
    "n_boxes",
)

EXCLUDED_COLUMNS: tuple[str, ...] = (
    "image_id",
    "split",
    "reason",
    "relative_path",
    "class_id",
    "breed_key",
    "sha256",
    "duplicate_of",
)

SPLITS: tuple[str, ...] = ("train", "val", "test")

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILES: tuple[str, ...] = (
    "weights.pt",
    "metadata.json",
    "class_mapping.json",
    "metrics.json",
)


class Verdict(StrEnum):
    """User feedback on a prediction."""

    LOOKS_CORRECT = "looks_correct"
    LOOKS_WRONG = "looks_wrong"
    UNKNOWN = "unknown"
    MIXED_BREED = "mixed_breed"
    BREED_NOT_SUPPORTED = "breed_not_supported"


@dataclass(frozen=True)
class Prediction:
    class_id: int
    breed_key: str
    name_ru: str
    name_en: str
    # Softmax score over the 120 supported classes; not a calibrated probability of correctness.
    score: float


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    model_version: str
    predictions: tuple[Prediction, ...]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    # Region actually used: the user's box before context (None = whole image) and the final
    # crop box after context, both in EXIF-oriented original pixels.
    user_roi: Roi | None = None
    crop_box: Roi | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predictions"] = [asdict(p) for p in self.predictions]
        d["warnings"] = list(self.warnings)
        d["user_roi"] = list(self.user_roi) if self.user_roi else None
        d["crop_box"] = list(self.crop_box) if self.crop_box else None
        return d
