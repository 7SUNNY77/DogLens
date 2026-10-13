"""Region-of-interest validation and preview <-> original coordinate mapping.

Coordinates are (xmin, ymin, xmax, ymax) in pixels of the EXIF-transposed image, before any
resize; xmax/ymax are exclusive, matching ``PIL.Image.crop``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from PIL import Image

from doglens.config import DEFAULT_LIMITS, ImageLimits
from doglens.errors import RoiError
from doglens.schemas import Roi

# Products like 0.1 * 30 land a hair away from an integer; snap those before floor/ceil.
_SNAP_EPS = 1e-6


@dataclass(frozen=True)
class PreviewGeometry:
    """Actual sizes of the original image and the downscaled copy shown to the user."""

    original_size: tuple[int, int]
    display_size: tuple[int, int]

    @property
    def scale_x(self) -> float:
        return self.original_size[0] / self.display_size[0]

    @property
    def scale_y(self) -> float:
        return self.original_size[1] / self.display_size[1]


def _coerce_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RoiError("roi_not_numeric", "Координаты области должны быть числами.", name)
    f = float(value)
    if not math.isfinite(f):
        raise RoiError("roi_not_finite", "Координаты области должны быть конечными числами.", name)
    return f


def _unpack4(box: Sequence[object]) -> tuple[object, object, object, object]:
    if isinstance(box, str | bytes) or not isinstance(box, Sequence) or len(box) != 4:
        raise RoiError(
            "roi_bad_shape", "Область задаётся четырьмя координатами.", f"got {type(box).__name__}"
        )
    return box[0], box[1], box[2], box[3]


def _snap(value: float) -> float:
    nearest = round(value)
    return float(nearest) if abs(value - nearest) < _SNAP_EPS else value


def validate_roi(
    roi: Sequence[object],
    image_size: tuple[int, int],
    limits: ImageLimits = DEFAULT_LIMITS,
) -> Roi:
    """Check an ROI in original-image pixels. Out-of-range values are rejected, never clamped."""
    width, height = image_size
    names = ("xmin", "ymin", "xmax", "ymax")
    values = [_coerce_number(v, n) for v, n in zip(_unpack4(roi), names, strict=True)]
    if not all(v.is_integer() for v in values):
        raise RoiError(
            "roi_not_integer",
            "Координаты области должны быть целыми числами пикселей.",
            str(values),
        )
    xmin, ymin, xmax, ymax = (int(v) for v in values)
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        raise RoiError(
            "roi_out_of_bounds",
            f"Область выходит за границы изображения ({width}×{height}).",
            f"roi={(xmin, ymin, xmax, ymax)} size={image_size}",
        )
    if xmin >= xmax or ymin >= ymax:
        raise RoiError(
            "roi_empty",
            "Область пустая: правая и нижняя границы должны быть больше левой и верхней.",
            f"roi={(xmin, ymin, xmax, ymax)}",
        )
    side = limits.min_roi_side
    if xmax - xmin < side or ymax - ymin < side:
        raise RoiError(
            "roi_too_small",
            f"Область слишком маленькая: минимум {side}×{side} пикселей исходного изображения.",
            f"roi={(xmin, ymin, xmax, ymax)}",
        )
    return xmin, ymin, xmax, ymax


def full_image_roi(image_size: tuple[int, int], limits: ImageLimits = DEFAULT_LIMITS) -> Roi:
    width, height = image_size
    return validate_roi((0, 0, width, height), image_size, limits)


def preview_geometry(original_size: tuple[int, int], max_side: int) -> PreviewGeometry:
    """Size of a preview whose longer side is at most ``max_side`` (never upscaled)."""
    if max_side < 1:
        raise ValueError("max_side must be >= 1")
    width, height = original_size
    if width < 1 or height < 1:
        raise ValueError("original_size must be positive")
    factor = min(1.0, max_side / max(width, height))
    display = (max(1, round(width * factor)), max(1, round(height * factor)))
    return PreviewGeometry(original_size=(width, height), display_size=display)


def make_preview(image: Image.Image, max_side: int) -> tuple[Image.Image, PreviewGeometry]:
    """Downscaled copy for display plus the geometry needed to map selections back."""
    geometry = preview_geometry(image.size, max_side)
    if geometry.display_size == image.size:
        return image.copy(), geometry
    preview = image.resize(geometry.display_size, Image.Resampling.LANCZOS)
    # Use the real resulting size, not the requested one.
    return preview, PreviewGeometry(original_size=image.size, display_size=preview.size)


def display_box_to_roi(
    box: Sequence[object],
    geometry: PreviewGeometry,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> Roi:
    """Map a selection on the preview to original pixels: floor for min, ceil for max.

    A selection may overshoot the preview border by at most ``limits.display_tolerance_px``
    (pointer/rounding jitter); such overshoot is clamped. Anything further is rejected.
    """
    dw, dh = geometry.display_size
    names = ("xmin", "ymin", "xmax", "ymax")
    x0, y0, x1, y1 = (_coerce_number(v, n) for v, n in zip(_unpack4(box), names, strict=True))
    tol = limits.display_tolerance_px
    if min(x0, x1) < -tol or min(y0, y1) < -tol or max(x0, x1) > dw + tol or max(y0, y1) > dh + tol:
        raise RoiError(
            "roi_out_of_bounds",
            "Область выходит за границы изображения.",
            f"display_box={(x0, y0, x1, y1)} display_size={geometry.display_size}",
        )
    if x0 >= x1 or y0 >= y1:
        raise RoiError(
            "roi_empty",
            "Область пустая: правая и нижняя границы должны быть больше левой и верхней.",
            f"display_box={(x0, y0, x1, y1)}",
        )
    x0, x1 = min(max(x0, 0.0), dw), min(max(x1, 0.0), dw)
    y0, y1 = min(max(y0, 0.0), dh), min(max(y1, 0.0), dh)

    width, height = geometry.original_size
    xmin = math.floor(_snap(x0 * geometry.scale_x))
    ymin = math.floor(_snap(y0 * geometry.scale_y))
    xmax = math.ceil(_snap(x1 * geometry.scale_x))
    ymax = math.ceil(_snap(y1 * geometry.scale_y))
    # The display-border clamp above maps to the original border; only float noise can exceed it.
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, width), min(ymax, height)
    return validate_roi((xmin, ymin, xmax, ymax), geometry.original_size, limits)


def roi_to_display(roi: Roi, geometry: PreviewGeometry) -> tuple[float, float, float, float]:
    """Inverse mapping for drawing an existing ROI on the preview."""
    xmin, ymin, xmax, ymax = roi
    return (
        xmin / geometry.scale_x,
        ymin / geometry.scale_y,
        xmax / geometry.scale_x,
        ymax / geometry.scale_y,
    )


def crop_roi(
    image: Image.Image, roi: Roi | None, limits: ImageLimits = DEFAULT_LIMITS
) -> Image.Image:
    """Crop a validated ROI; ``None`` means the whole image (still size-checked)."""
    checked = (
        full_image_roi(image.size, limits) if roi is None else validate_roi(roi, image.size, limits)
    )
    return image.crop(checked)


def expand_box(box: Roi, image_size: tuple[int, int], context: float) -> Roi:
    """Grow a box by ``context`` of its width/height on every side, clamped to the image."""
    if not math.isfinite(context) or context < 0:
        raise ValueError("context must be finite and >= 0")
    xmin, ymin, xmax, ymax = box
    width, height = image_size
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"box {box} is not inside image {image_size}")
    dx = (xmax - xmin) * context
    dy = (ymax - ymin) * context
    return (
        max(0, math.floor(_snap(xmin - dx))),
        max(0, math.floor(_snap(ymin - dy))),
        min(width, math.ceil(_snap(xmax + dx))),
        min(height, math.ceil(_snap(ymax + dy))),
    )


# --- model input region (shared by the UI preview and the predictor) ---------------------------

APP_CROP_POLICY = (
    "user ROI (validated, >= min side, before context) is expanded by `context` of its width on "
    "left/right and of its height on top/bottom, floor/ceil, clipped to the image; "
    "roi=None uses the whole image without any expansion"
)


@dataclass(frozen=True)
class ModelRegion:
    """The exact box fed to preprocessing; context is applied here and only here."""

    user_roi: Roi | None  # validated user box before context; None = whole image
    crop_box: Roi  # final box after context (or the full image)
    context: float

    @property
    def whole_image(self) -> bool:
        return self.user_roi is None


def model_region(
    image_size: tuple[int, int],
    roi: Sequence[object] | None,
    context: float,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> ModelRegion:
    """Validate ``roi`` (EXIF-oriented original pixels) and add context exactly once."""
    if roi is None:
        return ModelRegion(None, full_image_roi(image_size, limits), context)
    user = validate_roi(roi, image_size, limits)
    return ModelRegion(user, expand_box(user, image_size, context), context)


def crop_model_region(
    image: Image.Image,
    roi: Sequence[object] | None,
    context: float,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> tuple[Image.Image, ModelRegion]:
    region = model_region(image.size, roi, context, limits)
    return image.crop(region.crop_box), region
