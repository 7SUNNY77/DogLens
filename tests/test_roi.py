"""Unit tests for ROI validation and preview coordinate mapping."""

from __future__ import annotations

import math
import random

import pytest
from PIL import Image

from doglens.config import ImageLimits
from doglens.errors import RoiError
from doglens.roi import (
    PreviewGeometry,
    crop_roi,
    display_box_to_roi,
    expand_box,
    full_image_roi,
    make_preview,
    preview_geometry,
    roi_to_display,
    validate_roi,
)


def _code(fn, *args) -> str:
    with pytest.raises(RoiError) as info:
        fn(*args)
    assert info.value.user_message
    return info.value.code


# --- validate_roi ---------------------------------------------------------------------------


def test_valid_roi_and_integral_floats() -> None:
    assert validate_roi((10, 20, 110, 120), (200, 200)) == (10, 20, 110, 120)
    assert validate_roi([10.0, 20.0, 110.0, 120.0], (200, 200)) == (10, 20, 110, 120)


@pytest.mark.parametrize(
    ("roi", "code"),
    [
        ((10.5, 20, 110, 120), "roi_not_integer"),
        ((float("nan"), 0, 100, 100), "roi_not_finite"),
        ((0, 0, float("inf"), 100), "roi_not_finite"),
        ((True, 0, 100, 100), "roi_not_numeric"),
        (("0", 0, 100, 100), "roi_not_numeric"),
        ((None, 0, 100, 100), "roi_not_numeric"),
        ((0, 0, 100), "roi_bad_shape"),
        ("0,0,100,100", "roi_bad_shape"),
        ((-1, 0, 100, 100), "roi_out_of_bounds"),
        ((0, 0, 201, 100), "roi_out_of_bounds"),
        ((0, 0, 100, 201), "roi_out_of_bounds"),
        ((100, 0, 100, 100), "roi_empty"),
        ((150, 0, 50, 100), "roi_empty"),
        ((0, 0, 63, 100), "roi_too_small"),
        ((0, 0, 100, 63), "roi_too_small"),
    ],
)
def test_invalid_rois_are_rejected_not_clamped(roi, code: str) -> None:
    assert _code(validate_roi, roi, (200, 200)) == code


def test_minimum_side_is_inclusive() -> None:
    assert validate_roi((0, 0, 64, 64), (200, 200)) == (0, 0, 64, 64)
    assert validate_roi((136, 136, 200, 200), (200, 200)) == (136, 136, 200, 200)


def test_full_image_roi() -> None:
    assert full_image_roi((300, 200)) == (0, 0, 300, 200)
    assert _code(full_image_roi, (63, 200)) == "roi_too_small"


def test_custom_min_side() -> None:
    assert validate_roi((0, 0, 10, 10), (20, 20), ImageLimits(min_roi_side=10)) == (0, 0, 10, 10)


# --- preview geometry -----------------------------------------------------------------------


def test_preview_geometry_landscape_portrait_and_no_upscale() -> None:
    assert preview_geometry((4000, 3000), 800).display_size == (800, 600)
    assert preview_geometry((3000, 4000), 800).display_size == (600, 800)
    assert preview_geometry((500, 300), 800).display_size == (500, 300)
    with pytest.raises(ValueError):
        preview_geometry((100, 100), 0)


def test_make_preview_reports_actual_size() -> None:
    preview, geometry = make_preview(Image.new("RGB", (1001, 667)), 500)
    assert preview.size == geometry.display_size
    assert geometry.original_size == (1001, 667)
    assert max(preview.size) == 500


# --- display -> original mapping -------------------------------------------------------------


def test_display_mapping_floor_min_ceil_max() -> None:
    geometry = PreviewGeometry(original_size=(4000, 3000), display_size=(800, 600))
    roi = display_box_to_roi((10.3, 20.7, 100.2, 200.9), geometry)
    # 51.5 -> 51, 103.5 -> 103, 501.0 -> 501, 1004.5 -> 1005
    assert roi == (51, 103, 501, 1005)


def test_full_display_box_maps_to_full_image_with_non_integer_scale() -> None:
    geometry = preview_geometry((1000, 750), 333)
    dw, dh = geometry.display_size
    assert display_box_to_roi((0, 0, dw, dh), geometry) == (0, 0, 1000, 750)


def test_border_jitter_within_tolerance_is_clamped() -> None:
    geometry = PreviewGeometry(original_size=(4000, 3000), display_size=(800, 600))
    assert display_box_to_roi((-0.5, -1.0, 800.7, 601.0), geometry) == (0, 0, 4000, 3000)


@pytest.mark.parametrize(
    "box",
    [(-5, 0, 100, 100), (0, -1.5, 100, 100), (0, 0, 803, 100), (0, 0, 100, 610)],
)
def test_selection_far_outside_preview_is_rejected(box) -> None:
    geometry = PreviewGeometry(original_size=(4000, 3000), display_size=(800, 600))
    assert _code(display_box_to_roi, box, geometry) == "roi_out_of_bounds"


def test_inverted_or_degenerate_selection_is_rejected() -> None:
    geometry = PreviewGeometry(original_size=(4000, 3000), display_size=(800, 600))
    assert _code(display_box_to_roi, (100, 100, 50, 200), geometry) == "roi_empty"
    assert _code(display_box_to_roi, (100, 100, 100, 200), geometry) == "roi_empty"
    assert _code(display_box_to_roi, (0, 0, float("nan"), 10), geometry) == "roi_not_finite"


def test_minimum_size_is_checked_in_original_pixels() -> None:
    geometry = PreviewGeometry(original_size=(4000, 3000), display_size=(800, 600))
    assert _code(display_box_to_roi, (0, 0, 12, 100), geometry) == "roi_too_small"  # 60 px
    assert display_box_to_roi((0, 0, 13, 100), geometry) == (0, 0, 65, 500)


def test_mapping_always_covers_the_exact_selection() -> None:
    rng = random.Random(42)
    for _ in range(2000):
        original = (rng.randint(64, 6000), rng.randint(64, 6000))
        geometry = preview_geometry(original, rng.choice([300, 512, 777, 1024]))
        dw, dh = geometry.display_size
        x0, x1 = sorted(rng.uniform(0, dw) for _ in range(2))
        y0, y1 = sorted(rng.uniform(0, dh) for _ in range(2))
        try:
            xmin, ymin, xmax, ymax = display_box_to_roi((x0, y0, x1, y1), geometry)
        except RoiError as exc:
            assert exc.code in {"roi_too_small", "roi_empty"}
            continue
        ex0, ex1 = x0 * geometry.scale_x, x1 * geometry.scale_x
        ey0, ey1 = y0 * geometry.scale_y, y1 * geometry.scale_y
        assert xmin <= ex0 + 1e-6 and ex1 - 1e-6 <= xmax
        assert ymin <= ey0 + 1e-6 and ey1 - 1e-6 <= ymax
        assert ex0 - xmin < 1 + 1e-6 and xmax - ex1 < 1 + 1e-6
        assert 0 <= xmin < xmax <= original[0] and 0 <= ymin < ymax <= original[1]


def test_roi_to_display_round_trip() -> None:
    geometry = preview_geometry((4000, 3000), 800)
    roi = (100, 200, 1300, 2400)
    assert display_box_to_roi(roi_to_display(roi, geometry), geometry) == roi


# --- cropping and bbox expansion --------------------------------------------------------------


def test_crop_roi_uses_exclusive_max_and_none_means_full_image() -> None:
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    img.putpixel((10, 20), (255, 0, 0))
    img.putpixel((109, 99), (0, 255, 0))
    crop = crop_roi(img, (10, 20, 110, 100))
    assert crop.size == (100, 80)
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert crop.getpixel((99, 79)) == (0, 255, 0)
    assert crop_roi(img, None).size == (200, 100)
    with pytest.raises(RoiError):
        crop_roi(img, (0, 0, 300, 100))


def test_expand_box_adds_context_and_clamps() -> None:
    assert expand_box((50, 20, 150, 70), (300, 200), 0.1) == (40, 15, 160, 75)
    assert expand_box((10, 10, 110, 60), (200, 100), 0.1) == (0, 5, 120, 65)
    assert expand_box((0, 0, 200, 100), (200, 100), 0.1) == (0, 0, 200, 100)
    assert expand_box((50, 20, 150, 70), (300, 200), 0.0) == (50, 20, 150, 70)
    # Fractional context: floor on min, ceil on max.
    assert expand_box((10, 10, 25, 25), (100, 100), 0.1) == (8, 8, 27, 27)


@pytest.mark.parametrize(
    ("box", "context"),
    [((0, 0, 300, 100), 0.1), ((50, 50, 40, 60), 0.1), ((0, 0, 10, 10), -0.1)],
)
def test_expand_box_rejects_invalid_input(box, context: float) -> None:
    with pytest.raises(ValueError):
        expand_box(box, (200, 100), context)


def test_expand_box_rejects_non_finite_context() -> None:
    with pytest.raises(ValueError):
        expand_box((0, 0, 10, 10), (200, 100), math.nan)
