"""Shared model-region preparation: context is added exactly once, whole image never."""

from __future__ import annotations

import pytest
from PIL import Image

from doglens.errors import RoiError
from doglens.roi import crop_model_region, expand_box, model_region


def test_model_region_adds_context_once_and_clips() -> None:
    region = model_region((400, 300), (100, 100, 200, 180), 0.1)  # 100x80 user box
    assert region.user_roi == (100, 100, 200, 180)
    assert region.crop_box == (90, 92, 210, 188) == expand_box(region.user_roi, (400, 300), 0.1)
    assert not region.whole_image
    assert model_region((400, 300), (0, 0, 100, 300), 0.1).crop_box == (0, 0, 110, 300)


def test_model_region_whole_image_is_not_expanded() -> None:
    region = model_region((400, 300), None, 0.1)
    assert region.user_roi is None and region.whole_image
    assert region.crop_box == (0, 0, 400, 300)


def test_user_box_is_validated_before_context() -> None:
    # 60 px wide would be 72 px after context, but the *user* box itself must be >= 64 px.
    with pytest.raises(RoiError) as info:
        model_region((400, 300), (100, 100, 160, 200), 0.1)
    assert info.value.code == "roi_too_small"
    with pytest.raises(RoiError):
        model_region((400, 300), (350, 0, 410, 100), 0.1)  # out of bounds is not clamped


def test_crop_model_region_returns_the_exact_crop() -> None:
    img = Image.new("RGB", (400, 300))
    img.putpixel((90, 92), (255, 0, 0))
    crop, region = crop_model_region(img, (100, 100, 200, 180), 0.1)
    assert crop.size == (120, 96) and crop.getpixel((0, 0)) == (255, 0, 0)
    assert region.crop_box == (90, 92, 210, 188)
