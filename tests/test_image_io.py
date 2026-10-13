"""Unit tests for safe image decoding. All images are synthetic and generated in memory."""

from __future__ import annotations

import numpy as np
import pytest
from _imaging import encode, png_with_declared_size, solid, two_halves
from PIL import Image, ImageFile

from doglens.config import ImageLimits
from doglens.errors import ImageValidationError
from doglens.image_io import load_image, sniff_format


def _code(data: bytes, limits: ImageLimits | None = None) -> str:
    with pytest.raises(ImageValidationError) as info:
        load_image(data) if limits is None else load_image(data, limits)
    assert info.value.user_message  # always a user-facing Russian message
    return info.value.code


# --- valid inputs ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "kwargs"),
    [("JPEG", {"quality": 95}), ("PNG", {}), ("WEBP", {"lossless": True})],
)
def test_valid_formats_decode_to_rgb(fmt: str, kwargs: dict) -> None:
    loaded = load_image(encode(solid((80, 60)), fmt, **kwargs))
    assert loaded.source_format == fmt
    assert loaded.image.mode == "RGB"
    assert (loaded.width, loaded.height) == (80, 60) == loaded.image.size
    r, g, b = loaded.image.getpixel((40, 30))
    assert abs(r - 200) <= 3 and abs(g - 30) <= 3 and abs(b - 30) <= 3


def test_extension_and_name_are_irrelevant() -> None:
    # load_image only sees bytes; a PNG payload is a PNG regardless of how it was named.
    assert load_image(encode(solid(), "PNG")).source_format == "PNG"


def test_sniff_format() -> None:
    assert sniff_format(encode(solid(), "JPEG")) == "JPEG"
    assert sniff_format(encode(solid(), "PNG")) == "PNG"
    assert sniff_format(encode(solid(), "WEBP")) == "WEBP"
    assert sniff_format(encode(solid(), "GIF")) is None
    assert sniff_format(b"") is None


def test_rgba_png_is_composited_on_white() -> None:
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # fully transparent red
    img.paste((0, 0, 255, 255), (0, 0, 32, 64))  # opaque blue left half
    img.paste((0, 0, 0, 128), (32, 0, 64, 32))  # half-transparent black, top-right
    loaded = load_image(encode(img, "PNG"))
    assert loaded.had_alpha
    assert loaded.image.getpixel((10, 10)) == (0, 0, 255)
    assert loaded.image.getpixel((50, 50)) == (255, 255, 255)
    assert all(120 <= c <= 135 for c in loaded.image.getpixel((50, 10)))


def test_rgba_webp_is_composited_on_white() -> None:
    img = Image.new("RGBA", (64, 64), (0, 255, 0, 0))
    loaded = load_image(encode(img, "WEBP", lossless=True))
    assert loaded.image.getpixel((5, 5)) == (255, 255, 255)


def test_palette_with_transparency() -> None:
    img = Image.new("P", (64, 64), 0)
    img.putpalette([0, 0, 0, 10, 200, 10] + [0] * 762)
    img.paste(1, (0, 0, 32, 64))
    loaded = load_image(encode(img, "PNG", transparency=0))
    assert loaded.had_alpha
    assert loaded.image.getpixel((10, 10)) == (10, 200, 10)
    assert loaded.image.getpixel((50, 10)) == (255, 255, 255)


@pytest.mark.parametrize("mode", ["L", "LA"])
def test_grayscale_becomes_rgb(mode: str) -> None:
    color = 77 if mode == "L" else (77, 255)
    loaded = load_image(encode(solid((70, 70), color, mode), "PNG"))
    assert loaded.image.mode == "RGB"
    assert loaded.image.getpixel((5, 5)) == (77, 77, 77)


def test_grayscale_jpeg() -> None:
    loaded = load_image(encode(solid((70, 70), 128, "L"), "JPEG", quality=100))
    assert loaded.image.mode == "RGB"
    assert all(abs(c - 128) <= 2 for c in loaded.image.getpixel((5, 5)))


def test_16bit_grayscale_png_is_scaled_not_clipped() -> None:
    arr = np.full((70, 70), 65535, dtype=np.uint16)
    arr[:, :35] = 32896  # ~= 128 * 257
    loaded = load_image(encode(Image.fromarray(arr), "PNG"))
    assert loaded.image.getpixel((60, 5)) == (255, 255, 255)
    assert loaded.image.getpixel((5, 5)) == (128, 128, 128)


def test_cmyk_jpeg_becomes_rgb() -> None:
    loaded = load_image(encode(solid((70, 70), (0, 255, 255, 0), "CMYK"), "JPEG", quality=100))
    r, g, b = loaded.image.getpixel((5, 5))
    assert loaded.image.mode == "RGB" and r > 200 and g < 60 and b < 60


def test_multi_picture_jpeg_uses_primary_image() -> None:
    first, second = solid((70, 70), (250, 0, 0)), solid((70, 70), (0, 0, 250))
    data = encode(first, "MPO", save_all=True, append_images=[second], quality=95)
    loaded = load_image(data)
    assert loaded.source_format == "JPEG"
    r, _, b = loaded.image.getpixel((5, 5))
    assert r > 200 and b < 50


# --- EXIF orientation and metadata ----------------------------------------------------------


def _exif_bytes(orientation: int, with_gps: bool = False) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    if with_gps:
        exif[0x8825] = {1: "N", 2: (55.0, 45.0, 0.0)}  # GPSInfo IFD
    return exif.tobytes()


def test_exif_orientation_6_rotates_clockwise() -> None:
    # Stored 80x40 (red left, blue right); orientation 6 = rotate 90° CW to display.
    data = encode(two_halves((80, 40)), "JPEG", quality=95, exif=_exif_bytes(6))
    loaded = load_image(data)
    assert loaded.orientation_applied
    assert loaded.image.size == (40, 80)
    top, bottom = loaded.image.getpixel((20, 10)), loaded.image.getpixel((20, 70))
    assert top[0] > 200 and top[2] < 60  # red moved to the top
    assert bottom[2] > 200 and bottom[0] < 60


def test_exif_orientation_3_rotates_180() -> None:
    loaded = load_image(encode(two_halves((80, 40)), "JPEG", quality=95, exif=_exif_bytes(3)))
    assert loaded.image.size == (80, 40)
    left = loaded.image.getpixel((5, 20))
    assert left[2] > 200 and left[0] < 60  # blue is now on the left


def test_no_exif_means_no_orientation_change() -> None:
    loaded = load_image(encode(two_halves((80, 40)), "PNG"))
    assert not loaded.orientation_applied and loaded.image.size == (80, 40)


def test_metadata_is_stripped() -> None:
    data = encode(solid((80, 60)), "JPEG", exif=_exif_bytes(1, with_gps=True))
    loaded = load_image(data)
    assert loaded.image.info == {}
    assert len(loaded.image.getexif()) == 0


# --- refusals ---------------------------------------------------------------------------------


def test_empty_file() -> None:
    assert _code(b"") == "empty_file"


def test_oversized_bytes() -> None:
    data = encode(solid((200, 200)), "PNG")
    assert _code(data, ImageLimits(max_bytes=len(data) - 1)) == "file_too_large"
    assert load_image(data, ImageLimits(max_bytes=len(data))).width == 200


def test_default_byte_limit_is_10_mib() -> None:
    assert _code(b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024)) == "file_too_large"


def test_custom_pixel_limit() -> None:
    data = encode(solid((100, 100)), "PNG")
    assert _code(data, ImageLimits(max_pixels=9_999)) == "too_many_pixels"
    assert load_image(data, ImageLimits(max_pixels=10_000)).width == 100


def test_declared_canvas_over_project_limit_is_rejected_before_decode() -> None:
    assert _code(png_with_declared_size(5000, 5000)) == "too_many_pixels"  # 25 MP > 20 MP


def test_pillow_decompression_bomb_error() -> None:
    limits = ImageLimits(max_pixels=10**12)
    assert _code(png_with_declared_size(20000, 20000), limits) == "decompression_bomb"


def test_pillow_decompression_bomb_warning_is_refusal() -> None:
    limits = ImageLimits(max_pixels=10**12)
    # 100 MP: above Pillow's warning threshold, below its hard error threshold.
    assert _code(png_with_declared_size(10000, 10000), limits) == "decompression_bomb"


def test_truncated_jpeg() -> None:
    data = encode(two_halves((300, 300)), "JPEG", quality=95)
    assert _code(data[: len(data) // 2]) == "corrupt_image"


def test_truncated_png() -> None:
    data = encode(Image.effect_noise((300, 300), 50).convert("RGB"), "PNG")
    assert _code(data[: len(data) // 2]) == "corrupt_image"


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xd8\xff" + b"garbage" * 50,
        b"\x89PNG\r\n\x1a\n" + b"garbage" * 50,
        b"RIFF\x00\x00\x00\x00WEBP" + b"garbage" * 50,
    ],
)
def test_garbage_with_valid_magic(data: bytes) -> None:
    assert _code(data) == "corrupt_image"


def test_animated_png_is_rejected() -> None:
    frames = [solid((64, 64), (255, 0, 0)), solid((64, 64), (0, 255, 0))]
    data = encode(frames[0], "PNG", save_all=True, append_images=frames[1:])
    assert _code(data) == "animated_image"


def test_animated_webp_is_rejected() -> None:
    frames = [solid((64, 64), (255, 0, 0)), solid((64, 64), (0, 255, 0))]
    data = encode(frames[0], "WEBP", save_all=True, append_images=frames[1:], lossless=True)
    assert _code(data) == "animated_image"


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_raster_formats_are_unsupported(fmt: str) -> None:
    assert _code(encode(solid(), fmt)) == "unsupported_format"


@pytest.mark.parametrize(
    ("data", "hint"),
    [
        (b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>', "SVG"),
        (b"<svg width='10' height='10'></svg>", "SVG"),
        (b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n", "PDF"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic", "HEIC"),
        (b"just some text", "JPEG"),
    ],
)
def test_non_raster_inputs_are_unsupported(data: bytes, hint: str) -> None:
    with pytest.raises(ImageValidationError) as info:
        load_image(data)
    assert info.value.code == "unsupported_format"
    assert hint in info.value.user_message


def test_global_truncated_override_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    assert _code(encode(solid(), "PNG")) == "unsafe_decoder_settings"


def test_error_messages_do_not_leak_tracebacks() -> None:
    with pytest.raises(ImageValidationError) as info:
        load_image(b"\xff\xd8\xff" + b"x" * 100)
    assert "Traceback" not in info.value.user_message
    assert "File " not in info.value.user_message


def test_rejection_logs_contain_no_image_content(caplog: pytest.LogCaptureFixture) -> None:
    payload = b"\xff\xd8\xff" + b"SECRET-PIXELS" * 20
    with caplog.at_level("DEBUG", logger="doglens.image_io"), pytest.raises(ImageValidationError):
        load_image(payload)
    assert caplog.records, "rejections should be logged for diagnostics"
    assert "SECRET-PIXELS" not in caplog.text
