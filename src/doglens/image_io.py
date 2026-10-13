"""Safe decoding of user uploads into a metadata-free, correctly oriented RGB image.

Validation order (CLAUDE.md section 6):
byte size -> format by content -> dimensions/frames -> full decode -> EXIF transpose -> RGB.

The file extension and the client-reported MIME type are never trusted. Nothing here logs
image content, EXIF/GPS or the user's file name.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile, ImageOps

from doglens.config import DEFAULT_LIMITS, MIB, ImageLimits
from doglens.errors import ImageValidationError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
# Pillow reports JPEGs carrying an MPF index (common for phone cameras) as MPO. Only the primary
# image (frame 0) is used; the auxiliary images are not animation frames.
_ACCEPTED_PIL_FORMATS = {"JPEG": {"JPEG", "MPO"}, "PNG": {"PNG"}, "WEBP": {"WEBP"}}
_WHITE = (255, 255, 255)
_ISOBMFF_IMAGE_BRANDS = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1", b"avif"}


@dataclass(frozen=True)
class LoadedImage:
    """A validated upload. ``image`` is RGB, EXIF-transposed and stripped of metadata."""

    image: Image.Image
    source_format: str
    width: int
    height: int
    orientation_applied: bool
    had_alpha: bool


def _reject(code: str, message: str, detail: str | None = None) -> ImageValidationError:
    logger.info("image rejected: code=%s detail=%s", code, detail)
    return ImageValidationError(code, message, detail)


def sniff_format(data: bytes) -> str | None:
    """Return 'JPEG' | 'PNG' | 'WEBP' from magic bytes, or None for anything else."""
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def _describe_unsupported(data: bytes) -> str:
    head = data[:1024]
    if len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in _ISOBMFF_IMAGE_BRANDS:
        return "Формат HEIC/HEIF/AVIF не поддерживается. Сохраните фото как JPEG, PNG или WebP."
    if head.startswith(b"%PDF"):
        return "PDF не поддерживается. Загрузите фотографию в формате JPEG, PNG или WebP."
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if stripped.startswith((b"<?xml", b"<svg", b"<!doctype svg")) or b"<svg" in stripped:
        return "SVG не поддерживается. Загрузите фотографию в формате JPEG, PNG или WebP."
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "GIF не поддерживается. Загрузите фотографию в формате JPEG, PNG или WebP."
    return "Неподдерживаемый формат файла. Загрузите JPEG, PNG или WebP."


def to_rgb(img: Image.Image) -> tuple[Image.Image, bool]:
    """Convert any decoded mode to 8-bit RGB; transparent pixels are composited on white."""
    mode = img.mode
    has_transparency = mode in {"RGBA", "LA", "PA", "La", "RGBa"} or "transparency" in img.info
    if has_transparency:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, _WHITE)
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background, True
    if mode == "RGB":
        return img, False
    if mode.startswith("I;16") or mode in {"I", "F"}:
        # High bit depth grayscale: scale the 16-bit range to 8 bits instead of clipping.
        arr = np.asarray(img, dtype=np.float64) / 257.0
        arr8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return Image.fromarray(arr8).convert("RGB"), False
    return img.convert("RGB"), False


def load_image(data: bytes, limits: ImageLimits = DEFAULT_LIMITS) -> LoadedImage:
    """Validate and decode uploaded bytes. Raises ImageValidationError on any refusal."""
    if not isinstance(data, bytes | bytearray | memoryview):
        raise _reject("invalid_input", "Не удалось прочитать файл.", type(data).__name__)
    data = bytes(data)

    # 1. Size in bytes.
    if len(data) == 0:
        raise _reject("empty_file", "Файл пустой.")
    if len(data) > limits.max_bytes:
        raise _reject(
            "file_too_large",
            f"Файл слишком большой: максимум {limits.max_bytes / MIB:g} МиБ.",
            f"bytes={len(data)}",
        )

    # 2. Format by content.
    fmt = sniff_format(data)
    if fmt is None:
        raise _reject("unsupported_format", _describe_unsupported(data))

    if ImageFile.LOAD_TRUNCATED_IMAGES:
        # A global override elsewhere would let truncated files through silently.
        raise _reject(
            "unsafe_decoder_settings",
            "Сервис временно не может проверить изображение.",
            "PIL.ImageFile.LOAD_TRUNCATED_IMAGES is enabled",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data), formats=[fmt])
            if img.format not in _ACCEPTED_PIL_FORMATS[fmt]:
                raise _reject(
                    "format_mismatch",
                    "Содержимое файла не соответствует формату JPEG, PNG или WebP.",
                    f"sniffed={fmt} pillow={img.format}",
                )

            # 3. Dimensions and frames, from the header, before decoding pixel data.
            width, height = img.size
            if width <= 0 or height <= 0:
                raise _reject("invalid_dimensions", "Некорректные размеры изображения.")
            if width * height > limits.max_pixels:
                raise _reject(
                    "too_many_pixels",
                    "Изображение слишком большое: максимум "
                    f"{limits.max_pixels / 1_000_000:g} млн пикселей.",
                    f"size={width}x{height}",
                )
            if fmt != "JPEG" and (
                getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1
            ):
                raise _reject(
                    "animated_image",
                    "Анимированные изображения не поддерживаются. Загрузите один кадр.",
                    f"frames={getattr(img, 'n_frames', '?')}",
                )
            if img.format == "MPO":
                img.seek(0)

            # 4. Full decode: truncated or corrupt data raises here.
            img.load()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _reject(
            "decompression_bomb",
            "Изображение отклонено: слишком большие размеры после распаковки.",
            type(exc).__name__,
        ) from None
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a refusal, not a crash
        raise _reject(
            "corrupt_image",
            "Не удалось прочитать изображение: файл повреждён или неполон.",
            type(exc).__name__,
        ) from None

    # 5. EXIF orientation.
    try:
        orientation = img.getexif().get(0x0112, 1)
        transposed = ImageOps.exif_transpose(img)
    except Exception as exc:  # noqa: BLE001
        raise _reject(
            "bad_exif",
            "Не удалось применить ориентацию из метаданных изображения.",
            type(exc).__name__,
        ) from None
    if transposed is None:  # defensive: in_place=False always returns an image
        transposed = img

    # 6. RGB, metadata stripped (no EXIF/GPS/ICC/text chunks travel further).
    rgb, had_alpha = to_rgb(transposed)
    clean = rgb.copy()
    clean.info = {}
    w, h = clean.size
    return LoadedImage(
        image=clean,
        source_format=fmt,
        width=w,
        height=h,
        orientation_applied=orientation in range(2, 9),
        had_alpha=had_alpha,
    )
