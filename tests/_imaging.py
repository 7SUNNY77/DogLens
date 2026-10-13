"""Synthetic in-memory test images. Test-only: never used by the application."""

from __future__ import annotations

import io
import struct
import zlib

from PIL import Image


def encode(img: Image.Image, fmt: str, **save_kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **save_kwargs)
    return buf.getvalue()


def solid(size: tuple[int, int] = (80, 60), color: object = (200, 30, 30), mode: str = "RGB"):
    return Image.new(mode, size, color)


def two_halves(size: tuple[int, int] = (80, 40)) -> Image.Image:
    """Left half red, right half blue: makes orientation changes observable."""
    img = Image.new("RGB", size, (255, 0, 0))
    img.paste((0, 0, 255), (size[0] // 2, 0, size[0], size[1]))
    return img


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def png_with_declared_size(width: int, height: int) -> bytes:
    """A PNG whose header declares a huge canvas but carries almost no pixel data (bomb-like)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
