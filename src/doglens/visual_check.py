"""Render bbox / context / crop panels for manual verification of dataset preparation."""

from __future__ import annotations

from PIL import Image, ImageDraw

from doglens.schemas import Roi

PANEL_HEIGHT = 320
_GAP = 8
_CAPTION = 34
BOX_COLOR = (230, 20, 20)
CONTEXT_COLOR = (20, 200, 60)


def _stroke(size: tuple[int, int]) -> int:
    return max(2, round(max(size) / 160))


def _draw_box(img: Image.Image, box: Roi, color: tuple[int, int, int]) -> None:
    """Draw the stroke *inside* the exclusive box so it never paints outside it."""
    x0, y0, x1, y1 = box
    ImageDraw.Draw(img).rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=_stroke(img.size))


def _fit(img: Image.Image) -> Image.Image:
    scale = PANEL_HEIGHT / img.height
    return img.resize((max(1, round(img.width * scale)), PANEL_HEIGHT), Image.Resampling.LANCZOS)


def render_panel(image: Image.Image, bbox: Roi, ctx_box: Roi, caption: str) -> Image.Image:
    """Three panes: original + bbox (red); bbox + context (green); final context crop."""
    with_box = image.copy()
    _draw_box(with_box, bbox, BOX_COLOR)
    with_ctx = with_box.copy()
    _draw_box(with_ctx, ctx_box, CONTEXT_COLOR)
    panes = [_fit(with_box), _fit(with_ctx), _fit(image.crop(ctx_box))]
    width = sum(p.width for p in panes) + _GAP * (len(panes) + 1)
    canvas = Image.new("RGB", (width, PANEL_HEIGHT + _CAPTION + _GAP), (255, 255, 255))
    x = _GAP
    for pane in panes:
        canvas.paste(pane, (x, _CAPTION))
        x += pane.width + _GAP
    ImageDraw.Draw(canvas).text((_GAP, 6), caption, fill=(0, 0, 0))
    return canvas
