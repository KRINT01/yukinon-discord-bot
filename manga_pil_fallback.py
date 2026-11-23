"""
Slim Emergency Pure-Pillow Fallback (Lazy-Loaded ONLY if OpenCV/NumPy is absent).
Replaces the legacy 62 KB duplicate pure-Python BFS detector/renderer:
- Detection returns `(resized_b64_jpeg, [])` so `manga_translator.py` seamlessly
  uses `bubble_id=null + box_2d` grounding with zero maintenance drift.
- Rendering paints clean luminance-matched rounded overlays over `box_2d` regions
  and reuses `manga_cv_render`'s exact Thai/Latin typesetter & collision resolver.
"""

import io
import base64

from PIL import Image, ImageDraw, ImageStat

from config import THAI_CHAR_REGEX
from manga_cv_render import (
    MANGA_RENDER_JPEG_QUALITY,
    _sanitize_text_for_manga_font,
    _resolve_box_collisions,
    _fit_text_binary_search,
)


def _detect_page_bubbles_and_build_som_b64_pil_sync(
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    """Encodes a clean resized JPEG with no CV badges (`[]`) when OpenCV is unavailable."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    try:
        img.thumbnail((vision_max_dim, vision_max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(vision_quality))
        return base64.b64encode(buf.getvalue()).decode("utf-8"), []
    finally:
        img.close()


def _render_translated_manga_image_pil_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
    cv_bubbles: list[dict] | None = None,
) -> bytes | None:
    """Emergency pure-Pillow box_2d rounded-pill overlay renderer when OpenCV is unavailable."""
    if not raw_bytes or not bubbles:
        return None

    cv_map: dict[int, dict] = {
        int(c["bubble_id"]): c for c in (cv_bubbles or []) if c.get("bubble_id") is not None
    }

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    try:
        width, height = img.size
        draw = ImageDraw.Draw(img)
        prepared_renders: list[list] = []

        for b in bubbles:
            tr_txt = _sanitize_text_for_manga_font(b.get("translated_text") or "")
            if not tr_txt:
                continue

            box = b.get("box_2d")
            if not (isinstance(box, list) and len(box) == 4):
                bid = b.get("bubble_id")
                if bid is not None and int(bid) in cv_map:
                    box = cv_map[int(bid)].get("bbox_norm")
            if not (isinstance(box, list) and len(box) == 4):
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin

            rx1 = max(2, int((xmin / 1000.0) * width))
            ry1 = max(2, int((ymin / 1000.0) * height))
            rx2 = min(width - 2, int((xmax / 1000.0) * width))
            ry2 = min(height - 2, int((ymax / 1000.0) * height))
            bw, bh = rx2 - rx1, ry2 - ry1
            if bw < 18 or bh < 14:
                continue

            crop_g = base_gray.crop((rx1, ry1, rx2, ry2))
            med_lum = float(ImageStat.Stat(crop_g).median[0])
            crop_g.close()

            if med_lum < 125.0:
                bg_rgb = (24, 24, 28)
                fg_rgb = (250, 250, 250)
                stroke_rgb = (12, 12, 16)
            else:
                bg_rgb = (255, 255, 255)
                fg_rgb = (15, 15, 15)
                stroke_rgb = (255, 255, 255)

            rad = max(6, min(18, min(bw, bh) // 4))
            draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=rad, fill=bg_rgb)
            hb = (rx1 + 2, ry1 + 2, rx2 - 2, ry2 - 2)
            prepared_renders.append([hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, tr_txt, hb])

        if not prepared_renders:
            return None

        prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)
        max_dim = max(width, height)
        max_font_px = max(28, int(max_dim * 0.021))

        for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text, _ in prepared_renders:
            box_w = max(18, tx2 - tx1)
            box_h = max(14, ty2 - ty1)
            usable_w = max(14, int(box_w * 0.92))
            usable_h = max(12, int(box_h * 0.92))
            is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

            start_font_size = min(max_font_px, max(14, int(min(box_w * 0.42, box_h * 0.36))))
            chosen_font, chosen_lines, line_height, ref_top = _fit_text_binary_search(
                draw, trans_text, usable_w, usable_h, start_font_size, is_thai_text
            )
            total_text_h = line_height * len(chosen_lines)
            curr_y = ty1 + max(0, (box_h - total_text_h) // 2)

            for ln in chosen_lines:
                lbbox = draw.textbbox((0, 0), ln, font=chosen_font)
                lw = lbbox[2] - lbbox[0]
                curr_x = tx1 + max(0, (box_w - lw) // 2) - lbbox[0]
                draw.text(
                    (curr_x, curr_y - ref_top),
                    ln,
                    font=chosen_font,
                    fill=fg_text,
                    stroke_width=2,
                    stroke_fill=stroke_col,
                )
                curr_y += line_height

        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", quality=MANGA_RENDER_JPEG_QUALITY, optimize=True)
        return out_io.getvalue()
    finally:
        img.close()
        base_gray.close()