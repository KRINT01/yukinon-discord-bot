import io
import base64

from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageStat

from config import THAI_CHAR_REGEX
from manga_cv_render import (
    SOM_CV_MAX_DIM,
    MANGA_RENDER_JPEG_QUALITY,
    _load_manga_font,
    _sanitize_text_for_manga_font,
    _prune_nested_child_bubbles,
    _sort_bubbles_manga_reading_order,
    _split_text_for_two_lobes,
    _resolve_box_collisions,
    _fit_text_binary_search,
)


# ==============================================================================
# STAGE 1B: PURE-PIL FALLBACK CV DETECTOR (LAZY-LOADED ONLY IF OPENCV IS ABSENT)
# ==============================================================================
def _compute_rgb_chroma_band_pil(rgb_img: Image.Image) -> Image.Image:
    r, g, b = rgb_img.split()
    max_c = ImageChops.lighter(ImageChops.lighter(r, g), b)
    min_c = ImageChops.darker(ImageChops.darker(r, g), b)
    r.close()
    g.close()
    b.close()
    chroma = ImageChops.subtract(max_c, min_c)
    max_c.close()
    min_c.close()
    return chroma


def _extract_components_bytearray_bfs_pil(
    bin_img: Image.Image,
    min_w: int,
    min_h: int,
    max_w: int,
    max_h: int,
    max_area: int,
    step: int = 2,
    core_radius: int = 1,
    edge_pad: int = 1,
) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    sw, sh = bin_img.size
    buf = bytearray(bin_img.tobytes())
    total_len = sw * sh

    def _clear_true_margin_component(start_idx: int) -> None:
        # Probe-only BFS: only clear component if it is a genuine page margin
        # (area > max_area, >=2 edges touched, or >75% span) — preserves single-edge bubbles!
        visited = [start_idx]
        stk = [start_idx]
        buf[start_idx] = 0
        min_x = max_x = start_idx % sw
        min_y = max_y = start_idx // sw
        oversized = False
        area_cnt = 0
        while stk:
            p = stk.pop()
            area_cnt += 1
            x = p % sw
            y = p // sw
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
            if not oversized and area_cnt > max_area:
                oversized = True
                visited.clear()
            if x > 0 and buf[p - 1]:
                buf[p - 1] = 0
                stk.append(p - 1)
                if not oversized:
                    visited.append(p - 1)
            if x + 1 < sw and buf[p + 1]:
                buf[p + 1] = 0
                stk.append(p + 1)
                if not oversized:
                    visited.append(p + 1)
            if p >= sw and buf[p - sw]:
                buf[p - sw] = 0
                stk.append(p - sw)
                if not oversized:
                    visited.append(p - sw)
            if p + sw < total_len and buf[p + sw]:
                buf[p + sw] = 0
                stk.append(p + sw)
                if not oversized:
                    visited.append(p + sw)

        cw = (max_x - min_x) + 1
        ch = (max_y - min_y) + 1
        t_l, t_t = min_x <= edge_pad, min_y <= edge_pad
        t_r, t_b = (max_x + 1) >= sw - edge_pad, (max_y + 1) >= sh - edge_pad
        edge_touches = int(t_l) + int(t_t) + int(t_r) + int(t_b)
        spans_opposite = (t_l and t_r) or (t_t and t_b)
        is_margin = oversized or edge_touches >= 3 or spans_opposite or cw > int(sw * 0.75) or ch > int(sh * 0.75)
        if not is_margin:
            # Restore single-edge speech bubble pixels so main BFS extracts it
            for idx_p in visited:
                buf[idx_p] = 255

    for bx in (1, sw // 2, sw - 2):
        for by in (1, sh - 2):
            pos = by * sw + bx
            if buf[pos]:
                _clear_true_margin_component(pos)
    for by in (sh // 3, (2 * sh) // 3):
        for bx in (1, sw - 2):
            pos = by * sw + bx
            if buf[pos]:
                _clear_true_margin_component(pos)

    results: list[tuple[tuple[int, int, int, int], Image.Image]] = []
    margin = max(2, core_radius + 1)

    for sy in range(margin, sh - margin, step):
        row_off = sy * sw
        for sx in range(margin, sw - margin, step):
            pos = row_off + sx
            if not buf[pos]:
                continue
            if (
                not buf[pos - core_radius]
                or not buf[pos + core_radius]
                or not buf[pos - core_radius * sw]
                or not buf[pos + core_radius * sw]
            ):
                continue

            buf[pos] = 0
            stk = [pos]
            pixels = [pos]
            min_x = max_x = sx
            min_y = max_y = sy
            oversized = False
            area_cnt = 0

            while stk:
                p = stk.pop()
                area_cnt += 1
                x = p % sw
                y = p // sw
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                elif y > max_y:
                    max_y = y

                if not oversized:
                    if area_cnt > max_area:
                        oversized = True
                        pixels.clear()
                    else:
                        pixels.append(p)

                if x > 0 and buf[p - 1]:
                    buf[p - 1] = 0
                    stk.append(p - 1)
                if x + 1 < sw and buf[p + 1]:
                    buf[p + 1] = 0
                    stk.append(p + 1)
                if p >= sw and buf[p - sw]:
                    buf[p - sw] = 0
                    stk.append(p - sw)
                if p + sw < total_len and buf[p + sw]:
                    buf[p + sw] = 0
                    stk.append(p + sw)

            if oversized:
                continue

            cx1, cy1, cx2, cy2 = min_x, min_y, max_x + 1, max_y + 1
            cw, ch = cx2 - cx1, cy2 - cy1
            t_l, t_t = cx1 <= edge_pad, cy1 <= edge_pad
            t_r, t_b = cx2 >= sw - edge_pad, cy2 >= sh - edge_pad
            edge_touches = int(t_l) + int(t_t) + int(t_r) + int(t_b)
            if edge_touches >= 3 or (t_l and t_r) or (t_t and t_b) or cw < min_w or ch < min_h or cw > max_w or ch > max_h:
                continue

            crop_buf = bytearray(cw * ch)
            for p in pixels:
                crop_buf[(p // sw - cy1) * cw + (p % sw - cx1)] = 255
            comp_crop = Image.frombytes("L", (cw, ch), bytes(crop_buf))
            results.append(((cx1, cy1, cx2, cy2), comp_crop))

    return results


def _filter_white_components_pil(
    raw_comps: list[tuple[tuple[int, int, int, int], Image.Image]],
    small_rgb: Image.Image,
    small_gray: Image.Image,
    small_chroma: Image.Image,
    small_sat: Image.Image,
    sw: int,
    sh: int,
    already_covered: Image.Image,
    is_secondary_pass: bool = False,
    is_pastel_pass: bool = False,
) -> list[dict]:
    found: list[dict] = []
    max_area_px = int(sw * sh * 0.24)
    min_area_px = max(220, int(sw * sh * 0.0022))

    for (cx1, cy1, cx2, cy2), comp_crop in raw_comps:
        cw, ch = cx2 - cx1, cy2 - cy1

        if is_secondary_pass and not is_pastel_pass:
            cov_crop = already_covered.crop((cx1, cy1, cx2, cy2))
            ov_px = ImageChops.darker(comp_crop, cov_crop).histogram()[255]
            cov_crop.close()
            if ov_px > comp_crop.histogram()[255] * 0.25:
                comp_crop.close()
                continue

        padded = Image.new("L", (cw + 4, ch + 4), 0)
        padded.paste(comp_crop, (2, 2))
        ImageDraw.floodfill(padded, (0, 0), 128)
        holes_crop = padded.crop((2, 2, cw + 2, ch + 2)).point(lambda p: 255 if p == 0 else 0)
        padded.close()
        filled_crop = ImageChops.lighter(comp_crop, holes_crop)

        comp_area = comp_crop.histogram()[255]
        filled_area = max(1, filled_crop.histogram()[255])
        holes_area = holes_crop.histogram()[255]
        comp_crop.close()
        holes_crop.close()

        rect_fill = filled_area / float(max(1, cw * ch))
        aspect = max(cw, ch) / float(max(1, min(cw, ch)))

        if filled_area < min_area_px or filled_area > max_area_px:
            filled_crop.close()
            continue
        min_rf = 0.56 if is_pastel_pass else 0.50
        if rect_fill < min_rf or (cw > sw * 0.42 and rect_fill < 0.62):
            filled_crop.close()
            continue
        if (aspect > 4.2 and rect_fill < 0.65) or (min(cw, ch) < 14 and rect_fill < 0.62):
            filled_crop.close()
            continue
        min_holes = 6 if is_pastel_pass else 3
        if (
            holes_area < min_holes
            or (comp_area / float(filled_area)) < 0.18
            or (holes_area / float(filled_area)) > 0.78
        ):
            filled_crop.close()
            continue

        if is_pastel_pass:
            cov_crop = already_covered.crop((cx1, cy1, cx2, cy2))
            ov_px = ImageChops.darker(filled_crop, cov_crop).histogram()[255]
            cov_crop.close()
            if ov_px > filled_area * 0.65:
                filled_crop.close()
                continue

        ch_crop = small_chroma.crop((cx1, cy1, cx2, cy2))
        gray_crop = small_gray.crop((cx1, cy1, cx2, cy2))

        if not is_pastel_pass:
            ch_hist = ch_crop.histogram(mask=filled_crop)
            total_m_px = max(1, sum(ch_hist))
            chroma_ratio = sum(ch_hist[28:]) / float(total_m_px)
            mean_chroma = ImageStat.Stat(ch_crop, mask=filled_crop).mean[0]
            if chroma_ratio >= 0.20 or mean_chroma >= 18.0:
                ch_crop.close()
                gray_crop.close()
                filled_crop.close()
                continue
        else:
            sat_crop = small_sat.crop((cx1, cy1, cx2, cy2))
            s_hist = sat_crop.histogram(mask=filled_crop)
            mean_s = ImageStat.Stat(sat_crop, mask=filled_crop).mean[0]
            high_sat_ratio = sum(s_hist[95:]) / float(max(1, sum(s_hist)))
            sat_crop.close()
            if mean_s >= 75.0 or high_sat_ratio >= 0.30:
                ch_crop.close()
                gray_crop.close()
                filled_crop.close()
                continue

            bg_mask = ImageChops.darker(gray_crop.point(lambda p: 255 if p >= 170 else 0), filled_crop)
            bg_cnt = bg_mask.histogram()[255]
            if bg_cnt < int(filled_area * 0.45):
                bg_mask.close()
                ch_crop.close()
                gray_crop.close()
                filled_crop.close()
                continue
            rgb_crop = small_rgb.crop((cx1, cy1, cx2, cy2))
            bg_std = sum(ImageStat.Stat(rgb_crop, mask=bg_mask).stddev[:3]) / 3.0
            rgb_crop.close()
            bg_mask.close()
            if bg_std > 32.0:
                ch_crop.close()
                gray_crop.close()
                filled_crop.close()
                continue

        inner_core = filled_crop.filter(ImageFilter.MinFilter(5))
        if not inner_core.getbbox():
            inner_core.close()
            inner_core = filled_crop.filter(ImageFilter.MinFilter(3))

        dark_ink_bin = gray_crop.point(lambda p: 255 if p < 135 else 0)
        achrom_bin = ch_crop.point(lambda p: 255 if p < 32 else 0)
        inner_area = max(1, inner_core.histogram()[255])
        inner_ink = ImageChops.darker(ImageChops.darker(dark_ink_bin, achrom_bin), inner_core)
        inner_ink_px = inner_ink.histogram()[255]
        ink_density = inner_ink_px / float(inner_area)

        dark_ink_bin.close()
        achrom_bin.close()
        inner_ink.close()
        inner_core.close()
        ch_crop.close()
        gray_crop.close()

        if inner_ink_px < 12 or ink_density < 0.012 or ink_density > 0.45:
            filled_crop.close()
            continue

        if is_pastel_pass:
            # Close small concave rim bays after passing validation
            pad_m = Image.new("L", (cw + 8, ch + 8), 0)
            pad_m.paste(filled_crop, (4, 4))
            closed_m = pad_m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
            pad_m.close()
            filled_crop.close()
            filled_crop = closed_m.crop((4, 4, cw + 4, ch + 4))
            closed_m.close()

        already_covered.paste(255, (cx1, cy1, cx2, cy2), mask=filled_crop)
        bbox_norm = [
            int(round((cy1 / float(sh)) * 1000.0)),
            int(round((cx1 / float(sw)) * 1000.0)),
            int(round((cy2 / float(sh)) * 1000.0)),
            int(round((cx2 / float(sw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_small": filled_crop,
            "rect_fill": rect_fill,
            "is_dark": False,
            "is_pastel": bool(is_pastel_pass),
        })

    return found


def _detect_page_bubbles_and_build_som_b64_pil_sync(
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    try:
        vis_img = img.copy()
        vis_img.thumbnail((vision_max_dim, vision_max_dim), Image.Resampling.LANCZOS)
        vw, vh = vis_img.size

        scale = min(1.0, SOM_CV_MAX_DIM / float(max(vw, vh)))
        sw = max(32, int(round(vw * scale)))
        sh = max(32, int(round(vh * scale)))

        small_rgb = vis_img.resize((sw, sh), Image.Resampling.BILINEAR)
        small_gray = small_rgb.convert("L")
        small_chroma = _compute_rgb_chroma_band_pil(small_rgb)
        small_sat = small_rgb.convert("HSV").getchannel("S")

        vis_gray = vis_img.convert("L")
        vis_dark_strokes = vis_gray.point(lambda p: 255 if p < 185 else 0)
        vis_gray.close()
        small_dark_box = vis_dark_strokes.resize((sw, sh), Image.Resampling.BOX)
        vis_dark_strokes.close()

        min_w_px = max(11, int(sw * 0.042))
        min_h_px = max(11, int(sh * 0.028))
        max_w_px = int(sw * 0.70)
        max_h_px = int(sh * 0.62)
        max_area_px = int(sw * sh * 0.24)

        already_covered = Image.new("L", (sw, sh), 0)

        # Pass A1: White bubbles
        white_bin_base = small_gray.point(lambda p: 255 if p >= 232 else 0)
        raw_a1 = _extract_components_bytearray_bfs_pil(
            white_bin_base, min_w_px, min_h_px, max_w_px, max_h_px, max_area_px, step=2, core_radius=1
        )
        detected = _filter_white_components_pil(
            raw_a1, small_rgb, small_gray, small_chroma, small_sat, sw, sh, already_covered,
            is_secondary_pass=False, is_pastel_pass=False,
        )

        # Pass A2: Thin-outline white bubbles
        non_outline = small_dark_box.point(lambda p: 0 if p >= 24 else 255)
        small_dark_box.close()
        white_bin_outline = ImageChops.darker(white_bin_base, non_outline)
        white_bin_base.close()

        raw_a2 = _extract_components_bytearray_bfs_pil(
            white_bin_outline, min_w_px, min_h_px, max_w_px, max_h_px, max_area_px, step=2, core_radius=1
        )
        white_bin_outline.close()
        detected.extend(
            _filter_white_components_pil(
                raw_a2, small_rgb, small_gray, small_chroma, small_sat, sw, sh, already_covered,
                is_secondary_pass=True, is_pastel_pass=False,
            )
        )

        # Pass A3: Pastel / colored light bubbles
        g_pastel = small_gray.point(lambda p: 255 if p >= 178 else 0)
        s_pastel = small_sat.point(lambda p: 255 if p < 95 else 0)
        pastel_bin = ImageChops.darker(ImageChops.darker(g_pastel, s_pastel), non_outline)
        g_pastel.close()
        s_pastel.close()
        non_outline.close()
        pdraw = ImageDraw.Draw(pastel_bin)
        pdraw.rectangle([0, 0, sw - 1, 1], fill=0)
        pdraw.rectangle([0, sh - 2, sw - 1, sh - 1], fill=0)
        pdraw.rectangle([0, 0, 1, sh - 1], fill=0)
        pdraw.rectangle([sw - 2, 0, sw - 1, sh - 1], fill=0)

        raw_a3 = _extract_components_bytearray_bfs_pil(
            pastel_bin, min_w_px, min_h_px, max_w_px, max_h_px, max_area_px, step=2, core_radius=1, edge_pad=2
        )
        pastel_bin.close()
        detected.extend(
            _filter_white_components_pil(
                raw_a3, small_rgb, small_gray, small_chroma, small_sat, sw, sh, already_covered,
                is_secondary_pass=True, is_pastel_pass=True,
            )
        )
        small_rgb.close()

        detected = _prune_nested_child_bubbles(detected, close_pil=True)

        # Pass B: Dark bubbles
        dark_bin = small_gray.point(lambda p: 255 if p <= 95 else 0)
        raw_dark = _extract_components_bytearray_bfs_pil(
            dark_bin, min_w_px, min_h_px, int(sw * 0.50), int(sh * 0.50), max_area_px, step=3, core_radius=1
        )
        dark_bin.close()

        for (cx1, cy1, cx2, cy2), comp_crop in raw_dark:
            cw, ch = cx2 - cx1, cy2 - cy1
            padded = Image.new("L", (cw + 4, ch + 4), 0)
            padded.paste(comp_crop, (2, 2))
            ImageDraw.floodfill(padded, (0, 0), 128)
            holes_crop = padded.crop((2, 2, cw + 2, ch + 2)).point(lambda p: 255 if p == 0 else 0)
            padded.close()
            filled_crop = ImageChops.lighter(comp_crop, holes_crop)

            filled_area = max(1, filled_crop.histogram()[255])
            holes_area = holes_crop.histogram()[255]
            comp_crop.close()
            holes_crop.close()

            rect_fill = filled_area / float(max(1, cw * ch))
            if filled_area > max_area_px or holes_area < 5 or rect_fill < 0.65:
                filled_crop.close()
                continue

            cov_crop = already_covered.crop((cx1, cy1, cx2, cy2))
            ov_px = ImageChops.darker(filled_crop, cov_crop).histogram()[255]
            cov_crop.close()
            if ov_px > 20 or ov_px > filled_area * 0.04:
                filled_crop.close()
                continue

            g_crop = small_gray.crop((cx1, cy1, cx2, cy2))
            s_crop = small_sat.crop((cx1, cy1, cx2, cy2))
            s_hist = s_crop.histogram(mask=filled_crop)
            mean_s = ImageStat.Stat(s_crop, mask=filled_crop).mean[0]
            sat_ratio = sum(s_hist[28:]) / float(max(1, sum(s_hist)))

            if mean_s >= 22.0 or sat_ratio >= 0.16:
                g_crop.close()
                s_crop.close()
                filled_crop.close()
                continue

            dark_bg_mask = ImageChops.darker(g_crop.point(lambda p: 255 if p <= 135 else 0), filled_crop)
            dark_bg_px = dark_bg_mask.histogram()[255]
            dark_std = ImageStat.Stat(g_crop, mask=dark_bg_mask).stddev[0] if dark_bg_px > 0 else 99.0
            dark_bg_mask.close()
            if dark_bg_px < max(16, int(filled_area * 0.45)) or dark_std > 24.0:
                g_crop.close()
                s_crop.close()
                filled_crop.close()
                continue

            inner_core = filled_crop.filter(ImageFilter.MinFilter(5))
            if not inner_core.getbbox():
                inner_core.close()
                inner_core = filled_crop.filter(ImageFilter.MinFilter(3))
            achrom_b = s_crop.point(lambda p: 255 if p < 28 else 0)
            bright_b = g_crop.point(lambda p: 255 if p >= 210 else 0)
            inner_bright = ImageChops.darker(ImageChops.darker(bright_b, achrom_b), inner_core)
            inner_bright_px = inner_bright.histogram()[255]
            bright_txt = ImageChops.darker(ImageChops.darker(bright_b, achrom_b), filled_crop)
            bright_px = bright_txt.histogram()[255]
            inner_bright.close()
            inner_core.close()
            bright_txt.close()
            achrom_b.close()
            bright_b.close()
            g_crop.close()
            s_crop.close()

            if inner_bright_px >= 4 and 0.02 <= (bright_px / float(filled_area)) <= 0.38:
                already_covered.paste(255, (cx1, cy1, cx2, cy2), mask=filled_crop)
                bbox_norm = [
                    int(round((cy1 / float(sh)) * 1000.0)),
                    int(round((cx1 / float(sw)) * 1000.0)),
                    int(round((cy2 / float(sh)) * 1000.0)),
                    int(round((cx2 / float(sw)) * 1000.0)),
                ]
                detected.append({
                    "bbox_norm": bbox_norm,
                    "mask_small": filled_crop,
                    "rect_fill": rect_fill,
                    "is_dark": True,
                    "is_pastel": False,
                })
            else:
                filled_crop.close()

        already_covered.close()
        small_gray.close()
        small_chroma.close()
        small_sat.close()

        if len(detected) > 24:
            detected.sort(
                key=lambda d: (
                    (d["bbox_norm"][2] - d["bbox_norm"][0])
                    * (d["bbox_norm"][3] - d["bbox_norm"][1])
                    * d.get("rect_fill", 0.7)
                ),
                reverse=True,
            )
            for dropped in detected[24:]:
                dropped["mask_small"].close()
            detected = detected[:24]

        detected = _sort_bubbles_manga_reading_order(detected)
        for idx, b_obj in enumerate(detected, 1):
            b_obj["bubble_id"] = idx

        vdraw = ImageDraw.Draw(vis_img)
        badge_font_size = max(14, int(vw * 0.019))
        badge_font = _load_manga_font(badge_font_size, is_thai=False)

        for b_obj in detected:
            bid = b_obj["bubble_id"]
            ymin, xmin, ymax, xmax = b_obj["bbox_norm"]
            vx1 = max(0, int((xmin / 1000.0) * vw))
            vy1 = max(0, int((ymin / 1000.0) * vh))
            vx2 = min(vw - 1, int((xmax / 1000.0) * vw))
            vy2 = min(vh - 1, int((ymax / 1000.0) * vh))

            m_w = max(2, vx2 - vx1)
            m_h = max(2, vy2 - vy1)
            m_up = b_obj["mask_small"].resize((m_w, m_h), Image.Resampling.NEAREST)
            m_erode = m_up.filter(ImageFilter.MinFilter(3))
            contour = ImageChops.difference(m_up, m_erode)
            m_up.close()
            m_erode.close()
            red_patch = Image.new("RGB", (m_w, m_h), (235, 20, 20))
            vis_img.paste(red_patch, (vx1, vy1), mask=contour)
            red_patch.close()
            contour.close()

            label = str(bid)
            tb = vdraw.textbbox((0, 0), label, font=badge_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            pw = tw + 8
            ph = th + 6
            bx2 = min(vw - 3, vx2 + 2)
            bx1 = max(2, bx2 - pw)
            if vy1 < ph:
                by1 = min(vh - ph - 2, max(2, vy1 + 2))
                by2 = by1 + ph
            else:
                by2 = min(vh - 2, max(ph + 2, vy1 + 6))
                by1 = max(2, by2 - ph)
            vdraw.rectangle([max(0, bx1 - 2), max(0, by1 - 2), min(vw - 1, bx2 + 2), min(vh - 1, by2 + 2)], fill=(15, 15, 15))
            vdraw.rectangle([bx1 - 1, by1 - 1, bx2 + 1, by2 + 1], fill=(255, 255, 255))
            vdraw.rectangle([bx1, by1, bx2, by2], fill=(225, 18, 18))
            vdraw.text(
                (bx1 + 4 - tb[0], by1 + 3 - tb[1]),
                label,
                font=badge_font,
                fill=(255, 255, 255),
            )

        buf = io.BytesIO()
        vis_img.save(buf, format="JPEG", quality=vision_quality)
        vis_img.close()
        b64_som = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_som, detected
    finally:
        img.close()


# ==============================================================================
# STAGE 3B: PURE-PIL FALLBACK RENDERER (LAZY-LOADED ONLY IF OPENCV IS ABSENT)
# ==============================================================================
def _fast_dilate_mask_half_res_pil(mask_l: Image.Image, kernel_half: int = 3) -> Image.Image:
    ww, wh = mask_l.size
    if ww < 24 or wh < 24:
        return mask_l.filter(ImageFilter.MaxFilter(3))
    sw, sh = max(8, ww // 2), max(8, wh // 2)
    small = mask_l.resize((sw, sh), Image.Resampling.BILINEAR)
    dil = small.filter(ImageFilter.MaxFilter(kernel_half))
    small.close()
    up = dil.resize((ww, wh), Image.Resampling.BILINEAR)
    dil.close()
    out = up.point(lambda p: 255 if p >= 64 else 0)
    up.close()
    return out


def _build_true_chroma_art_shield_pil(
    sub_rgb: Image.Image,
    chroma_thresh: int = 34,
    is_dark: bool = False,
) -> Image.Image:
    chroma_band = _compute_rgb_chroma_band_pil(sub_rgb)
    ww, wh = chroma_band.size
    if is_dark:
        sat_band = sub_rgb.convert("HSV").getchannel("S")
        c_bin = chroma_band.point(lambda p: 255 if p >= 24 else 0)
        s_bin = sat_band.point(lambda p: 255 if p >= 26 else 0)
        chroma_band.close()
        sat_band.close()
        combined = ImageChops.lighter(c_bin, s_bin)
        c_bin.close()
        s_bin.close()
        if not combined.getbbox():
            combined.close()
            return Image.new("L", (ww, wh), 255)
        shield = _fast_dilate_mask_half_res_pil(combined, kernel_half=3)
        combined.close()
        non_chroma = ImageChops.invert(shield)
        shield.close()
        return non_chroma

    if ww >= 24 and wh >= 24:
        sw, sh = max(8, ww // 2), max(8, wh // 2)
        small_c = chroma_band.resize((sw, sh), Image.Resampling.BILINEAR)
        chroma_band.close()
        bin_c = small_c.point(lambda p: 255 if p >= chroma_thresh else 0)
        small_c.close()
        if not bin_c.getbbox():
            bin_c.close()
            return Image.new("L", (ww, wh), 255)
        opened = bin_c.filter(ImageFilter.MinFilter(3))
        bin_c.close()
        shield_small = opened.filter(ImageFilter.MaxFilter(5))
        opened.close()
        up = shield_small.resize((ww, wh), Image.Resampling.BILINEAR)
        shield_small.close()
        non_chroma = up.point(lambda p: 0 if p >= 64 else 255)
        up.close()
        return non_chroma

    bin_c = chroma_band.point(lambda p: 255 if p >= chroma_thresh else 0)
    chroma_band.close()
    if not bin_c.getbbox():
        bin_c.close()
        return Image.new("L", (ww, wh), 255)
    shield = bin_c.filter(ImageFilter.MaxFilter(3))
    bin_c.close()
    non_chroma = ImageChops.invert(shield)
    shield.close()
    return non_chroma


def _largest_inscribed_rect_in_mask_pil(mask_l: Image.Image) -> tuple[int, int, int, int] | None:
    mw, mh = mask_l.size
    if mw < 4 or mh < 4:
        return None

    scale = min(1.0, 96.0 / float(max(mw, mh)))
    gw = max(6, int(round(mw * scale)))
    gh = max(6, int(round(mh * scale)))
    grid = mask_l.resize((gw, gh), Image.Resampling.NEAREST) if scale < 1.0 else mask_l
    data = grid.tobytes()
    if scale < 1.0:
        grid.close()

    heights = [0] * (gw + 1)
    best_score = -1.0
    best_rect = None

    for y in range(gh):
        row_off = y * gw
        for x in range(gw):
            heights[x] = (heights[x] + 1) if data[row_off + x] >= 128 else 0
        heights[gw] = 0

        stack: list[int] = []
        for x in range(gw + 1):
            cur_h = heights[x]
            while stack and heights[stack[-1]] > cur_h:
                h_val = heights[stack.pop()]
                left_x = (stack[-1] + 1) if stack else 0
                w_val = x - left_x
                if w_val >= 3 and h_val >= 3:
                    area = w_val * h_val
                    min_side = min(w_val, h_val)
                    score = area * (0.75 + 0.25 * (min_side / float(max(w_val, h_val))))
                    if score > best_score:
                        best_score = score
                        best_rect = (left_x, y - h_val + 1, x, y + 1)
            stack.append(x)

    if best_rect is None:
        return None

    rx1, ry1, rx2, ry2 = best_rect
    return (
        max(0, int(round(rx1 / scale))),
        max(0, int(round(ry1 / scale))),
        min(mw, int(round(rx2 / scale))),
        min(mh, int(round(ry2 / scale))),
    )


def _detect_two_lobe_render_rects_pil(
    place_mask: Image.Image,
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, int, int]] | None:
    mw, mh = place_mask.size
    if mw < 100 and mh < 100:
        return None

    scale = min(1.0, 96.0 / float(max(mw, mh)))
    gw = max(12, int(round(mw * scale)))
    gh = max(12, int(round(mh * scale)))
    grid = place_mask.resize((gw, gh), Image.Resampling.NEAREST)
    data = grid.tobytes()

    area = grid.histogram()[255]
    if area / float(max(1, gw * gh)) >= 0.74:
        grid.close()
        return None

    best_split = None
    if mw >= 150 and (mw / float(max(1, mh))) >= 1.05:
        col_counts = [0] * gw
        for y in range(gh):
            row_off = y * gw
            for x in range(gw):
                if data[row_off + x] >= 128:
                    col_counts[x] += 1
        s_lo, s_hi = max(4, int(gw * 0.25)), min(gw - 4, int(gw * 0.75))
        if s_hi > s_lo:
            min_x = min(range(s_lo, s_hi), key=lambda ix: col_counts[ix])
            min_h = col_counts[min_x]
            l_max = max(col_counts[:min_x]) if min_x > 0 else 0
            r_max = max(col_counts[min_x + 1 :]) if (min_x + 1) < gw else 0
            peak_min = min(l_max, r_max)
            if peak_min >= 18 and min_h <= peak_min * 0.74:
                best_split = ("v", min_x, min_h / float(peak_min))

    if mh >= 150 and (mh / float(max(1, mw))) >= 1.05:
        row_counts = [0] * gh
        for y in range(gh):
            row_off = y * gw
            cnt = 0
            for x in range(gw):
                if data[row_off + x] >= 128:
                    cnt += 1
            row_counts[y] = cnt
        s_lo, s_hi = max(4, int(gh * 0.25)), min(gh - 4, int(gh * 0.75))
        if s_hi > s_lo:
            min_y = min(range(s_lo, s_hi), key=lambda iy: row_counts[iy])
            min_w = row_counts[min_y]
            t_max = max(row_counts[:min_y]) if min_y > 0 else 0
            b_max = max(row_counts[min_y + 1 :]) if (min_y + 1) < gh else 0
            peak_min = min(t_max, b_max)
            if peak_min >= 18 and min_w <= peak_min * 0.74:
                ratio = min_w / float(peak_min)
                if best_split is None or ratio < best_split[2]:
                    best_split = ("h", min_y, ratio)

    grid.close()
    if best_split is None:
        return None

    axis, cut_g, _ = best_split
    cut_full = int(round(cut_g / scale))

    if axis == "v":
        right_m = place_mask.crop((cut_full, 0, mw, mh))
        left_m = place_mask.crop((0, 0, cut_full, mh))
        r_rect = _largest_inscribed_rect_in_mask_pil(right_m)
        l_rect = _largest_inscribed_rect_in_mask_pil(left_m)
        right_m.close()
        left_m.close()
        if not (r_rect and l_rect):
            return None
        r1 = (offset_x + cut_full + r_rect[0] + 2, offset_y + r_rect[1] + 2, offset_x + cut_full + r_rect[2] - 2, offset_y + r_rect[3] - 2)
        r2 = (offset_x + l_rect[0] + 2, offset_y + l_rect[1] + 2, offset_x + l_rect[2] - 2, offset_y + l_rect[3] - 2)
    else:
        top_m = place_mask.crop((0, 0, mw, cut_full))
        bot_m = place_mask.crop((0, cut_full, mw, mh))
        t_rect = _largest_inscribed_rect_in_mask_pil(top_m)
        b_rect = _largest_inscribed_rect_in_mask_pil(bot_m)
        top_m.close()
        bot_m.close()
        if not (t_rect and b_rect):
            return None
        r1 = (offset_x + t_rect[0] + 2, offset_y + t_rect[1] + 2, offset_x + t_rect[2] - 2, offset_y + t_rect[3] - 2)
        r2 = (offset_x + b_rect[0] + 2, offset_y + cut_full + b_rect[1] + 2, offset_x + b_rect[2] - 2, offset_y + cut_full + b_rect[3] - 2)

    if (r1[2] - r1[0]) < 48 or (r1[3] - r1[1]) < 36 or (r2[2] - r2[0]) < 48 or (r2[3] - r2[1]) < 36:
        return None
    return [r1, r2]


def _refine_cv_bubble_mask_locally_pil(
    img: Image.Image,
    cv_obj: dict,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    other_cv_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int, Image.Image, float]:
    width, height = img.size
    bw, bh = x2 - x1, y2 - y1
    orig_fill = float(cv_obj.get("rect_fill") or 0.70)

    def _fallback_orig():
        m_full = cv_obj["mask_small"].resize((bw, bh), Image.Resampling.BILINEAR).point(
            lambda p: 255 if p >= 128 else 0
        )
        return x1, y1, x2, y2, m_full, orig_fill

    if bool(cv_obj.get("is_dark")) or bool(cv_obj.get("is_pastel")) or bw < 16 or bh < 14:
        return _fallback_orig()

    pad_x = min(int(width * 0.12), max(24, int(bw * 0.35)))
    pad_y = min(int(height * 0.16), max(32, int(bh * 0.65)))
    wx1 = max(0, x1 - pad_x)
    wy1 = max(0, y1 - pad_y)
    wx2 = min(width, x2 + pad_x)
    wy2 = min(height, y2 + pad_y)
    ww, wh = wx2 - wx1, wy2 - wy1

    scale = min(1.0, 220.0 / float(max(ww, wh)))
    sw = max(16, int(round(ww * scale)))
    sh = max(16, int(round(wh * scale)))

    win_full_rgb = img.crop((wx1, wy1, wx2, wy2))
    win_full_gray = win_full_rgb.convert("L")
    win_dark_full = win_full_gray.point(lambda p: 255 if p < 185 else 0)
    win_full_gray.close()
    win_dark_box = win_dark_full.resize((sw, sh), Image.Resampling.BOX)
    win_dark_full.close()

    win_rgb = win_full_rgb.resize((sw, sh), Image.Resampling.BILINEAR)
    win_full_rgb.close()
    win_gray = win_rgb.convert("L")
    win_chroma = _compute_rgb_chroma_band_pil(win_rgb)
    win_rgb.close()

    g_ok = win_gray.point(lambda p: 255 if p >= 220 else 0)
    c_ok = win_chroma.point(lambda p: 255 if p < 16 else 0)
    d_ok = win_dark_box.point(lambda p: 0 if p >= 24 else 255)
    win_gray.close()
    win_chroma.close()
    win_dark_box.close()
    local_bin = ImageChops.darker(ImageChops.darker(g_ok, c_ok), d_ok)
    g_ok.close()
    c_ok.close()
    d_ok.close()

    ox_s = int(round((x1 - wx1) * scale))
    oy_s = int(round((y1 - wy1) * scale))
    ow_s = max(4, int(round(bw * scale)))
    oh_s = max(4, int(round(bh * scale)))
    orig_small = cv_obj["mask_small"].resize((ow_s, oh_s), Image.Resampling.NEAREST)
    orig_area_s = max(1, orig_small.histogram()[255])

    seeds = []
    for fy in (0.50, 0.30, 0.70):
        sy_o = min(oh_s - 1, max(0, int(oh_s * fy)))
        for fx in (0.50, 0.30, 0.70):
            sx_o = min(ow_s - 1, max(0, int(ow_s * fx)))
            if orig_small.getpixel((sx_o, sy_o)) >= 128:
                sx_l = min(sw - 2, max(1, ox_s + sx_o))
                sy_l = min(sh - 2, max(1, oy_s + sy_o))
                if local_bin.getpixel((sx_l, sy_l)) == 255:
                    seeds.append((sx_l, sy_l))
    orig_small.close()

    if not seeds:
        local_bin.close()
        return _fallback_orig()

    work = local_bin.copy()
    local_bin.close()
    for sx, sy in seeds:
        if work.getpixel((sx, sy)) == 255:
            ImageDraw.floodfill(work, (sx, sy), 128)

    comp = work.point(lambda p: 255 if p == 128 else 0)
    work.close()
    c_bb = comp.getbbox()
    if not c_bb:
        comp.close()
        return _fallback_orig()

    cx1, cy1, cx2, cy2 = c_bb
    cw_s, ch_s = cx2 - cx1, cy2 - cy1
    edge_touches = (
        (1 if cx1 <= 1 else 0)
        + (1 if cy1 <= 1 else 0)
        + (1 if cx2 >= sw - 1 else 0)
        + (1 if cy2 >= sh - 1 else 0)
    )
    if edge_touches >= 2 or (cx1 <= 1 and cx2 >= sw - 1) or (cy1 <= 1 and cy2 >= sh - 1):
        comp.close()
        return _fallback_orig()

    comp_crop = comp.crop(c_bb)
    comp.close()
    padded = Image.new("L", (cw_s + 4, ch_s + 4), 0)
    padded.paste(comp_crop, (2, 2))
    ImageDraw.floodfill(padded, (0, 0), 128)
    holes_c = padded.crop((2, 2, cw_s + 2, ch_s + 2)).point(lambda p: 255 if p == 0 else 0)
    padded.close()
    filled_s = ImageChops.lighter(comp_crop, holes_c)
    comp_crop.close()
    holes_c.close()

    new_area_s = max(1, filled_s.histogram()[255])
    new_rect_fill = new_area_s / float(max(1, cw_s * ch_s))
    if (
        new_area_s > orig_area_s * 1.75
        or new_area_s < orig_area_s * 0.85
        or new_rect_fill < orig_fill * 0.78
    ):
        filled_s.close()
        return _fallback_orig()

    rx1 = max(0, wx1 + int(round(cx1 / scale)))
    ry1 = max(0, wy1 + int(round(cy1 / scale)))
    rx2 = min(width, wx1 + int(round(cx2 / scale)))
    ry2 = min(height, wy1 + int(round(cy2 / scale)))
    rw, rh = max(4, rx2 - rx1), max(4, ry2 - ry1)

    if other_cv_boxes:
        new_area_full = max(1, rw * rh)
        old_area_full = max(1, bw * bh)
        for ox1, oy1, ox2, oy2 in other_cv_boxes:
            other_area = max(1, (ox2 - ox1) * (oy2 - oy1))
            ov_x = max(0, min(rx2, ox2) - max(rx1, ox1))
            ov_y = max(0, min(ry2, oy2) - max(ry1, oy1))
            ov_new = (ov_x * ov_y) / float(min(new_area_full, other_area))
            if ov_new >= 0.15:
                o_ov_x = max(0, min(x2, ox2) - max(x1, ox1))
                o_ov_y = max(0, min(y2, oy2) - max(y1, oy1))
                ov_old = (o_ov_x * o_ov_y) / float(min(old_area_full, other_area))
                if ov_new > ov_old + 0.05:
                    filled_s.close()
                    return _fallback_orig()

    m_full = filled_s.resize((rw, rh), Image.Resampling.BILINEAR).point(lambda p: 255 if p >= 128 else 0)
    filled_s.close()
    return rx1, ry1, rx2, ry2, m_full, new_rect_fill


def _sample_dark_bg_rgb_and_lum_pil(
    img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> tuple[tuple[int, int, int], int, float]:
    crop_rgb = img.crop((x1, y1, x2, y2))
    crop_gray = base_gray.crop((x1, y1, x2, y2))
    bg_mask = crop_gray.point(lambda p: 255 if p < 145 else 0)
    if not bg_mask.getbbox():
        bg_mask.close()
        bg_mask = Image.new("L", crop_gray.size, 255)
    stat_rgb = ImageStat.Stat(crop_rgb, mask=bg_mask)
    stat_g = ImageStat.Stat(crop_gray, mask=bg_mask)
    bg_rgb = tuple(int(v) for v in stat_rgb.median[:3])
    bg_lum = int(stat_g.median[0])
    bg_std = float(stat_g.stddev[0])
    bg_mask.close()
    crop_rgb.close()
    crop_gray.close()
    return bg_rgb, bg_lum, bg_std


def _sample_light_bg_rgb_pil(
    img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> tuple[int, int, int]:
    crop_rgb = img.crop((x1, y1, x2, y2))
    crop_gray = base_gray.crop((x1, y1, x2, y2))
    light_mask = crop_gray.point(lambda p: 255 if p >= 175 else 0)
    if not light_mask.getbbox():
        light_mask.close()
        crop_rgb.close()
        crop_gray.close()
        return (255, 255, 255)
    stat_g = ImageStat.Stat(crop_gray, mask=light_mask)
    stat_rgb = ImageStat.Stat(crop_rgb, mask=light_mask)
    bg_rgb = tuple(int(v) for v in stat_rgb.median[:3])
    light_mask.close()
    crop_rgb.close()
    crop_gray.close()
    if stat_g.median[0] >= 244 and (max(bg_rgb) - min(bg_rgb)) <= 8:
        return (255, 255, 255)
    return bg_rgb


def _clean_exact_cv_bubble_and_build_render_slots_pil(
    img: Image.Image,
    base_gray: Image.Image,
    cv_obj: dict,
    trans_text: str,
    other_cv_boxes: list[tuple[int, int, int, int]] | None = None,
    white_exclusion_mask: Image.Image | None = None,
) -> list[list]:
    width, height = img.size
    ymin, xmin, ymax, xmax = cv_obj["bbox_norm"]
    x1 = max(0, int((xmin / 1000.0) * width))
    y1 = max(0, int((ymin / 1000.0) * height))
    x2 = min(width, int((xmax / 1000.0) * width))
    y2 = min(height, int((ymax / 1000.0) * height))
    if (x2 - x1) < 16 or (y2 - y1) < 14:
        return []

    is_dark = bool(cv_obj.get("is_dark"))
    is_pastel = bool(cv_obj.get("is_pastel"))
    rx1, ry1, rx2, ry2, mask_full, rect_fill = _refine_cv_bubble_mask_locally_pil(
        img, cv_obj, x1, y1, x2, y2, other_cv_boxes=other_cv_boxes
    )
    bw, bh = rx2 - rx1, ry2 - ry1
    if bw < 16 or bh < 14:
        mask_full.close()
        return []

    safe_core = mask_full.filter(ImageFilter.MinFilter(3))
    if not safe_core.getbbox():
        safe_core.close()
        safe_core = mask_full
    else:
        mask_full.close()

    if is_dark and white_exclusion_mask is not None:
        excl_crop = white_exclusion_mask.crop((rx1, ry1, rx2, ry2))
        inv_excl = ImageChops.invert(excl_crop)
        excl_crop.close()
        clipped_core = ImageChops.darker(safe_core, inv_excl)
        inv_excl.close()
        safe_core.close()
        safe_core = clipped_core
        if not safe_core.getbbox():
            safe_core.close()
            return []

    sub_gray = base_gray.crop((rx1, ry1, rx2, ry2))
    sub_rgb = img.crop((rx1, ry1, rx2, ry2))
    non_chroma = (
        Image.new("L", (bw, bh), 255)
        if is_pastel
        else _build_true_chroma_art_shield_pil(sub_rgb, chroma_thresh=34, is_dark=is_dark)
    )
    sub_rgb.close()

    try:
        if is_pastel:
            bg_rgb = _sample_light_bg_rgb_pil(img, base_gray, rx1, ry1, rx2, ry2)
            bg_lum = int(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
            ink_t = min(195, max(145, bg_lum - 24))
            ink_mask = sub_gray.point(lambda p: 255 if p < ink_t else 0)
            ink_dil = _fast_dilate_mask_half_res_pil(ink_mask, kernel_half=5)
            ink_mask.close()
            ink_dil2 = ink_dil.filter(ImageFilter.MaxFilter(5))
            ink_dil.close()
            erase_mask = ImageChops.darker(ink_dil2, safe_core)
            ink_dil2.close()
            fill_patch = Image.new("RGB", (bw, bh), bg_rgb)
            img.paste(fill_patch, (rx1, ry1), mask=erase_mask)
            fill_patch.close()
            fg_rgb = (15, 15, 15)
            stroke_rgb = bg_rgb
        elif is_dark:
            dark_rgb, bg_lum, _ = _sample_dark_bg_rgb_and_lum_pil(img, base_gray, rx1, ry1, rx2, ry2)
            diff_ink = sub_gray.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
            ink_dil = _fast_dilate_mask_half_res_pil(diff_ink, kernel_half=3)
            diff_ink.close()
            erase_mask = ImageChops.darker(ImageChops.darker(ink_dil, safe_core), non_chroma)
            ink_dil.close()

            fill_patch = Image.new("RGB", (bw, bh), dark_rgb)
            img.paste(fill_patch, (rx1, ry1), mask=erase_mask)
            fill_patch.close()
            fg_rgb = (250, 250, 250)
            stroke_rgb = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))
        else:
            ink_mask = sub_gray.point(lambda p: 255 if p < 246 else 0)
            ink_dil = _fast_dilate_mask_half_res_pil(ink_mask, kernel_half=3)
            ink_mask.close()
            erase_mask = ImageChops.darker(ImageChops.darker(ink_dil, safe_core), non_chroma)
            ink_dil.close()

            white_patch = Image.new("RGB", (bw, bh), (255, 255, 255))
            img.paste(white_patch, (rx1, ry1), mask=erase_mask)
            white_patch.close()
            fg_rgb = (15, 15, 15)
            stroke_rgb = (255, 255, 255)

        erased_bb = erase_mask.getbbox()
        erase_mask.close()

        place_mask = ImageChops.darker(safe_core, non_chroma)
        if not place_mask.getbbox():
            place_mask.close()
            place_mask = safe_core.copy()

        if not is_dark and len(trans_text) >= 28:
            lobe_rects = _detect_two_lobe_render_rects_pil(place_mask, rx1, ry1)
            if lobe_rects:
                r1, r2 = lobe_rects
                a1 = max(1, (r1[2] - r1[0]) * (r1[3] - r1[1]))
                a2 = max(1, (r2[2] - r2[0]) * (r2[3] - r2[1]))
                split_pair = _split_text_for_two_lobes(trans_text, a1 / float(a1 + a2))
                if split_pair:
                    t1, t2 = split_pair
                    place_mask.close()
                    return [
                        [r1[0], r1[1], r1[2], r1[3], fg_rgb, stroke_rgb, t1, r1],
                        [r2[0], r2[1], r2[2], r2[3], fg_rgb, stroke_rgb, t2, r2],
                    ]

        if (not is_dark) and rect_fill < 0.60 and erased_bb is not None:
            ex1, ey1, ex2, ey2 = erased_bb
            ew, eh = ex2 - ex1, ey2 - ey1
            if ew < bw * 0.65 or eh < bh * 0.65:
                pad_ix = max(24, int(bw * 0.14))
                pad_iy = max(16, int(bh * 0.08))
                clip_win = Image.new("L", (bw, bh), 0)
                ImageDraw.Draw(clip_win).rectangle(
                    [max(0, ex1 - pad_ix), max(0, ey1 - pad_iy), min(bw, ex2 + pad_ix), min(bh, ey2 + pad_iy)],
                    fill=255,
                )
                local_place = ImageChops.darker(place_mask, clip_win)
                clip_win.close()
                inscribed = _largest_inscribed_rect_in_mask_pil(local_place)
                local_place.close()
                if inscribed and (inscribed[2] - inscribed[0]) >= 36 and (inscribed[3] - inscribed[1]) >= 24:
                    ix1, iy1, ix2, iy2 = inscribed
                    place_mask.close()
                    hb = (rx1 + ix1 + 2, ry1 + iy1 + 2, rx1 + ix2 - 2, ry1 + iy2 - 2)
                    return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

        inscribed = _largest_inscribed_rect_in_mask_pil(place_mask)
        place_mask.close()
        if inscribed and (inscribed[2] - inscribed[0]) >= 18 and (inscribed[3] - inscribed[1]) >= 14:
            ix1, iy1, ix2, iy2 = inscribed
            hb = (rx1 + ix1 + 2, ry1 + iy1 + 2, rx1 + ix2 - 2, ry1 + iy2 - 2)
            return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

        core_bb = safe_core.getbbox() or (4, 4, bw - 4, bh - 4)
        hb = (rx1 + core_bb[0] + 3, ry1 + core_bb[1] + 3, rx1 + core_bb[2] - 3, ry1 + core_bb[3] - 3)
        return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]
    finally:
        safe_core.close()
        non_chroma.close()
        sub_gray.close()


def _clean_unmarked_open_bubble_and_measure_box_pil(
    img: Image.Image,
    base_gray: Image.Image,
    raw_x1: int,
    raw_y1: int,
    raw_x2: int,
    raw_y2: int,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    width, height = img.size
    bw, bh = raw_x2 - raw_x1, raw_y2 - raw_y1
    if bw < 18 or bh < 16:
        return None

    sub_g = base_gray.crop((raw_x1, raw_y1, raw_x2, raw_y2))
    median_lum = ImageStat.Stat(sub_g).median[0]
    box_hist = sub_g.histogram()
    total_px = max(1, sum(box_hist))

    sub_rgb_check = img.crop((raw_x1, raw_y1, raw_x2, raw_y2))
    ch_band = _compute_rgb_chroma_band_pil(sub_rgb_check)
    ch_hist = ch_band.histogram()
    mean_chroma = ImageStat.Stat(ch_band).mean[0]
    chroma_ratio = sum(ch_hist[28:]) / float(max(1, sum(ch_hist)))

    if median_lum < 125:
        ch_band.close()
        sub_g.close()
        if mean_chroma >= 18.0 or chroma_ratio >= 0.18:
            sub_rgb_check.close()
            return None
        s_band = sub_rgb_check.convert("HSV").getchannel("S")
        sub_rgb_check.close()
        s_hist = s_band.histogram()
        mean_s = ImageStat.Stat(s_band).mean[0]
        s_band.close()
        if mean_s >= 22.0 or (sum(s_hist[28:]) / float(max(1, sum(s_hist)))) >= 0.16:
            return None

        dark_rgb, bg_lum, bg_std = _sample_dark_bg_rgb_and_lum_pil(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        bright_ratio = sum(box_hist[220:]) / float(total_px)
        if bg_std > 24.0 or not (0.02 <= bright_ratio <= 0.38):
            return None

        pad = 6
        wx1, wy1 = max(0, raw_x1 - pad), max(0, raw_y1 - pad)
        wx2, wy2 = min(width, raw_x2 + pad), min(height, raw_y2 + pad)
        ww, wh = wx2 - wx1, wy2 - wy1
        sg = base_gray.crop((wx1, wy1, wx2, wy2))
        srgb_d = img.crop((wx1, wy1, wx2, wy2))
        non_chroma_d = _build_true_chroma_art_shield_pil(srgb_d, chroma_thresh=24, is_dark=True)
        srgb_d.close()
        diff = sg.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
        sg.close()
        erase_d = _fast_dilate_mask_half_res_pil(diff, kernel_half=3)
        diff.close()
        final_erase = ImageChops.darker(erase_d, non_chroma_d)
        erase_d.close()
        non_chroma_d.close()
        patch = Image.new("RGB", (ww, wh), dark_rgb)
        img.paste(patch, (wx1, wy1), mask=final_erase)
        patch.close()
        final_erase.close()
        stroke_c = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))
        return raw_x1, raw_y1, raw_x2, raw_y2, (250, 250, 250), stroke_c, (raw_x1, raw_y1, raw_x2, raw_y2)

    is_pastel_open = (mean_chroma >= 18.0 or chroma_ratio >= 0.18)

    if is_pastel_open:
        if median_lum < 170:
            ch_band.close()
            sub_g.close()
            sub_rgb_check.close()
            return None
        s_band = sub_rgb_check.convert("HSV").getchannel("S")
        s_hist = s_band.histogram()
        mean_s = ImageStat.Stat(s_band).mean[0]
        high_sat_ratio = sum(s_hist[100:]) / float(max(1, sum(s_hist)))
        s_band.close()
        if mean_s >= 85.0 or high_sat_ratio >= 0.30:
            ch_band.close()
            sub_g.close()
            sub_rgb_check.close()
            return None

        light_ratio = sum(box_hist[170:]) / float(total_px)
        ink_mask_chk = ImageChops.darker(
            sub_g.point(lambda p: 255 if p < 155 else 0),
            ch_band.point(lambda p: 255 if p < 36 else 0),
        )
        ink_ratio = ink_mask_chk.histogram()[255] / float(total_px)
        ink_mask_chk.close()
        ch_band.close()
        if light_ratio < 0.42 or ink_ratio < 0.010:
            sub_g.close()
            sub_rgb_check.close()
            return None

        bg_m = sub_g.point(lambda p: 255 if p >= 170 else 0)
        bg_std = sum(ImageStat.Stat(sub_rgb_check, mask=bg_m).stddev[:3]) / 3.0 if bg_m.getbbox() else 99.0
        bg_m.close()
        sub_g.close()
        sub_rgb_check.close()
        if bg_std > 32.0:
            return None

        pad_x, pad_y = 6, 8
        wx1, wy1 = max(0, raw_x1 - pad_x), max(0, raw_y1 - pad_y)
        wx2, wy2 = min(width, raw_x2 + pad_x), min(height, raw_y2 + pad_y)
        ww, wh = wx2 - wx1, wy2 - wy1
        bg_rgb = _sample_light_bg_rgb_pil(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        bg_lum = int(0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2])
        sg = base_gray.crop((wx1, wy1, wx2, wy2))
        srgb = img.crop((wx1, wy1, wx2, wy2))
        ssat = srgb.convert("HSV").getchannel("S")
        srgb.close()
        p_seed = ImageChops.darker(sg.point(lambda p: 255 if p >= 175 else 0), ssat.point(lambda p: 255 if p < 95 else 0))
        ssat.close()
        p_zone = _fast_dilate_mask_half_res_pil(p_seed, kernel_half=5)
        p_seed.close()
        ink_t = min(195, max(145, bg_lum - 24))
        ink_m = sg.point(lambda p: 255 if p < ink_t else 0)
        sg.close()
        ink_dil = _fast_dilate_mask_half_res_pil(ink_m, kernel_half=5)
        ink_m.close()
        erase_c = ImageChops.darker(ink_dil, p_zone)
        ink_dil.close()
        p_zone.close()
        if not erase_c.getbbox():
            erase_c.close()
            return None
        patch = Image.new("RGB", (ww, wh), bg_rgb)
        img.paste(patch, (wx1, wy1), mask=erase_c)
        patch.close()
        erase_c.close()
        return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), bg_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)

    ch_band.close()
    sub_g.close()
    sub_rgb_check.close()
    white_ratio = sum(box_hist[220:]) / float(total_px)
    ink_ratio = sum(box_hist[:155]) / float(total_px)
    if white_ratio < 0.38 or ink_ratio < 0.010:
        return None

    pad_x, pad_y = 8, 12
    wx1, wy1 = max(0, raw_x1 - pad_x), max(0, raw_y1 - pad_y)
    wx2, wy2 = min(width, raw_x2 + pad_x), min(height, raw_y2 + pad_y)
    ww, wh = wx2 - wx1, wy2 - wy1

    sg = base_gray.crop((wx1, wy1, wx2, wy2))
    srgb = img.crop((wx1, wy1, wx2, wy2))
    non_chroma = _build_true_chroma_art_shield_pil(srgb, chroma_thresh=30, is_dark=False)
    srgb.close()

    w_seed = sg.point(lambda p: 255 if p >= 228 else 0)
    w_zone = _fast_dilate_mask_half_res_pil(w_seed, kernel_half=5)
    w_seed.close()

    ink_m = sg.point(lambda p: 255 if p < 232 else 0)
    sg.close()
    ink_dil = _fast_dilate_mask_half_res_pil(ink_m, kernel_half=3)
    ink_m.close()

    safe_zone = ImageChops.darker(w_zone, non_chroma)
    w_zone.close()
    non_chroma.close()
    erase_c = ImageChops.darker(ink_dil, safe_zone)
    ink_dil.close()
    safe_zone.close()

    if not erase_c.getbbox():
        erase_c.close()
        return None

    bg_rgb = _sample_light_bg_rgb_pil(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
    patch = Image.new("RGB", (ww, wh), bg_rgb)
    img.paste(patch, (wx1, wy1), mask=erase_c)
    patch.close()
    erase_c.close()
    return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), bg_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)


def _render_translated_manga_image_pil_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
    cv_bubbles: list[dict] | None = None,
) -> bytes | None:
    cv_map: dict[int, dict] = {
        int(c["bubble_id"]): c for c in (cv_bubbles or []) if c.get("bubble_id") is not None
    }

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    width, height = img.size
    white_exclusion_mask: Image.Image | None = None

    try:
        cv_boxes_px: dict[int, tuple[int, int, int, int]] = {}
        has_dark_cv = False
        for cid, c_obj in cv_map.items():
            cy1_n, cx1_n, cy2_n, cx2_n = c_obj["bbox_norm"]
            cv_boxes_px[cid] = (
                max(0, int((cx1_n / 1000.0) * width)),
                max(0, int((cy1_n / 1000.0) * height)),
                min(width, int((cx2_n / 1000.0) * width)),
                min(height, int((cy2_n / 1000.0) * height)),
            )
            if c_obj.get("is_dark"):
                has_dark_cv = True

        if has_dark_cv:
            excl_buf = Image.new("L", (width, height), 0)
            for cid, c_obj in cv_map.items():
                if c_obj.get("is_dark") or c_obj.get("mask_small") is None:
                    continue
                bx1, by1, bx2, by2 = cv_boxes_px[cid]
                if (bx2 - bx1) >= 4 and (by2 - by1) >= 4:
                    m_up = c_obj["mask_small"].resize((bx2 - bx1, by2 - by1), Image.Resampling.NEAREST)
                    excl_buf.paste(255, (bx1, by1, bx2, by2), mask=m_up)
                    m_up.close()
            if excl_buf.getbbox():
                white_exclusion_mask = excl_buf.filter(ImageFilter.MaxFilter(5))
            excl_buf.close()

        prepared_renders: list[list] = []
        used_cv_ids: set[int] = set()
        rendered_rects: list[tuple[int, int, int, int]] = []

        for b in bubbles:
            bid = b.get("bubble_id")
            tr_txt = _sanitize_text_for_manga_font(b.get("translated_text") or "")
            if not tr_txt or bid is None or bid in used_cv_ids or bid not in cv_map:
                continue

            other_boxes = [box for cid, box in cv_boxes_px.items() if cid != bid]
            slots = _clean_exact_cv_bubble_and_build_render_slots_pil(
                img,
                base_gray,
                cv_map[bid],
                tr_txt,
                other_cv_boxes=other_boxes,
                white_exclusion_mask=white_exclusion_mask,
            )
            if not slots:
                continue
            used_cv_ids.add(bid)
            for slot in slots:
                prepared_renders.append(slot)
                rendered_rects.append(slot[7])

        for b in bubbles:
            tr_txt = _sanitize_text_for_manga_font(b.get("translated_text") or "")
            if not tr_txt:
                continue
            bid = b.get("bubble_id")
            if bid is not None and bid in used_cv_ids:
                continue

            box = b.get("box_2d")
            if not (isinstance(box, list) and len(box) == 4):
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            rx1 = max(0, int((xmin / 1000.0) * width))
            ry1 = max(0, int((ymin / 1000.0) * height))
            rx2 = min(width, int((xmax / 1000.0) * width))
            ry2 = min(height, int((ymax / 1000.0) * height))
            area_new = max(1, (rx2 - rx1) * (ry2 - ry1))

            snapped_cv_id = None
            best_ov = 0.0
            for cid, (cx1_p, cy1_p, cx2_p, cy2_p) in cv_boxes_px.items():
                if cid in used_cv_ids:
                    continue
                ov_x = max(0, min(rx2, cx2_p) - max(rx1, cx1_p))
                ov_y = max(0, min(ry2, cy2_p) - max(ry1, cy1_p))
                ov_ratio = (ov_x * ov_y) / float(min(area_new, max(1, (cx2_p - cx1_p) * (cy2_p - cy1_p))))
                if ov_ratio >= 0.40 and ov_ratio > best_ov:
                    best_ov = ov_ratio
                    snapped_cv_id = cid

            if snapped_cv_id is not None:
                other_boxes = [box for cid, box in cv_boxes_px.items() if cid != snapped_cv_id]
                slots = _clean_exact_cv_bubble_and_build_render_slots_pil(
                    img,
                    base_gray,
                    cv_map[snapped_cv_id],
                    tr_txt,
                    other_cv_boxes=other_boxes,
                    white_exclusion_mask=white_exclusion_mask,
                )
                if slots:
                    used_cv_ids.add(snapped_cv_id)
                    for slot in slots:
                        prepared_renders.append(slot)
                        rendered_rects.append(slot[7])
                    continue

            is_dup = False
            for ex1, ey1, ex2, ey2 in rendered_rects:
                ov_x = max(0, min(rx2, ex2) - max(rx1, ex1))
                ov_y = max(0, min(ry2, ey2) - max(ry1, ey1))
                if (ov_x * ov_y) / float(min(area_new, max(1, (ex2 - ex1) * (ey2 - ey1)))) >= 0.45:
                    is_dup = True
                    break
            if is_dup:
                continue

            cleaned_open = _clean_unmarked_open_bubble_and_measure_box_pil(img, base_gray, rx1, ry1, rx2, ry2)
            if cleaned_open is None:
                continue
            lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned_open
            prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
            rendered_rects.append(hard_bounds)

        if not prepared_renders:
            return None

        prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)

        max_dim = max(width, height)
        max_font_px = max(28, int(max_dim * 0.021))

        draw = ImageDraw.Draw(img)
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
        if white_exclusion_mask is not None:
            white_exclusion_mask.close()
        img.close()
        base_gray.close()