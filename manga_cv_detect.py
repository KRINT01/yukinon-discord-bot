import os
import re
import base64

OPENCV_THREADS = max(1, int(os.getenv("OPENCV_THREADS", "1") or "1"))

try:
    import numpy as np
    import cv2

    try:
        # Default 1 thread on Render 0.1 vCPU / 512MB RAM; scales via OPENCV_THREADS on multi-core VPS hosts
        cv2.setNumThreads(OPENCV_THREADS)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    HAS_CV2 = True
except ImportError:
    np = None
    cv2 = None
    HAS_CV2 = False

MANGA_CV_RENDER_VERSION = "v22"
SOM_CV_MAX_DIM = 360  # Fast 360px grid derived from single 1024px resize (~1-2.5s for 10 pages with OpenCV+NumPy!)
# Bundle 2 RAM & Discord upload optimization: quality=90 with standard 4:2:0 subsampling (~50% smaller than 94/4:4:4)
MANGA_RENDER_JPEG_QUALITY = 90

# Pre-allocated reusable OpenCV morphological kernels
_CV2_K3 = np.ones((3, 3), np.uint8) if HAS_CV2 else None
_CV2_K5 = np.ones((5, 5), np.uint8) if HAS_CV2 else None
_CV2_K7 = np.ones((7, 7), np.uint8) if HAS_CV2 else None
_CV2_K9 = np.ones((9, 9), np.uint8) if HAS_CV2 else None
_CV2_ELLIPSE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) if HAS_CV2 else None
_CV2_ELLIPSE_11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) if HAS_CV2 else None
_CV2_ELLIPSE_13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)) if HAS_CV2 else None
_CV2_ELLIPSE_17 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)) if HAS_CV2 else None


def _coerce_image_bytes_sync(raw_bytes_or_path: bytes | str | os.PathLike) -> bytes:
    """Loads raw image bytes from a `/tmp` disk path or returns `bytes` directly (streaming-safe)."""
    if isinstance(raw_bytes_or_path, (bytes, bytearray, memoryview)):
        return bytes(raw_bytes_or_path)
    if isinstance(raw_bytes_or_path, (str, os.PathLike)):
        path_str = os.fspath(raw_bytes_or_path)
        if not path_str or not os.path.exists(path_str):
            return b""
        with open(path_str, "rb") as f:
            return f.read()
    return b""


def _prune_nested_child_bubbles(detected: list[dict], close_pil: bool = False) -> list[dict]:
    """
    Drops tiny inner white text-glow halos (from Pass A1/A2) that sit >=70% inside a larger
    enclosing colored pastel speech bubble (`is_pastel=True`).
    Safety guard: NEVER lets a non-pastel B&W panel background prune white bubbles inside it,
    and if a pastel candidate somehow wraps >=2 independent large white speech bubbles,
    drops the enclosing panel instead of the real bubbles!
    """
    if len(detected) <= 1:
        return detected

    def _contained_ratio(child_box, parent_box) -> float:
        ay1, ax1, ay2, ax2 = child_box
        by1, bx1, by2, bx2 = parent_box
        a_area = max(1, (ay2 - ay1) * (ax2 - ax1))
        b_area = max(1, (by2 - by1) * (bx2 - bx1))
        if b_area <= a_area * 1.15:
            return 0.0
        ov_y = max(0, min(ay2, by2) - max(ay1, by1))
        ov_x = max(0, min(ax2, bx2) - max(ax1, bx1))
        return (ov_x * ov_y) / float(a_area)

    drop_indices: set[int] = set()

    # Check if any large enclosing region wraps >=2 distinct full-sized bubbles (panel leak guard!)
    for j, b in enumerate(detected):
        if not b.get("is_pastel"):
            continue
        contained_real = []
        for i, a in enumerate(detected):
            if i == j or a.get("is_pastel"):
                continue
            if _contained_ratio(a["bbox_norm"], b["bbox_norm"]) >= 0.70:
                ay1, ax1, ay2, ax2 = a["bbox_norm"]
                # Count substantial speech bubbles (not tiny narrow kanji-column halos)
                if (ax2 - ax1) >= 75 and (ay2 - ay1) >= 75 and float(a.get("rect_fill") or 0.0) >= 0.72:
                    contained_real.append(i)
        if len(contained_real) >= 2:
            drop_indices.add(j)

    # Drop tiny white text-glow halos inside surviving pastel speech bubbles
    for i, a in enumerate(detected):
        if i in drop_indices or a.get("is_pastel"):
            continue
        for j, b in enumerate(detected):
            if i == j or j in drop_indices or not b.get("is_pastel"):
                continue
            if _contained_ratio(a["bbox_norm"], b["bbox_norm"]) >= 0.70:
                drop_indices.add(i)
                break

    pruned: list[dict] = []
    for idx, item in enumerate(detected):
        if idx in drop_indices:
            if close_pil and item.get("mask_small") is not None:
                try:
                    item["mask_small"].close()
                except Exception:
                    pass
            item["mask_small"] = None
            continue
        pruned.append(item)
    return pruned


def _sort_bubbles_manga_reading_order(bubbles: list[dict]) -> list[dict]:
    """
    Sorts detected CV bubbles in natural Japanese Manga reading order:
    Top-to-Bottom panel rows, and Right-to-Left within each row.
    """
    if len(bubbles) <= 1:
        return bubbles

    ordered = sorted(bubbles, key=lambda d: (d["bbox_norm"][0], -d["bbox_norm"][3]))
    bands: list[list[dict]] = []
    for item in ordered:
        y1, _, y2, _ = item["bbox_norm"]
        placed = False
        for band in bands:
            b_y1 = min(x["bbox_norm"][0] for x in band)
            b_y2 = max(x["bbox_norm"][2] for x in band)
            ov = min(y2, b_y2) - max(y1, b_y1)
            min_h = max(1.0, min(y2 - y1, b_y2 - b_y1))
            if ov >= min_h * 0.42:
                band.append(item)
                placed = True
                break
        if not placed:
            bands.append([item])

    bands.sort(key=lambda b: min(x["bbox_norm"][0] for x in b))
    result: list[dict] = []
    for band in bands:
        band.sort(key=lambda x: (-((x["bbox_norm"][1] + x["bbox_norm"][3]) / 2.0), x["bbox_norm"][0]))
        result.extend(band)
    return result


def _release_cv_page_masks(page_cv: list[dict]) -> None:
    for c_obj in page_cv:
        m = c_obj.get("mask_small")
        if m is not None:
            try:
                m.close()
            except Exception:
                pass
        c_obj["mask_small"] = None
        c_obj["mask_np"] = None
    page_cv.clear()


# ==============================================================================
# STAGE 1A: VECTORIZED OPENCV + NUMPY CV BUBBLE & UI-LABEL DETECTOR (v21)
# ==============================================================================
def _fill_mask_holes_cv2(comp_crop: "np.ndarray") -> "np.ndarray":
    """Fills all enclosed interior holes (kanji strokes) inside a uint8 binary mask in C."""
    ch, cw = comp_crop.shape
    padded = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = comp_crop
    ff_mask = np.zeros((ch + 4, cw + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 128)
    return np.where(padded[1:-1, 1:-1] == 128, np.uint8(0), np.uint8(255))


def _close_rim_bays_cv2(
    filled_crop: "np.ndarray",
    gentle: bool = False,
    kernel_size: int | None = None,
) -> "np.ndarray":
    """
    Seals concave kanji/furigana-to-border bays on validated bubbles (e.g. `技 わざ` touching
    the diagonal right rim of a white bubble, or kanji touching gloom shading lines) using
    elliptical closing strictly inside the bubble's bounding box without bridging figure-8 waists.
    """
    ch, cw = filled_crop.shape
    if kernel_size is not None and kernel_size >= 3:
        ksz = int(kernel_size) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        pad = (ksz // 2) + 1
    else:
        kern = _CV2_ELLIPSE_11 if gentle else _CV2_ELLIPSE_13
        pad = 6 if gentle else 7
    buf = np.zeros((ch + pad * 2, cw + pad * 2), dtype=np.uint8)
    buf[pad:-pad, pad:-pad] = filled_crop
    closed = cv2.morphologyEx(buf, cv2.MORPH_CLOSE, kern)
    closed = _fill_mask_holes_cv2(closed)
    return closed[pad:-pad, pad:-pad]


def _collect_margin_labels_cv2(
    labels: "np.ndarray",
    stats: "np.ndarray",
    sw: int,
    sh: int,
    max_area_px: int,
    edge_pad: int = 1,
) -> set[int]:
    """
    Identifies ONLY true outer page-margin / gutter components (preserving corner speech
    bubbles touching 2 adjacent edges!): a border-probed label is margin only if its area
    exceeds `max_area_px`, it touches >= 3 edges, spans opposite edges, or spans >75% width/height.
    """
    margin_lbls: set[int] = {0}
    probes = []
    for bx in (edge_pad, sw // 2, sw - 1 - edge_pad):
        for by in (edge_pad, sh - 1 - edge_pad):
            probes.append((bx, by))
    for by in (sh // 3, (2 * sh) // 3):
        for bx in (edge_pad, sw - 1 - edge_pad):
            probes.append((bx, by))

    for bx, by in probes:
        lbl = int(labels[by, bx])
        if lbl <= 0 or lbl in margin_lbls:
            continue
        cx1 = int(stats[lbl, cv2.CC_STAT_LEFT])
        cy1 = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx2, cy2 = cx1 + cw, cy1 + ch
        t_l = cx1 <= edge_pad
        t_t = cy1 <= edge_pad
        t_r = cx2 >= sw - edge_pad
        t_b = cy2 >= sh - edge_pad
        et = int(t_l) + int(t_t) + int(t_r) + int(t_b)
        spans_opposite = (t_l and t_r) or (t_t and t_b)
        if area > max_area_px or et >= 3 or spans_opposite or cw > int(sw * 0.75) or ch > int(sh * 0.75):
            margin_lbls.add(lbl)
    return margin_lbls


def _extract_and_filter_white_components_cv2(
    bin_mask: "np.ndarray",
    small_bgr: "np.ndarray",
    small_gray: "np.ndarray",
    small_chroma: "np.ndarray",
    small_sat: "np.ndarray",
    sw: int,
    sh: int,
    already_covered: "np.ndarray",
    min_w: int,
    min_h: int,
    max_w: int,
    max_h: int,
    max_area_px: int,
    is_secondary_pass: bool = False,
    is_pastel_pass: bool = False,
    vis_gray: "np.ndarray | None" = None,
    vis_chroma: "np.ndarray | None" = None,
) -> list[dict]:
    """
    Vectorized `cv2.connectedComponentsWithStats` + NumPy bubble validator supporting BOTH
    classic white bubbles (Pass A1/A2) and colored/pastel light speech bubbles (Pass A3):
    - Allows corner speech bubbles (touching 2 adjacent edges); rejects >=3 edges or opposite spans.
    - Allows Pass A3 pastel bubbles to supersede tiny Pass A1 white text-glow halos inside them
      (`ov_px > filled_area * 0.65`), followed by `_prune_nested_child_bubbles()`.
    """
    found: list[dict] = []
    min_area_px = max(220, int(sw * sh * 0.0022))
    edge_pad = 2 if is_pastel_pass else 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=4)
    if num_labels <= 1:
        return found

    margin_lbls = _collect_margin_labels_cv2(labels, stats, sw, sh, max_area_px, edge_pad=edge_pad)

    for lbl in range(1, num_labels):
        if lbl in margin_lbls:
            continue
        cx1 = int(stats[lbl, cv2.CC_STAT_LEFT])
        cy1 = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        comp_area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx2 = cx1 + cw
        cy2 = cy1 + ch

        if comp_area > max_area_px or cw < min_w or ch < min_h or cw > max_w or ch > max_h:
            continue

        t_l = cx1 <= edge_pad
        t_t = cy1 <= edge_pad
        t_r = cx2 >= sw - edge_pad
        t_b = cy2 >= sh - edge_pad
        edge_touches = int(t_l) + int(t_t) + int(t_r) + int(t_b)
        if edge_touches >= 3 or (t_l and t_r) or (t_t and t_b):
            continue

        comp_crop = np.where(labels[cy1:cy2, cx1:cx2] == lbl, np.uint8(255), np.uint8(0))

        if is_secondary_pass and not is_pastel_pass:
            ov_px = int(np.count_nonzero(comp_crop & already_covered[cy1:cy2, cx1:cx2]))
            if ov_px > comp_area * 0.25:
                continue

        filled_crop = _fill_mask_holes_cv2(comp_crop)
        filled_area = max(1, int(np.count_nonzero(filled_crop)))
        holes_area = int(np.count_nonzero((filled_crop > 0) & (comp_crop == 0)))

        rect_fill = filled_area / float(max(1, cw * ch))
        aspect = max(cw, ch) / float(max(1, min(cw, ch)))

        if filled_area < min_area_px or filled_area > max_area_px:
            continue
        min_rf = 0.56 if is_pastel_pass else 0.50
        if rect_fill < min_rf or (cw > sw * 0.42 and rect_fill < 0.62):
            continue
        if (aspect > 4.2 and rect_fill < 0.65) or (min(cw, ch) < 14 and rect_fill < 0.62):
            continue
        min_holes = 6 if is_pastel_pass else 3
        if (
            holes_area < min_holes
            or (comp_area / float(filled_area)) < 0.18
            or (holes_area / float(filled_area)) > 0.78
        ):
            continue

        # Anti-False-Positive Guard 1: Convex Hull Solidity >= 0.79
        # Rejects irregular shirt sleeves, SFX outlines, and panel corners while keeping spiky shout bubbles!
        solidity = 1.0
        cnts, _ = cv2.findContours(filled_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_cnt = max(cnts, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(main_cnt))
            solidity = cv2.contourArea(main_cnt) / max(1.0, hull_area)
            if solidity < 0.79:
                continue

        if is_pastel_pass:
            ov_px = int(np.count_nonzero(filled_crop & already_covered[cy1:cy2, cx1:cx2]))
            if ov_px > filled_area * 0.65:
                continue

        c_sub = small_chroma[cy1:cy2, cx1:cx2]
        g_sub = small_gray[cy1:cy2, cx1:cx2]
        mask_bool = filled_crop > 0

        if not is_pastel_pass:
            c_vals = c_sub[mask_bool]
            chroma_ratio = float(np.count_nonzero(c_vals >= 28)) / float(max(1, c_vals.size))
            mean_chroma = float(c_vals.mean()) if c_vals.size else 255.0
            if chroma_ratio >= 0.20 or mean_chroma >= 18.0:
                continue
        else:
            s_sub = small_sat[cy1:cy2, cx1:cx2]
            s_vals = s_sub[mask_bool]
            mean_s = float(s_vals.mean()) if s_vals.size else 255.0
            mean_c_p = float(c_sub[mask_bool].mean()) if mask_bool.any() else 0.0
            high_sat_ratio = float(np.count_nonzero(s_vals >= 95)) / float(max(1, s_vals.size))
            # Pass A3 is strictly for colored/pastel bubbles (reject achromatic B&W gray panels/screentones!)
            if (mean_s < 10.0 and mean_c_p < 8.0) or mean_s >= 75.0 or high_sat_ratio >= 0.30:
                continue

            bgr_sub = small_bgr[cy1:cy2, cx1:cx2]
            bg_mask = mask_bool & (g_sub >= 170)
            if int(np.count_nonzero(bg_mask)) < int(filled_area * 0.45):
                continue
            bg_std = float(np.std(bgr_sub[bg_mask], axis=0).mean())
            if bg_std > 32.0:
                continue

        # Anti-False-Positive Guard 2: True dark achromatic ink count, density bounds, and glyph count
        # Kills blank white sleeves/skin (dens < 0.012), single-stroke SFX/folds (ccs < 3), and solid SFX blobs (dens > 0.45)
        inner_core = cv2.erode(filled_crop, _CV2_K5, iterations=1)
        if not np.any(inner_core):
            inner_core = cv2.erode(filled_crop, _CV2_K3, iterations=1)
        inner_area = max(1, int(np.count_nonzero(inner_core)))
        ink_chroma_max = 42 if is_pastel_pass else 32
        black_ink_px = int(
            np.count_nonzero(
                (inner_core > 0)
                & (g_sub < 145 if is_pastel_pass else g_sub < 135)
                & (c_sub < ink_chroma_max)
            )
        )
        ink_density = black_ink_px / float(inner_area)
        is_small_clean_oval = (filled_area < 550 and solidity >= 0.92 and rect_fill >= 0.72)
        min_blk_px = 6 if is_small_clean_oval else 12
        min_ink_dens = 0.025 if is_small_clean_oval else 0.012
        if black_ink_px < min_blk_px or ink_density < min_ink_dens or ink_density > 0.45:
            continue
        # Anti-Anime-Face Guard: irregular low-rect_fill/low-solidity white character faces have
        # sparser ink than real speech bubbles (e.g. P2/5 & P4/5 faces: rf=0.60-0.62, sol=0.81-0.82, dens<=0.035)
        if (not is_pastel_pass) and rect_fill < 0.65 and solidity < 0.86 and ink_density < 0.038:
            continue
        if (not is_pastel_pass) and filled_area > int(sw * sh * 0.035) and (rect_fill < 0.70 or solidity < 0.88) and ink_density < 0.045:
            continue

        ink_bin = np.where((inner_core > 0) & (g_sub < 145) & (c_sub < (45 if is_pastel_pass else 35)), np.uint8(255), np.uint8(0))
        num_ink_cc, _ = cv2.connectedComponents(ink_bin, connectivity=8)
        if (num_ink_cc - 1) < 3:
            continue

        # Anti-False-Positive Guard 3: Full-res (~1024px) compact text-glyph count!
        # Eliminates B&W & color anime faces/bodies (which have only 1-3 eye/mouth/crease marks inside the core)
        # while keeping 100% of real speech bubbles (including 2-char bubbles like `は？`, `…っ!!` and spiky shout bubbles!)
        glyph_count = int(num_ink_cc - 1)
        if vis_gray is not None and vis_chroma is not None:
            vh_v, vw_v = vis_gray.shape[:2]
            vy1_v = max(0, int(round((cy1 / float(sh)) * vh_v)))
            vx1_v = max(0, int(round((cx1 / float(sw)) * vw_v)))
            vy2_v = min(vh_v, int(round((cy2 / float(sh)) * vh_v)))
            vx2_v = min(vw_v, int(round((cx2 / float(sw)) * vw_v)))
            if (vx2_v - vx1_v) >= 12 and (vy2_v - vy1_v) >= 12:
                vm_full = cv2.resize(filled_crop, (vx2_v - vx1_v, vy2_v - vy1_v), interpolation=cv2.INTER_NEAREST)
                v_in = cv2.erode(vm_full, _CV2_K7, iterations=1)
                if not np.any(v_in):
                    v_in = cv2.erode(vm_full, _CV2_K5, iterations=1)
                vg_sub = vis_gray[vy1_v:vy2_v, vx1_v:vx2_v]
                vc_sub = vis_chroma[vy1_v:vy2_v, vx1_v:vx2_v]
                v_ink = np.where((v_in > 0) & (vg_sub < 145) & (vc_sub < (46 if is_pastel_pass else 38)), np.uint8(255), np.uint8(0))
                vn_cc, _, vcc_st, _ = cv2.connectedComponentsWithStats(v_ink, connectivity=8)
                reg_glyphs = 0
                dot_glyphs = 0
                max_ink_span_v = 0
                for k_cc in range(1, vn_cc):
                    gw_v = int(vcc_st[k_cc, cv2.CC_STAT_WIDTH])
                    gh_v = int(vcc_st[k_cc, cv2.CC_STAT_HEIGHT])
                    ga_v = int(vcc_st[k_cc, cv2.CC_STAT_AREA])
                    if ga_v >= 8:
                        max_ink_span_v = max(max_ink_span_v, gw_v, gh_v)
                    if 3 <= gw_v <= 40 and 3 <= gh_v <= 42 and ga_v >= 8:
                        reg_glyphs += 1
                    elif 2 <= gw_v <= 5 and 2 <= gh_v <= 5 and 4 <= ga_v <= 16:
                        dot_glyphs += 1
                glyph_count = reg_glyphs + dot_glyphs
                min_glyphs = 6 if (rect_fill < 0.62 or solidity < 0.86) else 3
                if glyph_count < min_glyphs:
                    continue
                gly_per_k = glyph_count / float(max(1, filled_area)) * 1000.0
                # Anti-Face / Anti-Hair-Gap Guard for irregular low-rect_fill (< 0.65) white shapes:
                # Rejects character faces (`gly/k ~ 4.0`) and hair/earring gaps (`gly/k ~ 7.4`, `max_span=135`)
                # across ALL resolutions (1.0x/1.5x/2.0x), while keeping 100% of real low-rf bubbles (`gly/k >= 9.7`)!
                if (not is_pastel_pass) and rect_fill < 0.65 and (gly_per_k < 8.5 or max_ink_span_v > 75):
                    continue
                if (
                    (not is_pastel_pass)
                    and filled_area > int(sw * sh * 0.035)
                    and (rect_fill < 0.70 or solidity < 0.88)
                    and gly_per_k < 8.0
                ):
                    continue

        # Seal concave rim-kanji bays ONLY AFTER passing all validation guards (keeps Page 3 調子乗/思 inside mask!)
        final_mask = _close_rim_bays_cv2(filled_crop, gentle=(not is_pastel_pass))
        already_covered[cy1:cy2, cx1:cx2] |= final_mask
        bbox_norm = [
            int(round((cy1 / float(sh)) * 1000.0)),
            int(round((cx1 / float(sw)) * 1000.0)),
            int(round((cy2 / float(sh)) * 1000.0)),
            int(round((cx2 / float(sw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_np": final_mask,
            "rect_fill": rect_fill,
            "solidity": solidity,
            "glyph_count": glyph_count,
            "is_dark": False,
            "is_pastel": bool(is_pastel_pass),
        })

    return found


def _recover_gutter_merged_white_bubbles_cv2(
    white_bin_base: "np.ndarray",
    small_gray: "np.ndarray",
    small_chroma: "np.ndarray",
    vis_gray: "np.ndarray",
    vis_chroma: "np.ndarray",
    sw: int,
    sh: int,
    already_covered: "np.ndarray",
    max_area_px: int,
) -> list[dict]:
    """
    Pass A4 (Gutter-Detached White Margin Recovery):
    When a borderless white speech oval inside a dark panel touches a thin white panel gutter
    (<=14px wide), Pass A1 merges it into the giant white page-margin component (`margin_lbls`).
    Opening ONLY the margin component with a 17x17 ellipse (after closing 7x7 kanji holes and
    trimming the 4px page edge) cleanly severs thin panel gutters and recovers the enclosed white
    oval — verified strictly by high solidity (>=0.88), dark surrounding panel (>=55%), and
    full-res 1024px black-ink glyph count (>=5).
    """
    found: list[dict] = []
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(white_bin_base, connectivity=4)
    if num_l <= 1:
        return found

    margin_lbls = _collect_margin_labels_cv2(labels, stats, sw, sh, max_area_px, edge_pad=1)
    margin_ids = [m for m in margin_lbls if m > 0]
    if not margin_ids:
        return found

    margin_mask = np.isin(labels, margin_ids).astype(np.uint8) * 255
    m_filled = cv2.morphologyEx(margin_mask, cv2.MORPH_CLOSE, _CV2_K7)
    m_filled[:4, :] = 0
    m_filled[-4:, :] = 0
    m_filled[:, :4] = 0
    m_filled[:, -4:] = 0
    m_open = cv2.morphologyEx(m_filled, cv2.MORPH_OPEN, _CV2_ELLIPSE_17)

    onl, olbls, ost, _ = cv2.connectedComponentsWithStats(m_open, connectivity=4)
    if onl <= 1:
        return found

    vh_v, vw_v = vis_gray.shape[:2]
    max_w = int(sw * 0.55)
    max_h = int(sh * 0.55)

    for ol in range(1, onl):
        ox1 = int(ost[ol, cv2.CC_STAT_LEFT])
        oy1 = int(ost[ol, cv2.CC_STAT_TOP])
        ow_ = int(ost[ol, cv2.CC_STAT_WIDTH])
        oh_ = int(ost[ol, cv2.CC_STAT_HEIGHT])
        oa = int(ost[ol, cv2.CC_STAT_AREA])
        ox2, oy2 = ox1 + ow_, oy1 + oh_

        if oa < 400 or oa > max_area_px or ow_ > max_w or oh_ > max_h:
            continue

        sub_m = np.where(olbls[oy1:oy2, ox1:ox2] == ol, np.uint8(255), np.uint8(0))
        if int(np.count_nonzero(sub_m & already_covered[oy1:oy2, ox1:ox2])) > oa * 0.20:
            continue

        rf = oa / float(max(1, ow_ * oh_))
        if rf < 0.68:
            continue
        cnts, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        mc = max(cnts, key=cv2.contourArea)
        sol = cv2.contourArea(mc) / max(1.0, cv2.contourArea(cv2.convexHull(mc)))
        if sol < 0.88:
            continue

        # Must be white achromatic interior
        c_sub = small_chroma[oy1:oy2, ox1:ox2]
        c_vals = c_sub[sub_m > 0]
        if c_vals.size == 0 or float(c_vals.mean()) >= 16.0:
            continue

        # Must sit predominantly inside a darker panel (not a blank white margin/corner!)
        px1, py1 = max(0, ox1 - 4), max(0, oy1 - 4)
        px2, py2 = min(sw, ox2 + 4), min(sh, oy2 + 4)
        fm = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
        fm[oy1 - py1 : oy2 - py1, ox1 - px1 : ox2 - px1] = sub_m
        oring = cv2.dilate(fm, _CV2_K7, iterations=1) - fm
        dark_border_ratio = float(
            np.count_nonzero((oring > 0) & (small_gray[py1:py2, px1:px2] < 140))
        ) / float(max(1, int(np.count_nonzero(oring))))
        if dark_border_ratio < 0.55:
            continue

        # Must contain real dark achromatic text ink inside eroded interior
        inn = cv2.erode(sub_m, _CV2_K5, iterations=1)
        if not np.any(inn):
            continue
        g_sub = small_gray[oy1:oy2, ox1:ox2]
        blk = int(np.count_nonzero((inn > 0) & (g_sub < 135) & (c_sub < 32)))
        dens = blk / float(max(1, int(np.count_nonzero(inn))))
        if blk < 14 or not (0.015 <= dens <= 0.42):
            continue

        # Full-res (~1024px) compact-glyph count verification
        vy1_v = max(0, int(round((oy1 / float(sh)) * vh_v)))
        vx1_v = max(0, int(round((ox1 / float(sw)) * vw_v)))
        vy2_v = min(vh_v, int(round((oy2 / float(sh)) * vh_v)))
        vx2_v = min(vw_v, int(round((ox2 / float(sw)) * vw_v)))
        if (vx2_v - vx1_v) < 16 or (vy2_v - vy1_v) < 16:
            continue
        vm_full = cv2.resize(sub_m, (vx2_v - vx1_v, vy2_v - vy1_v), interpolation=cv2.INTER_NEAREST)
        v_in = cv2.erode(vm_full, _CV2_K7, iterations=1)
        vg_sub = vis_gray[vy1_v:vy2_v, vx1_v:vx2_v]
        vc_sub = vis_chroma[vy1_v:vy2_v, vx1_v:vx2_v]
        v_ink = np.where((v_in > 0) & (vg_sub < 145) & (vc_sub < 38), np.uint8(255), np.uint8(0))
        vn_cc, _, vcc_st, _ = cv2.connectedComponentsWithStats(v_ink, connectivity=8)
        gly = sum(
            1
            for k_cc in range(1, vn_cc)
            if (3 <= int(vcc_st[k_cc, cv2.CC_STAT_WIDTH]) <= 40 and 3 <= int(vcc_st[k_cc, cv2.CC_STAT_HEIGHT]) <= 42 and int(vcc_st[k_cc, cv2.CC_STAT_AREA]) >= 8)
            or (2 <= int(vcc_st[k_cc, cv2.CC_STAT_WIDTH]) <= 5 and 2 <= int(vcc_st[k_cc, cv2.CC_STAT_HEIGHT]) <= 5 and 4 <= int(vcc_st[k_cc, cv2.CC_STAT_AREA]) <= 16)
        )
        if gly < 5:
            continue

        already_covered[oy1:oy2, ox1:ox2] |= sub_m
        bbox_norm = [
            int(round((oy1 / float(sh)) * 1000.0)),
            int(round((ox1 / float(sw)) * 1000.0)),
            int(round((oy2 / float(sh)) * 1000.0)),
            int(round((ox2 / float(sw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_np": sub_m,
            "rect_fill": rf,
            "solidity": sol,
            "glyph_count": gly,
            "is_dark": False,
            "is_pastel": False,
        })

    return found


def _extract_bw_soft_shaded_bubbles_cv2(
    small_gray: "np.ndarray",
    vis_gray: "np.ndarray",
    sw: int,
    sh: int,
    already_covered: "np.ndarray",
    min_w: int,
    min_h: int,
    max_area_px: int,
) -> list[dict]:
    """
    Pass A3-BW (runs ONLY on pure B&W manga pages):
    Detects soft-shaded / airbrush / gradient-rim vertical flashback & monologue ovals
    (`190 <= median_bg <= 244`, surrounded by a darker forest/screentone background with
    `in_lum - out_lum >= 35` and crisp vertical kanji glyphs) that fall below Pass A1's
    `gray >= 232` threshold. Marked `is_pastel=True` so Stage 3 erases ink cleanly via
    TELEA inpainting without losing the soft-gray oval gradient!
    """
    found: list[dict] = []
    vh_v, vw_v = vis_gray.shape[:2]
    # Hard-ink border mask (<140) preserves panel frames while keeping soft gray bubble interiors intact
    vis_hard_dark = np.where(vis_gray < 140, np.uint8(255), np.uint8(0))
    sdark_hard = cv2.resize(vis_hard_dark, (sw, sh), interpolation=cv2.INTER_AREA)
    del vis_hard_dark

    soft_bin = np.where((small_gray >= 195) & (sdark_hard < 28), np.uint8(255), np.uint8(0))
    soft_bin[:2, :] = 0
    soft_bin[-2:, :] = 0
    soft_bin[:, :2] = 0
    soft_bin[:, -2:] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(soft_bin, connectivity=4)
    if num_labels <= 1:
        return found

    margin_lbls = _collect_margin_labels_cv2(labels, stats, sw, sh, max_area_px, edge_pad=2)
    max_soft_w = int(sw * 0.40)
    max_soft_h = int(sh * 0.45)
    max_soft_area = int(sw * sh * 0.12)

    for lbl in range(1, num_labels):
        if lbl in margin_lbls:
            continue
        cx1 = int(stats[lbl, cv2.CC_STAT_LEFT])
        cy1 = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        comp_area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx2, cy2 = cx1 + cw, cy1 + ch

        if (
            comp_area > max_soft_area
            or cw < min_w
            or ch < min_h
            or cw > max_soft_w
            or ch > max_soft_h
            or ch < int(cw * 1.15)
        ):
            continue

        comp_crop = np.where(labels[cy1:cy2, cx1:cx2] == lbl, np.uint8(255), np.uint8(0))
        if int(np.count_nonzero(comp_crop & already_covered[cy1:cy2, cx1:cx2])) > comp_area * 0.20:
            continue

        filled_crop = _close_rim_bays_cv2(_fill_mask_holes_cv2(comp_crop))
        fa = max(1, int(np.count_nonzero(filled_crop)))
        if fa < 220 or fa > max_soft_area:
            continue

        rect_fill = fa / float(max(1, cw * ch))
        if rect_fill < 0.66:
            continue

        cnts, _ = cv2.findContours(filled_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        main_cnt = max(cnts, key=cv2.contourArea)
        solidity = cv2.contourArea(main_cnt) / max(1.0, cv2.contourArea(cv2.convexHull(main_cnt)))
        if solidity < 0.80:
            continue

        g_sub = small_gray[cy1:cy2, cx1:cx2]
        bg_mask = (filled_crop > 0) & (g_sub >= 165)
        if int(np.count_nonzero(bg_mask)) < int(fa * 0.45):
            continue
        in_bg_lum = float(np.median(g_sub[bg_mask]))
        in_bg_p95 = float(np.percentile(g_sub[bg_mask], 95))
        # Soft-shaded flashback oval (median 190..244 WITH bright white inner core p95 >= 246);
        # rejects BOTH pure-white 255 faces/panels AND flat gray cobblestone/screentone tiles (p95 <= 242)!
        if not (190.0 <= in_bg_lum <= 244.0) or in_bg_p95 < 246.0 or float(np.std(g_sub[bg_mask])) > 26.0:
            continue

        # Surrounding ring must be significantly darker than the soft-shaded oval
        px1, py1 = max(0, cx1 - 5), max(0, cy1 - 5)
        px2, py2 = min(sw, cx2 + 5), min(sh, cy2 + 5)
        full_m = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
        full_m[cy1 - py1 : cy2 - py1, cx1 - px1 : cx2 - px1] = filled_crop
        outer_ring = cv2.dilate(full_m, _CV2_K9, iterations=1) - full_m
        out_lum = float(np.median(small_gray[py1:py2, px1:px2][outer_ring > 0])) if np.any(outer_ring) else 255.0
        if out_lum > 195.0 or (in_bg_lum - out_lum) < 35.0:
            continue

        inner_core = cv2.erode(filled_crop, _CV2_K5, iterations=1)
        if not np.any(inner_core):
            inner_core = cv2.erode(filled_crop, _CV2_K3, iterations=1)
        blk = int(np.count_nonzero((inner_core > 0) & (g_sub < 135)))
        dens = blk / float(max(1, int(np.count_nonzero(inner_core))))
        if blk < 6 or dens < 0.045 or dens > 0.35:
            continue

        # Full-res (~1024px) compact-glyph verification & long line-art stroke rejection
        vy1_v = max(0, int(round((cy1 / float(sh)) * vh_v)))
        vx1_v = max(0, int(round((cx1 / float(sw)) * vw_v)))
        vy2_v = min(vh_v, int(round((cy2 / float(sh)) * vh_v)))
        vx2_v = min(vw_v, int(round((cx2 / float(sw)) * vw_v)))
        if (vx2_v - vx1_v) < 12 or (vy2_v - vy1_v) < 12:
            continue
        vm_full = cv2.resize(filled_crop, (vx2_v - vx1_v, vy2_v - vy1_v), interpolation=cv2.INTER_NEAREST)
        v_in = cv2.erode(vm_full, _CV2_K7, iterations=1)
        if not np.any(v_in):
            v_in = cv2.erode(vm_full, _CV2_K5, iterations=1)
        vg_sub = vis_gray[vy1_v:vy2_v, vx1_v:vx2_v]
        v_ink = np.where((v_in > 0) & (vg_sub < 145), np.uint8(255), np.uint8(0))
        vn_cc, _, vcc_st, _ = cv2.connectedComponentsWithStats(v_ink, connectivity=8)
        glyph_count = 0
        long_art = 0
        for k_cc in range(1, vn_cc):
            gw_v = int(vcc_st[k_cc, cv2.CC_STAT_WIDTH])
            gh_v = int(vcc_st[k_cc, cv2.CC_STAT_HEIGHT])
            ga_v = int(vcc_st[k_cc, cv2.CC_STAT_AREA])
            if (3 <= gw_v <= 40 and 3 <= gh_v <= 42 and ga_v >= 8) or (2 <= gw_v <= 5 and 2 <= gh_v <= 5 and 4 <= ga_v <= 16):
                glyph_count += 1
            elif max(gw_v, gh_v) > 48 or ga_v > 700:
                long_art += 1
        if glyph_count < 5 or long_art >= 1:
            continue

        already_covered[cy1:cy2, cx1:cx2] |= filled_crop
        bbox_norm = [
            int(round((cy1 / float(sh)) * 1000.0)),
            int(round((cx1 / float(sw)) * 1000.0)),
            int(round((cy2 / float(sh)) * 1000.0)),
            int(round((cx2 / float(sw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_np": filled_crop,
            "rect_fill": rect_fill,
            "solidity": solidity,
            "glyph_count": glyph_count,
            "is_dark": False,
            "is_pastel": True,
        })

    return found


def _detect_colored_ui_banners_cv2(
    vis_bgr: "np.ndarray",
    vis_gray: "np.ndarray",
    vis_chroma: "np.ndarray",
    vis_sat: "np.ndarray",
    covered_vis_mask: "np.ndarray",
) -> list[dict]:
    """
    Pass C-Banner (runs ONLY on color pages with 0 classic white/dark bubbles — e.g. character
    profile / infographic sheets, NEVER on standard B&W/color manga pages):
    Detects uniform medium/dark colored horizontal section banners & expression caption bars
    containing bright white text (e.g. `◆ 基本情報`, `◆ プロフィール`, `◆ 能力`, `通常`/`嬉`/`怒`/`哀`/`楽`/`照`),
    including detaching thin vertical artwork (neckties/ribbons) touching a horizontal caption bar.
    """
    vh, vw = vis_gray.shape[:2]
    col_bg_raw = (
        (vis_gray >= 40)
        & (vis_gray <= 185)
        & (vis_sat >= 45)
        & (vis_chroma >= 35)
    ).astype(np.uint8) * 255
    if not np.any(col_bg_raw):
        return []

    found: list[dict] = []

    def _validate_banner(comp: "np.ndarray", x1: int, y1: int, bw: int, bh: int) -> dict | None:
        if bw < 32 or bh < 14 or bh > int(vh * 0.065) or bw > int(vw * 0.55) or bw < int(bh * 1.4):
            return None
        filled = _fill_mask_holes_cv2(comp)
        fa = int(np.count_nonzero(filled))
        rf = fa / float(max(1, bw * bh))
        if rf < 0.72:
            return None
        if int(np.count_nonzero(filled & covered_vis_mask[y1 : y1 + bh, x1 : x1 + bw])) > fa * 0.25:
            return None
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        mc = max(cnts, key=cv2.contourArea)
        sol = cv2.contourArea(mc) / max(1.0, cv2.contourArea(cv2.convexHull(mc)))
        if sol < 0.86:
            return None

        sub_bgr = vis_bgr[y1 : y1 + bh, x1 : x1 + bw]
        sub_g = vis_gray[y1 : y1 + bh, x1 : x1 + bw]
        sub_c = vis_chroma[y1 : y1 + bh, x1 : x1 + bw]
        inner = cv2.erode(filled, _CV2_K3, iterations=1)
        if not np.any(inner):
            return None

        # Measure background color uniformity strictly inside eroded core (excludes JPEG edge halos & white text AA)
        bg_core = cv2.erode(filled, _CV2_K5, iterations=1)
        if not np.any(bg_core):
            bg_core = inner
        w_raw = np.where((bg_core > 0) & (sub_g >= 185) & (sub_c <= 55), np.uint8(255), np.uint8(0))
        w_fringe = cv2.dilate(w_raw, _CV2_K3, iterations=1) > 0
        pure_bg = (bg_core > 0) & (~w_fringe) & (comp > 0)
        if int(np.count_nonzero(pure_bg)) < max(16, int(fa * 0.20)):
            return None
        bg_std = float(np.std(sub_bgr[pure_bg], axis=0).mean())
        if bg_std > 22.0:
            return None

        # Verify crisp white achromatic text inside the banner
        w_text = np.where((inner > 0) & (sub_g >= 205) & (sub_c <= 42), np.uint8(255), np.uint8(0))
        w_cnt = int(np.count_nonzero(w_text))
        w_ratio = w_cnt / float(max(1, int(np.count_nonzero(inner))))
        if w_cnt < 20 or not (0.012 <= w_ratio <= 0.45):
            return None

        wn, _, wst, _ = cv2.connectedComponentsWithStats(w_text, connectivity=8)
        w_gly = sum(
            1
            for j in range(1, wn)
            if 2 <= int(wst[j, cv2.CC_STAT_WIDTH]) <= 32
            and 3 <= int(wst[j, cv2.CC_STAT_HEIGHT]) <= 36
            and int(wst[j, cv2.CC_STAT_AREA]) >= 6
        )
        if w_gly < 1:
            return None

        covered_vis_mask[y1 : y1 + bh, x1 : x1 + bw] |= filled
        ms_w = max(8, min(96, bw // 2))
        ms_h = max(8, min(96, bh // 2))
        mask_compact = np.where(
            cv2.resize(filled, (ms_w, ms_h), interpolation=cv2.INTER_NEAREST) >= 128,
            np.uint8(255),
            np.uint8(0),
        )
        bbox_norm = [
            int(round((y1 / float(vh)) * 1000.0)),
            int(round((x1 / float(vw)) * 1000.0)),
            int(round(((y1 + bh) / float(vh)) * 1000.0)),
            int(round(((x1 + bw) / float(vw)) * 1000.0)),
        ]
        return {
            "bbox_norm": bbox_norm,
            "mask_np": mask_compact,
            "rect_fill": rf,
            "solidity": sol,
            "glyph_count": int(w_gly),
            "is_dark": False,
            "is_pastel": False,
            "is_ui_banner": True,
        }

    n_cc, lbls, st, _ = cv2.connectedComponentsWithStats(col_bg_raw, connectivity=4)
    for k in range(1, n_cc):
        x1 = int(st[k, cv2.CC_STAT_LEFT])
        y1 = int(st[k, cv2.CC_STAT_TOP])
        bw = int(st[k, cv2.CC_STAT_WIDTH])
        bh = int(st[k, cv2.CC_STAT_HEIGHT])
        if bw < 32 or bh < 14 or bw > int(vw * 0.55) or bh > int(vh * 0.22):
            continue
        comp = np.where(lbls[y1 : y1 + bh, x1 : x1 + bw] == k, np.uint8(255), np.uint8(0))
        res = _validate_banner(comp, x1, y1, bw, bh)
        if res is not None:
            found.append(res)
            continue

        # Fallback: detach thin vertical art (e.g. blue tie/ribbon) touching a horizontal caption bar
        if bh > int(vh * 0.025):
            bar_filled = _fill_mask_holes_cv2(cv2.morphologyEx(comp, cv2.MORPH_CLOSE, _CV2_K5))
            kh_open = np.ones((5, max(17, min(41, bw // 2))), np.uint8)
            opened = cv2.morphologyEx(bar_filled, cv2.MORPH_OPEN, kh_open)
            on, ol, ost, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
            for j in range(1, on):
                ox = int(ost[j, cv2.CC_STAT_LEFT])
                oy = int(ost[j, cv2.CC_STAT_TOP])
                ow = int(ost[j, cv2.CC_STAT_WIDTH])
                oh = int(ost[j, cv2.CC_STAT_HEIGHT])
                sub_m = (
                    np.where(ol[oy : oy + oh, ox : ox + ow] == j, np.uint8(255), np.uint8(0))
                    & comp[oy : oy + oh, ox : ox + ow]
                )
                sub_res = _validate_banner(sub_m, x1 + ox, y1 + oy, ow, oh)
                if sub_res is not None:
                    found.append(sub_res)

    return found


def _detect_pastel_table_labels_cv2(
    vis_bgr: "np.ndarray",
    vis_gray: "np.ndarray",
    vis_chroma: "np.ndarray",
    vis_sat: "np.ndarray",
    covered_vis_mask: "np.ndarray",
) -> list[dict]:
    """
    Pass C-PastelLabel (runs ONLY on color pages with 0 classic white/dark bubbles):
    Detects small uniform light-pastel table row header cells (e.g. `年齢`, `身長`, `誕生日`, `所属`)
    with dark text that are too narrow/short (~20px tall at 1024px) for the 360px Pass A3 grid.
    """
    vh, vw = vis_gray.shape[:2]
    pastel_bg = (
        (vis_gray >= 185)
        & (vis_gray <= 242)
        & (vis_sat >= 18)
        & (vis_sat <= 85)
        & (vis_chroma >= 16)
    ).astype(np.uint8) * 255
    if not np.any(pastel_bg):
        return []

    found: list[dict] = []
    pn_cc, plbls, pst, _ = cv2.connectedComponentsWithStats(pastel_bg, connectivity=4)
    for k in range(1, pn_cc):
        x1 = int(pst[k, cv2.CC_STAT_LEFT])
        y1 = int(pst[k, cv2.CC_STAT_TOP])
        bw = int(pst[k, cv2.CC_STAT_WIDTH])
        bh = int(pst[k, cv2.CC_STAT_HEIGHT])
        if bw < 32 or bh < 14 or bh > int(vh * 0.06) or bw > int(vw * 0.35) or bw < int(bh * 1.4):
            continue

        comp = np.where(plbls[y1 : y1 + bh, x1 : x1 + bw] == k, np.uint8(255), np.uint8(0))
        filled = _fill_mask_holes_cv2(comp)
        fa = int(np.count_nonzero(filled))
        rf = fa / float(max(1, bw * bh))
        if rf < 0.82:
            continue
        if int(np.count_nonzero(filled & covered_vis_mask[y1 : y1 + bh, x1 : x1 + bw])) > fa * 0.25:
            continue

        sub_bgr = vis_bgr[y1 : y1 + bh, x1 : x1 + bw]
        sub_g = vis_gray[y1 : y1 + bh, x1 : x1 + bw]
        sub_c = vis_chroma[y1 : y1 + bh, x1 : x1 + bw]
        if float(np.std(sub_bgr[comp > 0], axis=0).mean()) > 16.0:
            continue

        inner = cv2.erode(filled, _CV2_K3, iterations=1)
        if not np.any(inner):
            continue
        d_text = np.where((inner > 0) & (sub_g <= 145) & (sub_c <= 45), np.uint8(255), np.uint8(0))
        d_cnt = int(np.count_nonzero(d_text))
        d_ratio = d_cnt / float(max(1, int(np.count_nonzero(inner))))
        if d_cnt < 20 or not (0.02 <= d_ratio <= 0.45):
            continue
        dn, _, dst, _ = cv2.connectedComponentsWithStats(d_text, connectivity=8)
        d_gly = sum(
            1
            for j in range(1, dn)
            if 2 <= int(dst[j, cv2.CC_STAT_WIDTH]) <= 28
            and 3 <= int(dst[j, cv2.CC_STAT_HEIGHT]) <= 32
            and int(dst[j, cv2.CC_STAT_AREA]) >= 6
        )
        if d_gly < 2:
            continue

        covered_vis_mask[y1 : y1 + bh, x1 : x1 + bw] |= filled
        ms_w = max(8, min(96, bw // 2))
        ms_h = max(8, min(96, bh // 2))
        mask_compact = np.where(
            cv2.resize(filled, (ms_w, ms_h), interpolation=cv2.INTER_NEAREST) >= 128,
            np.uint8(255),
            np.uint8(0),
        )
        bbox_norm = [
            int(round((y1 / float(vh)) * 1000.0)),
            int(round((x1 / float(vw)) * 1000.0)),
            int(round(((y1 + bh) / float(vh)) * 1000.0)),
            int(round(((x1 + bw) / float(vw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_np": mask_compact,
            "rect_fill": rf,
            "solidity": 0.96,
            "glyph_count": int(d_gly),
            "is_dark": False,
            "is_pastel": False,
            "is_pastel_label": True,
        })

    return found


def _detect_white_outlined_text_clusters_cv2(
    vis_gray: "np.ndarray",
    vis_chroma: "np.ndarray",
    is_color: bool = True,
    covered_vis_mask: "np.ndarray | None" = None,
    has_sheet_ui: bool = False,
) -> list[dict]:
    """
    Pass C fallback (runs when Passes A1/A2/A4/B find 0 classic white/dark enclosed bubbles,
    including character-sheet / CG pages where Pass A3 / C-Banner found only pills/banners):
    - Detects dark-on-white or white-halo-outlined Japanese text clusters while skipping any
      pixels already covered by Pass A3 / C-Banner / C-PastelLabel (`covered_vis_mask`).
    - When `has_sheet_ui=True` (character profile sheet with >=3 UI banners/labels), uses a
      tighter vertical kernel (`kh=max(7, int(vh*0.008))|1`) so table rows (`16歳`, `170cm`,
      `9月19日`, `私立岩戸高等学校`) and bilingual name lines (`浬 結衣` vs `Kairi Yui`) stay
      cleanly separated, and allows 3-character detail captions like `デカい`.
    - Anti-artwork glyph density guard (`gly_fill >= 0.040`) eliminates sparse wrist-cuff/skin/hair
      line-art clusters (`gly_fill ~ 0.015`) while preserving all real text blocks (`0.08-0.25`)!
    """
    vh, vw = vis_gray.shape[:2]
    white_px = (vis_gray >= (225 if is_color else 218)) & (vis_chroma <= 30)
    dark_any = (vis_gray <= 145)
    if not np.any(white_px) or not np.any(dark_any):
        return []

    ring_k = _CV2_K7 if is_color else _CV2_K5
    w_thresh = 0.42 if is_color else 0.46
    white_near = cv2.dilate(white_px.astype(np.uint8), ring_k, iterations=1) > 0
    dark_core_cand = np.where(dark_any & white_near, np.uint8(255), np.uint8(0))
    if covered_vis_mask is not None:
        dark_core_cand[covered_vis_mask > 0] = 0

    n_cc, lbls, st, _ = cv2.connectedComponentsWithStats(dark_core_cand, connectivity=8)
    if n_cc <= 1:
        return []

    glyph_mask = np.zeros((vh, vw), dtype=np.uint8)
    for k in range(1, n_cc):
        gw = int(st[k, cv2.CC_STAT_WIDTH])
        gh = int(st[k, cv2.CC_STAT_HEIGHT])
        ga = int(st[k, cv2.CC_STAT_AREA])
        if 2 <= gw <= 46 and 2 <= gh <= 48 and ga >= 6:
            x1 = int(st[k, cv2.CC_STAT_LEFT])
            y1 = int(st[k, cv2.CC_STAT_TOP])
            px1, py1 = max(0, x1 - 4), max(0, y1 - 4)
            px2, py2 = min(vw, x1 + gw + 4), min(vh, y1 + gh + 4)
            sub_c = (lbls[py1:py2, px1:px2] == k).astype(np.uint8)
            ring = cv2.dilate(sub_c, ring_k, iterations=1) - sub_c
            ring_cnt = max(1, int(np.count_nonzero(ring)))
            w_ratio = int(np.count_nonzero((ring > 0) & white_px[py1:py2, px1:px2])) / float(ring_cnt)
            if w_ratio >= w_thresh:
                glyph_mask[py1:py2, px1:px2][sub_c > 0] = 255

    if not np.any(glyph_mask):
        return []

    outlined_glyphs = cv2.dilate(glyph_mask, _CV2_K3, iterations=1)
    kw = max(15, int(vw * 0.030)) | 1
    kh = (max(7, int(vh * 0.008)) | 1) if has_sheet_ui else (max(11, int(vh * 0.015)) | 1)
    kern_grp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    grouped = cv2.dilate(outlined_glyphs, kern_grp, iterations=1)

    gn, glbls, gst, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    found: list[dict] = []
    min_gly_req = (3 if has_sheet_ui else 4) if is_color else 8
    min_th_req = 18 if has_sheet_ui else 24
    pad = 6 if has_sheet_ui else 10
    min_gly_fill = 0.045 if has_sheet_ui else 0.040

    for k in range(1, gn):
        gx1 = int(gst[k, cv2.CC_STAT_LEFT])
        gy1 = int(gst[k, cv2.CC_STAT_TOP])
        gw = int(gst[k, cv2.CC_STAT_WIDTH])
        gh = int(gst[k, cv2.CC_STAT_HEIGHT])
        gx2, gy2 = gx1 + gw, gy1 + gh
        sub_g = glyph_mask[gy1:gy2, gx1:gx2] & ((glbls[gy1:gy2, gx1:gx2] == k).astype(np.uint8) * 255)
        n_gly, _ = cv2.connectedComponents(sub_g, connectivity=8)
        n_gly -= 1
        if n_gly < min_gly_req:
            continue
        ys, xs = np.nonzero(sub_g)
        if ys.size == 0:
            continue
        tx1 = max(0, gx1 + int(xs.min()) - pad)
        ty1 = max(0, gy1 + int(ys.min()) - pad)
        tx2 = min(vw, gx1 + int(xs.max()) + pad + 1)
        ty2 = min(vh, gy1 + int(ys.max()) + pad + 1)
        tw, th = tx2 - tx1, ty2 - ty1
        if tw < 24 or th < min_th_req or (tw * th) > int(vw * vh * 0.35):
            continue
        # Anti-Artwork Density Guard: kills sparse cuff/hair/eye line-art clusters (`gly_fill ~ 0.015`)
        # while keeping real printed/outlined text blocks (`gly_fill ~ 0.08-0.25`)!
        gly_fill = float(np.count_nonzero(sub_g)) / float(max(1, tw * th))
        if gly_fill < min_gly_fill:
            continue
        # Character-sheet artwork guard: printed sheet text sits on clean white paper (`col_rat <= 0.22`);
        # rejects portrait-border hair/blouse/gap clusters (e.g. `col_rat=0.24..0.57`, `w_rat=0.25..0.50`, `n_gly <= 6`)!
        if has_sheet_ui:
            sub_g_box = vis_gray[ty1:ty2, tx1:tx2]
            sub_c_box = vis_chroma[ty1:ty2, tx1:tx2]
            col_rat = float(np.count_nonzero(sub_c_box >= 28)) / float(max(1, sub_c_box.size))
            white_rat = float(np.count_nonzero((sub_g_box >= 235) & (sub_c_box <= 18))) / float(max(1, sub_g_box.size))
            if col_rat > 0.22 or (n_gly <= 6 and white_rat < 0.55):
                continue
        if covered_vis_mask is not None:
            cov_ov = float(np.count_nonzero(covered_vis_mask[ty1:ty2, tx1:tx2])) / float(max(1, tw * th))
            if cov_ov >= 0.25:
                continue
        # On pure B&W pages, reject compact square artwork/hair/face clusters; keep real vertical text columns
        if (not is_color) and (th / float(max(1, tw))) < 1.90:
            continue

        # Store compact 64px-max solid mask for crisp SoM red outline drawing
        ms_w = max(8, min(96, tw // 2))
        ms_h = max(8, min(96, th // 2))
        mask_np = np.full((ms_h, ms_w), 255, dtype=np.uint8)
        bbox_norm = [
            int(round((ty1 / float(vh)) * 1000.0)),
            int(round((tx1 / float(vw)) * 1000.0)),
            int(round((ty2 / float(vh)) * 1000.0)),
            int(round((tx2 / float(vw)) * 1000.0)),
        ]
        found.append({
            "bbox_norm": bbox_norm,
            "mask_np": mask_np,
            "rect_fill": 0.92,
            "solidity": 0.96,
            "glyph_count": int(n_gly),
            "is_dark": False,
            "is_pastel": False,
            "is_outlined_text": True,
        })

    return found


def _detect_page_bubbles_and_build_som_b64_cv2_sync(
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    """
    Fully vectorized OpenCV + NumPy Stage 1 detector & Set-of-Mark JPEG builder (v21):
    - Pass A1 (`gray >= 232`): clean base white bubble pass (corner-bubble safe!).
    - Pass A2 (`INTER_AREA` dark-outline preservation): thin-bordered ovals on white panels.
    - Pass A3 (`gray >= 175 & (sat >= 16 | chroma >= 14) & sat < 95 & dark_box < 24`): enclosed
      OR borderless pastel/colored light speech bubbles & character-sheet pills.
    - Pass A4: Gutter-detached white ovals inside dark panels.
    - Pass B (`gray <= 95`): enclosed dark speech bubbles with strict anti-dark-panel guards.
    - Pass C-Banner & Pass C-PastelLabel (strictly on color pages with 0 classic white/dark bubbles):
      colored section banners (`◆ 基本情報`, `◆ 能力`, `通常`/`嬉`/...) and pastel table row labels (`年齢`, `身長`).
    - Pass C: White-paper / white-halo text clusters with tight row separation on character sheets.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        from manga_pil_fallback import _detect_page_bubbles_and_build_som_b64_pil_sync
        return _detect_page_bubbles_and_build_som_b64_pil_sync(raw_bytes, vision_max_dim, vision_quality)

    orig_h, orig_w = img_bgr.shape[:2]
    vis_scale = min(1.0, vision_max_dim / float(max(orig_w, orig_h)))
    vw = max(32, int(round(orig_w * vis_scale)))
    vh = max(32, int(round(orig_h * vis_scale)))
    # INTER_AREA gives anti-aliased LANCZOS-grade kanji sharpness for OCR at C speed
    vis_bgr = (
        cv2.resize(img_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
        if vis_scale < 1.0
        else img_bgr.copy()
    )
    del img_bgr

    cv_scale = min(1.0, SOM_CV_MAX_DIM / float(max(vw, vh)))
    sw = max(32, int(round(vw * cv_scale)))
    sh = max(32, int(round(vh * cv_scale)))
    small_bgr = cv2.resize(vis_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    small_chroma = small_bgr.max(axis=2) - small_bgr.min(axis=2)
    small_hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    small_sat = small_hsv[:, :, 1]
    del small_hsv

    # Zero-thickening dark-outline preservation via C box-area downscale
    vis_gray = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2GRAY)
    vis_chroma = vis_bgr.max(axis=2) - vis_bgr.min(axis=2)
    vis_dark = np.where(vis_gray < 185, np.uint8(255), np.uint8(0))
    small_dark_box = cv2.resize(vis_dark, (sw, sh), interpolation=cv2.INTER_AREA)
    del vis_dark
    page_is_color = (float(np.count_nonzero(small_chroma >= 16)) / float(max(1, small_chroma.size))) >= 0.03

    min_w_px = max(11, int(sw * 0.042))
    min_h_px = max(11, int(sh * 0.028))
    max_w_px = int(sw * 0.70)
    max_h_px = int(sh * 0.62)
    max_area_px = int(sw * sh * 0.24)

    already_covered = np.zeros((sh, sw), dtype=np.uint8)

    # Pass A1: Crisp base white bubble detection (`p >= 232`)
    white_bin_base = np.where(small_gray >= 232, np.uint8(255), np.uint8(0))
    detected = _extract_and_filter_white_components_cv2(
        white_bin_base,
        small_bgr,
        small_gray,
        small_chroma,
        small_sat,
        sw,
        sh,
        already_covered,
        min_w_px,
        min_h_px,
        max_w_px,
        max_h_px,
        max_area_px,
        is_secondary_pass=False,
        is_pastel_pass=False,
        vis_gray=vis_gray,
        vis_chroma=vis_chroma,
    )

    # Pass A2: Dark-outline-preserved pass (thin-outline ovals on white panels)
    white_bin_outline = np.where(
        (white_bin_base > 0) & (small_dark_box < 24), np.uint8(255), np.uint8(0)
    )
    detected.extend(
        _extract_and_filter_white_components_cv2(
            white_bin_outline,
            small_bgr,
            small_gray,
            small_chroma,
            small_sat,
            sw,
            sh,
            already_covered,
            min_w_px,
            min_h_px,
            max_w_px,
            max_h_px,
            max_area_px,
            is_secondary_pass=True,
            is_pastel_pass=False,
            vis_gray=vis_gray,
            vis_chroma=vis_chroma,
        )
    )

    # Pass A3: Pastel / colored light speech bubbles & borderless pastel pills — ONLY on actual color pages!
    if page_is_color:
        pastel_bin = np.where(
            (small_gray >= 175)
            & ((small_sat >= 16) | (small_chroma >= 14))
            & (small_sat < 95)
            & (small_dark_box < 24),
            np.uint8(255),
            np.uint8(0),
        )
        pastel_bin[:2, :] = 0
        pastel_bin[-2:, :] = 0
        pastel_bin[:, :2] = 0
        pastel_bin[:, -2:] = 0
        detected.extend(
            _extract_and_filter_white_components_cv2(
                pastel_bin,
                small_bgr,
                small_gray,
                small_chroma,
                small_sat,
                sw,
                sh,
                already_covered,
                min_w_px,
                min_h_px,
                max_w_px,
                max_h_px,
                max_area_px,
                is_secondary_pass=True,
                is_pastel_pass=True,
                vis_gray=vis_gray,
                vis_chroma=vis_chroma,
            )
        )
    else:
        # Pass A3-BW: Soft-shaded / gradient-rim vertical B&W flashback & monologue ovals
        detected.extend(
            _extract_bw_soft_shaded_bubbles_cv2(
                small_gray,
                vis_gray,
                sw,
                sh,
                already_covered,
                min_w_px,
                min_h_px,
                max_area_px,
            )
        )
    del small_bgr

    # Pass A4: Recover gutter-merged white speech ovals sitting inside dark panels
    detected.extend(
        _recover_gutter_merged_white_bubbles_cv2(
            white_bin_base,
            small_gray,
            small_chroma,
            vis_gray,
            vis_chroma,
            sw,
            sh,
            already_covered,
            max_area_px,
        )
    )

    # Remove tiny Pass A1 white text-glow halos nested inside larger Pass A3 pastel bubbles!
    detected = _prune_nested_child_bubbles(detected, close_pil=False)

    # Pass B: Enclosed dark speech bubbles / dark caption boxes
    dark_bin = np.where(small_gray <= 95, np.uint8(255), np.uint8(0))
    num_d, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark_bin, connectivity=4)
    if num_d > 1:
        d_margin = _collect_margin_labels_cv2(d_labels, d_stats, sw, sh, max_area_px, edge_pad=1)
        max_dw, max_dh = int(sw * 0.55), int(sh * 0.55)
        vh_d, vw_d = vis_gray.shape[:2]

        def _validate_dark_bubble_crop(
            f_crop: "np.ndarray",
            c_crop: "np.ndarray",
            bx1: int,
            by1: int,
            bw_d: int,
            bh_d: int,
        ) -> dict | None:
            bx2, by2 = bx1 + bw_d, by1 + bh_d
            if bw_d < min_w_px or bh_d < min_h_px or bw_d > max_dw or bh_d > max_dh:
                return None
            tl_d, tt_d, tr_d, tb_d = bx1 <= 1, by1 <= 1, bx2 >= sw - 1, by2 >= sh - 1
            if (int(tl_d) + int(tt_d) + int(tr_d) + int(tb_d)) >= 3 or (tl_d and tr_d) or (tt_d and tb_d):
                return None

            fa_d = max(1, int(np.count_nonzero(f_crop)))
            ha_d = int(np.count_nonzero((f_crop > 0) & (c_crop == 0)))
            rf_d = fa_d / float(max(1, bw_d * bh_d))
            if fa_d < 200 or fa_d > max_area_px or ha_d < 5 or rf_d < 0.54:
                return None

            cnts_d, _ = cv2.findContours(f_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_d:
                return None
            mc_d = max(cnts_d, key=cv2.contourArea)
            sol_d = cv2.contourArea(mc_d) / max(1.0, cv2.contourArea(cv2.convexHull(mc_d)))
            if sol_d < 0.78:
                return None

            # Guard 1: Reject dark panels that enclose/overlap already-detected light/pastel speech bubbles!
            ov_d = int(np.count_nonzero(f_crop & already_covered[by1:by2, bx1:bx2]))
            if ov_d > 20 or ov_d > fa_d * 0.04:
                return None

            m_bool = f_crop > 0
            s_sub = small_sat[by1:by2, bx1:bx2]
            g_sub = small_gray[by1:by2, bx1:bx2]
            s_vals = s_sub[m_bool]
            mean_s = float(s_vals.mean()) if s_vals.size else 255.0
            sat_ratio = float(np.count_nonzero(s_vals >= 28)) / float(max(1, s_vals.size))

            # Guard 2: True HSV Saturation rejects dark colored artwork (navy clothes, dark brown walls, shadows)
            if mean_s >= 22.0 or sat_ratio >= 0.16:
                return None

            inner_core = cv2.erode(f_crop, _CV2_K5, iterations=1)
            if not np.any(inner_core):
                inner_core = cv2.erode(f_crop, _CV2_K3, iterations=1)

            # Guard 3: Measure dark interior uniformity EXCLUDING white text + 1px AA fringe
            br_mask = np.where((g_sub >= 150) & m_bool, np.uint8(255), np.uint8(0))
            br_fringe = cv2.dilate(br_mask, _CV2_K3, iterations=1) > 0
            pure_bg = (inner_core > 0) & (~br_fringe) & (g_sub <= 135)
            if int(np.count_nonzero(pure_bg)) < 12:
                pure_bg = m_bool & (~br_fringe) & (g_sub <= 135)
            if int(np.count_nonzero(pure_bg)) < max(12, int(fa_d * 0.15)):
                return None
            bg_med_d = float(np.median(g_sub[pure_bg]))
            bg_std_d = float(np.std(g_sub[pure_bg]))
            if bg_med_d > 55.0 or bg_std_d > 24.0:
                return None

            # Guard 4: Bright achromatic text must sit strictly inside the eroded interior
            inner_bright_px = int(np.count_nonzero((inner_core > 0) & (g_sub >= 195) & (s_sub < 28)))
            bright_px = int(np.count_nonzero(m_bool & (g_sub >= 195) & (s_sub < 28)))
            if inner_bright_px < 4 or not (0.018 <= (bright_px / float(fa_d)) <= 0.38):
                return None

            # Guard 5: Full-res (~1024px) compact white-text glyph verification
            vy1_d = max(0, int(round((by1 / float(sh)) * vh_d)))
            vx1_d = max(0, int(round((bx1 / float(sw)) * vw_d)))
            vy2_d = min(vh_d, int(round((by2 / float(sh)) * vh_d)))
            vx2_d = min(vw_d, int(round((bx2 / float(sw)) * vw_d)))
            if (vx2_d - vx1_d) < 12 or (vy2_d - vy1_d) < 12:
                return None
            vm_d = cv2.resize(f_crop, (vx2_d - vx1_d, vy2_d - vy1_d), interpolation=cv2.INTER_NEAREST)
            vin_d = cv2.erode(vm_d, _CV2_K7, iterations=1)
            if not np.any(vin_d):
                vin_d = cv2.erode(vm_d, _CV2_K5, iterations=1)
            vg_d = vis_gray[vy1_d:vy2_d, vx1_d:vx2_d]
            vc_d = vis_chroma[vy1_d:vy2_d, vx1_d:vx2_d]
            v_br_d = np.where((vin_d > 0) & (vg_d >= 185) & (vc_d < 35), np.uint8(255), np.uint8(0))
            vn_cc_d, _, vst_d, _ = cv2.connectedComponentsWithStats(v_br_d, connectivity=8)
            w_gly = 0
            w_big = 0
            for k_d in range(1, vn_cc_d):
                gw_d = int(vst_d[k_d, cv2.CC_STAT_WIDTH])
                gh_d = int(vst_d[k_d, cv2.CC_STAT_HEIGHT])
                ga_d = int(vst_d[k_d, cv2.CC_STAT_AREA])
                if (3 <= gw_d <= 36 and 3 <= gh_d <= 38 and ga_d >= 8) or (2 <= gw_d <= 5 and 2 <= gh_d <= 5 and 4 <= ga_d <= 16):
                    w_gly += 1
                elif max(gw_d, gh_d) > 48 and ga_d > 220:
                    w_big += 1
            if w_gly < 3:
                return None
            if w_big >= 2 and w_gly < w_big * 3:
                return None

            already_covered[by1:by2, bx1:bx2] |= f_crop
            bbox_norm_d = [
                int(round((by1 / float(sh)) * 1000.0)),
                int(round((bx1 / float(sw)) * 1000.0)),
                int(round((by2 / float(sh)) * 1000.0)),
                int(round((bx2 / float(sw)) * 1000.0)),
            ]
            return {
                "bbox_norm": bbox_norm_d,
                "mask_np": f_crop,
                "rect_fill": rf_d,
                "solidity": sol_d,
                "glyph_count": w_gly,
                "is_dark": True,
                "is_pastel": False,
            }

        for lbl in range(1, num_d):
            if lbl in d_margin:
                continue
            cx1 = int(d_stats[lbl, cv2.CC_STAT_LEFT])
            cy1 = int(d_stats[lbl, cv2.CC_STAT_TOP])
            cw = int(d_stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(d_stats[lbl, cv2.CC_STAT_HEIGHT])
            comp_area = int(d_stats[lbl, cv2.CC_STAT_AREA])
            cx2, cy2 = cx1 + cw, cy1 + ch
            if comp_area > max_area_px or cw < min_w_px or ch < min_h_px:
                continue

            comp_crop = np.where(d_labels[cy1:cy2, cx1:cx2] == lbl, np.uint8(255), np.uint8(0))
            filled_crop = _fill_mask_holes_cv2(comp_crop)
            direct_ok = _validate_dark_bubble_crop(filled_crop, comp_crop, cx1, cy1, cw, ch)
            if direct_ok is not None:
                detected.append(direct_ok)
                continue

            # SFX / Panel-Edge Detachment & Staircase-Valley Splitter Fallback
            if int(np.count_nonzero(filled_crop)) >= 600 and cw <= int(sw * 0.72) and ch <= int(sh * 0.72):
                recovered_any = False
                opened_d = cv2.morphologyEx(filled_crop, cv2.MORPH_OPEN, _CV2_ELLIPSE_7)
                onl, olbls, ost, _ = cv2.connectedComponentsWithStats(opened_d, connectivity=4)
                for ol in range(1, onl):
                    ox1 = int(ost[ol, cv2.CC_STAT_LEFT])
                    oy1 = int(ost[ol, cv2.CC_STAT_TOP])
                    ow_d = int(ost[ol, cv2.CC_STAT_WIDTH])
                    oh_d = int(ost[ol, cv2.CC_STAT_HEIGHT])
                    oa_d = int(ost[ol, cv2.CC_STAT_AREA])
                    if oa_d < 220:
                        continue
                    sub_f = np.where(olbls[oy1 : oy1 + oh_d, ox1 : ox1 + ow_d] == ol, np.uint8(255), np.uint8(0))
                    sub_c = comp_crop[oy1 : oy1 + oh_d, ox1 : ox1 + ow_d] & sub_f
                    sub_ok = _validate_dark_bubble_crop(sub_f, sub_c, cx1 + ox1, cy1 + oy1, ow_d, oh_d)
                    if sub_ok is not None:
                        detected.append(sub_ok)
                        recovered_any = True

                rf_raw_d = float(np.count_nonzero(filled_crop)) / float(max(1, cw * ch))
                if not recovered_any and cw >= 28 and ch >= 28 and cw <= max_dw and ch <= max_dh and rf_raw_d < 0.58:
                    best_v = None
                    col_cnt = np.count_nonzero(filled_crop, axis=0)
                    s_lo, s_hi = max(6, int(cw * 0.22)), min(cw - 6, int(cw * 0.78))
                    if s_hi > s_lo:
                        mx = s_lo + int(np.argmin(col_cnt[s_lo:s_hi]))
                        pmin = min(int(col_cnt[:mx].max()), int(col_cnt[mx + 1 :].max()))
                        if pmin >= 20 and int(col_cnt[mx]) <= pmin * 0.48:
                            best_v = ("v", mx, int(col_cnt[mx]) / float(pmin))

                    row_cnt = np.count_nonzero(filled_crop, axis=1)
                    r_lo, r_hi = max(6, int(ch * 0.22)), min(ch - 6, int(ch * 0.78))
                    if r_hi > r_lo:
                        my = r_lo + int(np.argmin(row_cnt[r_lo:r_hi]))
                        pmin_r = min(int(row_cnt[:my].max()), int(row_cnt[my + 1 :].max()))
                        if pmin_r >= 20 and int(row_cnt[my]) <= pmin_r * 0.48:
                            rr = int(row_cnt[my]) / float(pmin_r)
                            if best_v is None or rr < best_v[2]:
                                best_v = ("h", my, rr)

                    if best_v is not None:
                        v_axis, v_cut, _ = best_v
                        halves = (
                            [(filled_crop[:, :v_cut], comp_crop[:, :v_cut], 0, 0), (filled_crop[:, v_cut:], comp_crop[:, v_cut:], v_cut, 0)]
                            if v_axis == "v"
                            else [(filled_crop[:v_cut, :], comp_crop[:v_cut, :], 0, 0), (filled_crop[v_cut:, :], comp_crop[v_cut:, :], 0, v_cut)]
                        )
                        for hf, hc, dx_h, dy_h in halves:
                            ys_h, xs_h = np.nonzero(hf)
                            if ys_h.size < 200:
                                continue
                            hx0, hx1 = int(xs_h.min()), int(xs_h.max()) + 1
                            hy0, hy1 = int(ys_h.min()), int(ys_h.max()) + 1
                            sub_ok = _validate_dark_bubble_crop(
                                hf[hy0:hy1, hx0:hx1],
                                hc[hy0:hy1, hx0:hx1],
                                cx1 + dx_h + hx0,
                                cy1 + dy_h + hy0,
                                hx1 - hx0,
                                hy1 - hy0,
                            )
                            if sub_ok is not None:
                                detected.append(sub_ok)

    # Pass C-Banner, Pass C-PastelLabel & Pass C White-Paper / White-Halo Text Clusters.
    # Runs ONLY when 0 classic white/dark enclosed bubbles were found (character sheets / infographics / CG),
    # guaranteeing ZERO regressions on standard B&W or color manga pages!
    has_classic_bubble = any((not d.get("is_pastel")) for d in detected)
    has_sheet_ui = False
    if (not detected) or (page_is_color and not has_classic_bubble):
        cov_vis = (
            cv2.dilate(
                cv2.resize(already_covered, (vw, vh), interpolation=cv2.INTER_NEAREST),
                _CV2_K7,
                iterations=1,
            )
            if np.any(already_covered)
            else np.zeros((vh, vw), dtype=np.uint8)
        )

        if page_is_color:
            vis_sat = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
            ui_banners = _detect_colored_ui_banners_cv2(vis_bgr, vis_gray, vis_chroma, vis_sat, cov_vis)
            pastel_labels = _detect_pastel_table_labels_cv2(vis_bgr, vis_gray, vis_chroma, vis_sat, cov_vis)
            del vis_sat
            detected.extend(ui_banners)
            detected.extend(pastel_labels)
            has_sheet_ui = (len(ui_banners) + len(pastel_labels)) >= 3
            if ui_banners or pastel_labels:
                cov_vis = cv2.dilate(cov_vis, _CV2_K5, iterations=1)

        detected.extend(
            _detect_white_outlined_text_clusters_cv2(
                vis_gray,
                vis_chroma,
                is_color=page_is_color,
                covered_vis_mask=(cov_vis if np.any(cov_vis) else None),
                has_sheet_ui=has_sheet_ui,
            )
        )
    del vis_gray, vis_chroma

    # Cap dense pages (allow up to 36 badges on structured character profile sheets; keep 24 on normal manga)
    max_badges = 36 if has_sheet_ui else 24
    if len(detected) > max_badges:
        detected.sort(
            key=lambda d: (
                (d["bbox_norm"][2] - d["bbox_norm"][0])
                * (d["bbox_norm"][3] - d["bbox_norm"][1])
                * d.get("rect_fill", 0.7)
            ),
            reverse=True,
        )
        detected = detected[:max_badges]

    detected = _sort_bubbles_manga_reading_order(detected)
    for idx, b_obj in enumerate(detected, 1):
        b_obj["bubble_id"] = idx

    # Draw crisp RED contours + high-contrast RED/WHITE/BLACK numbered badges
    font_scale = max(0.50, vw / 1450.0)
    thickness = 2
    for b_obj in detected:
        bid = b_obj["bubble_id"]
        ymin, xmin, ymax, xmax = b_obj["bbox_norm"]
        vx1 = max(0, int((xmin / 1000.0) * vw))
        vy1 = max(0, int((ymin / 1000.0) * vh))
        vx2 = min(vw - 1, int((xmax / 1000.0) * vw))
        vy2 = min(vh - 1, int((ymax / 1000.0) * vh))

        m_w = max(2, vx2 - vx1)
        m_h = max(2, vy2 - vy1)
        m_up = cv2.resize(b_obj["mask_np"], (m_w, m_h), interpolation=cv2.INTER_NEAREST)
        contour = cv2.subtract(
            m_up,
            cv2.erode(m_up, _CV2_K3, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0),
        )
        vis_bgr[vy1 : vy1 + m_h, vx1 : vx1 + m_w][contour > 0] = (20, 20, 235)

        label = str(bid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pw = tw + 10
        ph = th + 8
        bx2 = min(vw - 3, vx2 + 2)
        bx1 = max(2, bx2 - pw)
        if vy1 < ph:
            by1 = min(vh - ph - 2, max(2, vy1 + 2))
            by2 = by1 + ph
        else:
            by2 = min(vh - 2, max(ph + 2, vy1 + 6))
            by1 = max(2, by2 - ph)
        cv2.rectangle(vis_bgr, (max(0, bx1 - 2), max(0, by1 - 2)), (min(vw - 1, bx2 + 2), min(vh - 1, by2 + 2)), (15, 15, 15), -1)
        cv2.rectangle(vis_bgr, (bx1 - 1, by1 - 1), (bx2 + 1, by2 + 1), (255, 255, 255), -1)
        cv2.rectangle(vis_bgr, (bx1, by1), (bx2, by2), (18, 18, 225), -1)
        cv2.putText(
            vis_bgr,
            label,
            (bx1 + 5, by1 + ph - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    ok, enc = cv2.imencode(".jpg", vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(vision_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for SoM JPEG")
    b64_som = base64.b64encode(enc.tobytes()).decode("utf-8")
    return b64_som, detected


def _detect_page_bubbles_and_build_som_b64_sync(
    raw_bytes_or_path: bytes | str | os.PathLike,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    raw_bytes = _coerce_image_bytes_sync(raw_bytes_or_path)
    if not raw_bytes:
        return "", []
    if HAS_CV2:
        return _detect_page_bubbles_and_build_som_b64_cv2_sync(raw_bytes, vision_max_dim, vision_quality)
    from manga_pil_fallback import _detect_page_bubbles_and_build_som_b64_pil_sync
    return _detect_page_bubbles_and_build_som_b64_pil_sync(raw_bytes, vision_max_dim, vision_quality)