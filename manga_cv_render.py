import os
import io
import re
import base64
import functools

from PIL import Image, ImageDraw, ImageFont

try:
    import numpy as np
    import cv2

    try:
        # Lock OpenCV to 1 thread so Render 0.1 vCPU / 512MB RAM containers avoid thread-pool RSS bloat
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    HAS_CV2 = True
except ImportError:
    np = None
    cv2 = None
    HAS_CV2 = False

from config import THAI_CHAR_REGEX

MANGA_CV_RENDER_VERSION = "v10"
SOM_CV_MAX_DIM = 360  # Fast 360px grid derived from single 1024px resize (~1-2.5s for 10 pages with OpenCV+NumPy!)
# Bundle 2 RAM & Discord upload optimization: quality=90 with standard 4:2:0 subsampling (~50% smaller than 94/4:4:4)
MANGA_RENDER_JPEG_QUALITY = 90

# Garuda-Bold MUST come first because NotoSansThai-Bold on Debian/Ubuntu lacks ASCII '.', '!', '?', "'", '-', '0-9'
MANGA_THAI_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
)

# Zero-dependency Thai syllable tokenizer constants
_THAI_CLUSTER_REGEX = re.compile(
    r"[เแโใไ]?[ก-ฮ][\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]*(?:[ะาำ]|[\u0E47-\u0E4E])*|[A-Za-z0-9]+|[^\s]"
)
_THAI_LEADING_VOWELS = set("เแโใไ")
_THAI_VOWEL_SIGNALS = set("เแโใไะาำ\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47")
_THAI_CLUSTER_FIRST = set("กขคจตทบปผพฟสศษ")
_THAI_CLUSTER_SECOND = set("รลว")
_THAI_HO_NAM_FOLLOW = set("งญนมยรลว")
_THAI_PREFIX_LEAD = set("สปกตฉผถข")
_THAI_PUNCT_ATTACH = set("ๆฯ?!.,:;'\"')]-~")
_THAI_OPEN_ENDINGS = set("าำ\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47\u0E48\u0E49\u0E4A\u0E4Bอว")
_THAI_CAN_TAKE_CODA_VOWEL = set("เแโา\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47อว")
_THAI_CODA_CONSONANTS = set("กขคฆงจชซฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬญ")

# Strict font-safe whitelist (ASCII printable + Thai Unicode block + Latin-1 accented letters)
# Guarantees ZERO missing-glyph `[]` tofu boxes in Garuda-Bold.ttf / NotoSans-Bold.ttf!
_FONT_UNSAFE_CHARS_REGEX = re.compile(r"[^\x20-\x7E\u00C0-\u00FF\u0E00-\u0E5B]")
# Unicode invariant: fix malformed Thai tone-before-vowel order (e.g. อ๊ิก -> อิ๊ก) so Garuda-Bold never collides marks
_THAI_TONE_BEFORE_VOWEL_REGEX = re.compile(r"([\u0E48-\u0E4B])([\u0E31\u0E34-\u0E3A])")

# Pre-allocated reusable OpenCV morphological kernels
_CV2_K3 = np.ones((3, 3), np.uint8) if HAS_CV2 else None
_CV2_K5 = np.ones((5, 5), np.uint8) if HAS_CV2 else None
_CV2_K7 = np.ones((7, 7), np.uint8) if HAS_CV2 else None
_CV2_ELLIPSE_13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)) if HAS_CV2 else None


# ==============================================================================
# SHARED FONTS, TEXT SANITIZATION, READING ORDER & LAYOUT ENGINE
# ==============================================================================
@functools.lru_cache(maxsize=64)
def _load_manga_font(size: int, is_thai: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = MANGA_THAI_FONT_CANDIDATES if is_thai else LATIN_FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _sanitize_text_for_manga_font(text: str) -> str:
    """
    Normalizes Unicode punctuation, fixes Thai combining-mark order (vowel before tone),
    and enforces a strict font-safe whitelist (`ASCII + Thai + Latin-1`) so `♥`, `♡`, `♪`,
    Hangul, Arabic, CJK, and emojis NEVER render as missing-glyph `[]` tofu boxes!
    """
    if not text:
        return ""
    s = str(text)
    s = (
        s.replace("…", "...")
        .replace("⋯", "...")
        .replace("〜", "~")
        .replace("～", "~")
        .replace("—", "-")
        .replace("–", "-")
        .replace("―", "-")
        .replace("・", " ")
        .replace("“", "'")
        .replace("”", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("「", "'")
        .replace("」", "'")
        .replace("『", "'")
        .replace("』", "'")
        .replace("！", "!")
        .replace("？", "?")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("。", ".")
    )
    # Remove any character outside Garuda-Bold / NotoSans-Bold glyph coverage (kills ♥/♡/♪/Hangul/Arabic/CJK/emoji tofu!)
    s = _FONT_UNSAFE_CHARS_REGEX.sub("", s)
    # Unicode combining-mark invariant: ensure Thai above/below vowels always precede tone marks
    if THAI_CHAR_REGEX.search(s):
        s = _THAI_TONE_BEFORE_VOWEL_REGEX.sub(r"\2\1", s)
    s = re.sub(r"\(\s*\)|\[\s*\]|'\s*'", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def _tokenize_thai_syllables(word: str) -> list[str]:
    raw = _THAI_CLUSTER_REGEX.findall(word)
    if len(raw) <= 1:
        return raw or [word]

    merged_onsets: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        tok = raw[i]
        nxt = raw[i + 1] if (i + 1) < n else ""

        if nxt:
            is_ho_nam = (
                ((tok == "ห") or (len(tok) == 2 and tok[0] in _THAI_LEADING_VOWELS and tok[1] == "ห"))
                and nxt[0] in _THAI_HO_NAM_FOLLOW
            )
            # Restrict อ-นำ-ย strictly to the 4 real Thai words (อย่า, อยู่, อย่าง, อยาก)
            # so คอย, หน่อย, บ่อย, น้อย, อร่อย, ปล่อย are never split!
            is_o_nam = (
                tok == "อ"
                and (
                    nxt.startswith(("ย่า", "ยู่"))
                    or (nxt == "ยา" and (i + 2) < n and raw[i + 2] == "ก")
                )
            )
            if is_ho_nam or is_o_nam:
                merged_onsets.append(tok + nxt)
                i += 2
                continue

            if nxt[0] in _THAI_CLUSTER_SECOND:
                is_bare_c = (len(tok) == 1 and tok in _THAI_CLUSTER_FIRST)
                is_lv_c = (len(tok) == 2 and tok[0] in _THAI_LEADING_VOWELS and tok[1] in _THAI_CLUSTER_FIRST)
                nxt_has_vowel = any(c in _THAI_VOWEL_SIGNALS for c in nxt) or tok.startswith("เ")
                if is_lv_c or (is_bare_c and nxt_has_vowel):
                    merged_onsets.append(tok + nxt)
                    i += 2
                    continue

        merged_onsets.append(tok)
        i += 1

    merged_vowels: list[str] = []
    i = 0
    m = len(merged_onsets)
    while i < m:
        tok = merged_onsets[i]
        nxt = merged_onsets[i + 1] if (i + 1) < m else ""
        if nxt:
            if tok.startswith("เ"):
                if nxt in ("อะ", "อ", "าะ") or (("\u0E35" in tok) and nxt.startswith("ย")) or (("\u0E37" in tok) and nxt.startswith("อ")):
                    merged_vowels.append(tok + nxt)
                    i += 2
                    continue
            # Merge 'ว' as part of สระอัว (ไม้หันอากาศ '\u0E31' + 'ว', e.g. ตัว/หัว/กลัว/ด้วย/ช่วย)
            if nxt == "ว" and "\u0E31" in tok and any("\u0E01" <= c <= "\u0E2E" for c in tok):
                merged_vowels.append(tok + nxt)
                i += 2
                continue
            # Merge bare 'อ' (including after '\u0E47' as in ช็อก/ล็อก) or 'ว' (สระอัวลดรูป like พวก/สวย)
            tok_other_v = any(c in _THAI_VOWEL_SIGNALS and c != "\u0E47" for c in tok)
            if (nxt == "อ" and not tok_other_v and any("\u0E01" <= c <= "\u0E2E" for c in tok)) or (
                nxt == "ว" and not any(c in _THAI_VOWEL_SIGNALS for c in tok) and any("\u0E01" <= c <= "\u0E2E" for c in tok)
            ):
                merged_vowels.append(tok + nxt)
                i += 2
                continue
        merged_vowels.append(tok)
        i += 1

    syllables: list[str] = []
    i = 0
    k = len(merged_vowels)
    while i < k:
        tok = merged_vowels[i]
        nxt = merged_vowels[i + 1] if (i + 1) < k else ""

        if not syllables:
            if (
                nxt
                and len(tok) == 1
                and tok in _THAI_PREFIX_LEAD
                and any(c in _THAI_VOWEL_SIGNALS for c in nxt)
            ):
                syllables.append(tok + nxt)
                i += 2
                continue
            syllables.append(tok)
            i += 1
            continue

        if all(c in _THAI_PUNCT_ATTACH for c in tok) or ("\u0E4C" in tok):
            syllables[-1] += tok
            i += 1
            continue

        is_bare_single = (len(tok) == 1 and "\u0E01" <= tok <= "\u0E2E")
        prev = syllables[-1]
        # Attach coda consonant to an open previous syllable BEFORE considering prefix merge
        # (fixes 'ช็อก' -> ['ช็อก'], 'หรอก' -> ['หรอก'], 'พวก' -> ['พวก'])
        prev_closed_vowel = prev.endswith(("ะ", "ำ")) or (prev[0] in ("ใ", "ไ"))
        # Recognize สระเอีย (เ-ีย like เยี่ย+ม, เรีย+ก, เสีย+ง) as open vowel ready for coda
        prev_is_ia_vowel = (prev.startswith("เ") and "\u0E35" in prev and prev.endswith("ย"))
        # Recognize สระอัว (ั+ว like ตัว, ด้วย) — only accepts 'ย' coda (ด้วย/ช่วย/ป่วย), never steals next onset!
        prev_is_ua_vowel = ("\u0E31" in prev and prev.endswith("ว"))
        if prev_is_ua_vowel:
            prev_open_for_coda = (tok == "ย")
        else:
            prev_open_for_coda = (
                not prev_closed_vowel
                and len(prev) <= 6
                and (
                    prev_is_ia_vowel
                    or (len(prev) >= 2 and prev[-1] in ("อ", "ว"))
                    or (
                        any(c in _THAI_CAN_TAKE_CODA_VOWEL for c in prev)
                        and (prev[-1] in _THAI_OPEN_ENDINGS or (len(prev) == 2 and prev[0] in ("เ", "แ", "โ")))
                    )
                )
            )
        if is_bare_single and tok in _THAI_CODA_CONSONANTS and prev_open_for_coda:
            syllables[-1] += tok
            i += 1
            continue

        if (
            nxt
            and len(tok) == 1
            and tok in _THAI_PREFIX_LEAD
            and any(c in _THAI_VOWEL_SIGNALS for c in nxt)
            and len(prev) >= 3
        ):
            syllables.append(tok + nxt)
            i += 2
            continue

        syllables.append(tok)
        i += 1

    final_syls: list[str] = []
    i = 0
    fs_len = len(syllables)
    while i < fs_len:
        tok = syllables[i]
        if len(tok) == 1 and "\u0E01" <= tok <= "\u0E2E":
            if (i + 1) < fs_len:
                final_syls.append(tok + syllables[i + 1])
                i += 2
                continue
            elif final_syls:
                final_syls[-1] += tok
                i += 1
                continue
        final_syls.append(tok)
        i += 1

    return final_syls


def _split_thai_word_cleanly(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    syllables = _tokenize_thai_syllables(word)
    out_lines: list[str] = []
    current = ""

    for syl in syllables:
        test_line = current + syl
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current = test_line
        else:
            out_lines.append(current)
            current = syl

    if current:
        out_lines.append(current)

    final_lines: list[str] = []
    for ln in out_lines:
        lb = draw.textbbox((0, 0), ln, font=font)
        if (lb[2] - lb[0]) <= max_width or len(ln) <= 3:
            final_lines.append(ln)
        else:
            chunk = ""
            for ch in ln:
                t_ch = chunk + ch
                cb = draw.textbbox((0, 0), t_ch, font=font)
                is_comb = ("\u0E31" <= ch <= "\u0E3A") or ("\u0E47" <= ch <= "\u0E4E")
                if (cb[2] - cb[0]) > max_width and chunk and not is_comb:
                    if len(chunk) > 2 and chunk[-1] in _THAI_LEADING_VOWELS:
                        final_lines.append(chunk[:-1])
                        chunk = chunk[-1] + ch
                    else:
                        final_lines.append(chunk)
                        chunk = ch
                else:
                    chunk = t_ch
            if chunk:
                final_lines.append(chunk)

    return final_lines


def _wrap_text_to_pixel_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    allow_midword_split: bool = False,
) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    is_thai = bool(THAI_CHAR_REGEX.search(clean_text))
    has_spaces = " " in clean_text
    words = clean_text.split(" ") if has_spaces else [clean_text]
    lines = []
    current = ""

    for word in words:
        if not word:
            continue
        test_line = f"{current} {word}".strip() if current else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
                current = ""
            wbbox = draw.textbbox((0, 0), word, font=font)
            if (wbbox[2] - wbbox[0]) <= max_width or (not is_thai and not allow_midword_split):
                current = word
            elif is_thai:
                thai_parts = _split_thai_word_cleanly(draw, word, font, max_width)
                lines.extend(thai_parts[:-1])
                current = thai_parts[-1] if thai_parts else ""
            else:
                chunk = ""
                for ch in word:
                    test_ch = chunk + ch
                    cbbox = draw.textbbox((0, 0), test_ch, font=font)
                    if (cbbox[2] - cbbox[0]) > max_width and chunk:
                        lines.append(chunk)
                        chunk = ch
                    else:
                        chunk = test_ch
                current = chunk
    if current:
        lines.append(current)
    return lines


def _split_text_for_two_lobes(text: str, ratio_first: float) -> tuple[str, str] | None:
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) < 28:
        return None

    ratio_first = max(0.25, min(0.75, ratio_first))
    target = int(round(len(s) * ratio_first))

    sent_breaks = [m.end() for m in re.finditer(r"(?:\.\.\.|[.!?])\s+", s)]
    valid_sent = [idx for idx in sent_breaks if 6 <= idx <= len(s) - 6]
    if valid_sent:
        best_s = min(valid_sent, key=lambda idx: abs(idx - target))
        if abs(best_s - target) <= max(18, int(len(s) * 0.30)):
            p1, p2 = s[:best_s].strip(), s[best_s:].strip()
            if p1 and p2:
                return p1, p2

    space_breaks = [m.start() for m in re.finditer(r"\s+", s)]
    valid_sp = [idx for idx in space_breaks if 6 <= idx <= len(s) - 6]
    if valid_sp:
        best_sp = min(valid_sp, key=lambda idx: abs(idx - target))
        if abs(best_sp - target) <= max(14, int(len(s) * 0.25)):
            p1, p2 = s[:best_sp].strip(), s[best_sp:].strip()
            if p1 and p2:
                return p1, p2

    if THAI_CHAR_REGEX.search(s):
        syls = _tokenize_thai_syllables(s)
        if len(syls) >= 4:
            acc = 0
            best_k = None
            best_diff = 10**9
            for k in range(1, len(syls)):
                acc += len(syls[k - 1])
                diff = abs(acc - target)
                if diff < best_diff:
                    best_diff = diff
                    best_k = k
            if best_k is not None:
                p1 = "".join(syls[:best_k]).strip()
                p2 = "".join(syls[best_k:]).strip()
                if len(p1) >= 4 and len(p2) >= 4:
                    return p1, p2

    return None


def _resolve_box_collisions(renders: list[list], min_gap: int = 6) -> list[list]:
    n = len(renders)
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                ax1, ay1, ax2, ay2 = renders[i][0], renders[i][1], renders[i][2], renders[i][3]
                bx1, by1, bx2, by2 = renders[j][0], renders[j][1], renders[j][2], renders[j][3]

                ov_x = min(ax2, bx2) - max(ax1, bx1)
                ov_y = min(ay2, by2) - max(ay1, by1)
                if ov_x <= -min_gap or ov_y <= -min_gap:
                    continue

                acx, acy = (ax1 + ax2) // 2, (ay1 + ay2) // 2
                bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2

                if abs(acy - bcy) >= abs(acx - bcx) * 0.8:
                    mid_y = (min(ay2, by2) + max(ay1, by1)) // 2
                    if acy <= bcy:
                        renders[i][3] = max(ay1 + 16, mid_y - (min_gap // 2))
                        renders[j][1] = min(by2 - 16, mid_y + (min_gap // 2))
                    else:
                        renders[j][3] = max(by1 + 16, mid_y - (min_gap // 2))
                        renders[i][1] = min(ay2 - 16, mid_y + (min_gap // 2))
                else:
                    mid_x = (min(ax2, bx2) + max(ax1, bx1)) // 2
                    if acx <= bcx:
                        renders[i][2] = max(ax1 + 18, mid_x - (min_gap // 2))
                        renders[j][0] = min(bx2 - 18, mid_x + (min_gap // 2))
                    else:
                        renders[j][2] = max(bx1 + 18, mid_x - (min_gap // 2))
                        renders[i][0] = min(ax2 - 18, mid_x + (min_gap // 2))

                for idx_k in (i, j):
                    hx1, hy1, hx2, hy2 = renders[idx_k][7]
                    renders[idx_k][0] = max(hx1, min(hx2 - 18, renders[idx_k][0]))
                    renders[idx_k][1] = max(hy1, min(hy2 - 16, renders[idx_k][1]))
                    renders[idx_k][2] = min(hx2, max(renders[idx_k][0] + 18, renders[idx_k][2]))
                    renders[idx_k][3] = min(hy2, max(renders[idx_k][1] + 16, renders[idx_k][3]))
    return renders


def _fit_text_binary_search(
    draw: ImageDraw.ImageDraw,
    trans_text: str,
    usable_w: int,
    usable_h: int,
    start_font_size: int,
    is_thai_text: bool,
) -> tuple[ImageFont.ImageFont, list[str], int, int]:
    sample_probe = "Agกข้ำ" if is_thai_text else "Agjpqy!"

    def _measure(fsize: int) -> tuple[bool, ImageFont.ImageFont, list[str], int, int]:
        font = _load_manga_font(fsize, is_thai=is_thai_text)
        wrapped = _wrap_text_to_pixel_width(
            draw, trans_text, font, usable_w, allow_midword_split=(fsize <= 9)
        )
        sample_bbox = draw.textbbox((0, 0), sample_probe, font=font)
        lh = max(fsize + 3, int((sample_bbox[3] - sample_bbox[1]) * 1.16))
        total_h = lh * len(wrapped)
        max_line_w = 0
        for ln in wrapped:
            lb = draw.textbbox((0, 0), ln, font=font)
            w_ln = lb[2] - lb[0]
            if w_ln > max_line_w:
                max_line_w = w_ln
        fits = (total_h <= usable_h and max_line_w <= usable_w)
        return fits, font, wrapped, lh, sample_bbox[1]

    lo, hi = 9, max(9, start_font_size)
    _, best_font, best_lines, best_lh, best_top = _measure(lo)

    while lo <= hi:
        mid = (lo + hi) // 2
        fits, font, wrapped, lh, ref_top = _measure(mid)
        if fits:
            best_font, best_lines, best_lh, best_top = font, wrapped, lh, ref_top
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_lines, best_lh, best_top


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
# STAGE 1A: VECTORIZED OPENCV + NUMPY CV BUBBLE DETECTOR (~10x FASTER ON 0.1 vCPU)
# ==============================================================================
def _fill_mask_holes_cv2(comp_crop: "np.ndarray") -> "np.ndarray":
    """Fills all enclosed interior holes (kanji strokes) inside a uint8 binary mask in C."""
    ch, cw = comp_crop.shape
    padded = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = comp_crop
    ff_mask = np.zeros((ch + 4, cw + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 128)
    return np.where(padded[1:-1, 1:-1] == 128, np.uint8(0), np.uint8(255))


def _close_rim_bays_cv2(filled_crop: "np.ndarray") -> "np.ndarray":
    """
    Seals concave kanji-to-border bays on validated pastel/borderless bubbles (e.g. kanji
    touching dark gloom shading lines near the bubble rim) using a 13x13 elliptical closing
    strictly inside the bubble's own bounding box without leaking outward.
    """
    ch, cw = filled_crop.shape
    pad = 7
    buf = np.zeros((ch + pad * 2, cw + pad * 2), dtype=np.uint8)
    buf[pad:-pad, pad:-pad] = filled_crop
    closed = cv2.morphologyEx(buf, cv2.MORPH_CLOSE, _CV2_ELLIPSE_13)
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
        black_ink_px = int(np.count_nonzero((inner_core > 0) & (g_sub < 135) & (c_sub < 32)))
        ink_density = black_ink_px / float(inner_area)
        is_small_clean_oval = (filled_area < 550 and solidity >= 0.92 and rect_fill >= 0.72)
        min_blk_px = 6 if is_small_clean_oval else 12
        min_ink_dens = 0.025 if is_small_clean_oval else 0.012
        if black_ink_px < min_blk_px or ink_density < min_ink_dens or ink_density > 0.45:
            continue

        ink_bin = np.where((inner_core > 0) & (g_sub < 145) & (c_sub < 35), np.uint8(255), np.uint8(0))
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
                v_ink = np.where((v_in > 0) & (vg_sub < 145) & (vc_sub < 38), np.uint8(255), np.uint8(0))
                vn_cc, _, vcc_st, _ = cv2.connectedComponentsWithStats(v_ink, connectivity=8)
                reg_glyphs = 0
                dot_glyphs = 0
                for k_cc in range(1, vn_cc):
                    gw_v = int(vcc_st[k_cc, cv2.CC_STAT_WIDTH])
                    gh_v = int(vcc_st[k_cc, cv2.CC_STAT_HEIGHT])
                    ga_v = int(vcc_st[k_cc, cv2.CC_STAT_AREA])
                    if 3 <= gw_v <= 40 and 3 <= gh_v <= 42 and ga_v >= 8:
                        reg_glyphs += 1
                    elif 2 <= gw_v <= 5 and 2 <= gh_v <= 5 and 4 <= ga_v <= 16:
                        dot_glyphs += 1
                glyph_count = reg_glyphs + dot_glyphs
                min_glyphs = 6 if (rect_fill < 0.62 or solidity < 0.86) else 3
                if glyph_count < min_glyphs:
                    continue

        # Seal concave rim-kanji bays ONLY AFTER passing all validation guards (keeps Page 3 調子乗/思 inside mask!)
        final_mask = _close_rim_bays_cv2(filled_crop) if is_pastel_pass else filled_crop
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


def _detect_page_bubbles_and_build_som_b64_cv2_sync(
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    """
    Fully vectorized OpenCV + NumPy Stage 1 detector & Set-of-Mark JPEG builder:
    - Pass A1 (`gray >= 232`): clean base white bubble pass (now corner-bubble safe!).
    - Pass A2 (`INTER_AREA` dark-outline preservation): thin-bordered ovals on white panels.
    - Pass A3 (`gray >= 178 & sat < 95 & dark_box < 24`): enclosed pastel/colored light speech
      bubbles (pale blue, pink, lavender, green, cream, yellow) with 2px border trim + nested
      white-halo deduplication.
    - Pass B (`gray <= 95`): enclosed dark speech bubbles with strict anti-dark-panel guards.
    - Draws crisp RED bubble contours + high-contrast RED/WHITE/BLACK numbered badges so
      badges stand out clearly even next to red SFX (`ガチャ`, `ぎゅっ`) or top edges!
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

    # Pass A3: Enclosed pastel / colored light speech bubbles — ONLY on actual color pages!
    # (Prevents pure B&W screentone panels/faces from being misclassified as pastel bubbles!)
    if page_is_color:
        pastel_bin = np.where(
            (small_gray >= 178) & (small_sat < 95) & (small_dark_box < 24),
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
    del small_bgr, vis_gray, vis_chroma

    # Remove tiny Pass A1 white text-glow halos nested inside larger Pass A3 pastel bubbles!
    detected = _prune_nested_child_bubbles(detected, close_pil=False)

    # Pass B: Enclosed dark speech bubbles / dark caption boxes (strict anti-dark-panel guards!)
    dark_bin = np.where(small_gray <= 95, np.uint8(255), np.uint8(0))
    num_d, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark_bin, connectivity=4)
    if num_d > 1:
        d_margin = _collect_margin_labels_cv2(d_labels, d_stats, sw, sh, max_area_px, edge_pad=1)
        max_dw, max_dh = int(sw * 0.50), int(sh * 0.50)
        for lbl in range(1, num_d):
            if lbl in d_margin:
                continue
            cx1 = int(d_stats[lbl, cv2.CC_STAT_LEFT])
            cy1 = int(d_stats[lbl, cv2.CC_STAT_TOP])
            cw = int(d_stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(d_stats[lbl, cv2.CC_STAT_HEIGHT])
            comp_area = int(d_stats[lbl, cv2.CC_STAT_AREA])
            cx2, cy2 = cx1 + cw, cy1 + ch
            if comp_area > max_area_px or cw < min_w_px or ch < min_h_px or cw > max_dw or ch > max_dh:
                continue
            t_l, t_t, t_r, t_b = cx1 <= 1, cy1 <= 1, cx2 >= sw - 1, cy2 >= sh - 1
            if (int(t_l) + int(t_t) + int(t_r) + int(t_b)) >= 3 or (t_l and t_r) or (t_t and t_b):
                continue

            comp_crop = np.where(d_labels[cy1:cy2, cx1:cx2] == lbl, np.uint8(255), np.uint8(0))
            filled_crop = _fill_mask_holes_cv2(comp_crop)
            filled_area = max(1, int(np.count_nonzero(filled_crop)))
            holes_area = int(np.count_nonzero((filled_crop > 0) & (comp_crop == 0)))
            rect_fill = filled_area / float(max(1, cw * ch))
            if filled_area > max_area_px or holes_area < 5 or rect_fill < 0.65:
                continue

            # Guard 1: Reject dark panels that enclose/overlap already-detected light/pastel speech bubbles!
            ov_px = int(np.count_nonzero(filled_crop & already_covered[cy1:cy2, cx1:cx2]))
            if ov_px > 20 or ov_px > filled_area * 0.04:
                continue

            mask_bool = filled_crop > 0
            s_sub = small_sat[cy1:cy2, cx1:cx2]
            g_sub = small_gray[cy1:cy2, cx1:cx2]
            s_vals = s_sub[mask_bool]
            mean_s = float(s_vals.mean()) if s_vals.size else 255.0
            sat_ratio = float(np.count_nonzero(s_vals >= 28)) / float(max(1, s_vals.size))

            # Guard 2: True HSV Saturation rejects dark colored artwork (navy clothes, dark brown walls, shadows)
            if mean_s >= 22.0 or sat_ratio >= 0.16:
                continue

            # Guard 3: Dark bubble background must be uniform (rejects shaded character panels)
            dark_vals = g_sub[mask_bool & (g_sub <= 135)]
            if dark_vals.size < max(16, int(filled_area * 0.45)) or float(dark_vals.std()) > 24.0:
                continue

            # Guard 4: Bright achromatic text must sit strictly inside the eroded interior
            inner_core = cv2.erode(filled_crop, _CV2_K5, iterations=1)
            if not np.any(inner_core):
                inner_core = cv2.erode(filled_crop, _CV2_K3, iterations=1)
            inner_bright_px = int(np.count_nonzero((inner_core > 0) & (g_sub >= 205) & (s_sub < 28)))
            bright_px = int(np.count_nonzero(mask_bool & (g_sub >= 210) & (s_sub < 28)))

            if inner_bright_px >= 4 and 0.02 <= (bright_px / float(filled_area)) <= 0.38:
                already_covered[cy1:cy2, cx1:cx2] |= filled_crop
                bbox_norm = [
                    int(round((cy1 / float(sh)) * 1000.0)),
                    int(round((cx1 / float(sw)) * 1000.0)),
                    int(round((cy2 / float(sh)) * 1000.0)),
                    int(round((cx2 / float(sw)) * 1000.0)),
                ]
                detected.append({
                    "bbox_norm": bbox_norm,
                    "mask_np": filled_crop,
                    "rect_fill": rect_fill,
                    "solidity": 0.95,
                    "glyph_count": max(6, int(inner_bright_px)),
                    "is_dark": True,
                    "is_pastel": False,
                })

    # On dense B&W pages, cap to top 24 highest-solidity/largest candidates
    if len(detected) > 24:
        detected.sort(
            key=lambda d: (
                (d["bbox_norm"][2] - d["bbox_norm"][0])
                * (d["bbox_norm"][3] - d["bbox_norm"][1])
                * d.get("rect_fill", 0.7)
            ),
            reverse=True,
        )
        detected = detected[:24]

    detected = _sort_bubbles_manga_reading_order(detected)
    for idx, b_obj in enumerate(detected, 1):
        b_obj["bubble_id"] = idx

    # Draw crisp RED contours + high-contrast RED/WHITE/BLACK numbered badges
    # (stands out unmistakably against white, pastel bubbles, AND bright red SFX like ガチャ!)
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
        contour = cv2.subtract(m_up, cv2.erode(m_up, _CV2_K3, iterations=1))
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
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    if HAS_CV2:
        return _detect_page_bubbles_and_build_som_b64_cv2_sync(raw_bytes, vision_max_dim, vision_quality)
    from manga_pil_fallback import _detect_page_bubbles_and_build_som_b64_pil_sync
    return _detect_page_bubbles_and_build_som_b64_pil_sync(raw_bytes, vision_max_dim, vision_quality)


# ==============================================================================
# STAGE 3A: VECTORIZED OPENCV + NUMPY RENDER MASKING & FULL-RES TYPESETTER
# ==============================================================================
def _fast_dilate_mask_cv2(mask_u8: "np.ndarray", kernel_half: int = 3) -> "np.ndarray":
    wh, ww = mask_u8.shape
    if ww < 24 or wh < 24:
        return cv2.dilate(mask_u8, _CV2_K3, iterations=1)
    sw, sh = max(8, ww // 2), max(8, wh // 2)
    small = cv2.resize(mask_u8, (sw, sh), interpolation=cv2.INTER_LINEAR)
    kern = _CV2_K3 if kernel_half <= 3 else _CV2_K5
    dil = cv2.dilate(small, kern, iterations=1)
    up = cv2.resize(dil, (ww, wh), interpolation=cv2.INTER_LINEAR)
    return np.where(up >= 64, np.uint8(255), np.uint8(0))


def _build_true_chroma_art_shield_cv2(
    sub_bgr: "np.ndarray",
    chroma_thresh: int = 34,
    is_dark: bool = False,
) -> "np.ndarray":
    """
    Returns a uint8 `non_chroma` mask (`255` = safe to erase, `0` = protected colored art).
    - Light bubbles: uses pure RGB chroma (`>= 34`) so warm-black kanji is never shielded.
    - Dark bubbles: combines HSV saturation (`>= 26`) + RGB chroma (`>= 24`) to shield pale skin/eyes/hair.
    """
    wh, ww = sub_bgr.shape[:2]
    if is_dark:
        if ww >= 24 and wh >= 24:
            sw, sh = max(8, ww // 2), max(8, wh // 2)
            small_bgr = cv2.resize(sub_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            small_c = small_bgr.max(axis=2) - small_bgr.min(axis=2)
            small_s = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
            bin_c = np.where((small_c >= 24) | (small_s >= 26), np.uint8(255), np.uint8(0))
            if not np.any(bin_c):
                return np.full((wh, ww), 255, dtype=np.uint8)
            opened = cv2.erode(bin_c, _CV2_K3, iterations=1)
            shield_small = cv2.dilate(opened, _CV2_K5, iterations=1)
            up = cv2.resize(shield_small, (ww, wh), interpolation=cv2.INTER_LINEAR)
            return np.where(up >= 64, np.uint8(0), np.uint8(255))
        c_band = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)
        s_band = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        bin_c = np.where((c_band >= 24) | (s_band >= 26), np.uint8(255), np.uint8(0))
        if not np.any(bin_c):
            return np.full((wh, ww), 255, dtype=np.uint8)
        shield = cv2.dilate(bin_c, _CV2_K3, iterations=1)
        return np.where(shield > 0, np.uint8(0), np.uint8(255))

    chroma_band = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)
    if ww >= 24 and wh >= 24:
        sw, sh = max(8, ww // 2), max(8, wh // 2)
        small_c = cv2.resize(chroma_band, (sw, sh), interpolation=cv2.INTER_LINEAR)
        bin_c = np.where(small_c >= chroma_thresh, np.uint8(255), np.uint8(0))
        if not np.any(bin_c):
            return np.full((wh, ww), 255, dtype=np.uint8)
        opened = cv2.erode(bin_c, _CV2_K3, iterations=1)
        shield_small = cv2.dilate(opened, _CV2_K5, iterations=1)
        up = cv2.resize(shield_small, (ww, wh), interpolation=cv2.INTER_LINEAR)
        return np.where(up >= 64, np.uint8(0), np.uint8(255))

    bin_c = np.where(chroma_band >= chroma_thresh, np.uint8(255), np.uint8(0))
    if not np.any(bin_c):
        return np.full((wh, ww), 255, dtype=np.uint8)
    shield = cv2.dilate(bin_c, _CV2_K3, iterations=1)
    return np.where(shield > 0, np.uint8(0), np.uint8(255))


def _inpaint_pastel_bubble_region_cv2(
    sub_bgr: "np.ndarray",
    sub_gray: "np.ndarray",
    safe_core: "np.ndarray",
) -> tuple["np.ndarray", tuple[int, int, int]]:
    """
    Erases dark Japanese ink + any white text glow/halo inside a pastel/colored light bubble
    and fills the erased region via fast half-res TELEA inpainting (~3-10ms/bubble) so the
    bubble's pastel blue/pink/green/yellow/mint gradient stays completely seamless!
    Returns `(erase_bool, median_bg_bgr)`.
    """
    bh, bw = sub_gray.shape
    core_bool = safe_core > 0
    sub_chroma = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)

    non_dark = core_bool & (sub_gray >= 165)
    if np.count_nonzero(non_dark) < 16:
        non_dark = core_bool
    bg_lum = int(np.median(sub_gray[non_dark]))
    med_bgr = np.median(sub_bgr[non_dark], axis=0)
    bg_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))
    bg_chroma = int(max(bg_bgr) - min(bg_bgr))

    ink_thresh = min(195, max(145, bg_lum - 24))
    dark_ink = np.where(
        core_bool & (sub_gray < ink_thresh) & (sub_chroma < max(42, bg_chroma + 22)),
        np.uint8(255),
        np.uint8(0),
    )

    # Dilate dark ink proportionally to bubble size to swallow full anti-aliased strokes + wide white halos
    ink_halo = _fast_dilate_mask_cv2(dark_ink, kernel_half=5)
    extra_iters = 2 if max(bw, bh) >= 260 else 1
    ink_halo = cv2.dilate(ink_halo, _CV2_K5, iterations=extra_iters)
    if bg_lum < 242 or bg_chroma >= 10:
        near_ink = cv2.dilate(ink_halo, _CV2_K5, iterations=extra_iters + 1)
        glow_lum_thresh = min(236, max(210, bg_lum + 14))
        glow_chroma_max = max(14, int(bg_chroma * 0.72))
        white_halo = (
            (near_ink > 0)
            & core_bool
            & (sub_gray >= glow_lum_thresh)
            & (sub_chroma <= glow_chroma_max)
        )
        # Dilate detected white glow by 2px so soft outer fringes never leave cloudy ghost outlines
        if np.any(white_halo):
            wh_u8 = cv2.dilate(np.where(white_halo, np.uint8(255), np.uint8(0)), _CV2_K5, iterations=1)
            erase_bool = ((ink_halo > 0) | (wh_u8 > 0)) & core_bool
        else:
            erase_bool = (ink_halo > 0) & core_bool
    else:
        erase_bool = (ink_halo > 0) & core_bool

    if not np.any(erase_bool):
        return erase_bool, bg_bgr

    erase_u8 = np.where(erase_bool, np.uint8(255), np.uint8(0))
    inp_scale = min(0.5, 180.0 / float(max(bw, bh)))
    if bw >= 32 and bh >= 32 and inp_scale < 1.0:
        iw = max(12, int(round(bw * inp_scale)))
        ih = max(12, int(round(bh * inp_scale)))
        sm_bgr = cv2.resize(sub_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        sm_mask = cv2.resize(erase_u8, (iw, ih), interpolation=cv2.INTER_LINEAR)
        sm_mask = np.where(sm_mask >= 32, np.uint8(255), np.uint8(0))
        # Guard temporary scratch buffer so TELEA never pulls dark black bubble-outline pixels inward!
        sm_gray_tmp = cv2.cvtColor(sm_bgr, cv2.COLOR_BGR2GRAY)
        sm_dark_border = (sm_mask == 0) & (sm_gray_tmp < ink_thresh)
        if np.any(sm_dark_border):
            sm_bgr = sm_bgr.copy()
            sm_bgr[sm_dark_border] = bg_bgr
        sm_inp = cv2.inpaint(sm_bgr, sm_mask, 3, cv2.INPAINT_TELEA)
        full_inp = cv2.resize(sm_inp, (bw, bh), interpolation=cv2.INTER_LINEAR)
        sub_bgr[erase_bool] = full_inp[erase_bool]
    else:
        tmp_bgr = sub_bgr.copy()
        tmp_dark_border = (erase_u8 == 0) & (sub_gray < ink_thresh)
        if np.any(tmp_dark_border):
            tmp_bgr[tmp_dark_border] = bg_bgr
        full_inp = cv2.inpaint(tmp_bgr, erase_u8, 3, cv2.INPAINT_TELEA)
        sub_bgr[erase_bool] = full_inp[erase_bool]

    return erase_bool, bg_bgr


def _largest_inscribed_rect_in_mask_cv2(mask_u8: "np.ndarray") -> tuple[int, int, int, int] | None:
    mh, mw = mask_u8.shape
    if mw < 4 or mh < 4:
        return None

    scale = min(1.0, 96.0 / float(max(mw, mh)))
    gw = max(6, int(round(mw * scale)))
    gh = max(6, int(round(mh * scale)))
    grid = (
        cv2.resize(mask_u8, (gw, gh), interpolation=cv2.INTER_NEAREST)
        if scale < 1.0
        else mask_u8
    )
    grid_bool = grid >= 128

    heights = np.zeros(gw + 1, dtype=np.int32)
    best_score = -1.0
    best_rect = None

    for y in range(gh):
        row = grid_bool[y]
        heights[:gw] = np.where(row, heights[:gw] + 1, 0)
        heights[gw] = 0
        h_list = heights.tolist()

        stack: list[int] = []
        for x in range(gw + 1):
            cur_h = h_list[x]
            while stack and h_list[stack[-1]] > cur_h:
                h_val = h_list[stack.pop()]
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


def _detect_two_lobe_render_rects_cv2(
    place_mask: "np.ndarray",
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, int, int]] | None:
    mh, mw = place_mask.shape
    if mw < 100 and mh < 100:
        return None

    scale = min(1.0, 96.0 / float(max(mw, mh)))
    gw = max(12, int(round(mw * scale)))
    gh = max(12, int(round(mh * scale)))
    grid = cv2.resize(place_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)
    grid_bool = grid >= 128

    area = int(np.count_nonzero(grid_bool))
    if area / float(max(1, gw * gh)) >= 0.74:
        return None

    best_split = None
    if mw >= 150 and (mw / float(max(1, mh))) >= 1.05:
        col_counts = np.count_nonzero(grid_bool, axis=0)
        s_lo, s_hi = max(4, int(gw * 0.25)), min(gw - 4, int(gw * 0.75))
        if s_hi > s_lo:
            min_x = s_lo + int(np.argmin(col_counts[s_lo:s_hi]))
            min_h = int(col_counts[min_x])
            l_max = int(col_counts[:min_x].max()) if min_x > 0 else 0
            r_max = int(col_counts[min_x + 1 :].max()) if (min_x + 1) < gw else 0
            peak_min = min(l_max, r_max)
            if peak_min >= 18 and min_h <= peak_min * 0.74:
                best_split = ("v", min_x, min_h / float(peak_min))

    if mh >= 150 and (mh / float(max(1, mw))) >= 1.05:
        row_counts = np.count_nonzero(grid_bool, axis=1)
        s_lo, s_hi = max(4, int(gh * 0.25)), min(gh - 4, int(gh * 0.75))
        if s_hi > s_lo:
            min_y = s_lo + int(np.argmin(row_counts[s_lo:s_hi]))
            min_w = int(row_counts[min_y])
            t_max = int(row_counts[:min_y].max()) if min_y > 0 else 0
            b_max = int(row_counts[min_y + 1 :].max()) if (min_y + 1) < gh else 0
            peak_min = min(t_max, b_max)
            if peak_min >= 18 and min_w <= peak_min * 0.74:
                ratio = min_w / float(peak_min)
                if best_split is None or ratio < best_split[2]:
                    best_split = ("h", min_y, ratio)

    if best_split is None:
        return None

    axis, cut_g, _ = best_split
    cut_full = int(round(cut_g / scale))

    if axis == "v":
        r_rect = _largest_inscribed_rect_in_mask_cv2(place_mask[:, cut_full:])
        l_rect = _largest_inscribed_rect_in_mask_cv2(place_mask[:, :cut_full])
        if not (r_rect and l_rect):
            return None
        r1 = (offset_x + cut_full + r_rect[0] + 2, offset_y + r_rect[1] + 2, offset_x + cut_full + r_rect[2] - 2, offset_y + r_rect[3] - 2)
        r2 = (offset_x + l_rect[0] + 2, offset_y + l_rect[1] + 2, offset_x + l_rect[2] - 2, offset_y + l_rect[3] - 2)
    else:
        t_rect = _largest_inscribed_rect_in_mask_cv2(place_mask[:cut_full, :])
        b_rect = _largest_inscribed_rect_in_mask_cv2(place_mask[cut_full:, :])
        if not (t_rect and b_rect):
            return None
        r1 = (offset_x + t_rect[0] + 2, offset_y + t_rect[1] + 2, offset_x + t_rect[2] - 2, offset_y + t_rect[3] - 2)
        r2 = (offset_x + b_rect[0] + 2, offset_y + cut_full + b_rect[1] + 2, offset_x + b_rect[2] - 2, offset_y + cut_full + b_rect[3] - 2)

    if (r1[2] - r1[0]) < 48 or (r1[3] - r1[1]) < 36 or (r2[2] - r2[0]) < 48 or (r2[3] - r2[1]) < 36:
        return None
    return [r1, r2]


def _refine_cv_bubble_mask_locally_cv2(
    img_bgr: "np.ndarray",
    cv_obj: dict,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    other_cv_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int, "np.ndarray", float]:
    """
    Locally refines a light white CV bubble with strict anti-leak guards:
    - Preserves thin 1px dark borders via `INTER_AREA` downscale of `gray < 185`.
    - Skips white-only chroma refinement on dark or pastel bubbles (which already have exact Pass A3/B masks).
    - Caps area growth at `1.75x`, enforces solidity retention (`>= 0.78x`),
      and rejects any expansion that spills into a neighboring CV bubble.
    """
    height, width = img_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    orig_fill = float(cv_obj.get("rect_fill") or 0.70)
    orig_mask_np = cv_obj.get("mask_np")
    if orig_mask_np is None and cv_obj.get("mask_small") is not None:
        orig_mask_np = np.array(cv_obj["mask_small"], dtype=np.uint8)

    def _fallback_orig():
        m_full = np.where(
            cv2.resize(orig_mask_np, (bw, bh), interpolation=cv2.INTER_LINEAR) >= 128,
            np.uint8(255),
            np.uint8(0),
        )
        return x1, y1, x2, y2, m_full, orig_fill

    if bool(cv_obj.get("is_dark")) or bool(cv_obj.get("is_pastel")) or bw < 16 or bh < 14 or orig_mask_np is None:
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

    win_full_bgr = img_bgr[wy1:wy2, wx1:wx2]
    win_full_gray = cv2.cvtColor(win_full_bgr, cv2.COLOR_BGR2GRAY)
    win_dark_full = np.where(win_full_gray < 185, np.uint8(255), np.uint8(0))
    win_dark_box = cv2.resize(win_dark_full, (sw, sh), interpolation=cv2.INTER_AREA)

    win_bgr = cv2.resize(win_full_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    win_gray = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)
    win_chroma = win_bgr.max(axis=2) - win_bgr.min(axis=2)
    local_bin = np.where(
        (win_gray >= 220) & (win_chroma < 16) & (win_dark_box < 24),
        np.uint8(255),
        np.uint8(0),
    )

    ox_s = int(round((x1 - wx1) * scale))
    oy_s = int(round((y1 - wy1) * scale))
    ow_s = max(4, int(round(bw * scale)))
    oh_s = max(4, int(round(bh * scale)))
    orig_small = cv2.resize(orig_mask_np, (ow_s, oh_s), interpolation=cv2.INTER_NEAREST)
    orig_area_s = max(1, int(np.count_nonzero(orig_small)))

    _, loc_labels = cv2.connectedComponents(local_bin, connectivity=4)
    hit_labels: set[int] = set()
    for fy in (0.50, 0.30, 0.70):
        sy_o = min(oh_s - 1, max(0, int(oh_s * fy)))
        for fx in (0.50, 0.30, 0.70):
            sx_o = min(ow_s - 1, max(0, int(ow_s * fx)))
            if orig_small[sy_o, sx_o] >= 128:
                sx_l = min(sw - 2, max(1, ox_s + sx_o))
                sy_l = min(sh - 2, max(1, oy_s + sy_o))
                lbl_val = int(loc_labels[sy_l, sx_l])
                if lbl_val > 0:
                    hit_labels.add(lbl_val)

    if not hit_labels:
        return _fallback_orig()

    comp = np.isin(loc_labels, list(hit_labels)).astype(np.uint8) * 255
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return _fallback_orig()

    cx1, cx2 = int(xs.min()), int(xs.max()) + 1
    cy1, cy2 = int(ys.min()), int(ys.max()) + 1
    cw_s, ch_s = cx2 - cx1, cy2 - cy1

    t_l, t_t, t_r, t_b = cx1 <= 1, cy1 <= 1, cx2 >= sw - 1, cy2 >= sh - 1
    if (int(t_l) + int(t_t) + int(t_r) + int(t_b)) >= 3 or (t_l and t_r) or (t_t and t_b):
        return _fallback_orig()

    filled_s = _fill_mask_holes_cv2(comp[cy1:cy2, cx1:cx2])
    new_area_s = max(1, int(np.count_nonzero(filled_s)))
    new_rect_fill = new_area_s / float(max(1, cw_s * ch_s))
    if (
        new_area_s > orig_area_s * 1.75
        or new_area_s < orig_area_s * 0.85
        or new_rect_fill < orig_fill * 0.78
    ):
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
                    return _fallback_orig()

    m_full = np.where(
        cv2.resize(filled_s, (rw, rh), interpolation=cv2.INTER_LINEAR) >= 128,
        np.uint8(255),
        np.uint8(0),
    )
    return rx1, ry1, rx2, ry2, m_full, new_rect_fill


def _sample_dark_bg_bgr_and_lum_cv2(
    img_bgr: "np.ndarray", base_gray: "np.ndarray", x1: int, y1: int, x2: int, y2: int
) -> tuple[tuple[int, int, int], int, float]:
    sub_bgr = img_bgr[y1:y2, x1:x2]
    sub_g = base_gray[y1:y2, x1:x2]
    bg_bool = sub_g < 145
    if not np.any(bg_bool):
        bg_bool = np.ones(sub_g.shape, dtype=bool)
    med_bgr = np.median(sub_bgr[bg_bool], axis=0)
    bg_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))
    g_vals = sub_g[bg_bool]
    bg_lum = int(np.median(g_vals))
    bg_std = float(np.std(g_vals))
    return bg_bgr, bg_lum, bg_std


def _sample_light_bg_bgr_cv2(
    img_bgr: "np.ndarray", base_gray: "np.ndarray", x1: int, y1: int, x2: int, y2: int
) -> tuple[int, int, int]:
    sub_bgr = img_bgr[y1:y2, x1:x2]
    sub_g = base_gray[y1:y2, x1:x2]
    light_bool = sub_g >= 175
    if not np.any(light_bool):
        return (255, 255, 255)
    med_bgr = np.median(sub_bgr[light_bool], axis=0)
    bg_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))
    if float(np.median(sub_g[light_bool])) >= 244.0 and (max(bg_bgr) - min(bg_bgr)) <= 8:
        return (255, 255, 255)
    return bg_bgr


def _clean_exact_cv_bubble_and_build_render_slots_cv2(
    img_bgr: "np.ndarray",
    base_gray: "np.ndarray",
    cv_obj: dict,
    trans_text: str,
    other_cv_boxes: list[tuple[int, int, int, int]] | None = None,
    white_exclusion_mask: "np.ndarray | None" = None,
) -> list[list]:
    height, width = img_bgr.shape[:2]
    ymin, xmin, ymax, xmax = cv_obj["bbox_norm"]
    x1 = max(0, int((xmin / 1000.0) * width))
    y1 = max(0, int((ymin / 1000.0) * height))
    x2 = min(width, int((xmax / 1000.0) * width))
    y2 = min(height, int((ymax / 1000.0) * height))
    if (x2 - x1) < 16 or (y2 - y1) < 14:
        return []

    is_dark = bool(cv_obj.get("is_dark"))
    is_pastel = bool(cv_obj.get("is_pastel"))
    rx1, ry1, rx2, ry2, mask_full, rect_fill = _refine_cv_bubble_mask_locally_cv2(
        img_bgr, cv_obj, x1, y1, x2, y2, other_cv_boxes=other_cv_boxes
    )
    bw, bh = rx2 - rx1, ry2 - ry1
    if bw < 16 or bh < 14:
        return []

    if is_pastel:
        # Keep full sealed mask on pastel/borderless bubbles (only trim 1px at very outer dark borders)
        # so rim-touching kanji like 調子乗/思 on gloom bubbles are erased cleanly!
        outer_ring = cv2.subtract(mask_full, cv2.erode(mask_full, _CV2_K3, iterations=1))
        hard_border_px = (outer_ring > 0) & (base_gray[ry1:ry2, rx1:rx2] < 105)
        safe_core = np.where(hard_border_px, np.uint8(0), mask_full)
        if not np.any(safe_core):
            safe_core = mask_full
    else:
        safe_core = cv2.erode(mask_full, _CV2_K3, iterations=1)
        if not np.any(safe_core):
            safe_core = mask_full

    # Never let a dark bubble erase or overwrite an enclosed white/pastel speech bubble
    if is_dark and white_exclusion_mask is not None:
        safe_core = np.where(white_exclusion_mask[ry1:ry2, rx1:rx2] > 0, np.uint8(0), safe_core)
        if not np.any(safe_core):
            return []

    sub_gray = base_gray[ry1:ry2, rx1:rx2]
    sub_bgr = img_bgr[ry1:ry2, rx1:rx2]

    if is_pastel:
        erase_bool, bg_bgr = _inpaint_pastel_bubble_region_cv2(sub_bgr, sub_gray, safe_core)
        fg_rgb = (15, 15, 15)
        stroke_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        place_mask = safe_core
    else:
        non_chroma = _build_true_chroma_art_shield_cv2(sub_bgr, chroma_thresh=34, is_dark=is_dark)
        if is_dark:
            dark_bgr, bg_lum, _ = _sample_dark_bg_bgr_and_lum_cv2(img_bgr, base_gray, rx1, ry1, rx2, ry2)
            diff_ink = np.where(np.abs(sub_gray.astype(np.int16) - bg_lum) > 18, np.uint8(255), np.uint8(0))
            ink_dil = _fast_dilate_mask_cv2(diff_ink, kernel_half=3)
            erase_bool = (ink_dil > 0) & (safe_core > 0) & (non_chroma > 0)
            sub_bgr[erase_bool] = dark_bgr
            fg_rgb = (250, 250, 250)
            sc_val = max(0, bg_lum - 20)
            stroke_rgb = (sc_val, sc_val, sc_val)
        else:
            ink_mask = np.where(sub_gray < 246, np.uint8(255), np.uint8(0))
            ink_dil = _fast_dilate_mask_cv2(ink_mask, kernel_half=3)
            erase_bool = (ink_dil > 0) & (safe_core > 0) & (non_chroma > 0)
            sub_bgr[erase_bool] = (255, 255, 255)
            fg_rgb = (15, 15, 15)
            stroke_rgb = (255, 255, 255)

        place_mask = np.where((safe_core > 0) & (non_chroma > 0), np.uint8(255), np.uint8(0))
        if not np.any(place_mask):
            place_mask = safe_core

    eys, exs = np.nonzero(erase_bool)
    erased_bb = (int(exs.min()), int(eys.min()), int(exs.max()) + 1, int(eys.max()) + 1) if eys.size else None

    # 1. Connected figure-8 double bubble -> flow text cleanly across BOTH lobes!
    if not is_dark and len(trans_text) >= 28:
        lobe_rects = _detect_two_lobe_render_rects_cv2(place_mask, rx1, ry1)
        if lobe_rects:
            r1, r2 = lobe_rects
            a1 = max(1, (r1[2] - r1[0]) * (r1[3] - r1[1]))
            a2 = max(1, (r2[2] - r2[0]) * (r2[3] - r2[1]))
            split_pair = _split_text_for_two_lobes(trans_text, a1 / float(a1 + a2))
            if split_pair:
                t1, t2 = split_pair
                return [
                    [r1[0], r1[1], r1[2], r1[3], fg_rgb, stroke_rgb, t1, r1],
                    [r2[0], r2[1], r2[2], r2[3], fg_rgb, stroke_rgb, t2, r2],
                ]

    # 2. Concave white panel wrapped around a hand/character
    if (not is_dark) and rect_fill < 0.60 and erased_bb is not None:
        ex1, ey1, ex2, ey2 = erased_bb
        ew, eh = ex2 - ex1, ey2 - ey1
        if ew < bw * 0.65 or eh < bh * 0.65:
            pad_ix = max(24, int(bw * 0.14))
            pad_iy = max(16, int(bh * 0.08))
            local_place = np.zeros_like(place_mask)
            ly1, ly2 = max(0, ey1 - pad_iy), min(bh, ey2 + pad_iy)
            lx1, lx2 = max(0, ex1 - pad_ix), min(bw, ex2 + pad_ix)
            local_place[ly1:ly2, lx1:lx2] = place_mask[ly1:ly2, lx1:lx2]
            inscribed = _largest_inscribed_rect_in_mask_cv2(local_place)
            if inscribed and (inscribed[2] - inscribed[0]) >= 36 and (inscribed[3] - inscribed[1]) >= 24:
                ix1, iy1, ix2, iy2 = inscribed
                hb = (rx1 + ix1 + 2, ry1 + iy1 + 2, rx1 + ix2 - 2, ry1 + iy2 - 2)
                return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

    # 3. Standard / tilted oval: exact Largest Inscribed Rectangle
    inscribed = _largest_inscribed_rect_in_mask_cv2(place_mask)
    if inscribed and (inscribed[2] - inscribed[0]) >= 18 and (inscribed[3] - inscribed[1]) >= 14:
        ix1, iy1, ix2, iy2 = inscribed
        hb = (rx1 + ix1 + 2, ry1 + iy1 + 2, rx1 + ix2 - 2, ry1 + iy2 - 2)
        return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

    cys, cxs = np.nonzero(safe_core)
    if cys.size:
        hb = (rx1 + int(cxs.min()) + 3, ry1 + int(cys.min()) + 3, rx1 + int(cxs.max()) - 2, ry1 + int(cys.max()) - 2)
    else:
        hb = (rx1 + 4, ry1 + 4, rx2 - 4, ry2 - 4)
    return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]


def _clean_direct_on_art_box_cv2(
    img_bgr: "np.ndarray",
    base_gray: "np.ndarray",
    raw_x1: int,
    raw_y1: int,
    raw_x2: int,
    raw_y2: int,
    has_cv_bubbles: bool = False,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    """
    Case 4 fallback for `box_2d` dialogue that has NO classic white/pastel/black enclosed bubble
    (e.g. full-color CG/fantasy stories with white/black outlined text directly on dark or colored
    artwork, semi-transparent visual-novel dialogue bars, or highly saturated colored bubbles):
    1) Verifies genuine high-contrast text strokes (`p95 - p05 >= 60` + compact glyph CCs) so
       blank artwork/faces are never overwritten.
    2) Erases the original text + outline/drop-shadow ring via fast TELEA inpainting.
    3) If the underlying art is visually busy (`bg_std > 42`), blends a subtle feathered scrim
       so the translated text remains effortlessly legible without harsh rectangular blocks.
    """
    height, width = img_bgr.shape[:2]
    bw, bh = raw_x2 - raw_x1, raw_y2 - raw_y1
    if bw < 18 or bh < 16:
        return None

    sub_g = base_gray[raw_y1:raw_y2, raw_x1:raw_x2]
    sub_bgr = img_bgr[raw_y1:raw_y2, raw_x1:raw_x2]
    sub_ch = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)
    total_px = max(1, sub_g.size)

    p05, p50, p95 = [float(v) for v in np.percentile(sub_g, (5, 50, 95))]
    if (p95 - p05) < 60.0:
        return None

    def _count_compact_glyphs(mask_bin: "np.ndarray") -> int:
        n_cc, _, cc_st, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        max_gw = max(50, int(bw * 0.48))
        max_gh = max(55, int(bh * 0.48))
        return sum(
            1
            for k in range(1, n_cc)
            if 2 <= int(cc_st[k, cv2.CC_STAT_WIDTH]) <= max_gw
            and 2 <= int(cc_st[k, cv2.CC_STAT_HEIGHT]) <= max_gh
            and int(cc_st[k, cv2.CC_STAT_AREA]) >= 4
        )

    # First check standard achromatic white/black text; if 0-CV-bubble CG page, also allow colored text!
    bright_achrom = (sub_g >= max(208.0, p50 + 50.0)) & (sub_ch <= 60)
    dark_achrom = (sub_g <= min(110.0, p50 - 45.0)) & (sub_ch <= 68)
    bright_ratio = float(np.count_nonzero(bright_achrom)) / float(total_px)
    dark_ratio = float(np.count_nonzero(dark_achrom)) / float(total_px)

    # Pick natural text polarity from background luminance
    if p50 >= 140.0:
        primary_white = False if dark_ratio >= 0.012 else (bright_ratio > dark_ratio * 1.5)
    else:
        primary_white = True if bright_ratio >= 0.008 else (bright_ratio >= dark_ratio)

    min_req_glyphs = 3 if has_cv_bubbles else 2
    use_white_text = primary_white
    colored_text_mode = False

    prim_mask = np.where(bright_achrom if primary_white else dark_achrom, np.uint8(255), np.uint8(0))
    if _count_compact_glyphs(prim_mask) < min_req_glyphs:
        alt_mask = np.where(dark_achrom if primary_white else bright_achrom, np.uint8(255), np.uint8(0))
        if _count_compact_glyphs(alt_mask) >= min_req_glyphs:
            use_white_text = not primary_white
        elif not has_cv_bubbles:
            # Luminance-only fallback for colored text (e.g. yellow/pink/cyan text on dark CG, red/blue on light)
            bright_lum = np.where(sub_g >= max(195.0, p50 + 55.0), np.uint8(255), np.uint8(0))
            dark_lum = np.where(sub_g <= min(115.0, p50 - 50.0), np.uint8(255), np.uint8(0))
            gb = _count_compact_glyphs(bright_lum)
            gd = _count_compact_glyphs(dark_lum)
            if max(gb, gd) < min_req_glyphs:
                return None
            use_white_text = (p50 < 140.0 and gb >= min_req_glyphs) or (gb > gd)
            colored_text_mode = True
        else:
            return None

    pad = 6
    wx1, wy1 = max(0, raw_x1 - pad), max(0, raw_y1 - pad)
    wx2, wy2 = min(width, raw_x2 + pad), min(height, raw_y2 + pad)
    ww, wh = wx2 - wx1, wy2 - wy1
    w_bgr = img_bgr[wy1:wy2, wx1:wx2]
    w_g = base_gray[wy1:wy2, wx1:wx2]
    w_ch = w_bgr.max(axis=2) - w_bgr.min(axis=2)

    in_box = np.zeros((wh, ww), dtype=bool)
    in_box[raw_y1 - wy1 : raw_y2 - wy1, raw_x1 - wx1 : raw_x2 - wx1] = True

    if use_white_text:
        if colored_text_mode:
            text_core = np.where(in_box & (w_g >= max(195.0, p50 + 52.0)), np.uint8(255), np.uint8(0))
        else:
            text_core = np.where(in_box & (w_g >= max(205.0, p50 + 48.0)) & (w_ch <= 62), np.uint8(255), np.uint8(0))
        ink_dil = _fast_dilate_mask_cv2(text_core, kernel_half=5)
        ring = cv2.dilate(ink_dil, _CV2_K5, iterations=1)
        dark_outline = (ring > 0) & in_box & (w_g < min(95.0, max(45.0, p50 - 25.0)))
        erase_u8 = cv2.dilate(np.where((ink_dil > 0) | dark_outline, np.uint8(255), np.uint8(0)), _CV2_K3, iterations=1)
        fg_rgb = (252, 252, 252)
        stroke_rgb = (16, 16, 16)
    else:
        if colored_text_mode:
            text_core = np.where(in_box & (w_g <= min(115.0, p50 - 45.0)), np.uint8(255), np.uint8(0))
        else:
            text_core = np.where(in_box & (w_g <= min(115.0, p50 - 40.0)) & (w_ch <= 70), np.uint8(255), np.uint8(0))
        ink_dil = _fast_dilate_mask_cv2(text_core, kernel_half=5)
        ring = cv2.dilate(ink_dil, _CV2_K5, iterations=1)
        white_outline = (ring > 0) & in_box & (w_g >= max(220.0, p50 + 25.0)) & (w_ch <= 34)
        erase_u8 = cv2.dilate(np.where((ink_dil > 0) | white_outline, np.uint8(255), np.uint8(0)), _CV2_K3, iterations=1)
        fg_rgb = (15, 15, 15)
        stroke_rgb = (250, 250, 250)

    erase_bool = (erase_u8 > 0) & in_box
    if np.any(erase_bool):
        inp_mask = np.where(erase_bool, np.uint8(255), np.uint8(0))
        inp_scale = min(0.5, 220.0 / float(max(ww, wh)))
        if ww >= 40 and wh >= 40 and inp_scale < 1.0:
            iw, ih = max(16, int(round(ww * inp_scale))), max(16, int(round(wh * inp_scale)))
            sm_bgr = cv2.resize(w_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
            sm_m = np.where(cv2.resize(inp_mask, (iw, ih), interpolation=cv2.INTER_LINEAR) >= 32, np.uint8(255), np.uint8(0))
            sm_inp = cv2.inpaint(sm_bgr, sm_m, 4, cv2.INPAINT_TELEA)
            full_inp = cv2.resize(sm_inp, (ww, wh), interpolation=cv2.INTER_LINEAR)
            w_bgr[erase_bool] = full_inp[erase_bool]
        else:
            full_inp = cv2.inpaint(w_bgr, inp_mask, 4, cv2.INPAINT_TELEA)
            w_bgr[erase_bool] = full_inp[erase_bool]

    # If the background art is busy after inpainting, blend a soft feathered translucent scrim
    bg_after = w_bgr[raw_y1 - wy1 : raw_y2 - wy1, raw_x1 - wx1 : raw_x2 - wx1]
    bg_std_after = float(np.std(bg_after, axis=(0, 1)).mean())
    if bg_std_after > 42.0:
        scrim_col = (
            np.array([18, 18, 22], dtype=np.float32)
            if use_white_text
            else np.array([248, 248, 252], dtype=np.float32)
        )
        alpha = 0.42 if use_white_text else 0.52
        m_pill = np.zeros((bh, bw), dtype=np.float32)
        rad = max(6, min(18, min(bw, bh) // 5))
        cv2.rectangle(m_pill, (rad, 2), (bw - rad - 1, bh - 3), 1.0, -1)
        cv2.rectangle(m_pill, (2, rad), (bw - 3, bh - rad - 1), 1.0, -1)
        for cx_c, cy_c in ((rad, rad), (bw - rad - 1, rad), (rad, bh - rad - 1), (bw - rad - 1, bh - rad - 1)):
            cv2.circle(m_pill, (cx_c, cy_c), max(1, rad - 2), 1.0, -1)
        m_pill = cv2.GaussianBlur(m_pill, (0, 0), sigmaX=4.0) * alpha
        blended = bg_after.astype(np.float32) * (1.0 - m_pill[:, :, None]) + scrim_col * m_pill[:, :, None]
        img_bgr[raw_y1:raw_y2, raw_x1:raw_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return raw_x1, raw_y1, raw_x2, raw_y2, fg_rgb, stroke_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)


def _clean_unmarked_open_bubble_and_measure_box_cv2(
    img_bgr: "np.ndarray",
    base_gray: "np.ndarray",
    raw_x1: int,
    raw_y1: int,
    raw_x2: int,
    raw_y2: int,
    has_cv_bubbles: bool = False,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    height, width = img_bgr.shape[:2]
    bw, bh = raw_x2 - raw_x1, raw_y2 - raw_y1
    if bw < 18 or bh < 16:
        return None

    sub_g = base_gray[raw_y1:raw_y2, raw_x1:raw_x2]
    sub_bgr_chk = img_bgr[raw_y1:raw_y2, raw_x1:raw_x2]
    total_px = max(1, sub_g.size)
    median_lum = float(np.median(sub_g))

    ch_band = sub_bgr_chk.max(axis=2) - sub_bgr_chk.min(axis=2)
    mean_chroma = float(ch_band.mean())
    chroma_ratio = float(np.count_nonzero(ch_band >= 28)) / float(total_px)

    def _fallback_case4():
        return _clean_direct_on_art_box_cv2(
            img_bgr, base_gray, raw_x1, raw_y1, raw_x2, raw_y2, has_cv_bubbles=has_cv_bubbles
        )

    # Case 1: Dark speech bubble (strictly achromatic uniform dark background + bright text)
    if median_lum < 125.0:
        if mean_chroma >= 18.0 or chroma_ratio >= 0.18:
            return _fallback_case4()
        s_band = cv2.cvtColor(sub_bgr_chk, cv2.COLOR_BGR2HSV)[:, :, 1]
        if float(s_band.mean()) >= 22.0 or (float(np.count_nonzero(s_band >= 28)) / float(total_px)) >= 0.16:
            return _fallback_case4()
        dark_bgr, bg_lum, bg_std = _sample_dark_bg_bgr_and_lum_cv2(img_bgr, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        bright_ratio = float(np.count_nonzero(sub_g >= 220)) / float(total_px)
        if bg_std > 24.0 or not (0.02 <= bright_ratio <= 0.38):
            return _fallback_case4()

        pad = 6
        wx1, wy1 = max(0, raw_x1 - pad), max(0, raw_y1 - pad)
        wx2, wy2 = min(width, raw_x2 + pad), min(height, raw_y2 + pad)
        sg = base_gray[wy1:wy2, wx1:wx2]
        s_bgr_d = img_bgr[wy1:wy2, wx1:wx2]
        non_chroma_d = _build_true_chroma_art_shield_cv2(s_bgr_d, chroma_thresh=24, is_dark=True)
        diff = np.where(np.abs(sg.astype(np.int16) - bg_lum) > 18, np.uint8(255), np.uint8(0))
        erase_d = _fast_dilate_mask_cv2(diff, kernel_half=3)
        s_bgr_d[(erase_d > 0) & (non_chroma_d > 0)] = dark_bgr
        sc_val = max(0, bg_lum - 20)
        return raw_x1, raw_y1, raw_x2, raw_y2, (250, 250, 250), (sc_val, sc_val, sc_val), (raw_x1, raw_y1, raw_x2, raw_y2)

    is_pastel_open = (mean_chroma >= 18.0 or chroma_ratio >= 0.18)

    # Case 2: Pastel / colored light bubble (`box_2d` fallback for borderless pastel bubbles)
    if is_pastel_open:
        if median_lum < 170.0:
            return _fallback_case4()
        s_band = cv2.cvtColor(sub_bgr_chk, cv2.COLOR_BGR2HSV)[:, :, 1]
        mean_s = float(s_band.mean())
        high_sat_ratio = float(np.count_nonzero(s_band >= 100)) / float(total_px)
        if mean_s >= 85.0 or high_sat_ratio >= 0.30:
            return _fallback_case4()

        light_bool = sub_g >= 170
        light_ratio = float(np.count_nonzero(light_bool)) / float(total_px)
        ink_ratio = float(np.count_nonzero((sub_g < 155) & (ch_band < 36))) / float(total_px)
        if light_ratio < 0.42 or ink_ratio < 0.010:
            return _fallback_case4()
        bg_std = float(np.std(sub_bgr_chk[light_bool], axis=0).mean())
        if bg_std > 32.0:
            return _fallback_case4()

        pad_x, pad_y = 6, 8
        wx1, wy1 = max(0, raw_x1 - pad_x), max(0, raw_y1 - pad_y)
        wx2, wy2 = min(width, raw_x2 + pad_x), min(height, raw_y2 + pad_y)
        sg = base_gray[wy1:wy2, wx1:wx2]
        s_bgr = img_bgr[wy1:wy2, wx1:wx2]
        s_sat = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]

        p_seed = np.where((sg >= 175) & (s_sat < 95), np.uint8(255), np.uint8(0))
        p_zone = _fast_dilate_mask_cv2(p_seed, kernel_half=5)
        erase_bool, bg_bgr = _inpaint_pastel_bubble_region_cv2(s_bgr, sg, p_zone)
        if not np.any(erase_bool):
            return _fallback_case4()
        stroke_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), stroke_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)

    # Case 3: Classic white / light achromatic open bubble
    white_ratio = float(np.count_nonzero(sub_g >= 220)) / float(total_px)
    ink_ratio = float(np.count_nonzero(sub_g < 155)) / float(total_px)
    if white_ratio < 0.38 or ink_ratio < 0.010:
        return _fallback_case4()

    pad_x, pad_y = 8, 12
    wx1, wy1 = max(0, raw_x1 - pad_x), max(0, raw_y1 - pad_y)
    wx2, wy2 = min(width, raw_x2 + pad_x), min(height, raw_y2 + pad_y)

    sg = base_gray[wy1:wy2, wx1:wx2]
    s_bgr = img_bgr[wy1:wy2, wx1:wx2]
    non_chroma = _build_true_chroma_art_shield_cv2(s_bgr, chroma_thresh=30, is_dark=False)

    w_seed = np.where(sg >= 228, np.uint8(255), np.uint8(0))
    w_zone = _fast_dilate_mask_cv2(w_seed, kernel_half=5)
    ink_m = np.where(sg < 232, np.uint8(255), np.uint8(0))
    ink_dil = _fast_dilate_mask_cv2(ink_m, kernel_half=3)

    erase_bool = (ink_dil > 0) & (w_zone > 0) & (non_chroma > 0)
    if not np.any(erase_bool):
        return _fallback_case4()

    bg_bgr = _sample_light_bg_bgr_cv2(img_bgr, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
    s_bgr[erase_bool] = bg_bgr
    stroke_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), stroke_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)


def _render_translated_manga_image_cv2_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
    cv_bubbles: list[dict] | None = None,
) -> bytes | None:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        from manga_pil_fallback import _render_translated_manga_image_pil_sync
        return _render_translated_manga_image_pil_sync(raw_bytes, bubbles, cv_bubbles)

    base_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = img_bgr.shape[:2]

    cv_map: dict[int, dict] = {
        int(c["bubble_id"]): c for c in (cv_bubbles or []) if c.get("bubble_id") is not None
    }
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

    white_exclusion_mask: "np.ndarray | None" = None
    if has_dark_cv:
        white_exclusion_mask = np.zeros((height, width), dtype=np.uint8)
        for cid, c_obj in cv_map.items():
            if c_obj.get("is_dark"):
                continue
            bx1, by1, bx2, by2 = cv_boxes_px[cid]
            if (bx2 - bx1) >= 4 and (by2 - by1) >= 4:
                m_src = c_obj.get("mask_np")
                if m_src is None and c_obj.get("mask_small") is not None:
                    m_src = np.array(c_obj["mask_small"], dtype=np.uint8)
                if m_src is not None:
                    m_up = np.where(
                        cv2.resize(m_src, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_NEAREST) >= 128,
                        np.uint8(255),
                        np.uint8(0),
                    )
                    white_exclusion_mask[by1:by2, bx1:bx2] |= m_up
        if np.any(white_exclusion_mask):
            white_exclusion_mask = cv2.dilate(white_exclusion_mask, _CV2_K5, iterations=1)
        else:
            white_exclusion_mask = None

    prepared_renders: list[list] = []
    used_cv_ids: set[int] = set()
    rendered_rects: list[tuple[int, int, int, int]] = []

    # Pass 1: Render all CV-anchored Set-of-Mark bubbles
    for b in bubbles:
        bid = b.get("bubble_id")
        tr_txt = _sanitize_text_for_manga_font(b.get("translated_text") or "")
        if not tr_txt or bid is None or bid in used_cv_ids or bid not in cv_map:
            continue

        other_boxes = [box for cid, box in cv_boxes_px.items() if cid != bid]
        slots = _clean_exact_cv_bubble_and_build_render_slots_cv2(
            img_bgr,
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

    # Pass 2: Render unmarked open/borderless bubbles (`bubble_id == None` + `box_2d`)
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
            slots = _clean_exact_cv_bubble_and_build_render_slots_cv2(
                img_bgr,
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

        cleaned_open = _clean_unmarked_open_bubble_and_measure_box_cv2(img_bgr, base_gray, rx1, ry1, rx2, ry2, has_cv_bubbles=bool(cv_map))
        if cleaned_open is None:
            continue
        lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned_open
        prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
        rendered_rects.append(hard_bounds)

    white_exclusion_mask = None
    if not prepared_renders:
        return None

    prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)
    del base_gray

    # Single BGR -> RGB Pillow handoff for crisp Thai/Latin TrueType typesetting
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    del img_bgr
    pil_img = Image.fromarray(img_rgb)
    del img_rgb

    try:
        max_dim = max(width, height)
        max_font_px = max(28, int(max_dim * 0.021))
        draw = ImageDraw.Draw(pil_img)

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
                s_width = 3 if (fg_text == (252, 252, 252) or stroke_col == (250, 250, 250)) else 2
                draw.text(
                    (curr_x, curr_y - ref_top),
                    ln,
                    font=chosen_font,
                    fill=fg_text,
                    stroke_width=s_width,
                    stroke_fill=stroke_col,
                )
                curr_y += line_height

        out_io = io.BytesIO()
        pil_img.save(out_io, format="JPEG", quality=MANGA_RENDER_JPEG_QUALITY, optimize=True)
        return out_io.getvalue()
    finally:
        pil_img.close()


def render_translated_manga_image_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
    cv_bubbles: list[dict] | None = None,
) -> bytes | None:
    """
    Renders translated dialogue at FULL ORIGINAL RESOLUTION using vectorized OpenCV + NumPy
    masking when available, falling back lazily to pure Pillow (`manga_pil_fallback.py`) otherwise.
    """
    if not raw_bytes or not bubbles:
        return None
    if HAS_CV2:
        return _render_translated_manga_image_cv2_sync(raw_bytes, bubbles, cv_bubbles)
    from manga_pil_fallback import _render_translated_manga_image_pil_sync
    return _render_translated_manga_image_pil_sync(raw_bytes, bubbles, cv_bubbles)