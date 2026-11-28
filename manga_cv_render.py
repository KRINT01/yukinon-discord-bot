import os
import io
import re
import glob
import functools

from PIL import Image, ImageDraw, ImageFont

from config import THAI_CHAR_REGEX
from manga_cv_detect import (
    OPENCV_THREADS,
    HAS_CV2,
    np,
    cv2,
    MANGA_CV_RENDER_VERSION,
    SOM_CV_MAX_DIM,
    MANGA_RENDER_JPEG_QUALITY,
    _CV2_K3,
    _CV2_K5,
    _CV2_K7,
    _CV2_K9,
    _CV2_ELLIPSE_7,
    _CV2_ELLIPSE_11,
    _CV2_ELLIPSE_13,
    _CV2_ELLIPSE_17,
    _coerce_image_bytes_sync,
    _prune_nested_child_bubbles,
    _sort_bubbles_manga_reading_order,
    _release_cv_page_masks,
    _fill_mask_holes_cv2,
    _close_rim_bays_cv2,
    _collect_margin_labels_cv2,
    _extract_and_filter_white_components_cv2,
    _recover_gutter_merged_white_bubbles_cv2,
    _extract_bw_soft_shaded_bubbles_cv2,
    _detect_colored_ui_banners_cv2,
    _detect_pastel_table_labels_cv2,
    _detect_white_outlined_text_clusters_cv2,
    _detect_page_bubbles_and_build_som_b64_cv2_sync,
    _detect_page_bubbles_and_build_som_b64_sync,
)

# Debian `fonts-tlwg-garuda-otf` installs OpenType (.otf) under `/usr/share/fonts/opentype/tlwg/`.
# Garuda OTF/TTF MUST come first because Debian's `NotoSansThai-Bold.ttf` is Thai-only and lacks
# ASCII letters, digits (`0-9`), quotes (`'`, `"`), parentheses, colons, commas, and periods!
MANGA_THAI_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/tlwg/Garuda-Bold.otf",
    "/usr/share/fonts/opentype/tlwg/Garuda.otf",
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",
    "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf",
    "/usr/share/fonts/opentype/tlwg/Waree-Bold.otf",
    "/usr/share/fonts/truetype/tlwg/Waree-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/tlwg/Garuda-Bold.otf",
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
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
# Guarantees ZERO missing-glyph `[]` tofu boxes in Garuda-Bold / NotoSans-Bold!
_FONT_UNSAFE_CHARS_REGEX = re.compile(r"[^\x20-\x7E\u00C0-\u00FF\u0E00-\u0E5B]")
# Unicode invariant: fix malformed Thai tone-before-vowel order (e.g. อ๊ิก -> อิ๊ก) so Garuda-Bold never collides marks
_THAI_TONE_BEFORE_VOWEL_REGEX = re.compile(r"([\u0E48-\u0E4B])([\u0E31\u0E34-\u0E3A])")
# Split lines into Thai vs non-Thai runs when dual-font fallback is active
_THAI_RUN_SPLIT_REGEX = re.compile(r"([\u0E00-\u0E7F]+)")


# ==============================================================================
# SHARED FONTS, DUAL-FONT GLYPH SAFETY, TEXT SANITIZATION & LAYOUT ENGINE
# ==============================================================================
def _font_has_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    """Checks whether `font` renders `ch` as a real glyph rather than the `.notdef` tofu box."""
    try:
        m_ch = font.getmask(ch)
        if m_ch.size[0] == 0 or m_ch.size[1] == 0:
            return ch.isspace()
        m_nd = font.getmask("\uffff")
        if m_ch.size == m_nd.size and bytes(m_ch) == bytes(m_nd):
            return False
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def _font_path_supports_ascii(path: str) -> bool:
    try:
        probe_font = ImageFont.truetype(path, size=20)
        return all(_font_has_glyph(probe_font, ch) for ch in ("A", "0", '"', "'", "(", ","))
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def _font_path_supports_thai(path: str) -> bool:
    try:
        probe_font = ImageFont.truetype(path, size=20)
        return all(_font_has_glyph(probe_font, ch) for ch in ("ก", "ข", "น"))
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _discover_extra_thai_font_paths() -> tuple[str, ...]:
    """Discovers TLWG Garuda/Loma/Waree OTF/TTF fonts anywhere under `/usr/share/fonts` if not at default paths."""
    found: list[str] = []
    for pattern in (
        "/usr/share/fonts/**/Garuda*Bold*.[ot]tf",
        "/usr/share/fonts/**/Garuda*.[ot]tf",
        "/usr/share/fonts/**/Loma*Bold*.[ot]tf",
        "/usr/share/fonts/**/Waree*Bold*.[ot]tf",
    ):
        for p in sorted(glob.glob(pattern, recursive=True)):
            if p not in found:
                found.append(p)
    return tuple(found)


@functools.lru_cache(maxsize=64)
def _load_manga_font(size: int, is_thai: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if is_thai:
        all_candidates = list(dict.fromkeys(list(MANGA_THAI_FONT_CANDIDATES) + list(_discover_extra_thai_font_paths())))
        # Pass 1: Strongly prefer a unified font with BOTH Thai AND ASCII/digits/quotes (e.g. Garuda-Bold.otf)
        for path in all_candidates:
            if os.path.exists(path) and _font_path_supports_thai(path) and _font_path_supports_ascii(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
        # Pass 2: Fall back to any Thai font (paired with Latin fallback at draw time!)
        for path in all_candidates:
            if os.path.exists(path) and _font_path_supports_thai(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
    else:
        for path in LATIN_FONT_CANDIDATES:
            if os.path.exists(path) and _font_path_supports_ascii(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue

    for path in (MANGA_THAI_FONT_CANDIDATES if is_thai else LATIN_FONT_CANDIDATES):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _font_needs_latin_fallback(font: ImageFont.ImageFont) -> bool:
    """True only when a Thai-only subset font (e.g. Debian NotoSansThai-Bold.ttf) is loaded without ASCII glyphs."""
    fpath = getattr(font, "path", None)
    if not fpath:
        return False
    return not _font_path_supports_ascii(str(fpath))


def _get_latin_companion_font(primary_font: ImageFont.ImageFont) -> ImageFont.ImageFont:
    fsize = int(getattr(primary_font, "size", 16) or 16)
    return _load_manga_font(fsize, is_thai=False)


def _measure_text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> int:
    """Measures rendered pixel width accurately, using dual-font runs only when primary font lacks ASCII glyphs."""
    if not text:
        return 0
    if not _font_needs_latin_fallback(font):
        bb = draw.textbbox((0, 0), text, font=font)
        return int(bb[2] - bb[0])
    latin_font = _get_latin_companion_font(font)
    total_w = 0.0
    for seg in _THAI_RUN_SPLIT_REGEX.split(text):
        if not seg:
            continue
        seg_font = font if THAI_CHAR_REGEX.search(seg) else latin_font
        try:
            total_w += float(draw.textlength(seg, font=seg_font))
        except Exception:
            bb = draw.textbbox((0, 0), seg, font=seg_font)
            total_w += float(bb[2] - bb[0])
    return int(round(total_w))


def _draw_text_line_safe(
    draw: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int],
) -> None:
    """Draws a single line cleanly; automatically routes non-Thai segments to Latin font if primary lacks ASCII."""
    if not text:
        return
    if not _font_needs_latin_fallback(font):
        draw.text(
            pos,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return

    latin_font = _get_latin_companion_font(font)
    cur_x, cur_y = float(pos[0]), float(pos[1])
    for seg in _THAI_RUN_SPLIT_REGEX.split(text):
        if not seg:
            continue
        seg_font = font if THAI_CHAR_REGEX.search(seg) else latin_font
        draw.text(
            (cur_x, cur_y),
            seg,
            font=seg_font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        try:
            cur_x += float(draw.textlength(seg, font=seg_font))
        except Exception:
            bb = draw.textbbox((0, 0), seg, font=seg_font)
            cur_x += float(bb[2] - bb[0])


def _sanitize_text_for_manga_font(text: str) -> str:
    """
    Normalizes Japanese brackets (`「」「」『』`) & fullwidth punctuation into clean ASCII
    equivalents, inserts tidy spacing where Thai and Latin/quote/paren tokens touch directly,
    fixes Thai combining-mark order (vowel before tone), and enforces a strict font-safe
    whitelist (`ASCII + Thai + Latin-1`) so missing-glyph `[]` tofu boxes NEVER appear!
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
        .replace("　", " ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("〝", '"')
        .replace("〞", '"')
        .replace("「", '"')
        .replace("」", '"')
        .replace("『", '"')
        .replace("』", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("【", "[")
        .replace("】", "]")
        .replace("！", "!")
        .replace("？", "?")
        .replace("（", "(")
        .replace("）", ")")
        .replace("：", ": ")
        .replace("；", "; ")
        .replace("，", ", ")
        .replace("、", ", ")
        .replace("。", ".")
    )
    # Remove any character outside Garuda-Bold / NotoSans-Bold glyph coverage (kills ♥/♡/♪/Hangul/Arabic/CJK/emoji tofu!)
    s = _FONT_UNSAFE_CHARS_REGEX.sub("", s)
    # Unicode combining-mark invariant: ensure Thai above/below vowels always precede tone marks
    if THAI_CHAR_REGEX.search(s):
        s = _THAI_TONE_BEFORE_VOWEL_REGEX.sub(r"\2\1", s)
        # Insert clean spacing where Thai text collides directly with Latin words, opening parens, or opening quotes
        s = re.sub(r"([\u0E00-\u0E5B])([A-Za-z(])", r"\1 \2", s)
        s = re.sub(r"([A-Za-z)])([\u0E00-\u0E5B])", r"\1 \2", s)
        # Pad paired double/single quotes cleanly when touching Thai text on either side
        # (e.g. `ฉันชื่อ"ยุย"ค่ะ` -> `ฉันชื่อ "ยุย" ค่ะ`) without misplacing spaces inside closing quotes or English contractions
        s = re.sub(r'([\u0E00-\u0E5B])"([^"\n]+)"', r'\1 "\2"', s)
        s = re.sub(r'"([^"\n]+)"([\u0E00-\u0E5B])', r'"\1" \2', s)
        s = re.sub(r"([\u0E00-\u0E5B])'([^'\n]+)'", r"\1 '\2'", s)
        s = re.sub(r"'([^'\n]+)'([\u0E00-\u0E5B])", r"'\1' \2", s)
    s = re.sub(r"\(\s*\)|\[\s*\]|'\s*'|\"\s*\"", "", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
        prev_closed_vowel = prev.endswith(("ะ", "ำ")) or (prev[0] in ("ใ", "ไ"))
        prev_is_ia_vowel = (prev.startswith("เ") and "\u0E35" in prev and prev.endswith("ย"))
        prev_is_ua_vowel = ("\u0E31" in prev and prev.endswith("ว"))
        prev_is_ho_nam_bare = (len(prev) == 2 and prev[0] == "ห" and prev[1] in _THAI_HO_NAM_FOLLOW)
        if prev_is_ua_vowel:
            prev_open_for_coda = (tok == "ย")
        else:
            prev_open_for_coda = (
                not prev_closed_vowel
                and len(prev) <= 6
                and (
                    prev_is_ia_vowel
                    or prev_is_ho_nam_bare
                    or (len(prev) >= 2 and prev[-1] in ("อ", "ว"))
                    or (
                        any(c in _THAI_CAN_TAKE_CODA_VOWEL for c in prev)
                        and (
                            prev[-1] in _THAI_OPEN_ENDINGS
                            or (prev[0] in ("เ", "แ", "โ") and (len(prev) == 2 or (len(prev) == 3 and prev[2] in _THAI_CLUSTER_SECOND)))
                        )
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
        if _measure_text_width(draw, test_line, font) <= max_width or not current:
            current = test_line
        else:
            out_lines.append(current)
            current = syl

    if current:
        out_lines.append(current)

    final_lines: list[str] = []
    for ln in out_lines:
        if _measure_text_width(draw, ln, font) <= max_width or len(ln) <= 3:
            final_lines.append(ln)
        else:
            chunk = ""
            for ch in ln:
                t_ch = chunk + ch
                is_comb = ("\u0E31" <= ch <= "\u0E3A") or ("\u0E47" <= ch <= "\u0E4E") or (ch in _THAI_PUNCT_ATTACH)
                if _measure_text_width(draw, t_ch, font) > max_width and chunk and not is_comb:
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
        if _measure_text_width(draw, test_line, font) <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
                current = ""
            if _measure_text_width(draw, word, font) <= max_width or (not is_thai and not allow_midword_split):
                current = word
            elif is_thai:
                thai_parts = _split_thai_word_cleanly(draw, word, font, max_width)
                lines.extend(thai_parts[:-1])
                current = thai_parts[-1] if thai_parts else ""
            else:
                chunk = ""
                for ch in word:
                    test_ch = chunk + ch
                    if _measure_text_width(draw, test_ch, font) > max_width and chunk:
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
                        renders[i][3] = max(ay1 + 14, mid_y - (min_gap // 2))
                        renders[j][1] = min(by2 - 14, mid_y + (min_gap // 2))
                    else:
                        renders[j][3] = max(by1 + 14, mid_y - (min_gap // 2))
                        renders[i][1] = min(ay2 - 14, mid_y + (min_gap // 2))
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
                    renders[idx_k][1] = max(hy1, min(hy2 - 14, renders[idx_k][1]))
                    renders[idx_k][2] = min(hx2, max(renders[idx_k][0] + 18, renders[idx_k][2]))
                    renders[idx_k][3] = min(hy2, max(renders[idx_k][1] + 14, renders[idx_k][3]))
    return renders


def _fit_text_binary_search(
    draw: ImageDraw.ImageDraw,
    trans_text: str,
    usable_w: int,
    usable_h: int,
    start_font_size: int,
    is_thai_text: bool,
) -> tuple[ImageFont.ImageFont, list[str], int, int]:
    sample_probe = "กข้ำุ" if is_thai_text else "Agjpqy!"

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
            w_ln = _measure_text_width(draw, ln, font)
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
        if mw >= 140 and mh >= 140 and (area / float(max(1, gw * gh))) < 0.68:
            r_a = _largest_inscribed_rect_in_mask_cv2(place_mask)
            if r_a is not None:
                a_w, a_h = r_a[2] - r_a[0], r_a[3] - r_a[1]
                a_area = a_w * a_h
                if a_w >= 52 and a_h >= 52:
                    rem = place_mask.copy()
                    pad_c = 6
                    rem[
                        max(0, r_a[1] - pad_c) : min(mh, r_a[3] + pad_c),
                        max(0, r_a[0] - pad_c) : min(mw, r_a[2] + pad_c),
                    ] = 0
                    r_b = _largest_inscribed_rect_in_mask_cv2(rem)
                    if r_b is not None:
                        b_w, b_h = r_b[2] - r_b[0], r_b[3] - r_b[1]
                        b_area = b_w * b_h
                        if b_w >= 52 and b_h >= 52 and b_area >= a_area * 0.28:
                            acx, acy = (r_a[0] + r_a[2]) / 2.0, (r_a[1] + r_a[3]) / 2.0
                            bcx, bcy = (r_b[0] + r_b[2]) / 2.0, (r_b[1] + r_b[3]) / 2.0
                            if abs(acx - bcx) >= min(a_w, b_w) * 0.55 and abs(acy - bcy) >= min(a_h, b_h) * 0.45:
                                ra_abs = (offset_x + r_a[0] + 2, offset_y + r_a[1] + 2, offset_x + r_a[2] - 2, offset_y + r_a[3] - 2)
                                rb_abs = (offset_x + r_b[0] + 2, offset_y + r_b[1] + 2, offset_x + r_b[2] - 2, offset_y + r_b[3] - 2)
                                pair = [ra_abs, rb_abs]
                                pair.sort(key=lambda rr: (-((rr[0] + rr[2]) / 2.0), rr[1]))
                                return pair
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

    raw_filled_s = _fill_mask_holes_cv2(comp[cy1:cy2, cx1:cx2])
    orig_sol_ref = float(cv_obj.get("solidity") or 0.90)
    if orig_sol_ref >= 0.84 and orig_fill >= 0.64:
        adapt_k = min(29, max(13, int(min(cw_s, ch_s) * 0.28))) | 1
        cand_filled_s = _close_rim_bays_cv2(raw_filled_s, kernel_size=adapt_k)
        raw_a_cnt = max(1, int(np.count_nonzero(raw_filled_s)))
        filled_s = (
            cand_filled_s
            if int(np.count_nonzero(cand_filled_s)) <= int(raw_a_cnt * 1.28)
            else _close_rim_bays_cv2(raw_filled_s, gentle=True)
        )
    else:
        filled_s = _close_rim_bays_cv2(raw_filled_s, gentle=True)
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
    br_mask = np.where(sub_g >= 150, np.uint8(255), np.uint8(0))
    br_fringe = cv2.dilate(br_mask, _CV2_K5, iterations=1) > 0
    bg_bool = (sub_g < 145) & (~br_fringe)
    if int(np.count_nonzero(bg_bool)) < 16:
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
    if (x2 - x1) < 16 or (y2 - y1) < 12:
        return []

    is_dark = bool(cv_obj.get("is_dark"))
    is_pastel = bool(cv_obj.get("is_pastel"))

    # Pass C-Banner: Colored horizontal section banner / expression caption bar (white text on solid color)
    if bool(cv_obj.get("is_ui_banner")):
        bw_b, bh_b = x2 - x1, y2 - y1
        sub_bgr = img_bgr[y1:y2, x1:x2]
        sub_g = base_gray[y1:y2, x1:x2]
        sub_ch = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)

        m_src = cv_obj.get("mask_np")
        if m_src is not None:
            mask_full = np.where(
                cv2.resize(m_src, (bw_b, bh_b), interpolation=cv2.INTER_NEAREST) >= 128,
                np.uint8(255),
                np.uint8(0),
            )
        else:
            mask_full = np.full((bh_b, bw_b), 255, dtype=np.uint8)

        inner_core = cv2.erode(mask_full, _CV2_K3, iterations=1)
        if not np.any(inner_core):
            inner_core = mask_full

        # Sample pure banner fill color (excluding white text + 1px AA fringe)
        w_text_raw = np.where((inner_core > 0) & (sub_g >= 180) & (sub_ch <= 60), np.uint8(255), np.uint8(0))
        w_fringe = cv2.dilate(w_text_raw, _CV2_K3, iterations=1) > 0
        pure_bg = (inner_core > 0) & (~w_fringe) & (sub_ch >= 20)
        if int(np.count_nonzero(pure_bg)) < 12:
            pure_bg = (inner_core > 0) & (~w_fringe)
        if not np.any(pure_bg):
            pure_bg = inner_core > 0
        med_bgr = np.median(sub_bgr[pure_bg], axis=0)
        bg_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

        # Fill eroded interior cleanly with banner color (preserves chevron/pill outer edges 100%!)
        sub_bgr[inner_core > 0] = bg_bgr
        fg_rgb = (255, 255, 255)
        stroke_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])

        inscribed = _largest_inscribed_rect_in_mask_cv2(inner_core)
        if inscribed and (inscribed[2] - inscribed[0]) >= max(24, int(bw_b * 0.55)):
            ix1, iy1, ix2, iy2 = inscribed
            hb = (x1 + ix1 + 2, y1 + iy1 + 1, x1 + ix2 - 2, y1 + iy2 - 1)
        else:
            pad_x = max(4, min(10, int(bw_b * 0.08)))
            hb = (x1 + pad_x, y1 + 1, x2 - pad_x, y2 - 1)
        return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

    # Pass C-PastelLabel: Compact uniform pastel table row header (dark text on soft lavender/pastel pill)
    if bool(cv_obj.get("is_pastel_label")):
        bw_p, bh_p = x2 - x1, y2 - y1
        sub_bgr = img_bgr[y1:y2, x1:x2]
        sub_g = base_gray[y1:y2, x1:x2]

        m_src = cv_obj.get("mask_np")
        if m_src is not None:
            mask_full = np.where(
                cv2.resize(m_src, (bw_p, bh_p), interpolation=cv2.INTER_NEAREST) >= 128,
                np.uint8(255),
                np.uint8(0),
            )
        else:
            mask_full = np.full((bh_p, bw_p), 255, dtype=np.uint8)

        inner_core = cv2.erode(mask_full, _CV2_K3, iterations=1)
        if not np.any(inner_core):
            inner_core = mask_full

        d_raw = np.where((inner_core > 0) & (sub_g <= 170), np.uint8(255), np.uint8(0))
        d_fringe = cv2.dilate(d_raw, _CV2_K3, iterations=1) > 0
        pure_bg = (inner_core > 0) & (~d_fringe)
        if int(np.count_nonzero(pure_bg)) < 12:
            pure_bg = inner_core > 0
        med_bgr = np.median(sub_bgr[pure_bg], axis=0)
        bg_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

        sub_bgr[inner_core > 0] = bg_bgr
        fg_rgb = (20, 20, 20)
        stroke_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        hb = (x1 + 3, y1 + 1, x2 - 3, y2 - 1)
        return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

    # Pass C White-Paper / White-Halo Outlined Text Cluster
    if bool(cv_obj.get("is_outlined_text")):
        bw_c = x2 - x1
        bh_c = y2 - y1
        sub_bgr_raw = img_bgr[y1:y2, x1:x2]
        sub_g_raw = base_gray[y1:y2, x1:x2]
        sub_ch_raw = sub_bgr_raw.max(axis=2) - sub_bgr_raw.min(axis=2)
        p75_g = float(np.percentile(sub_g_raw, 75))
        white_bg_ratio = float(np.count_nonzero((sub_g_raw >= 235) & (sub_ch_raw <= 18))) / float(max(1, sub_g_raw.size))

        # Clean white paper (character sheets / open white panels): erase ONLY compact text glyphs
        # to pure white while shielding BOTH colored artwork AND large dark artwork blobs (e.g. dark
        # tights/legs or portrait frames intruding into a corner/edge of the text bounding box)!
        safe_tx1, safe_ty1, safe_tx2, safe_ty2 = x1, y1, x2, y2
        if p75_g >= 245.0 and white_bg_ratio >= 0.52:
            pad_w = 4
            ex1, ey1 = max(0, x1 - pad_w), max(0, y1 - pad_w)
            ex2, ey2 = min(width, x2 + pad_w), min(height, y2 + pad_w)
            sub_bgr = img_bgr[ey1:ey2, ex1:ex2]
            sub_g = base_gray[ey1:ey2, ex1:ex2]
            sub_ch = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)
            non_chroma = _build_true_chroma_art_shield_cv2(sub_bgr, chroma_thresh=30, is_dark=False)

            max_gly_dim = max(44, int(max(width, height) * 0.045))
            max_gly_area = max(650, int(max_gly_dim * max_gly_dim * 0.35))
            dark_all = np.where((sub_g < 240) & (sub_ch < 35), np.uint8(255), np.uint8(0))
            dn_cc, d_lbls, d_st, _ = cv2.connectedComponentsWithStats(dark_all, connectivity=8)
            glyph_only = np.zeros_like(dark_all)
            dark_art = np.zeros_like(dark_all)
            for dk in range(1, dn_cc):
                dgw = int(d_st[dk, cv2.CC_STAT_WIDTH])
                dgh = int(d_st[dk, cv2.CC_STAT_HEIGHT])
                dga = int(d_st[dk, cv2.CC_STAT_AREA])
                if dgw > max_gly_dim or dgh > max_gly_dim or dga > max_gly_area:
                    dark_art[d_lbls == dk] = 255
                else:
                    glyph_only[d_lbls == dk] = 255

            art_shield = np.where((non_chroma == 0) | (dark_art > 0), np.uint8(255), np.uint8(0))
            if np.any(art_shield):
                art_shield = cv2.dilate(art_shield, _CV2_K7, iterations=1)
            erase_u8 = cv2.dilate(glyph_only, _CV2_K5, iterations=1)
            sub_bgr[(erase_u8 > 0) & (art_shield == 0)] = (255, 255, 255)

            # Trim text placement box away from intruding legs/ribbons/portrait frames
            art_in_box = art_shield[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
            if np.any(art_in_box):
                safe_place = np.where(art_in_box == 0, np.uint8(255), np.uint8(0))
                inscribed = _largest_inscribed_rect_in_mask_cv2(safe_place)
                if inscribed and (inscribed[2] - inscribed[0]) >= max(24, int(bw_c * 0.45)) and (inscribed[3] - inscribed[1]) >= max(14, int(bh_c * 0.35)):
                    safe_tx1 = x1 + inscribed[0]
                    safe_ty1 = y1 + inscribed[1]
                    safe_tx2 = x1 + inscribed[2]
                    safe_ty2 = y1 + inscribed[3]
            fg_rgb = (15, 15, 15)
            stroke_rgb = (255, 255, 255)
        else:
            pad_e = 8
            ex1, ey1 = max(0, x1 - pad_e), max(0, y1 - pad_e)
            ex2, ey2 = min(width, x2 + pad_e), min(height, y2 + pad_e)
            ebw, ebh = ex2 - ex1, ey2 - ey1
            sub_bgr = img_bgr[ey1:ey2, ex1:ex2]
            sub_g = base_gray[ey1:ey2, ex1:ex2]
            sub_ch = sub_bgr.max(axis=2) - sub_bgr.min(axis=2)

            is_very_dark_bg = float(np.percentile(sub_g, 25)) < 55.0
            w_thr_e = 165 if is_very_dark_bg else 218
            wp = (sub_g >= w_thr_e) & (sub_ch <= 36)
            da = (sub_g <= 158)
            w_near_d = wp & (cv2.dilate(da.astype(np.uint8), _CV2_K7, iterations=1) > 0)
            d_near_w = da & (cv2.dilate(wp.astype(np.uint8), _CV2_K7, iterations=1) > 0)
            erase_u8 = cv2.dilate(np.where(w_near_d | d_near_w, np.uint8(255), np.uint8(0)), _CV2_K5, iterations=1)
            erase_u8 = cv2.morphologyEx(erase_u8, cv2.MORPH_CLOSE, _CV2_K5)
            erase_bool = erase_u8 > 0

            if np.any(erase_bool):
                inp_scale = min(0.5, 220.0 / float(max(ebw, ebh)))
                if ebw >= 40 and ebh >= 40 and inp_scale < 1.0:
                    iw, ih = max(16, int(round(ebw * inp_scale))), max(16, int(round(ebh * inp_scale)))
                    sm_bgr = cv2.resize(sub_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
                    sm_m = np.where(cv2.resize(erase_u8, (iw, ih), interpolation=cv2.INTER_LINEAR) >= 32, np.uint8(255), np.uint8(0))
                    sm_inp = cv2.inpaint(sm_bgr, sm_m, 4, cv2.INPAINT_TELEA)
                    full_inp = cv2.resize(sm_inp, (ebw, ebh), interpolation=cv2.INTER_LINEAR)
                    sub_bgr[erase_bool] = full_inp[erase_bool]
                else:
                    full_inp = cv2.inpaint(sub_bgr, erase_u8, 4, cv2.INPAINT_TELEA)
                    sub_bgr[erase_bool] = full_inp[erase_bool]

            bg_after = img_bgr[y1:y2, x1:x2]
            bg_lum_after = float(np.median(cv2.cvtColor(bg_after, cv2.COLOR_BGR2GRAY)))
            if bg_lum_after < 155.0:
                fg_rgb = (252, 252, 252)
                stroke_rgb = (16, 16, 16)
            else:
                fg_rgb = (15, 15, 15)
                stroke_rgb = (250, 250, 250)

        # Only widen horizontally when the source cluster is a vertical JP column (`bh > bw * 1.15`)
        # AND no artwork was trimmed from the bounding box!
        box_was_trimmed = (safe_tx1, safe_ty1, safe_tx2, safe_ty2) != (x1, y1, x2, y2)
        if (not box_was_trimmed) and bh_c > int(bw_c * 1.15):
            extra_x = min(28, max(16, int(bw_c * 0.28)))
            tx1 = max(4, safe_tx1 - extra_x)
            tx2 = min(width - 4, safe_tx2 + extra_x)
        else:
            tx1, tx2 = safe_tx1, safe_tx2
        hb = (tx1, safe_ty1, tx2, safe_ty2)
        return [[hb[0], hb[1], hb[2], hb[3], fg_rgb, stroke_rgb, trans_text, hb]]

    rx1, ry1, rx2, ry2, mask_full, rect_fill = _refine_cv_bubble_mask_locally_cv2(
        img_bgr, cv_obj, x1, y1, x2, y2, other_cv_boxes=other_cv_boxes
    )
    bw, bh = rx2 - rx1, ry2 - ry1
    if bw < 16 or bh < 14:
        return []

    if is_pastel:
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

    # 1. Connected figure-8 double bubble (light OR dark) -> flow text cleanly across BOTH lobes!
    if len(trans_text) >= 28:
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

    if has_cv_bubbles:
        w_halo_px = (sub_g >= 225) & (sub_ch <= 30)
        d_any_px = (sub_g <= 140)
        if not np.any(w_halo_px) or not np.any(d_any_px):
            return None
        w_near_d = cv2.dilate(w_halo_px.astype(np.uint8), _CV2_K7, iterations=1) > 0
        d_cand = np.where(d_any_px & w_near_d, np.uint8(255), np.uint8(0))
        n_hc, h_lbls, h_st, _ = cv2.connectedComponentsWithStats(d_cand, connectivity=8)
        halo_glyphs = 0
        for hk in range(1, n_hc):
            hgw, hgh, hga = int(h_st[hk, cv2.CC_STAT_WIDTH]), int(h_st[hk, cv2.CC_STAT_HEIGHT]), int(h_st[hk, cv2.CC_STAT_AREA])
            if 3 <= hgw <= 44 and 3 <= hgh <= 46 and hga >= 8:
                hx1, hy1 = int(h_st[hk, cv2.CC_STAT_LEFT]), int(h_st[hk, cv2.CC_STAT_TOP])
                p_x1, p_y1 = max(0, hx1 - 4), max(0, hy1 - 4)
                p_x2, p_y2 = min(bw, hx1 + hgw + 4), min(bh, hy1 + hgh + 4)
                sc_m = (h_lbls[p_y1:p_y2, p_x1:p_x2] == hk).astype(np.uint8)
                hring = cv2.dilate(sc_m, _CV2_K7, iterations=1) - sc_m
                hrcnt = max(1, int(np.count_nonzero(hring)))
                if (int(np.count_nonzero((hring > 0) & w_halo_px[p_y1:p_y2, p_x1:p_x2])) / float(hrcnt)) >= 0.45:
                    halo_glyphs += 1
        if halo_glyphs < 4:
            return None

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

    bright_achrom = (sub_g >= max(208.0, p50 + 50.0)) & (sub_ch <= 60)
    dark_achrom = (sub_g <= min(110.0, p50 - 45.0)) & (sub_ch <= 68)
    bright_ratio = float(np.count_nonzero(bright_achrom)) / float(total_px)
    dark_ratio = float(np.count_nonzero(dark_achrom)) / float(total_px)

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
        dark_outline = (ring > 0) & in_box & (w_g < min(155.0, max(95.0, p50 + 25.0)))
        erase_u8 = cv2.dilate(np.where((ink_dil > 0) | dark_outline, np.uint8(255), np.uint8(0)), _CV2_K3, iterations=1)
        erase_u8 = cv2.morphologyEx(erase_u8, cv2.MORPH_CLOSE, _CV2_K5)
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
        bright_ratio = float(np.count_nonzero(sub_g >= 195)) / float(total_px)
        if bg_lum > 60 or bg_std > 28.0 or not (0.012 <= bright_ratio <= 0.40):
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
    if has_cv_bubbles and bw >= 28 and bh >= 24:
        mx_i, my_i = max(2, int(bw * 0.10)), max(2, int(bh * 0.10))
        inner_g = sub_g[my_i : bh - my_i, mx_i : bw - mx_i]
        inner_ink = np.where(inner_g < 145, np.uint8(255), np.uint8(0))
        in_cc, _, in_st, _ = cv2.connectedComponentsWithStats(inner_ink, connectivity=8)
        inner_gly = sum(
            1
            for k_i in range(1, in_cc)
            if (3 <= int(in_st[k_i, cv2.CC_STAT_WIDTH]) <= 44 and 3 <= int(in_st[k_i, cv2.CC_STAT_HEIGHT]) <= 46 and int(in_st[k_i, cv2.CC_STAT_AREA]) >= 8)
            or (2 <= int(in_st[k_i, cv2.CC_STAT_WIDTH]) <= 5 and 2 <= int(in_st[k_i, cv2.CC_STAT_HEIGHT]) <= 5 and 4 <= int(in_st[k_i, cv2.CC_STAT_AREA]) <= 16)
        )
        if inner_gly < 3:
            return None

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
    render_stats: dict | None = None,
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
    rendered_bubble_count = 0

    # Pass 1: Render all CV-anchored Set-of-Mark bubbles & UI banners/labels
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
        rendered_bubble_count += 1
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
                rendered_bubble_count += 1
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

        cleaned_open = _clean_unmarked_open_bubble_and_measure_box_cv2(
            img_bgr, base_gray, rx1, ry1, rx2, ry2, has_cv_bubbles=bool(cv_map)
        )
        if cleaned_open is None:
            continue
        lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned_open
        rendered_bubble_count += 1
        prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
        rendered_rects.append(hard_bounds)

    if isinstance(render_stats, dict):
        render_stats["rendered_bubbles"] = rendered_bubble_count
        render_stats["rendered_slots"] = len(prepared_renders)
        render_stats["total_bubbles"] = len(bubbles)

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
            box_h = max(12, ty2 - ty1)
            is_compact_label = (box_h <= 36 and box_w >= int(box_h * 1.4))
            usable_w = max(14, int(box_w * (0.94 if is_compact_label else 0.92)))
            usable_h = max(11, int(box_h * (0.94 if is_compact_label else 0.92)))
            is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

            if is_compact_label:
                start_font_size = min(max_font_px, max(16, int(min(box_w * 0.42, box_h * 0.64))))
            else:
                start_font_size = min(max_font_px, max(14, int(min(box_w * 0.42, box_h * 0.36))))

            chosen_font, chosen_lines, line_height, ref_top = _fit_text_binary_search(
                draw, trans_text, usable_w, usable_h, start_font_size, is_thai_text
            )

            total_text_h = line_height * len(chosen_lines)
            curr_y = ty1 + max(0, (box_h - total_text_h) // 2)

            for ln in chosen_lines:
                lw = _measure_text_width(draw, ln, chosen_font)
                lb_left = 0 if _font_needs_latin_fallback(chosen_font) else draw.textbbox((0, 0), ln, font=chosen_font)[0]
                curr_x = tx1 + max(0, (box_w - lw) // 2) - lb_left
                if is_compact_label:
                    s_width = 1
                elif fg_text == (252, 252, 252) or stroke_col == (250, 250, 250):
                    s_width = 3
                else:
                    s_width = 2
                _draw_text_line_safe(
                    draw,
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
    raw_bytes_or_path: bytes | str | os.PathLike,
    bubbles: list[dict],
    cv_bubbles: list[dict] | None = None,
    render_stats: dict | None = None,
) -> bytes | None:
    """
    Renders translated dialogue at FULL ORIGINAL RESOLUTION using vectorized OpenCV + NumPy
    masking when available (accepts either raw `bytes` or a `/tmp` disk path for low-RAM
    streaming), falling back lazily to slim pure-Pillow (`manga_pil_fallback.py`) otherwise.
    """
    raw_bytes = _coerce_image_bytes_sync(raw_bytes_or_path)
    if not raw_bytes or not bubbles:
        return None
    if HAS_CV2:
        return _render_translated_manga_image_cv2_sync(raw_bytes, bubbles, cv_bubbles, render_stats=render_stats)
    from manga_pil_fallback import _render_translated_manga_image_pil_sync
    res = _render_translated_manga_image_pil_sync(raw_bytes, bubbles, cv_bubbles)
    if isinstance(render_stats, dict) and res is not None:
        render_stats["rendered_bubbles"] = len(bubbles)
        render_stats["rendered_slots"] = len(bubbles)
        render_stats["total_bubbles"] = len(bubbles)
    return res