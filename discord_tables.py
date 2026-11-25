import re
import unicodedata

_CODE_FENCE_SPLIT_REGEX = re.compile(r"(```[\s\S]*?```)")
_MD_SEP_LINE_REGEX = re.compile(
    r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$"
)
_MD_LINK_REGEX = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HTML_BR_REGEX = re.compile(r"<br\s*/?>", re.IGNORECASE)
_THAI_CHAR_REGEX = re.compile(r"[\u0E00-\u0E7F]")
_THAI_LEADING_VOWELS = frozenset("เแโใไ")

# Zero-dependency Thai syllable tokenizer (matches manga_cv_render quality for clean cell wrapping)
_THAI_CLUSTER_REGEX = re.compile(
    r"[เแโใไ]?[ก-ฮ][\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]*(?:[ะาำ]|[\u0E47-\u0E4E])*|[A-Za-z0-9]+|[^\s]"
)
_THAI_VOWEL_SIGNALS = set("เแโใไะาำ\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47")
_THAI_CLUSTER_FIRST = set("กขคจตทบปผพฟสศษ")
_THAI_CLUSTER_SECOND = set("รลว")
_THAI_HO_NAM_FOLLOW = set("งญนมยรลว")
_THAI_PREFIX_LEAD = set("สปกตฉผถข")
_THAI_PUNCT_ATTACH = set("ๆฯ?!.,:;'\"')]-~")
_THAI_OPEN_ENDINGS = set("าำ\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47\u0E48\u0E49\u0E4A\u0E4Bอว")
_THAI_CAN_TAKE_CODA_VOWEL = set("เแโา\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E47อว")
_THAI_CODA_CONSONANTS = set("กขคฆงจชซฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬญ")


def _char_display_width(ch: str) -> int:
    """
    Returns monospace display width for a single Unicode character:
    - 0: combining marks (Thai above/below vowels & tone marks), format chars, variation selectors
    - 2: East Asian Wide/Fullwidth characters (Japanese Kanji/Kana, CJK, Hangul, wide emoji)
    - 1: ASCII, Thai base consonants & spacing vowels (including ำ U+0E33), Latin-1, box-drawing
    """
    if not ch:
        return 0
    cp = ord(ch)
    if cp in (0x200D, 0xFE0E, 0xFE0F):
        return 0
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me", "Cf"):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def _display_width(text: str) -> int:
    return sum(_char_display_width(ch) for ch in (text or ""))


def _split_grapheme_clusters(text: str) -> list[str]:
    clusters: list[str] = []
    for ch in text:
        if clusters and _char_display_width(ch) == 0:
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _tokenize_thai_syllables(word: str) -> list[str]:
    """Zero-dependency Thai syllable tokenizer so long Thai words never break mid-syllable."""
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
            is_o_nam = (
                tok == "อ"
                and (nxt.startswith(("ย่า", "ยู่")) or (nxt == "ยา" and (i + 2) < n and raw[i + 2] == "ก"))
            )
            if is_ho_nam or is_o_nam:
                merged_onsets.append(tok + nxt)
                i += 2
                continue
            if nxt[0] in _THAI_CLUSTER_SECOND:
                is_bare_c = len(tok) == 1 and tok in _THAI_CLUSTER_FIRST
                is_lv_c = len(tok) == 2 and tok[0] in _THAI_LEADING_VOWELS and tok[1] in _THAI_CLUSTER_FIRST
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
                if (
                    nxt in ("อะ", "อ", "าะ")
                    or ("\u0E35" in tok and nxt.startswith("ย"))
                    or ("\u0E37" in tok and nxt.startswith("อ"))
                ):
                    merged_vowels.append(tok + nxt)
                    i += 2
                    continue
            if nxt == "ว" and "\u0E31" in tok and any("\u0E01" <= c <= "\u0E2E" for c in tok):
                merged_vowels.append(tok + nxt)
                i += 2
                continue
            tok_other_v = any(c in _THAI_VOWEL_SIGNALS and c != "\u0E47" for c in tok)
            if (
                nxt == "อ"
                and not tok_other_v
                and any("\u0E01" <= c <= "\u0E2E" for c in tok)
            ) or (
                nxt == "ว"
                and not any(c in _THAI_VOWEL_SIGNALS for c in tok)
                and any("\u0E01" <= c <= "\u0E2E" for c in tok)
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
            if nxt and len(tok) == 1 and tok in _THAI_PREFIX_LEAD and any(c in _THAI_VOWEL_SIGNALS for c in nxt):
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

        is_bare_single = len(tok) == 1 and "\u0E01" <= tok <= "\u0E2E"
        prev = syllables[-1]
        prev_closed_vowel = prev.endswith(("ะ", "ำ")) or (prev[0] in ("ใ", "ไ"))
        prev_is_ia_vowel = prev.startswith("เ") and "\u0E35" in prev and prev.endswith("ย")
        prev_is_ua_vowel = "\u0E31" in prev and prev.endswith("ว")
        prev_is_ho_nam_bare = len(prev) == 2 and prev[0] == "ห" and prev[1] in _THAI_HO_NAM_FOLLOW
        if prev_is_ua_vowel:
            prev_open_for_coda = tok == "ย"
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
                            or (
                                prev[0] in ("เ", "แ", "โ")
                                and (len(prev) == 2 or (len(prev) == 3 and prev[2] in _THAI_CLUSTER_SECOND))
                            )
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
            if final_syls:
                final_syls[-1] += tok
                i += 1
                continue
        final_syls.append(tok)
        i += 1
    return final_syls


def _clean_cell_markdown(raw: str) -> str:
    """Strips Markdown inline styling that Discord does not render inside fenced code blocks."""
    s = (raw or "").strip()
    s = s.replace("\\|", "|")
    s = _HTML_BR_REGEX.sub("\n", s)
    s = _MD_LINK_REGEX.sub(r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _split_pipe_row(line: str) -> list[str]:
    """Splits a Markdown pipe table row while respecting escaped `\\|` inside cells."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]

    cells: list[str] = []
    cur: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n and s[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _parse_alignments(sep_line: str, num_cols: int) -> list[str]:
    parts = _split_pipe_row(sep_line)
    aligns: list[str] = []
    for idx in range(num_cols):
        tok = parts[idx].strip() if idx < len(parts) else "---"
        left = tok.startswith(":")
        right = tok.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _wrap_cell_text(text: str, max_col_width: int) -> list[str]:
    """
    Wraps cell text to <= `max_col_width` display columns, preferring whitespace breaks,
    then Thai syllable boundaries (never splitting `สูงสุด` into `สูงสุ`/`ด`), and finally
    grapheme clusters (never detaching Thai combining marks or stranding `เแโใไ`).
    """
    if not text:
        return [""]
    limit = max(4, int(max_col_width))
    out_lines: list[str] = []

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if _display_width(para) <= limit:
            out_lines.append(para)
            continue

        words = para.split(" ")
        cur = ""
        cur_w = 0
        for word in words:
            if not word:
                continue
            ww = _display_width(word)
            sep_w = 1 if cur else 0
            if cur_w + sep_w + ww <= limit:
                cur = f"{cur} {word}" if cur else word
                cur_w += sep_w + ww
                continue

            if cur:
                out_lines.append(cur)
                cur = ""
                cur_w = 0

            if ww <= limit:
                cur = word
                cur_w = ww
                continue

            # Split oversized Thai words along syllable boundaries first
            tokens = (
                _tokenize_thai_syllables(word)
                if _THAI_CHAR_REGEX.search(word)
                else _split_grapheme_clusters(word)
            )
            chunk = ""
            chunk_w = 0
            for tok in tokens:
                tw = _display_width(tok)
                if tw > limit:
                    if chunk:
                        out_lines.append(chunk)
                        chunk = ""
                        chunk_w = 0
                    for cl in _split_grapheme_clusters(tok):
                        cw = _display_width(cl)
                        if chunk and (chunk_w + cw > limit):
                            if len(chunk) >= 2 and chunk[-1] in _THAI_LEADING_VOWELS:
                                carry = chunk[-1]
                                out_lines.append(chunk[:-1])
                                chunk = carry + cl
                                chunk_w = _display_width(chunk)
                            else:
                                out_lines.append(chunk)
                                chunk = cl
                                chunk_w = cw
                        else:
                            chunk += cl
                            chunk_w += cw
                    continue

                if chunk and (chunk_w + tw > limit):
                    out_lines.append(chunk)
                    chunk = tok
                    chunk_w = tw
                else:
                    chunk += tok
                    chunk_w += tw

            if chunk:
                cur = chunk
                cur_w = chunk_w

        if cur:
            out_lines.append(cur)

    return out_lines or [""]


def _pad_cell(line: str, width: int, align: str = "left") -> str:
    vis_w = _display_width(line)
    pad = max(0, width - vis_w)
    if align == "center":
        left_p = pad // 2
        right_p = pad - left_p
        return (" " * left_p) + line + (" " * right_p)
    if align == "right":
        return (" " * pad) + line
    return line + (" " * pad)


def _compute_column_widths(
    headers: list[str], rows: list[list[str]], max_table_width: int = 74
) -> list[int] | None:
    """
    Allocates per-column display widths within `max_table_width` using clean inner separators
    (`" │ "`, border overhead = `3 * (N - 1)`). Returns `None` for 6+ columns or unwrappable
    wide tables so the renderer switches to structured Discord Markdown cards.
    """
    num_cols = len(headers)
    if num_cols <= 0 or num_cols >= 6:
        return None

    border_overhead = 3 * (num_cols - 1)  # inner " │ " between columns (open outer edges!)
    avail = max_table_width - border_overhead
    if avail < num_cols * 5:
        return None

    natural_widths: list[int] = []
    body_max_widths: list[int] = []
    for c in range(num_cols):
        h_w = max((_display_width(ln) for ln in (headers[c] or "").split("\n")), default=1)
        b_w = max(
            (
                max((_display_width(ln) for ln in (r[c] or "").split("\n")), default=0)
                for r in rows
            ),
            default=0,
        )
        natural_widths.append(max(2, h_w, b_w))
        body_max_widths.append(max(2, b_w))

    if sum(natural_widths) <= avail:
        return natural_widths

    widths = list(natural_widths)
    min_floors: list[int] = []
    for c in range(num_cols):
        if body_max_widths[c] <= 14:
            min_floors.append(max(4, min(widths[c], max(body_max_widths[c], min(8, natural_widths[c])))))
        else:
            min_floors.append(max(14, min(20, widths[c])))

    if sum(min_floors) > avail:
        avail_expanded = 84 - border_overhead
        if sum(min_floors) > avail_expanded:
            return None
        avail = avail_expanded

    while sum(widths) > avail:
        shrinkable = [c for c in range(num_cols) if widths[c] > min_floors[c]]
        if not shrinkable:
            break
        widest = max(shrinkable, key=lambda c: widths[c] - min_floors[c])
        widths[widest] -= 1

    return widths


def _render_cards_fallback(headers: list[str], raw_rows: list[list[str]]) -> str:
    """Clean Discord Markdown card layout for ultra-wide (6+ column) tables."""
    blocks: list[str] = []
    clean_headers = [_clean_cell_markdown(h) or f"Col {i + 1}" for i, h in enumerate(headers)]
    for idx, row in enumerate(raw_rows, 1):
        title_val = (row[0] or "").strip() if row else f"รายการที่ {idx}"
        title_clean = _clean_cell_markdown(title_val) or f"รายการที่ {idx}"
        lines = [f"**{idx}. {title_clean}**"]
        for c in range(1, len(clean_headers)):
            val = (row[c] if c < len(row) else "").strip()
            if val:
                lines.append(f"- **{clean_headers[c]}:** {val}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_discord_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
    max_width: int = 74,
) -> str:
    """
    Renders a table cleanly for Discord chat:
    - 2-5 columns fitting within `max_width` -> clean open-edge monospace table inside ```text```
      (`─┼─` header & row dividers, `│` column separators, no crooked right-side wall on proportional
      Thai font fallbacks), with Thai syllable-aware wrapping and 0-width combining mark padding.
    - 6+ columns or unwrappable wide tables -> clean structured Discord Markdown cards.
    """
    if not headers or not rows:
        return ""

    num_cols = len(headers)
    norm_raw_rows = [
        [(r[c] if c < len(r) else "") for c in range(num_cols)]
        for r in rows
    ]
    clean_headers = [_clean_cell_markdown(h) for h in headers]
    clean_rows = [[_clean_cell_markdown(cell) for cell in r] for r in norm_raw_rows]
    aligns = alignments if (alignments and len(alignments) == num_cols) else ["left"] * num_cols

    col_widths = _compute_column_widths(clean_headers, clean_rows, max_table_width=max_width)
    if col_widths is None:
        return _render_cards_fallback(headers, norm_raw_rows)

    def _divider(bar: str = "─", cross: str = "┼") -> str:
        segs = []
        for idx_c, w in enumerate(col_widths):
            if idx_c == 0:
                segs.append(bar * (w + 1))
            elif idx_c == num_cols - 1:
                segs.append(bar * (w + 1))
            else:
                segs.append(bar * (w + 2))
        return cross.join(segs)

    def _render_row_lines(cells: list[str], row_aligns: list[str]) -> list[str]:
        wrapped_cols = [_wrap_cell_text(cells[c], col_widths[c]) for c in range(num_cols)]
        max_h = max(len(col_lns) for col_lns in wrapped_cols)
        rendered: list[str] = []
        for line_i in range(max_h):
            seg = []
            for c in range(num_cols):
                txt = wrapped_cols[c][line_i] if line_i < len(wrapped_cols[c]) else ""
                # Leave rightmost column unpadded on the right to avoid trailing whitespace
                if c == num_cols - 1 and row_aligns[c] == "left":
                    seg.append(txt)
                else:
                    seg.append(_pad_cell(txt, col_widths[c], row_aligns[c]))
            rendered.append(" │ ".join(seg).rstrip())
        return rendered

    header_sep = _divider("━", "┿")
    row_sep = _divider("─", "┼")

    header_lines = _render_row_lines(clean_headers, ["center"] * num_cols)
    rendered_body_groups = [_render_row_lines(r, aligns) for r in clean_rows]
    any_multiline = any(len(g) > 1 for g in rendered_body_groups)

    def _assemble(use_row_dividers: bool) -> str:
        out = []
        out.extend(header_lines)
        out.append(header_sep)
        for idx_g, g in enumerate(rendered_body_groups):
            out.extend(g)
            if use_row_dividers and idx_g < len(rendered_body_groups) - 1:
                out.append(row_sep)
        return "```text\n" + "\n".join(out) + "\n```"

    result = _assemble(use_row_dividers=any_multiline)
    if any_multiline and len(result) > 1600:
        result = _assemble(use_row_dividers=False)
    return result


def _convert_tables_in_prose_segment(segment: str, max_width: int = 74) -> str:
    if "|" not in segment:
        return segment

    lines = segment.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        if (
            i + 1 < n
            and "|" in lines[i]
            and _MD_SEP_LINE_REGEX.match(lines[i + 1])
        ):
            headers = _split_pipe_row(lines[i])
            if len(headers) >= 2:
                aligns = _parse_alignments(lines[i + 1], len(headers))
                j = i + 2
                raw_rows: list[list[str]] = []
                while j < n:
                    cand = lines[j]
                    if not cand.strip() or "|" not in cand:
                        break
                    if _MD_SEP_LINE_REGEX.match(cand):
                        break
                    row_cells = _split_pipe_row(cand)
                    if len(row_cells) < 2:
                        break
                    raw_rows.append(row_cells)
                    j += 1

                if raw_rows:
                    out.append(render_discord_table(headers, raw_rows, aligns, max_width=max_width))
                    i = j
                    continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def format_markdown_tables_for_discord(text: str, max_width: int = 74) -> str:
    """
    Converts any raw Markdown pipe tables (`| A | B |` + `| --- | --- |`) located OUTSIDE
    fenced code blocks into aligned Discord-safe monospace tables (or clean structured cards
    for 6+ column tables). Leaves existing code blocks and non-table prose 100% untouched.
    """
    if not text or "|" not in text:
        return text

    parts = _CODE_FENCE_SPLIT_REGEX.split(text)
    if len(parts) == 1:
        return _convert_tables_in_prose_segment(text, max_width=max_width)

    rebuilt: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            rebuilt.append(part)
        else:
            rebuilt.append(_convert_tables_in_prose_segment(part, max_width=max_width))
    return "".join(rebuilt)