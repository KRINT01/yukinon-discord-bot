import os
import re
import json
import time
import uuid
import shutil
import tempfile
import asyncio
from urllib.parse import quote_plus

import aiohttp
import discord

from config import (
    HTTPS,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_MANGA_MODEL,
    THAI_CHAR_REGEX,
    IMAGE_EXTENSIONS,
    manga_semaphore,
    state,
    clean_url,
    api_log,
    get_http_session,
    get_cached_channel_attachments,
    load_temp_image_cache,
    release_unused_memory,
)
from llm import (
    call_gemini_generate_content_self_healing,
    call_gemma_ai_studio_vision,
    call_cloudflare_vision_chat,
    call_cloudflare_vision_fallback_qwen,
    call_cloudflare_gemma_chat,
    call_groq_vision_chat,
    call_openrouter_multi_image_manga_batch,
    _fetch_google_ai_studio_model_names,
    get_google_model_cooldown,
    set_google_model_cooldown,
)
import manga_cv_render as _mcr_mod
from manga_cv_render import (
    HAS_CV2,
    _sanitize_text_for_manga_font,
    _detect_page_bubbles_and_build_som_b64_sync,
    render_translated_manga_image_sync,
    _release_cv_page_masks,
)

MANGA_TRANSLATOR_VERSION = "v17.9"
MANGA_CV_VER = getattr(_mcr_mod, "MANGA_CV_RENDER_VERSION", "old")
MAX_MANGA_BATCH_PAGES = 12  # Supports up to 12 pages across multi-part threads (split into <=10 files per Discord message)
# Bundle 2 dynamic parallel hedge: faster 15s recovery on typical 2-4p X threads, patient 25s on 5-12p batches
MANGA_BATCH_HEDGE_DELAY_SMALL_SEC = 15.0
MANGA_BATCH_HEDGE_DELAY_LARGE_SEC = 25.0

# Batch 3: Adaptive Stage 2 polish threshold & disk-backed /tmp streaming root
MAX_ADAPTIVE_SKIP_POLISH_BUBBLES = 4
MANGA_STREAM_TMP_ROOT = os.path.join(tempfile.gettempdir(), "yukinon_manga_jobs")
_MANGA_JOB_STALE_SEC = 1200.0  # 20-minute orphan cleanup guard

_FLASH_LITE_EXCLUDED = ("2.5", "2.0", "1.5", "latest", "image", "tts", "audio", "live", "embedding")
_HANGUL_REGEX = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]")
_SOURCE_EAST_ASIAN_REGEX = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")
_ASCII_LATIN_WORD_REGEX = re.compile(r"[A-Za-z]{2,}")
_COMMON_SAFE_THAI_LOAN_ABBREVS = frozenset({
    "ok", "dm", "hp", "mp", "xp", "tv", "pc", "ai", "sns", "id", "ng", "vr", "gps", "rpg", "npc", "bgm", "op", "ed",
})
_LEAKED_CONTAINER_ARTIFACT_REGEX = re.compile(
    r"^(?:\{\s*['\"][^'\"]+['\"]\s*:.*\}|\[\s*['\"\{].*\])$",
    re.DOTALL,
)

# Set-of-Mark (SoM) JSON responseSchema with explicit `propertyOrdering`:
# - Every numbered CV bubble `[k]` maps 1-to-1 via `"bubble_id": k` (zero coordinate drift!).
# - Unmarked borderless/open bubbles use `"bubble_id": null` + `"box_2d"` emitted AFTER `"source_text"`.
MANGA_BATCH_SOM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "pages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page_index": {"type": "INTEGER"},
                    "summary": {"type": "STRING"},
                    "bubbles": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "INTEGER"},
                                "bubble_id": {"type": "INTEGER", "nullable": True},
                                "should_translate": {"type": "BOOLEAN"},
                                "speaker_or_note": {"type": "STRING"},
                                "source_text": {"type": "STRING"},
                                "box_2d": {
                                    "type": "ARRAY",
                                    "items": {"type": "INTEGER"},
                                    "nullable": True,
                                },
                                "translated_text": {"type": "STRING"},
                            },
                            "propertyOrdering": [
                                "id",
                                "bubble_id",
                                "speaker_or_note",
                                "source_text",
                                "should_translate",
                                "box_2d",
                                "translated_text",
                            ],
                            "required": [
                                "id",
                                "bubble_id",
                                "should_translate",
                                "speaker_or_note",
                                "source_text",
                                "translated_text",
                            ],
                        },
                    },
                },
                "propertyOrdering": ["page_index", "summary", "bubbles"],
                "required": ["page_index", "summary", "bubbles"],
            },
        }
    },
    "propertyOrdering": ["pages"],
    "required": ["pages"],
}

MANGA_SINGLE_SOM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "bubbles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "bubble_id": {"type": "INTEGER", "nullable": True},
                    "should_translate": {"type": "BOOLEAN"},
                    "speaker_or_note": {"type": "STRING"},
                    "source_text": {"type": "STRING"},
                    "box_2d": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "nullable": True,
                    },
                    "translated_text": {"type": "STRING"},
                },
                "propertyOrdering": [
                    "id",
                    "bubble_id",
                    "speaker_or_note",
                    "source_text",
                    "should_translate",
                    "box_2d",
                    "translated_text",
                ],
                "required": [
                    "id",
                    "bubble_id",
                    "should_translate",
                    "speaker_or_note",
                    "source_text",
                    "translated_text",
                ],
            },
        },
    },
    "propertyOrdering": ["summary", "bubbles"],
    "required": ["summary", "bubbles"],
}


# ==============================================================================
# BATCH 3: DISK-BACKED /tmp PAGE STREAMING HELPERS (ZERO RAW-PAGE RAM RESIDENCY)
# ==============================================================================
def _create_manga_job_dir_sync(req_id: str) -> str:
    """Creates an isolated `/tmp` job directory and opportunistically prunes stale orphan jobs."""
    os.makedirs(MANGA_STREAM_TMP_ROOT, exist_ok=True)
    now = time.time()
    try:
        for entry in os.scandir(MANGA_STREAM_TMP_ROOT):
            if entry.is_dir():
                try:
                    if (now - entry.stat().st_mtime) > _MANGA_JOB_STALE_SEC:
                        shutil.rmtree(entry.path, ignore_errors=True)
                except OSError:
                    pass
    except OSError:
        pass

    safe_req = re.sub(r"[^A-Za-z0-9_-]+", "", req_id or "")[:24] or uuid.uuid4().hex[:8]
    job_dir = os.path.join(MANGA_STREAM_TMP_ROOT, f"{safe_req}_{uuid.uuid4().hex[:6]}")
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def _write_manga_page_to_job_dir_sync(job_dir: str, page_idx: int, raw_bytes: bytes) -> str:
    out_path = os.path.join(job_dir, f"in_p{page_idx:02d}.bin")
    with open(out_path, "wb") as f:
        f.write(raw_bytes)
    return out_path


def _spill_pages_to_job_dir_sync(
    pages_bytes: list[tuple[bytes, str]], req_id: str
) -> tuple[str, list[tuple[str, str]]]:
    """Spills in-memory `(raw_bytes, fname)` pages into an isolated `/tmp` job directory."""
    job_dir = _create_manga_job_dir_sync(req_id)
    disk_pages: list[tuple[str, str]] = []
    for idx, (raw_b, fname) in enumerate(pages_bytes, 1):
        p_path = _write_manga_page_to_job_dir_sync(job_dir, idx, raw_b)
        disk_pages.append((p_path, fname))
    return job_dir, disk_pages


def _read_page_bytes_sync(page_source: bytes | str) -> bytes:
    """Reads raw image bytes from either a `/tmp` file path (`str`) or an in-memory `bytes` buffer."""
    if isinstance(page_source, (bytes, bytearray)):
        return bytes(page_source)
    if isinstance(page_source, str) and page_source and os.path.exists(page_source):
        with open(page_source, "rb") as f:
            return f.read()
    return b""


def _detect_page_bubbles_from_source_sync(
    page_source: bytes | str, max_dim: int, quality: int
) -> tuple[str, list[dict]]:
    raw_b = _read_page_bytes_sync(page_source)
    if not raw_b:
        return "", []
    return _detect_page_bubbles_and_build_som_b64_sync(raw_b, max_dim, quality)


def _render_manga_page_to_disk_sync(
    page_source: bytes | str,
    bubbles: list[dict],
    page_cv: list[dict],
    out_jpg_path: str,
    render_stats: dict | None = None,
) -> bool:
    """
    Streams one manga page from `/tmp` (or raw bytes), renders translated bubbles,
    and writes the finished JPEG directly to `out_jpg_path` so rendered buffers
    never accumulate simultaneously in container RAM across 1-12 pages.
    """
    raw_b = _read_page_bytes_sync(page_source)
    if not raw_b:
        return False
    rendered_jpg = render_translated_manga_image_sync(
        raw_b, bubbles, page_cv, render_stats=render_stats
    )
    if not rendered_jpg:
        return False
    with open(out_jpg_path, "wb") as f:
        f.write(rendered_jpg)
    return True


def _cleanup_manga_job_dir_sync(job_dir: str) -> None:
    if job_dir and os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)


def _is_gemini_model_on_cooldown(model_id: str) -> bool:
    return time.time() < get_google_model_cooldown(model_id)


def _mark_gemini_model_cooldown(model_id: str, duration_sec: float) -> None:
    set_google_model_cooldown(model_id, duration_sec)


def _detect_target_translation_lang(user_prompt: str) -> tuple[str, str]:
    lower_p = user_prompt.lower()
    if any(k in lower_p for k in ("english", "eng", "อังกฤษ", "เป็นอิ้ง", "แปลอิ้ง", " to en")):
        return "en", "English"
    if any(k in lower_p for k in ("thai", "ไทย", "เป็นไทย", "แปลไทย", " to th")):
        return "th", "Thai"
    if THAI_CHAR_REGEX.search(user_prompt):
        return "th", "Thai"
    return "en", "English"


def _is_pure_english_or_latin_source(src_text: str) -> bool:
    """
    True when `src_text` contains English/Latin text (or numbers+Latin units like `170cm`,
    `Character Profile`, `Kairi Yui`, `HP 100`) and NO East Asian (JP/ZH/KR) or Thai characters.
    Used when translating to a non-English target (e.g. Thai) so original crisp English typography
    on character sheets/posters is preserved untouched on the image!
    """
    s = (src_text or "").strip()
    if not s:
        return False
    if _SOURCE_EAST_ASIAN_REGEX.search(s) or THAI_CHAR_REGEX.search(s):
        return False
    return bool(re.search(r"[A-Za-z]", s))


def _parse_flash_lite_version(model_id: str) -> tuple[int, int]:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?-flash-lite", (model_id or "").lower())
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


async def _discover_manga_batch_models() -> list[tuple[str, bool]]:
    """
    Returns the ordered model chain for multi-page `MangaBatch` using `llm.py`'s shared
    1-hour Google AI Studio catalog cache:
    1. `gemini-3.5-flash-lite`
    2. `gemini-3.1-flash-lite`
    3. Any strictly newer (>3.5) `flash-lite` model discovered via shared catalog
    4. `gemma-4-31b-it`
    5. `gemma-4-26b-a4b-it`
    Each entry is `(model_id, is_gemma)`.
    """
    newer_flash_lites: list[str] = []
    if GEMINI_API_KEY:
        all_models = await _fetch_google_ai_studio_model_names()
        found = []
        for name in all_models:
            lower_name = name.lower()
            if "flash-lite" in lower_name and not any(bad in lower_name for bad in _FLASH_LITE_EXCLUDED):
                ver = _parse_flash_lite_version(name)
                if ver > (3, 5):
                    found.append((ver, name))
        found.sort(reverse=True)
        newer_flash_lites = [n for _, n in found]

    chain: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for m_id in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"] + newer_flash_lites:
        if m_id and m_id not in seen:
            seen.add(m_id)
            chain.append((m_id, False))

    for g_id in ("gemma-4-31b-it", "gemma-4-26b-a4b-it"):
        if g_id not in seen:
            seen.add(g_id)
            chain.append((g_id, True))

    return chain


# ==============================================================================
# STAGE 2: JSON NORMALIZATION, WRONG-LANGUAGE AUTO-REPAIR & BATCH ORCHESTRATION
# ==============================================================================
def _extract_norm_box_2d(raw_box) -> list[float] | None:
    if raw_box is None:
        return None

    coords: list[float] | None = None
    if isinstance(raw_box, (list, tuple)):
        if len(raw_box) == 1 and isinstance(raw_box[0], (list, tuple)) and len(raw_box[0]) == 4:
            raw_box = raw_box[0]
        if len(raw_box) == 4:
            try:
                coords = [float(v) for v in raw_box]
            except Exception:
                coords = None
    elif isinstance(raw_box, dict):
        lower_d = {str(k).lower(): v for k, v in raw_box.items()}
        y1 = lower_d.get("ymin", lower_d.get("y_min", lower_d.get("top", lower_d.get("y1"))))
        x1 = lower_d.get("xmin", lower_d.get("x_min", lower_d.get("left", lower_d.get("x1"))))
        y2 = lower_d.get("ymax", lower_d.get("y_max", lower_d.get("bottom", lower_d.get("y2"))))
        x2 = lower_d.get("xmax", lower_d.get("x_max", lower_d.get("right", lower_d.get("x2"))))
        if None not in (y1, x1, y2, x2):
            try:
                coords = [float(y1), float(x1), float(y2), float(x2)]
            except Exception:
                coords = None
    elif isinstance(raw_box, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_box)
        if len(nums) == 4:
            try:
                coords = [float(v) for v in nums]
            except Exception:
                coords = None

    if not coords or len(coords) != 4:
        return None

    if all(0.0 <= v <= 1.0 for v in coords) and max(coords) > 0.0:
        coords = [v * 1000.0 for v in coords]

    ymin, xmin, ymax, xmax = coords
    if abs(ymax - ymin) < 2.0 or abs(xmax - xmin) < 2.0:
        return None

    return coords


def _is_valid_target_script(tr_text: str, src_text: str, target_lang_name: str) -> bool:
    """
    Universal script sanity check (zero ad-hoc typo regexes):
    Rejects empty outputs, leaked Python/JSON dict/list artifacts (e.g. `{'1': '...'}`),
    leaked Korean Hangul, or translations missing any target-language script character
    when `source_text` contained Japanese/East Asian text.
    """
    raw_tr = str(tr_text or "").strip()
    if not raw_tr or _HANGUL_REGEX.search(raw_tr):
        return False
    if _LEAKED_CONTAINER_ARTIFACT_REGEX.match(raw_tr):
        return False
    cleaned = _sanitize_text_for_manga_font(raw_tr)
    if not cleaned:
        return False
    if target_lang_name.lower() == "thai":
        if not THAI_CHAR_REGEX.search(cleaned) and _SOURCE_EAST_ASIAN_REGEX.search(src_text or ""):
            return False
    elif target_lang_name.lower() == "english":
        if not re.search(r"[A-Za-z]", cleaned) and _SOURCE_EAST_ASIAN_REGEX.search(src_text or ""):
            return False
    return True


def _should_skip_gemma_polish_for_simple_page(
    batch_results: list[tuple[list[dict], str]],
    target_lang_name: str,
) -> bool:
    """
    Adaptive Stage 2 Proofreading Gate:
    Auto-skips the extra Gemma 4 polish round (~6-12s) ONLY on simple 1-page strips
    (`<= MAX_ADAPTIVE_SKIP_POLISH_BUBBLES` bubbles) when every bubble and the scene summary
    already pass strict script, cleanliness, length-proportionality, and zero-leak checks.
    Multi-page batches (`> 1` page) and denser pages (`> 4` bubbles) ALWAYS run Gemma 4 polish.
    """
    if len(batch_results) != 1:
        return False
    bubbles, summary = batch_results[0]
    if not (1 <= len(bubbles) <= MAX_ADAPTIVE_SKIP_POLISH_BUBBLES):
        return False

    is_thai = target_lang_name.lower() == "thai"

    if summary:
        if not _is_valid_target_script(summary, "", target_lang_name):
            return False
        if _SOURCE_EAST_ASIAN_REGEX.search(summary):
            return False

    for b in bubbles:
        # If sanitizer had to strip emojis/Hangul/CJK/malformed tone marks, polish it properly
        if b.get("_raw_translated_text"):
            return False
        src_txt = str(b.get("source_text") or "").strip()
        tr_txt = str(b.get("translated_text") or "").strip()
        if len(tr_txt) < 2:
            return False
        if not _is_valid_target_script(tr_txt, src_txt, target_lang_name):
            return False
        if _SOURCE_EAST_ASIAN_REGEX.search(tr_txt):
            return False
        # Suspiciously truncated translation compared to a long Japanese source sentence
        if len(src_txt) >= 16 and len(tr_txt) < 5:
            return False
        if is_thai:
            # Flag mixed English/romanized syllables inside Thai unless verbatim in source_text or safe abbrev
            src_lower = src_txt.lower()
            for tok in _ASCII_LATIN_WORD_REGEX.findall(tr_txt):
                tl = tok.lower()
                if tl not in _COMMON_SAFE_THAI_LOAN_ABBREVS and tl not in src_lower:
                    return False

    return True


def _dedupe_hallucinated_page_bubbles(
    batch_results: list[tuple[list[dict], str]],
    cv_bubbles_per_page: list[list[dict]],
    req_id: str = "",
) -> list[tuple[list[dict], str]]:
    """
    Universal, zero-regex safeguard against LLM duplicate hallucinations on false-positive
    face/body/clothing badges (e.g., `#5 == #6` or `#8 == #9` on the same page):
    If two distinct `bubble_id`s on the same page carry the exact same non-trivial `source_text`
    (>= 4 non-punctuation characters), keeps ONLY the bubble with the stronger CV text-glyph
    confidence score (`glyph_count`, `rect_fill`, `solidity`) and drops the hallucinated copy.
    """
    total_dropped = 0
    cleaned_results: list[tuple[list[dict], str]] = []

    for p_idx, (bubbles, summary) in enumerate(batch_results):
        if len(bubbles) <= 1:
            cleaned_results.append((bubbles, summary))
            continue

        page_cv = cv_bubbles_per_page[p_idx] if p_idx < len(cv_bubbles_per_page) else []
        cv_by_id = {int(c["bubble_id"]): c for c in page_cv if c.get("bubble_id") is not None}

        def _cv_text_score(b_item: dict) -> float:
            bid = b_item.get("bubble_id")
            if bid is None or int(bid) not in cv_by_id:
                return 0.0
            c = cv_by_id[int(bid)]
            gly = float(c.get("glyph_count") or 0.0)
            rf = float(c.get("rect_fill") or 0.70)
            sol = float(c.get("solidity") or 0.90)
            return (gly * 2.0) + (rf * 10.0) + (sol * 10.0)

        # Collect exact short labels owned by real UI banners (`嬉`, `怒`, `通常`, `基本情報`, etc.)
        # so an adjacent non-banner false positive duplicating a 1-3 char banner label is always dropped!
        banner_src_keys: set[str] = set()
        for b_item in bubbles:
            bid = b_item.get("bubble_id")
            if bid is None:
                continue
            if (cv_by_id.get(int(bid)) or {}).get("is_ui_banner"):
                sk = re.sub(r"[\s\.…!！?？~〜～、。,，・\-―—\x27\x22「」『』()（）]+", "", b_item.get("source_text") or "")
                if sk:
                    banner_src_keys.add(sk)

        # Group by normalized `source_text` (ignore whitespace & common punctuation)
        groups: dict[str, list[int]] = {}
        drop_idx: set[int] = set()
        for idx_b, b_item in enumerate(bubbles):
            bid = b_item.get("bubble_id")
            if bid is None:
                continue
            cv_meta = cv_by_id.get(int(bid)) or {}
            # Never dedupe legitimate repeated UI banners/table headers on character sheets
            if cv_meta.get("is_ui_banner") or cv_meta.get("is_pastel_label"):
                continue
            src_key = re.sub(r"[\s\.…!！?？~〜～、。,，・\-―—\x27\x22「」『』()（）]+", "", b_item.get("source_text") or "")
            if src_key and src_key in banner_src_keys and cv_meta.get("is_outlined_text"):
                drop_idx.add(idx_b)
                continue
            if len(src_key) >= 4:
                groups.setdefault(src_key, []).append(idx_b)

        for src_key, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            # Keep the single highest-scoring CV bubble (tie-break: keep first)
            best_i = max(idxs, key=lambda ii: (_cv_text_score(bubbles[ii]), -ii))
            for ii in idxs:
                if ii != best_i:
                    drop_idx.add(ii)

        if drop_idx:
            total_dropped += len(drop_idx)
            kept = [b for ii, b in enumerate(bubbles) if ii not in drop_idx]
            cleaned_results.append((kept, summary))
        else:
            cleaned_results.append((bubbles, summary))

    if total_dropped:
        api_log(
            f"MangaDedupe [req={req_id}] -> dropped {total_dropped} hallucinated duplicate bubble(s) "
            "(kept highest CV text-glyph match)"
        )
    return cleaned_results


async def _recover_omitted_pages_async(
    pages_input: list[tuple[bytes | str, str]],
    batch_results: list[tuple[list[dict], str]],
    cv_bubbles_per_page: list[list[dict]],
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    max_pages_to_recover: int = 2,
) -> tuple[list[tuple[list[dict], str]], list[list[dict]]]:
    """
    Automatic safeguard for multi-page batches: if a page had >=2 CV bubbles and Gemini
    omitted >=50% of them OR skipped a high-confidence rectangular narration box (`rect_fill >= 0.90`),
    re-runs ONLY that single page (up to `max_pages_to_recover` pages, ~2-3s each) so white
    caption boxes (e.g. Page 1 narration) are never silently lost!
    """
    num_pages = len(pages_input)
    if num_pages <= 1:
        return batch_results, cv_bubbles_per_page

    suspect_indices: list[tuple[int, list[int]]] = []
    for p_idx in range(num_pages):
        page_cv = cv_bubbles_per_page[p_idx] if p_idx < len(cv_bubbles_per_page) else []
        if not page_cv:
            continue
        bubbles, _ = batch_results[p_idx]
        returned_bids = {
            int(b["bubble_id"]) for b in bubbles if b.get("bubble_id") is not None
        }
        # Exclude character-sheet UI banners / pastel labels / white-paper clusters so intentionally-preserved
        # English labels (e.g. `Character Profile`, `Kairi Yui`, `170cm`) never trigger unnecessary retries!
        missing_rect_boxes = [
            int(c["bubble_id"])
            for c in page_cv
            if (not c.get("is_pastel"))
            and (not c.get("is_dark"))
            and (not c.get("is_ui_banner"))
            and (not c.get("is_pastel_label"))
            and (not c.get("is_outlined_text"))
            and float(c.get("rect_fill") or 0.0) >= 0.90
            and int(c["bubble_id"]) not in returned_bids
        ]
        severe_drop = (len(page_cv) >= 2 and len(bubbles) < max(2, int(len(page_cv) * 0.55)))
        if missing_rect_boxes or severe_drop:
            missing_all = [int(c["bubble_id"]) for c in page_cv if int(c["bubble_id"]) not in returned_bids]
            suspect_indices.append((p_idx, missing_all))

    if not suspect_indices:
        return batch_results, cv_bubbles_per_page

    recovered_pages = 0
    for p_idx, missing_ids in suspect_indices[:max_pages_to_recover]:
        page_src, fname = pages_input[p_idx]
        old_bubbles, old_summary = batch_results[p_idx]
        api_log(
            f"MangaPageRecover [req={req_id}] -> Page {p_idx + 1}/{num_pages} ({fname}) "
            f"returned {len(old_bubbles)}/{len(cv_bubbles_per_page[p_idx])} bubbles (omitted={missing_ids}); re-translating single page"
        )
        try:
            new_bubbles, new_summary, new_cv = await extract_and_translate_manga_bubbles(
                page_src,
                fname,
                target_lang_name,
                series_context_hint=series_context_hint,
                req_id=f"{req_id}-p{p_idx + 1}",
            )
            if len(new_bubbles) > len(old_bubbles):
                _release_cv_page_masks(cv_bubbles_per_page[p_idx])
                cv_bubbles_per_page[p_idx] = new_cv
                batch_results[p_idx] = (new_bubbles, new_summary or old_summary)
                recovered_pages += 1
                api_log(
                    f"MangaPageRecover [req={req_id}] -> Page {p_idx + 1} RECOVERED "
                    f"({len(old_bubbles)} -> {len(new_bubbles)} bubbles)"
                )
            else:
                _release_cv_page_masks(new_cv)
        except Exception as e:
            api_log(f"MangaPageRecover [req={req_id}] Page {p_idx + 1} note ({type(e).__name__})")

    if recovered_pages:
        api_log(f"MangaPageRecover [req={req_id}] -> recovered {recovered_pages} page(s) cleanly")
    return batch_results, cv_bubbles_per_page


def _unwrap_polish_text_value(val, depth: int = 0) -> str:
    """
    Safely extracts a plain string from a Stage 2 polish JSON value without ever
    stringifying a nested Python `dict`/`list` into `"{'1': '...'}"`:
    - Plain `str` -> returned immediately (zero impact on standard flat responses).
    - Nested `dict` with known text keys (`tr`, `translated_text`, `text`, ...) -> unwraps that key.
    - Single-entry nested `dict` (e.g. `{"1": "อ๊ะ..."}`) or 1-element `list` -> unwraps recursively.
    - Ambiguous multi-key dicts/lists -> returns `""` so Stage 1's clean draft is preserved.
    """
    if depth > 3 or val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for key in ("tr", "translated_text", "translation", "text", "polished", "draft", "sum", "summary"):
            if key in val:
                unwrapped = _unwrap_polish_text_value(val[key], depth + 1)
                if unwrapped:
                    return unwrapped
        if len(val) == 1:
            only_val = next(iter(val.values()))
            return _unwrap_polish_text_value(only_val, depth + 1)
        return ""
    if isinstance(val, list):
        if len(val) == 1:
            return _unwrap_polish_text_value(val[0], depth + 1)
        return ""
    return ""


def _parse_polish_json_response(raw_out: str) -> tuple[dict[int, str], dict[int, str]]:
    """
    Parses Stage 2 Gemma 4 polish JSON (`{"s": {"1": "..."}, "b": {"0": "...", ...}}` or
    flat `{"0": "..."}` fallback). Safely unwraps nested single-entry or page-grouped dicts
    (e.g. `{"0": {"1": "อ๊ะ..."}}` or `{"1": {"0": "...", "1": "..."}}`) without ever leaking
    `str(dict)` artifacts. Returns `(summary_map, bubble_map)`.
    """
    if not raw_out:
        return {}, {}
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_out)
    cleaned = re.sub(r"<\|channel>thought[\s\S]*?<channel\|>", "", cleaned)
    cleaned = re.sub(r"<\|channel>thought\s*", "", cleaned).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    m_obj = re.search(r"\{[\s\S]*\}", cleaned)
    if not m_obj:
        return {}, {}
    target = m_obj.group(0)
    parsed = None
    for cand in (
        target,
        re.sub(r",\s*([}\]])", r"\1", target),
    ):
        try:
            parsed = json.loads(cand)
            break
        except Exception:
            continue
    if not isinstance(parsed, dict):
        return {}, {}

    summary_map: dict[int, str] = {}
    bubble_map: dict[int, str] = {}

    raw_s = parsed.get("s") or parsed.get("summaries")
    if isinstance(raw_s, dict):
        for k_str, val in raw_s.items():
            try:
                unwrapped_s = _unwrap_polish_text_value(val)
                if unwrapped_s:
                    summary_map[int(k_str)] = unwrapped_s
            except Exception:
                continue

    raw_b = parsed.get("b") or parsed.get("bubbles")
    if isinstance(raw_b, dict):
        for k_str, val in raw_b.items():
            try:
                if isinstance(val, str):
                    txt = val.strip()
                    if txt:
                        bubble_map[int(k_str)] = txt
                elif isinstance(val, dict):
                    # Handle page-grouped nested maps like `"1": {"0": "...", "1": "..."}`
                    if (
                        len(val) >= 2
                        and all(str(ik).strip().isdigit() for ik in val.keys())
                        and all(isinstance(iv, str) for iv in val.values())
                    ):
                        for ik, iv in val.items():
                            txt = iv.strip()
                            if txt:
                                bubble_map[int(ik)] = txt
                    else:
                        unwrapped_b = _unwrap_polish_text_value(val)
                        if unwrapped_b:
                            bubble_map[int(k_str)] = unwrapped_b
                else:
                    unwrapped_b = _unwrap_polish_text_value(val)
                    if unwrapped_b:
                        bubble_map[int(k_str)] = unwrapped_b
            except Exception:
                continue
    elif isinstance(raw_b, list):
        for idx_i, val in enumerate(raw_b):
            if isinstance(val, dict):
                try:
                    ki = int(val.get("i", val.get("idx", idx_i)))
                    txt = _unwrap_polish_text_value(val)
                    if txt:
                        bubble_map[ki] = txt
                except Exception:
                    continue
            elif isinstance(val, str):
                txt = val.strip()
                if txt:
                    bubble_map[idx_i] = txt
    else:
        # Flat `{"0": "...", "1": "..."}` format
        for k_str, val in parsed.items():
            if k_str in ("s", "summaries", "b", "bubbles"):
                continue
            try:
                unwrapped_b = _unwrap_polish_text_value(val)
                if unwrapped_b:
                    bubble_map[int(k_str)] = unwrapped_b
            except Exception:
                continue

    return summary_map, bubble_map


async def _call_gemma_ai_studio_polish_json(
    model_id: str,
    prompt_text: str,
    req_id: str,
    timeout_sec: float = 22.0,
) -> tuple[str, float]:
    """
    Calls a Google AI Studio Gemma 4 model (`gemma-4-31b-it` or `gemma-4-26b-a4b-it`) for
    Stage 2 text-only proofreading with automatic HTTP 400 fallback if `responseMimeType` is rejected.
    """
    if not GEMINI_API_KEY or _is_gemini_model_on_cooldown(model_id):
        return "", 0.0

    session = get_http_session()
    url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }

    for attempt in range(2):
        t0 = time.monotonic()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            elapsed = time.monotonic() - t0
            _mark_gemini_model_cooldown(model_id, 60.0)
            api_log(f"MangaPolish [req={req_id}] -> {model_id} FAILED ({type(e).__name__}, {elapsed:.2f}s)")
            return "", elapsed

        if status == 200:
            data = json.loads(body_text)
            state.active_gemma_model = model_id
            state.record_provider_call("gemma_ai_studio")
            r_parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            raw_out = "".join(
                p.get("text", "") for p in r_parts
                if p.get("text") and not p.get("thought", False)
            ).strip()
            return raw_out, elapsed

        if status == 400 and attempt == 0:
            changed = False
            if "thinkingConfig" in payload["generationConfig"]:
                payload["generationConfig"].pop("thinkingConfig", None)
                changed = True
            if "responseMimeType" in payload["generationConfig"]:
                payload["generationConfig"].pop("responseMimeType", None)
                changed = True
            if changed:
                api_log(f"MangaPolish [req={req_id}] -> {model_id} HTTP 400 ({elapsed:.2f}s); retrying without thinkingConfig/responseMimeType")
                continue

        if status == 429:
            _mark_gemini_model_cooldown(model_id, 60.0)
        elif status in (500, 503):
            _mark_gemini_model_cooldown(model_id, 180.0)
        elif status == 404:
            _mark_gemini_model_cooldown(model_id, 3600.0)
        api_log(f"MangaPolish [req={req_id}] -> {model_id} HTTP {status} ({elapsed:.2f}s)")
        return "", elapsed

    return "", 0.0


def _split_polish_pages_into_chunks(
    compact_pages: list[dict],
    max_bubbles_per_chunk: int = 22,
) -> list[list[dict]]:
    """
    Groups sequential manga pages into balanced chunks of <= `max_bubbles_per_chunk` bubbles
    so no trailing tiny chunk stalls a wave and every chunk stays <= ~750-850 Thai output tokens
    (~8-12s on Cloudflare Gemma 4, comfortably under `llm.py`'s 28s HTTP timeout).
    """
    total_b = sum(len(p.get("b") or []) for p in compact_pages)
    if total_b <= max_bubbles_per_chunk:
        return [compact_pages] if compact_pages else []

    # Target balanced chunk size so e.g. 47 bubbles splits ~[14, 15, 18] instead of [21, 22, 4]
    num_target = max(2, (total_b + max_bubbles_per_chunk - 1) // max_bubbles_per_chunk)
    soft_cap = max(12, min(max_bubbles_per_chunk, (total_b + num_target - 1) // num_target + 2))

    chunks: list[list[dict]] = []
    cur_chunk: list[dict] = []
    cur_cnt = 0

    for page_obj in compact_pages:
        p_bubbles = page_obj.get("b") or []
        p_cnt = len(p_bubbles)

        # Rare edge case: a single dense page with > max_bubbles_per_chunk bubbles
        if p_cnt > max_bubbles_per_chunk:
            if cur_chunk:
                chunks.append(cur_chunk)
                cur_chunk = []
                cur_cnt = 0
            for start_i in range(0, p_cnt, max_bubbles_per_chunk):
                sub_p = {
                    "p": page_obj["p"],
                    "sum": page_obj.get("sum", "") if start_i == 0 else "",
                    "b": p_bubbles[start_i : start_i + max_bubbles_per_chunk],
                }
                chunks.append([sub_p])
            continue

        cap_now = max_bubbles_per_chunk if len(chunks) >= (num_target - 1) else soft_cap
        if cur_chunk and (cur_cnt + p_cnt) > cap_now:
            chunks.append(cur_chunk)
            cur_chunk = [page_obj]
            cur_cnt = p_cnt
        else:
            cur_chunk.append(page_obj)
            cur_cnt += p_cnt

    if cur_chunk:
        chunks.append(cur_chunk)
    return chunks


async def _polish_single_chunk_gemma_async(
    chunk_idx: int,
    total_chunks: int,
    chunk_pages: list[dict],
    story_overview: str,
    target_lang_name: str,
    series_context_hint: str,
    req_id: str,
    ai_sem: asyncio.Semaphore,
    cf_sem: asyncio.Semaphore,
) -> tuple[dict[int, str], dict[int, str], str, float]:
    """
    Polishes one <=22-bubble chunk using:
    1) Google AI Studio `gemma-4-26b-a4b-it` (fast 4B-active MoE; saves Cloudflare Neurons,
       15s timeout, fails in ~0.27s on HTTP 503).
    2) Cloudflare Workers AI `@cf/google/gemma-4-26b-a4b-it` (~6-8s per chunk).
    3) Google AI Studio dense `gemma-4-31b-it` (35s timeout fallback if both MoEs fail).
    4) Text-only Gemini Flash-Lite fallback.
    """
    chunk_bubble_cnt = sum(len(p.get("b") or []) for p in chunk_pages)
    if chunk_bubble_cnt == 0:
        return {}, {}, "", 0.0

    chunk_tag = f"{req_id}-c{chunk_idx}/{total_chunks}" if total_chunks > 1 else req_id
    ctx_parts = []
    if series_context_hint:
        ctx_parts.append(f"Series context: {series_context_hint[:450]}")
    if total_chunks > 1 and story_overview:
        ctx_parts.append(f"Full chapter scene flow (for cross-page continuity): {story_overview[:700]}")
    ctx_block = ("\n".join(ctx_parts) + "\n\n") if ctx_parts else ""

    if target_lang_name.lower() == "thai":
        lang_guidance = (
            "- แปลและเกลาทุกช่องให้เป็นภาษาไทยธรรมชาติ ลื่นไหล สะกดถูกต้อง 100% (ไม่มีคำสะกดผิด ไม่มีสระ/วรรณยุกต์สลับตำแหน่ง)\n"
            "- ใช้เฉพาะอักษรไทยและเครื่องหมาย ASCII มาตรฐาน (`...`, `~`, `-`, `!`, `?`, `\"`, `'`, `(`, `)`) — แปลงวงเล็บญี่ปุ่น `「」『』` เป็น `\"...\"` เสมอ ห้ามใส่อีโมจิ สัญลักษณ์หัวใจ/โน้ตเพลง หรือภาษาเกาหลี/จีน/อาหรับ\n"
            "- หากในต้นฉบับ `src` มีคำภาษาอังกฤษ/ชื่อภาษาอังกฤษอยู่แล้ว (เช่น `Kairi Yui`, `Iwato Pharmaceutical`, `OK`, `HP`) ให้คงคำภาษาอังกฤษเหล่านั้นไว้ตามเดิมพร้อมเว้นวรรคให้สวยงาม ห้ามแปลหรือทับศัพท์คำอังกฤษที่มีอยู่ในต้นฉบับเป็นไทย\n"
            "- ศัพท์เฉพาะ/ศัพท์เทคนิคในเรื่อง ชื่อระบบ ชื่ออาวุธ และชื่อสกิลที่มาจากคำทับศัพท์คาตาคานะภาษาอังกฤษ (เช่น `レリック`/`レリクス` = `Relic`/`Relics`, `スート:ハート` = `Suit: Heart`, `ハートビート` = `Heartbeat`, `ブースト` = `Boost`, `レイピア` = `Rapier`, `マナアクセラレーション` = `Mana Acceleration`) ให้ใช้เป็นคำภาษาอังกฤษที่สม่ำเสมอเหมือนกันทุกแบนเนอร์ ป้ายหัวข้อ และย่อหน้าในทุกหน้า ห้ามปนกันระหว่างภาษาอังกฤษ (`Relic`) กับคำทับศัพท์ไทย (`รีลิกส์`) ส่วนเสียงอุทาน/คำพูดญี่ปุ่นทั่วไป (เช่น `えへへ` = `เอะเฮะ`, `…っ!!` = `...อ๊ะ!!`, `コイツ` = `ไอ้หมอนี่`) ให้แปลเป็นไทยล้วน\n"
            "- สำหรับหัวข้อแบนเนอร์ ป้ายตาราง หรือป้ายสีหน้าสั้นๆ (เช่น `基本情報`, `年齢`, `通常`, `嬉`, `怒`, `能力`, `必殺技`) ให้แปลเป็นภาษาไทยที่สั้น กระชับที่สุด (1-3 คำ) เพื่อไม่ให้ล้นกรอบป้าย\n"
            "- ใช้คำลงท้ายให้ตรงเพศและบุคลิกของตัวละคร (ตัวละครหญิง/แต่งหญิงใช้ `ค่ะ`/`คะ`/`นะคะ`/`นะ` ตัวเอกชายพูดกับเพื่อนสนิทใช้ภาษาเป็นกันเอง)\n"
            "- คงชื่อศิลปิน/ผู้วาดตามต้นฉบับทุกตัวอักษรโดยห้ามแปลหรือทับศัพท์เป็นไทย ส่วนชื่อตัวละครและศัพท์เฉพาะญี่ปุ่นให้คงตาม Series context / เสียงอ่านทางการหรือเสียงคานะญี่ปุ่นที่ถูกต้อง (เช่น `あおい` = `อาโออิ`, `せな` = `เซนะ`, `なつき` = `นัตสึกิ`) หรือ Romaji ให้สม่ำเสมอ ห้ามแปลชื่อเฉพาะตรงตัว\n"
            "- หากต้นฉบับ `src` มีอักษรญี่ปุ่น OCR เพี้ยนเล็กน้อย หรือ `draft` ขาดหายกลางประโยค ให้ดูบริบทช่องก่อนหน้าแล้วแปลให้ครบสมบูรณ์\n"
            "- เกลา `sum` (บริบทของฉาก) ของแต่ละหน้าในชุดนี้ให้สั้น กระชับ เป็นภาษาไทยสะอาดตาด้วย"
        )
    else:
        lang_guidance = (
            f"- Proofread and polish every bubble into natural, idiomatic, accurately spelled {target_lang_name} manga dialogue.\n"
            f"- Use ONLY pure {target_lang_name} script and standard ASCII punctuation (`...`, `~`, `-`, `!`, `?`, `\"`, `'`) — convert Japanese brackets `「」『』` to standard quotes `\"...\"`, with no emojis or heart/music symbols.\n"
            "- Keep section banners, table labels, and expression tags concise (1-3 words) so they fit cleanly inside compact label boxes.\n"
            "- Preserve Japanese honorifics directly attached with a hyphen (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).\n"
            "- If `src` has a minor OCR typo or `draft` is truncated/mistranslated, infer the true meaning from surrounding dialogue and complete it cleanly.\n"
            f"- Also polish each page's 1-sentence `sum` into clean {target_lang_name}."
        )

    system_rules = (
        f"You are a senior Manga Localization Proofreader ({target_lang_name}).\n"
        "Review EVERY bubble `i` (`src` = Japanese OCR, `draft` = initial translation) and EVERY page `sum` in the JSON input, "
        f"and return polished, natural, typo-free {target_lang_name}.\n"
        f"RULES:\n{lang_guidance}\n"
        "- Keep lines concise enough to fit cleanly inside manga speech bubbles and compact UI banners.\n"
        "- Return ONLY valid JSON with this exact compact flat-string schema (include EVERY bubble index `i` and page number `p` present in INPUT; values MUST be plain strings, NEVER nested objects):\n"
        '{"s": {"1": "polished summary"}, "b": {"0": "polished bubble 0"}}'
    )
    compact_json_input = json.dumps({"pages": chunk_pages}, ensure_ascii=False, separators=(",", ":"))
    user_payload = f"{ctx_block}INPUT:\n{compact_json_input}"
    combined_prompt = f"{system_rules}\n\n{user_payload}"

    min_accept = max(1, int(chunk_bubble_cnt * 0.5))

    # Tier 1: Google AI Studio `gemma-4-26b-a4b-it` (4B-active MoE — fast when healthy, saves Cloudflare Neurons,
    # fails in ~0.27s on HTTP 503; never stalls on dense 31B before trying Cloudflare!)
    async with ai_sem:
        raw_out, el = await _call_gemma_ai_studio_polish_json(
            "gemma-4-26b-a4b-it", combined_prompt, req_id=chunk_tag, timeout_sec=15.0
        )
    if raw_out:
        s_m, b_m = _parse_polish_json_response(raw_out)
        if len(b_m) >= min_accept:
            return s_m, b_m, "gemma-4-26b-a4b-it", el
        api_log(
            f"MangaPolish [req={chunk_tag}] -> gemma-4-26b-a4b-it incomplete "
            f"({len(b_m)}/{chunk_bubble_cnt}); trying Cloudflare Gemma fallback"
        )

    # Tier 2: Cloudflare Workers AI Gemma 4 (`@cf/google/gemma-4-26b-a4b-it`, ~6-8s per chunk!)
    # Passing an explicit `system` role activates `llm.py`'s anti-thinking directive!
    async with cf_sem:
        t_cf = time.monotonic()
        try:
            cf_out = await call_cloudflare_gemma_chat(
                [
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                max_tokens=4096,
                purpose=f"MangaPolish[{chunk_tag}]",
            )
            if cf_out:
                s_m, b_m = _parse_polish_json_response(cf_out)
                if len(b_m) >= min_accept:
                    return s_m, b_m, "cf-gemma-4-26b", time.monotonic() - t_cf
                api_log(
                    f"MangaPolish [req={chunk_tag}] -> cf-gemma-4-26b incomplete "
                    f"({len(b_m)}/{chunk_bubble_cnt}); trying AI Studio 31B fallback"
                )
        except Exception as e:
            api_log(f"MangaPolish [req={chunk_tag}] Cloudflare Gemma note ({type(e).__name__})")

    # Tier 3: Google AI Studio dense `gemma-4-31b-it` fallback (realistic 35s timeout for dense 31B)
    async with ai_sem:
        raw_out_31, el_31 = await _call_gemma_ai_studio_polish_json(
            "gemma-4-31b-it", combined_prompt, req_id=chunk_tag, timeout_sec=35.0
        )
    if raw_out_31:
        s_m, b_m = _parse_polish_json_response(raw_out_31)
        if len(b_m) >= min_accept:
            return s_m, b_m, "gemma-4-31b-it", el_31
        api_log(
            f"MangaPolish [req={chunk_tag}] -> gemma-4-31b-it incomplete "
            f"({len(b_m)}/{chunk_bubble_cnt}); trying Flash-Lite fallback"
        )

    # Tier 4: Text-only Gemini Flash-Lite fallback (only if all Gemma tiers unavailable)
    if GEMINI_API_KEY:
        t_fb = time.monotonic()
        fb_payload = {
            "contents": [{"parts": [{"text": combined_prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(
                fb_payload,
                timeout_sec=16,
                purpose=f"MangaPolishFB[{chunk_tag}]",
                max_models=2,
                max_wall_sec=22.0,
                preferred_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            )
            if data:
                r_parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                raw_out = "".join(
                    p.get("text", "") for p in r_parts
                    if p.get("text") and not p.get("thought", False)
                ).strip()
                s_m, b_m = _parse_polish_json_response(raw_out)
                if b_m:
                    return s_m, b_m, "gemini-flash-lite-fb", time.monotonic() - t_fb
        except Exception as e:
            api_log(f"MangaPolish [req={chunk_tag}] Flash-Lite fallback note ({type(e).__name__})")

    return {}, {}, "", 0.0


async def _polish_manga_translations_gemma_async(
    batch_results: list[tuple[list[dict], str]],
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    progress_cb=None,
) -> list[tuple[list[dict], str]]:
    """
    Adaptive Stage 2 Text-Only Proofreading & Localization Polish (Parallel-Chunked Gemma 4):
    - Auto-skips cleanly on tiny 1-page strips (`<= 4` bubbles) when every bubble already passes
      strict script, cleanliness, and zero-leak checks (saves ~6-12s on quick single-panel strips).
    - Always runs on multi-page batches (`> 1` page) and denser 1-page strips (`> 4` bubbles),
      splitting >22 bubbles into balanced concurrent chunks with cross-page `story_overview`.
    - Per-chunk priority: AI Studio Gemma 4 (saves CF Neurons) -> Cloudflare Gemma 4 -> Flash-Lite text-only.
    """
    if _should_skip_gemma_polish_for_simple_page(batch_results, target_lang_name):
        bubbles_p0, sum_p0 = batch_results[0]
        for b in bubbles_p0:
            b["translated_text"] = _sanitize_text_for_manga_font(b.get("translated_text") or "")
            b.pop("_raw_translated_text", None)
        clean_sum = _sanitize_text_for_manga_font(sum_p0 or "")
        api_log(
            f"MangaPolish [req={req_id}] ADAPTIVE_SKIP -> 1p ({len(bubbles_p0)} clean bubble(s)) "
            f"passed script/leak checks; skipped Gemma 4 pass ({target_lang_name})"
        )
        return [(bubbles_p0, clean_sum)]

    flat_bubbles: list[dict] = []
    compact_pages: list[dict] = []
    overview_lines: list[str] = []

    for p_idx, (bubbles, summary) in enumerate(batch_results, 1):
        p_items = []
        for b in bubbles:
            g_idx = len(flat_bubbles)
            flat_bubbles.append(b)
            draft_val = b.get("_raw_translated_text") or b.get("translated_text") or ""
            p_items.append({
                "i": g_idx,
                "spk": b.get("speaker_or_note") or "",
                "src": b.get("source_text") or "",
                "draft": draft_val,
            })
        if summary:
            overview_lines.append(f"P{p_idx}: {summary.strip()}")
        if p_items or summary:
            compact_pages.append({"p": p_idx, "sum": summary or "", "b": p_items})

    if not flat_bubbles:
        return batch_results

    num_pages = len(batch_results)
    chunks = _split_polish_pages_into_chunks(compact_pages, max_bubbles_per_chunk=22)
    total_chunks = len(chunks)
    story_overview = " | ".join(overview_lines)

    api_log(
        f"MangaPolish [req={req_id}] START -> polishing {len(flat_bubbles)} bubble(s) "
        f"across {num_pages} page(s) in {total_chunks} parallel chunk(s) via AI Studio Gemma 4 ({target_lang_name})"
    )
    if progress_cb is not None:
        try:
            await progress_cb("✨ **2/3** กำลังเกลาสำนวนคำแปลให้เนียนขึ้นนะคะ...")
        except Exception:
            pass

    wall_t0 = time.monotonic()
    # Cap concurrent AI Studio Gemma calls to 2 (respects free-tier burst RPM) and Cloudflare fallback to 4
    ai_sem = asyncio.Semaphore(2)
    cf_sem = asyncio.Semaphore(4)
    chunk_tasks = [
        _polish_single_chunk_gemma_async(
            c_idx,
            total_chunks,
            c_pages,
            story_overview,
            target_lang_name,
            series_context_hint,
            req_id,
            ai_sem,
            cf_sem,
        )
        for c_idx, c_pages in enumerate(chunks, 1)
    ]
    chunk_outputs = await asyncio.gather(*chunk_tasks, return_exceptions=True)

    summary_map: dict[int, str] = {}
    bubble_map: dict[int, str] = {}
    used_models: list[str] = []

    for out in chunk_outputs:
        if isinstance(out, Exception) or not isinstance(out, tuple):
            continue
        s_m, b_m, m_name, _ = out
        if s_m:
            summary_map.update(s_m)
        if b_m:
            bubble_map.update(b_m)
        if m_name and m_name not in used_models:
            used_models.append(m_name)

    wall_elapsed = time.monotonic() - wall_t0

    # Apply polished results with per-bubble safety validation
    updated_bubbles = 0
    still_broken: list[dict] = []

    for g_idx, b in enumerate(flat_bubbles):
        src_txt = b.get("source_text") or ""
        cand_raw = bubble_map.get(g_idx)
        if cand_raw and _is_valid_target_script(cand_raw, src_txt, target_lang_name):
            b["translated_text"] = _sanitize_text_for_manga_font(cand_raw)
            updated_bubbles += 1
        else:
            b["translated_text"] = _sanitize_text_for_manga_font(b.get("translated_text") or "")

        b.pop("_raw_translated_text", None)

        if not _is_valid_target_script(b.get("translated_text") or "", src_txt, target_lang_name):
            still_broken.append(b)

    # Emergency per-bubble Google NMT rescue ONLY if a bubble still failed basic script validation
    if still_broken:
        tl_code = "th" if target_lang_name.lower() == "thai" else "en"
        session = get_http_session()
        rescued = 0
        for b in still_broken:
            src_q = (b.get("source_text") or "").strip()
            if not src_q:
                continue
            try:
                gtrans_host = "translate." + "googleapis.com"
                gt_url = (
                    f"{HTTPS}{gtrans_host}/translate_a/single"
                    f"?client=gtx&sl=auto&tl={tl_code}&dt=t&q={quote_plus(src_q)}"
                )
                async with session.get(
                    gt_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=aiohttp.ClientTimeout(total=6)
                ) as resp:
                    if resp.status == 200:
                        gt_data = await resp.json(content_type=None)
                        segs = [seg[0] for seg in (gt_data[0] or []) if seg and seg[0]]
                        cand = _sanitize_text_for_manga_font("".join(segs))
                        if cand:
                            b["translated_text"] = cand
                            rescued += 1
            except Exception:
                continue
        if rescued:
            api_log(f"MangaPolish [req={req_id}] -> NMT emergency rescue fixed {rescued}/{len(still_broken)} bubble(s)")

    updated_summaries = 0
    polished_batch: list[tuple[list[dict], str]] = []
    for p_idx, (bubbles, summary) in enumerate(batch_results, 1):
        cand_sum = summary_map.get(p_idx)
        if cand_sum and _is_valid_target_script(cand_sum, "", target_lang_name):
            final_sum = _sanitize_text_for_manga_font(cand_sum)
            updated_summaries += 1
        else:
            final_sum = _sanitize_text_for_manga_font(summary or "")
        polished_batch.append((bubbles, final_sum))

    if used_models:
        api_log(
            f"MangaPolish [req={req_id}] DONE -> model={'+'.join(used_models)} "
            f"({wall_elapsed:.2f}s wall, {total_chunks} chunk(s), updated {updated_bubbles}/{len(flat_bubbles)} bubble(s), "
            f"{updated_summaries}/{num_pages} summaries)"
        )
    else:
        api_log(f"MangaPolish [req={req_id}] SKIPPED -> kept Stage 1 drafts ({len(flat_bubbles)} bubbles)")

    return polished_batch


def _normalize_and_validate_bubbles(
    raw_bubbles: list,
    target_lang_name: str = "",
) -> list[dict]:
    valid = []
    preserve_english_only = bool(target_lang_name and target_lang_name.lower() != "english")

    for idx, item in enumerate(raw_bubbles, 1):
        if not isinstance(item, dict):
            continue

        flag_val = item.get("should_translate")
        if flag_val is None:
            flag_val = item.get("is_dialogue")
        _tmp_src = str(
            item.get("source_text")
            or item.get("original_text")
            or item.get("japanese_text")
            or item.get("source")
            or ""
        ).strip()

        # When translating to a non-English target (e.g. Thai), keep standalone pure-English/Latin
        # banners, names, and unit labels (e.g. `Character Profile`, `Kairi Yui`, `170cm`) untouched on the image!
        if preserve_english_only and _is_pure_english_or_latin_source(_tmp_src):
            continue

        if flag_val is False:
            # Safety guard: only skip if there is truly no translated sentence OR explicitly marked SFX/blank/English!
            _tmp_spk = str(item.get("speaker_or_note") or item.get("speaker") or "").lower()
            _tmp_tr = str(item.get("translated_text") or item.get("translation") or "").strip()
            _is_explicit_non_speech = any(
                k in _tmp_spk
                for k in ("sfx", "sound", "blank", "empty", "background", "artwork", "sleeve", "english", "เอฟเฟกต์")
            )
            if _is_explicit_non_speech or not _tmp_tr or not _tmp_src:
                continue

        speaker = (
            item.get("speaker_or_note")
            or item.get("speaker")
            or item.get("character")
            or f"Bubble {idx}"
        )
        src_txt = _tmp_src
        tr_txt = (
            item.get("translated_text")
            or item.get("translation")
            or item.get("translated")
            or item.get("english_text")
            or item.get("target_text")
            or item.get("text")
            or ""
        )
        raw_tr_before_sanitize = str(tr_txt).strip()
        tr_txt = _sanitize_text_for_manga_font(raw_tr_before_sanitize)
        tr_txt = re.sub(
            r"-\s+(san|kun|chan|sama|sensei|senpai|dono|shi)\b",
            r"-\1",
            tr_txt,
            flags=re.IGNORECASE,
        )
        src_txt = re.sub(r"\s+", " ", str(src_txt)).strip()
        if not tr_txt and not (raw_tr_before_sanitize and src_txt):
            continue

        spk_lower = str(speaker).strip().lower()
        is_sfx = any(k in spk_lower for k in ("sfx", "sound effect", "onomatopoeia", "เอฟเฟกต์"))

        raw_bid = item.get("bubble_id")
        bubble_id = None
        if raw_bid is not None and not is_sfx:
            try:
                bid_int = int(raw_bid)
                if bid_int >= 1:
                    bubble_id = bid_int
            except Exception:
                bubble_id = None

        raw_box = (
            item.get("box_2d")
            or item.get("bbox")
            or item.get("bbox_2d")
            or item.get("bounding_box")
            or item.get("box")
        )
        norm_box = None if is_sfx else _extract_norm_box_2d(raw_box)

        entry = {
            "id": item.get("id", idx),
            "bubble_id": bubble_id,
            "speaker_or_note": str(speaker).strip(),
            "source_text": src_txt,
            "translated_text": tr_txt,
            "box_2d": norm_box,
        }
        if raw_tr_before_sanitize and raw_tr_before_sanitize != tr_txt:
            entry["_raw_translated_text"] = raw_tr_before_sanitize
        valid.append(entry)
    return valid


def _repair_truncated_manga_batch_json(raw_target: str) -> dict | None:
    if '"pages"' not in raw_target:
        return None
    last_close_brace = raw_target.rfind("}")
    while last_close_brace > 20:
        candidate = raw_target[: last_close_brace + 1]
        for suffix in ("]}", "]}\n]}", "]}"):
            try:
                obj = json.loads(candidate + suffix)
                if isinstance(obj, dict) and isinstance(obj.get("pages"), list) and obj["pages"]:
                    return obj
            except Exception:
                pass
        last_close_brace = raw_target.rfind("}", 0, last_close_brace)
    return None


def _parse_manga_pages_batch_json(
    raw_text: str,
    expected_pages: int,
    target_lang_name: str = "",
) -> list[tuple[list[dict], str]]:
    if not raw_text:
        return [([], "") for _ in range(expected_pages)]

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*", cleaned)
    target = json_match.group(0) if json_match else cleaned
    target = re.sub(r"```\s*$", "", target).strip()

    parsed = None
    for candidate_str in (
        target,
        re.sub(
            r",\s*([}\]])",
            r"\1",
            re.sub(r'(["\d\]}])\s*\n(\s*["{])', r"\1,\n\2", target),
        ),
    ):
        try:
            parsed = json.loads(candidate_str)
            break
        except Exception:
            continue

    if parsed is None:
        parsed = _repair_truncated_manga_batch_json(target)

    if isinstance(parsed, dict):
        raw_pages = parsed.get("pages")
        if isinstance(raw_pages, list) and raw_pages:
            results = [([], "") for _ in range(expected_pages)]
            for idx, p_obj in enumerate(raw_pages):
                if not isinstance(p_obj, dict):
                    continue
                p_idx = int(p_obj.get("page_index") or (idx + 1)) - 1
                if not (0 <= p_idx < expected_pages):
                    p_idx = min(idx, expected_pages - 1)
                bubbles = _normalize_and_validate_bubbles(
                    p_obj.get("bubbles") or [], target_lang_name=target_lang_name
                )
                summary = _sanitize_text_for_manga_font(str(p_obj.get("summary") or "").strip())
                results[p_idx] = (bubbles, summary)
            if any(b or s for b, s in results):
                return results

        if expected_pages == 1 and isinstance(parsed, dict) and "bubbles" in parsed:
            bubbles = _normalize_and_validate_bubbles(
                parsed.get("bubbles") or [], target_lang_name=target_lang_name
            )
            summary = _sanitize_text_for_manga_font(str(parsed.get("summary") or "").strip())
            return [(bubbles, summary)]

    if expected_pages == 1:
        b, s = _parse_manga_bubbles_json(raw_text, target_lang_name=target_lang_name)
        return [(b, s)]

    return [([], "") for _ in range(expected_pages)]


def _parse_manga_bubbles_json(
    raw_text: str,
    target_lang_name: str = "",
) -> tuple[list[dict], str]:
    if not raw_text:
        return [], ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    target = json_match.group(0) if json_match else cleaned

    try:
        parsed = json.loads(target)
        bubbles = _normalize_and_validate_bubbles(
            parsed.get("bubbles") or [], target_lang_name=target_lang_name
        )
        summary = _sanitize_text_for_manga_font(str(parsed.get("summary") or "").strip())
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    try:
        repaired = target
        repaired = re.sub(r'(["\d\]}])\s*\n(\s*["{])', r"\1,\n\2", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        parsed = json.loads(repaired)
        bubbles = _normalize_and_validate_bubbles(
            parsed.get("bubbles") or [], target_lang_name=target_lang_name
        )
        summary = _sanitize_text_for_manga_font(str(parsed.get("summary") or "").strip())
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    summary_m = re.search(r'"summary"\s*:\s*"([^"\n]+)"', target)
    summary = _sanitize_text_for_manga_font(summary_m.group(1).strip()) if summary_m else ""

    extracted_bubbles = []
    for idx, obj_m in enumerate(re.finditer(r"\{[^{}]*?\}", target), 1):
        block = obj_m.group(0)
        tr_m = re.search(
            r'"(?:translated_text|translation|translated|text)"\s*:\s*"(.+?)"\s*(?:,\s*"|\n|\})',
            block,
            re.DOTALL,
        )
        if not tr_m:
            continue
        src_m = re.search(
            r'"(?:source_text|original_text|source)"\s*:\s*"(.+?)"\s*(?:,\s*"|\n|\})',
            block,
            re.DOTALL,
        )
        spk_m = re.search(
            r'"(?:speaker_or_note|speaker|character)"\s*:\s*"(.+?)"\s*(?:,\s*"|\n|\})',
            block,
            re.DOTALL,
        )
        bid_m = re.search(r'"bubble_id"\s*:\s*(\d+)', block)
        box_m = re.search(r'"(?:box_2d|bbox|bbox_2d|bounding_box|box)"\s*:\s*\[([^\]]+)\]', block)
        norm_box = _extract_norm_box_2d(box_m.group(1)) if box_m else None

        raw_tr_val = tr_m.group(1).replace('\\"', '"').strip()
        extracted_bubbles.append({
            "id": idx,
            "bubble_id": int(bid_m.group(1)) if bid_m else None,
            "should_translate": True,
            "speaker_or_note": spk_m.group(1).strip() if spk_m else f"Bubble {idx}",
            "source_text": src_m.group(1).strip() if src_m else "",
            "translated_text": raw_tr_val,
            "box_2d": norm_box,
        })

    valid_extracted = _normalize_and_validate_bubbles(
        extracted_bubbles, target_lang_name=target_lang_name
    )
    return valid_extracted, summary


async def _waiting_heartbeat_logger(
    req_id: str,
    label: str,
    model_id: str,
    interval_sec: float = 15.0,
    on_tick=None,
):
    start_t = time.monotonic()
    try:
        while True:
            await asyncio.sleep(interval_sec)
            elapsed = time.monotonic() - start_t
            api_log(f"{label} [req={req_id}] WAITING -> model={model_id} still running ({elapsed:.1f}s elapsed...)")
            if on_tick is not None:
                try:
                    await on_tick(model_id, elapsed)
                except Exception:
                    pass
    except asyncio.CancelledError:
        return


async def _invoke_manga_batch_single_model(
    model_id: str,
    is_gemma: bool,
    parts: list[dict],
    req_id: str,
    num_pages: int,
    batch_dim: int,
    per_call_timeout: float,
    wall_start: float,
    max_wall_budget: float,
    target_lang_name: str = "",
    total_cv: int = 0,
    notify_fn=None,
) -> tuple[list[tuple[list[dict], str]] | None, list[tuple[list[dict], str]] | None]:
    """
    Executes one model from the `MangaBatch` chain with shared Gemini cooldown tracking,
    provider/cache telemetry recording, and throttled live Discord progress heartbeats.
    Returns `(anchored_results, text_only_fallback)`.
    """
    session = get_http_session()
    label_prefix = f"MangaBatch({num_pages}p)"
    url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    gen_cfg = {
        "temperature": 0.0,
        "maxOutputTokens": 24576 if is_gemma else 32768,
        "responseMimeType": "application/json",
    }
    if not is_gemma:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}
        gen_cfg["responseSchema"] = MANGA_BATCH_SOM_SCHEMA

    payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
    max_attempts = 2
    text_only_fb = None

    for attempt in range(max_attempts):
        rem_now = max_wall_budget - (time.monotonic() - wall_start)
        if rem_now <= 8.0:
            break
        req_t = min(per_call_timeout, rem_now)
        t0 = time.monotonic()

        api_log(
            f"{label_prefix} [req={req_id}] REQUEST -> model={model_id} "
            f"(attempt {attempt + 1}/{max_attempts}, timeout={req_t:.0f}s, dim={batch_dim}px)"
        )

        async def _on_hb_tick(m_id: str, el: float):
            if notify_fn is not None:
                await notify_fn(
                    f"🌐 **2/3** กำลังแปล `{total_cv}` ช่องคำพูดอยู่ค่ะ (`{el:.0f}s`)..."
                )

        hb_task = asyncio.create_task(
            _waiting_heartbeat_logger(
                req_id, label_prefix, model_id, interval_sec=15.0, on_tick=_on_hb_tick
            )
        )
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=req_t)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0
        except asyncio.CancelledError:
            hb_task.cancel()
            raise
        except Exception as e:
            hb_task.cancel()
            elapsed = time.monotonic() - t0
            _mark_gemini_model_cooldown(model_id, 60.0)
            api_log(
                f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} "
                f"FAILED ({type(e).__name__}, {elapsed:.2f}s); cooldown 60s"
            )
            break
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except Exception:
                pass

        if status == 200:
            data = json.loads(body_text)
            state.record_provider_call("gemma_ai_studio" if is_gemma else "gemini_flash")
            usage = data.get("usageMetadata") if isinstance(data, dict) else None
            cache_note = ""
            if isinstance(usage, dict):
                cached_cnt = int(usage.get("cachedContentTokenCount") or 0)
                if cached_cnt > 0:
                    state.record_gemini_cache_hit(cached_cnt)
                    cache_note = f", cache_hit={cached_cnt}"

            resp_parts = (
                ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            )
            raw_json_str = "".join(
                p.get("text", "") for p in resp_parts
                if p.get("text") and not p.get("thought", False)
            ).strip()
            batch_results = _parse_manga_pages_batch_json(
                raw_json_str, num_pages, target_lang_name=target_lang_name
            )
            total_b = sum(len(b) for b, _ in batch_results)
            total_anchored = sum(
                1 for b_list, _ in batch_results for b in b_list
                if (b.get("bubble_id") is not None)
                or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
            )
            if total_b > 0 and total_anchored > 0:
                matched_som = sum(
                    1 for b_list, _ in batch_results for b in b_list
                    if b.get("bubble_id") is not None
                )
                per_page_llm = [len(b_list) for b_list, _ in batch_results]
                api_log(
                    f"{label_prefix}/{MANGA_TRANSLATOR_VERSION} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 OK "
                    f"({elapsed:.2f}s{cache_note}, {total_b} bubbles: {matched_som} SoM-anchored, {total_anchored - matched_som} box_2d, per_page={per_page_llm})"
                )
                return batch_results, None

            if total_b > 0 and total_anchored == 0:
                text_only_fb = batch_results
                api_log(
                    f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 "
                    f"but 0/{total_b} anchored bubbles ({elapsed:.2f}s); trying next model"
                )
                break

            api_log(
                f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 "
                f"empty JSON ({elapsed:.2f}s); trying next model"
            )
            break

        if status == 400 and attempt == 0:
            changed = False
            if "responseSchema" in payload["generationConfig"] and (
                "schema" in body_text.lower() or "propertyOrdering" in body_text or "responseSchema" in body_text
            ):
                payload["generationConfig"].pop("responseSchema", None)
                changed = True
            if "thinkingConfig" in payload["generationConfig"] and "thinkingConfig" in body_text:
                payload["generationConfig"].pop("thinkingConfig", None)
                changed = True
            if is_gemma and "responseMimeType" in payload["generationConfig"]:
                payload["generationConfig"].pop("responseMimeType", None)
                changed = True
            if changed:
                api_log(
                    f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 400 ({elapsed:.2f}s); retrying without schema/thinking"
                )
                continue

        if status == 503 and attempt == 0 and elapsed < 6.0 and (max_wall_budget - (time.monotonic() - wall_start)) > 25.0:
            api_log(
                f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} fast HTTP 503 ({elapsed:.2f}s); retrying once in 1.5s"
            )
            await asyncio.sleep(1.5)
            continue

        if status == 429:
            _mark_gemini_model_cooldown(model_id, 60.0)
        elif status in (500, 503):
            _mark_gemini_model_cooldown(model_id, 180.0)
        elif status == 404:
            _mark_gemini_model_cooldown(model_id, 3600.0)

        api_log(
            f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP {status} ({elapsed:.2f}s); trying next model"
        )
        break

    return None, text_only_fb


def _build_english_preservation_prompt_rule(target_lang_name: str) -> str:
    if target_lang_name.lower() == "english":
        return ""
    return (
        " - PRESERVE EXISTING ENGLISH TEXT: If a badge contains ONLY English/Latin text, numbers, or Latin units "
        "(e.g. `Character Profile`, `Kairi Yui`, `170cm`) with NO Japanese/East Asian text, set `\"should_translate\": false`, "
        "`\"source_text\": \"<exact English text>\"`, `\"translated_text\": \"\"` so the original crisp English typography stays untouched on the image! "
        "If a single badge mixes Japanese + English (e.g. `浬 結衣 Kairi Yui`), translate ONLY the Japanese part and keep the English part verbatim with a clean space.\n"
        " - CONSISTENT ENGLISH TECHNICAL TERMS: For in-world English katakana technical terms, system/class/weapon/skill/code names "
        "(e.g. `レリック`/`レリクス` -> `Relic`/`Relics`, `スート:ハート` -> `Suit: Heart`, `ハートビート` -> `Heartbeat`, `ブースト` -> `Boost`, `レイピア` -> `Rapier`, `マナアクセラレーション` -> `Mana Acceleration`), "
        "keep them consistently in English across EVERY section banner, label, and body paragraph on ALL pages — NEVER mix English (`Relic`) in one bubble with Thai transliteration (`รีลิกส์`) in another!\n"
    )


async def extract_and_translate_manga_pages_batch(
    pages_input: list[tuple[bytes | str, str]],
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    progress_cb=None,
) -> tuple[list[tuple[list[dict], str]], list[list[dict]], bool]:
    """
    Set-of-Mark (SoM) CV-Anchored Multi-Page Manga Batch Pipeline:
    1) Streams each page from `/tmp` (or raw `bytes`) 1-at-a-time to detect enclosed speech bubbles,
       colored UI section banners, pastel table row labels, and dark/light text blocks via vectorized
       OpenCV+NumPy and stamps high-contrast RED/WHITE/BLACK badges without keeping raw page buffers in RAM.
    2) Calls `MangaBatch` across:
       `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 `flash-lite`)
       -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
       using shared `llm.py` catalog/cooldowns AND dynamic parallel hedging.
    3) Returns `(batch_results, cv_bubbles_per_page, is_service_unavailable)`.
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return [], [], False

    req_id = req_id or f"mb-{uuid.uuid4().hex[:6]}"

    async def _notify(stage_text: str):
        if progress_cb is not None:
            try:
                await progress_cb(stage_text)
            except Exception:
                pass

    if num_pages == 1:
        b, s, cv_list = await extract_and_translate_manga_bubbles(
            pages_input[0][0],
            pages_input[0][1],
            target_lang_name,
            series_context_hint=series_context_hint,
            req_id=req_id,
            progress_cb=progress_cb,
        )
        return [(b, s)], [cv_list], False

    if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
        return [([], "") for _ in range(num_pages)], [[] for _ in range(num_pages)], True

    if num_pages >= 9:
        batch_dim, batch_q = 1024, 84
    elif num_pages >= 5:
        batch_dim, batch_q = 1120, 86
    else:
        batch_dim, batch_q = 1280, 90

    som_t0 = time.monotonic()
    b64_pages: list[tuple[str, str, int]] = []
    cv_bubbles_per_page: list[list[dict]] = []
    for page_src, fname in pages_input:
        b64_str, cv_bubbles = await asyncio.to_thread(
            _detect_page_bubbles_from_source_sync, page_src, batch_dim, batch_q
        )
        b64_pages.append((b64_str, fname, len(cv_bubbles)))
        cv_bubbles_per_page.append(cv_bubbles)

    total_cv = sum(len(c) for c in cv_bubbles_per_page)
    cv_elapsed = time.monotonic() - som_t0
    engine_tag = "cv2" if HAS_CV2 else "pil"
    per_page_cv_counts = [len(c) for c in cv_bubbles_per_page]
    api_log(
        f"MangaBatch({num_pages}p/{MANGA_TRANSLATOR_VERSION}) [req={req_id}] CV_DETECT ({engine_tag}/{MANGA_CV_VER}) -> "
        f"found {total_cv} enclosed bubble(s) across {num_pages} page(s) per_page={per_page_cv_counts} in {cv_elapsed:.2f}s"
    )
    await _notify(
        f"🌐 **2/3** เจอ `{total_cv}` ช่องคำพูด กำลังแปลอยู่นะคะ..."
    )

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster (use accurate names & terms):\n{series_context_hint[:1400]}\n"
        if series_context_hint else ""
    )
    en_preserve_rule = _build_english_preservation_prompt_rule(target_lang_name)

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga / character-profile page image(s) (labeled PAGE 1 to PAGE {num_pages}).
Every enclosed speech bubble, rectangular narration/caption box, colored section banner, and table label has ALREADY been outlined in RED with a high-contrast RED/WHITE/BLACK NUMBERED BADGE (`1`, `2`, `3`, ...) at its top-right rim!
{context_block}
Analyze all {num_pages} page(s) in sequence so character names, titles/ranks, pronouns, and ongoing story stay consistent across every page,
and translate EVERY SINGLE Japanese speech bubble, narration/caption box, section banner, table header, and expression label on EVERY page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (NUMBERED BADGE) & TRANSLATION RULES:
1. MANDATORY 1-TO-1 NUMBERED BADGE CHECKLIST (`bubble_id`):
 - Each page header lists the exact numbered badge IDs present on that page (`[1, 2, ..., K]`).
 - You MUST inspect and output an entry for EVERY SINGLE badge ID `1..K` on that page — NEVER skip a badge ID, NEVER merge two separate badges into one entry, and NEVER put text from a DIFFERENT bubble into another bubble's badge ID!
 - Double-check which exact RED/WHITE NUMBERED badge touches the bubble, banner, table label, or caption whose Japanese text you are reading before assigning `"bubble_id"`.
 - MANDATORY: Translate spoken speech bubbles, rectangular narration/caption boxes, AND EVERY badged Japanese section banner, table row header, and expression/detail label (e.g. `基本情報`, `年齢`, `身長`, `誕生日`, `所属`, `表情差分`, `通常`, `嬉`, `怒`, `哀`, `楽`, `照`, `能力`, `固有武器`, `必殺技`, `弱点・制約`, `武器・装備`, `デカい`)! Keep compact banner/table/expression labels concise (1-3 short words) so they fit cleanly inside the pill.
{en_preserve_rule} - CRITICAL ANTI-DUPLICATE RULE: If badge `k` outlines a character face, body, clothing, hand, background art, or pure SFX with NO dialogue/label text visually inside badge `k`, you MUST set `"should_translate": false`, `"source_text": ""`, `"translated_text": ""` — NEVER copy or duplicate text from a neighboring bubble onto an empty face/body badge!
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
 - ALSO scan every panel for real Japanese dialogue, narration, or profile text written WITHOUT a numbered badge around it.
 - For each unmarked Japanese text block, add a SEPARATE entry with `"bubble_id": null`, `"should_translate": true`, transcribe `"source_text"`, and provide `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY that Japanese text in its OWN panel — NEVER over faces, hands, or artwork!).
 - NEVER merge unbadged text from another panel into an earlier panel's badge!
 - NEVER output floating handwritten SFX (e.g. ドーン, トスッ, ポスン, ムチッ, ぐっ, カチャッ, ガチャ, ぎゅっ, にま, べた, もじっ) as `box_2d`.
3. Accurate Speaker Attribution, Names & Visual Gender:
 - Identify the actual character speaking each bubble visually (follow the tail; never copy one character's name onto every bubble).
 - Keep character names, titles/ranks, and faction names consistent across all pages using accurate Japanese phonetics.
 - Match pronouns and politeness particles strictly to each speaker's visual gender and setting: in Thai, visually female characters (including female subordinates, guards, and attendants) must use female/neutral endings (`ค่ะ`, `คะ`, `นะคะ`, `นะ`, `ล่ะ`, `สิ`) — NEVER use male `ครับ`/`ผม` unless the speaker is clearly drawn as male!
 - PURE TARGET LANGUAGE & FONT-SAFE OUTPUT: Write EVERY `translated_text` and `summary` purely in natural, correctly spelled {target_lang_name} using ONLY standard {target_lang_name} script and ASCII punctuation (`...`, `~`, `-`, `"`, `'`, `!`, `?`) — convert Japanese brackets `「」『』` into standard `"..."` quotes, with no emojis or heart/music symbols, and no mixed romanized syllables for Japanese interjections inside Thai (e.g. render `えへへ` as `เอะเฮะ`, `…っ!!` as `...อ๊ะ!!`, `コイツ` as `ไอ้หมอนี่`).
 - When translating to English, preserve Japanese honorifics attached directly with a hyphen and NO space (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).
4. Return ONLY valid JSON matching this exact structure for all {num_pages} page(s) (write `summary` in {target_lang_name}):
{{
  "pages": [
    {{
      "page_index": 1,
      "summary": "Brief 1-sentence scene context for Page 1 in {target_lang_name}",
      "bubbles": [
        {{
          "id": 1,
          "bubble_id": 1,
          "should_translate": true,
          "speaker_or_note": "Actual speaker name / Section Banner / Narration",
          "source_text": "Original Japanese text inside badge 1",
          "box_2d": null,
          "translated_text": "Natural {target_lang_name} translation"
        }}
      ]
    }}
  ]
}}
""".strip()

    fallback_text_only_results = None

    try:
        parts = [{"text": manga_batch_instruction}]
        for idx, (b64_img, fname, cv_cnt) in enumerate(b64_pages, 1):
            if cv_cnt > 0:
                badge_ids_str = ", ".join(str(i) for i in range(1, cv_cnt + 1))
                badge_note = f"NUMBERED BADGES ON THIS PAGE: [{badge_ids_str}] (account for EVERY badge ID strictly inside its own red outline; add bubble_id=null for any unbadged Japanese dialogue/text in other panels)"
            else:
                badge_note = "no numbered badges (use bubble_id=null + box_2d for any dialogue)"
            parts.append({"text": f"=== PAGE {idx} ({fname} | {badge_note}) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        full_chain = (await _discover_manga_batch_models()) if GEMINI_API_KEY else []
        active_chain = [m for m in full_chain if not _is_gemini_model_on_cooldown(m[0])]
        if not active_chain and full_chain:
            active_chain = full_chain

        per_call_timeout = min(85.0, 40.0 + (4.5 * num_pages))
        max_wall_budget = 180.0
        wall_start = time.monotonic()
        label_prefix = f"MangaBatch({num_pages}p)"
        hedge_delay_sec = (
            MANGA_BATCH_HEDGE_DELAY_SMALL_SEC
            if num_pages <= 4
            else MANGA_BATCH_HEDGE_DELAY_LARGE_SEC
        )

        start_idx = 0
        # Dynamic Parallel Hedge on primary `gemini-3.5-flash-lite` -> races `gemini-3.1-flash-lite`!
        if (
            len(active_chain) >= 2
            and active_chain[0][0] == "gemini-3.5-flash-lite"
            and not active_chain[1][1]
        ):
            m1_id, m1_gemma = active_chain[0]
            m2_id, m2_gemma = active_chain[1]
            t1 = asyncio.create_task(
                _invoke_manga_batch_single_model(
                    m1_id, m1_gemma, parts, req_id, num_pages, batch_dim,
                    per_call_timeout, wall_start, max_wall_budget,
                    target_lang_name=target_lang_name,
                    total_cv=total_cv, notify_fn=_notify,
                )
            )
            done, _ = await asyncio.wait({t1}, timeout=hedge_delay_sec)
            if t1 in done:
                try:
                    res1, fb1 = t1.result()
                except Exception:
                    res1, fb1 = None, None
                if res1 is not None:
                    return res1, cv_bubbles_per_page, False
                if fb1 is not None and fallback_text_only_results is None:
                    fallback_text_only_results = fb1
                start_idx = 1
            else:
                api_log(
                    f"{label_prefix} [req={req_id}] HEDGE -> model={m1_id} >{hedge_delay_sec:.0f}s; "
                    f"racing {m2_id} in parallel"
                )
                await _notify("🌐 **2/3** ตอบช้าไปหน่อย กำลังเร่งแปลให้อีกทางนะคะ...")
                t2 = asyncio.create_task(
                    _invoke_manga_batch_single_model(
                        m2_id, m2_gemma, parts, req_id, num_pages, batch_dim,
                        per_call_timeout, wall_start, max_wall_budget,
                        target_lang_name=target_lang_name,
                        total_cv=total_cv, notify_fn=_notify,
                    )
                )
                tasks = {t1, t2}
                try:
                    while tasks:
                        done_set, pending_set = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for finished in done_set:
                            tasks.discard(finished)
                            try:
                                res_h, fb_h = finished.result()
                            except Exception:
                                res_h, fb_h = None, None
                            if res_h is not None:
                                for p_task in pending_set:
                                    p_task.cancel()
                                if pending_set:
                                    await asyncio.gather(*pending_set, return_exceptions=True)
                                return res_h, cv_bubbles_per_page, False
                            if fb_h is not None and fallback_text_only_results is None:
                                fallback_text_only_results = fb_h
                finally:
                    for p_task in tasks:
                        if not p_task.done():
                            p_task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                start_idx = 2

        for offset_i, (model_id, is_gemma) in enumerate(active_chain[start_idx:]):
            if (max_wall_budget - (time.monotonic() - wall_start)) <= 10.0:
                break
            if start_idx > 0 or offset_i > 0:
                await _notify("🌐 **2/3** กำลังลองเปลี่ยนฝั่งแปลให้ใหม่นะคะ...")
            res_m, fb_m = await _invoke_manga_batch_single_model(
                model_id, is_gemma, parts, req_id, num_pages, batch_dim,
                per_call_timeout, wall_start, max_wall_budget,
                target_lang_name=target_lang_name,
                total_cv=total_cv, notify_fn=_notify,
            )
            if res_m is not None:
                return res_m, cv_bubbles_per_page, False
            if fb_m is not None and fallback_text_only_results is None:
                fallback_text_only_results = fb_m

        # Emergency Multi-Page Fallback via OpenRouter (`nvidia/nemotron-3.5-super-vl:free`, 1M ctx)
        # Fires ONLY when all Google AI Studio batch models fail/timeout (single multi-page batch, no per-page split!)
        if (
            OPENROUTER_API_KEY
            and OPENROUTER_MANGA_MODEL
            and time.time() >= getattr(state, "openrouter_manga_cooldown_until", 0.0)
        ):
            await _notify("🌐 **2/3** ฝั่ง Google ขัดข้องชั่วคราว กำลังสลับไปแปลผ่าน OpenRouter ให้แทนนะคะ...")
            or_timeout = min(85.0, max(45.0, per_call_timeout))
            or_max_tokens = min(16384, max(4096, 1600 * num_pages))
            api_log(
                f"{label_prefix} [req={req_id}] REQUEST -> OpenRouter model={OPENROUTER_MANGA_MODEL} "
                f"(timeout={int(or_timeout)}s, dim={batch_dim}px)"
            )
            or_t0 = time.monotonic()

            async def _on_or_hb_tick(m_id: str, el: float):
                await _notify(
                    f"🌐 **2/3** กำลังแปลผ่าน OpenRouter (`{el:.0f}s`)..."
                )

            hb_or = asyncio.create_task(
                _waiting_heartbeat_logger(
                    req_id,
                    label_prefix,
                    OPENROUTER_MANGA_MODEL,
                    interval_sec=15.0,
                    on_tick=_on_or_hb_tick,
                )
            )
            try:
                raw_or_json = await call_openrouter_multi_image_manga_batch(
                    b64_pages,
                    manga_batch_instruction,
                    mime_type=mime_type,
                    timeout_sec=or_timeout,
                    max_tokens=or_max_tokens,
                    purpose=f"{label_prefix} [req={req_id}]",
                )
            finally:
                hb_or.cancel()
                try:
                    await hb_or
                except Exception:
                    pass

            if raw_or_json:
                or_elapsed = time.monotonic() - or_t0
                or_results = _parse_manga_pages_batch_json(
                    raw_or_json, num_pages, target_lang_name=target_lang_name
                )
                total_b = sum(len(b) for b, _ in or_results)
                total_anchored = sum(
                    1 for b_list, _ in or_results for b in b_list
                    if (b.get("bubble_id") is not None)
                    or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                )
                if total_b > 0 and total_anchored > 0:
                    matched_som = sum(
                        1 for b_list, _ in or_results for b in b_list
                        if b.get("bubble_id") is not None
                    )
                    per_page_llm = [len(b_list) for b_list, _ in or_results]
                    api_log(
                        f"{label_prefix}/{MANGA_TRANSLATOR_VERSION} [req={req_id}] RESPONSE -> "
                        f"model={OPENROUTER_MANGA_MODEL} HTTP 200 OK ({or_elapsed:.2f}s, "
                        f"{total_b} bubbles: {matched_som} SoM-anchored, {total_anchored - matched_som} box_2d, per_page={per_page_llm})"
                    )
                    return or_results, cv_bubbles_per_page, False
                if total_b > 0 and fallback_text_only_results is None:
                    fallback_text_only_results = or_results
    finally:
        b64_pages.clear()

    if fallback_text_only_results is not None:
        return fallback_text_only_results, cv_bubbles_per_page, False

    api_log(f"MangaBatch({num_pages}p) [req={req_id}] EXHAUSTED -> all models unavailable/timed out")
    return [([], "") for _ in range(num_pages)], cv_bubbles_per_page, True


async def extract_and_translate_manga_bubbles(
    page_source: bytes | str,
    filename: str,
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    progress_cb=None,
) -> tuple[list[dict], str, list[dict]]:
    req_id = req_id or f"ms-{uuid.uuid4().hex[:6]}"
    cv_t0 = time.monotonic()
    b64_img, cv_bubbles = await asyncio.to_thread(
        _detect_page_bubbles_from_source_sync, page_source, 1280, 90
    )
    cv_el = time.monotonic() - cv_t0
    if not b64_img:
        return [], "", cv_bubbles

    if progress_cb is not None:
        try:
            await progress_cb(
                f"🌐 **2/3** เจอ `{len(cv_bubbles)}` ช่องคำพูด กำลังแปลอยู่นะคะ..."
            )
        except Exception:
            pass

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster:\n{series_context_hint[:1000]}\n"
        if series_context_hint else ""
    )
    if cv_bubbles:
        badge_ids_str = ", ".join(str(i) for i in range(1, len(cv_bubbles) + 1))
        badge_note = f"RED/WHITE NUMBERED BADGES ON THIS PAGE: [{badge_ids_str}] (account for EVERY badge ID strictly inside its own red outline!)"
    else:
        badge_note = "no numbered badges (use bubble_id=null + box_2d)"

    en_preserve_rule = _build_english_preservation_prompt_rule(target_lang_name)

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
This page has {badge_note}.
{context_block}
Translate EVERY SINGLE Japanese speech bubble, narration/caption box, section banner, table header, and expression label on this page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (NUMBERED BADGE) & TRANSLATION RULES:
1. MANDATORY 1-TO-1 NUMBERED BADGE CHECKLIST (`bubble_id`):
 - Output an entry for EVERY SINGLE badge ID listed above — NEVER skip a badge ID, NEVER merge two separate badges, and NEVER put text from a different bubble into another bubble's badge ID!
 - MANDATORY: Translate spoken speech bubbles, rectangular narration/caption boxes, AND EVERY badged Japanese section banner, table row header, and expression/detail label (e.g. `基本情報`, `年齢`, `身長`, `誕生日`, `所属`, `表情差分`, `通常`, `嬉`, `怒`, `哀`, `楽`, `照`, `能力`, `固有武器`, `必殺技`, `弱点・制約`, `武器・装備`, `デカい`)! Keep compact banner/table/expression labels concise (1-3 short words) so they fit cleanly inside the pill.
{en_preserve_rule} - CRITICAL ANTI-DUPLICATE RULE: If badge `k` outlines a character face, body, clothing, hand, background art, or pure SFX with NO dialogue/label text inside it, set `"should_translate": false`, `"source_text": ""`, `"translated_text": ""` — NEVER copy text from a nearby bubble onto an empty face/body badge!
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
 - For real Japanese dialogue, narration, or profile text WITHOUT a numbered badge (e.g. vertical text on open white panels), output a SEPARATE entry with `"bubble_id": null`, `"should_translate": true`, `"source_text"`, and `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY the Japanese text in its own panel).
 - NEVER concatenate or repeat text that already belongs to numbered badges into an extra `bubble_id: null` summary entry, and NEVER shift a banner/title onto an adjacent paragraph's badge ID.
3. Accurate Speaker Attribution, Names & Visual Gender:
 - Identify the actual speaker visually; keep character names and franchise-specific terms consistent with the Series / Post Context using accurate Japanese phonetics (e.g. `あおい` -> `อาโออิ`, `せな` -> `เซนะ`) or Romaji — NEVER translate Japanese proper names literally; match Thai/English tone to visual gender (never use male Thai `ครับ`/`ผม` for female characters/guards).
 - PURE TARGET LANGUAGE & FONT-SAFE OUTPUT: Write EVERY `translated_text` and `summary` purely in natural, correctly spelled {target_lang_name} using ONLY standard {target_lang_name} script and ASCII punctuation (`...`, `~`, `-`, `"`, `'`, `!`, `?`) — convert Japanese brackets `「」『』` into standard `"..."` quotes, with no emojis or heart/music symbols, and no mixed romanized syllables for Japanese interjections inside Thai (e.g. render `えへへ` as `เอะเฮะ`, `…っ!!` as `...อ๊ะ!!`, `コイツ` as `ไอ้หมอนี่`).
4. Return ONLY valid JSON matching this exact structure (write `summary` in {target_lang_name}):
{{
  "summary": "Brief 1-sentence scene context in {target_lang_name}",
  "bubbles": [
    {{
      "id": 1,
      "bubble_id": 1,
      "should_translate": true,
      "speaker_or_note": "Actual speaker name / Section Banner / Narration",
      "source_text": "Original Japanese text inside badge 1",
      "box_2d": null,
      "translated_text": "Natural {target_lang_name} translation"
    }}
  ]
}}
""".strip()

    valid_cv_ids = {int(c["bubble_id"]) for c in cv_bubbles if c.get("bubble_id") is not None}

    def _evaluate_grounding_quality(cand_bubbles: list[dict]) -> tuple[int, float, float]:
        """
        Scores how well `cand_bubbles` can actually be rendered onto the image:
        Returns `(anchored_count, anchored_char_ratio, composite_score)`.
        Prevents accepting a fallback output where a long bio paragraph was dumped into an
        unanchored `bubble_id: null` + `box_2d: null` entry (which prints in chat but drops from image!).
        """
        if not cand_bubbles:
            return 0, 0.0, 0.0
        total_chars = 0
        anchored_chars = 0
        anchored_cnt = 0
        used_bids: set[int] = set()
        for b in cand_bubbles:
            tr = (b.get("translated_text") or "").strip()
            ch_len = max(1, len(tr))
            total_chars += ch_len
            bid = b.get("bubble_id")
            box = b.get("box_2d")
            has_valid_bid = (bid is not None) and (int(bid) in valid_cv_ids) and (int(bid) not in used_bids)
            has_valid_box = isinstance(box, list) and len(box) == 4
            if has_valid_bid:
                used_bids.add(int(bid))
                anchored_cnt += 1
                anchored_chars += ch_len
            elif has_valid_box:
                anchored_cnt += 1
                anchored_chars += ch_len
        char_ratio = anchored_chars / float(max(1, total_chars))
        count_ratio = anchored_cnt / float(max(1, len(cand_bubbles)))
        score = (char_ratio * 0.70) + (count_ratio * 0.30)
        return anchored_cnt, char_ratio, score

    best_candidate: tuple[float, list[dict], str, str] | None = None

    def _consider_candidate(tier_label: str, cand_bubbles: list[dict], cand_summary: str) -> bool:
        nonlocal best_candidate
        anc_cnt, char_ratio, score = _evaluate_grounding_quality(cand_bubbles)
        if best_candidate is None or score > best_candidate[0]:
            best_candidate = (score, cand_bubbles, cand_summary, tier_label)
        # Accept immediately when anchored text covers >= 70% of all translated characters
        if anc_cnt > 0 and char_ratio >= 0.70:
            api_log(
                f"MangaSingle [req={req_id}] RESPONSE -> {tier_label} OK "
                f"({anc_cnt}/{len(cand_bubbles)} anchored, {int(round(char_ratio * 100))}% text grounded)"
            )
            return True
        if cand_bubbles:
            api_log(
                f"MangaSingle [req={req_id}] {tier_label} partial grounding "
                f"({anc_cnt}/{len(cand_bubbles)} anchored, {int(round(char_ratio * 100))}% text); trying next tier"
            )
        return False

    if GEMINI_API_KEY:
        api_log(
            f"MangaSingle/{MANGA_TRANSLATOR_VERSION} [req={req_id}] REQUEST -> Tier 1 Gemini ({filename}, {len(cv_bubbles)} CV bubbles, "
            f"engine={'cv2' if HAS_CV2 else 'pil'}, cv={cv_el:.2f}s)"
        )
        gemini_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": manga_system_instruction},
                        {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": MANGA_SINGLE_SOM_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(
                gemini_payload,
                timeout_sec=35,
                purpose=f"MangaSingle[{req_id}]",
                max_models=3,
                preferred_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview"],
            )
            if data:
                parts = (
                    ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                )
                raw_json_str = "".join(
                    p.get("text", "") for p in parts if p.get("text") and not p.get("thought", False)
                ).strip()
                bubbles, summary = _parse_manga_bubbles_json(
                    raw_json_str, target_lang_name=target_lang_name
                )
                if bubbles and _consider_candidate("Tier 1 Gemini", bubbles, summary):
                    return bubbles, summary, cv_bubbles
        except Exception as e:
            api_log(f"MangaSingle [req={req_id}] Tier 1 Gemini warning ({type(e).__name__}): {e}")

    try:
        gemma_content = await call_gemma_ai_studio_vision(
            manga_system_instruction,
            mime_type,
            b64_img,
            response_mime_type="application/json",
            timeout_sec=35,
            purpose=f"MangaSingle[{req_id}]",
        )
        if gemma_content:
            bubbles, summary = _parse_manga_bubbles_json(
                gemma_content, target_lang_name=target_lang_name
            )
            if bubbles and _consider_candidate("Tier 2 AI Studio Gemma 4", bubbles, summary):
                return bubbles, summary, cv_bubbles
    except Exception as e:
        api_log(f"MangaSingle [req={req_id}] Tier 2 AI Studio Gemma 4 warning: {e}")

    # Tier 2.5: OpenRouter Vision Fallback (`nvidia/nemotron-3.5-super-vl:free`) when Google AI Studio is down
    if (
        OPENROUTER_API_KEY
        and OPENROUTER_MANGA_MODEL
        and time.time() >= getattr(state, "openrouter_manga_cooldown_until", 0.0)
    ):
        try:
            or_content = await call_openrouter_multi_image_manga_batch(
                [(b64_img, filename, len(cv_bubbles))],
                manga_system_instruction,
                mime_type=mime_type,
                timeout_sec=45.0,
                max_tokens=4096,
                purpose=f"MangaSingleOpenRouter[{req_id}]",
            )
            if or_content:
                bubbles, summary = _parse_manga_bubbles_json(
                    or_content, target_lang_name=target_lang_name
                )
                if bubbles and _consider_candidate("Tier 2.5 OpenRouter Nemotron", bubbles, summary):
                    return bubbles, summary, cv_bubbles
        except Exception as e:
            api_log(f"MangaSingle [req={req_id}] Tier 2.5 OpenRouter warning: {e}")

    try:
        cf_content = await call_cloudflare_vision_chat(
            manga_system_instruction,
            mime_type,
            b64_img,
            max_tokens=4096,
            purpose=f"MangaSingleCF[{req_id}]",
        )
        if cf_content:
            bubbles, summary = _parse_manga_bubbles_json(
                cf_content, target_lang_name=target_lang_name
            )
            if bubbles and _consider_candidate("Tier 3 Cloudflare Gemma 4", bubbles, summary):
                return bubbles, summary, cv_bubbles
    except Exception as e:
        api_log(f"MangaSingle [req={req_id}] Tier 3 Cloudflare Gemma 4 warning: {e}")

    # Tier 4: Groq Qwen 3.8 27B Vision (`qwen/qwen3.8-27b`)
    if GROQ_API_KEY:
        try:
            gq_content = await call_groq_vision_chat(
                manga_system_instruction,
                mime_type,
                b64_img,
                max_tokens=4096,
                timeout_sec=25,
                purpose=f"MangaSingle[{req_id}]",
            )
            if gq_content:
                bubbles, summary = _parse_manga_bubbles_json(
                    gq_content, target_lang_name=target_lang_name
                )
                if bubbles and _consider_candidate("Tier 4 Groq Qwen", bubbles, summary):
                    return bubbles, summary, cv_bubbles
        except Exception as e:
            api_log(f"MangaSingle [req={req_id}] Tier 4 Groq Qwen warning: {e}")

    # Tier 5: Cloudflare Qwen 3.8 27B Vision (`@cf/qwen/qwen3.8-27b`, `reasoning_effort="low"`)
    try:
        cfq_content = await call_cloudflare_vision_fallback_qwen(
            manga_system_instruction,
            mime_type,
            b64_img,
            max_tokens=4096,
            purpose=f"MangaSingleCFQwen[{req_id}]",
            timeout_sec=30,
        )
        if cfq_content:
            bubbles, summary = _parse_manga_bubbles_json(
                cfq_content, target_lang_name=target_lang_name
            )
            if bubbles and _consider_candidate("Tier 5 Cloudflare Qwen", bubbles, summary):
                return bubbles, summary, cv_bubbles
    except Exception as e:
        api_log(f"MangaSingle [req={req_id}] Tier 5 Cloudflare Qwen warning: {e}")

    if best_candidate is not None:
        b_score, b_bubs, b_sum, b_tier = best_candidate
        api_log(
            f"MangaSingle [req={req_id}] RESPONSE -> Best available fallback ({b_tier}, "
            f"{len(b_bubs)} bubbles, grounding={b_score:.2f})"
        )
        return b_bubs, b_sum, cv_bubbles

    return [], "", cv_bubbles


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    """
    Translates 1 to 12 manga pages (Color & Black-and-White, Thai & English) from direct
    Discord attachments, zero-RAM `/tmp` image cache (instant on In-Bubble follow-ups!), OR
    embedded links using:
    - `manga_semaphore` (`Semaphore(1)`) OOM guard with queue waiting notice
    - Disk-backed `/tmp` job page streaming (input & rendered output pages stay on disk, keeping
      raw 1-12p buffers out of container RAM during 20-60s Vision API waits and Stage 3 uploads)
    - Vectorized OpenCV + NumPy CV Set-of-Mark (SoM) detection (white, pastel, corner, dark, UI banners/labels)
      & render masking
    - Shared `llm.py` Google AI Studio catalog & cooldowns + dynamic `15s`/`25s` `3.5-flash-lite` -> `3.1-flash-lite` hedge
    - Adaptive Stage 2 Gemma 4 text-only localization polish (auto-skips on clean <=4-bubble 1p strips;
      always-on parallel-chunked for multi-page batches & dense pages)
    - Throttled live Discord progress messages (1/3 -> 2/3 -> 3/3 -> auto-delete)
    """
    from media_context import resolve_referenced_message, collect_embedded_images_and_metadata
    from image_engine import _send_or_reply, send_chunked_response

    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel
    ref_msg = await resolve_referenced_message(msg) if getattr(msg, "reference", None) else None

    job_dir = ""
    req_id = ""
    disk_pages: list[tuple[str, str]] = []
    embed_meta_sections: list[str] = []
    cv_bubbles_per_page: list[list[dict]] = []
    discord_files: list[discord.File] = []

    try:
        job_seed_id = f"mjob-{uuid.uuid4().hex[:6]}"
        # 1. Check zero-RAM `/tmp` image cache first (instant hit when clicking 'แปลลงภาพ (In-Bubble)'!)
        cached_hit = None
        for candidate_m in (msg, ref_msg):
            m_id = getattr(candidate_m, "id", 0) if candidate_m is not None else 0
            if m_id:
                cached_hit = await asyncio.to_thread(load_temp_image_cache, m_id)
                if cached_hit:
                    break

        if cached_hit:
            raw_pages, embed_meta_sections = cached_hit
            raw_pages = raw_pages[:MAX_MANGA_BATCH_PAGES]
            job_dir, disk_pages = await asyncio.to_thread(_spill_pages_to_job_dir_sync, raw_pages, job_seed_id)
            raw_pages.clear()
            api_log(f"MangaCache HIT -> streamed {len(disk_pages)} page(s) to job /tmp (skipped re-download)")
        else:
            image_attachments = [
                a for a in (getattr(msg, "attachments", None) or [])
                if a.filename.lower().endswith(IMAGE_EXTENSIONS)
            ]
            if not image_attachments and ref_msg:
                image_attachments = [
                    a for a in ref_msg.attachments
                    if a.filename.lower().endswith(IMAGE_EXTENSIONS)
                ]

            if image_attachments:
                job_dir = await asyncio.to_thread(_create_manga_job_dir_sync, job_seed_id)
                for p_i, att in enumerate(image_attachments[:MAX_MANGA_BATCH_PAGES], 1):
                    raw_b = await att.read()
                    p_path = await asyncio.to_thread(_write_manga_page_to_job_dir_sync, job_dir, p_i, raw_b)
                    disk_pages.append((p_path, att.filename))
                    raw_b = None
                image_attachments.clear()

            if not disk_pages:
                embedded_imgs, embed_meta_sections, _ = await collect_embedded_images_and_metadata(
                    msg, ref_msg=ref_msg, max_images=MAX_MANGA_BATCH_PAGES
                )
                if embedded_imgs:
                    raw_embed_pairs = [(emb_bytes, emb_fname) for emb_bytes, emb_fname, _ in embedded_imgs[:MAX_MANGA_BATCH_PAGES]]
                    embedded_imgs.clear()
                    job_dir, disk_pages = await asyncio.to_thread(_spill_pages_to_job_dir_sync, raw_embed_pairs, job_seed_id)
                    raw_embed_pairs.clear()

        if not disk_pages:
            await channel.send(
                "ถ้าไม่แนบรูปหน้ามังงะหรือวางลิงก์ (X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วย ฉันก็ช่วยแปลให้ไม่ได้นะคะ "
                "แนบหลายหน้าพร้อมกันได้เลย เช่น พิมพ์ `!manga en` หรือ `!manga th` ค่ะ"
            )
            return

        # Reclaim upload/download buffers immediately before entering queued API waits
        release_unused_memory()

        context_hints = []
        if embed_meta_sections:
            context_hints.append("\n".join(embed_meta_sections)[:800])
        if channel and getattr(channel, "id", None):
            cached_ocr = get_cached_channel_attachments(channel.id, ttl_sec=1800, max_total_chars=5000)
            if cached_ocr:
                # Match the LATEST identification block from single-image Gemini, single-image Gemma, OR multi-image OCR
                id_matches = re.findall(
                    r"((?:1\.\s*\[(?:Series\s*/\s*Franchise|Identification\s*&\s*Visual\s*Context)|2\.\s*\[Deduced\s*Character\s*/\s*Franchise)[\s\S]{0,800}?(?=\n(?:2\.\s*\[Ordered|3\.\s*\[Verbatim|2\.\s*\[Page-by-Page|===)|\Z))",
                    cached_ocr,
                )
                if id_matches:
                    context_hints.append(id_matches[-1].strip()[:800])
        series_context_hint = "\n\n".join(context_hints)[:1400]

        _, target_lang_name = _detect_target_translation_lang(user_prompt)
        disk_pages = disk_pages[:MAX_MANGA_BATCH_PAGES]
        num_pages = len(disk_pages)
        req_id = f"mb-{uuid.uuid4().hex[:6]}" if num_pages > 1 else f"ms-{uuid.uuid4().hex[:6]}"

        queue_notice_msg = None
        if manga_semaphore.locked():
            api_log(f"MangaQueue [req={req_id}] -> waiting for active manga job to finish (OOM protection)")
            try:
                queue_notice_msg = await _send_or_reply(
                    channel,
                    msg,
                    content="-# ⏳ ตอนนี้ฉันติดแปลอีกชุดอยู่นะคะ เสร็จแล้วจะรีบจัดการหน้านี้ให้ต่อเลยค่ะ",
                )
            except Exception:
                queue_notice_msg = None

        progress_prefix = f"❄️ **[แปลมังงะ {num_pages} หน้า → {target_lang_name}]**"
        status_msg: discord.Message | None = None
        last_edit_ts = 0.0

        async def _update_progress(detail_text: str, force: bool = False):
            nonlocal status_msg, last_edit_ts
            now = time.monotonic()
            if not force and status_msg is not None and (now - last_edit_ts) < 2.2:
                return
            full_text = f"{progress_prefix} {detail_text}"
            try:
                if status_msg is None:
                    status_msg = await _send_or_reply(channel, msg, content=full_text)
                else:
                    await status_msg.edit(content=full_text)
                last_edit_ts = time.monotonic()
            except Exception:
                pass

        async def _cleanup_status(replace_with: str | None = None):
            nonlocal status_msg
            if status_msg is None:
                return
            try:
                if replace_with:
                    await status_msg.edit(content=replace_with)
                else:
                    await status_msg.delete()
            except Exception:
                if replace_with:
                    await _send_or_reply(channel, msg, content=replace_with)
            finally:
                status_msg = None

        async with manga_semaphore:
            if queue_notice_msg is not None:
                try:
                    await queue_notice_msg.delete()
                except Exception:
                    pass
                queue_notice_msg = None

            async with channel.typing():
                try:
                    await _update_progress(
                        "🔍 **1/3** กำลังหาช่องคำพูด...",
                        force=True,
                    )

                    batch_results, cv_bubbles_per_page, is_unavailable = await extract_and_translate_manga_pages_batch(
                        disk_pages,
                        target_lang_name,
                        series_context_hint=series_context_hint,
                        req_id=req_id,
                        progress_cb=lambda txt: _update_progress(txt, force=False),
                    )

                    if is_unavailable:
                        await _cleanup_status(
                            replace_with="ตอนนี้ฝั่งแปลหลายหน้าขัดข้องชั่วคราวค่ะ ลองใหม่อีกสักครู่นะคะ"
                        )
                        return

                    if not any(bubbles or summary for bubbles, summary in batch_results):
                        await _cleanup_status(
                            replace_with="อ่านข้อความในภาพไม่ได้เลยค่ะ ลองเช็คดูหน่อยนะคะว่าภาพชัดพอไหม"
                        )
                        return

                    batch_results = _dedupe_hallucinated_page_bubbles(
                        batch_results, cv_bubbles_per_page, req_id=req_id
                    )
                    batch_results, cv_bubbles_per_page = await _recover_omitted_pages_async(
                        disk_pages,
                        batch_results,
                        cv_bubbles_per_page,
                        target_lang_name,
                        series_context_hint=series_context_hint,
                        req_id=req_id,
                    )
                    batch_results = await _polish_manga_translations_gemma_async(
                        batch_results,
                        target_lang_name,
                        series_context_hint=series_context_hint,
                        req_id=req_id,
                        progress_cb=lambda txt: _update_progress(txt, force=True),
                    )

                    header_title = (
                        f"**📖 คำแปลมังงะ {num_pages} หน้า ตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
                        if num_pages > 1
                        else f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
                    )
                    transcript_lines = [header_title]
                    render_t0 = time.monotonic()
                    api_log(
                        f"MangaRender [req={req_id}] START ({'cv2' if HAS_CV2 else 'pil'}/{MANGA_CV_VER}, disk-stream) -> rendering {num_pages} page(s)..."
                    )
                    await _update_progress(
                        f"🎨 **3/3** กำลังใส่คำแปลลงภาพ (`0/{num_pages}` หน้า)...",
                        force=True,
                    )

                    for page_idx in range(1, num_pages + 1):
                        page_path, fname = disk_pages[page_idx - 1]
                        bubbles, scene_summary = batch_results[page_idx - 1]
                        page_cv = cv_bubbles_per_page[page_idx - 1] if (page_idx - 1) < len(cv_bubbles_per_page) else []

                        if num_pages > 1:
                            transcript_lines.append(f"\n__**📄 หน้าที่ {page_idx}/{num_pages} (`{fname}`)**__")
                        if scene_summary:
                            clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
                            transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

                        if not bubbles:
                            transcript_lines.append("*(ไม่พบช่องคำพูดในหน้านี้)*")
                            _release_cv_page_masks(page_cv)
                            continue

                        for idx, b in enumerate(bubbles, 1):
                            speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
                            src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
                            tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
                            if tr_txt:
                                transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n ↳ **{tr_txt}**")

                        try:
                            out_jpg_path = os.path.join(job_dir, f"out_p{page_idx:02d}.jpg")
                            page_rstats: dict = {}
                            rendered_ok = await asyncio.to_thread(
                                _render_manga_page_to_disk_sync,
                                page_path,
                                bubbles,
                                page_cv,
                                out_jpg_path,
                                page_rstats,
                            )
                            if rendered_ok:
                                base_name = os.path.splitext(fname)[0]
                                discord_files.append(
                                    discord.File(
                                        out_jpg_path,
                                        filename=f"translated_p{page_idx}_{base_name}.jpg",
                                    )
                                )
                                r_cnt = page_rstats.get("rendered_bubbles")
                                t_cnt = page_rstats.get("total_bubbles")
                                if r_cnt is not None and t_cnt is not None and r_cnt < t_cnt:
                                    api_log(
                                        f"MangaRender [req={req_id}] page {page_idx}/{num_pages} ({fname}) "
                                        f"rendered {r_cnt}/{t_cnt} bubble(s) (cv={len(page_cv)})"
                                    )
                            else:
                                api_log(
                                    f"MangaRender [req={req_id}] page {page_idx}/{num_pages} ({fname}) returned None "
                                    f"(cv={len(page_cv)}, llm={len(bubbles)})"
                                )
                        except Exception as e:
                            api_log(f"MangaRender [req={req_id}] page {page_idx} fallback to text-only: {e}")
                        finally:
                            _release_cv_page_masks(page_cv)

                        if num_pages > 1 and page_idx < num_pages:
                            await _update_progress(
                                f"🎨 **3/3** กำลังใส่คำแปลลงภาพ (`{page_idx}/{num_pages}` หน้า)...",
                                force=False,
                            )
                        await asyncio.sleep(0.05)

                    render_elapsed = time.monotonic() - render_t0
                    api_log(
                        f"MangaRender [req={req_id}] DONE ({'cv2' if HAS_CV2 else 'pil'}) -> "
                        f"rendered {len(discord_files)}/{num_pages} page(s) in {render_elapsed:.2f}s"
                    )
                    transcript_text = "\n".join(transcript_lines)

                    await _cleanup_status()

                    if discord_files:
                        file_chunks = [discord_files[i : i + 10] for i in range(0, len(discord_files), 10)]
                        if len(transcript_text) <= 1950:
                            await _send_or_reply(channel, msg, content=transcript_text, files=file_chunks[0])
                            for extra_files in file_chunks[1:]:
                                await _send_or_reply(channel, None, files=extra_files)
                        else:
                            for idx_fc, fc in enumerate(file_chunks):
                                await _send_or_reply(channel, msg if idx_fc == 0 else None, files=fc)
                            await send_chunked_response(channel, transcript_text, reply_to=None)
                    else:
                        await send_chunked_response(channel, transcript_text, reply_to=msg)
                finally:
                    await _cleanup_status()
    finally:
        for df in discord_files:
            try:
                df.close()
            except Exception:
                pass
        discord_files.clear()
        for page_cv in cv_bubbles_per_page:
            _release_cv_page_masks(page_cv)
        cv_bubbles_per_page.clear()
        disk_pages.clear()
        if job_dir:
            await asyncio.to_thread(_cleanup_manga_job_dir_sync, job_dir)
        if req_id or job_dir:
            before_mb, after_mb = release_unused_memory()
            freed_mb = round(max(0.0, before_mb - after_mb), 1)
            api_log(
                f"MangaCleanup [req={req_id or 'pre'}] -> RAM {before_mb} MB -> {after_mb} MB"
                + (f" (freed {freed_mb} MB)" if freed_mb > 0 else "")
            )