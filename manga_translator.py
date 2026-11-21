import os
import io
import re
import json
import time
import uuid
import asyncio

import aiohttp
import discord

from config import (
    HTTPS,
    GEMINI_API_KEY,
    GROQ_API_KEY,
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
    discover_active_groq_vision_models,
    _fetch_google_ai_studio_model_names,
    get_google_model_cooldown,
    set_google_model_cooldown,
)
from manga_cv_render import (
    HAS_CV2,
    _sanitize_text_for_manga_font,
    _detect_page_bubbles_and_build_som_b64_sync,
    render_translated_manga_image_sync,
    _release_cv_page_masks,
)

MAX_MANGA_BATCH_PAGES = 12  # Supports up to 12 pages across multi-part threads (split into <=10 files per Discord message)
# Bundle 2 dynamic parallel hedge: faster 15s recovery on typical 2-4p X threads, patient 25s on 5-12p batches
MANGA_BATCH_HEDGE_DELAY_SMALL_SEC = 15.0
MANGA_BATCH_HEDGE_DELAY_LARGE_SEC = 25.0

_FLASH_LITE_EXCLUDED = ("2.5", "2.0", "1.5", "latest", "image", "tts", "audio", "live", "embedding")

# Set-of-Mark (SoM) JSON responseSchema with explicit `propertyOrdering`:
# - Every red-numbered CV bubble `[k]` maps 1-to-1 via `"bubble_id": k` (zero coordinate drift!).
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
                                "is_dialogue": {"type": "BOOLEAN"},
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
                                "is_dialogue",
                                "speaker_or_note",
                                "source_text",
                                "box_2d",
                                "translated_text",
                            ],
                            "required": [
                                "id",
                                "bubble_id",
                                "is_dialogue",
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
                    "is_dialogue": {"type": "BOOLEAN"},
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
                    "is_dialogue",
                    "speaker_or_note",
                    "source_text",
                    "box_2d",
                    "translated_text",
                ],
                "required": [
                    "id",
                    "bubble_id",
                    "is_dialogue",
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
# STAGE 2: JSON NORMALIZATION & HEDGED BATCH VISION ORCHESTRATION
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


def _normalize_and_validate_bubbles(raw_bubbles: list) -> list[dict]:
    valid = []
    for idx, item in enumerate(raw_bubbles, 1):
        if not isinstance(item, dict):
            continue

        if item.get("is_dialogue") is False:
            continue

        speaker = (
            item.get("speaker_or_note")
            or item.get("speaker")
            or item.get("character")
            or f"Bubble {idx}"
        )
        src_txt = (
            item.get("source_text")
            or item.get("original_text")
            or item.get("japanese_text")
            or item.get("source")
            or ""
        )
        tr_txt = (
            item.get("translated_text")
            or item.get("translation")
            or item.get("translated")
            or item.get("english_text")
            or item.get("target_text")
            or item.get("text")
            or ""
        )
        tr_txt = _sanitize_text_for_manga_font(str(tr_txt))
        tr_txt = re.sub(
            r"-\s+(san|kun|chan|sama|sensei|senpai|dono|shi)\b",
            r"-\1",
            tr_txt,
            flags=re.IGNORECASE,
        )
        src_txt = re.sub(r"\s+", " ", str(src_txt)).strip()
        if not tr_txt:
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

        valid.append({
            "id": item.get("id", idx),
            "bubble_id": bubble_id,
            "speaker_or_note": str(speaker).strip(),
            "source_text": src_txt,
            "translated_text": tr_txt,
            "box_2d": norm_box,
        })
    return valid


def _repair_truncated_manga_batch_json(raw_target: str) -> dict | None:
    if '"pages"' not in raw_target:
        return None
    last_close_brace = raw_target.rfind("}")
    while last_close_brace > 20:
        candidate = raw_target[: last_close_brace + 1]
        for suffix in ("]}", "]}]}", "]}"):
            try:
                obj = json.loads(candidate + suffix)
                if isinstance(obj, dict) and isinstance(obj.get("pages"), list) and obj["pages"]:
                    return obj
            except Exception:
                pass
        last_close_brace = raw_target.rfind("}", 0, last_close_brace)
    return None


def _parse_manga_pages_batch_json(
    raw_text: str, expected_pages: int
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
                bubbles = _normalize_and_validate_bubbles(p_obj.get("bubbles") or [])
                summary = str(p_obj.get("summary") or "").strip()
                results[p_idx] = (bubbles, summary)
            if any(b or s for b, s in results):
                return results

        if expected_pages == 1 and isinstance(parsed, dict) and "bubbles" in parsed:
            bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
            summary = str(parsed.get("summary") or "").strip()
            return [(bubbles, summary)]

    if expected_pages == 1:
        b, s = _parse_manga_bubbles_json(raw_text)
        return [(b, s)]

    return [([], "") for _ in range(expected_pages)]


def _parse_manga_bubbles_json(raw_text: str) -> tuple[list[dict], str]:
    if not raw_text:
        return [], ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    target = json_match.group(0) if json_match else cleaned

    try:
        parsed = json.loads(target)
        bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
        summary = str(parsed.get("summary") or "").strip()
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    try:
        repaired = target
        repaired = re.sub(r'(["\d\]}])\s*\n(\s*["{])', r"\1,\n\2", repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        parsed = json.loads(repaired)
        bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
        summary = str(parsed.get("summary") or "").strip()
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    summary_m = re.search(r'"summary"\s*:\s*"([^"\n]+)"', target)
    summary = summary_m.group(1).strip() if summary_m else ""

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
            "is_dialogue": True,
            "speaker_or_note": spk_m.group(1).strip() if spk_m else f"Bubble {idx}",
            "source_text": src_m.group(1).strip() if src_m else "",
            "translated_text": raw_tr_val,
            "box_2d": norm_box,
        })

    valid_extracted = _normalize_and_validate_bubbles(extracted_bubbles)
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
        "temperature": 0.1,
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
            raw_json_str = "\n".join(
                p.get("text", "") for p in resp_parts
                if p.get("text") and not p.get("thought", False)
            ).strip()
            batch_results = _parse_manga_pages_batch_json(raw_json_str, num_pages)
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
                api_log(
                    f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 OK "
                    f"({elapsed:.2f}s{cache_note}, {total_b} bubbles: {matched_som} SoM-anchored, {total_anchored - matched_som} box_2d)"
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


async def extract_and_translate_manga_pages_batch(
    pages_input: list[tuple[bytes, str]],
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    progress_cb=None,
) -> tuple[list[tuple[list[dict], str]], list[list[dict]], bool]:
    """
    Set-of-Mark (SoM) CV-Anchored Multi-Page Manga Batch Pipeline:
    1) Detects enclosed speech bubbles via vectorized OpenCV+NumPy (or lazy PIL fallback)
       and stamps compact top-right-rim red badges `[1]`, `[2]`, ...
    2) Calls `MangaBatch` strictly across:
         `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 `flash-lite`)
         -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
       using shared `llm.py` catalog/cooldowns AND dynamic parallel hedging
       (15s for 2-4p batches, 25s for 5-12p batches) on `3.5-flash-lite`!
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

    if not GEMINI_API_KEY:
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
    for raw_b, fname in pages_input:
        b64_str, cv_bubbles = await asyncio.to_thread(
            _detect_page_bubbles_and_build_som_b64_sync, raw_b, batch_dim, batch_q
        )
        b64_pages.append((b64_str, fname, len(cv_bubbles)))
        cv_bubbles_per_page.append(cv_bubbles)

    total_cv = sum(len(c) for c in cv_bubbles_per_page)
    cv_elapsed = time.monotonic() - som_t0
    engine_tag = "cv2" if HAS_CV2 else "pil"
    api_log(
        f"MangaBatch({num_pages}p) [req={req_id}] CV_DETECT ({engine_tag}) -> found {total_cv} enclosed bubble(s) "
        f"across {num_pages} page(s) in {cv_elapsed:.2f}s"
    )
    await _notify(
        f"🌐 **2/3** เจอ `{total_cv}` ช่องคำพูด กำลังแปลอยู่นะคะ..."
    )

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster (use accurate names & terms):\n{series_context_hint[:1400]}\n"
        if series_context_hint else ""
    )

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
Every enclosed speech bubble and caption box has ALREADY been outlined in RED with a RED NUMBERED BADGE (`1`, `2`, `3`, ...) at its top-right rim!
{context_block}
Analyze all {num_pages} page(s) in sequence so character names, titles/ranks, pronouns, and ongoing story stay consistent across every page,
and translate EVERY SINGLE speech bubble and narration/caption box on EVERY page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (RED BADGE) & TRANSLATION RULES:
1. MANDATORY 1-TO-1 RED BADGE CHECKLIST (`bubble_id`):
   - Each page header lists the exact red badge IDs present on that page (`[1, 2, ..., K]`).
   - You MUST inspect and output an entry for EVERY SINGLE red badge ID `1..K` on that page — NEVER skip a badge ID, NEVER merge two separate red badges into one entry, and NEVER put text from a DIFFERENT panel into another panel's red badge!
   - If red badge `k` contains Japanese dialogue, monologue, or a caption box: set `"bubble_id": k`, `"is_dialogue": true`, `"box_2d": null`, transcribe ONLY the exact `"source_text"` visually INSIDE badge `k`, and write its natural {target_lang_name} `"translated_text"`.
   - If red badge `k` is NOT dialogue (for example, a sound effect / SFX like ゲプ…, a non-speech diagram/icon, or a blank background/face/tile strip with no dialogue): set `"bubble_id": k`, `"is_dialogue": false`, `"source_text": ""`, `"translated_text": ""` so it is left untouched!
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
   - ALSO scan every panel for real spoken dialogue or monologue written directly on white/light areas WITHOUT a red badge around it (such as borderless vertical monologue lines in a panel with no red badge!).
   - For each unmarked dialogue line, add a SEPARATE entry with `"bubble_id": null`, `"is_dialogue": true`, transcribe `"source_text"`, and provide `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY that Japanese text in its OWN panel — NEVER over faces, hands, or artwork!).
   - NEVER merge unbadged dialogue from another panel into an earlier panel's red badge!
   - NEVER output floating handwritten SFX (e.g. ドーン, トスッ, ポスン, ムチッ, ぐっ, カチャッ, 脱, ぎゅ) as `box_2d`.
3. Accurate Speaker Attribution, Names & Visual Gender:
   - Identify the actual character speaking each bubble visually (follow the tail; never copy one character's name onto every bubble).
   - Keep character names, titles/ranks, and faction names consistent across all pages using accurate Japanese phonetics.
   - Match pronouns and politeness particles strictly to each speaker's visual gender and setting: in Thai, visually female characters (including female subordinates, guards, and attendants) must use female/neutral endings (`ค่ะ`, `คะ`, `นะคะ`, `นะ`, `ล่ะ`, `สิ`) — NEVER use male `ครับ`/`ผม` unless the speaker is clearly drawn as male!
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
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
          "is_dialogue": true,
          "speaker_or_note": "Actual speaker name / Narration",
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
                badge_note = f"RED BADGES ON THIS PAGE: [{badge_ids_str}] (account for EVERY badge ID strictly inside its own red outline; add bubble_id=null for any unbadged dialogue in other panels)"
            else:
                badge_note = "no red badges (use bubble_id=null + box_2d for any dialogue)"
            parts.append({"text": f"=== PAGE {idx} ({fname} | {badge_note}) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        full_chain = await _discover_manga_batch_models()
        active_chain = [m for m in full_chain if not _is_gemini_model_on_cooldown(m[0])]
        if not active_chain:
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
                total_cv=total_cv, notify_fn=_notify,
            )
            if res_m is not None:
                return res_m, cv_bubbles_per_page, False
            if fb_m is not None and fallback_text_only_results is None:
                fallback_text_only_results = fb_m
    finally:
        b64_pages.clear()

    if fallback_text_only_results is not None:
        return fallback_text_only_results, cv_bubbles_per_page, False

    api_log(f"MangaBatch({num_pages}p) [req={req_id}] EXHAUSTED -> all models unavailable/timed out")
    return [([], "") for _ in range(num_pages)], cv_bubbles_per_page, True


async def extract_and_translate_manga_bubbles(
    raw_bytes: bytes,
    filename: str,
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
    progress_cb=None,
) -> tuple[list[dict], str, list[dict]]:
    req_id = req_id or f"ms-{uuid.uuid4().hex[:6]}"
    cv_t0 = time.monotonic()
    b64_img, cv_bubbles = await asyncio.to_thread(
        _detect_page_bubbles_and_build_som_b64_sync, raw_bytes, 1280, 90
    )
    cv_el = time.monotonic() - cv_t0
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
        badge_note = f"RED BADGES ON THIS PAGE: [{badge_ids_str}] (account for EVERY badge ID strictly inside its own red outline!)"
    else:
        badge_note = "no red badges (use bubble_id=null + box_2d)"

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
This page has {badge_note}.
{context_block}
Translate EVERY SINGLE speech bubble and narration/caption box on this page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (RED BADGE) & TRANSLATION RULES:
1. MANDATORY 1-TO-1 RED BADGE CHECKLIST (`bubble_id`):
   - Output an entry for EVERY SINGLE red badge ID listed above — NEVER skip a badge ID, NEVER merge two separate red badges, and NEVER put text from a different panel into another panel's red badge!
   - If red badge `k` contains Japanese dialogue/narration: set `"bubble_id": k`, `"is_dialogue": true`, `"box_2d": null`, exact `"source_text"` inside ONLY badge `k`, and its natural {target_lang_name} `"translated_text"`.
   - If red badge `k` is NOT dialogue (SFX, diagram, background object, face): set `"bubble_id": k`, `"is_dialogue": false`, `"source_text": ""`, `"translated_text": ""`.
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
   - For real spoken dialogue/narration WITHOUT a red badge (e.g. vertical text on open white panels), output a SEPARATE entry with `"bubble_id": null`, `"is_dialogue": true`, `"source_text"`, and `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY the Japanese text in its own panel).
3. Accurate Speaker Attribution, Names & Visual Gender:
   - Identify the actual speaker visually; match Thai/English tone to visual gender (never use male Thai `ครับ`/`ผม` for female characters/guards).
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
4. Return ONLY valid JSON matching this exact structure (write `summary` in {target_lang_name}):
{{
  "summary": "Brief 1-sentence scene context in {target_lang_name}",
  "bubbles": [
    {{
      "id": 1,
      "bubble_id": 1,
      "is_dialogue": true,
      "speaker_or_note": "Actual speaker name / Narration",
      "source_text": "Original Japanese text inside badge 1",
      "box_2d": null,
      "translated_text": "Natural {target_lang_name} translation"
    }}
  ]
}}
""".strip()

    fallback_bubbles_no_boxes = []
    fallback_summary = ""

    if GEMINI_API_KEY:
        api_log(
            f"MangaSingle [req={req_id}] REQUEST -> Tier 1 Gemini ({filename}, {len(cv_bubbles)} CV bubbles, "
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
                "temperature": 0.1,
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
                raw_json_str = "\n".join(
                    p.get("text", "") for p in parts if p.get("text") and not p.get("thought", False)
                ).strip()
                bubbles, summary = _parse_manga_bubbles_json(raw_json_str)
                if bubbles:
                    anchored = sum(
                        1 for b in bubbles
                        if (b.get("bubble_id") is not None)
                        or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                    )
                    if anchored > 0:
                        api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 1 Gemini OK ({anchored}/{len(bubbles)} anchored)")
                        return bubbles, summary, cv_bubbles
                    fallback_bubbles_no_boxes = bubbles
                    fallback_summary = summary
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
            bubbles, summary = _parse_manga_bubbles_json(gemma_content)
            if bubbles:
                has_anchor = any(
                    (b.get("bubble_id") is not None)
                    or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                    for b in bubbles
                )
                if has_anchor:
                    api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 2 AI Studio Gemma 4 OK ({len(bubbles)} bubbles)")
                    return bubbles, summary, cv_bubbles
                if not fallback_bubbles_no_boxes:
                    fallback_bubbles_no_boxes = bubbles
                    fallback_summary = summary
    except Exception as e:
        api_log(f"MangaSingle [req={req_id}] Tier 2 AI Studio Gemma 4 warning: {e}")

    try:
        cf_content = await call_cloudflare_vision_chat(manga_system_instruction, mime_type, b64_img)
        if cf_content:
            bubbles, summary = _parse_manga_bubbles_json(cf_content)
            if bubbles:
                has_anchor = any(
                    (b.get("bubble_id") is not None)
                    or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                    for b in bubbles
                )
                if has_anchor:
                    api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 3 Cloudflare Gemma 4 OK ({len(bubbles)} bubbles)")
                    return bubbles, summary, cv_bubbles
                if not fallback_bubbles_no_boxes:
                    fallback_bubbles_no_boxes = bubbles
                    fallback_summary = summary
    except Exception as e:
        api_log(f"MangaSingle [req={req_id}] Tier 3 Cloudflare Gemma 4 warning: {e}")

    if GROQ_API_KEY:
        groq_vision_candidates = await discover_active_groq_vision_models()
        groq_host = "api." + "groq.com"
        chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        session = get_http_session()
        for g_model in groq_vision_candidates:
            try:
                groq_payload = {
                    "model": g_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": manga_system_instruction},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
                async with session.post(
                    chat_url, headers=headers, json=groq_payload, timeout=aiohttp.ClientTimeout(total=25)
                ) as resp:
                    if resp.status == 200:
                        state.active_groq_vision_model = g_model
                        state.record_provider_call("groq")
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        bubbles, summary = _parse_manga_bubbles_json(content)
                        if bubbles:
                            has_anchor = any(
                                (b.get("bubble_id") is not None)
                                or (isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                                for b in bubbles
                            )
                            if has_anchor:
                                api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 4 Groq ({g_model}) OK ({len(bubbles)} bubbles)")
                                return bubbles, summary, cv_bubbles
                            if not fallback_bubbles_no_boxes:
                                fallback_bubbles_no_boxes = bubbles
                                fallback_summary = summary
            except Exception as e:
                api_log(f"MangaSingle [req={req_id}] Tier 4 Groq ({g_model}) warning: {e}")

    if fallback_bubbles_no_boxes:
        return fallback_bubbles_no_boxes, fallback_summary, cv_bubbles

    return [], "", cv_bubbles


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    """
    Translates 1 to 12 manga pages (Color & Black-and-White, Thai & English) from direct
    Discord attachments, zero-RAM `/tmp` image cache (instant on In-Bubble follow-ups!), OR
    embedded links using:
    - `manga_semaphore` (`Semaphore(1)`) OOM guard with queue waiting notice
    - Vectorized OpenCV + NumPy CV Set-of-Mark (SoM) detection & render masking (with lazy PIL fallback)
    - Shared `llm.py` Google AI Studio catalog & cooldowns + dynamic `15s`/`25s` `3.5-flash-lite` -> `3.1-flash-lite` hedge
    - Throttled live Discord progress messages (1/3 -> 2/3 -> 3/3 -> auto-delete)
    - Border-safe local mask refinement, HSV dark-panel rejection, two-lobe figure-8 typesetting,
      and syllable-aware Thai wrapping
    """
    from media_context import resolve_referenced_message, collect_embedded_images_and_metadata
    from image_engine import _send_or_reply, send_chunked_response

    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel
    ref_msg = await resolve_referenced_message(msg) if getattr(msg, "reference", None) else None

    pages_input: list[tuple[bytes, str]] = []
    embed_meta_sections: list[str] = []

    # 1. Check zero-RAM `/tmp` image cache first (instant hit when clicking 'แปลลงภาพ (In-Bubble)'!)
    cached_hit = None
    for candidate_m in (msg, ref_msg):
        m_id = getattr(candidate_m, "id", 0) if candidate_m is not None else 0
        if m_id:
            cached_hit = await asyncio.to_thread(load_temp_image_cache, m_id)
            if cached_hit:
                break

    if cached_hit:
        pages_input, embed_meta_sections = cached_hit
        api_log(f"MangaCache HIT -> loaded {len(pages_input)} page(s) from /tmp (skipped re-download)")
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

        for att in image_attachments[:MAX_MANGA_BATCH_PAGES]:
            raw_b = await att.read()
            pages_input.append((raw_b, att.filename))
        image_attachments.clear()

        if not pages_input:
            embedded_imgs, embed_meta_sections, _ = await collect_embedded_images_and_metadata(
                msg, ref_msg=ref_msg, max_images=MAX_MANGA_BATCH_PAGES
            )
            for emb_bytes, emb_fname, _ in embedded_imgs:
                pages_input.append((emb_bytes, emb_fname))
            embedded_imgs.clear()

    if not pages_input:
        await channel.send(
            "แนบรูปหน้ามังงะหรือวางลิงก์ (X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วยสิคะ "
            "แนบหลายหน้าพร้อมกันได้เลย เช่น พิมพ์ `!manga en` หรือ `!manga th` ค่ะ"
        )
        return

    context_hints = []
    if embed_meta_sections:
        context_hints.append("\n".join(embed_meta_sections)[:800])
    if channel and getattr(channel, "id", None):
        cached_ocr = get_cached_channel_attachments(channel.id, ttl_sec=1800, max_total_chars=4000)
        if cached_ocr:
            id_match = re.search(
                r"(1\.\s*\[Series\s*/\s*Franchise[\s\S]{0,700})", cached_ocr
            )
            if id_match:
                context_hints.append(id_match.group(1).strip())
    series_context_hint = "\n\n".join(context_hints)[:1400]

    _, target_lang_name = _detect_target_translation_lang(user_prompt)
    pages_input = pages_input[:MAX_MANGA_BATCH_PAGES]
    num_pages = len(pages_input)
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

    cv_bubbles_per_page: list[list[dict]] = []
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
                    pages_input,
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

                header_title = (
                    f"**📖 คำแปลมังงะ {num_pages} หน้า ตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
                    if num_pages > 1
                    else f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
                )
                transcript_lines = [header_title]
                discord_files = []
                render_t0 = time.monotonic()
                api_log(
                    f"MangaRender [req={req_id}] START ({'cv2' if HAS_CV2 else 'pil'}) -> rendering {num_pages} page(s)..."
                )
                await _update_progress(
                    f"🎨 **3/3** กำลังใส่คำแปลลงภาพ (`0/{num_pages}` หน้า)...",
                    force=True,
                )

                for page_idx in range(1, num_pages + 1):
                    raw_bytes, fname = pages_input[page_idx - 1]
                    bubbles, scene_summary = batch_results[page_idx - 1]
                    page_cv = cv_bubbles_per_page[page_idx - 1] if (page_idx - 1) < len(cv_bubbles_per_page) else []

                    if num_pages > 1:
                        transcript_lines.append(f"\n__**📄 หน้าที่ {page_idx}/{num_pages} (`{fname}`)**__")
                    if scene_summary:
                        clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
                        transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

                    if not bubbles:
                        transcript_lines.append("*(ไม่พบช่องคำพูดในหน้านี้)*")
                        pages_input[page_idx - 1] = (b"", fname)
                        _release_cv_page_masks(page_cv)
                        continue

                    for idx, b in enumerate(bubbles, 1):
                        speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
                        src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
                        tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
                        if tr_txt:
                            transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n  ↳ **{tr_txt}**")

                    try:
                        rendered_jpg = await asyncio.to_thread(
                            render_translated_manga_image_sync, raw_bytes, bubbles, page_cv
                        )
                        if rendered_jpg:
                            base_name = os.path.splitext(fname)[0]
                            discord_files.append(
                                discord.File(
                                    io.BytesIO(rendered_jpg),
                                    filename=f"translated_p{page_idx}_{base_name}.jpg",
                                )
                            )
                    except Exception as e:
                        api_log(f"MangaRender [req={req_id}] page {page_idx} fallback to text-only: {e}")
                    finally:
                        pages_input[page_idx - 1] = (b"", fname)
                        raw_bytes = None
                        rendered_jpg = None
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

                for df in discord_files:
                    try:
                        df.close()
                    except Exception:
                        pass
                discord_files.clear()
            finally:
                await _cleanup_status()
                for page_cv in cv_bubbles_per_page:
                    _release_cv_page_masks(page_cv)
                cv_bubbles_per_page.clear()
                pages_input.clear()
                release_unused_memory()