import os
import io
import re
import json
import time
import base64
import asyncio
from urllib.parse import quote_plus

import aiohttp
import discord
import functools
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

from config import (
    HTTPS,
    CF_ACCOUNT_ID,
    CF_API_TOKEN,
    CF_IMAGE_MODEL,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    GROQ_CHAT_FALLBACK_MODEL,
    ACTIVE_FAST_MODEL,
    THAI_CHAR_REGEX,
    CAT_TOPIC_REGEX,
    FONT_CANDIDATES,
    IMAGE_EXTENSIONS,
    state,
    clean_url,
    api_log,
    get_http_session,
    release_unused_memory,
)
from llm import (
    call_gemini_generate_content_self_healing,
    call_gemma_ai_studio_chat,
    call_gemma_ai_studio_vision,
    call_cloudflare_gemma_chat,
    call_modal_ollama_chat,
    call_groq_chat,
    call_cloudflare_vision_chat,
    discover_active_groq_vision_models,
)

LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
)


def split_discord_message_smart(text: str, max_len: int = 1900) -> list[str]:
    if len(text) <= 1990:
        return [text]

    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_len = 0
    in_code_block = False
    code_fence_header = "```"

    for line in lines:
        stripped = line.strip()
        if len(line) > max_len:
            if current_lines:
                if in_code_block:
                    current_lines.append("```")
                chunks.append("\n".join(current_lines))
                current_lines = [code_fence_header] if in_code_block else []
                current_len = len(code_fence_header) + 1 if in_code_block else 0

            while len(line) > max_len:
                split_at = line.rfind(" ", 0, max_len)
                if split_at == -1:
                    split_at = max_len
                part = line[:split_at]
                if in_code_block:
                    chunks.append(f"{code_fence_header}\n{part}\n```")
                else:
                    chunks.append(part)
                line = line[split_at:].lstrip()

        add_len = len(line) + 1
        if current_len + add_len > max_len and current_lines:
            if in_code_block:
                current_lines.append("```")
            chunks.append("\n".join(current_lines))
            current_lines = [code_fence_header] if in_code_block else []
            current_len = (len(code_fence_header) + 1) if in_code_block else 0

        current_lines.append(line)
        current_len += add_len

        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_fence_header = stripped
            else:
                in_code_block = False
                code_fence_header = "```"

    if current_lines:
        if in_code_block:
            current_lines.append("```")
        chunks.append("\n".join(current_lines))

    return [c for c in chunks if c.strip()]


async def _send_or_reply(
    target_channel,
    reply_to: discord.Message | None,
    content: str = None,
    file: discord.File = None,
    files: list[discord.File] = None,
):
    kwargs = {"content": content}
    if files:
        kwargs["files"] = files
    elif file is not None:
        kwargs["file"] = file

    if reply_to is not None:
        try:
            return await reply_to.reply(**kwargs, mention_author=False)
        except Exception:
            pass
    return await target_channel.send(**kwargs)


async def send_chunked_response(
    ctx_or_channel, text: str, filename_fallback: str = "output.md", reply_to: discord.Message = None
):
    if not text or not text.strip():
        return
    target_channel = ctx_or_channel.channel if hasattr(ctx_or_channel, "channel") else ctx_or_channel
    if len(text) > 7500:
        await _send_or_reply(
            target_channel,
            reply_to,
            content="**เนื้อหายาวเกิน 7,500 ตัวอักษร ฉันแนบเป็นไฟล์ไว้ให้ด้านล่างนะคะ:**",
            file=discord.File(io.StringIO(text), filename=filename_fallback),
        )
        return
    chunks = split_discord_message_smart(text, max_len=1900)
    for idx, chunk in enumerate(chunks):
        await _send_or_reply(target_channel, reply_to if idx == 0 else None, content=chunk)


# ==============================================================================
# CLOUDFLARE WORKERS AI IMAGE GENERATION (100% FREE NEURONS TIER)
# ==============================================================================
def _clean_conversational_image_prefix(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(
        r"^(?:ยูกิโนะ|ยูกิน่อน|บอท)?\s*(?:ช่วย|ลอง|ช่วยลอง|อยากให้|ไหนลอง)?\s*"
        r"(?:วาดรูป|สร้างรูป|เจนรูป|ขอรูป|วาดภาพ|สร้างภาพ|เจนภาพ|"
        r"draw\s+(?:an?\s+|me\s+an?\s+)?(?:image|picture|illustration)\s+(?:of\s+)?|"
        r"generate\s+(?:an?\s+)?(?:image|picture)\s+(?:of\s+)?)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(
        r"\s*(?:ให้ดูหน่อย|ให้หน่อย|ดูหน่อย|ให้ที|ให้ดูที|ดูที|หน่อย)?\s*(?:นะคะ|นะครับ|ค่ะ|ครับ|นะ|จ้ะ|จ๊ะ|สิ)?\s*$",
        "",
        cleaned,
    ).strip()
    return cleaned or raw_text.strip()


async def translate_prompt_to_english_for_flux(raw_prompt: str, is_anime_style: bool = False) -> str:
    prefix = "2D anime illustration, " if is_anime_style else ""

    # Tier 1: Keyless Google Translate NMT
    t0 = time.monotonic()
    try:
        gtrans_host = "translate." + "googleapis.com"
        gt_url = (
            f"{HTTPS}{gtrans_host}/translate_a/single"
            f"?client=gtx&sl=th&tl=en&dt=t&q={quote_plus(raw_prompt)}"
        )
        session = get_http_session()
        async with session.get(
            gt_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status == 200:
                gt_data = await resp.json(content_type=None)
                translated_segments = [seg[0] for seg in (gt_data[0] or []) if seg and seg[0]]
                en_literal = "".join(translated_segments).strip()
                if en_literal and not THAI_CHAR_REGEX.search(en_literal):
                    api_log(f"Prompt Translate (Tier 1 Google NMT) -> 200 OK ({time.monotonic() - t0:.2f}s)")
                    return f"{prefix}{en_literal}"
    except Exception as e:
        api_log(f"Prompt Translate (Tier 1 Google NMT) -> failed ({type(e).__name__}); fallback to Tier 2")

    strict_sys_msg = (
        "You are a strict, literal Thai-to-English translator. "
        "Translate the user's Thai text into English accurately and faithfully.\n"
        "CRITICAL RULES:\n"
        "1. Translate ONLY the words present in the original Thai text.\n"
        "2. Do NOT add any extra words, adjectives, lighting, furniture, or background details.\n"
        "3. Do NOT change colors or subjects (for example, 'แมวส้ม' means 'orange cat').\n"
        "4. Output ONLY the direct English translation in plain text, with no quotes or explanations."
    )
    trans_messages = [
        {"role": "system", "content": strict_sys_msg},
        {"role": "user", "content": raw_prompt},
    ]

    # Tier 2: Google AI Studio Gemma 4 (14.4K RPD) & Cloudflare Gemma 4 (10K Neurons)
    for gemma_fn, p_label in (
        (call_gemma_ai_studio_chat, "Prompt Translate Tier 2 AI Studio Gemma 4"),
        (call_cloudflare_gemma_chat, "Prompt Translate Tier 2 CF Gemma 4"),
    ):
        try:
            res = await gemma_fn(trans_messages, temperature=0.0, purpose=p_label)
            if res and not THAI_CHAR_REGEX.search(res):
                clean_res = res.strip(' "\'\n\r\t')
                return f"{prefix}{clean_res}"
        except Exception:
            pass

    # Tier 3: Modal GPU Ollama
    try:
        res = await call_modal_ollama_chat(
            trans_messages, ACTIVE_FAST_MODEL, 0.0, purpose="Prompt Translate Tier 3"
        )
        if res and not THAI_CHAR_REGEX.search(res):
            clean_res = res.strip(' "\'\n\r\t')
            return f"{prefix}{clean_res}"
    except Exception:
        pass

    # Tier 4: Groq LPU
    if GROQ_API_KEY:
        try:
            res = await call_groq_chat(
                trans_messages, GROQ_CHAT_FALLBACK_MODEL, 0.0, is_coding=False, purpose="Prompt Translate Tier 4"
            )
            if res and not THAI_CHAR_REGEX.search(res):
                clean_res = res.strip(' "\'\n\r\t')
                return f"{prefix}{clean_res}"
        except Exception:
            pass

    return raw_prompt


async def generate_image_cloudflare_flux(user_prompt: str) -> tuple[bytes, str, str]:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError("ยังไม่ได้ตั้งค่า `CF_ACCOUNT_ID` และ `CF_API_TOKEN` ใน Render Environment ค่ะ")

    raw_p = user_prompt.strip()
    selected_model = CF_IMAGE_MODEL
    mode_tag = "FLUX.1-schnell (8 steps)"
    is_anime_style = False

    lower_p = raw_p.lower()
    if lower_p.startswith(("--hq ", "--phoenix ")):
        selected_model = "@cf/leonardo/phoenix-1.0"
        mode_tag = "Leonardo Phoenix 1.0 (HQ)"
        raw_p = raw_p.split(" ", 1)[1].strip()
    elif lower_p.startswith("--fl2hq "):
        selected_model = "@cf/black-forest-labs/flux-2-klein-9b"
        mode_tag = "FLUX.2 Klein 9B"
        raw_p = raw_p.split(" ", 1)[1].strip()
    elif lower_p.startswith(("--fl2 ", "--flux2 ")):
        selected_model = "@cf/black-forest-labs/flux-2-klein-4b"
        mode_tag = "FLUX.2 Klein 4B"
        raw_p = raw_p.split(" ", 1)[1].strip()
    elif lower_p.startswith("--anime "):
        selected_model = "@cf/black-forest-labs/flux-1-schnell"
        mode_tag = "FLUX.1-schnell (Anime Style)"
        is_anime_style = True
        raw_p = raw_p.split(" ", 1)[1].strip()

    cleaned_subject = _clean_conversational_image_prefix(raw_p)
    english_prompt = cleaned_subject
    if THAI_CHAR_REGEX.search(cleaned_subject) or is_anime_style:
        english_prompt = await translate_prompt_to_english_for_flux(cleaned_subject, is_anime_style=is_anime_style)

    cf_host = "api." + "cloudflare.com"
    cf_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{selected_model}"
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}

    t0 = time.monotonic()
    session = get_http_session()
    req_timeout = aiohttp.ClientTimeout(total=60)
    if "flux-2-" in selected_model:
        form = aiohttp.FormData()
        form.add_field("prompt", english_prompt)
        form.add_field("width", "1024")
        form.add_field("height", "1024")
        req_ctx = session.post(cf_url, headers=headers, data=form, timeout=req_timeout)
    elif "phoenix" in selected_model:
        headers["Content-Type"] = "application/json"
        payload = {
            "prompt": english_prompt,
            "negative_prompt": "blurry, low quality, deformed hands, extra fingers, watermark, bad anatomy",
            "num_steps": 25,
            "guidance": 3.5,
            "width": 1024,
            "height": 1024,
        }
        req_ctx = session.post(cf_url, headers=headers, json=payload, timeout=req_timeout)
    else:
        headers["Content-Type"] = "application/json"
        payload = {"prompt": english_prompt, "steps": 8}
        req_ctx = session.post(cf_url, headers=headers, json=payload, timeout=req_timeout)

    async with req_ctx as resp:
        elapsed = time.monotonic() - t0
        if resp.status != 200:
            err_txt = await resp.text()
            api_log(f"Cloudflare Image -> {selected_model} HTTP {resp.status} ({elapsed:.2f}s)")
            raise RuntimeError(f"Cloudflare Workers AI HTTP {resp.status}: {err_txt[:200]}")

        api_log(f"Cloudflare Image -> {selected_model} 200 OK ({elapsed:.2f}s)")
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "image/" in content_type or "octet-stream" in content_type:
            return await resp.read(), f"[{mode_tag}] {english_prompt}", cleaned_subject

        data = await resp.json()
        b64_img = (data.get("result") or {}).get("image")
        if not b64_img:
            raise RuntimeError("Cloudflare Workers AI did not return an image payload.")
        return base64.b64decode(b64_img), f"[{mode_tag}] {english_prompt}", cleaned_subject


def build_yukino_image_caption(subject_text: str, formatted_en_prompt: str) -> str:
    combined = f"{subject_text} {formatted_en_prompt}"
    if CAT_TOPIC_REGEX.search(combined):
        is_panda = bool(re.search(r"(แพนซัง|แพนด้า|pan-san|pansan|panda)", combined, re.IGNORECASE))
        fav_item = "แพนซัง" if is_panda else "แมว"
        return (
            f"ฉ-ฉันไม่ได้ตั้งใจวาดเพราะชอบ{fav_item}เป็นพิเศษหรอกนะคะ! "
            f"ก-ก็แค่นายยืนกรานขอร้องมา ฉันเลยยอมทำให้ตามหน้าที่ของชมรมบริการเท่านั้นแหละค่ะ (`{subject_text}`)\n"
            f"-# Prompt: *{formatted_en_prompt[:500]}*"
        )
    return (
        f"ฉันสร้างภาพ `{subject_text}` ออกมาให้ตามที่นายต้องการแล้วค่ะ ลองตรวจสอบดูสิคะ\n"
        f"-# Prompt: *{formatted_en_prompt[:500]}*"
    )


async def send_generated_image_to_channel(channel, prompt_text: str, reply_to: discord.Message = None):
    try:
        img_bytes, en_prompt_tag, clean_subj = await generate_image_cloudflare_flux(prompt_text)
        caption = build_yukino_image_caption(clean_subj, en_prompt_tag)
        await _send_or_reply(
            channel,
            reply_to,
            content=caption,
            file=discord.File(io.BytesIO(img_bytes), filename="yukinon_flux.png"),
        )
    except Exception as e:
        await _send_or_reply(channel, reply_to, content=f"ไม่สามารถสร้างรูปภาพได้ในขณะนี้ค่ะ (`{e}`)")


# ==============================================================================
# IN-CONTEXT MANGA / IMAGE TRANSLATION & SMART SPEECH-BUBBLE TYPESETTING ENGINE
# ==============================================================================
@functools.lru_cache(maxsize=64)
def _load_manga_font(size: int, is_thai: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_CANDIDATES if is_thai else LATIN_FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _detect_target_translation_lang(user_prompt: str) -> tuple[str, str]:
    lower_p = user_prompt.lower()
    if any(k in lower_p for k in ("english", "eng", "อังกฤษ", "เป็นอิ้ง", "แปลอิ้ง", " to en")):
        return "en", "English"
    if any(k in lower_p for k in ("thai", "ไทย", "เป็นไทย", "แปลไทย", " to th")):
        return "th", "Thai"
    if THAI_CHAR_REGEX.search(user_prompt):
        return "th", "Thai"
    return "en", "English"


def _prepare_manga_image_for_vision(raw_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalize_and_validate_bubbles(raw_bubbles: list) -> list[dict]:
    valid = []
    for idx, item in enumerate(raw_bubbles, 1):
        if not isinstance(item, dict):
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
        tr_txt = re.sub(r"\s+", " ", str(tr_txt)).strip()
        # Remove any accidental space after hyphenated Japanese honorifics (e.g., 'Melt- san?' -> 'Melt-san?')
        tr_txt = re.sub(
            r"-\s+(san|kun|chan|sama|sensei|senpai|dono|shi)\b",
            r"-\1",
            tr_txt,
            flags=re.IGNORECASE,
        )
        src_txt = re.sub(r"\s+", " ", str(src_txt)).strip()
        if not tr_txt:
            continue

        box = item.get("box_2d") or item.get("bbox") or item.get("box")
        norm_box = None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                norm_box = [float(v) for v in box]
            except Exception:
                norm_box = None
        elif isinstance(box, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", box)
            if len(nums) == 4:
                norm_box = [float(v) for v in nums]

        valid.append({
            "id": item.get("id", idx),
            "speaker_or_note": str(speaker).strip(),
            "source_text": src_txt,
            "translated_text": tr_txt,
            "box_2d": norm_box,
        })
    return valid


def _parse_manga_pages_batch_json(
    raw_text: str, expected_pages: int
) -> list[tuple[list[dict], str]]:
    """
    Parses a multi-page batch JSON response:
    {"pages": [{"page_index": 1, "summary": "...", "bubbles": [...]}, ...]}
    Also seamlessly handles single-page {"summary": "...", "bubbles": [...]} responses.
    """
    if not raw_text:
        return [([], "") for _ in range(expected_pages)]

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    target = json_match.group(0) if json_match else cleaned

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

    if isinstance(parsed, dict):
        # Case A: Multi-page "pages" array returned
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
            if any(b for b, _ in results):
                return results

        # Case B: Single-page {"bubbles": [...]} format returned
        if expected_pages == 1 and "bubbles" in parsed:
            bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
            summary = str(parsed.get("summary") or "").strip()
            return [(bubbles, summary)]

    # Fallback for 1-page regex extraction
    if expected_pages == 1:
        b, s = _parse_manga_bubbles_json(raw_text)
        return [(b, s)]

    return [([], "") for _ in range(expected_pages)]


async def extract_and_translate_manga_pages_batch(
    pages_input: list[tuple[bytes, str]], target_lang_name: str
) -> list[tuple[list[dict], str]]:
    """
    Translates 1 to 10 manga pages in a SINGLE multimodal API request to conserve RPD quota.
    pages_input: list of (raw_bytes, filename)
    Returns: list of (bubbles, scene_summary) corresponding to each page in order.
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return []

    b64_pages = []
    for raw_b, fname in pages_input:
        b64_str = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_b)
        b64_pages.append((b64_str, fname))

    mime_type = "image/jpeg"

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
Analyze all pages in order so character names, pronouns, tone, and ongoing dialogue stay consistent across pages,
and translate ALL speech bubbles on every page into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. Reading Order: Within each page, order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. In-Context Localization & Honorifics:
   - Adapt idioms, pronouns, tone, and emotion so the dialogue sounds natural in {target_lang_name} while fitting inside a speech bubble. Do NOT use double quotes inside `translated_text` (use single quotes if needed).
   - When translating to English, ALWAYS preserve Japanese honorifics attached directly with a hyphen and NO space (translate さん as `-san`, くん/君 as `-kun`, ちゃん as `-chan`, 様/さま as `-sama`, 先生 as `-sensei`, 先輩 as `-senpai` — e.g., 'Melt-san', NEVER 'Melt- san' or 'Miss Melt').
3. Speech Bubble Bounding Boxes (`box_2d`):
   - For each speech bubble on a page, provide `box_2d` as `[ymin, xmin, ymax, xmax]` normalized from 0 to 1000 relative to THAT page's dimensions.
   - `box_2d` must tightly cover the Japanese text block inside that specific bubble lobe ( NEVER place `box_2d` on a character's face/hair outside the bubble, and NEVER merge two separate bubbles into one `box_2d`).
4. Return ONLY valid JSON matching this exact structure for all {num_pages} page(s):
{{
  "pages": [
    {{
      "page_index": 1,
      "summary": "Brief 1-sentence scene context for Page 1",
      "bubbles": [
        {{
          "id": 1,
          "speaker_or_note": "Character / SFX / Narration",
          "source_text": "Original Japanese text",
          "translated_text": "Natural {target_lang_name} translation",
          "box_2d": [ymin, xmin, ymax, xmax]
        }}
      ]
    }}
  ]
}}
""".strip()

    # Tier 1: Google AI Studio (gemini-3-flash-preview) — All pages in 1 single RPD request!
    if GEMINI_API_KEY:
        parts = [{"text": manga_batch_instruction}]
        for idx, (b64_img, fname) in enumerate(b64_pages, 1):
            parts.append({"text": f"=== PAGE {idx} ({fname}) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        gemini_payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 512},
            },
        }
        # Scale timeout with page count (e.g. 30s for 1 page, up to 90s for 10 pages)
        timeout_sec = min(90, 22 + (8 * num_pages))
        try:
            data = await call_gemini_generate_content_self_healing(
                gemini_payload, timeout_sec=timeout_sec, purpose=f"MangaBatch({num_pages}p)"
            )
            if data:
                resp_parts = (
                    ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                )
                raw_json_str = "\n".join(p.get("text", "") for p in resp_parts if p.get("text")).strip()
                batch_results = _parse_manga_pages_batch_json(raw_json_str, num_pages)
                if any(bubbles for bubbles, _ in batch_results):
                    total_b = sum(len(b) for b, _ in batch_results)
                    print(
                        f"[Manga Vision Batch] Completed {num_pages} page(s) in 1 request via Tier 1 Gemini "
                        f"({state.active_gemini_model}) - {total_b} total bubbles",
                        flush=True,
                    )
                    return batch_results
        except Exception as e:
            print(f"Gemini Manga Batch Vision warning: {e}", flush=True)

    # Fallback: If Tier 1 Gemini is unavailable/quota-exhausted, process per-page via Tier 2/3 (Cloudflare / Groq)
    fallback_results = []
    for raw_b, fname in pages_input:
        b, s = await extract_and_translate_manga_bubbles(raw_b, fname, target_lang_name)
        fallback_results.append((b, s))
    return fallback_results


def _parse_manga_bubbles_json(raw_text: str) -> tuple[list[dict], str]:
    if not raw_text:
        return [], ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    target = json_match.group(0) if json_match else cleaned

    # Stage 1: Standard json.loads
    try:
        parsed = json.loads(target)
        bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
        summary = str(parsed.get("summary") or "").strip()
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    # Stage 2: Repair common LLM JSON syntax errors
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

    # Stage 3: Regex fallback
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
        box_m = re.search(r'"(?:box_2d|bbox|box)"\s*:\s*\[([^\]]+)\]', block)
        norm_box = None
        if box_m:
            nums = re.findall(r"-?\d+(?:\.\d+)?", box_m.group(1))
            if len(nums) == 4:
                norm_box = [float(v) for v in nums]

        raw_tr_val = tr_m.group(1).replace('\\"', '"').strip()
        extracted_bubbles.append({
            "id": idx,
            "speaker_or_note": spk_m.group(1).strip() if spk_m else f"Bubble {idx}",
            "source_text": src_m.group(1).strip() if src_m else "",
            "translated_text": raw_tr_val,
            "box_2d": norm_box,
        })

    valid_extracted = _normalize_and_validate_bubbles(extracted_bubbles)
    return valid_extracted, summary


async def extract_and_translate_manga_bubbles(
    raw_bytes: bytes, filename: str, target_lang_name: str
) -> tuple[list[dict], str]:
    b64_img = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_bytes)
    mime_type = "image/jpeg"

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
Analyze the entire image visually (character expressions, action, speaker relationships, and panel layout)
and translate ALL speech bubbles (including short exclamations like '!' or 'Huh?') into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. Reading Order: Within each page, order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. In-Context Localization & Honorifics:
   - Adapt idioms, pronouns, tone, and emotion so the dialogue sounds natural in {target_lang_name} while fitting inside a speech bubble. Do NOT use double quotes inside `translated_text` (use single quotes if needed).
   - When translating to English, ALWAYS preserve Japanese honorifics attached directly with a hyphen and NO space (translate さん as `-san`, くん/君 as `-kun`, ちゃん as `-chan`, 様/さま as `-sama`, 先生 as `-sensei`, 先輩 as `-senpai` — e.g., 'Melt-san', NEVER 'Melt- san' or 'Miss Melt').
3. Speech Bubble Bounding Boxes (`box_2d`):
   - For each speech bubble on a page, provide `box_2d` as `[ymin, xmin, ymax, xmax]` normalized from 0 to 1000 relative to THAT page's dimensions.
   - `box_2d` must tightly cover the Japanese text block inside that specific bubble lobe ( NEVER place `box_2d` on a character's face/hair outside the bubble, and NEVER merge two separate bubbles into one `box_2d`).
4. Return ONLY valid JSON matching this exact structure (every bubble MUST have a non-empty `translated_text`):
{{
  "summary": "Brief 1-sentence scene context",
  "bubbles": [
    {{
      "id": 1,
      "speaker_or_note": "Character / SFX / Narration",
      "source_text": "Original Japanese text",
      "translated_text": "Natural {target_lang_name} translation",
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ]
}}
""".strip()

    fallback_bubbles_no_boxes = []
    fallback_summary = ""

    # Tier 1: Google AI Studio (gemini-3-flash-preview)
    if GEMINI_API_KEY:
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
                "thinkingConfig": {"thinkingBudget": 256},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(gemini_payload, timeout_sec=25)
            if data:
                parts = (
                    ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                )
                raw_json_str = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
                bubbles, summary = _parse_manga_bubbles_json(raw_json_str)
                if bubbles:
                    print(
                        f"[Manga Vision] Completed via Tier 1 Gemini ({state.active_gemini_model}) "
                        f"- {len(bubbles)} bubbles"
                    )
                    return bubbles, summary
        except Exception as e:
            print(f"Gemini Manga Vision warning: {e}")

    # Tier 2: Google AI Studio Gemma 4 Vision (gemma-4-26b-a4b-it — 14.4K RPD backup when Gemini 3 Flash 20 RPD is exhausted)
    try:
        gemma_content = await call_gemma_ai_studio_vision(
            manga_system_instruction,
            mime_type,
            b64_img,
            response_mime_type="application/json",
            timeout_sec=30,
            purpose="MangaSingle",
        )
        if gemma_content:
            bubbles, summary = _parse_manga_bubbles_json(gemma_content)
            if bubbles:
                has_boxes_and_text = any(
                    isinstance(b.get("box_2d"), list)
                    and len(b["box_2d"]) == 4
                    and bool(b.get("translated_text"))
                    for b in bubbles
                )
                if has_boxes_and_text:
                    print(f"[Manga Vision] Completed via Tier 2 AI Studio Gemma 4 - {len(bubbles)} bubbles", flush=True)
                    return bubbles, summary
                fallback_bubbles_no_boxes = bubbles
                fallback_summary = summary
    except Exception as e:
        print(f"AI Studio Gemma 4 Manga Vision warning: {e}", flush=True)

    # Tier 3: Cloudflare Workers AI (@cf/google/gemma-4-26b-a4b-it)
    try:
        cf_content = await call_cloudflare_vision_chat(manga_system_instruction, mime_type, b64_img)
        if cf_content:
            bubbles, summary = _parse_manga_bubbles_json(cf_content)
            if bubbles:
                has_boxes_and_text = any(
                    isinstance(b.get("box_2d"), list)
                    and len(b["box_2d"]) == 4
                    and bool(b.get("translated_text"))
                    for b in bubbles
                )
                if has_boxes_and_text:
                    print(f"[Manga Vision] Completed via Tier 3 Cloudflare Gemma 4 - {len(bubbles)} bubbles")
                    return bubbles, summary
                fallback_bubbles_no_boxes = bubbles
                fallback_summary = summary
    except Exception as e:
        print(f"Cloudflare Gemma 4 Manga Vision warning: {e}")

    # Tier 4: Groq Vision (qwen/qwen3.8-27b)
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
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        bubbles, summary = _parse_manga_bubbles_json(content)
                        if bubbles:
                            has_boxes_and_text = any(
                                isinstance(b.get("box_2d"), list)
                                and len(b["box_2d"]) == 4
                                and bool(b.get("translated_text"))
                                for b in bubbles
                            )
                            if has_boxes_and_text:
                                print(f"[Manga Vision] Completed via Tier 4 Groq ({g_model}) - {len(bubbles)} bubbles")
                                return bubbles, summary
                            if not fallback_bubbles_no_boxes:
                                fallback_bubbles_no_boxes = bubbles
                                fallback_summary = summary
                    else:
                        err_txt = await resp.text()
                        print(f"Groq Manga Vision ({g_model}) HTTP {resp.status}: {err_txt[:150]}")
            except Exception as e:
                print(f"Groq Manga Vision ({g_model}) warning: {e}")

    if fallback_bubbles_no_boxes:
        return fallback_bubbles_no_boxes, fallback_summary

    return [], ""


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
            else:
                chunk = ""
                for ch in word:
                    test_ch = chunk + ch
                    cbbox = draw.textbbox((0, 0), test_ch, font=font)
                    is_combining = "\u0E31" <= ch <= "\u0E3A" or "\u0E47" <= ch <= "\u0E4E"
                    if (cbbox[2] - cbbox[0]) > max_width and chunk and not is_combining:
                        lines.append(chunk)
                        chunk = ch
                    else:
                        chunk = test_ch
                current = chunk
    if current:
        lines.append(current)
    return lines


def _detect_bubble_polarity(
    base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> bool:
    """
    Returns True if the bubble at (x1, y1, x2, y2) is a Dark/Black bubble (white text on black fill),
    or False if it is a Light/White bubble.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw < 10 or bh < 10:
        return False
    ix1, iy1 = x1 + 2, y1 + 2
    ix2, iy2 = x2 - 2, y2 - 2
    sub = base_gray.crop((ix1, iy1, ix2, iy2))
    pixels = sorted(sub.getdata())
    median_lum = pixels[len(pixels) // 2] if pixels else 255
    return median_lum < 115


def _compute_voronoi_lobe_bounds(
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    other_peer_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    box_w = max(10, x2 - x1)
    box_h = max(10, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Generous reach in unobstructed directions so tall bubbles ('…………', '私は別に――')
    # and SFX-displaced boxes are 100% contained inside the window, while still clamping
    # strictly at the midpoint between neighboring speech bubbles.
    vor_x1 = max(0, x1 - max(110, int(box_w * 1.40)))
    vor_y1 = max(0, y1 - max(115, int(box_h * 1.60)))
    vor_x2 = min(width, x2 + max(110, int(box_w * 1.40)))
    vor_y2 = min(height, y2 + max(115, int(box_h * 1.60)))

    for ox1, oy1, ox2, oy2 in other_peer_boxes:
        ocx = (ox1 + ox2) // 2
        ocy = (oy1 + oy2) // 2
        dx = abs(cx - ocx)
        dy = abs(cy - ocy)
        if dx > (box_w + (ox2 - ox1)) * 1.35 and dy > (box_h + (oy2 - oy1)) * 1.35:
            continue

        h_gap = max(ox1 - x2, x1 - ox2)
        v_gap = max(oy1 - y2, y1 - oy2)
        if v_gap >= h_gap:
            if cy < ocy:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y2 + oy1) // 2
                vor_y2 = min(vor_y2, max(y1 + 10, mid_y - 2))
            else:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y1 + oy2) // 2
                vor_y1 = max(vor_y1, min(y2 - 10, mid_y + 2))
        else:
            if cx < ocx:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x2 + ox1) // 2
                vor_x2 = min(vor_x2, max(x1 + 10, mid_x - 2))
            else:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x1 + ox2) // 2
                vor_x1 = max(vor_x1, min(x2 - 10, mid_x + 2))

    return vor_x1, vor_y1, vor_x2, vor_y2


def _find_best_enclosed_white_bubble(
    sub_gray: Image.Image,
    rx1_rel: int,
    ry1_rel: int,
    rx2_rel: int,
    ry2_rel: int,
    has_kanji_kana: bool = True,
) -> Image.Image | None:
    """
    Fast 0.5x-scale enclosed bubble finder:
    1) Runs flood-fill candidate discovery at 0.5x resolution (4x fewer pixels -> ~15x faster on Render CPU!).
    2) Tests both standard white (p >= 232) AND 2px-pinched white (seals thin white SFX halos like 'リンゴーン'
       on the left border of '私は別に――').
    3) Rejects empty white cheeks/faces by requiring `holes_area` (enclosed dark character islands inside the
       white bubble) to be >= 22 at half-scale (~88px full-scale) whenever `has_kanji_kana=True`.
    """
    ww, wh = sub_gray.size
    if ww < 16 or wh < 16:
        return None

    # Downscale by 0.5x so pure-Python ImageDraw.floodfill runs in <1ms
    sw = max(8, ww // 2)
    sh = max(8, wh // 2)
    small_gray = sub_gray.resize((sw, sh), Image.Resampling.BILINEAR)

    s_rx1 = max(1, rx1_rel // 2)
    s_ry1 = max(1, ry1_rel // 2)
    s_rx2 = min(sw - 2, max(s_rx1 + 4, rx2_rel // 2))
    s_ry2 = min(sh - 2, max(s_ry1 + 4, ry2_rel // 2))
    s_bw = max(4, s_rx2 - s_rx1)
    s_bh = max(4, s_ry2 - s_ry1)

    # Candidate seed points at 0.5x scale (inside box + horizontal probes left/right for SFX-shifted boxes)
    seed_candidates = []
    for fy in (0.50, 0.25, 0.75):
        sy = max(1, min(sh - 2, int(s_ry1 + s_bh * fy)))
        for fx in (0.50, 0.25, 0.75):
            sx = max(1, min(sw - 2, int(s_rx1 + s_bw * fx)))
            seed_candidates.append((sx, sy))
        for extra_dx in (-10, 10, -22, 22, -34, 34, -46, 46):
            sx_ext = (s_rx1 + extra_dx) if extra_dx < 0 else (s_rx2 + extra_dx)
            if 1 <= sx_ext < sw - 1:
                seed_candidates.append((sx_ext, sy))

    base_bin = small_gray.point(lambda p: 255 if p >= 232 else 0)
    # Pass 2 erodes white by 1px at 0.5x scale (= 2px at full scale) to pinch shut thin white SFX halos like 'リンゴーン'
    pinched_bin = base_bin.filter(ImageFilter.MinFilter(3))

    best_small_filled = None
    best_was_pinched = False
    best_score = -1.0

    for pass_idx, bin_img in enumerate((base_bin, pinched_bin)):
        visited = Image.new("L", (sw, sh), 0)
        for sx, sy in seed_candidates:
            if bin_img.getpixel((sx, sy)) != 255 or visited.getpixel((sx, sy)) != 0:
                continue

            work = bin_img.copy()
            ImageDraw.floodfill(work, (sx, sy), 128)
            comp = work.point(lambda p: 255 if p == 128 else 0)
            visited = ImageChops.lighter(visited, comp)

            c_bbox = comp.getbbox()
            if not c_bbox:
                continue
            cx1, cy1, cx2, cy2 = c_bbox
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw < 9 or ch < 9:
                continue

            # Reject open background/speedlines that span across opposite window boundaries
            edge_touches = (
                (1 if cx1 <= 1 else 0)
                + (1 if cy1 <= 1 else 0)
                + (1 if cx2 >= sw - 1 else 0)
                + (1 if cy2 >= sh - 1 else 0)
            )
            if edge_touches >= 3 or (cx1 <= 1 and cx2 >= sw - 1) or (cy1 <= 1 and cy2 >= sh - 1):
                continue

            # Find enclosed dark character holes inside `comp` using a 2px exterior moat
            padded = Image.new("L", (sw + 4, sh + 4), 0)
            padded.paste(comp, (2, 2))
            ImageDraw.floodfill(padded, (0, 0), 128)
            holes = padded.crop((2, 2, sw + 2, sh + 2)).point(lambda p: 255 if p == 0 else 0)
            filled = ImageChops.lighter(comp, holes)

            comp_area = comp.crop(c_bbox).histogram()[255]
            filled_area = max(1, filled.crop(c_bbox).histogram()[255])
            holes_area = holes.crop(c_bbox).histogram()[255]

            if comp_area < 55 or (comp_area / filled_area) < 0.48:
                continue

            # CRITICAL: An enclosed white cheek has almost ZERO enclosed dark holes (holes_area < 8),
            # whereas a real speech bubble with kanji/kana ('私は別に――') has holes_area >= 18 at 0.5x scale!
            if has_kanji_kana and holes_area < 16:
                continue

            ov_x = max(0, min(cx2, s_rx2) - max(cx1, s_rx1))
            ov_y = max(0, min(cy2, s_ry2) - max(cy1, s_ry1))
            overlap_area = ov_x * ov_y

            # Score heavily by enclosed character holes (`holes_area`) + box overlap
            score = (overlap_area * 2.5) + (holes_area * 10.0) + (filled_area * 0.25)
            if pass_idx == 1:
                score *= 0.92  # Prefer un-pinched pass when both find the same bubble
            if score > best_score:
                best_score = score
                best_small_filled = filled
                best_was_pinched = (pass_idx == 1)

        # If Pass 1 already found a high-confidence enclosed bubble with text holes, skip Pass 2
        if best_small_filled is not None and best_score > 450:
            break

    if best_small_filled is None:
        return None

    if best_was_pinched:
        best_small_filled = best_small_filled.filter(ImageFilter.MaxFilter(3))

    # Upscale the winning bubble mask back to full resolution (ww, wh) and smooth edges
    full_filled = best_small_filled.resize((ww, wh), Image.Resampling.BILINEAR).point(
        lambda p: 255 if p >= 128 else 0
    )
    return full_filled


def _erase_bubble_japanese_text_and_measure_lobe(
    img: Image.Image,
    base_gray: Image.Image,
    other_peer_boxes: list[tuple[int, int, int, int]],
    raw_x1: int,
    raw_y1: int,
    raw_x2: int,
    raw_y2: int,
    is_dark_bubble: bool = False,
    has_kanji_kana: bool = True,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int]]:
    width, height = img.size
    x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2

    vor_x1, vor_y1, vor_x2, vor_y2 = _compute_voronoi_lobe_bounds(
        width, height, x1, y1, x2, y2, other_peer_boxes
    )

    wx1, wy1, wx2, wy2 = vor_x1, vor_y1, vor_x2, vor_y2
    ww = max(8, wx2 - wx1)
    wh = max(8, wy2 - wy1)
    sub_gray = base_gray.crop((wx1, wy1, wx2, wy2))

    rx1_rel = max(2, x1 - wx1)
    ry1_rel = max(2, y1 - wy1)
    rx2_rel = min(ww - 2, x2 - wx1)
    ry2_rel = min(wh - 2, y2 - wy1)
    rel_cx = (rx1_rel + rx2_rel) // 2

    # =========================================================================
    # CASE A: DARK / BLACK BUBBLE (White text on Black/Dark fill — Never enlarges bubble!)
    # =========================================================================
    if is_dark_bubble:
        core_crop = base_gray.crop((max(0, raw_x1), max(0, raw_y1), min(width, raw_x2), min(height, raw_y2)))
        core_pixels = sorted(core_crop.getdata())
        bg_shade = int(core_pixels[len(core_pixels) // 10]) if core_pixels else 10
        bg_shade = max(4, min(38, bg_shade))
        dark_fill_rgb = (bg_shade, bg_shade, bg_shade)
        fg_text_rgb = (250, 250, 250)
        stroke_rgb = (10, 10, 10)

        closed_dark_bg = sub_gray.filter(ImageFilter.MinFilter(9)).filter(ImageFilter.MaxFilter(11))
        strict_dark_silhouette = closed_dark_bg.point(lambda p: 255 if p <= 68 else 0)
        ImageDraw.Draw(strict_dark_silhouette).rectangle([0, 0, ww - 1, wh - 1], outline=0)

        seeded_dark = 0
        for frac_y in (0.18, 0.34, 0.50, 0.66, 0.82):
            sy = int(ry1_rel + (ry2_rel - ry1_rel) * frac_y)
            for frac_x in (0.50, 0.30, 0.70, 0.20, 0.80):
                sx = int(rx1_rel + (rx2_rel - rx1_rel) * frac_x)
                if 2 <= sx < ww - 2 and 2 <= sy < wh - 2 and strict_dark_silhouette.getpixel((sx, sy)) == 255:
                    ImageDraw.floodfill(strict_dark_silhouette, (sx, sy), 128)
                    seeded_dark += 1

        if seeded_dark > 0:
            verified_dark_interior = strict_dark_silhouette.point(lambda p: 255 if p == 128 else 0)
        else:
            verified_dark_interior = strict_dark_silhouette

        tx_clean_x1 = max(2, rx1_rel - 8)
        tx_clean_y1 = max(2, ry1_rel - 8)
        tx_clean_x2 = min(ww - 2, rx2_rel + 8)
        tx_clean_y2 = min(wh - 2, ry2_rel + 8)

        text_box_bounds = Image.new("L", (ww, wh), 0)
        ImageDraw.Draw(text_box_bounds).rectangle(
            [tx_clean_x1, tx_clean_y1, tx_clean_x2, tx_clean_y2], fill=255
        )
        safe_dark_erase_zone = ImageChops.darker(verified_dark_interior, text_box_bounds)

        bright_ink = sub_gray.point(lambda p: 255 if p > (bg_shade + 8) else 0)
        erase_mask_dark = ImageChops.darker(
            bright_ink.filter(ImageFilter.MaxFilter(5)),
            safe_dark_erase_zone,
        )
        img.paste(Image.new("RGB", (ww, wh), dark_fill_rgb), (wx1, wy1), mask=erase_mask_dark)

        mid_y_start = ry1_rel + int((ry2_rel - ry1_rel) * 0.22)
        mid_y_end = ry1_rel + int((ry2_rel - ry1_rel) * 0.78)
        row_lefts = []
        row_rights = []
        for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), max(2, (mid_y_end - mid_y_start) // 6 or 1)):
            if not (0 <= sy < wh):
                continue
            lx = rel_cx
            while lx > 2 and verified_dark_interior.getpixel((lx - 1, sy)) > 128:
                lx -= 1
            rx = rel_cx
            while rx < ww - 2 and verified_dark_interior.getpixel((rx, sy)) > 128:
                rx += 1
            if rx - lx >= 16:
                row_lefts.append(lx)
                row_rights.append(rx)

        if row_lefts and row_rights:
            row_lefts.sort()
            row_rights.sort()
            inner_l = row_lefts[int(len(row_lefts) * 0.75)] + 6
            inner_r = row_rights[int(len(row_rights) * 0.25)] - 6
            if inner_r - inner_l >= 22:
                return (
                    wx1 + inner_l,
                    wy1 + ry1_rel,
                    wx1 + inner_r,
                    wy1 + ry2_rel,
                    fg_text_rgb,
                    stroke_rgb,
                )

        return (
            max(vor_x1 + 4, raw_x1),
            max(vor_y1 + 4, raw_y1),
            min(vor_x2 - 4, raw_x2),
            min(vor_y2 - 4, raw_y2),
            fg_text_rgb,
            stroke_rgb,
        )

    # =========================================================================
    # CASE B: ENCLOSED WHITE SPEECH BUBBLE (Fast 0.5x Finder + Full-Res Clean)
    # =========================================================================
    filled_bubble = _find_best_enclosed_white_bubble(
        sub_gray, rx1_rel, ry1_rel, rx2_rel, ry2_rel, has_kanji_kana=has_kanji_kana
    )

    if filled_bubble is not None:
        # Shrink inward by 2px (MinFilter(5)) so the black bubble outline is 100% preserved,
        # and erase all ink (including faint gray '……' up to p < 248) inside the bubble!
        safe_inner_core = filled_bubble.filter(ImageFilter.MinFilter(5))
        ink_inside_bubble = sub_gray.point(lambda p: 255 if p < 248 else 0)
        erase_mask = ImageChops.darker(
            ink_inside_bubble.filter(ImageFilter.MaxFilter(3)),
            safe_inner_core,
        )
        img.paste(Image.new("RGB", (ww, wh), (255, 255, 255)), (wx1, wy1), mask=erase_mask)

        lobe_bbox = safe_inner_core.getbbox()
        if lobe_bbox:
            lb_x1, lb_y1, lb_x2, lb_y2 = lobe_bbox
            lobe_cx = (lb_x1 + lb_x2) // 2
            rel_y1 = max(lb_y1 + 4, min(ry1_rel, lb_y1 + int((lb_y2 - lb_y1) * 0.12)))
            rel_y2 = min(lb_y2 - 4, max(ry2_rel, lb_y1 + int((lb_y2 - lb_y1) * 0.88)))
            if rel_y2 - rel_y1 < 18:
                rel_y1, rel_y2 = lb_y1 + 3, lb_y2 - 3

            mid_y_start = rel_y1 + int((rel_y2 - rel_y1) * 0.18)
            mid_y_end = rel_y1 + int((rel_y2 - rel_y1) * 0.82)
            row_lefts = []
            row_rights = []
            for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), max(2, (mid_y_end - mid_y_start) // 7 or 1)):
                if not (0 <= sy < wh):
                    continue
                lx = lobe_cx
                while lx > lb_x1 and safe_inner_core.getpixel((lx - 1, sy)) > 128:
                    lx -= 1
                rx = lobe_cx
                while rx < lb_x2 and safe_inner_core.getpixel((rx, sy)) > 128:
                    rx += 1
                if rx - lx >= 14:
                    row_lefts.append(lx)
                    row_rights.append(rx)

            if row_lefts and row_rights:
                row_lefts.sort()
                row_rights.sort()
                inner_l = row_lefts[int(len(row_lefts) * 0.65)] + 4
                inner_r = row_rights[int(len(row_rights) * 0.35)] - 4
                if inner_r - inner_l >= 18:
                    return (
                        wx1 + inner_l,
                        wy1 + rel_y1,
                        wx1 + inner_r,
                        wy1 + rel_y2,
                        (15, 15, 15),
                        (255, 255, 255),
                    )

            return (
                wx1 + lb_x1 + 4,
                wy1 + rel_y1,
                wx1 + lb_x2 - 4,
                wy1 + rel_y2,
                (15, 15, 15),
                (255, 255, 255),
            )

    # =========================================================================
    # CASE C: FAST C-FILTER FALLBACK FOR OPEN / SFX-BROKEN BUBBLES ('Hmph!' / '私は別に――')
    # =========================================================================
    # Uses compiled C-level BoxBlur + MaxFilter to identify dark text strokes sitting on pure white
    # bubble paper (excluding gray screentone hair/faces and thick dark SFX like 'リンゴーン').
    local_mean = sub_gray.filter(ImageFilter.BoxBlur(6))
    local_max = sub_gray.filter(ImageFilter.MaxFilter(5))
    white_paper_zone = ImageChops.darker(
        local_mean.point(lambda p: 255 if p >= 214 else 0),
        local_max.point(lambda p: 255 if p >= 246 else 0),
    ).filter(ImageFilter.MinFilter(3))

    # Exclude outer 2px window border
    ImageDraw.Draw(white_paper_zone).rectangle([0, 0, ww - 1, wh - 1], outline=0, width=2)

    dark_strokes = sub_gray.point(lambda p: 255 if p < 235 else 0)
    candidate_text_ink = ImageChops.darker(dark_strokes, white_paper_zone)

    ink_bbox = candidate_text_ink.getbbox()
    if ink_bbox:
        erase_c = ImageChops.darker(
            candidate_text_ink.filter(ImageFilter.MaxFilter(5)),
            white_paper_zone.filter(ImageFilter.MaxFilter(3)),
        )
        img.paste(Image.new("RGB", (ww, wh), (255, 255, 255)), (wx1, wy1), mask=erase_c)
        ix1, iy1, ix2, iy2 = ink_bbox
        if (ix2 - ix1) >= 12 and (iy2 - iy1) >= 12:
            pad_x = max(8, int((ix2 - ix1) * 0.25))
            return (
                max(vor_x1 + 4, wx1 + ix1 - pad_x),
                max(vor_y1 + 4, wy1 + iy1),
                min(vor_x2 - 4, wx1 + ix2 + pad_x),
                min(vor_y2 - 4, wy1 + iy2),
                (15, 15, 15),
                (255, 255, 255),
            )

    return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), (255, 255, 255)


def _resolve_box_collisions(
    renders: list[list],
    min_gap: int = 6,
) -> list[list]:
    """
    Guarantees that no two bubble layout boxes overlap in 2D space.
    """
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
    return renders


def render_translated_manga_image_sync(raw_bytes: bytes, bubbles: list[dict]) -> bytes | None:
    valid_boxes = [
        b for b in bubbles
        if isinstance(b.get("box_2d"), list)
        and len(b["box_2d"]) == 4
        and (b.get("translated_text") or "").strip()
    ]
    if not valid_boxes:
        return None

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    width, height = img.size

    # Step 1: Convert normalized box_2d to pixel coordinates, detect polarity, and check if source_text has kanji/kana
    pixel_boxes = []
    for b in valid_boxes:
        ymin, xmin, ymax, xmax = [float(v) for v in b["box_2d"]]
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        raw_x1 = max(0, int((xmin / 1000.0) * width))
        raw_y1 = max(0, int((ymin / 1000.0) * height))
        raw_x2 = min(width, int((xmax / 1000.0) * width))
        raw_y2 = min(height, int((ymax / 1000.0) * height))
        is_dark = _detect_bubble_polarity(base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        src_txt = str(b.get("source_text") or "")
        has_kanji_kana = bool(re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]", src_txt))
        clean_tr = re.sub(r"\s+", " ", b["translated_text"]).strip()
        pixel_boxes.append((raw_x1, raw_y1, raw_x2, raw_y2, is_dark, has_kanji_kana, clean_tr))

    # Step 2: Erase Japanese text (matching white or black bubble background) and measure each lobe
    prepared_renders = []
    for i, (rx1, ry1, rx2, ry2, is_dark, has_kk, trans_text) in enumerate(pixel_boxes):
        if (rx2 - rx1) < 3 or (ry2 - ry1) < 3:
            continue
        other_boxes = [
            (ox1, oy1, ox2, oy2)
            for j, (ox1, oy1, ox2, oy2, _, _, _) in enumerate(pixel_boxes)
            if j != i
        ]
        lx1, ly1, lx2, ly2, fg_text, stroke_col = _erase_bubble_japanese_text_and_measure_lobe(
            img,
            base_gray,
            other_boxes,
            rx1,
            ry1,
            rx2,
            ry2,
            is_dark_bubble=is_dark,
            has_kanji_kana=has_kk,
        )
        prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, trans_text])

    # Step 3: Resolve any 2D bounding-box collisions and typeset with a uniform baseline offset
    prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)

    draw = ImageDraw.Draw(img)
    for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text in prepared_renders:
        box_w = max(20, tx2 - tx1)
        box_h = max(16, ty2 - ty1)
        usable_w = max(18, int(box_w * 0.94))
        usable_h = max(14, int(box_h * 0.94))
        is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

        start_font_size = min(24, max(11, int(min(box_w, box_h) * 0.27)))
        chosen_font = _load_manga_font(8, is_thai=is_thai_text)
        chosen_lines = [trans_text]
        line_height = 10
        ref_top = 0

        sample_probe = "Agกข้ำ" if is_thai_text else "Agjpqy!"
        for fsize in range(start_font_size, 7, -1):
            font = _load_manga_font(fsize, is_thai=is_thai_text)
            wrapped = _wrap_text_to_pixel_width(
                draw, trans_text, font, usable_w, allow_midword_split=(fsize <= 7)
            )
            sample_bbox = draw.textbbox((0, 0), sample_probe, font=font)
            lh = max(fsize + 3, int((sample_bbox[3] - sample_bbox[1]) * 1.18))
            total_h = lh * len(wrapped)
            max_line_w = max(
                (draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0])
                for ln in wrapped
            )
            chosen_font = font
            chosen_lines = wrapped
            line_height = lh
            ref_top = sample_bbox[1]
            if total_h <= usable_h and max_line_w <= usable_w:
                break

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
    # Use compress_level=3 instead of optimize=True (10x faster on Render CPU!)
    img.save(out_io, format="PNG", compress_level=3)
    img.close()
    base_gray.close()
    return out_io.getvalue()


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    from media_context import resolve_referenced_message

    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel

    image_attachments = [
        a for a in msg.attachments if a.filename.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not image_attachments and msg.reference:
        ref_msg = await resolve_referenced_message(msg)
        if ref_msg:
            image_attachments = [
                a for a in ref_msg.attachments
                if a.filename.lower().endswith(IMAGE_EXTENSIONS)
            ]

    if not image_attachments:
        await channel.send(
            "กรุณาแนบรูปภาพหน้ามังงะหรือข้อความที่ต้องการให้แปลมาด้วยนะคะ "
            "(สามารถแนบพร้อมกันหลายหน้าในข้อความเดียวได้เลยค่ะ เช่น แนบ 5 หน้าพร้อมพิมพ์ `!manga en` หรือ `!manga th`)"
        )
        return

    _, target_lang_name = _detect_target_translation_lang(user_prompt)
    # Discord allows up to 10 attachments per message
    target_attachments = image_attachments[:10]
    num_pages = len(target_attachments)

    async with channel.typing():
        pages_input = []
        for att in target_attachments:
            raw_b = await att.read()
            pages_input.append((raw_b, att.filename))

        batch_results = await extract_and_translate_manga_pages_batch(
            pages_input, target_lang_name
        )

        if not any(bubbles or summary for bubbles, summary in batch_results):
            await _send_or_reply(
                channel,
                msg,
                content="ไม่สามารถตรวจจับข้อความในภาพที่แนบมาได้ค่ะ กรุณาตรวจสอบว่าภาพมีความคมชัดนะคะ",
            )
            return

        header_title = (
            f"**📖 คำแปลมังงะ {num_pages} หน้า ตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
            if num_pages > 1
            else f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
        )
        transcript_lines = [header_title]
        discord_files = []

        for page_idx, ((raw_bytes, fname), (bubbles, scene_summary)) in enumerate(
            zip(pages_input, batch_results), 1
        ):
            if num_pages > 1:
                transcript_lines.append(f"\n__**📄 หน้าที่ {page_idx}/{num_pages} (`{fname}`)**__")
            if scene_summary:
                clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
                transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

            if not bubbles:
                transcript_lines.append("*(ไม่พบช่องคำพูดในหน้านี้)*")
                continue

            for idx, b in enumerate(bubbles, 1):
                speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
                src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
                tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
                if tr_txt:
                    transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n   ↳ **{tr_txt}**")

            try:
                rendered_png = await asyncio.to_thread(
                    render_translated_manga_image_sync, raw_bytes, bubbles
                )
                if rendered_png:
                    base_name = os.path.splitext(fname)[0]
                    discord_files.append(
                        discord.File(
                            io.BytesIO(rendered_png),
                            filename=f"translated_p{page_idx}_{base_name}.png",
                        )
                    )
            except Exception as e:
                print(f"Manga page {page_idx} rendering fallback to text-only: {e}", flush=True)

        transcript_text = "\n".join(transcript_lines)

        if discord_files:
            if len(transcript_text) <= 1950:
                await _send_or_reply(channel, msg, content=transcript_text, files=discord_files)
            else:
                await _send_or_reply(channel, msg, files=discord_files)
                await send_chunked_response(channel, transcript_text, reply_to=None)
        else:
            await send_chunked_response(channel, transcript_text, reply_to=msg)

        for df in discord_files:
            try:
                df.close()
            except Exception:
                pass
        discord_files.clear()
        pages_input.clear()
        rendered_png = raw_bytes = raw_b = None
        release_unused_memory()