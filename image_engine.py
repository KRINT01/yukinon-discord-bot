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
from PIL import Image, ImageDraw, ImageFont, ImageChops

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
)
from llm import (
    call_gemini_generate_content_self_healing,
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
    target_channel, reply_to: discord.Message | None, content: str = None, file: discord.File = None
):
    if reply_to is not None:
        try:
            return await reply_to.reply(content=content, file=file, mention_author=False)
        except Exception:
            pass
    return await target_channel.send(content=content, file=file)


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

    # Tier 2: Google AI Studio
    if GEMINI_API_KEY:
        try:
            g_payload = {
                "contents": [{"parts": [{"text": f"{strict_sys_msg}\n\nThai text: {raw_prompt}"}]}],
                "generationConfig": {"temperature": 0.0},
            }
            g_data = await call_gemini_generate_content_self_healing(
                g_payload, timeout_sec=15, purpose="Prompt Translate Tier 2"
            )
            if g_data:
                parts = ((g_data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                raw_parts_text = " ".join(p.get("text", "") for p in parts if p.get("text")).strip()
                clean_txt = raw_parts_text.strip(' "\'\n\r\t')
                if clean_txt and not THAI_CHAR_REGEX.search(clean_txt):
                    return f"{prefix}{clean_txt}"
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
        tr_txt = str(tr_txt).strip()
        src_txt = str(src_txt).strip()
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
1. Reading Order: Order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. In-Context Localization: Adapt Japanese/Korean/Chinese idioms, pronouns, tone, and emotion so the dialogue sounds completely natural in {target_lang_name} while fitting inside a speech bubble. Do NOT use double quotes inside `translated_text` (use single quotes if needed).
3. Speech Bubble Bounding Boxes (`box_2d`):
   - For each speech bubble, provide `box_2d` as `[ymin, xmin, ymax, xmax]` normalized from 0 to 1000.
   - `box_2d` must tightly cover the Japanese text block inside that specific bubble lobe (if a double bubble has two lobes, output each lobe as its own separate bubble entry covering only that lobe's text).
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
                "thinkingConfig": {"thinkingBudget": 0},
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

    # Tier 2: Cloudflare Workers AI (@cf/google/gemma-4-26b-a4b-it)
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
                    print(f"[Manga Vision] Completed via Tier 2 Cloudflare Gemma 4 - {len(bubbles)} bubbles")
                    return bubbles, summary
                fallback_bubbles_no_boxes = bubbles
                fallback_summary = summary
    except Exception as e:
        print(f"Cloudflare Gemma 4 Manga Vision warning: {e}")

    # Tier 3: Groq Vision (qwen/qwen3.8-27b)
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
                                print(f"[Manga Vision] Completed via Tier 3 Groq ({g_model}) - {len(bubbles)} bubbles")
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
    is_thai_text = bool(THAI_CHAR_REGEX.search(text))
    words = text.split(" ") if (" " in text or not is_thai_text) else [text]
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
            # Never split English/Latin words mid-word unless allow_midword_split is True (smallest fallback font)
            if (wbbox[2] - wbbox[0]) <= max_width or (not is_thai_text and not allow_midword_split):
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


def _clean_single_bubble_lobe_fast(
    img: Image.Image,
    base_gray: Image.Image,
    other_peer_boxes: list[tuple[int, int, int, int]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int]]:
    width, height = img.size
    box_w = max(10, x2 - x1)
    box_h = max(10, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 1. FIRST: Erase the full original Japanese text box (with a +6px right margin for vertical rightmost columns like 'ああ')
    # BEFORE applying Voronoi layout clamping, so right-edge Japanese characters are never left behind!
    ex1 = max(0, x1 - 4)
    ey1 = max(0, y1 - 4)
    ex2 = min(width, x2 + 6)
    ey2 = min(height, y2 + 4)
    ew = max(4, ex2 - ex1)
    eh = max(4, ey2 - ey1)

    sub_gray = base_gray.crop((ex1, ey1, ex2, ey2))
    dark_mask = sub_gray.point(lambda p: 255 if p < 212 else 0)

    # Protect only continuous outer bubble borders touching the 1px perimeter
    border_connected = dark_mask.copy()
    for bx in range(ew):
        if border_connected.getpixel((bx, 0)) == 255:
            ImageDraw.floodfill(border_connected, (bx, 0), 128)
        if border_connected.getpixel((bx, eh - 1)) == 255:
            ImageDraw.floodfill(border_connected, (bx, eh - 1), 128)
    for by in range(eh):
        if border_connected.getpixel((0, by)) == 255:
            ImageDraw.floodfill(border_connected, (0, by), 128)
        if border_connected.getpixel((ew - 1, by)) == 255:
            ImageDraw.floodfill(border_connected, (ew - 1, by), 128)

    text_only_mask = border_connected.point(lambda p: 255 if p == 255 else 0)
    # Guarantee the core text area (including the right-side first column) is wiped clean
    inner_m = ImageDraw.Draw(text_only_mask)
    ix1, iy1 = max(2, int(ew * 0.04)), max(2, int(eh * 0.05))
    ix2, iy2 = min(ew - 2, int(ew * 0.96)), min(eh - 2, int(eh * 0.95))
    if ix2 > ix1 and iy2 > iy1:
        inner_m.rectangle([ix1, iy1, ix2, iy2], fill=255)

    white_patch = Image.new("RGB", (ew, eh), (255, 255, 255))
    img.paste(white_patch, (ex1, ey1), mask=text_only_mask)

    # 2. Compute Voronoi midpoint bounds against nearby peer bubbles for text placement
    max_expand_x = max(16, int(box_w * 0.24))
    max_expand_y = max(12, int(box_h * 0.16))
    vor_x1 = max(0, x1 - max_expand_x)
    vor_y1 = max(0, y1 - max_expand_y)
    vor_x2 = min(width, x2 + max_expand_x)
    vor_y2 = min(height, y2 + max_expand_y)

    has_close_peer = False
    for ox1, oy1, ox2, oy2 in other_peer_boxes:
        ocx = (ox1 + ox2) // 2
        ocy = (oy1 + oy2) // 2
        dx = abs(cx - ocx)
        dy = abs(cy - ocy)
        if dx > (box_w + (ox2 - ox1)) and dy > (box_h + (oy2 - oy1)):
            continue
        has_close_peer = True

        h_gap = max(ox1 - x2, x1 - ox2)
        v_gap = max(oy1 - y2, y1 - oy2)
        if v_gap >= h_gap:
            if cy < ocy:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y2 + oy1) // 2
                vor_y2 = min(vor_y2, max(y1 + 12, mid_y - 2))
            else:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y1 + oy2) // 2
                vor_y1 = max(vor_y1, min(y2 - 12, mid_y + 2))
        else:
            if cx < ocx:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x2 + ox1) // 2
                vor_x2 = min(vor_x2, max(x1 + 12, mid_x - 2))
            else:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x1 + ox2) // 2
                vor_x1 = max(vor_x1, min(x2 - 12, mid_x + 2))

    cx1 = max(vor_x1, min(x1, vor_x2 - 10))
    cy1 = max(vor_y1, min(y1, vor_y2 - 10))
    cx2 = min(vor_x2, max(x2, cx1 + 10))
    cy2 = min(vor_y2, max(y2, cy1 + 10))

    # 3. Scan the CLEANED image (not base_gray) from the horizontal center of the box outward
    # and enforce symmetric expansion so double-bubble text never drifts to the left!
    cleaned_gray = img.convert("L")
    mid_y_start = cy1 + int((cy2 - cy1) * 0.22)
    mid_y_end = cy1 + int((cy2 - cy1) * 0.78)
    safe_lefts = []
    safe_rights = []
    for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), max(2, (mid_y_end - mid_y_start) // 6 or 1)):
        if not (0 <= sy < height):
            continue
        lx = cx
        while lx > vor_x1 and cleaned_gray.getpixel((lx - 1, sy)) >= 210:
            lx -= 1
        rx = cx
        while rx < vor_x2 and cleaned_gray.getpixel((rx, sy)) >= 210:
            rx += 1
        safe_lefts.append(lx)
        safe_rights.append(rx)

    if safe_lefts and safe_rights:
        wall_l = sorted(safe_lefts)[len(safe_lefts) // 2] + 5
        wall_r = sorted(safe_rights)[len(safe_rights) // 2] - 5
        # Keep expansion symmetric around the true bubble center so left expansion never pulls text left
        avail_left = max(0, cx - wall_l)
        avail_right = max(0, wall_r - cx)
        sym_half = min(avail_left, avail_right)
        if sym_half >= (box_w // 2):
            tx1 = cx - sym_half
            tx2 = cx + sym_half
        else:
            tx1 = max(vor_x1 + 4, min(cx1, wall_l))
            tx2 = min(vor_x2 - 4, max(cx2, wall_r))
    else:
        tx1, tx2 = cx1, cx2

    # Subtle +5px rightward optical balance for double bubbles so text sits dead-center in the lobe
    if has_close_peer and (tx2 + 5) < vor_x2:
        tx1 += 5
        tx2 += 5

    ty1 = max(vor_y1 + 2, cy1)
    ty2 = min(vor_y2 - 2, cy2)
    return tx1, ty1, tx2, ty2, (15, 15, 15), (255, 255, 255)


def _resolve_box_collisions(
    renders: list[list],
    min_gap: int = 6,
) -> list[list]:
    """
    Guarantees that no two bubble layout boxes overlap in 2D space.
    If two boxes intersect, splits the overlap at their midpoint along the axis of smaller overlap.
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

                # Separate along the axis where centers are further apart relative to box size
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

    pixel_boxes = []
    for b in valid_boxes:
        ymin, xmin, ymax, xmax = [float(v) for v in b["box_2d"]]
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        x1 = max(0, int((xmin / 1000.0) * width))
        y1 = max(0, int((ymin / 1000.0) * height))
        x2 = min(width, int((xmax / 1000.0) * width))
        y2 = min(height, int((ymax / 1000.0) * height))
        pixel_boxes.append((x1, y1, x2, y2, b["translated_text"].strip()))

    prepared_renders = []
    for i, (x1, y1, x2, y2, trans_text) in enumerate(pixel_boxes):
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            continue
        other_boxes = [
            (ox1, oy1, ox2, oy2)
            for j, (ox1, oy1, ox2, oy2, _) in enumerate(pixel_boxes)
            if j != i
        ]
        tx1, ty1, tx2, ty2, fg_text, stroke_col = _clean_single_bubble_lobe_fast(
            img, base_gray, other_boxes, x1, y1, x2, y2
        )
        prepared_renders.append([tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text])

    prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)

    draw = ImageDraw.Draw(img)
    for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text in prepared_renders:
        box_w = max(20, tx2 - tx1)
        box_h = max(16, ty2 - ty1)
        usable_w = max(18, int(box_w * 0.94))
        usable_h = max(14, int(box_h * 0.94))
        is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

        start_font_size = min(26, max(11, int(min(box_w, box_h) * 0.28)))
        chosen_font = _load_manga_font(9, is_thai=is_thai_text)
        chosen_lines = [trans_text]
        line_height = 11

        sample_probe = "Agกข้ำ" if is_thai_text else "Agjpqy!"
        for fsize in range(start_font_size, 7, -1):
            font = _load_manga_font(fsize, is_thai=is_thai_text)
            wrapped = _wrap_text_to_pixel_width(
                draw, trans_text, font, usable_w, allow_midword_split=(fsize <= 8)
            )
            sample_bbox = draw.textbbox((0, 0), sample_probe, font=font)
            lh = max(fsize + 2, int((sample_bbox[3] - sample_bbox[1]) * 1.18))
            total_h = lh * len(wrapped)
            max_line_w = max(
                (draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0])
                for ln in wrapped
            )
            chosen_font = font
            chosen_lines = wrapped
            line_height = lh
            if total_h <= usable_h and max_line_w <= usable_w:
                break

        total_text_h = line_height * len(chosen_lines)
        curr_y = ty1 + max(0, (box_h - total_text_h) // 2)

        for ln in chosen_lines:
            lbbox = draw.textbbox((0, 0), ln, font=chosen_font)
            lw = lbbox[2] - lbbox[0]
            curr_x = tx1 + max(0, (box_w - lw) // 2) - lbbox[0]
            draw.text(
                (curr_x, curr_y - lbbox[1]),
                ln,
                font=chosen_font,
                fill=fg_text,
                stroke_width=2,
                stroke_fill=stroke_col,
            )
            curr_y += line_height

    out_io = io.BytesIO()
    img.save(out_io, format="PNG", optimize=True)
    return out_io.getvalue()


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel

    image_attachments = [
        a for a in msg.attachments if a.filename.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not image_attachments and msg.reference and isinstance(msg.reference.resolved, discord.Message):
        image_attachments = [
            a for a in msg.reference.resolved.attachments
            if a.filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

    if not image_attachments:
        await channel.send(
            "กรุณาแนบรูปภาพหน้ามังงะหรือข้อความที่ต้องการให้แปลมาด้วยนะคะ "
            "(เช่น แนบรูปพร้อมพิมพ์ `!manga en` เพื่อแปลเป็นภาษาอังกฤษ หรือ `!manga th` เพื่อแปลเป็นภาษาไทยค่ะ)"
        )
        return

    _, target_lang_name = _detect_target_translation_lang(user_prompt)
    att = image_attachments[0]

    async with channel.typing():
        raw_bytes = await att.read()
        bubbles, scene_summary = await extract_and_translate_manga_bubbles(
            raw_bytes, att.filename, target_lang_name
        )

        if not bubbles and scene_summary:
            await send_chunked_response(
                channel,
                f"**📖 คำแปลข้อความจากภาพ ({target_lang_name}):**\n\n{scene_summary}",
                reply_to=msg,
            )
            return

        if not bubbles:
            await _send_or_reply(
                channel,
                msg,
                content="ไม่สามารถตรวจจับข้อความในภาพนี้ได้ค่ะ กรุณาตรวจสอบว่าภาพมีความคมชัดนะคะ",
            )
            return

        transcript_lines = [
            f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
        ]
        if scene_summary:
            clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
            transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

        for idx, b in enumerate(bubbles, 1):
            speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
            src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
            tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
            if tr_txt:
                transcript_lines.append(f"**[{idx}] [{speaker}]** `{src_txt}`\n   ↳ **{tr_txt}**")

        transcript_text = "\n".join(transcript_lines)

        rendered_png = None
        try:
            rendered_png = await asyncio.to_thread(
                render_translated_manga_image_sync, raw_bytes, bubbles
            )
        except Exception as e:
            print(f"Manga image rendering fallback to text-only: {e}")

        if rendered_png:
            base_name = os.path.splitext(att.filename)[0]
            discord_file = discord.File(io.BytesIO(rendered_png), filename=f"translated_{base_name}.png")
            if len(transcript_text) <= 1950:
                await _send_or_reply(channel, msg, content=transcript_text, file=discord_file)
            else:
                await _send_or_reply(channel, msg, file=discord_file)
                await send_chunked_response(channel, transcript_text, reply_to=None)
        else:
            await send_chunked_response(channel, transcript_text, reply_to=msg)