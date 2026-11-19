import os
import io
import gc
import re
import json
import time
import uuid
import base64
import asyncio
from urllib.parse import quote_plus

import aiohttp
import discord
import functools
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter, ImageStat

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
    get_cached_channel_attachments,
    release_unused_memory,
)
from llm import (
    call_gemini_generate_content_self_healing,
    call_gemini_decision_orchestrator,
    call_gemma_ai_studio_chat,
    call_gemma_ai_studio_vision,
    call_cloudflare_gemma_chat,
    call_modal_ollama_chat,
    call_groq_chat,
    call_cloudflare_vision_chat,
    discover_active_groq_vision_models,
)

MAX_MANGA_BATCH_PAGES = 12  # Supports up to 12 pages across multi-part threads (split into <=10 files per Discord message)

# Cache for self-healing discovery of strictly newer (>3.5) `flash-lite` models on Google AI Studio
_FLASH_LITE_DISCOVERY_CACHE: list[str] = []
_FLASH_LITE_DISCOVERY_TS: float = 0.0
_FLASH_LITE_EXCLUDED = ("2.5", "2.0", "1.5", "latest", "image", "tts", "audio", "live", "embedding")

# Strict JSON responseSchema for Gemini models so `box_2d` is ALWAYS emitted as a 4-integer [ymin, xmin, ymax, xmax] array
MANGA_BATCH_RESPONSE_SCHEMA = {
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
                                "speaker_or_note": {"type": "STRING"},
                                "panel_position": {"type": "STRING"},
                                "source_text": {"type": "STRING"},
                                "translated_text": {"type": "STRING"},
                                "box_2d": {
                                    "type": "ARRAY",
                                    "items": {"type": "INTEGER"},
                                    "nullable": True,
                                },
                            },
                            "required": ["id", "speaker_or_note", "source_text", "translated_text", "box_2d"],
                        },
                    },
                },
                "required": ["page_index", "summary", "bubbles"],
            },
        }
    },
    "required": ["pages"],
}

MANGA_SINGLE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "bubbles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "speaker_or_note": {"type": "STRING"},
                    "panel_position": {"type": "STRING"},
                    "source_text": {"type": "STRING"},
                    "translated_text": {"type": "STRING"},
                    "box_2d": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "nullable": True,
                    },
                },
                "required": ["id", "speaker_or_note", "source_text", "translated_text", "box_2d"],
            },
        },
    },
    "required": ["summary", "bubbles"],
}

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

CODE_FENCE_BLOCK_REGEX = re.compile(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```")
LANG_DEFAULT_EXT = {
    "java": "Solution.java",
    "python": "script.py",
    "py": "script.py",
    "javascript": "index.js",
    "js": "index.js",
    "typescript": "index.ts",
    "ts": "index.ts",
    "tsx": "Component.tsx",
    "jsx": "Component.jsx",
    "sql": "query.sql",
    "plsql": "package.sql",
    "xml": "pom.xml",
    "yaml": "config.yaml",
    "yml": "config.yml",
    "json": "config.json",
    "dockerfile": "Dockerfile",
    "docker": "Dockerfile",
    "bash": "script.sh",
    "sh": "script.sh",
    "powershell": "script.ps1",
    "ps1": "script.ps1",
    "html": "index.html",
    "css": "styles.css",
    "scss": "styles.scss",
    "properties": "application.properties",
    "diff": "changes.diff",
    "patch": "changes.patch",
}
EXPLICIT_FILENAME_COMMENT_REGEX = re.compile(
    r"^(?://|#|--|/\*|<!--)\s*([A-Za-z0-9_.-]+\.(?:java|py|js|ts|tsx|jsx|sql|pls|pkb|pks|xml|yaml|yml|json|sh|ps1|html|css|scss|properties|gradle|md|txt|diff|patch)|Dockerfile|docker-compose\.ya?ml|pom\.xml)\b",
    re.IGNORECASE,
)


def extract_named_code_files(text: str) -> tuple[str, list[discord.File]]:
    files: list[discord.File] = []
    used_names: set[str] = set()

    def _replace_block(match: re.Match) -> str:
        lang = (match.group(1) or "").strip().lower()
        code_body = match.group(2) or ""
        lines = code_body.splitlines()
        fname = None

        if lines:
            first_line = lines[0].strip()
            m_fn = EXPLICIT_FILENAME_COMMENT_REGEX.match(first_line)
            if m_fn:
                fname = m_fn.group(1)

        if not fname:
            fname = LANG_DEFAULT_EXT.get(lang, f"snippet_{len(files) + 1}.txt")

        base, ext = os.path.splitext(fname)
        candidate = fname
        counter = 2
        while candidate.lower() in used_names:
            candidate = f"{base}_{counter}{ext}" if ext else f"{base}_{counter}"
            counter += 1
        used_names.add(candidate.lower())

        files.append(discord.File(io.StringIO(code_body), filename=candidate))
        return f"📎 **[Attached Source File: `{candidate}`]**"

    summary_text = CODE_FENCE_BLOCK_REGEX.sub(_replace_block, text)
    return summary_text, files[:10]


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
        kwargs["files"] = files[:10]
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
        summary_text, code_files = extract_named_code_files(text)
        if code_files:
            chunks = split_discord_message_smart(summary_text, max_len=1900)
            if len(chunks) <= 3:
                for idx, chunk in enumerate(chunks):
                    await _send_or_reply(
                        target_channel,
                        reply_to if idx == 0 else None,
                        content=chunk,
                        files=code_files if idx == len(chunks) - 1 else None,
                    )
                return
            code_files.insert(0, discord.File(io.StringIO(text), filename=filename_fallback))
            await _send_or_reply(
                target_channel,
                reply_to,
                content="**เนื้อหาและโค้ดมีความยาวมาก ฉันแยกไฟล์ซอร์สโค้ดพร้อมฉบับเต็มแนบไว้ให้ด้านล่างนะคะ:**",
                files=code_files[:10],
            )
            return

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

    try:
        res = await call_modal_ollama_chat(
            trans_messages, ACTIVE_FAST_MODEL, 0.0, purpose="Prompt Translate Tier 3"
        )
        if res and not THAI_CHAR_REGEX.search(res):
            clean_res = res.strip(' "\'\n\r\t')
            return f"{prefix}{clean_res}"
    except Exception:
        pass

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
    Normalizes Unicode punctuation (`…` -> `...`, `〜` -> `~`, `—` -> `-`, smart quotes -> `'`),
    removes stray CJK/emoji symbols, and cleans empty `()` / `[]` leftovers so `Garuda-Bold.ttf`
    renders every character cleanly with zero `[]` tofu boxes.
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
    )
    s = re.sub(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00010000-\U0010FFFF]", "", s)
    s = re.sub(r"\(\s*\)|\[\s*\]|'\s*'", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_target_translation_lang(user_prompt: str) -> tuple[str, str]:
    lower_p = user_prompt.lower()
    if any(k in lower_p for k in ("english", "eng", "อังกฤษ", "เป็นอิ้ง", "แปลอิ้ง", " to en")):
        return "en", "English"
    if any(k in lower_p for k in ("thai", "ไทย", "เป็นไทย", "แปลไทย", " to th")):
        return "th", "Thai"
    if THAI_CHAR_REGEX.search(user_prompt):
        return "th", "Thai"
    return "en", "English"


def _prepare_manga_image_for_vision(raw_bytes: bytes, max_dim: int = 1280, quality: int = 90) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    img.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_flash_lite_version(model_id: str) -> tuple[int, int]:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?-flash-lite", (model_id or "").lower())
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


async def _discover_manga_batch_models() -> list[tuple[str, bool]]:
    """
    Returns the exact ordered model chain for multi-page `MangaBatch`:
      1. `gemini-3.5-flash-lite`
      2. `gemini-3.1-flash-lite`
      3. Any strictly newer (>3.5) `flash-lite` model discovered via self-healing (no duplicate -preview aliases of 3.1/3.5!)
      4. `gemma-4-31b-it`
      5. `gemma-4-26b-a4b-it`
    Each entry is `(model_id, is_gemma)`.
    """
    global _FLASH_LITE_DISCOVERY_CACHE, _FLASH_LITE_DISCOVERY_TS
    now = time.time()

    if GEMINI_API_KEY and (not _FLASH_LITE_DISCOVERY_CACHE or (now - _FLASH_LITE_DISCOVERY_TS >= 3600.0)):
        list_url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        )
        try:
            session = get_http_session()
            async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    found = []
                    for m in data.get("models") or []:
                        name = (m.get("name") or "").replace("models/", "")
                        lower_name = name.lower()
                        methods = m.get("supportedGenerationMethods") or []
                        if "generateContent" in methods and "flash-lite" in lower_name:
                            if any(bad in lower_name for bad in _FLASH_LITE_EXCLUDED):
                                continue
                            ver = _parse_flash_lite_version(name)
                            # Only add strictly newer versions (> 3.5) so older `-preview` aliases of 3.1/3.5 never burn wall time
                            if ver > (3, 5):
                                found.append((ver, name))
                    found.sort(reverse=True)
                    _FLASH_LITE_DISCOVERY_CACHE = [n for _, n in found]
                    _FLASH_LITE_DISCOVERY_TS = now
        except Exception:
            pass

    chain: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for m_id in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"] + _FLASH_LITE_DISCOVERY_CACHE:
        if m_id and m_id not in seen:
            seen.add(m_id)
            chain.append((m_id, False))

    for g_id in ("gemma-4-31b-it", "gemma-4-26b-a4b-it"):
        if g_id not in seen:
            seen.add(g_id)
            chain.append((g_id, True))

    return chain


def _extract_norm_box_2d(raw_box) -> list[float] | None:
    """
    Flexibly parses bounding boxes returned in any VLM format and auto-scales
    `[0.0..1.0]` normalized floats up to `[0..1000]`.
    """
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

        raw_box = (
            item.get("box_2d")
            or item.get("bbox")
            or item.get("bbox_2d")
            or item.get("bounding_box")
            or item.get("box")
            or item.get("coordinates")
            or item.get("rect")
        )
        norm_box = None if is_sfx else _extract_norm_box_2d(raw_box)
        panel_pos = str(item.get("panel_position") or "").strip().lower()

        valid.append({
            "id": item.get("id", idx),
            "speaker_or_note": str(speaker).strip(),
            "panel_position": panel_pos,
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


async def _waiting_heartbeat_logger(req_id: str, label: str, model_id: str, interval_sec: float = 15.0):
    """Logs periodic waiting status lines to Render stderr while a long vision request is in flight."""
    start_t = time.monotonic()
    try:
        while True:
            await asyncio.sleep(interval_sec)
            elapsed = time.monotonic() - start_t
            api_log(f"{label} [req={req_id}] WAITING -> model={model_id} still running ({elapsed:.1f}s elapsed...)")
    except asyncio.CancelledError:
        return


async def extract_and_translate_manga_pages_batch(
    pages_input: list[tuple[bytes, str]],
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
) -> tuple[list[tuple[list[dict], str]], bool]:
    """
    Processes 1 to 12 manga pages:
      - If `num_pages == 1`: calls `extract_and_translate_manga_bubbles()`.
      - If `num_pages > 1`: ALWAYS runs unified multi-page `MangaBatch` strictly across:
        `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 `flash-lite`)
        -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
        NEVER falls back to per-page `MangaSingle`!
      - Emits correlated `[req=...]` REQUEST / WAITING / RESPONSE logs to Render.
    Returns `(results, is_service_unavailable)`.
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return [], False

    req_id = req_id or f"mb-{uuid.uuid4().hex[:6]}"

    if num_pages == 1:
        b, s = await extract_and_translate_manga_bubbles(
            pages_input[0][0],
            pages_input[0][1],
            target_lang_name,
            series_context_hint=series_context_hint,
            req_id=req_id,
        )
        return [(b, s)], False

    if not GEMINI_API_KEY:
        return [([], "") for _ in range(num_pages)], True

    # Keep vision images crisp enough (1024px for 9-12p, 1120px for 5-8p, 1280px for 2-4p) so small bubbles are never skipped
    if num_pages >= 9:
        batch_dim, batch_q = 1024, 84
    elif num_pages >= 5:
        batch_dim, batch_q = 1120, 86
    else:
        batch_dim, batch_q = 1280, 90

    b64_pages = []
    for raw_b, fname in pages_input:
        b64_str = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_b, batch_dim, batch_q)
        b64_pages.append((b64_str, fname))

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster (use these exact names & terms):\n{series_context_hint[:1400]}\n"
        if series_context_hint else ""
    )

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
{context_block}
Analyze all {num_pages} page(s) in sequence so character names, titles/ranks, pronouns, and ongoing story stay consistent across every page,
and translate EVERY SINGLE speech bubble, narration/monologue box, and caption box on EVERY page into natural, idiomatic {target_lang_name}.

CRITICAL COMPLETENESS, 1-TO-1 BUBBLE SEPARATION & BOUNDING-BOX RULES:
1. STRICT 1-BUBBLE = 1-OBJECT RULE (NEVER MERGE SEPARATE BUBBLES!):
   - Every physically separate speech oval, cloud, or rectangular caption box MUST be its own separate object in `bubbles` — EVEN IF the same character speaks across 2 or 3 separate bubbles in the same panel!
   - NEVER combine text from two separate speech bubbles into one `translated_text`!
   - Scan every panel of every page systematically from top-right to bottom-left and extract EVERY speech bubble (including short 1-word replies, narrow vertical bubbles, rectangular time/location caption boxes, and secondary bubbles in crowded panels).
   - Within each page, order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. Consistent Character Names, Visual Gender & Localization:
   - Keep character names, titles/ranks, and faction names consistent across all pages, cross-checking with the post context above and accurate Japanese phonetics.
   - Match pronouns and politeness particles to each character's visual gender and personality (for example, when translating to Thai, a female character who uses casual/boyish Japanese pronouns like `僕` or `妾` must still use female/neutral Thai endings like `นะ`/`ล่ะ`/`สิ`/`ค่ะ` — NEVER male `ครับ`/`ผม` unless the speaker is visually male).
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
   - When translating to English, preserve Japanese honorifics attached directly with a hyphen and NO space (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).
3. Spatial Anchor & Bounding Boxes (`panel_position` + `box_2d`) — MANDATORY FOR IN-BUBBLE RENDERING:
   - First identify `panel_position` on that specific page (`"top-right"`, `"top-left"`, `"middle-right"`, `"middle-left"`, `"bottom-right"`, `"bottom-left"`), then output `box_2d` as 4 INTEGERS `[ymin, xmin, ymax, xmax]` in `0..1000` scale relative to THAT single page.
   - You MUST provide a non-null `[ymin, xmin, ymax, xmax]` `box_2d` for EVERY dialogue bubble, thought bubble, and narration/caption box!
   - `box_2d` MUST tightly enclose ONLY the Japanese text INSIDE that specific bubble — NEVER shift `box_2d` onto a character's face, eyes, hands, arms, hair, empty background, or another panel!
   - ONLY set `"box_2d": null` for floating handwritten SFX / sound effects (e.g. ドーン, トスッ, ポスン, ムチッ, スラ〜〜) or tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble!
   - NEVER assign the same `box_2d` to two different bubbles!
4. Return ONLY valid JSON matching this exact structure for all {num_pages} page(s) (write `summary` in {target_lang_name}):
{{
  "pages": [
    {{
      "page_index": 1,
      "summary": "Brief 1-sentence scene context for Page 1 in {target_lang_name}",
      "bubbles": [
        {{
          "id": 1,
          "speaker_or_note": "Character / Narration / SFX",
          "panel_position": "top-right",
          "source_text": "Original Japanese text",
          "translated_text": "Natural {target_lang_name} translation",
          "box_2d": [120, 450, 280, 620]
        }}
      ]
    }}
  ]
}}
""".strip()

    fallback_text_only_results = None

    try:
        parts = [{"text": manga_batch_instruction}]
        for idx, (b64_img, fname) in enumerate(b64_pages, 1):
            parts.append({"text": f"=== PAGE {idx} ({fname}) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        model_chain = await _discover_manga_batch_models()
        per_call_timeout = min(55.0, 30.0 + (2.5 * num_pages))
        max_wall_budget = 165.0
        wall_start = time.monotonic()
        session = get_http_session()
        label_prefix = f"MangaBatch({num_pages}p)"

        for model_id, is_gemma in model_chain:
            remaining = max_wall_budget - (time.monotonic() - wall_start)
            if remaining <= 8.0:
                break

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
                gen_cfg["responseSchema"] = MANGA_BATCH_RESPONSE_SCHEMA

            payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
            max_attempts = 2

            for attempt in range(max_attempts):
                rem_now = max_wall_budget - (time.monotonic() - wall_start)
                if rem_now <= 6.0:
                    break
                req_t = min(per_call_timeout, rem_now)
                t0 = time.monotonic()

                api_log(
                    f"{label_prefix} [req={req_id}] REQUEST -> model={model_id} "
                    f"(attempt {attempt + 1}/{max_attempts}, timeout={req_t:.0f}s, dim={batch_dim}px)"
                )
                hb_task = asyncio.create_task(
                    _waiting_heartbeat_logger(req_id, label_prefix, model_id, interval_sec=15.0)
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
                    api_log(
                        f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} "
                        f"FAILED ({type(e).__name__}, {elapsed:.2f}s); trying next model"
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
                    resp_parts = (
                        ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                    )
                    raw_json_str = "\n".join(
                        p.get("text", "") for p in resp_parts
                        if p.get("text") and not p.get("thought", False)
                    ).strip()
                    batch_results = _parse_manga_pages_batch_json(raw_json_str, num_pages)
                    total_b = sum(len(b) for b, _ in batch_results)
                    total_boxes = sum(
                        1 for b_list, _ in batch_results for b in b_list
                        if isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4
                    )
                    if total_b > 0 and total_boxes > 0:
                        api_log(
                            f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 OK "
                            f"({elapsed:.2f}s, {total_boxes}/{total_b} boxes)"
                        )
                        return batch_results, False

                    if total_b > 0 and total_boxes == 0:
                        if fallback_text_only_results is None:
                            fallback_text_only_results = batch_results
                        api_log(
                            f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 "
                            f"but 0/{total_b} valid box_2d ({elapsed:.2f}s); trying next model"
                        )
                        break

                    api_log(
                        f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 200 "
                        f"empty JSON ({elapsed:.2f}s); trying next model"
                    )
                    break

                if status == 400 and attempt == 0:
                    changed = False
                    if "responseSchema" in payload["generationConfig"] and ("schema" in body_text.lower() or "responseSchema" in body_text):
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

                if status == 503 and attempt == 0 and (max_wall_budget - (time.monotonic() - wall_start)) > 20.0:
                    api_log(
                        f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP 503 ({elapsed:.2f}s); retrying once in 1.5s"
                    )
                    await asyncio.sleep(1.5)
                    continue

                api_log(
                    f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} HTTP {status} ({elapsed:.2f}s); trying next model"
                )
                break
    finally:
        b64_pages.clear()

    if fallback_text_only_results is not None:
        return fallback_text_only_results, False

    api_log(f"MangaBatch({num_pages}p) [req={req_id}] EXHAUSTED -> all models unavailable/timed out")
    return [([], "") for _ in range(num_pages)], True


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
        pos_m = re.search(r'"panel_position"\s*:\s*"([^"]+)"', block)
        box_m = re.search(r'"(?:box_2d|bbox|bbox_2d|bounding_box|box)"\s*:\s*\[([^\]]+)\]', block)
        norm_box = _extract_norm_box_2d(box_m.group(1)) if box_m else None

        raw_tr_val = tr_m.group(1).replace('\\"', '"').strip()
        extracted_bubbles.append({
            "id": idx,
            "speaker_or_note": spk_m.group(1).strip() if spk_m else f"Bubble {idx}",
            "panel_position": pos_m.group(1).strip() if pos_m else "",
            "source_text": src_m.group(1).strip() if src_m else "",
            "translated_text": raw_tr_val,
            "box_2d": norm_box,
        })

    valid_extracted = _normalize_and_validate_bubbles(extracted_bubbles)
    return valid_extracted, summary


async def extract_and_translate_manga_bubbles(
    raw_bytes: bytes,
    filename: str,
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
) -> tuple[list[dict], str]:
    req_id = req_id or f"ms-{uuid.uuid4().hex[:6]}"
    b64_img = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_bytes, 1280, 90)
    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster:\n{series_context_hint[:1000]}\n"
        if series_context_hint else ""
    )

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
{context_block}
Analyze this single manga page visually (character expressions, action, speaker relationships, and panel layout)
and translate EVERY SINGLE speech bubble, narration/monologue box, and caption box into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. STRICT 1-BUBBLE = 1-OBJECT RULE (NEVER MERGE SEPARATE BUBBLES!):
   - Every physically separate speech oval, cloud, or rectangular caption box MUST be its own separate object in `bubbles` — EVEN IF the same character speaks across 2 or 3 separate bubbles!
   - Scan every panel systematically and translate EVERY speech bubble on the page without skipping any bubble.
   - Order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. Consistent Character Names, Visual Gender & Localization:
   - Keep character names, titles/ranks, and faction names consistent with the post context above and accurate Japanese phonetics.
   - Match pronouns and politeness particles to each character's visual gender (never use male Thai `ครับ`/`ผม` for visually female characters).
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
   - When translating to English, preserve Japanese honorifics attached directly with a hyphen and NO space (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).
3. Spatial Anchor & Bounding Boxes (`panel_position` + `box_2d`) — MANDATORY FOR IN-BUBBLE RENDERING:
   - First identify `panel_position` (`"top-right"`, `"top-left"`, `"middle-right"`, `"middle-left"`, `"bottom-right"`, `"bottom-left"`), then output `box_2d` as 4 INTEGERS `[ymin, xmin, ymax, xmax]` in `0..1000` scale.
   - Provide a non-null `[ymin, xmin, ymax, xmax]` `box_2d` for EVERY dialogue bubble, thought bubble, and narration/caption box!
   - `box_2d` MUST tightly enclose ONLY the Japanese text INSIDE that specific bubble — NEVER shift onto character faces, hands, arms, hair, or empty background!
   - ONLY set `"box_2d": null` for floating handwritten SFX / sound effects or tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble!
4. Return ONLY valid JSON matching this exact structure (every bubble MUST have a non-empty `translated_text`; write `summary` in {target_lang_name}):
{{
  "summary": "Brief 1-sentence scene context in {target_lang_name}",
  "bubbles": [
    {{
      "id": 1,
      "speaker_or_note": "Character / Narration / SFX",
      "panel_position": "top-right",
      "source_text": "Original Japanese text",
      "translated_text": "Natural {target_lang_name} translation",
      "box_2d": [120, 450, 280, 620]
    }}
  ]
}}
""".strip()

    fallback_bubbles_no_boxes = []
    fallback_summary = ""

    if GEMINI_API_KEY:
        api_log(f"MangaSingle [req={req_id}] REQUEST -> Tier 1 Gemini ({filename})")
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
                "responseSchema": MANGA_SINGLE_RESPONSE_SCHEMA,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(
                gemini_payload,
                timeout_sec=30,
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
                    box_cnt = sum(1 for b in bubbles if isinstance(b.get("box_2d"), list) and len(b["box_2d"]) == 4)
                    if box_cnt > 0:
                        api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 1 Gemini OK ({box_cnt}/{len(bubbles)} boxes)")
                        return bubbles, summary
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
            timeout_sec=30,
            purpose=f"MangaSingle[{req_id}]",
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
                    api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 2 AI Studio Gemma 4 OK ({len(bubbles)} bubbles)")
                    return bubbles, summary
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
                has_boxes_and_text = any(
                    isinstance(b.get("box_2d"), list)
                    and len(b["box_2d"]) == 4
                    and bool(b.get("translated_text"))
                    for b in bubbles
                )
                if has_boxes_and_text:
                    api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 3 Cloudflare Gemma 4 OK ({len(bubbles)} bubbles)")
                    return bubbles, summary
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
                                api_log(f"MangaSingle [req={req_id}] RESPONSE -> Tier 4 Groq ({g_model}) OK ({len(bubbles)} bubbles)")
                                return bubbles, summary
                            if not fallback_bubbles_no_boxes:
                                fallback_bubbles_no_boxes = bubbles
                                fallback_summary = summary
            except Exception as e:
                api_log(f"MangaSingle [req={req_id}] Tier 4 Groq ({g_model}) warning: {e}")

    if fallback_bubbles_no_boxes:
        return fallback_bubbles_no_boxes, fallback_summary

    return [], ""


def _split_thai_word_cleanly(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    leading_vowels = set("เแโใไ")
    trailing_keep = set("ะาำๆฯ?!.,")
    out_lines: list[str] = []
    chunk = ""

    for ch in word:
        test_ch = chunk + ch
        cbbox = draw.textbbox((0, 0), test_ch, font=font)
        is_combining = ("\u0E31" <= ch <= "\u0E3A") or ("\u0E47" <= ch <= "\u0E4E")
        if (cbbox[2] - cbbox[0]) > max_width and chunk and not is_combining:
            if len(chunk) > 2 and chunk[-1] in leading_vowels:
                out_lines.append(chunk[:-1])
                chunk = chunk[-1] + ch
            elif ch in trailing_keep and len(chunk) > 2:
                cut = len(chunk) - 1
                while cut > 1 and (("\u0E31" <= chunk[cut] <= "\u0E3A") or ("\u0E47" <= chunk[cut] <= "\u0E4E")):
                    cut -= 1
                if cut > 1 and chunk[cut - 1] in leading_vowels:
                    cut -= 1
                if cut > 1:
                    out_lines.append(chunk[:cut])
                    chunk = chunk[cut:] + ch
                else:
                    out_lines.append(chunk)
                    chunk = ch
            else:
                out_lines.append(chunk)
                chunk = ch
        else:
            chunk = test_ch

    if chunk:
        out_lines.append(chunk)
    return out_lines


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


def _fast_dilate_mask_half_res(mask_l: Image.Image, kernel_half: int = 3) -> Image.Image:
    """
    Runs morphological dilation at 0.5x resolution (equivalent to `MaxFilter(5..6)` at 1x)
    which is ~16x faster on 0.1 vCPU while preserving full-resolution mask crispness.
    """
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


def _measure_crop_chroma_stats(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """
    Returns `(mean_sat, chroma_ratio)` where `chroma_ratio` is the fraction of pixels with
    HSV saturation >= 22. Pure white/gray speech bubbles have `chroma_ratio < 0.03`, whereas
    anime faces, blush, skin/hands, hair, and colored wallpaper have `chroma_ratio >= 0.14`
    even when diluted by surrounding white pixels!
    """
    crop_rgb = img.crop((x1, y1, x2, y2))
    crop_hsv = crop_rgb.convert("HSV")
    crop_rgb.close()
    sat_band = crop_hsv.getchannel("S")
    crop_hsv.close()
    sat_hist = sat_band.histogram()
    mean_sat = ImageStat.Stat(sat_band).mean[0]
    sat_band.close()
    total_px = max(1, sum(sat_hist))
    chroma_ratio = sum(sat_hist[22:]) / total_px
    return float(mean_sat), float(chroma_ratio)


def _detect_bubble_polarity(
    base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> bool:
    bw = x2 - x1
    bh = y2 - y1
    if bw < 10 or bh < 10:
        return False
    sub = base_gray.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2))
    median_lum = ImageStat.Stat(sub).median[0]
    sub.close()
    return median_lum < 125


def _has_dark_panel_gutter_between(
    base_gray: Image.Image,
    cx1: int,
    cy1: int,
    cx2: int,
    cy2: int,
) -> bool:
    steps = 12
    for s in range(1, steps):
        px = int(cx1 + (cx2 - cx1) * (s / steps))
        py = int(cy1 + (cy2 - cy1) * (s / steps))
        if 0 <= px < base_gray.width and 0 <= py < base_gray.height:
            if base_gray.getpixel((px, py)) < 65:
                return True
    return False


def _compute_peer_clip_bounds(
    base_gray: Image.Image,
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

    clip_x1 = 2
    clip_y1 = 2
    clip_x2 = width - 2
    clip_y2 = height - 2

    for ox1, oy1, ox2, oy2 in other_peer_boxes:
        ocx = (ox1 + ox2) // 2
        ocy = (oy1 + oy2) // 2
        dx = abs(cx - ocx)
        dy = abs(cy - ocy)
        if dx > (box_w + (ox2 - ox1)) * 1.45 and dy > (box_h + (oy2 - oy1)) * 1.45:
            continue

        if _has_dark_panel_gutter_between(base_gray, cx, cy, ocx, ocy):
            continue

        h_gap = max(ox1 - x2, x1 - ox2)
        v_gap = max(oy1 - y2, y1 - oy2)
        if v_gap >= h_gap:
            if cy < ocy:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y2 + oy1) // 2
                clip_y2 = min(clip_y2, max(y1 + 12, mid_y - 3))
            else:
                mid_y = (min(y2, oy2) + max(y1, oy1)) // 2 if v_gap < 0 else (y1 + oy2) // 2
                clip_y1 = max(clip_y1, min(y2 - 12, mid_y + 3))
        else:
            if cx < ocx:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x2 + ox1) // 2
                clip_x2 = min(clip_x2, max(x1 + 12, mid_x - 3))
            else:
                mid_x = (min(x2, ox2) + max(x1, ox1)) // 2 if h_gap < 0 else (x1 + ox2) // 2
                clip_x1 = max(clip_x1, min(x2 - 12, mid_x + 3))

    return clip_x1, clip_y1, clip_x2, clip_y2


def _find_best_enclosed_white_bubble(
    sub_gray: Image.Image,
    rx1_rel: int,
    ry1_rel: int,
    rx2_rel: int,
    ry2_rel: int,
    has_kanji_kana: bool = True,
) -> Image.Image | None:
    """
    Fast 0.5x-scale enclosed pure-white speech bubble finder with proportional window support
    so tall full-resolution ovals are never clipped open at crop borders.
    """
    ww, wh = sub_gray.size
    if ww < 14 or wh < 14:
        return None

    sw = max(7, ww // 2)
    sh = max(7, wh // 2)
    small_gray = sub_gray.resize((sw, sh), Image.Resampling.BILINEAR)

    s_rx1 = max(1, rx1_rel // 2)
    s_ry1 = max(1, ry1_rel // 2)
    s_rx2 = min(sw - 2, max(s_rx1 + 3, rx2_rel // 2))
    s_ry2 = min(sh - 2, max(s_ry1 + 3, ry2_rel // 2))
    s_bw = max(3, s_rx2 - s_rx1)
    s_bh = max(3, s_ry2 - s_ry1)

    seed_candidates = []
    for fy in (0.50, 0.25, 0.75):
        sy = max(1, min(sh - 2, int(s_ry1 + s_bh * fy)))
        for fx in (0.50, 0.25, 0.75):
            sx = max(1, min(sw - 2, int(s_rx1 + s_bw * fx)))
            seed_candidates.append((sx, sy))
        for extra_dx in (-14, 14, -28, 28):
            sx_ext = (s_rx1 + extra_dx) if extra_dx < 0 else (s_rx2 + extra_dx)
            if 1 <= sx_ext < sw - 1:
                seed_candidates.append((sx_ext, sy))

    base_bin = small_gray.point(lambda p: 255 if p >= 238 else 0)
    small_gray.close()
    pinch_3 = base_bin.filter(ImageFilter.MinFilter(3))

    best_small_filled = None
    best_pinch_level = 0
    best_score = -1.0
    best_is_single_lobe = False

    for pass_idx, bin_img in enumerate((base_bin, pinch_3)):
        visited = Image.new("L", (sw, sh), 0)
        for sx, sy in seed_candidates:
            if bin_img.getpixel((sx, sy)) != 255 or visited.getpixel((sx, sy)) != 0:
                continue

            work = bin_img.copy()
            ImageDraw.floodfill(work, (sx, sy), 128)
            comp = work.point(lambda p: 255 if p == 128 else 0)
            work.close()

            new_visited = ImageChops.lighter(visited, comp)
            visited.close()
            visited = new_visited

            c_bbox = comp.getbbox()
            if not c_bbox:
                comp.close()
                continue
            cx1, cy1, cx2, cy2 = c_bbox
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw < 7 or ch < 7:
                comp.close()
                continue

            edge_touches = (
                (1 if cx1 <= 1 else 0)
                + (1 if cy1 <= 1 else 0)
                + (1 if cx2 >= sw - 1 else 0)
                + (1 if cy2 >= sh - 1 else 0)
            )
            if edge_touches >= 3 or (cx1 <= 1 and cx2 >= sw - 1) or (cy1 <= 1 and cy2 >= sh - 1):
                comp.close()
                continue

            padded = Image.new("L", (sw + 4, sh + 4), 0)
            padded.paste(comp, (2, 2))
            ImageDraw.floodfill(padded, (0, 0), 128)
            cropped_pad = padded.crop((2, 2, sw + 2, sh + 2))
            padded.close()
            holes = cropped_pad.point(lambda p: 255 if p == 0 else 0)
            cropped_pad.close()
            filled = ImageChops.lighter(comp, holes)

            comp_crop = comp.crop(c_bbox)
            comp_area = comp_crop.histogram()[255]
            comp_crop.close()
            comp.close()

            filled_crop = filled.crop(c_bbox)
            filled_area = max(1, filled_crop.histogram()[255])
            filled_crop.close()

            holes_crop = holes.crop(c_bbox)
            holes_area = holes_crop.histogram()[255]
            holes_crop.close()
            holes.close()

            if comp_area < 32 or (comp_area / filled_area) < 0.24:
                filled.close()
                continue

            if has_kanji_kana and holes_area < 3:
                filled.close()
                continue

            ov_x = max(0, min(cx2, s_rx2) - max(cx1, s_rx1))
            ov_y = max(0, min(cy2, s_ry2) - max(cy1, s_ry1))
            overlap_area = ov_x * ov_y
            if overlap_area < 12:
                filled.close()
                continue

            is_single = (cw <= s_bw * 2.15)
            oversize_penalty = 1.0 if is_single else 0.65
            score = ((overlap_area * 2.5) + (holes_area * 10.0) + (filled_area * 0.25)) * oversize_penalty
            if pass_idx > 0:
                score *= 0.94

            if score > best_score:
                if best_small_filled is not None:
                    best_small_filled.close()
                best_score = score
                best_small_filled = filled
                best_pinch_level = pass_idx
                best_is_single_lobe = is_single
            else:
                filled.close()

        visited.close()
        if best_small_filled is not None and best_is_single_lobe:
            break

    base_bin.close()
    pinch_3.close()

    if best_small_filled is None:
        return None

    if best_pinch_level == 1:
        expanded = best_small_filled.filter(ImageFilter.MaxFilter(3))
        best_small_filled.close()
        best_small_filled = expanded

    resized = best_small_filled.resize((ww, wh), Image.Resampling.BILINEAR)
    best_small_filled.close()
    full_filled = resized.point(lambda p: 255 if p >= 128 else 0)
    resized.close()
    return full_filled


def _sample_dark_bubble_bg_rgb_and_lum(img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int, int], int]:
    width, height = img.size
    sx1 = max(0, x1 - 4)
    sy1 = max(0, y1 - 4)
    sx2 = min(width, x2 + 4)
    sy2 = min(height, y2 + 4)
    crop_rgb = img.crop((sx1, sy1, sx2, sy2))
    crop_gray = base_gray.crop((sx1, sy1, sx2, sy2))

    bg_mask = crop_gray.point(lambda p: 255 if p < 145 else 0)
    if not bg_mask.getbbox():
        bg_mask.close()
        bg_mask = Image.new("L", crop_gray.size, 255)

    stat_rgb = ImageStat.Stat(crop_rgb, mask=bg_mask)
    stat_g = ImageStat.Stat(crop_gray, mask=bg_mask)
    bg_rgb = tuple(int(v) for v in stat_rgb.median[:3])
    bg_lum = int(stat_g.median[0])

    bg_mask.close()
    crop_rgb.close()
    crop_gray.close()
    return bg_rgb, bg_lum


def _sample_light_bubble_bg_rgb(img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int]:
    crop_rgb = img.crop((x1, y1, x2, y2))
    crop_gray = base_gray.crop((x1, y1, x2, y2))
    light_mask = crop_gray.point(lambda p: 255 if p >= 200 else 0)
    if not light_mask.getbbox():
        light_mask.close()
        crop_rgb.close()
        crop_gray.close()
        return (255, 255, 255)

    stat_g = ImageStat.Stat(crop_gray, mask=light_mask)
    if stat_g.median[0] >= 244:
        light_mask.close()
        crop_rgb.close()
        crop_gray.close()
        return (255, 255, 255)

    stat_rgb = ImageStat.Stat(crop_rgb, mask=light_mask)
    bg_rgb = tuple(int(v) for v in stat_rgb.median[:3])
    light_mask.close()
    crop_rgb.close()
    crop_gray.close()
    return bg_rgb


def _find_enclosed_dark_caption_box(
    sub_gray: Image.Image,
    rx1_rel: int,
    ry1_rel: int,
    rx2_rel: int,
    ry2_rel: int,
) -> tuple[int, int, int, int] | None:
    ww, wh = sub_gray.size
    box_w = max(8, rx2_rel - rx1_rel)
    box_h = max(8, ry2_rel - ry1_rel)
    if box_w < 14 or box_h < 20:
        return None

    cap_bin = sub_gray.point(lambda p: 255 if p <= 165 else 0)
    seed_pt = None
    for fy in (0.50, 0.25, 0.75):
        sy = max(1, min(wh - 2, int(ry1_rel + box_h * fy)))
        for fx in (0.50, 0.25, 0.75):
            sx = max(1, min(ww - 2, int(rx1_rel + box_w * fx)))
            if cap_bin.getpixel((sx, sy)) == 255:
                seed_pt = (sx, sy)
                break
        if seed_pt:
            break

    if seed_pt is None:
        cap_bin.close()
        return None

    ImageDraw.floodfill(cap_bin, seed_pt, 128)
    comp = cap_bin.point(lambda p: 255 if p == 128 else 0)
    cap_bin.close()

    padded = Image.new("L", (ww + 4, wh + 4), 0)
    padded.paste(comp, (2, 2))
    ImageDraw.floodfill(padded, (0, 0), 128)
    cropped_pad = padded.crop((2, 2, ww + 2, wh + 2))
    padded.close()
    holes = cropped_pad.point(lambda p: 255 if p == 0 else 0)
    cropped_pad.close()
    filled = ImageChops.lighter(comp, holes)
    comp.close()
    holes.close()

    c_bbox = filled.getbbox()
    if not c_bbox:
        filled.close()
        return None
    bx1, by1, bx2, by2 = c_bbox
    bw, bh = bx2 - bx1, by2 - by1

    if bw > box_w * 3.2 or bh > box_h * 1.45 or bw > ww * 0.65 or bh > wh * 0.65:
        filled.close()
        return None

    filled_crop = filled.crop(c_bbox)
    area = filled_crop.histogram()[255]
    filled_crop.close()
    filled.close()

    rect_fill_ratio = area / float(max(1, bw * bh))
    if rect_fill_ratio >= 0.76:
        return bx1 + 3, by1 + 3, bx2 - 3, by2 - 3
    return None


def _build_red_box_qa_crop_b64(img: Image.Image, x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Extracts a lightweight 320px context crop around `(x1, y1, x2, y2)` with a bright red
    rectangle marking the exact proposed `box_2d` for `gemini-decision-1` pre-render audit.
    """
    width, height = img.size
    bw = max(16, x2 - x1)
    bh = max(16, y2 - y1)
    pad_x = max(45, int(bw * 0.55))
    pad_y = max(45, int(bh * 0.55))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(width, x2 + pad_x)
    cy2 = min(height, y2 + pad_y)

    crop = img.crop((cx1, cy1, cx2, cy2)).copy()
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
        outline=(255, 25, 25),
        width=max(2, int(max(crop.size) * 0.008)),
    )
    crop.thumbnail((320, 320), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=76)
    crop.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_ambiguous_qa_crops_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
) -> tuple[list[tuple[int, int, int, int, bool, bool, str, str]], list[tuple[int, str]]]:
    """
    Fast pre-scan before rendering: identifies only ambiguous/risky boxes (boxes with
    significant chromatic/skin pixels `chroma_ratio >= 0.08` OR non-enclosed Case C boxes)
    and prepares tiny 320px red-box crops for `gemini-decision-1` pre-render verification.
    Clean enclosed white speech bubbles (`chroma_ratio < 0.08`) pass with ZERO API calls!
    """
    valid_boxes = [
        b for b in bubbles
        if isinstance(b.get("box_2d"), list)
        and len(b["box_2d"]) == 4
        and _sanitize_text_for_manga_font(b.get("translated_text") or "")
    ]
    if not valid_boxes:
        return [], []

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    width, height = img.size

    try:
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
            if (raw_x2 - raw_x1) < 18 or (raw_y2 - raw_y1) < 16:
                continue
            is_dark = _detect_bubble_polarity(base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
            src_txt = str(b.get("source_text") or "")
            has_kanji_kana = bool(re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]", src_txt))
            clean_tr = _sanitize_text_for_manga_font(b["translated_text"])
            panel_pos = str(b.get("panel_position") or "").strip().lower()
            if clean_tr:
                pixel_boxes.append((raw_x1, raw_y1, raw_x2, raw_y2, is_dark, has_kanji_kana, clean_tr, panel_pos))

        pixel_boxes = _deduplicate_overlapping_pixel_boxes(pixel_boxes, height)

        qa_crops: list[tuple[int, str]] = []
        for idx, (rx1, ry1, rx2, ry2, is_dark, _, _, _) in enumerate(pixel_boxes):
            mean_sat, chroma_ratio = _measure_crop_chroma_stats(img, rx1, ry1, rx2, ry2)
            # Flag for surgical `gemini-decision-1` verification if the box contains chromatic/skin pixels
            # (`chroma_ratio >= 0.08` or `mean_sat >= 11.0`) so drifted boxes over faces/hands are blocked!
            if chroma_ratio >= 0.08 or mean_sat >= 11.0:
                b64_crop = _build_red_box_qa_crop_b64(img, rx1, ry1, rx2, ry2)
                qa_crops.append((idx, b64_crop))

        return pixel_boxes, qa_crops
    finally:
        img.close()
        base_gray.close()


async def _audit_ambiguous_boxes_with_decision_1(
    qa_crops: list[tuple[int, str]],
    page_num: int,
    req_id: str,
) -> set[int]:
    """
    Uses `gemini-decision-1` (200 RPM, ~250ms, zero generateContent quota) to audit ONLY
    suspicious/chromatic `box_2d` crops BEFORE erasing or drawing.
    Returns the set of `idx` indices that MUST BE BLOCKED (character face/hand/body or no Japanese text).
    """
    blocked: set[int] = set()
    if not qa_crops or not GEMINI_API_KEY or time.time() < state.gemini_decision_cooldown_until:
        return blocked

    sem = asyncio.Semaphore(6)

    async def _check_one(box_idx: int, b64_crop: str) -> tuple[int, bool]:
        shared_input = [
            {
                "type": "text",
                "text": (
                    "Inspect the area strictly INSIDE the red rectangle on this manga crop.\n"
                    "Determine if erasing inside the red rectangle would safely erase Japanese text "
                    "or if it would damage a character's face/eyes/hands/body or erase blank artwork."
                ),
            },
            {
                "type": "image",
                "data": b64_crop,
                "mime_type": "image/jpeg",
                "resolution": "low",
            },
        ]
        questions = {
            "has_japanese_text": {
                "type": "noul",
                "instructions": (
                    "Answer YES if there is clearly readable Japanese dialogue or narration text INSIDE the red rectangle. "
                    "Answer NO if the red rectangle contains mostly character artwork, blank space, or a face/hand."
                ),
            },
            "damages_character_art": {
                "type": "noul",
                "instructions": (
                    "Answer YES if the red rectangle sits directly over a character's face, eyes, nose, hair, fingers, hand, or body "
                    "outside a speech bubble. Answer NO if it is cleanly inside a speech bubble or caption box."
                ),
            },
        }
        async with sem:
            answers = await call_gemini_decision_orchestrator(
                shared_input,
                questions,
                timeout_sec=3.5,
                purpose=f"BoxQA[{req_id}]-p{page_num}b{box_idx + 1}",
            )
        if not answers:
            return box_idx, False
        has_txt = float((answers.get("has_japanese_text") or {}).get("noul") or 1.0)
        damages_art = float((answers.get("damages_character_art") or {}).get("noul") or 0.0)
        should_block = bool(has_txt < 0.45 or damages_art >= 0.58)
        return box_idx, should_block

    results = await asyncio.gather(*[_check_one(idx, b64_c) for idx, b64_c in qa_crops[:8]])
    for box_idx, should_block in results:
        if should_block:
            blocked.add(box_idx)
    return blocked


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
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    """
    Fast, artifact-free Japanese text erasure & bubble interior measurement:
      - Proportional crop padding (`0.24 * W`, `0.22 * H`) so tall full-res ovals are never cut open.
      - 0.5x mask dilation + C-level `getbbox()` row scanning (~10x faster on 0.1 vCPU).
      - Chromatic skin/art guard (`chroma_ratio`) + skin-edge protection in Case C so fingers/faces
        are NEVER damaged by white artifacts.
    """
    width, height = img.size
    x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2
    box_w = max(8, x2 - x1)
    box_h = max(8, y2 - y1)

    if box_w < 18 or box_h < 16:
        return None

    # Proportional local window so tall full-res ovals (e.g. 500px tall on 2100px canvas) stay 100% enclosed
    pad_x = min(int(width * 0.24), max(120, int(box_w * 1.40)))
    pad_y = min(int(height * 0.22), max(120, int(box_h * 1.40)))
    wx1 = max(0, x1 - pad_x)
    wy1 = max(0, y1 - pad_y)
    wx2 = min(width, x2 + pad_x)
    wy2 = min(height, y2 + pad_y)

    ww = max(8, wx2 - wx1)
    wh = max(8, wy2 - wy1)
    sub_gray = base_gray.crop((wx1, wy1, wx2, wy2))

    rx1_rel = max(2, x1 - wx1)
    ry1_rel = max(2, y1 - wy1)
    rx2_rel = min(ww - 2, x2 - wx1)
    ry2_rel = min(wh - 2, y2 - wy1)

    peer_clip_x1, peer_clip_y1, peer_clip_x2, peer_clip_y2 = _compute_peer_clip_bounds(
        base_gray, width, height, x1, y1, x2, y2, other_peer_boxes
    )

    try:
        # =====================================================================
        # CASE A: DARK OR CHARCOAL BUBBLE / NARRATION CAPTION BOX
        # =====================================================================
        if is_dark_bubble:
            mean_sat, chroma_ratio = _measure_crop_chroma_stats(img, x1, y1, x2, y2)
            sub_g_a = base_gray.crop((x1, y1, x2, y2))
            box_hist = sub_g_a.histogram()
            sub_g_a.close()
            total_box_px = max(1, sum(box_hist))
            bright_ratio = sum(box_hist[205:]) / total_box_px
            if mean_sat >= 36.0 or bright_ratio < 0.008:
                return None

            dark_fill_rgb, bg_lum = _sample_dark_bubble_bg_rgb_and_lum(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
            fg_text_rgb = (250, 250, 250)
            stroke_rgb = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))

            cap_rect = _find_enclosed_dark_caption_box(
                sub_gray, rx1_rel, ry1_rel, rx2_rel, ry2_rel
            )
            if cap_rect is not None:
                tx_clean_x1 = max(2, cap_rect[0])
                tx_clean_y1 = max(2, cap_rect[1])
                tx_clean_x2 = min(ww - 2, cap_rect[2])
                tx_clean_y2 = min(wh - 2, cap_rect[3])
            else:
                tx_clean_x1 = max(2, rx1_rel - 6)
                tx_clean_y1 = max(2, ry1_rel - 6)
                tx_clean_x2 = min(ww - 2, rx2_rel + 6)
                tx_clean_y2 = min(wh - 2, ry2_rel + 6)

            text_box_bounds = Image.new("L", (ww, wh), 0)
            ImageDraw.Draw(text_box_bounds).rectangle(
                [tx_clean_x1, tx_clean_y1, tx_clean_x2, tx_clean_y2], fill=255
            )

            diff_from_bg = sub_gray.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
            diff_dilated = _fast_dilate_mask_half_res(diff_from_bg, kernel_half=3)
            diff_from_bg.close()
            erase_mask_dark = ImageChops.darker(diff_dilated, text_box_bounds)
            diff_dilated.close()
            text_box_bounds.close()

            fill_patch = Image.new("RGB", (ww, wh), dark_fill_rgb)
            img.paste(fill_patch, (wx1, wy1), mask=erase_mask_dark)
            fill_patch.close()
            erase_mask_dark.close()

            if cap_rect is not None:
                hard_b = (
                    max(peer_clip_x1, wx1 + cap_rect[0]),
                    max(peer_clip_y1, wy1 + cap_rect[1]),
                    min(peer_clip_x2, wx1 + cap_rect[2]),
                    min(peer_clip_y2, wy1 + cap_rect[3]),
                )
                return hard_b[0], hard_b[1], hard_b[2], hard_b[3], fg_text_rgb, stroke_rgb, hard_b

            hard_b = (
                max(peer_clip_x1, raw_x1),
                max(peer_clip_y1, raw_y1),
                min(peer_clip_x2, raw_x2),
                min(peer_clip_y2, raw_y2),
            )
            return hard_b[0], hard_b[1], hard_b[2], hard_b[3], fg_text_rgb, stroke_rgb, hard_b

        # =====================================================================
        # CASE B: ENCLOSED PURE-WHITE SPEECH BUBBLE
        # =====================================================================
        filled_bubble = _find_best_enclosed_white_bubble(
            sub_gray, rx1_rel, ry1_rel, rx2_rel, ry2_rel, has_kanji_kana=has_kanji_kana
        )

        if filled_bubble is not None:
            safe_inner_core = filled_bubble.filter(ImageFilter.MinFilter(3))
            filled_bubble.close()
            lobe_bbox = safe_inner_core.getbbox()
            if lobe_bbox:
                peer_mask = Image.new("L", (ww, wh), 0)
                pm_x1 = max(0, peer_clip_x1 - wx1)
                pm_y1 = max(0, peer_clip_y1 - wy1)
                pm_x2 = min(ww, peer_clip_x2 - wx1)
                pm_y2 = min(wh, peer_clip_y2 - wy1)
                if pm_x2 > pm_x1 and pm_y2 > pm_y1:
                    ImageDraw.Draw(peer_mask).rectangle([pm_x1, pm_y1, pm_x2, pm_y2], fill=255)
                else:
                    ImageDraw.Draw(peer_mask).rectangle([0, 0, ww, wh], fill=255)

                clipped_core = ImageChops.darker(safe_inner_core, peer_mask)
                safe_inner_core.close()
                peer_mask.close()

                ink_inside_bubble = sub_gray.point(lambda p: 255 if p < 242 else 0)
                ink_dilated = _fast_dilate_mask_half_res(ink_inside_bubble, kernel_half=3)
                ink_inside_bubble.close()
                erase_mask = ImageChops.darker(ink_dilated, clipped_core)
                ink_dilated.close()

                white_patch = Image.new("RGB", (ww, wh), (255, 255, 255))
                img.paste(white_patch, (wx1, wy1), mask=erase_mask)
                white_patch.close()
                erase_mask.close()

                clipped_bbox = clipped_core.getbbox() or lobe_bbox
                lb_x1, lb_y1, lb_x2, lb_y2 = clipped_bbox

                rel_y1 = lb_y1 + max(4, int((lb_y2 - lb_y1) * 0.08))
                rel_y2 = lb_y2 - max(4, int((lb_y2 - lb_y1) * 0.08))
                if rel_y2 - rel_y1 < 18:
                    rel_y1, rel_y2 = lb_y1 + 3, lb_y2 - 3

                # Fast C-level row horizontal span measurement via `getbbox()`
                mid_y_start = rel_y1 + int((rel_y2 - rel_y1) * 0.16)
                mid_y_end = rel_y1 + int((rel_y2 - rel_y1) * 0.84)
                step_y = max(2, (mid_y_end - mid_y_start) // 7 or 1)
                row_lefts = []
                row_rights = []
                for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), step_y):
                    if not (0 <= sy < wh):
                        continue
                    row_strip = clipped_core.crop((lb_x1, sy, lb_x2, sy + 1))
                    rb = row_strip.getbbox()
                    row_strip.close()
                    if rb and (rb[2] - rb[0]) >= 14:
                        row_lefts.append(lb_x1 + rb[0])
                        row_rights.append(lb_x1 + rb[2])
                clipped_core.close()

                if row_lefts and row_rights:
                    row_lefts.sort()
                    row_rights.sort()
                    inner_l = row_lefts[int(len(row_lefts) * 0.65)] + 4
                    inner_r = row_rights[int(len(row_rights) * 0.35)] - 4
                    if inner_r - inner_l >= 18:
                        tx1, ty1, tx2, ty2 = wx1 + inner_l, wy1 + rel_y1, wx1 + inner_r, wy1 + rel_y2
                        hard_b = (wx1 + lb_x1 + 3, wy1 + lb_y1 + 3, wx1 + lb_x2 - 3, wy1 + lb_y2 - 3)
                        return tx1, ty1, tx2, ty2, (15, 15, 15), (255, 255, 255), hard_b

                tx1, ty1, tx2, ty2 = wx1 + lb_x1 + 4, wy1 + rel_y1, wx1 + lb_x2 - 4, wy1 + rel_y2
                hard_b = (wx1 + lb_x1 + 3, wy1 + lb_y1 + 3, wx1 + lb_x2 - 3, wy1 + lb_y2 - 3)
                return tx1, ty1, tx2, ty2, (15, 15, 15), (255, 255, 255), hard_b
            else:
                safe_inner_core.close()

        # =====================================================================
        # CASE C: BORDER-TOUCHING WHITE OR SEMI-TRANSPARENT MIST BUBBLE
        # (Chromatic-pixel rejection + skin-outline protection so hands, arms,
        #  faces, and colored art are NEVER turned into white artifacts!)
        # =====================================================================
        tx_b1 = max(2, rx1_rel - 4)
        ty_b1 = max(2, ry1_rel - 4)
        tx_b2 = min(ww - 2, rx2_rel + 4)
        ty_b2 = min(wh - 2, ry2_rel + 4)

        mean_sat_c, chroma_ratio_c = _measure_crop_chroma_stats(img, x1, y1, x2, y2)

        box_crop = sub_gray.crop((tx_b1, ty_b1, tx_b2, ty_b2))
        box_hist = box_crop.histogram()
        box_crop.close()
        total_box_px = max(1, sum(box_hist))
        white_paper_px = sum(box_hist[230:])
        midtone_px = sum(box_hist[135:230])
        dark_ink_px = sum(box_hist[:135])

        white_paper_ratio = white_paper_px / total_box_px
        midtone_ratio = midtone_px / total_box_px
        ink_ratio = dark_ink_px / total_box_px

        # Strict `chroma_ratio_c < 0.12` blocks pale skin/hands/faces even when surrounded by white background!
        is_valid_pure_white = (
            mean_sat_c < 10.0
            and chroma_ratio_c < 0.12
            and white_paper_ratio >= 0.45
            and midtone_ratio <= 0.24
            and ink_ratio >= 0.015
        )
        light_mist_px = sum(box_hist[205:])
        is_valid_mist_bubble = (
            mean_sat_c < 7.0
            and chroma_ratio_c < 0.08
            and (light_mist_px / total_box_px) >= 0.62
            and ink_ratio >= 0.015
        )

        if is_valid_pure_white or is_valid_mist_bubble:
            exp_x1 = max(2, max(peer_clip_x1 - wx1, rx1_rel - 8))
            exp_y1 = max(2, max(peer_clip_y1 - wy1, ry1_rel - 14))
            exp_x2 = min(ww - 2, min(peer_clip_x2 - wx1, rx2_rel + 8))
            exp_y2 = min(wh - 2, min(peer_clip_y2 - wy1, ry2_rel + 14))

            box_clamp = Image.new("L", (ww, wh), 0)
            ImageDraw.Draw(box_clamp).rectangle([exp_x1, exp_y1, exp_x2, exp_y2], fill=255)

            bg_rgb = (255, 255, 255) if is_valid_pure_white else _sample_light_bubble_bg_rgb(
                img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2
            )
            ink_cutoff = 228 if is_valid_pure_white else 188
            text_ink_mask = sub_gray.point(lambda p: 255 if p < ink_cutoff else 0)
            text_ink_dil = _fast_dilate_mask_half_res(text_ink_mask, kernel_half=3)
            text_ink_mask.close()

            # Build chromatic-art shield: protect any pixel within ~5px of colored skin/hair/clothes (`sat >= 20`)
            sub_rgb_shield = img.crop((wx1, wy1, wx2, wy2))
            sub_sat_band = sub_rgb_shield.convert("HSV").getchannel("S")
            sub_rgb_shield.close()
            chroma_px = sub_sat_band.point(lambda p: 255 if p >= 20 else 0)
            sub_sat_band.close()
            chroma_shield = _fast_dilate_mask_half_res(chroma_px, kernel_half=5)
            chroma_px.close()
            non_chroma_mask = ImageChops.invert(chroma_shield)
            chroma_shield.close()

            safe_clamp = ImageChops.darker(box_clamp, non_chroma_mask)
            box_clamp.close()
            non_chroma_mask.close()

            if is_valid_pure_white:
                white_seed = sub_gray.point(lambda p: 255 if p >= 235 else 0)
                white_zone = _fast_dilate_mask_half_res(white_seed, kernel_half=5)
                white_seed.close()
                safe_zone = ImageChops.darker(white_zone, safe_clamp)
                white_zone.close()
                safe_clamp.close()
                erase_c = ImageChops.darker(text_ink_dil, safe_zone)
                safe_zone.close()
            else:
                erase_c = ImageChops.darker(text_ink_dil, safe_clamp)
                safe_clamp.close()

            text_ink_dil.close()

            if erase_c.getbbox():
                bg_patch = Image.new("RGB", (ww, wh), bg_rgb)
                img.paste(bg_patch, (wx1, wy1), mask=erase_c)
                bg_patch.close()
                erase_c.close()
                hard_b = (raw_x1, raw_y1, raw_x2, raw_y2)
                return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), bg_rgb, hard_b
            erase_c.close()

        return None
    finally:
        sub_gray.close()


def _deduplicate_overlapping_pixel_boxes(
    pixel_boxes: list[tuple[int, int, int, int, bool, bool, str, str]],
    height: int,
) -> list[tuple[int, int, int, int, bool, bool, str, str]]:
    def _pos_matches_y(pos: str, y1: int, y2: int) -> bool:
        if not pos:
            return True
        y_norm = ((y1 + y2) / 2.0) / max(1.0, float(height)) * 1000.0
        if pos.startswith("top"):
            return y_norm <= 480
        if pos.startswith("middle"):
            return 240 <= y_norm <= 740
        if pos.startswith("bottom"):
            return y_norm >= 520
        return True

    kept: list[tuple[int, int, int, int, bool, bool, str, str]] = []
    for item in pixel_boxes:
        rx1, ry1, rx2, ry2, _, _, _, r_pos = item
        area_a = max(1, (rx2 - rx1) * (ry2 - ry1))
        match_a = _pos_matches_y(r_pos, ry1, ry2)

        replaced_or_dropped = False
        for idx_k, k_item in enumerate(kept):
            kx1, ky1, kx2, ky2, _, _, _, k_pos = k_item
            area_b = max(1, (kx2 - kx1) * (ky2 - ky1))
            ov_x = max(0, min(rx2, kx2) - max(rx1, kx1))
            ov_y = max(0, min(ry2, ky2) - max(ry1, ky1))
            ov_area = ov_x * ov_y
            if ov_area / float(min(area_a, area_b)) >= 0.55:
                match_b = _pos_matches_y(k_pos, ky1, ky2)
                if (match_a and not match_b) or (match_a == match_b and area_a > area_b * 1.15):
                    kept[idx_k] = item
                replaced_or_dropped = True
                break

        if not replaced_or_dropped:
            kept.append(item)
    return kept


def _resolve_box_collisions(
    renders: list[list],
    min_gap: int = 6,
) -> list[list]:
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
    """
    Finds the exact largest font size in `[9..start_font_size]` that fits `(usable_w, usable_h)`
    in O(log N) (~5 iterations instead of 45), speeding up full-res typesetting by ~8x.
    """
    sample_probe = "Agกข้ำ" if is_thai_text else "Agjpqy!"

    def _measure(fsize: int) -> tuple[bool, ImageFont.ImageFont, list[str], int, int]:
        font = _load_manga_font(fsize, is_thai=is_thai_text)
        wrapped = _wrap_text_to_pixel_width(
            draw, trans_text, font, usable_w, allow_midword_split=(fsize <= 9)
        )
        sample_bbox = draw.textbbox((0, 0), sample_probe, font=font)
        lh = max(fsize + 3, int((sample_bbox[3] - sample_bbox[1]) * 1.18))
        total_h = lh * len(wrapped)
        max_line_w = max(
            (draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0])
            for ln in wrapped
        )
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


def render_translated_manga_image_sync(
    raw_bytes: bytes,
    bubbles: list[dict],
    approved_pixel_boxes: list[tuple[int, int, int, int, bool, bool, str, str]] | None = None,
) -> bytes | None:
    """
    Renders translated dialogue cleanly inside speech bubbles at FULL ORIGINAL RESOLUTION
    using fast 0.5x morphological masks, C-level row bounds, and binary-search font fitting.
    """
    if not raw_bytes:
        return None

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    width, height = img.size

    try:
        if approved_pixel_boxes is not None:
            pixel_boxes = approved_pixel_boxes
        else:
            valid_boxes = [
                b for b in bubbles
                if isinstance(b.get("box_2d"), list)
                and len(b["box_2d"]) == 4
                and _sanitize_text_for_manga_font(b.get("translated_text") or "")
            ]
            if not valid_boxes:
                return None

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
                clean_tr = _sanitize_text_for_manga_font(b["translated_text"])
                panel_pos = str(b.get("panel_position") or "").strip().lower()
                if clean_tr:
                    pixel_boxes.append((raw_x1, raw_y1, raw_x2, raw_y2, is_dark, has_kanji_kana, clean_tr, panel_pos))

            pixel_boxes = _deduplicate_overlapping_pixel_boxes(pixel_boxes, height)

        if not pixel_boxes:
            return None

        prepared_renders = []
        for i, (rx1, ry1, rx2, ry2, is_dark, has_kk, trans_text, _) in enumerate(pixel_boxes):
            if (rx2 - rx1) < 3 or (ry2 - ry1) < 3:
                continue
            other_boxes = [
                (ox1, oy1, ox2, oy2)
                for j, (ox1, oy1, ox2, oy2, _, _, _, _) in enumerate(pixel_boxes)
                if j != i
            ]
            cleaned_lobe = _erase_bubble_japanese_text_and_measure_lobe(
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
            if cleaned_lobe is None:
                continue
            lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned_lobe
            prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, trans_text, hard_bounds])

        if not prepared_renders:
            return None

        prepared_renders = _resolve_box_collisions(prepared_renders, min_gap=6)

        max_dim = max(width, height)
        max_font_px = max(26, int(max_dim * 0.019))

        draw = ImageDraw.Draw(img)
        for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text, _ in prepared_renders:
            box_w = max(18, tx2 - tx1)
            box_h = max(14, ty2 - ty1)
            usable_w = max(14, int(box_w * 0.88))
            usable_h = max(12, int(box_h * 0.90))
            is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

            start_font_size = min(max_font_px, max(12, int(min(box_w, box_h) * 0.26)))
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
        img.save(out_io, format="JPEG", quality=94, subsampling=0)
        return out_io.getvalue()
    finally:
        img.close()
        base_gray.close()


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    """
    Translates 1 to 12 manga pages from direct Discord attachments OR embedded links
    in 1 unified batch call using:
      `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 flash-lite) -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
    with surgical `gemini-decision-1` pre-render crop auditing on chromatic/drifted boxes,
    correlated Render request/waiting/response logs, and fast full-resolution rendering.
    """
    from media_context import resolve_referenced_message, collect_embedded_images_and_metadata

    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel
    ref_msg = await resolve_referenced_message(msg) if msg.reference else None

    image_attachments = [
        a for a in msg.attachments if a.filename.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not image_attachments and ref_msg:
        image_attachments = [
            a for a in ref_msg.attachments
            if a.filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

    pages_input: list[tuple[bytes, str]] = []
    for att in image_attachments[:MAX_MANGA_BATCH_PAGES]:
        raw_b = await att.read()
        pages_input.append((raw_b, att.filename))
    image_attachments.clear()

    embed_meta_sections: list[str] = []
    if not pages_input:
        embedded_imgs, embed_meta_sections, _ = await collect_embedded_images_and_metadata(
            msg, ref_msg=ref_msg, max_images=MAX_MANGA_BATCH_PAGES
        )
        for emb_bytes, emb_fname, _ in embedded_imgs:
            pages_input.append((emb_bytes, emb_fname))
        embedded_imgs.clear()

    if not pages_input:
        await channel.send(
            "กรุณาแนบรูปภาพหน้ามังงะ หรือวางลิงก์ที่มีรูปภาพ (เช่น X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วยนะคะ "
            "(สามารถแนบหรือวางลิงก์ที่มีหลายหน้าพร้อมกันได้เลยค่ะ เช่น พิมพ์ `!manga en` หรือ `!manga th`)"
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

    async with channel.typing():
        batch_results, is_unavailable = await extract_and_translate_manga_pages_batch(
            pages_input, target_lang_name, series_context_hint=series_context_hint, req_id=req_id
        )

        if is_unavailable:
            pages_input.clear()
            release_unused_memory()
            await _send_or_reply(
                channel,
                msg,
                content=(
                    "ขณะนี้ระบบแปลมังงะแบบหลายหน้าไม่พร้อมใช้งานชั่วคราว "
                    "(API ต้นทางขัดข้องหรือหมดเวลาเชื่อมต่อ) กรุณาลองใหม่อีกครั้งในอีกสักครู่นะคะ"
                ),
            )
            return

        if not any(bubbles or summary for bubbles, summary in batch_results):
            pages_input.clear()
            release_unused_memory()
            await _send_or_reply(
                channel,
                msg,
                content="ไม่สามารถตรวจจับข้อความในภาพที่แนบมาหรือจากลิงก์ได้ค่ะ กรุณาตรวจสอบว่าภาพมีความคมชัดนะคะ",
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
        total_qa_audited = 0
        total_qa_blocked = 0
        api_log(f"MangaRender [req={req_id}] START -> rendering {num_pages} page(s)...")

        for page_idx in range(1, num_pages + 1):
            raw_bytes, fname = pages_input[page_idx - 1]
            bubbles, scene_summary = batch_results[page_idx - 1]

            if num_pages > 1:
                transcript_lines.append(f"\n__**📄 หน้าที่ {page_idx}/{num_pages} (`{fname}`)**__")
            if scene_summary:
                clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
                transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

            if not bubbles:
                transcript_lines.append("*(ไม่พบช่องคำพูดในหน้านี้)*")
                pages_input[page_idx - 1] = (b"", fname)
                continue

            for idx, b in enumerate(bubbles, 1):
                speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
                src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
                tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
                if tr_txt:
                    transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n  ↳ **{tr_txt}**")

            try:
                # Step 1: Fast pre-scan & surgical `gemini-decision-1` check ONLY on chromatic/suspicious boxes
                pixel_boxes, qa_crops = await asyncio.to_thread(
                    _extract_ambiguous_qa_crops_sync, raw_bytes, bubbles
                )
                if qa_crops:
                    total_qa_audited += len(qa_crops)
                    blocked_ids = await _audit_ambiguous_boxes_with_decision_1(qa_crops, page_idx, req_id)
                    if blocked_ids:
                        total_qa_blocked += len(blocked_ids)
                        pixel_boxes = [pb for idx_b, pb in enumerate(pixel_boxes) if idx_b not in blocked_ids]

                # Step 2: Render full-resolution page with only verified speech-bubble boxes
                rendered_jpg = await asyncio.to_thread(
                    render_translated_manga_image_sync, raw_bytes, bubbles, pixel_boxes
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

            await asyncio.sleep(0.15)

        render_elapsed = time.monotonic() - render_t0
        api_log(
            f"MangaRender [req={req_id}] DONE -> rendered {len(discord_files)}/{num_pages} page(s) "
            f"in {render_elapsed:.2f}s (BoxQA audited={total_qa_audited}, blocked={total_qa_blocked})"
        )
        transcript_text = "\n".join(transcript_lines)

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
        pages_input.clear()
        batch_results.clear()
        release_unused_memory()