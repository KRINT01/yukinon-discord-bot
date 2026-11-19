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
    call_gemma_ai_studio_chat,
    call_gemma_ai_studio_vision,
    call_cloudflare_gemma_chat,
    call_modal_ollama_chat,
    call_groq_chat,
    call_cloudflare_vision_chat,
    discover_active_groq_vision_models,
)

MAX_MANGA_BATCH_PAGES = 12  # Supports up to 12 pages across multi-part threads (split into <=10 files per Discord message)
SOM_CV_MAX_DIM = 420        # Fast 420px CV detection grid (~60ms/page, zero Discord Gateway lag!)

# Cache for self-healing discovery of strictly newer (>3.5) `flash-lite` models on Google AI Studio
_FLASH_LITE_DISCOVERY_CACHE: list[str] = []
_FLASH_LITE_DISCOVERY_TS: float = 0.0
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
_THAI_PUNCT_ATTACH = set("ๆฯ?!.,:;'\")]-~")


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
# SET-OF-MARK (SoM) CV-ANCHORED MANGA TRANSLATION & FULL-RES TYPESETTING ENGINE
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
    Normalizes Unicode punctuation, strips stray CJK/emoji symbols, and cleans empty `()` / `[]`
    so `Garuda-Bold.ttf` renders every character cleanly with zero `[]` tofu boxes.
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
      3. Any strictly newer (>3.5) `flash-lite` model discovered via self-healing
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


# ------------------------------------------------------------------------------
# STAGE 1: DETERMINISTIC CV SPEECH-BUBBLE DETECTOR + SET-OF-MARK (SoM) OVERLAY
# ------------------------------------------------------------------------------
def _split_concave_double_bubble(
    crop_mask: Image.Image,
    offset_x: int,
    offset_y: int,
    sw: int,
    sh: int,
    depth: int = 0,
) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    """
    Detects connected figure-8 double speech bubbles (horizontal or vertical) via
    projection-profile neck detection and splits them cleanly at the narrowest waist
    into separate single-oval masks.
    """
    cw, ch = crop_mask.size
    if depth >= 2 or (cw < 24 and ch < 24):
        return [((offset_x, offset_y, offset_x + cw, offset_y + ch), crop_mask)]

    area = crop_mask.histogram()[255]
    rect_fill = area / float(max(1, cw * ch))
    if rect_fill >= 0.74:
        return [((offset_x, offset_y, offset_x + cw, offset_y + ch), crop_mask)]

    raw_bytes = crop_mask.tobytes()

    # Check horizontal figure-8 neck (wide double oval)
    best_split = None
    if cw >= max(28, int(sw * 0.14)):
        col_counts = [0] * cw
        for y in range(ch):
            row_off = y * cw
            for x in range(cw):
                if raw_bytes[row_off + x] > 128:
                    col_counts[x] += 1

        s_lo = max(4, int(cw * 0.22))
        s_hi = min(cw - 4, int(cw * 0.78))
        if s_hi > s_lo:
            min_x = min(range(s_lo, s_hi), key=lambda ix: col_counts[ix])
            min_h = col_counts[min_x]
            left_max = max(col_counts[:min_x]) if min_x > 0 else 0
            right_max = max(col_counts[min_x + 1 :]) if (min_x + 1) < cw else 0
            peak_min = min(left_max, right_max)
            if peak_min >= 14 and min_h <= peak_min * 0.60:
                best_split = ("v", min_x, min_h / float(peak_min))

    # Check vertical figure-8 neck (stacked double oval)
    if ch >= max(28, int(sh * 0.12)):
        row_counts = [0] * ch
        for y in range(ch):
            row_off = y * cw
            cnt = 0
            for x in range(cw):
                if raw_bytes[row_off + x] > 128:
                    cnt += 1
            row_counts[y] = cnt

        s_lo = max(4, int(ch * 0.22))
        s_hi = min(ch - 4, int(ch * 0.78))
        if s_hi > s_lo:
            min_y = min(range(s_lo, s_hi), key=lambda iy: row_counts[iy])
            min_w = row_counts[min_y]
            top_max = max(row_counts[:min_y]) if min_y > 0 else 0
            bot_max = max(row_counts[min_y + 1 :]) if (min_y + 1) < ch else 0
            peak_min = min(top_max, bot_max)
            if peak_min >= 14 and min_w <= peak_min * 0.60:
                ratio = min_w / float(peak_min)
                if best_split is None or ratio < best_split[2]:
                    best_split = ("h", min_y, ratio)

    if best_split is None:
        return [((offset_x, offset_y, offset_x + cw, offset_y + ch), crop_mask)]

    axis, cut_pos, _ = best_split
    sliced = crop_mask.copy()
    sdraw = ImageDraw.Draw(sliced)
    if axis == "v":
        sdraw.rectangle([max(0, cut_pos - 1), 0, min(cw - 1, cut_pos + 1), ch - 1], fill=0)
    else:
        sdraw.rectangle([0, max(0, cut_pos - 1), cw - 1, min(ch - 1, cut_pos + 1)], fill=0)

    # Extract separated child components from `sliced`
    children: list[tuple[tuple[int, int, int, int], Image.Image]] = []
    work = sliced
    for sy in range(1, ch - 1, 3):
        for sx in range(1, cw - 1, 3):
            if work.getpixel((sx, sy)) != 255:
                continue
            ImageDraw.floodfill(work, (sx, sy), 128)
            comp = work.point(lambda p: 255 if p == 128 else 0)
            new_work = work.point(lambda p: 0 if p == 128 else p)
            work.close()
            work = new_work

            bb = comp.getbbox()
            if not bb:
                comp.close()
                continue
            bw_c = bb[2] - bb[0]
            bh_c = bb[3] - bb[1]
            sub_m = comp.crop(bb)
            comp.close()
            sub_area = sub_m.histogram()[255]
            if bw_c >= 10 and bh_c >= 10 and sub_area >= 110:
                children.extend(
                    _split_concave_double_bubble(
                        sub_m, offset_x + bb[0], offset_y + bb[1], sw, sh, depth=depth + 1
                    )
                )
            else:
                sub_m.close()
    work.close()

    if len(children) >= 2:
        crop_mask.close()
        return children

    for _, cm in children:
        cm.close()
    return [((offset_x, offset_y, offset_x + cw, offset_y + ch), crop_mask)]


def _sort_bubbles_manga_reading_order(bubbles: list[dict], sw: int, sh: int) -> list[dict]:
    """
    Sorts detected CV bubbles in natural Japanese Manga reading order:
    Top-to-Bottom panel rows (grouping bubbles whose vertical spans overlap significantly),
    and Right-to-Left within each row!
    """
    if len(bubbles) <= 1:
        return bubbles

    # Sort primarily by top edge `y1` so we can form horizontal reading bands
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
        # Within each band, sort Right-to-Left by horizontal center, breaking ties Top-to-Bottom
        band.sort(key=lambda x: (-((x["bbox_norm"][1] + x["bbox_norm"][3]) / 2.0), x["bbox_norm"][0]))
        result.extend(band)
    return result


def _detect_page_bubbles_and_build_som_b64_sync(
    raw_bytes: bytes,
    vision_max_dim: int = 1024,
    vision_quality: int = 84,
) -> tuple[str, list[dict]]:
    """
    Detects enclosed white speech bubbles/caption boxes (and dark speech bubbles) directly
    via fast C-level PIL connected components at 420px scale, splits figure-8 double bubbles
    at their narrow waists, sorts them in Japanese Manga reading order, and draws crisp red
    numbered Set-of-Mark (SoM) badges `[1]`, `[2]`, ... onto the vision JPEG sent to Gemini.

    Returns `(b64_som_jpeg, detected_cv_bubbles)` where each item in `detected_cv_bubbles` has:
      - `bubble_id`: 1-based integer matching the red badge on the image
      - `bbox_norm`: `[ymin, xmin, ymax, xmax]` in `0..1000`
      - `mask_small`: cropped L-mode mask (`255` = bubble interior)
      - `is_dark`: bool
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    try:
        # 1. Fast 420px detection buffers
        scale = min(1.0, SOM_CV_MAX_DIM / float(max(orig_w, orig_h)))
        sw = max(32, int(round(orig_w * scale)))
        sh = max(32, int(round(orig_h * scale)))
        small_rgb = img.resize((sw, sh), Image.Resampling.BILINEAR)
        small_gray = small_rgb.convert("L")
        small_sat = small_rgb.convert("HSV").getchannel("S")
        small_rgb.close()

        detected: list[dict] = []

        # --- Pass A: Enclosed White Speech Bubbles & White Rectangular Caption Boxes ---
        white_bin = small_gray.point(lambda p: 255 if p >= 232 else 0)
        work = white_bin.copy()
        white_bin.close()

        min_w_px = max(11, int(sw * 0.040))
        min_h_px = max(11, int(sh * 0.028))
        max_area_px = int(sw * sh * 0.24)

        for sy in range(2, sh - 2, 4):
            for sx in range(2, sw - 2, 4):
                if work.getpixel((sx, sy)) != 255:
                    continue

                ImageDraw.floodfill(work, (sx, sy), 128)
                comp = work.point(lambda p: 255 if p == 128 else 0)
                new_work = work.point(lambda p: 0 if p == 128 else p)
                work.close()
                work = new_work

                c_bbox = comp.getbbox()
                if not c_bbox:
                    comp.close()
                    continue
                cx1, cy1, cx2, cy2 = c_bbox
                cw, ch = cx2 - cx1, cy2 - cy1

                # Reject outer page margin / full-page backgrounds & tiny speckles
                edge_touches = (
                    (1 if cx1 <= 1 else 0)
                    + (1 if cy1 <= 1 else 0)
                    + (1 if cx2 >= sw - 1 else 0)
                    + (1 if cy2 >= sh - 1 else 0)
                )
                if (
                    edge_touches >= 2
                    or cw > sw * 0.72
                    or ch > sh * 0.65
                    or cw < min_w_px
                    or ch < min_h_px
                ):
                    comp.close()
                    continue

                # Hole-fill inside the local crop (2px padding ensures complete exterior floodfill)
                comp_crop = comp.crop(c_bbox)
                comp.close()

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

                # Must contain interior text strokes (`holes_area >= 4`) and not be an entire panel
                if (
                    filled_area > max_area_px
                    or holes_area < 4
                    or (comp_area / float(filled_area)) < 0.25
                    or (holes_area / float(filled_area)) > 0.68
                ):
                    filled_crop.close()
                    continue

                # Verify bubble interior is achromatic (white paper + dark ink, NOT colored food/skin/art)
                sat_crop = small_sat.crop(c_bbox)
                gray_crop = small_gray.crop(c_bbox)
                sat_hist = sat_crop.histogram(mask=filled_crop)
                total_m_px = max(1, sum(sat_hist))
                chroma_ratio = sum(sat_hist[24:]) / float(total_m_px)
                mean_sat = ImageStat.Stat(sat_crop, mask=filled_crop).mean[0]

                # Verify there is actual dark text ink (`gray < 175` & `sat < 22`) inside `filled_crop`
                dark_ink_bin = gray_crop.point(lambda p: 255 if p < 175 else 0)
                achrom_bin = sat_crop.point(lambda p: 255 if p < 22 else 0)
                achrom_ink = ImageChops.darker(ImageChops.darker(dark_ink_bin, achrom_bin), filled_crop)
                dark_ink_px = achrom_ink.histogram()[255]

                dark_ink_bin.close()
                achrom_bin.close()
                achrom_ink.close()
                sat_crop.close()
                gray_crop.close()

                if chroma_ratio >= 0.20 or mean_sat >= 20.0 or dark_ink_px < 4:
                    filled_crop.close()
                    continue

                # Split connected figure-8 double bubbles cleanly at their narrow waists
                sub_lobes = _split_concave_double_bubble(filled_crop, cx1, cy1, sw, sh, depth=0)
                for (lx1, ly1, lx2, ly2), lobe_mask in sub_lobes:
                    lw, lh = lx2 - lx1, ly2 - ly1
                    if lw < min_w_px or lh < min_h_px:
                        lobe_mask.close()
                        continue
                    # Re-verify each split lobe actually contains dark text ink
                    l_gray = small_gray.crop((lx1, ly1, lx2, ly2))
                    l_sat = small_sat.crop((lx1, ly1, lx2, ly2))
                    l_ink = ImageChops.darker(
                        ImageChops.darker(
                            l_gray.point(lambda p: 255 if p < 175 else 0),
                            l_sat.point(lambda p: 255 if p < 22 else 0),
                        ),
                        lobe_mask,
                    )
                    l_ink_px = l_ink.histogram()[255]
                    l_ink.close()
                    l_gray.close()
                    l_sat.close()
                    if l_ink_px < 4:
                        lobe_mask.close()
                        continue

                    bbox_norm = [
                        int(round((ly1 / float(sh)) * 1000.0)),
                        int(round((lx1 / float(sw)) * 1000.0)),
                        int(round((ly2 / float(sh)) * 1000.0)),
                        int(round((lx2 / float(sw)) * 1000.0)),
                    ]
                    detected.append({
                        "bbox_norm": bbox_norm,
                        "mask_small": lobe_mask,
                        "is_dark": False,
                    })

        work.close()

        # --- Pass B: Enclosed Dark Speech Bubbles / Dark Caption Boxes with Bright White Text ---
        dark_bin = small_gray.point(lambda p: 255 if p <= 95 else 0)
        work_d = dark_bin.copy()
        dark_bin.close()

        for sy in range(2, sh - 2, 5):
            for sx in range(2, sw - 2, 5):
                if work_d.getpixel((sx, sy)) != 255:
                    continue
                ImageDraw.floodfill(work_d, (sx, sy), 128)
                comp = work_d.point(lambda p: 255 if p == 128 else 0)
                new_wd = work_d.point(lambda p: 0 if p == 128 else p)
                work_d.close()
                work_d = new_wd

                c_bbox = comp.getbbox()
                if not c_bbox:
                    comp.close()
                    continue
                cx1, cy1, cx2, cy2 = c_bbox
                cw, ch = cx2 - cx1, cy2 - cy1
                if cw < min_w_px or ch < min_h_px or cw > sw * 0.55 or ch > sh * 0.55:
                    comp.close()
                    continue

                comp_crop = comp.crop(c_bbox)
                comp.close()
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

                g_crop = small_gray.crop(c_bbox)
                s_crop = small_sat.crop(c_bbox)
                mean_s = ImageStat.Stat(s_crop, mask=filled_crop).mean[0]
                bright_txt = ImageChops.darker(g_crop.point(lambda p: 255 if p >= 210 else 0), filled_crop)
                bright_px = bright_txt.histogram()[255]
                bright_txt.close()
                g_crop.close()
                s_crop.close()

                if mean_s < 22.0 and 0.02 <= (bright_px / float(filled_area)) <= 0.42:
                    bbox_norm = [
                        int(round((cy1 / float(sh)) * 1000.0)),
                        int(round((cx1 / float(sw)) * 1000.0)),
                        int(round((cy2 / float(sh)) * 1000.0)),
                        int(round((cx2 / float(sw)) * 1000.0)),
                    ]
                    detected.append({
                        "bbox_norm": bbox_norm,
                        "mask_small": filled_crop,
                        "is_dark": True,
                    })
                else:
                    filled_crop.close()

        work_d.close()
        small_gray.close()
        small_sat.close()

        # Sort in Manga reading order & assign 1-based `bubble_id`
        detected = _sort_bubbles_manga_reading_order(detected, sw, sh)
        for idx, b_obj in enumerate(detected, 1):
            b_obj["bubble_id"] = idx

        # 2. Build annotated Set-of-Mark JPEG copy for Gemini Vision
        vis_img = img.copy()
        vis_img.thumbnail((vision_max_dim, vision_max_dim), Image.Resampling.LANCZOS)
        vw, vh = vis_img.size
        vdraw = ImageDraw.Draw(vis_img)
        badge_font_size = max(15, int(vw * 0.022))
        badge_font = _load_manga_font(badge_font_size, is_thai=False)

        for b_obj in detected:
            bid = b_obj["bubble_id"]
            ymin, xmin, ymax, xmax = b_obj["bbox_norm"]
            vx1 = max(0, int((xmin / 1000.0) * vw))
            vy1 = max(0, int((ymin / 1000.0) * vh))
            vx2 = min(vw - 1, int((xmax / 1000.0) * vw))
            vy2 = min(vh - 1, int((ymax / 1000.0) * vh))

            # Draw crisp red contour of the actual detected oval/rectangle mask
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

            # Draw bold red pill badge with white ID number at the top-right edge of the bubble
            label = str(bid)
            tb = vdraw.textbbox((0, 0), label, font=badge_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            pw = tw + 10
            ph = th + 6
            bx2 = min(vw - 1, vx2 + 2)
            bx1 = max(0, bx2 - pw)
            by1 = max(0, vy1 - (ph // 3))
            by2 = min(vh - 1, by1 + ph)
            vdraw.rectangle([bx1, by1, bx2, by2], fill=(225, 15, 15), outline=(255, 255, 255), width=1)
            vdraw.text(
                (bx1 + 5 - tb[0], by1 + 3 - tb[1]),
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


# ------------------------------------------------------------------------------
# STAGE 2: JSON NORMALIZATION & STRICT BATCH VISION ORCHESTRATION
# ------------------------------------------------------------------------------
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

        is_dialogue = item.get("is_dialogue")
        if is_dialogue is False:
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


async def _waiting_heartbeat_logger(req_id: str, label: str, model_id: str, interval_sec: float = 15.0):
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
) -> tuple[list[tuple[list[dict], str]], list[list[dict]], bool]:
    """
    Set-of-Mark (SoM) CV-Anchored Multi-Page Manga Batch Pipeline:
      1) Detects enclosed speech bubbles on every page via fast CV (`<=420px`) and draws
         red numbered badges `[1]`, `[2]`, ... on the vision copies.
      2) Calls `MangaBatch` strictly across:
         `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 `flash-lite`)
         -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
         NEVER falling back to `MangaSingle` for multi-page batches!
      3) Returns `(batch_results, cv_bubbles_per_page, is_service_unavailable)`.
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return [], [], False

    req_id = req_id or f"mb-{uuid.uuid4().hex[:6]}"

    if num_pages == 1:
        b, s, cv_list = await extract_and_translate_manga_bubbles(
            pages_input[0][0],
            pages_input[0][1],
            target_lang_name,
            series_context_hint=series_context_hint,
            req_id=req_id,
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
    api_log(
        f"MangaBatch({num_pages}p) [req={req_id}] CV_DETECT -> found {total_cv} enclosed bubble(s) "
        f"across {num_pages} page(s) in {time.monotonic() - som_t0:.2f}s"
    )

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster (use accurate names & terms):\n{series_context_hint[:1400]}\n"
        if series_context_hint else ""
    )

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
Every enclosed speech bubble and caption box has ALREADY been outlined in RED with a RED NUMBERED BADGE (`1`, `2`, `3`, ...) on each page!
{context_block}
Analyze all {num_pages} page(s) in sequence so character names, titles/ranks, pronouns, and ongoing story stay consistent across every page,
and translate EVERY SINGLE speech bubble and narration/caption box on EVERY page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (RED BADGE) & TRANSLATION RULES:
1. 1-TO-1 RED BADGE MATCHING (`bubble_id`):
   - Inspect EVERY red numbered badge (`1`, `2`, ..., `K`) on each page!
   - If red badge `k` contains Japanese dialogue, monologue, or a caption box: output an entry with `"bubble_id": k`, `"is_dialogue": true`, `"box_2d": null`, the exact `"source_text"` inside ONLY badge `k`, and its natural {target_lang_name} `"translated_text"`.
   - NEVER merge text from two different red badges into one entry! Each red badge `k` MUST have its OWN separate object with `"bubble_id": k`!
   - If a red badge `k` is NOT dialogue (for example, a sound effect / SFX like ゲプ…, a non-speech diagram/icon like 脂肪/免疫, or a blank background object): set `"bubble_id": k`, `"is_dialogue": false`, `"translated_text": ""` so the artwork is left untouched!
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
   - If a page has real spoken dialogue or narration that does NOT have a red badge around it (such as borderless/open-sided text over a white panel background), add an entry with `"bubble_id": null`, `"is_dialogue": true`, transcribe `"source_text"`, and provide `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY that Japanese text — NEVER over faces, hands, or artwork!).
   - NEVER output floating handwritten SFX (e.g. ドーン, トスッ, ポスン, ムチッ, ぐっ, カチャッ, 脱, ぎゅ) as `box_2d` — skip SFX or set `"box_2d": null`.
3. Accurate Speaker Attribution, Names & Visual Gender:
   - Identify the actual character speaking each bubble visually (follow the tail; never copy one character's name onto every bubble).
   - Keep character names, titles/ranks, and faction names consistent across all pages using accurate Japanese phonetics.
   - Match pronouns and politeness particles to each character's visual gender (in Thai, visually female characters must NEVER use male endings like `ครับ`/`ผม`).
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
            badge_note = f"red badges 1..{cv_cnt}" if cv_cnt > 0 else "no red badges"
            parts.append({"text": f"=== PAGE {idx} ({fname} | {badge_note}) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        model_chain = await _discover_manga_batch_models()
        # Client-side failover timeout (not an API limit): generous 80s for 10p so valid slow responses finish cleanly
        per_call_timeout = min(85.0, 40.0 + (4.5 * num_pages))
        max_wall_budget = 180.0
        wall_start = time.monotonic()
        session = get_http_session()
        label_prefix = f"MangaBatch({num_pages}p)"

        for model_id, is_gemma in model_chain:
            remaining = max_wall_budget - (time.monotonic() - wall_start)
            if remaining <= 10.0:
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
                gen_cfg["responseSchema"] = MANGA_BATCH_SOM_SCHEMA

            payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
            max_attempts = 2

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
                            f"({elapsed:.2f}s, {total_b} bubbles: {matched_som} SoM-anchored, {total_anchored - matched_som} box_2d)"
                        )
                        return batch_results, cv_bubbles_per_page, False

                    if total_b > 0 and total_anchored == 0:
                        if fallback_text_only_results is None:
                            fallback_text_only_results = batch_results
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

                # Only retry HTTP 503 if it failed fast (< 6s); slow 503 means pool saturation -> switch immediately!
                if status == 503 and attempt == 0 and elapsed < 6.0 and (max_wall_budget - (time.monotonic() - wall_start)) > 25.0:
                    api_log(
                        f"{label_prefix} [req={req_id}] RESPONSE -> model={model_id} fast HTTP 503 ({elapsed:.2f}s); retrying once in 1.5s"
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
        return fallback_text_only_results, cv_bubbles_per_page, False

    api_log(f"MangaBatch({num_pages}p) [req={req_id}] EXHAUSTED -> all models unavailable/timed out")
    return [([], "") for _ in range(num_pages)], cv_bubbles_per_page, True


async def extract_and_translate_manga_bubbles(
    raw_bytes: bytes,
    filename: str,
    target_lang_name: str,
    series_context_hint: str = "",
    req_id: str = "",
) -> tuple[list[dict], str, list[dict]]:
    req_id = req_id or f"ms-{uuid.uuid4().hex[:6]}"
    b64_img, cv_bubbles = await asyncio.to_thread(
        _detect_page_bubbles_and_build_som_b64_sync, raw_bytes, 1280, 90
    )
    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster:\n{series_context_hint[:1000]}\n"
        if series_context_hint else ""
    )
    badge_note = f"red numbered badges 1..{len(cv_bubbles)}" if cv_bubbles else "no red badges"

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
This page has {badge_note} marking enclosed speech bubbles/caption boxes.
{context_block}
Translate EVERY SINGLE speech bubble and narration/caption box on this page into natural, idiomatic {target_lang_name}.

CRITICAL SET-OF-MARK (RED BADGE) & TRANSLATION RULES:
1. 1-TO-1 RED BADGE MATCHING (`bubble_id`):
   - Inspect EVERY red numbered badge (`1`, `2`, ...) on the page.
   - If red badge `k` contains Japanese dialogue/narration: output `"bubble_id": k`, `"is_dialogue": true`, `"box_2d": null`, exact `"source_text"` inside ONLY badge `k`, and its natural {target_lang_name} `"translated_text"`.
   - NEVER merge text from two different red badges into one entry!
   - If a red badge `k` is NOT dialogue (SFX, diagram, background object): set `"bubble_id": k`, `"is_dialogue": false`, `"translated_text": ""`.
2. UNMARKED / BORDERLESS DIALOGUE (`bubble_id: null`):
   - For dialogue/narration WITHOUT a red badge, output `"bubble_id": null`, `"is_dialogue": true`, `"source_text"`, and `"box_2d": [ymin, xmin, ymax, xmax]` (4 INTEGERS in `0..1000` tightly enclosing ONLY the Japanese text).
3. Accurate Speaker Attribution, Names & Visual Gender:
   - Identify the actual speaker visually; match Thai/English tone to visual gender (never use male Thai `ครับ`/`ผม` for visually female characters).
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
        api_log(f"MangaSingle [req={req_id}] REQUEST -> Tier 1 Gemini ({filename}, {len(cv_bubbles)} CV bubbles)")
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


# ------------------------------------------------------------------------------
# STAGE 3: THAI SYLLABLE WRAPPING + EXACT-MASK FULL-RES RENDERER
# ------------------------------------------------------------------------------
def _tokenize_thai_syllables(word: str) -> list[str]:
    """
    Syllable-aware Thai tokenizer supporting onset clusters (`ห` นำ, `อย`, `กร/ปล/ปร/คร`),
    compound vowels (`เ-อะ`, `เ-อ`, `เ-ีย`, `เ-ือ`, `เ-าะ` -> `เถอะ`, `เยอะ`, `เหงื่อ`, `เพราะ`),
    `อักษรนำ` prefixes (`สภาพ`, `สมา`, `ประ`), `์` karan, trailing coda consonants, and punctuation.
    """
    raw = _THAI_CLUSTER_REGEX.findall(word)
    if len(raw) <= 1:
        return raw or [word]

    # Pass 1: Merge onset clusters (`[เแโใไ]?ห` + `งญนมยรลว`, `อ` + `ย`, `[เแโใไ]?C` + `รลว`)
    merged_onsets: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        tok = raw[i]
        nxt = raw[i + 1] if (i + 1) < n else ""

        if nxt:
            # `ห` นำ (including `เหงื่อ`, `เหรอ`, `หรือ`, `หลาย`) or `อ` นำ `ย` (`อยู่`)
            is_ho_nam = (
                ((tok == "ห") or (len(tok) == 2 and tok[0] in _THAI_LEADING_VOWELS and tok[1] == "ห"))
                and nxt[0] in _THAI_HO_NAM_FOLLOW
            )
            is_o_nam = (tok == "อ" and nxt[0] == "ย")
            if is_ho_nam or is_o_nam:
                merged_onsets.append(tok + nxt)
                i += 2
                continue

            # True consonant clusters (`เปล` / `เปล่า`, `จริง`, `ปรับ`, `เพราะ`)
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

    # Pass 2: Merge Thai compound vowels (`เ-อะ`, `เ-อ`, `เ-ีย`, `เ-ือ`, `เ-าะ` -> `เถอะ`, `เยอะ`, `เหงื่อ`, `เพราะ`, `เหรอ`)
    merged_vowels: list[str] = []
    i = 0
    m = len(merged_onsets)
    while i < m:
        tok = merged_onsets[i]
        nxt = merged_onsets[i + 1] if (i + 1) < m else ""
        if nxt and tok.startswith("เ"):
            # `เ_ + อะ`, `เ_ + อ`, `เ_ + าะ`, `เ_ี + ย`, `เ_ื + อ`
            if nxt in ("อะ", "อ", "าะ") or (("\u0E35" in tok) and nxt.startswith("ย")) or (("\u0E37" in tok) and nxt.startswith("อ")):
                merged_vowels.append(tok + nxt)
                i += 2
                continue
        merged_vowels.append(tok)
        i += 1

    # Pass 3: Merge `อักษรนำ` bare prefix (`ส+ภาพ`, `ส+มา`, `ป+ระ`) & attach coda consonants / karan / punctuation
    syllables: list[str] = []
    i = 0
    k = len(merged_vowels)
    while i < k:
        tok = merged_vowels[i]
        nxt = merged_vowels[i + 1] if (i + 1) < k else ""

        # Leading bare prefix before an explicit-vowel syllable (e.g. `ส` + `ภา` -> `สภา`, `ส` + `มา` -> `สมา`)
        if (
            nxt
            and len(tok) == 1
            and tok in _THAI_PREFIX_LEAD
            and any(c in _THAI_VOWEL_SIGNALS for c in nxt)
            and (not syllables or len(syllables[-1]) >= 3)
        ):
            syllables.append(tok + nxt)
            i += 2
            continue

        if not syllables:
            syllables.append(tok)
            i += 1
            continue

        if all(c in _THAI_PUNCT_ATTACH for c in tok) or ("\u0E4C" in tok):
            syllables[-1] += tok
            i += 1
            continue

        # Single bare Thai consonant is a coda consonant when previous syllable has a vowel/tone nucleus
        is_bare_single = (len(tok) == 1 and "\u0E01" <= tok <= "\u0E2E")
        prev = syllables[-1]
        prev_has_nucleus = any(c in _THAI_VOWEL_SIGNALS or ("\u0E48" <= c <= "\u0E4B") for c in prev)
        if is_bare_single and prev_has_nucleus and len(prev) <= 6:
            syllables[-1] += tok
            i += 1
            continue

        syllables.append(tok)
        i += 1

    return syllables


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


def _fast_dilate_mask_half_res(mask_l: Image.Image, kernel_half: int = 3) -> Image.Image:
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


def _sample_dark_bg_rgb_and_lum(img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int, int], int, float]:
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


def _sample_light_bg_rgb(img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int]:
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


def _clean_exact_cv_bubble_and_measure_box(
    img: Image.Image,
    base_gray: Image.Image,
    cv_obj: dict,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    """
    Cleans Japanese text inside a CV-anchored speech bubble using its exact pixel mask:
      - Erodes mask 2px inside the black outline so speech-bubble borders stay razor-sharp.
      - Protects colored chibi-art pixels (`sat >= 26`) inside the bubble while wiping 100%
        of achromatic Japanese text strokes (`p < 246`).
      - Measures the true inscribed text rectangle inside the oval via C-level `getbbox()` rows.
    """
    width, height = img.size
    ymin, xmin, ymax, xmax = cv_obj["bbox_norm"]
    x1 = max(0, int((xmin / 1000.0) * width))
    y1 = max(0, int((ymin / 1000.0) * height))
    x2 = min(width, int((xmax / 1000.0) * width))
    y2 = min(height, int((ymax / 1000.0) * height))
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 14:
        return None

    is_dark = bool(cv_obj.get("is_dark"))
    mask_full = cv_obj["mask_small"].resize((bw, bh), Image.Resampling.BILINEAR).point(
        lambda p: 255 if p >= 128 else 0
    )
    # Erode 2px inward so the outer black speech-bubble stroke is 100% preserved
    safe_core = mask_full.filter(ImageFilter.MinFilter(5))
    mask_full.close()
    if not safe_core.getbbox():
        safe_core.close()
        safe_core = cv_obj["mask_small"].resize((bw, bh), Image.Resampling.BILINEAR).point(
            lambda p: 255 if p >= 128 else 0
        )

    sub_gray = base_gray.crop((x1, y1, x2, y2))
    sub_rgb = img.crop((x1, y1, x2, y2))
    sub_sat = sub_rgb.convert("HSV").getchannel("S")
    sub_rgb.close()

    try:
        # Preserve colored chibi faces/icons drawn inside speech bubbles (`sat >= 26`)
        chroma_px = sub_sat.point(lambda p: 255 if p >= 26 else 0)
        chroma_shield = _fast_dilate_mask_half_res(chroma_px, kernel_half=3)
        chroma_px.close()
        non_chroma = ImageChops.invert(chroma_shield)
        chroma_shield.close()

        if is_dark:
            dark_rgb, bg_lum, _ = _sample_dark_bg_rgb_and_lum(img, base_gray, x1, y1, x2, y2)
            diff_ink = sub_gray.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
            ink_dil = _fast_dilate_mask_half_res(diff_ink, kernel_half=3)
            diff_ink.close()
            erase_mask = ImageChops.darker(ImageChops.darker(ink_dil, safe_core), non_chroma)
            ink_dil.close()

            fill_patch = Image.new("RGB", (bw, bh), dark_rgb)
            img.paste(fill_patch, (x1, y1), mask=erase_mask)
            fill_patch.close()
            erase_mask.close()
            fg_rgb = (250, 250, 250)
            stroke_rgb = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))
        else:
            ink_mask = sub_gray.point(lambda p: 255 if p < 246 else 0)
            ink_dil = _fast_dilate_mask_half_res(ink_mask, kernel_half=3)
            ink_mask.close()
            erase_mask = ImageChops.darker(ImageChops.darker(ink_dil, safe_core), non_chroma)
            ink_dil.close()

            white_patch = Image.new("RGB", (bw, bh), (255, 255, 255))
            img.paste(white_patch, (x1, y1), mask=erase_mask)
            white_patch.close()
            erase_mask.close()
            fg_rgb = (15, 15, 15)
            stroke_rgb = (255, 255, 255)

        non_chroma.close()

        # Measure inscribed text bounds strictly inside `safe_core`
        core_bb = safe_core.getbbox()
        if not core_bb:
            return x1 + 4, y1 + 4, x2 - 4, y2 - 4, fg_rgb, stroke_rgb, (x1 + 3, y1 + 3, x2 - 3, y2 - 3)

        lb_x1, lb_y1, lb_x2, lb_y2 = core_bb
        rel_y1 = lb_y1 + max(4, int((lb_y2 - lb_y1) * 0.08))
        rel_y2 = lb_y2 - max(4, int((lb_y2 - lb_y1) * 0.08))
        if rel_y2 - rel_y1 < 18:
            rel_y1, rel_y2 = lb_y1 + 2, lb_y2 - 2

        mid_y_start = rel_y1 + int((rel_y2 - rel_y1) * 0.16)
        mid_y_end = rel_y1 + int((rel_y2 - rel_y1) * 0.84)
        step_y = max(2, (mid_y_end - mid_y_start) // 7 or 1)
        row_lefts = []
        row_rights = []
        for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), step_y):
            if not (0 <= sy < bh):
                continue
            row_strip = safe_core.crop((lb_x1, sy, lb_x2, sy + 1))
            rb = row_strip.getbbox()
            row_strip.close()
            if rb and (rb[2] - rb[0]) >= 14:
                row_lefts.append(lb_x1 + rb[0])
                row_rights.append(lb_x1 + rb[2])

        hard_b = (x1 + lb_x1 + 3, y1 + lb_y1 + 3, x1 + lb_x2 - 3, y1 + lb_y2 - 3)
        if row_lefts and row_rights:
            row_lefts.sort()
            row_rights.sort()
            inner_l = row_lefts[int(len(row_lefts) * 0.65)] + 4
            inner_r = row_rights[int(len(row_rights) * 0.35)] - 4
            if inner_r - inner_l >= 18:
                return x1 + inner_l, y1 + rel_y1, x1 + inner_r, y1 + rel_y2, fg_rgb, stroke_rgb, hard_b

        return x1 + lb_x1 + 4, y1 + rel_y1, x1 + lb_x2 - 4, y1 + rel_y2, fg_rgb, stroke_rgb, hard_b
    finally:
        safe_core.close()
        sub_gray.close()
        sub_sat.close()


def _clean_unmarked_open_bubble_and_measure_box(
    img: Image.Image,
    base_gray: Image.Image,
    raw_x1: int,
    raw_y1: int,
    raw_x2: int,
    raw_y2: int,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int, int]] | None:
    """
    Safely cleans borderless/open-sided dialogue (`bubble_id == None`, e.g. text on a white
    panel background like Page 3 middle-left or Page 8 left-top) while shielding colored skin,
    hands, hair, and panel borders from any white artifacts.
    """
    width, height = img.size
    bw, bh = raw_x2 - raw_x1, raw_y2 - raw_y1
    if bw < 18 or bh < 16:
        return None

    sub_g = base_gray.crop((raw_x1, raw_y1, raw_x2, raw_y2))
    median_lum = ImageStat.Stat(sub_g).median[0]
    box_hist = sub_g.histogram()
    sub_g.close()
    total_px = max(1, sum(box_hist))
    mean_sat, chroma_ratio = _measure_crop_chroma_stats(img, raw_x1, raw_y1, raw_x2, raw_y2)

    # Reject chromatic/colored artwork (faces, hands, wallpaper)
    if mean_sat >= 12.0 or chroma_ratio >= 0.14:
        return None

    # Dark unmarked bubble/caption
    if median_lum < 125:
        dark_rgb, bg_lum, bg_std = _sample_dark_bg_rgb_and_lum(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        bright_ratio = sum(box_hist[220:]) / float(total_px)
        if bg_std > 26.0 or not (0.02 <= bright_ratio <= 0.38):
            return None

        pad = 6
        wx1, wy1 = max(0, raw_x1 - pad), max(0, raw_y1 - pad)
        wx2, wy2 = min(width, raw_x2 + pad), min(height, raw_y2 + pad)
        ww, wh = wx2 - wx1, wy2 - wy1
        sg = base_gray.crop((wx1, wy1, wx2, wy2))
        diff = sg.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
        sg.close()
        erase_d = _fast_dilate_mask_half_res(diff, kernel_half=3)
        diff.close()
        patch = Image.new("RGB", (ww, wh), dark_rgb)
        img.paste(patch, (wx1, wy1), mask=erase_d)
        patch.close()
        erase_d.close()
        stroke_c = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))
        return raw_x1, raw_y1, raw_x2, raw_y2, (250, 250, 250), stroke_c, (raw_x1, raw_y1, raw_x2, raw_y2)

    # Light open-sided bubble / white panel background
    white_ratio = sum(box_hist[228:]) / float(total_px)
    ink_ratio = sum(box_hist[:145]) / float(total_px)
    if white_ratio < 0.42 or ink_ratio < 0.012:
        return None

    pad_x, pad_y = 8, 12
    wx1, wy1 = max(0, raw_x1 - pad_x), max(0, raw_y1 - pad_y)
    wx2, wy2 = min(width, raw_x2 + pad_x), min(height, raw_y2 + pad_y)
    ww, wh = wx2 - wx1, wy2 - wy1

    sg = base_gray.crop((wx1, wy1, wx2, wy2))
    srgb = img.crop((wx1, wy1, wx2, wy2))
    ssat = srgb.convert("HSV").getchannel("S")
    srgb.close()

    # Chromatic-art shield protects hands/fingers/hair within ~5px of colored pixels
    c_px = ssat.point(lambda p: 255 if p >= 20 else 0)
    ssat.close()
    c_shield = _fast_dilate_mask_half_res(c_px, kernel_half=5)
    c_px.close()
    non_chroma = ImageChops.invert(c_shield)
    c_shield.close()

    # Only erase dark ink surrounded by white paper (protects solid black panel frames!)
    w_seed = sg.point(lambda p: 255 if p >= 235 else 0)
    w_zone = _fast_dilate_mask_half_res(w_seed, kernel_half=5)
    w_seed.close()

    ink_m = sg.point(lambda p: 255 if p < 230 else 0)
    sg.close()
    ink_dil = _fast_dilate_mask_half_res(ink_m, kernel_half=3)
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

    bg_rgb = _sample_light_bg_rgb(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
    patch = Image.new("RGB", (ww, wh), bg_rgb)
    img.paste(patch, (wx1, wy1), mask=erase_c)
    patch.close()
    erase_c.close()
    return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), bg_rgb, (raw_x1, raw_y1, raw_x2, raw_y2)


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
    cv_bubbles: list[dict] | None = None,
) -> bytes | None:
    """
    Renders translated dialogue at FULL ORIGINAL RESOLUTION:
      1) Matched Set-of-Mark CV bubbles (`bubble_id`) are cleaned and typeset using their
         exact pre-detected pixel masks (zero drift, zero overlap, zero leftover Japanese!).
      2) Unmarked open/borderless bubbles (`bubble_id == None` + `box_2d`) are cleaned safely
         with chromatic-art shielding after verifying they don't overlap an already-rendered CV bubble.
    """
    if not raw_bytes or not bubbles:
        return None

    cv_map: dict[int, dict] = {
        int(c["bubble_id"]): c for c in (cv_bubbles or []) if c.get("bubble_id") is not None
    }

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    base_gray = img.convert("L")
    width, height = img.size

    try:
        prepared_renders: list[list] = []
        used_cv_ids: set[int] = set()
        rendered_rects: list[tuple[int, int, int, int]] = []

        # Pass 1: Render all CV-anchored Set-of-Mark bubbles (`bubble_id` match)
        for b in bubbles:
            bid = b.get("bubble_id")
            tr_txt = _sanitize_text_for_manga_font(b.get("translated_text") or "")
            if not tr_txt or bid is None or bid in used_cv_ids or bid not in cv_map:
                continue

            cv_obj = cv_map[bid]
            cleaned = _clean_exact_cv_bubble_and_measure_box(img, base_gray, cv_obj)
            if cleaned is None:
                continue
            used_cv_ids.add(bid)
            lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned
            prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
            rendered_rects.append(hard_bounds)

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

            # Snap to an unused CV bubble if this `box_2d` overlaps it by >= 40%!
            snapped_cv_id = None
            best_ov = 0.0
            for cid, c_obj in cv_map.items():
                if cid in used_cv_ids:
                    continue
                cy1_n, cx1_n, cy2_n, cx2_n = c_obj["bbox_norm"]
                cx1_p = int((cx1_n / 1000.0) * width)
                cy1_p = int((cy1_n / 1000.0) * height)
                cx2_p = int((cx2_n / 1000.0) * width)
                cy2_p = int((cy2_n / 1000.0) * height)
                ov_x = max(0, min(rx2, cx2_p) - max(rx1, cx1_p))
                ov_y = max(0, min(ry2, cy2_p) - max(ry1, cy1_p))
                ov_ratio = (ov_x * ov_y) / float(min(area_new, max(1, (cx2_p - cx1_p) * (cy2_p - cy1_p))))
                if ov_ratio >= 0.40 and ov_ratio > best_ov:
                    best_ov = ov_ratio
                    snapped_cv_id = cid

            if snapped_cv_id is not None:
                cleaned = _clean_exact_cv_bubble_and_measure_box(img, base_gray, cv_map[snapped_cv_id])
                if cleaned is not None:
                    used_cv_ids.add(snapped_cv_id)
                    lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned
                    prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
                    rendered_rects.append(hard_bounds)
                    continue

            # Reject duplicate overlap with an already-rendered bubble
            is_dup = False
            for ex1, ey1, ex2, ey2 in rendered_rects:
                ov_x = max(0, min(rx2, ex2) - max(rx1, ex1))
                ov_y = max(0, min(ry2, ey2) - max(ry1, ey1))
                if (ov_x * ov_y) / float(min(area_new, max(1, (ex2 - ex1) * (ey2 - ey1)))) >= 0.45:
                    is_dup = True
                    break
            if is_dup:
                continue

            cleaned_open = _clean_unmarked_open_bubble_and_measure_box(img, base_gray, rx1, ry1, rx2, ry2)
            if cleaned_open is None:
                continue
            lx1, ly1, lx2, ly2, fg_text, stroke_col, hard_bounds = cleaned_open
            prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, tr_txt, hard_bounds])
            rendered_rects.append(hard_bounds)

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
    using CV Set-of-Mark (SoM) bubble anchoring, strict `MangaBatch` model ordering:
      `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite` -> (newer >3.5 flash-lite) -> `gemma-4-31b-it` -> `gemma-4-26b-a4b-it`
    correlated Render request/waiting/response logs, syllable-aware Thai wrapping,
    and fast full-original-resolution rendering.
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

    cv_bubbles_per_page: list[list[dict]] = []

    async with channel.typing():
        try:
            batch_results, cv_bubbles_per_page, is_unavailable = await extract_and_translate_manga_pages_batch(
                pages_input, target_lang_name, series_context_hint=series_context_hint, req_id=req_id
            )

            if is_unavailable:
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
            api_log(f"MangaRender [req={req_id}] START -> rendering {num_pages} page(s)...")

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
                    for c_obj in page_cv:
                        m = c_obj.get("mask_small")
                        if m is not None:
                            try:
                                m.close()
                            except Exception:
                                pass
                    page_cv.clear()

                await asyncio.sleep(0.10)

            render_elapsed = time.monotonic() - render_t0
            api_log(
                f"MangaRender [req={req_id}] DONE -> rendered {len(discord_files)}/{num_pages} page(s) "
                f"in {render_elapsed:.2f}s"
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
        finally:
            for page_cv in cv_bubbles_per_page:
                for c_obj in page_cv:
                    m = c_obj.get("mask_small")
                    if m is not None:
                        try:
                            m.close()
                        except Exception:
                            pass
                page_cv.clear()
            cv_bubbles_per_page.clear()
            pages_input.clear()
            release_unused_memory()