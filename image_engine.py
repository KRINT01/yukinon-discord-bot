import os
import io
import gc
import re
import json
import time
import ctypes
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
    GEMINI_HEAVY_MODEL,
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
    call_gemini_hydrogen_generate,
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
    Normalizes Unicode punctuation (`…` -> `...`, `〜` -> `~`, `—` -> `-`, smart quotes -> `'`)
    and removes stray CJK/emoji symbols so `Garuda-Bold.ttf` renders every character cleanly
    with zero `[]` missing-glyph tofu boxes.
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


def _prepare_manga_image_for_vision(raw_bytes: bytes, max_dim: int = 1280, quality: int = 92) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    img.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_qa_thumbnail_sync(rendered_jpg_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(rendered_jpg_bytes)).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    img.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _audit_rendered_manga_page_with_decision_1(
    rendered_jpg_bytes: bytes, page_num: int
) -> bool:
    """
    Uses Google's discriminative `gemini-decision-1` API (200 RPM, 10K RPD free, ~250ms)
    as a fast post-render visual QA auditor to detect:
      1) Uncleaned vertical Japanese dialogue left inside speech bubbles
      2) Translated text placed outside bubbles over character faces/hair/artwork or overlapping boxes
    Returns True if the page needs single-page spatial re-grounding & re-render, False if clean.
    """
    if not rendered_jpg_bytes or not GEMINI_API_KEY:
        return False
    if time.time() < state.gemini_decision_cooldown_until:
        return False

    try:
        b64_thumb = await asyncio.to_thread(_encode_qa_thumbnail_sync, rendered_jpg_bytes)
    except Exception:
        return False

    shared_input = [
        {
            "type": "text",
            "text": (
                f"Visual Quality Assurance Audit for Translated Manga Page #{page_num}.\n"
                "Inspect the speech bubbles and rendered text on this page:\n"
            ),
        },
        {
            "type": "image",
            "data": b64_thumb,
            "mime_type": "image/jpeg",
            "resolution": "low",
        },
    ]

    questions = {
        "uncleaned_japanese": {
            "type": "noul",
            "instructions": (
                "Answer YES only if there are prominent blocks of vertical Japanese dialogue still remaining "
                "inside white or dark speech bubbles (ignore small handwritten SFX sound effects outside bubbles)."
            ),
        },
        "misplaced_or_overlapping_text": {
            "type": "noul",
            "instructions": (
                "Answer YES if translated Thai or English text is drawn outside speech bubbles over character faces, "
                "hair, or background artwork, or if two separate translations are crammed on top of each other in one bubble."
            ),
        },
        "qa_verdict": {
            "type": "choice",
            "instructions": "Select whether this translated manga page passes visual QA or needs re-rendering.",
            "criteria": {
                "pass": "Speech bubbles are cleanly erased and translated text sits neatly inside the bubbles.",
                "rerender": "Major speech bubbles still show untranslated Japanese dialogue or translated text is placed in the wrong panel/outside bubbles.",
            },
        },
    }

    answers = await call_gemini_decision_orchestrator(
        shared_input, questions, timeout_sec=4.0, purpose=f"MangaQA-p{page_num}"
    )
    if not answers:
        return False

    uncleaned_prob = float((answers.get("uncleaned_japanese") or {}).get("noul") or 0.0)
    misplaced_prob = float((answers.get("misplaced_or_overlapping_text") or {}).get("noul") or 0.0)
    verdict_choice = (answers.get("qa_verdict") or {}).get("choice") or "pass"
    verdict_conf = float((answers.get("qa_verdict") or {}).get("confidence") or 0.0)

    needs_rerender = bool(
        (verdict_choice == "rerender" and verdict_conf >= 0.65)
        or uncleaned_prob >= 0.72
        or misplaced_prob >= 0.72
    )
    if needs_rerender:
        api_log(
            f"Manga QA Audit (Page {page_num}) -> FLAGGED for re-grounding "
            f"(verdict={verdict_choice}:{verdict_conf:.2f}, uncleaned={uncleaned_prob:.2f}, misplaced={misplaced_prob:.2f})"
        )
    return needs_rerender


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

        box = item.get("box_2d") or item.get("bbox") or item.get("box")
        norm_box = None
        if not is_sfx:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    norm_box = [float(v) for v in box]
                except Exception:
                    norm_box = None
            elif isinstance(box, str):
                nums = re.findall(r"-?\d+(?:\.\d+)?", box)
                if len(nums) == 4:
                    norm_box = [float(v) for v in nums]

        # Cross-check `panel_position` hint against `box_2d` vertical bounds to reject drifted boxes
        panel_pos = str(item.get("panel_position") or "").strip().lower()
        if norm_box and panel_pos:
            ymin, _, ymax, _ = norm_box
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            y_mid = (ymin + ymax) / 2.0
            if panel_pos.startswith("top") and y_mid > 520:
                norm_box = None
            elif panel_pos.startswith("middle") and (y_mid < 200 or y_mid > 760):
                norm_box = None
            elif panel_pos.startswith("bottom") and y_mid < 480:
                norm_box = None

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
    """
    Recovers completed page objects if an 8-12 page batch JSON hit maxOutputTokens
    mid-way through the final page so valid pages are never discarded.
    """
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
            if any(b for b, _ in results):
                return results

        if expected_pages == 1 and isinstance(parsed, dict) and "bubbles" in parsed:
            bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
            summary = str(parsed.get("summary") or "").strip()
            return [(bubbles, summary)]

    if expected_pages == 1:
        b, s = _parse_manga_bubbles_json(raw_text)
        return [(b, s)]

    return [([], "") for _ in range(expected_pages)]


async def extract_and_translate_manga_pages_batch(
    pages_input: list[tuple[bytes, str]],
    target_lang_name: str,
    series_context_hint: str = "",
) -> list[tuple[list[dict], str]]:
    """
    Processes 1 to 12 manga pages (including multi-part X/Twitter threads) in a SINGLE unified
    multimodal batch call:
      - Tier 1: Spatial Gemini Flash (`gemini-3-flash-preview` -> `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite`)
      - Tier 2: `gemini-4-hydrogen` (`thinkingLevel: "LOW"`, `maxOutputTokens: 65536`, no sampling params)
      - Tier 3: Concurrent `Semaphore(2)` per-page fallback
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return []

    if num_pages == 1:
        b, s = await extract_and_translate_manga_bubbles(
            pages_input[0][0], pages_input[0][1], target_lang_name, series_context_hint=series_context_hint
        )
        return [(b, s)]

    if num_pages >= 9:
        batch_dim, batch_q = 1080, 85
    elif num_pages >= 5:
        batch_dim, batch_q = 1152, 86
    else:
        batch_dim, batch_q = 1280, 90

    b64_pages = []
    for raw_b, fname in pages_input:
        b64_str = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_b, batch_dim, batch_q)
        b64_pages.append((b64_str, fname))
        await asyncio.sleep(0.05)

    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster (use these exact names & terms):\n{series_context_hint[:1400]}\n"
        if series_context_hint else ""
    )

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
{context_block}
Analyze all pages together in sequence so series title, character names, titles/ranks, pronouns, and ongoing story stay consistent across every page,
and translate EVERY SINGLE speech bubble, narration box, and SFX on EVERY page into natural, idiomatic {target_lang_name}.

CRITICAL COMPLETENESS, NAME ACCURACY & BOUNDING-BOX RULES:
1. Complete Coverage (ZERO Skipped Bubbles):
   - Scan every panel of every page systematically from top-right to bottom-left and extract EVERY speech bubble (including short replies, secondary bubbles in crowded panels, and background bubbles)! Do NOT skip any speech bubble!
   - Within each page, order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. Consistent Character Names & Localization:
   - Keep character names, titles/ranks, and faction names consistent across all pages, cross-checking with the post context above and accurate Japanese phonetics.
   - Adapt idioms, pronouns, tone, and emotion so the dialogue sounds natural in {target_lang_name} while fitting inside a speech bubble.
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
   - When translating to English, preserve Japanese honorifics attached directly with a hyphen and NO space (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).
3. Spatial Anchor & Bounding Boxes (`panel_position` + `box_2d`) — STRICT ALIGNMENT RULES:
   - First identify `panel_position` on that specific page (`"top-right"`, `"top-left"`, `"middle-right"`, `"middle-left"`, `"bottom-right"`, `"bottom-left"`), then output `box_2d` (`[ymin, xmin, ymax, xmax]`, normalized 0..1000 relative to THAT single page).
   - ONLY provide `box_2d` for dialogue/narration inside an actual speech bubble or caption box!
   - `box_2d` MUST tightly enclose the ENTIRE Japanese text block INSIDE that specific bubble (including all vertical columns in a rectangular caption box) — NEVER shift `box_2d` into a different panel, empty background, panel gutters, or character heads/bodies!
   - For floating SFX / sound effects OR tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble, ALWAYS set `"box_2d": null` so the artwork/character is NEVER overwritten!
   - NEVER merge two separate speech bubbles into a single `box_2d` and NEVER assign the same `box_2d` to two different bubbles!
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
          "box_2d": [ymin, xmin, ymax, xmax]
        }}
      ]
    }}
  ]
}}
""".strip()

    try:
        if GEMINI_API_KEY:
            parts = [{"text": manga_batch_instruction}]
            for idx, (b64_img, fname) in enumerate(b64_pages, 1):
                parts.append({"text": f"=== PAGE {idx} ({fname}) ==="})
                parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

            timeout_sec = min(145, 55 + (10 * num_pages))

            # Tier 1: Spatial Gemini Flash (`gemini-3-flash-preview` -> `gemini-3.5-flash-lite` -> `gemini-3.1-flash-lite`)
            flash_payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 32768,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                },
            }
            try:
                data = await call_gemini_generate_content_self_healing(
                    flash_payload,
                    timeout_sec=timeout_sec,
                    purpose=f"MangaBatch({num_pages}p)",
                    max_models=3,
                    max_wall_sec=float(timeout_sec + 25),
                    preferred_models=["gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
                )
                if data:
                    resp_parts = (
                        ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                    )
                    raw_json_str = "\n".join(
                        p.get("text", "") for p in resp_parts
                        if p.get("text") and not p.get("thought", False)
                    ).strip()
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
                print(f"Gemini Flash Manga Batch warning ({type(e).__name__}): {e}", flush=True)

            # Tier 2: Gemini 4 Hydrogen Heavy Specialist fallback (`thinkingLevel: "LOW"`, `maxOutputTokens: 65536`)
            hydrogen_payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "maxOutputTokens": 65536,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                },
            }
            try:
                h_data = await call_gemini_hydrogen_generate(
                    hydrogen_payload, timeout_sec=timeout_sec, purpose=f"MangaBatchHydrogen({num_pages}p)"
                )
                if h_data:
                    resp_parts = (
                        ((h_data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                    )
                    raw_json_str = "\n".join(
                        p.get("text", "") for p in resp_parts
                        if p.get("text") and not p.get("thought", False)
                    ).strip()
                    batch_results = _parse_manga_pages_batch_json(raw_json_str, num_pages)
                    if any(bubbles for bubbles, _ in batch_results):
                        total_b = sum(len(b) for b, _ in batch_results)
                        print(
                            f"[Manga Vision Batch] Completed {num_pages} page(s) in 1 request via Tier 2 Hydrogen "
                            f"({GEMINI_HEAVY_MODEL}) - {total_b} total bubbles",
                            flush=True,
                        )
                        return batch_results
            except Exception as e:
                print(f"Gemini Hydrogen Manga Batch warning ({type(e).__name__}): {e}", flush=True)
    finally:
        b64_pages.clear()

    # Tier 3: Concurrent per-page fallback (2 pages at a time)
    sem = asyncio.Semaphore(2)

    async def _bounded_single(rb: bytes, fn: str) -> tuple[list[dict], str]:
        async with sem:
            return await extract_and_translate_manga_bubbles(
                rb, fn, target_lang_name, series_context_hint=series_context_hint
            )

    fallback_results = await asyncio.gather(
        *[_bounded_single(raw_b, fname) for raw_b, fname in pages_input]
    )
    return list(fallback_results)


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
) -> tuple[list[dict], str]:
    b64_img = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_bytes)
    mime_type = "image/jpeg"
    context_block = (
        f"\nSeries / Post Context & Character Roster:\n{series_context_hint[:1000]}\n"
        if series_context_hint else ""
    )

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
{context_block}
Analyze this single manga page visually (character expressions, action, speaker relationships, and panel layout)
and translate EVERY SINGLE speech bubble (including small bubbles and short exclamations) into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. Complete Coverage & Reading Order:
   - Scan every panel systematically and translate EVERY speech bubble on the page without skipping any bubble!
   - Order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. Consistent Character Names & Localization:
   - Keep character names, titles/ranks, and faction names consistent with the post context above and accurate Japanese phonetics.
   - Use standard ASCII punctuation (`...`, `~`, `-`, single quotes `'`).
   - When translating to English, preserve Japanese honorifics attached directly with a hyphen and NO space (`-san`, `-kun`, `-chan`, `-sama`, `-sensei`, `-senpai`).
3. Spatial Anchor & Bounding Boxes (`panel_position` + `box_2d`) — STRICT ALIGNMENT RULES:
   - First identify `panel_position` (`"top-right"`, `"top-left"`, `"middle-right"`, `"middle-left"`, `"bottom-right"`, `"bottom-left"`), then output `box_2d` (`[ymin, xmin, ymax, xmax]`, normalized 0..1000).
   - ONLY provide `box_2d` for dialogue/narration inside an actual speech bubble or caption box!
   - `box_2d` MUST tightly enclose the ENTIRE Japanese text block INSIDE that specific bubble (including all vertical columns in a rectangular caption box) — NEVER shift into a different panel, empty background, or character heads/bodies!
   - For floating SFX / sound effects OR tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble, ALWAYS set `"box_2d": null` so the artwork/character is NEVER overwritten!
   - NEVER merge two separate speech bubbles into a single `box_2d` and NEVER assign the same `box_2d` to two different bubbles!
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
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ]
}}
""".strip()

    fallback_bubbles_no_boxes = []
    fallback_summary = ""

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
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(
                gemini_payload,
                timeout_sec=30,
                max_models=3,
                preferred_models=["gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
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
                    print(
                        f"[Manga Vision] Completed via Tier 1 Gemini ({state.active_gemini_model}) "
                        f"- {len(bubbles)} bubbles"
                    )
                    return bubbles, summary
        except Exception as e:
            print(f"Gemini Manga Vision warning ({type(e).__name__}): {e}")

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
                    print(f"[Manga Vision] Completed via Tier 3 AI Studio Gemma 4 - {len(bubbles)} bubbles", flush=True)
                    return bubbles, summary
                if not fallback_bubbles_no_boxes:
                    fallback_bubbles_no_boxes = bubbles
                    fallback_summary = summary
    except Exception as e:
        print(f"AI Studio Gemma 4 Manga Vision warning: {e}", flush=True)

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


def _split_thai_word_cleanly(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """
    Splits an unspaced Thai phrase across lines without leaving orphaned leading vowels
    (`เ`, `แ`, `โ`, `ใ`, `ไ`) at the end of a line or splitting combining tone/vowel marks.
    """
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


def _detect_bubble_polarity(
    base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int
) -> bool:
    """
    Fast C-level median luminance check using `ImageStat.Stat`.
    Returns True if the bubble/panel region at (x1, y1, x2, y2) is Dark/Charcoal (< 125).
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw < 10 or bh < 10:
        return False
    sub = base_gray.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2))
    median_lum = ImageStat.Stat(sub).median[0]
    sub.close()
    return median_lum < 125


def _compute_peer_clip_bounds(
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    other_peer_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """
    Computes safe midpoint boundaries between `(x1, y1, x2, y2)` and neighboring bubbles
    so an expanded bubble lobe never crosses into a neighboring peer bubble.
    """
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
    Fast, memory-safe 0.5x-scale enclosed pure-white speech bubble finder:
      - Uses 2 morphological passes (`base_bin` and `MinFilter(3)`) and breaks immediately on pass 0
        when a single-lobe bubble is found (`cw <= s_bw * 2.15`) to minimize CPU & C-heap allocations.
      - Explicitly `.close()`s every intermediate `Image` so 512MB Render containers never fragment RAM.
    """
    ww, wh = sub_gray.size
    if ww < 16 or wh < 16:
        return None

    sw = max(8, ww // 2)
    sh = max(8, wh // 2)
    small_gray = sub_gray.resize((sw, sh), Image.Resampling.BILINEAR)

    s_rx1 = max(1, rx1_rel // 2)
    s_ry1 = max(1, ry1_rel // 2)
    s_rx2 = min(sw - 2, max(s_rx1 + 4, rx2_rel // 2))
    s_ry2 = min(sh - 2, max(s_ry1 + 4, ry2_rel // 2))
    s_bw = max(4, s_rx2 - s_rx1)
    s_bh = max(4, s_ry2 - s_ry1)

    seed_candidates = []
    for fy in (0.50, 0.25, 0.75):
        sy = max(1, min(sh - 2, int(s_ry1 + s_bh * fy)))
        for fx in (0.50, 0.25, 0.75):
            sx = max(1, min(sw - 2, int(s_rx1 + s_bw * fx)))
            seed_candidates.append((sx, sy))
        for extra_dx in (-12, 12, -24, 24, -36, 36):
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
            if cw < 9 or ch < 9:
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

            if comp_area < 40 or (comp_area / filled_area) < 0.24:
                filled.close()
                continue

            if has_kanji_kana and holes_area < 4:
                filled.close()
                continue

            ov_x = max(0, min(cx2, s_rx2) - max(cx1, s_rx1))
            ov_y = max(0, min(cy2, s_ry2) - max(cy1, s_ry1))
            overlap_area = ov_x * ov_y
            if overlap_area < 16:
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
    """
    Zero-allocation C-level dark background sampler using `ImageStat` & `histogram()`.
    """
    width, height = img.size
    sx1 = max(0, x1 - 4)
    sy1 = max(0, y1 - 4)
    sx2 = min(width, x2 + 4)
    sy2 = min(height, y2 + 4)
    crop_rgb = img.crop((sx1, sy1, sx2, sy2))
    crop_gray = base_gray.crop((sx1, sy1, sx2, sy2))

    # Build mask of dark/midtone background pixels (< 145) so white text strokes are ignored
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
    """Zero-allocation C-level light background sampler using `ImageStat`."""
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
    bg_lum: int,
) -> tuple[int, int, int, int] | None:
    """
    Detects sharp-edged dark rectangular narration/caption boxes (e.g. `一ヶ月後 玄龍門`)
    even when `box_2d` only covered one vertical column of the box, returning `(bx1, by1, bx2, by2)`
    in `sub_gray` coordinates so ALL columns inside the caption box are wiped clean.
    """
    ww, wh = sub_gray.size
    box_w = max(8, rx2_rel - rx1_rel)
    box_h = max(8, ry2_rel - ry1_rel)
    if box_w < 14 or box_h < 20:
        return None

    # Dark interior pixels matching `bg_lum` (±24) OR bright white text strokes (>= 195) inside the box
    cap_bin = sub_gray.point(lambda p: 255 if (abs(int(p) - bg_lum) <= 24 or p >= 195) else 0)
    cx = (rx1_rel + rx2_rel) // 2
    cy = (ry1_rel + ry2_rel) // 2
    if not (0 <= cx < ww and 0 <= cy < wh) or cap_bin.getpixel((cx, cy)) != 255:
        cap_bin.close()
        return None

    ImageDraw.floodfill(cap_bin, (cx, cy), 128)
    comp = cap_bin.point(lambda p: 255 if p == 128 else 0)
    cap_bin.close()

    c_bbox = comp.getbbox()
    if not c_bbox:
        comp.close()
        return None
    bx1, by1, bx2, by2 = c_bbox
    bw, bh = bx2 - bx1, by2 - by1

    # Must be a bounded rectangular caption box (not a whole dark panel background)
    if bw > box_w * 2.8 or bh > box_h * 1.45 or bw > ww * 0.65 or bh > wh * 0.65:
        comp.close()
        return None

    comp_crop = comp.crop(c_bbox)
    area = comp_crop.histogram()[255]
    comp_crop.close()
    comp.close()

    rect_fill_ratio = area / float(max(1, bw * bh))
    if rect_fill_ratio >= 0.78:
        return bx1 + 3, by1 + 3, bx2 - 3, by2 - 3
    return None


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
    Erases Japanese text cleanly inside the speech bubble and returns:
      `(tx1, ty1, tx2, ty2, fg_text_rgb, stroke_rgb, hard_bubble_bounds)`
    """
    width, height = img.size
    x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2
    box_w = max(10, x2 - x1)
    box_h = max(10, y2 - y1)

    if box_w < 34 or box_h < 22 or (box_w < 65 and (box_h / float(box_w)) > 3.6):
        return None

    pad_x = min(240, max(120, int(box_w * 1.35)))
    pad_y = min(240, max(120, int(box_h * 1.35)))
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
    rel_cx = (rx1_rel + rx2_rel) // 2

    peer_clip_x1, peer_clip_y1, peer_clip_x2, peer_clip_y2 = _compute_peer_clip_bounds(
        width, height, x1, y1, x2, y2, other_peer_boxes
    )

    try:
        # =====================================================================
        # CASE A: DARK OR CHARCOAL BUBBLE / NARRATION CAPTION BOX
        # =====================================================================
        if is_dark_bubble:
            sub_rgb_a = img.crop((x1, y1, x2, y2))
            sub_hsv = sub_rgb_a.convert("HSV")
            sub_rgb_a.close()
            mean_sat = ImageStat.Stat(sub_hsv).mean[1]
            sub_hsv.close()

            sub_g_a = base_gray.crop((x1, y1, x2, y2))
            box_hist = sub_g_a.histogram()
            sub_g_a.close()
            total_box_px = max(1, sum(box_hist))
            bright_ratio = sum(box_hist[205:]) / total_box_px
            if mean_sat >= 34.0 or bright_ratio < 0.008:
                return None

            dark_fill_rgb, bg_lum = _sample_dark_bubble_bg_rgb_and_lum(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
            fg_text_rgb = (250, 250, 250)
            stroke_rgb = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))

            # Expand to full dark caption box rectangle if `box_2d` only covered one vertical column
            cap_rect = _find_enclosed_dark_caption_box(
                sub_gray, rx1_rel, ry1_rel, rx2_rel, ry2_rel, bg_lum
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
            diff_dilated = diff_from_bg.filter(ImageFilter.MaxFilter(5))
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

            dark_region_mask = sub_gray.point(lambda p: 255 if p <= 135 else 0)
            mid_y_start = ry1_rel + int((ry2_rel - ry1_rel) * 0.22)
            mid_y_end = ry1_rel + int((ry2_rel - ry1_rel) * 0.78)
            row_lefts = []
            row_rights = []
            for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), max(2, (mid_y_end - mid_y_start) // 6 or 1)):
                if not (0 <= sy < wh):
                    continue
                lx = rel_cx
                while lx > 2 and dark_region_mask.getpixel((lx - 1, sy)) > 128:
                    lx -= 1
                rx = rel_cx
                while rx < ww - 2 and dark_region_mask.getpixel((rx, sy)) > 128:
                    rx += 1
                if rx - lx >= 16:
                    row_lefts.append(lx)
                    row_rights.append(rx)
            dark_region_mask.close()

            if row_lefts and row_rights:
                row_lefts.sort()
                row_rights.sort()
                inner_l = max(peer_clip_x1, wx1 + row_lefts[int(len(row_lefts) * 0.75)] + 4)
                inner_r = min(peer_clip_x2, wx1 + row_rights[int(len(row_rights) * 0.25)] - 4)
                if inner_r - inner_l >= 22:
                    hard_b = (inner_l, raw_y1, inner_r, raw_y2)
                    return inner_l, raw_y1, inner_r, raw_y2, fg_text_rgb, stroke_rgb, hard_b

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
                ink_dilated = ink_inside_bubble.filter(ImageFilter.MaxFilter(5))
                ink_inside_bubble.close()
                erase_mask = ImageChops.darker(ink_dilated, clipped_core)
                ink_dilated.close()

                white_patch = Image.new("RGB", (ww, wh), (255, 255, 255))
                img.paste(white_patch, (wx1, wy1), mask=erase_mask)
                white_patch.close()
                erase_mask.close()

                clipped_bbox = clipped_core.getbbox() or lobe_bbox
                lb_x1, lb_y1, lb_x2, lb_y2 = clipped_bbox
                lobe_cx = (lb_x1 + lb_x2) // 2

                rel_y1 = lb_y1 + max(4, int((lb_y2 - lb_y1) * 0.08))
                rel_y2 = lb_y2 - max(4, int((lb_y2 - lb_y1) * 0.08))
                if rel_y2 - rel_y1 < 18:
                    rel_y1, rel_y2 = lb_y1 + 3, lb_y2 - 3

                mid_y_start = rel_y1 + int((rel_y2 - rel_y1) * 0.16)
                mid_y_end = rel_y1 + int((rel_y2 - rel_y1) * 0.84)
                row_lefts = []
                row_rights = []
                for sy in range(mid_y_start, max(mid_y_start + 1, mid_y_end), max(2, (mid_y_end - mid_y_start) // 7 or 1)):
                    if not (0 <= sy < wh):
                        continue
                    lx = lobe_cx
                    while lx > lb_x1 and clipped_core.getpixel((lx - 1, sy)) > 128:
                        lx -= 1
                    rx = lobe_cx
                    while rx < lb_x2 and clipped_core.getpixel((rx, sy)) > 128:
                        rx += 1
                    if rx - lx >= 14:
                        row_lefts.append(lx)
                        row_rights.append(rx)
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
        # =====================================================================
        tx_b1 = max(2, rx1_rel - 4)
        ty_b1 = max(2, ry1_rel - 4)
        tx_b2 = min(ww - 2, rx2_rel + 4)
        ty_b2 = min(wh - 2, ry2_rel + 4)

        sub_rgb_c = img.crop((x1, y1, x2, y2))
        sub_hsv_c = sub_rgb_c.convert("HSV")
        sub_rgb_c.close()
        mean_sat_c = ImageStat.Stat(sub_hsv_c).mean[1]
        sub_hsv_c.close()

        box_crop = sub_gray.crop((tx_b1, ty_b1, tx_b2, ty_b2))
        box_hist = box_crop.histogram()
        box_crop.close()
        total_box_px = max(1, sum(box_hist))
        white_paper_px = sum(box_hist[228:])
        midtone_px = sum(box_hist[135:228])
        dark_ink_px = sum(box_hist[:135])

        white_paper_ratio = white_paper_px / total_box_px
        midtone_ratio = midtone_px / total_box_px
        ink_ratio = dark_ink_px / total_box_px

        # Catches narrow dense 2-column border-touching white bubbles while rejecting pink polka-dot walls & skin
        is_valid_pure_white = (
            mean_sat_c < 16.0
            and white_paper_ratio >= 0.46
            and midtone_ratio <= 0.25
            and ink_ratio >= 0.015
        )
        light_mist_px = sum(box_hist[205:])
        is_valid_mist_bubble = (
            mean_sat_c < 8.0
            and (light_mist_px / total_box_px) >= 0.62
            and ink_ratio >= 0.015
        )

        if is_valid_pure_white or is_valid_mist_bubble:
            box_clamp = Image.new("L", (ww, wh), 0)
            ImageDraw.Draw(box_clamp).rectangle([tx_b1, ty_b1, tx_b2, ty_b2], fill=255)

            bg_rgb = (255, 255, 255) if is_valid_pure_white else _sample_light_bubble_bg_rgb(
                img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2
            )
            ink_cutoff = 228 if is_valid_pure_white else 188
            text_ink_mask = sub_gray.point(lambda p: 255 if p < ink_cutoff else 0)
            text_ink_dil = text_ink_mask.filter(ImageFilter.MaxFilter(5))
            text_ink_mask.close()
            erase_c = ImageChops.darker(text_ink_dil, box_clamp)
            text_ink_dil.close()
            box_clamp.close()

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
    pixel_boxes: list[tuple[int, int, int, int, bool, bool, str]]
) -> list[tuple[int, int, int, int, bool, bool, str]]:
    """
    Prevents duplicate or drifted `box_2d` coordinates pointing to the exact same speech
    bubble from cramming two translations into one bubble.
    """
    kept: list[tuple[int, int, int, int, bool, bool, str]] = []
    for item in pixel_boxes:
        rx1, ry1, rx2, ry2, _, _, _ = item
        area_a = max(1, (rx2 - rx1) * (ry2 - ry1))
        is_dup = False
        for k_item in kept:
            kx1, ky1, kx2, ky2, _, _, _ = k_item
            area_b = max(1, (kx2 - kx1) * (ky2 - ky1))
            ov_x = max(0, min(rx2, kx2) - max(rx1, kx1))
            ov_y = max(0, min(ry2, ky2) - max(ry1, ky1))
            ov_area = ov_x * ov_y
            if ov_area / float(min(area_a, area_b)) >= 0.55:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
    return kept


def _resolve_box_collisions(
    renders: list[list],
    min_gap: int = 6,
) -> list[list]:
    """
    Separates overlapping text boxes while strictly clamping every render box inside its
    own `hard_bubble_bounds` (`renders[k][7]`) so collision shifts can NEVER push text
    outside its speech bubble onto the page margin or character artwork.
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

                for idx_k in (i, j):
                    hx1, hy1, hx2, hy2 = renders[idx_k][7]
                    renders[idx_k][0] = max(hx1, min(hx2 - 18, renders[idx_k][0]))
                    renders[idx_k][1] = max(hy1, min(hy2 - 16, renders[idx_k][1]))
                    renders[idx_k][2] = min(hx2, max(renders[idx_k][0] + 18, renders[idx_k][2]))
                    renders[idx_k][3] = min(hy2, max(renders[idx_k][1] + 16, renders[idx_k][3]))
    return renders


def render_translated_manga_image_sync(raw_bytes: bytes, bubbles: list[dict]) -> bytes | None:
    """
    Renders translated dialogue cleanly inside speech bubbles at FULL ORIGINAL RESOLUTION
    (no downscaling of output images) with explicit C-heap cleanup + worker-thread `malloc_trim`.
    """
    if not raw_bytes:
        return None
    valid_boxes = [
        b for b in bubbles
        if isinstance(b.get("box_2d"), list)
        and len(b["box_2d"]) == 4
        and _sanitize_text_for_manga_font(b.get("translated_text") or "")
    ]
    if not valid_boxes:
        return None

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
            is_dark = _detect_bubble_polarity(base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
            src_txt = str(b.get("source_text") or "")
            has_kanji_kana = bool(re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]", src_txt))
            clean_tr = _sanitize_text_for_manga_font(b["translated_text"])
            if clean_tr:
                pixel_boxes.append((raw_x1, raw_y1, raw_x2, raw_y2, is_dark, has_kanji_kana, clean_tr))

        pixel_boxes = _deduplicate_overlapping_pixel_boxes(pixel_boxes)

        prepared_renders = []
        for i, (rx1, ry1, rx2, ry2, is_dark, has_kk, trans_text) in enumerate(pixel_boxes):
            if (rx2 - rx1) < 3 or (ry2 - ry1) < 3:
                continue
            other_boxes = [
                (ox1, oy1, ox2, oy2)
                for j, (ox1, oy1, ox2, oy2, _, _, _) in enumerate(pixel_boxes)
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
        max_font_px = max(24, int(max_dim * 0.019))

        draw = ImageDraw.Draw(img)
        for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text, _ in prepared_renders:
            box_w = max(20, tx2 - tx1)
            box_h = max(16, ty2 - ty1)
            usable_w = max(18, int(box_w * 0.94))
            usable_h = max(14, int(box_h * 0.94))
            is_thai_text = bool(THAI_CHAR_REGEX.search(trans_text))

            start_font_size = min(max_font_px, max(12, int(min(box_w, box_h) * 0.27)))
            chosen_font = _load_manga_font(9, is_thai=is_thai_text)
            chosen_lines = [trans_text]
            line_height = 11
            ref_top = 0

            sample_probe = "Agกข้ำ" if is_thai_text else "Agjpqy!"
            for fsize in range(start_font_size, 8, -1):
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
        img.save(out_io, format="JPEG", quality=92)
        return out_io.getvalue()
    finally:
        img.close()
        base_gray.close()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    """
    Translates 1 to 12 manga pages from direct Discord attachments OR embedded links
    (including multi-part X/Twitter threads up to 12 images total, Pixiv/phixiv, Bluesky,
    Discord embeds, or direct image URLs) at FULL ORIGINAL RESOLUTION with `gemini-decision-1`
    visual QA auditing and proactive C-heap RAM release.
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

    async with channel.typing():
        batch_results = await extract_and_translate_manga_pages_batch(
            pages_input, target_lang_name, series_context_hint=series_context_hint
        )
        release_unused_memory()

        if not any(bubbles or summary for bubbles, summary in batch_results):
            pages_input.clear()
            release_unused_memory()
            await _send_or_reply(
                channel,
                msg,
                content="ไม่สามารถตรวจจับข้อความในภาพที่แนบมาหรือจากลิงก์ได้ค่ะ กรุณาตรวจสอบว่าภาพมีความคมชัดนะคะ",
            )
            return

        # Pass 1: Render each page while retaining `raw_bytes` only until QA audit finishes
        rendered_pages_bytes: list[bytes | None] = [None] * num_pages
        for page_idx in range(1, num_pages + 1):
            raw_bytes, fname = pages_input[page_idx - 1]
            bubbles, _ = batch_results[page_idx - 1]
            if not bubbles:
                pages_input[page_idx - 1] = (b"", fname)
                continue
            try:
                rendered_pages_bytes[page_idx - 1] = await asyncio.to_thread(
                    render_translated_manga_image_sync, raw_bytes, bubbles
                )
            except Exception as e:
                print(f"Manga page {page_idx} initial render note: {e}", flush=True)
            if page_idx % 2 == 0:
                release_unused_memory()
            await asyncio.sleep(0.20)

        # Pass 2: Fast parallel `gemini-decision-1` QA audit (only when multi-page batch was used)
        if num_pages > 1 and GEMINI_API_KEY and time.time() >= state.gemini_decision_cooldown_until:
            qa_sem = asyncio.Semaphore(4)

            async def _qa_check(idx_0: int) -> tuple[int, bool]:
                r_bytes = rendered_pages_bytes[idx_0]
                if not r_bytes:
                    return idx_0, False
                async with qa_sem:
                    flagged = await _audit_rendered_manga_page_with_decision_1(r_bytes, idx_0 + 1)
                    return idx_0, flagged

            qa_outcomes = await asyncio.gather(*[_qa_check(i) for i in range(num_pages)])
            flagged_indices = [idx_0 for idx_0, is_flagged in qa_outcomes if is_flagged][:3]

            # Surgical single-page spatial re-grounding + re-render for any QA-flagged page
            for idx_0 in flagged_indices:
                raw_bytes, fname = pages_input[idx_0]
                if not raw_bytes:
                    continue
                try:
                    single_bubbles, single_summary = await extract_and_translate_manga_bubbles(
                        raw_bytes,
                        fname,
                        target_lang_name,
                        series_context_hint=series_context_hint,
                    )
                    if single_bubbles:
                        old_bubbles, old_summary = batch_results[idx_0]
                        batch_results[idx_0] = (single_bubbles, single_summary or old_summary)
                        rerendered = await asyncio.to_thread(
                            render_translated_manga_image_sync, raw_bytes, single_bubbles
                        )
                        if rerendered:
                            rendered_pages_bytes[idx_0] = rerendered
                            api_log(f"Manga QA Self-Heal -> Re-grounded & re-rendered Page {idx_0 + 1} ({len(single_bubbles)} bubbles)")
                except Exception as e:
                    print(f"Manga QA self-heal page {idx_0 + 1} note: {e}", flush=True)

        # Release all raw input image buffers immediately before building Discord payload
        for i in range(num_pages):
            pages_input[i] = (b"", pages_input[i][1])
        pages_input.clear()
        release_unused_memory()

        header_title = (
            f"**📖 คำแปลมังงะ {num_pages} หน้า ตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
            if num_pages > 1
            else f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
        )
        transcript_lines = [header_title]
        discord_files = []

        for page_idx in range(1, num_pages + 1):
            bubbles, scene_summary = batch_results[page_idx - 1]
            r_jpg = rendered_pages_bytes[page_idx - 1]
            fname = f"page_{page_idx}.jpg"

            if num_pages > 1:
                transcript_lines.append(f"\n__**📄 หน้าที่ {page_idx}/{num_pages}**__")
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
                    transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n  ↳ **{tr_txt}**")

            if r_jpg:
                base_name = os.path.splitext(fname)[0]
                discord_files.append(
                    discord.File(
                        io.BytesIO(r_jpg),
                        filename=f"translated_p{page_idx}_{base_name}.jpg",
                    )
                )
            rendered_pages_bytes[page_idx - 1] = None

        rendered_pages_bytes.clear()
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
        batch_results.clear()
        release_unused_memory()