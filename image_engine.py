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
    release_unused_memory,
)
from llm import (
    call_gemini_hydrogen_generate,
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

LATIN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
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


def _prepare_manga_image_for_vision(raw_bytes: bytes, max_dim: int = 1280, quality: int = 92) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    img.close()
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
        # Never place box_2d on floating SFX over characters/artwork (keep SFX in text transcript only)
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

        valid.append({
            "id": item.get("id", idx),
            "speaker_or_note": str(speaker).strip(),
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
    pages_input: list[tuple[bytes, str]], target_lang_name: str
) -> list[tuple[list[dict], str]]:
    """
    Processes 1 to 12 manga pages (including multi-part X/Twitter threads) in a SINGLE unified
    multimodal batch call (matching `image_engine_v8.py` architecture):
      - Tier 1: `gemini-4-hydrogen` (`thinkingLevel: "LOW"`, `maxOutputTokens: 65536`, no sampling params)
      - Tier 2: `gemini-3.1-flash-lite` -> `gemini-3.5-flash-lite` (`thinkingLevel: "LOW"`, `maxOutputTokens: 16384`)
      - Tier 3: Concurrent `Semaphore(2)` per-page fallback
    """
    num_pages = len(pages_input)
    if num_pages == 0:
        return []

    if num_pages == 1:
        b, s = await extract_and_translate_manga_bubbles(pages_input[0][0], pages_input[0][1], target_lang_name)
        return [(b, s)]

    if num_pages >= 9:
        batch_dim, batch_q = 832, 78
    elif num_pages >= 5:
        batch_dim, batch_q = 928, 82
    else:
        batch_dim, batch_q = 1080, 88

    b64_pages = []
    for raw_b, fname in pages_input:
        b64_str = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_b, batch_dim, batch_q)
        b64_pages.append((b64_str, fname))

    mime_type = "image/jpeg"

    manga_batch_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
You are given {num_pages} sequential manga page image(s) (labeled PAGE 1 to PAGE {num_pages}).
Analyze all pages together in sequence so series title, character names, pronouns, tone, and ongoing story stay consistent across pages,
and translate ALL speech bubbles, narration boxes, and SFX on every page into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. Reading Order: Within each page, order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. In-Context Localization & Honorifics:
   - Adapt idioms, pronouns, tone, and emotion so the dialogue sounds natural in {target_lang_name} while fitting inside a speech bubble. Do NOT use double quotes inside `translated_text` (use single quotes if needed).
   - When translating to English, ALWAYS preserve Japanese honorifics attached directly with a hyphen and NO space (translate さん as `-san`, くん/君 as `-kun`, ちゃん as `-chan`, 様/さま as `-sama`, 先生 as `-sensei`, 先輩 as `-senpai`).
3. Speech Bubble Bounding Boxes (`box_2d`) — STRICT PLACEMENT RULES:
   - ONLY provide `box_2d` (`[ymin, xmin, ymax, xmax]`, normalized 0..1000) for dialogue/narration inside an actual speech bubble or caption box!
   - `box_2d` MUST tightly enclose the Japanese text INSIDE the bubble (never point to empty white margin outside the bubble, and NEVER place `box_2d` over a character's face, body, or background artwork!).
   - For floating SFX / sound effects (e.g. ドーン, ドドン, スラ〜〜, ほう…？) OR tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble, ALWAYS set `"box_2d": null` so the artwork/character is NEVER overwritten!
   - NEVER merge two separate speech bubbles (or a speech bubble + an outside whisper) into a single `box_2d`! Keep each speech bubble strictly separate.
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

            timeout_sec = min(135, 50 + (9 * num_pages))

            # Tier 1: Gemini 4 Hydrogen (no sampling params, 65536 maxOutputTokens, `thinkingLevel: "LOW"`)
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
                    hydrogen_payload, timeout_sec=timeout_sec, purpose=f"MangaBatch({num_pages}p)"
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
                            f"[Manga Vision Batch] Completed {num_pages} page(s) in 1 request via Tier 1 Hydrogen "
                            f"({GEMINI_HEAVY_MODEL}) - {total_b} total bubbles",
                            flush=True,
                        )
                        return batch_results
            except Exception as e:
                print(f"Gemini Hydrogen Manga Batch warning ({type(e).__name__}): {e}", flush=True)

            # Tier 2: Gemini 3.1 Flash-Lite -> Gemini 3.5 Flash-Lite (`thinkingLevel: "LOW"`)
            flash_payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 16384,
                    "responseMimeType": "application/json",
                    "thinkingConfig": {"thinkingLevel": "LOW"},
                },
            }
            try:
                data = await call_gemini_generate_content_self_healing(
                    flash_payload, timeout_sec=timeout_sec, purpose=f"MangaBatchFallback({num_pages}p)"
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
                            f"[Manga Vision Batch] Completed {num_pages} page(s) in 1 request via Tier 2 Gemini Flash "
                            f"({state.active_gemini_model}) - {total_b} total bubbles",
                            flush=True,
                        )
                        return batch_results
            except Exception as e:
                print(f"Gemini Flash Manga Batch warning ({type(e).__name__}): {e}", flush=True)
    finally:
        b64_pages.clear()

    # Tier 3: Concurrent per-page fallback (2 pages at a time)
    sem = asyncio.Semaphore(2)

    async def _bounded_single(rb: bytes, fn: str) -> tuple[list[dict], str]:
        async with sem:
            return await extract_and_translate_manga_bubbles(rb, fn, target_lang_name)

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
3. Speech Bubble Bounding Boxes (`box_2d`) — STRICT PLACEMENT RULES:
   - ONLY provide `box_2d` (`[ymin, xmin, ymax, xmax]`, normalized 0..1000) for dialogue/narration inside an actual speech bubble or caption box!
   - `box_2d` MUST tightly enclose the Japanese text INSIDE the bubble (never point to empty white margin outside the bubble, and NEVER place `box_2d` over a character's face, body, or background artwork!).
   - For floating SFX / sound effects (e.g. ドーン, ドドン, スラ〜〜, ほう…？) OR tiny unenclosed whispers drawn directly over a character's body/shirt outside a speech bubble, ALWAYS set `"box_2d": null` so the artwork/character is NEVER overwritten!
   - NEVER merge two separate speech bubbles (or a speech bubble + an outside whisper) into a single `box_2d`! Keep each speech bubble strictly separate.
4. Return ONLY valid JSON matching this exact structure (every bubble MUST have a non-empty `translated_text`; write `summary` in {target_lang_name}):
{{
  "summary": "Brief 1-sentence scene context in {target_lang_name}",
  "bubbles": [
    {{
      "id": 1,
      "speaker_or_note": "Character / Narration / SFX",
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
            data = await call_gemini_generate_content_self_healing(gemini_payload, timeout_sec=30)
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
    Returns True if the bubble/panel region at (x1, y1, x2, y2) is a Dark/Charcoal background
    (median luminance < 125), or False if it is a Light/White bubble.
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
    return median_lum < 125


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
    Fast 0.5x-scale enclosed white speech bubble finder (exact `image_engine_v8.py` algorithm):
      - Rejects blank white cheeks/faces/backgrounds by checking enclosed character loops (`holes_area`).
      - Accepts thin-font speech bubbles where `holes_area >= 5` at 0.5x scale so they never miss
        enclosed bubble cleaning.
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
        for extra_dx in (-10, 10, -22, 22, -34, 34, -46, 46):
            sx_ext = (s_rx1 + extra_dx) if extra_dx < 0 else (s_rx2 + extra_dx)
            if 1 <= sx_ext < sw - 1:
                seed_candidates.append((sx_ext, sy))

    base_bin = small_gray.point(lambda p: 255 if p >= 232 else 0)
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

            edge_touches = (
                (1 if cx1 <= 1 else 0)
                + (1 if cy1 <= 1 else 0)
                + (1 if cx2 >= sw - 1 else 0)
                + (1 if cy2 >= sh - 1 else 0)
            )
            if edge_touches >= 3 or (cx1 <= 1 and cx2 >= sw - 1) or (cy1 <= 1 and cy2 >= sh - 1):
                continue

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

            if has_kanji_kana and holes_area < 5:
                continue

            ov_x = max(0, min(cx2, s_rx2) - max(cx1, s_rx1))
            ov_y = max(0, min(cy2, s_ry2) - max(cy1, s_ry1))
            overlap_area = ov_x * ov_y
            if overlap_area < 20:
                continue

            score = (overlap_area * 2.5) + (holes_area * 10.0) + (filled_area * 0.25)
            if pass_idx == 1:
                score *= 0.92
            if score > best_score:
                best_score = score
                best_small_filled = filled
                best_was_pinched = (pass_idx == 1)

        if best_small_filled is not None and best_score > 450:
            break

    if best_small_filled is None:
        return None

    if best_was_pinched:
        best_small_filled = best_small_filled.filter(ImageFilter.MaxFilter(3))

    full_filled = best_small_filled.resize((ww, wh), Image.Resampling.BILINEAR).point(
        lambda p: 255 if p >= 128 else 0
    )
    return full_filled


def _sample_dark_bubble_bg_rgb_and_lum(img: Image.Image, base_gray: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int, int], int]:
    """
    Samples the true background color (pure black OR charcoal gray) around the text box
    so dark/charcoal panels are filled with their EXACT matching background color.
    """
    width, height = img.size
    sx1 = max(0, x1 - 4)
    sy1 = max(0, y1 - 4)
    sx2 = min(width, x2 + 4)
    sy2 = min(height, y2 + 4)
    crop_rgb = img.crop((sx1, sy1, sx2, sy2))
    crop_gray = base_gray.crop((sx1, sy1, sx2, sy2))

    gray_vals = list(crop_gray.getdata())
    rgb_vals = list(crop_rgb.getdata())
    if not gray_vals:
        return (20, 20, 20), 20

    bg_indices = [i for i, g in enumerate(gray_vals) if g < 145]
    if not bg_indices:
        bg_indices = list(range(len(gray_vals)))
    bg_indices.sort(key=lambda idx: gray_vals[idx])
    mid_idx = bg_indices[len(bg_indices) // 2]
    bg_lum = int(gray_vals[mid_idx])
    bg_rgb = rgb_vals[mid_idx]
    return bg_rgb, bg_lum


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
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int]] | None:
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
    # CASE A: DARK OR CHARCOAL BUBBLE / PANEL (Verify low color saturation + bright white text inside)
    # =========================================================================
    if is_dark_bubble:
        sub_hsv = img.crop((x1, y1, x2, y2)).convert("HSV")
        mean_sat = ImageStat.Stat(sub_hsv).mean[1]
        box_pixels = list(base_gray.crop((x1, y1, x2, y2)).getdata())
        bright_ratio = sum(1 for p in box_pixels if p >= 205) / max(1, len(box_pixels))
        # Reject colorful character hair/clothes/skin that have no white text strokes
        if mean_sat >= 42.0 or bright_ratio < 0.006:
            return None

        dark_fill_rgb, bg_lum = _sample_dark_bubble_bg_rgb_and_lum(img, base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        fg_text_rgb = (250, 250, 250)
        stroke_rgb = (max(0, bg_lum - 20), max(0, bg_lum - 20), max(0, bg_lum - 20))

        tx_clean_x1 = max(2, rx1_rel - 6)
        tx_clean_y1 = max(2, ry1_rel - 6)
        tx_clean_x2 = min(ww - 2, rx2_rel + 6)
        tx_clean_y2 = min(wh - 2, ry2_rel + 6)

        text_box_bounds = Image.new("L", (ww, wh), 0)
        ImageDraw.Draw(text_box_bounds).rectangle(
            [tx_clean_x1, tx_clean_y1, tx_clean_x2, tx_clean_y2], fill=255
        )

        diff_from_bg = sub_gray.point(lambda p: 255 if abs(int(p) - bg_lum) > 18 else 0)
        erase_mask_dark = ImageChops.darker(
            diff_from_bg.filter(ImageFilter.MaxFilter(5)),
            text_box_bounds,
        )
        img.paste(Image.new("RGB", (ww, wh), dark_fill_rgb), (wx1, wy1), mask=erase_mask_dark)

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

        if row_lefts and row_rights:
            row_lefts.sort()
            row_rights.sort()
            inner_l = max(rx1_rel - 10, row_lefts[int(len(row_lefts) * 0.75)] + 4)
            inner_r = min(rx2_rel + 10, row_rights[int(len(row_rights) * 0.25)] - 4)
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
    # CASE B: ENCLOSED WHITE SPEECH BUBBLE (Exact `image_engine_v8.py` Lobe Snap)
    # =========================================================================
    filled_bubble = _find_best_enclosed_white_bubble(
        sub_gray, rx1_rel, ry1_rel, rx2_rel, ry2_rel, has_kanji_kana=has_kanji_kana
    )

    if filled_bubble is not None:
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
    # CASE C: OPEN-SIDED WHITE SPEECH BUBBLE (Strictly Clamped to `box_2d`)
    # =========================================================================
    tx_b1 = max(2, rx1_rel - 6)
    ty_b1 = max(2, ry1_rel - 6)
    tx_b2 = min(ww - 2, rx2_rel + 6)
    ty_b2 = min(wh - 2, ry2_rel + 6)

    box_crop = sub_gray.crop((tx_b1, ty_b1, tx_b2, ty_b2))
    box_hist = box_crop.histogram()
    total_box_px = max(1, sum(box_hist))
    pure_white_px = sum(box_hist[242:])
    white_ratio = pure_white_px / total_box_px

    if white_ratio >= 0.45:
        local_max = sub_gray.filter(ImageFilter.MaxFilter(7))
        white_paper_zone = local_max.point(lambda p: 255 if p >= 242 else 0).filter(ImageFilter.MinFilter(3))

        box_clamp = Image.new("L", (ww, wh), 0)
        ImageDraw.Draw(box_clamp).rectangle([tx_b1, ty_b1, tx_b2, ty_b2], fill=255)
        safe_open_zone = ImageChops.darker(white_paper_zone, box_clamp)

        dark_strokes = sub_gray.point(lambda p: 255 if p < 238 else 0)
        candidate_text_ink = ImageChops.darker(dark_strokes, safe_open_zone)

        if candidate_text_ink.getbbox():
            erase_c = ImageChops.darker(
                candidate_text_ink.filter(ImageFilter.MaxFilter(5)),
                box_clamp,
            )
            img.paste(Image.new("RGB", (ww, wh), (255, 255, 255)), (wx1, wy1), mask=erase_c)
            return raw_x1, raw_y1, raw_x2, raw_y2, (15, 15, 15), (255, 255, 255)

    # Reject completely blank white margins (no text ink inside `box_2d`) or colorful artwork
    return None


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
        raw_x1 = max(0, int((xmin / 1000.0) * width))
        raw_y1 = max(0, int((ymin / 1000.0) * height))
        raw_x2 = min(width, int((xmax / 1000.0) * width))
        raw_y2 = min(height, int((ymax / 1000.0) * height))
        is_dark = _detect_bubble_polarity(base_gray, raw_x1, raw_y1, raw_x2, raw_y2)
        src_txt = str(b.get("source_text") or "")
        has_kanji_kana = bool(re.search(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]", src_txt))
        clean_tr = re.sub(r"\s+", " ", b["translated_text"]).strip()
        pixel_boxes.append((raw_x1, raw_y1, raw_x2, raw_y2, is_dark, has_kanji_kana, clean_tr))

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
        lx1, ly1, lx2, ly2, fg_text, stroke_col = cleaned_lobe
        prepared_renders.append([lx1, ly1, lx2, ly2, fg_text, stroke_col, trans_text])

    if not prepared_renders:
        img.close()
        base_gray.close()
        return None

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
    img.save(out_io, format="PNG", compress_level=2)
    img.close()
    base_gray.close()
    return out_io.getvalue()


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    """
    Translates 1 to 12 manga pages from direct Discord attachments OR embedded links
    (including multi-part X/Twitter threads up to 12 images total, Pixiv/phixiv, Bluesky,
    Discord embeds, or direct image URLs).
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

    if not pages_input:
        embedded_imgs, _, _ = await collect_embedded_images_and_metadata(
            msg, ref_msg=ref_msg, max_images=MAX_MANGA_BATCH_PAGES
        )
        for emb_bytes, emb_fname, _ in embedded_imgs:
            pages_input.append((emb_bytes, emb_fname))

    if not pages_input:
        await channel.send(
            "กรุณาแนบรูปภาพหน้ามังงะ หรือวางลิงก์ที่มีรูปภาพ (เช่น X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วยนะคะ "
            "(สามารถแนบหรือวางลิงก์ที่มีหลายหน้าพร้อมกันได้เลยค่ะ เช่น พิมพ์ `!manga en` หรือ `!manga th`)"
        )
        return

    _, target_lang_name = _detect_target_translation_lang(user_prompt)
    pages_input = pages_input[:MAX_MANGA_BATCH_PAGES]
    num_pages = len(pages_input)

    async with channel.typing():
        batch_results = await extract_and_translate_manga_pages_batch(
            pages_input, target_lang_name
        )

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
                    transcript_lines.append(f"**[#{idx} | {speaker}]** `{src_txt}`\n  ↳ **{tr_txt}**")

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
                    # Yield briefly between CPU-heavy Pillow renders so Discord Gateway heartbeat never lags
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Manga page {page_idx} rendering fallback to text-only: {e}", flush=True)

        transcript_text = "\n".join(transcript_lines)

        # Discord allows up to 10 attachments per single message; split into chunks of 10 when >10 pages
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
        rendered_png = raw_bytes = None
        release_unused_memory()