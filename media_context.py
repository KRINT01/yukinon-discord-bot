import io
import re
import json
import time
import base64
import asyncio
import urllib.request
from urllib.parse import urlparse, parse_qs, quote_plus

import aiohttp
import discord
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

from config import (
    HTTPS,
    YOUTUBE_PROXY_URL,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    ACTIVE_VISION_MODEL,
    AUDIO_EXTENSIONS,
    CODE_AND_LOG_EXTENSIONS,
    PLAIN_TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    CUSTOM_EMOJI_REGEX,
    URL_REGEX,
    ollama_lock,
    channel_last_attachment,
    state,
    clean_url,
    api_log,
    get_http_session,
    get_modal_headers,
    prune_stale_attachments,
)
from persona import TYPHOON_OCR_PROMPT
from search_engine import fetch_webpage_sync, run_safe_web_search_sync
from llm import (
    discover_active_groq_vision_models,
    call_cloudflare_vision_chat,
    call_gemini_generate_content_self_healing,
)


def parse_xlsx_bytes_to_markdown(raw_bytes: bytes, max_chars: int = 25000) -> str:
    import openpyxl  # Lazy-loaded to save startup RAM on 512MB Render container

    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True, read_only=True)
    out_sections = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_iter = ws.iter_rows(values_only=True)
        non_empty_rows = [
            row for row in rows_iter if any(cell is not None and str(cell).strip() != "" for cell in row)
        ]
        if not non_empty_rows:
            continue

        def clean_cell(val):
            return (
                str(val).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|").strip()
                if val is not None else ""
            )

        header = [clean_cell(c) for c in non_empty_rows[0]]
        col_count = len(header)
        md_lines = [
            f"### Sheet: {sheet_name}",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * col_count) + " |"
        ]
        for row in non_empty_rows[1:350]:
            padded = [clean_cell(row[i]) if i < len(row) else "" for i in range(col_count)]
            md_lines.append("| " + " | ".join(padded) + " |")
        out_sections.append("\n".join(md_lines))
    wb.close()
    return "\n\n".join(out_sections)[:max_chars] if out_sections else "(Empty Excel workbook)"


def parse_pdf_bytes(raw_bytes: bytes, max_chars: int = 30000) -> str:
    from pypdf import PdfReader  # Lazy-loaded to save startup RAM

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text = [
        f"--- Page {i+1} ---\n{p.extract_text().strip()}"
        for i, p in enumerate(reader.pages) if p.extract_text()
    ]
    return "\n\n".join(pages_text)[:max_chars]


def parse_docx_bytes(raw_bytes: bytes, max_chars: int = 30000) -> str:
    import docx  # Lazy-loaded to save startup RAM

    doc = docx.Document(io.BytesIO(raw_bytes))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append("| " + " | ".join(c.text.replace("\n", " ").strip() for c in row.cells) + " |")
    return "\n\n".join(parts)[:max_chars]


def format_code_or_log_attachment(text: str, is_log: bool, max_chars: int = 120000) -> str:
    if is_log:
        if len(text) <= 30000:
            return text
        return f"{text[:4000]}\n\n... [ omitted middle log lines ] ...\n\n{text[-25000:]}"

    lines = text.splitlines()
    numbered_lines = [f"{idx:4d} | {line}" for idx, line in enumerate(lines, 1)]
    numbered_text = "\n".join(numbered_lines)

    if len(numbered_text) <= max_chars:
        return numbered_text
    half = max_chars // 2
    return (
        f"{numbered_text[:half]}\n\n"
        f"... [ file exceeds {max_chars} chars; middle section omitted ] ...\n\n"
        f"{numbered_text[-half:]}"
    )


def extract_youtube_id(url: str):
    try:
        parsed = urlparse(clean_url(url))
        host = (parsed.hostname or "").lower()
        if host in ("youtu.be", "www.youtu.be"):
            return parsed.path.lstrip("/").split("?")[0][:11]
        if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                vid = parse_qs(parsed.query).get("v", [None])[0]
                return vid[:11] if vid else None
            for prefix in ("/shorts/", "/embed/", "/live/", "/v/"):
                if parsed.path.startswith(prefix):
                    return parsed.path[len(prefix):].split("/")[0].split("?")[0][:11]
    except Exception:
        pass
    return None


def fetch_youtube_context_sync(
    video_id: str, discord_embed_info: dict | None = None, max_chars: int = 20000
) -> str:
    """
    4-Layer YouTube Context Extractor (Japanese, Thai, & English):
    1) Uses Discord's own message.embeds metadata.
    2) Queries noembed.com + youtube.com/oembed for Video Title & Channel Name.
    3) Extracts the full uploader shortDescription from the watch page HTML.
    4) Fetches subtitles/transcripts prioritizing ['th', 'ja', 'en', 'en-US', 'en-GB'].
    """
    t0 = time.monotonic()
    watch_url = clean_url(f"{HTTPS}www.youtube.com/watch?v={video_id}")
    title = (discord_embed_info or {}).get("title", "").strip()
    author = (discord_embed_info or {}).get("author", "").strip()
    description = (discord_embed_info or {}).get("description", "").strip()
    transcript_text = ""

    # Layer 1: Noembed & YouTube oEmbed
    if not title or not author:
        for oe_endpoint in (
            clean_url(f"{HTTPS}noembed.com/embed?url={quote_plus(watch_url)}"),
            clean_url(f"{HTTPS}www.youtube.com/oembed?url={quote_plus(watch_url)}&format=json"),
        ):
            try:
                req = urllib.request.Request(oe_endpoint, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    oe_data = json.loads(resp.read().decode("utf-8"))
                    title = title or (oe_data.get("title") or "").strip()
                    author = author or (oe_data.get("author_name") or "").strip()
                    if title and author:
                        break
            except Exception:
                continue

    # Layer 2: Watch Page HTML
    if len(description) < 200 or not title:
        try:
            req_page = urllib.request.Request(
                watch_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept-Language": "ja,th;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            )
            with urllib.request.urlopen(req_page, timeout=10) as resp:
                html_str = resp.read().decode("utf-8", errors="replace")

            if not title:
                t_m = re.search(r'<meta\s+name="title"\s+content="([^"]+)"', html_str)
                if t_m:
                    title = t_m.group(1).strip()

            desc_m = re.search(r'"shortDescription"\s*:\s*"((?:\\.|[^"\\])*)"', html_str)
            if desc_m:
                try:
                    full_d = json.loads(f'"{desc_m.group(1)}"').strip()
                except Exception:
                    full_d = desc_m.group(1).replace("\\n", "\n").strip()
                if len(full_d) > len(description):
                    description = full_d
        except Exception as e:
            api_log(f"YouTube HTML metadata note ({video_id}): {type(e).__name__}")

    # Layer 3: Subtitles / Transcript strictly in Thai, Japanese, or English
    proxy_cfg = (
        GenericProxyConfig(http_url=YOUTUBE_PROXY_URL, https_url=YOUTUBE_PROXY_URL)
        if YOUTUBE_PROXY_URL else None
    )
    try:
        ytt_api = YouTubeTranscriptApi(proxy_config=proxy_cfg)
        preferred_langs = ["th", "ja", "en", "en-US", "en-GB"]
        try:
            fetched = ytt_api.fetch(video_id, languages=preferred_langs)
            transcript_text = " ".join(s.text for s in fetched).strip()
        except Exception:
            transcript_list = ytt_api.list(video_id)
            for lang_code in preferred_langs:
                try:
                    tr_obj = transcript_list.find_transcript([lang_code])
                    fetched = tr_obj.fetch()
                    transcript_text = " ".join(s.text for s in fetched).strip()
                    if transcript_text:
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # Layer 4: Supplemental Web Search if both transcript and description are sparse
    search_enrichment = ""
    if len(transcript_text) < 80 and len(description) < 80 and title:
        try:
            search_enrichment = run_safe_web_search_sync(f"{title} {author}".strip(), max_results=3)
        except Exception:
            pass

    api_log(
        f"YouTube -> id={video_id} (title={'yes' if title else 'no'}, "
        f"desc={len(description)}c, transcript={len(transcript_text)}c, {time.monotonic() - t0:.2f}s)"
    )

    sections = [f"YouTube Video ID: {video_id}"]
    if title:
        sections.append(f"Video Title: {title}")
    if author:
        sections.append(f"Channel / Uploader: {author}")
    if description:
        sections.append(f"Official Video Description:\n{description[:6000]}")
    if transcript_text:
        sections.append(f"Video Subtitles / Transcript:\n{transcript_text[:14000]}")
    if search_enrichment:
        sections.append(f"Supplemental Web Search Context:\n{search_enrichment}")

    return "\n\n".join(sections)[:max_chars]


async def extract_image_via_transient_ocr(raw_bytes: bytes, filename: str) -> str:
    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
    fname_lower = filename.lower()
    if fname_lower.endswith(".png"):
        mime_type = "image/png"
    elif fname_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    ocr_prompt = (
        "Analyze this image thoroughly for a roleplay character (Yukino Yukinoshita) who needs to both "
        "read any text AND give her sharp personal impression/critique of the image.\n"
        "Provide your output in two concise sections:\n"
        "1. [Visual Scene & Details]: Describe the subject, characters/people/animals (especially any cats or pandas), "
        "expressions, environment, aesthetic quality, composition, and any notable, messy, or amusing details.\n"
        "2. [Verbatim OCR Text]: Transcribe ALL visible text (Thai, Japanese, English) verbatim. "
        "Pay close attention to Thai tone marks (วรรณยุกต์), vowels (สระ), and visually similar consonants "
        "(ก/ภ/ถ, ข/ช, ค/ด/ต, พ/ผ/ฝ/ฟ, ร/ธ), and note if the original text contains any spelling errors."
    )

    # Tier 1: Groq LPU Vision (qwen/qwen3.8-27b)
    if GROQ_API_KEY:
        groq_candidates = await discover_active_groq_vision_models()
        chat_url = clean_url(f"{HTTPS}api.groq.com/openai/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        session = get_http_session()
        for g_mod in groq_candidates:
            t0 = time.monotonic()
            try:
                groq_vision_payload = {
                    "model": g_mod,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ocr_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                }
                async with session.post(
                    chat_url, headers=headers, json=groq_vision_payload, timeout=aiohttp.ClientTimeout(total=35)
                ) as resp:
                    elapsed = time.monotonic() - t0
                    if resp.status == 200:
                        state.active_groq_vision_model = g_mod
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                        if content:
                            api_log(f"Image OCR (Tier 1 Groq) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                            return content
                    api_log(f"Image OCR (Tier 1 Groq) -> {g_mod} HTTP {resp.status}; fallback to Tier 2")
            except Exception as e:
                api_log(f"Image OCR (Tier 1 Groq) -> {g_mod} failed ({type(e).__name__}); fallback to Tier 2")

    # Tier 2: Cloudflare Workers AI Vision (@cf/google/gemma-4-26b-a4b-it)
    cf_ocr = await call_cloudflare_vision_chat(ocr_prompt, mime_type, b64_img, purpose="Image OCR Tier 2")
    if cf_ocr:
        return cf_ocr

    # Tier 3: Google AI Studio Backup (gemini-3-flash-preview)
    if GEMINI_API_KEY:
        g_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": ocr_prompt},
                        {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        g_data = await call_gemini_generate_content_self_healing(
            g_payload, timeout_sec=25, purpose="Image OCR Tier 3"
        )
        if g_data:
            parts = ((g_data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            g_text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
            if g_text:
                return g_text

    # Tier 4: Modal T4 GPU (scb10x/typhoon-ocr1.5-3b)
    payload = {
        "model": ACTIVE_VISION_MODEL,
        "messages": [
            {"role": "user", "content": TYPHOON_OCR_PROMPT, "images": [b64_img]}
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.1, "top_p": 0.6, "repeat_penalty": 1.1},
    }
    t0 = time.monotonic()
    try:
        async with ollama_lock:
            session = get_http_session()
            async with session.post(
                clean_url(f"{OLLAMA_BASE_URL}/api/chat"),
                headers=get_modal_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        api_log(f"Image OCR (Tier 4 Modal) -> {ACTIVE_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
                api_log(f"Image OCR (Tier 4 Modal) -> {ACTIVE_VISION_MODEL} HTTP {resp.status}")
    except Exception as e:
        api_log(f"Image OCR (Tier 4 Modal) -> {ACTIVE_VISION_MODEL} failed ({type(e).__name__})")

    return "[OCR could not extract text from image]"


async def extract_all_context(message: discord.Message) -> tuple[str, str]:
    from voice_engine import convert_audio_bytes_to_wav_sync, transcribe_wav_bytes_hybrid

    blocks, voice_clip_transcripts = [], []
    has_code_file = False
    for att in message.attachments:
        if att.size > 15 * 1024 * 1024:
            continue
        try:
            raw_bytes = await att.read()
            fname = att.filename.lower()
            if fname.endswith(AUDIO_EXTENSIONS):
                wav_bytes = await asyncio.to_thread(convert_audio_bytes_to_wav_sync, raw_bytes)
                spoken_txt = await transcribe_wav_bytes_hybrid(wav_bytes)
                if spoken_txt:
                    voice_clip_transcripts.append(spoken_txt)
                    blocks.append(f"=== Transcribed Voice Message (`{att.filename}`) ===\n{spoken_txt}")
            elif fname.endswith(CODE_AND_LOG_EXTENSIONS) or fname.endswith(PLAIN_TEXT_EXTENSIONS):
                is_log = fname.endswith(".log")
                if fname.endswith(CODE_AND_LOG_EXTENSIONS):
                    has_code_file = True
                formatted_code = format_code_or_log_attachment(
                    raw_bytes.decode("utf-8", errors="replace"), is_log=is_log
                )
                blocks.append(
                    f"=== Complete Attached Source File: `{att.filename}` (with line numbers) ===\n"
                    f"```\n{formatted_code}\n```"
                )
            elif fname.endswith(".xlsx"):
                blocks.append(
                    f"=== Attached Excel: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_xlsx_bytes_to_markdown, raw_bytes)
                )
            elif fname.endswith(".pdf"):
                blocks.append(
                    f"=== Attached PDF: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_pdf_bytes, raw_bytes)
                )
            elif fname.endswith(".docx"):
                blocks.append(
                    f"=== Attached Word: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_docx_bytes, raw_bytes)
                )
            elif fname.endswith(IMAGE_EXTENSIONS):
                blocks.append(
                    f"=== Image OCR (`{att.filename}`) ===\n"
                    + await extract_image_via_transient_ocr(raw_bytes, att.filename)
                )
        except Exception as e:
            blocks.append(f"[Error reading `{att.filename}`: {e}]")

    # Extract Discord Stickers (Name + Optional Vision Inspection for PNG/APNG/GIF stickers)
    for stk in getattr(message, "stickers", []) or []:
        stk_name = getattr(stk, "name", "Unknown Sticker")
        stk_fmt = str(getattr(stk, "format", "")).lower()
        stk_desc = f"User sent a Discord Sticker named: `{stk_name}`"
        if any(ext in stk_fmt for ext in ("png", "apng", "gif")) and getattr(stk, "url", None):
            try:
                session = get_http_session()
                async with session.get(
                    clean_url(str(stk.url)), timeout=aiohttp.ClientTimeout(total=8)
                ) as s_resp:
                    if s_resp.status == 200:
                        stk_bytes = await s_resp.read()
                        if len(stk_bytes) <= 8 * 1024 * 1024:
                            stk_vis = await extract_image_via_transient_ocr(stk_bytes, f"{stk_name}.png")
                            if stk_vis and "[OCR could not" not in stk_vis:
                                stk_desc += f"\nVisual/Sticker Details:\n{stk_vis}"
            except Exception as e:
                print(f"Sticker vision note ({stk_name}): {e}")
        blocks.append(f"=== Discord Sticker ===\n{stk_desc}")

    # Translate Custom Discord Emojis (<:name:id> / <a:name:id>) into readable emote names
    custom_emojis = CUSTOM_EMOJI_REGEX.findall(message.content or "")
    if custom_emojis:
        emote_names = [f":{name}:" for name, _ in custom_emojis]
        blocks.append(
            f"=== Custom Discord Emojis in User Message ===\n"
            f"User included custom emoji(s): {', '.join(emote_names)}"
        )

    urls_in_msg = URL_REGEX.findall(message.content)[:2]
    if urls_in_msg and not message.embeds:
        try:
            await asyncio.sleep(1.2)
            message = await message.channel.fetch_message(message.id)
        except Exception:
            pass

    for url in urls_in_msg:
        yt_id = extract_youtube_id(url)
        if yt_id:
            embed_info = {}
            for emb in message.embeds:
                if emb.title or emb.description or (emb.author and emb.author.name):
                    embed_info = {
                        "title": emb.title or "",
                        "author": emb.author.name if emb.author else "",
                        "description": emb.description or "",
                    }
                    break
            try:
                yt_context = await asyncio.to_thread(fetch_youtube_context_sync, yt_id, embed_info)
                if yt_context:
                    blocks.append(f"=== YouTube Video Information (`{url}`) ===\n{yt_context}")
            except Exception as e:
                print(f"YouTube context error ({yt_id}): {e}")
        else:
            try:
                blocks.append(
                    f'<untrusted_web_search_results url="{url}">\n'
                    + await asyncio.to_thread(fetch_webpage_sync, url)
                    + "\n</untrusted_web_search_results>"
                )
            except Exception as e:
                blocks.append(f"[Web extraction error: {e}]")

    joined_context = "\n\n".join(blocks)[:125000]
    prune_stale_attachments()
    if joined_context and message.channel:
        channel_last_attachment[message.channel.id] = {
            "content": joined_context,
            "is_code": has_code_file,
            "timestamp": time.time(),
        }
    return joined_context, " ".join(voice_clip_transcripts).strip()