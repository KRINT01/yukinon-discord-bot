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
    SAUCENAO_API_KEY,
    OLLAMA_BASE_URL,
    ACTIVE_VISION_MODEL,
    AUDIO_EXTENSIONS,
    CODE_AND_LOG_EXTENSIONS,
    PLAIN_TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    IMAGE_SOURCE_INTENT_PATTERN,
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
    call_gemma_ai_studio_vision,
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

    # Layer 4: Supplemental Web Search whenever a video has no/minimal spoken transcript (e.g. game trailers / PVs)
    search_enrichment = ""
    if len(transcript_text) < 120 and title:
        clean_title_for_search = re.sub(r"[|_—–-]\s*(?:Release Date Trailer|Official Trailer|Trailer|PV|Teaser).*", "", title, flags=re.IGNORECASE).strip()
        try:
            search_enrichment = run_safe_web_search_sync(f"{clean_title_for_search or title} {author}".strip(), max_results=4)
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


async def _search_saucenao_async(raw_bytes: bytes, filename: str) -> str:
    """
    Queries SauceNAO API (if SAUCENAO_API_KEY is configured) to find exact Pixiv, Twitter/X,
    Danbooru, and MangaDex sources (including artist name, Pixiv link, and manga chapter/part).
    """
    if not SAUCENAO_API_KEY:
        return ""
    t0 = time.monotonic()
    saucenao_url = clean_url(
        f"{HTTPS}saucenao.com/search.php?output_type=2&numres=4&api_key={SAUCENAO_API_KEY}"
    )
    try:
        form = aiohttp.FormData()
        form.add_field("file", raw_bytes, filename=filename, content_type="application/octet-stream")
        session = get_http_session()
        async with session.post(
            saucenao_url, data=form, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)

        results = data.get("results") or []
        matches = []
        for item in results:
            hdr = item.get("header") or {}
            sim = float(hdr.get("similarity") or 0.0)
            if sim < 68.0:
                continue
            d = item.get("data") or {}
            title = d.get("title") or d.get("source") or d.get("eng_name") or d.get("jp_name") or ""
            part = d.get("part") or ""
            author = d.get("member_name") or d.get("author") or d.get("creator") or ""
            if isinstance(author, list):
                author = ", ".join(str(x) for x in author)
            characters = d.get("characters") or ""
            material = d.get("material") or ""
            pixiv_id = d.get("pixiv_id")
            ext_urls = d.get("ext_urls") or []
            if pixiv_id and not any("pixiv.net" in u for u in ext_urls):
                ext_urls.insert(0, f"{HTTPS}www.pixiv.net/artworks/{pixiv_id}")

            info_parts = [f"Similarity: {sim:.1f}%"]
            if title:
                info_parts.append(f"Title/Source: {title}{f' ({part})' if part else ''}")
            if material:
                info_parts.append(f"Series/Material: {material}")
            if characters:
                info_parts.append(f"Character(s): {characters}")
            if author:
                info_parts.append(f"Artist/Author: {author}")
            if ext_urls:
                info_parts.append(f"URLs: {' | '.join(ext_urls[:2])}")
            matches.append("- " + " | ".join(info_parts))

        if matches:
            api_log(f"ReverseImage (SauceNAO) -> {len(matches)} match(es) ({time.monotonic() - t0:.2f}s)")
            return "=== SauceNAO Reverse Image Matches (Pixiv / Manga / Art) ===\n" + "\n".join(matches[:3])
    except Exception as e:
        api_log(f"ReverseImage (SauceNAO) -> skipped ({type(e).__name__})")
    return ""


async def _search_iqdb_keyless_async(raw_bytes: bytes, filename: str) -> str:
    """
    Keyless fallback for anime/manga illustrations: posts the image to iqdb.org
    and extracts high-similarity (>72%) Danbooru/Yande.re/Konachan matches and source tags.
    """
    t0 = time.monotonic()
    iqdb_url = f"{HTTPS}iqdb.org/"
    try:
        # Resize thumbnail to <= 600px JPEG so IQDB upload is fast (<1s) and under size limits
        from PIL import Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.thumbnail((600, 600), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        thumb_bytes = buf.getvalue()
        img.close()

        form = aiohttp.FormData()
        form.add_field("file", thumb_bytes, filename="query.jpg", content_type="image/jpeg")
        session = get_http_session()
        async with session.post(
            iqdb_url,
            data=form,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return ""
            html_text = await resp.text()

        # Look for 'Best match' block with similarity percentage
        if "Best match" not in html_text:
            return ""
        sim_m = re.search(r"(\d+)%\s*similarity", html_text)
        if not sim_m or int(sim_m.group(1)) < 72:
            return ""
        sim_pct = int(sim_m.group(1))

        link_m = re.search(
            r"Best match.*?href=[\"']((?:https?:)?//[^\"']+)[\"'][^>]*>\s*<img[^>]*alt=[\"']([^\"']*)[\"']",
            html_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not link_m:
            return ""
        match_url = link_m.group(1)
        if match_url.startswith("//"):
            match_url = "https:" + match_url
        alt_tags = link_m.group(2).strip()

        api_log(f"ReverseImage (IQDB Keyless) -> {sim_pct}% match ({time.monotonic() - t0:.2f}s)")
        return (
            f"=== IQDB Reverse Illustration Match ===\n"
            f"- Similarity: {sim_pct}% | Source Archive URL: {match_url}\n"
            f"- Detected Tags / Character / Series: {alt_tags}"
        )
    except Exception:
        return ""


async def _search_tracemoe_async(raw_bytes: bytes, mime_type: str) -> str:
    """
    Keyless Anime Screenshot Finder via trace.moe API (https://api.trace.moe/search?anilistInfo&cutBorders).
    Identifies exact Anime Title, Episode number, and Timestamp (MM:SS) when similarity >= 87%.
    """
    t0 = time.monotonic()
    trace_url = f"{HTTPS}api.trace.moe/search?anilistInfo&cutBorders"
    try:
        session = get_http_session()
        async with session.post(
            trace_url,
            data=raw_bytes,
            headers={"Content-Type": mime_type},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()

        docs = data.get("result") or []
        if not docs:
            return ""
        top = docs[0]
        sim = float(top.get("similarity") or 0.0) * 100.0
        if sim < 87.0:
            return ""

        anilist = top.get("anilist") or {}
        title_obj = anilist.get("title") if isinstance(anilist, dict) else {}
        romaji = (title_obj or {}).get("romaji") or ""
        english = (title_obj or {}).get("english") or ""
        native = (title_obj or {}).get("native") or ""
        mal_id = anilist.get("idMal") if isinstance(anilist, dict) else None
        ani_id = anilist.get("id") if isinstance(anilist, dict) else None

        ep = top.get("episode") or "Movie/OVA"
        from_sec = int(float(top.get("from") or 0))
        mm, ss = divmod(from_sec, 60)
        hh, mm = divmod(mm, 60)
        ts_str = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh > 0 else f"{mm:02d}:{ss:02d}"

        mal_link = f"{HTTPS}myanimelist.net/anime/{mal_id}" if mal_id else (
            f"{HTTPS}anilist.co/anime/{ani_id}" if ani_id else ""
        )
        api_log(f"ReverseImage (trace.moe) -> '{romaji or native}' Ep.{ep} at {ts_str} ({sim:.1f}%, {time.monotonic() - t0:.2f}s)")
        return (
            f"=== trace.moe Anime Screenshot Match ===\n"
            f"- Anime Title: {romaji} / {english} ({native})\n"
            f"- Episode: {ep} | Exact Scene Timestamp: {ts_str} | Similarity: {sim:.1f}%\n"
            + (f"- Reference Link: {mal_link}" if mal_link else "")
        ).strip()
    except Exception as e:
        api_log(f"ReverseImage (trace.moe) -> skipped ({type(e).__name__})")
    return ""


async def extract_image_via_transient_ocr(
    raw_bytes: bytes, filename: str, run_source_lookup: bool = True
) -> str:
    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
    fname_lower = filename.lower()
    if fname_lower.endswith(".png"):
        mime_type = "image/png"
    elif fname_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    ocr_prompt = (
        "Analyze this image thoroughly for Yukino Yukinoshita so she can identify its exact source, "
        "read any text, and give her sharp personal critique.\n"
        "Provide your output in 3 concise sections:\n"
        "1. [Identification & Origin]: Identify what type of image this is:\n"
        "   - If Illustration/Anime/Manga: State the likely Franchise/Series, Character name(s), Artist signature/handle (if visible), and Manga Title/Chapter clues.\n"
        "   - If Real-World Photo/Product/Hardware: Identify the exact Brand, Product Line, Model Name/Number (e.g., phone model, espresso machine, scale model kit, audio gear), and distinguishing physical features (camera layout, buttons, logos).\n"
        "   - Visual Scene & Details: Describe the scene, expressions, environment, aesthetic quality, composition, any cats or pandas, and any notable, messy, or amusing details.\n"
        "2. [Verbatim OCR Text]: Transcribe ALL visible text (Thai, Japanese, English) verbatim.\n"
        "   Pay close attention to Thai tone marks (วรรณยุกต์), vowels (สระ), and visually similar consonants "
        "(ก/ภ/ถ, ข/ช, ค/ด/ต, พ/ผ/ฝ/ฟ, ร/ธ), and note if the original text contains any spelling errors.\n"
        "3. [WEB_VERIFY_QUERY]: On the very last line, if the image is a manga page, anime, artwork, or physical product/device whose exact chapter/model/source can be verified via Google Search, output:\n"
        "   `[WEB_VERIFY_QUERY: <best 4-8 word search query using exact Japanese/English quotes, character names, or product model features>]`"
    )

    # Run Reverse Image Search (SauceNAO / trace.moe) concurrently with Vision OCR when source lookup is enabled
    reverse_tasks = []
    if run_source_lookup:
        reverse_tasks = [
            asyncio.create_task(_search_saucenao_async(raw_bytes, filename)),
            asyncio.create_task(_search_tracemoe_async(raw_bytes, mime_type)),
        ]

    vision_result = ""

    # Tier 1: Google AI Studio Gemma 4 Vision (gemma-4-26b-a4b-it / 31b-it — 14.4K RPD)
    gemma_ocr = await call_gemma_ai_studio_vision(
        ocr_prompt, mime_type, b64_img, timeout_sec=25, purpose="Image OCR Tier 1"
    )
    if gemma_ocr:
        vision_result = gemma_ocr

    # Tier 2: Cloudflare Workers AI Vision (@cf/google/gemma-4-26b-a4b-it — 10K Neurons/day)
    if not vision_result:
        cf_ocr = await call_cloudflare_vision_chat(ocr_prompt, mime_type, b64_img, purpose="Image OCR Tier 2")
        if cf_ocr:
            vision_result = cf_ocr

    # Tier 3: Groq LPU Vision (qwen/qwen3.8-27b)
    if not vision_result and GROQ_API_KEY:
        groq_candidates = await discover_active_groq_vision_models()
        groq_host = "api." + "groq.com"
        chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
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
                            api_log(f"Image OCR (Tier 3 Groq) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                            vision_result = content
                            break
                    api_log(f"Image OCR (Tier 3 Groq) -> {g_mod} HTTP {resp.status}; fallback to Tier 4")
            except Exception as e:
                api_log(f"Image OCR (Tier 3 Groq) -> {g_mod} failed ({type(e).__name__}); fallback to Tier 4")

    # Tier 4: Modal T4 GPU (scb10x/typhoon-ocr1.5-3b)
    if not vision_result:
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
                            vision_result = content
        except Exception as e:
            api_log(f"Image OCR (Tier 4 Modal) -> {ACTIVE_VISION_MODEL} failed ({type(e).__name__})")

    out_parts = []
    if reverse_tasks:
        rev_results = await asyncio.gather(*reverse_tasks, return_exceptions=True)
        saucenao_txt = rev_results[0] if isinstance(rev_results[0], str) else ""
        tracemoe_txt = rev_results[1] if isinstance(rev_results[1], str) else ""

        if saucenao_txt:
            out_parts.append(saucenao_txt)
        else:
            # If SauceNAO had no match (or no API key set), try keyless IQDB for anime/illustration matches
            iqdb_txt = await _search_iqdb_keyless_async(raw_bytes, filename)
            if iqdb_txt:
                out_parts.append(iqdb_txt)

        if tracemoe_txt:
            out_parts.append(tracemoe_txt)

    if vision_result:
        out_parts.append(f"=== Vision Analysis & OCR ===\n{vision_result}")

        # If Vision emitted a [WEB_VERIFY_QUERY: ...] tag and reverse search didn't already give a Pixiv match,
        # run a fast web search to verify the exact Manga Title/Chapter or Product Model!
        q_match = re.search(r"\[WEB_VERIFY_QUERY:\s*([^\]]+?)\s*\]", vision_result, re.IGNORECASE)
        if run_source_lookup and q_match:
            verify_q = q_match.group(1).strip()[:120]
            if verify_q and verify_q.lower() not in ("none", "n/a"):
                try:
                    web_verify = await asyncio.to_thread(run_safe_web_search_sync, verify_q, 3)
                    if web_verify:
                        out_parts.append(f"=== Web Verification for Image (`{verify_q}`) ===\n{web_verify}")
                except Exception:
                    pass

    return "\n\n".join(out_parts) if out_parts else "[OCR could not extract text from image]"


async def extract_all_context(message: discord.Message) -> tuple[str, str]:
    from voice_engine import convert_audio_bytes_to_wav_sync, transcribe_wav_bytes_hybrid

    blocks, voice_clip_transcripts = [], []
    has_code_file = False

    # Support replying to a previous message containing an image (e.g., replying 'ขอวาร์ป' or 'รุ่นอะไร')
    target_attachments = list(message.attachments)
    if (
        not target_attachments
        and message.reference
        and isinstance(message.reference.resolved, discord.Message)
    ):
        ref_atts = [
            a for a in message.reference.resolved.attachments
            if a.filename.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if ref_atts:
            target_attachments = ref_atts

    for att in target_attachments:
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
                    f"=== Image Inspection & Source Lookup (`{att.filename}`) ===\n"
                    + await extract_image_via_transient_ocr(raw_bytes, att.filename, run_source_lookup=True)
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