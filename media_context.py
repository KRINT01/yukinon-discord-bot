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
    CF_ACCOUNT_ID,
    CF_API_TOKEN,
    CF_VISION_MODEL,
    SAUCENAO_API_KEY,
    OLLAMA_BASE_URL,
    ACTIVE_VISION_MODEL,
    AUDIO_EXTENSIONS,
    CODE_AND_LOG_EXTENSIONS,
    PLAIN_TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    CUSTOM_EMOJI_REGEX,
    URL_REGEX,
    STACK_TRACE_SIGNATURE_REGEX,
    ollama_lock,
    channel_last_attachment,
    state,
    clean_url,
    api_log,
    get_http_session,
    get_modal_headers,
    prune_stale_attachments,
    release_unused_memory,
)
from persona import TYPHOON_OCR_PROMPT
from search_engine import (
    fetch_webpage_sync,
    run_safe_web_search_sync,
    validate_safe_public_url,
)
from llm import (
    discover_active_groq_vision_models,
    call_gemma_ai_studio_vision,
    call_cloudflare_vision_chat,
    call_gemini_generate_content_self_healing,
)

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024   # 25 MB cap (supports standard Discord PDF/doc uploads safely)
MAX_EMBED_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per embedded image cap for 512MB Render container

TWITTER_STATUS_URL_REGEX = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:x\.com|twitter\.com|fixupx\.com|fxtwitter\.com|vxtwitter\.com|fixvx\.com|twittpr\.com)/([^/\s]+)/status/(\d+)",
    re.IGNORECASE,
)
PIXIV_ILLUST_URL_REGEX = re.compile(
    r"https?://(?:www\.)?(?:pixiv\.net|phixiv\.net|ppxiv\.net)/(?:[a-z]{2}/)?(?:artworks/|i/|member_illust\.php\?[^#\s]*illust_id=)(\d+)",
    re.IGNORECASE,
)
BSKY_POST_URL_REGEX = re.compile(
    r"https?://(?:www\.)?(?:bsky\.app|fxbsky\.app)/profile/([^/\s]+)/post/([a-zA-Z0-9]+)",
    re.IGNORECASE,
)
DIRECT_IMAGE_URL_REGEX = re.compile(
    r"https?://[^\s<>'\"]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s<>'\"]*)?",
    re.IGNORECASE,
)

# Common Japanese manga SFX tokens that should NEVER trigger a web search query
_SFX_TOKEN_REGEX = re.compile(
    r"^(?:タッ|ドシッ|ドッ|ドドン|ピュ+|ピシャ+ン|ズシ+|ズゥ+ン|ズーン|ヴォ+|ブル+|ムチ+|ムク+|"
    r"ぷくっ?|ぷる+|だぷっ?|ずり+|ガタッ|クッ|カッ|ボフッ|ハッ|ハァ|フゥ|ゴゴ+|ザワ+|ドキッ|PAN|[!?…・ーッっ]+)$",
    re.IGNORECASE,
)


def _is_pure_sfx_query(q: str) -> bool:
    """Returns True if a proposed web verification query consists only of comic sound effects (SFX)."""
    if not q:
        return True
    cleaned = re.sub(r"[\s\u3000!?…・ー「」『』（）()]+", " ", q).strip()
    if len(cleaned) < 4:
        return True
    tokens = [t for t in cleaned.split(" ") if t]
    if not tokens:
        return True
    non_sfx = [t for t in tokens if not _SFX_TOKEN_REGEX.match(t)]
    return len(non_sfx) == 0


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


def parse_pdf_bytes(raw_bytes: bytes, max_chars: int = 32000) -> str:
    from pypdf import PdfReader  # Lazy-loaded to save startup RAM

    reader = PdfReader(io.BytesIO(raw_bytes))
    total_pages = len(reader.pages)
    pages_text = []
    collected_chars = 0

    for i, p in enumerate(reader.pages):
        txt = (p.extract_text() or "").strip()
        if txt:
            block = f"--- Page {i+1}/{total_pages} ---\n{txt}"
            pages_text.append(block)
            collected_chars += len(block)
            if collected_chars >= max_chars:
                break

    joined = "\n\n".join(pages_text)[:max_chars]
    if len(joined.strip()) < 60:
        return (
            f"[PDF Notice: This PDF has {total_pages} page(s), but contains almost no selectable text "
            f"(likely a scanned/image-only PDF). Ask the user to send page screenshots or export text if OCR is needed.]"
        )
    return joined


def parse_docx_bytes(raw_bytes: bytes, max_chars: int = 30000) -> str:
    import docx  # Lazy-loaded to save startup RAM

    doc = docx.Document(io.BytesIO(raw_bytes))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append("| " + " | ".join(c.text.replace("\n", " ").strip() for c in row.cells) + " |")
    return "\n\n".join(parts)[:max_chars]


def extract_key_stack_trace_windows(log_text: str, max_chars: int = 28000) -> str:
    if len(log_text) <= max_chars:
        return log_text

    lines = log_text.splitlines()
    total_lines = len(lines)
    keep_indices: set[int] = set()

    for i in range(min(45, total_lines)):
        keep_indices.add(i)
    for i in range(max(0, total_lines - 180), total_lines):
        keep_indices.add(i)

    for idx, line in enumerate(lines):
        if STACK_TRACE_SIGNATURE_REGEX.search(line) or re.search(r"\b(ERROR|FATAL|SEVERE|PANIC)\b", line):
            start_w = max(0, idx - 6)
            end_w = min(total_lines, idx + 28)
            for j in range(start_w, end_w):
                keep_indices.add(j)

    sorted_idx = sorted(keep_indices)
    out_lines = []
    prev_i = None
    for i in sorted_idx:
        if prev_i is not None and i > prev_i + 1:
            skipped = i - prev_i - 1
            out_lines.append(f"... [ omitted {skipped} routine log line(s) ] ...")
        out_lines.append(f"{i+1:5d} | {lines[i]}")
        prev_i = i

    assembled = "\n".join(out_lines)
    if len(assembled) <= max_chars:
        return assembled
    head = assembled[:4500]
    tail = assembled[-(max_chars - 4600):]
    return f"{head}\n\n... [ truncated intermediate trace frames ] ...\n\n{tail}"


def format_code_or_log_attachment(text: str, is_log: bool, max_chars: int = 120000) -> str:
    if is_log or STACK_TRACE_SIGNATURE_REGEX.search(text[:8000]):
        return extract_key_stack_trace_windows(text, max_chars=28000)

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


# ==============================================================================
# EMBEDDED LINKS & SOCIAL MEDIA MEDIA/METADATA RESOLVER (X, PIXIV, BSKY, EMBEDS)
# ==============================================================================
def _normalize_image_dedup_key(url: str) -> str:
    cu = clean_url(url)
    tw_media_m = re.search(r"pbs\.twimg\.com/media/([A-Za-z0-9_-]+)", cu)
    if tw_media_m:
        return f"twimg:{tw_media_m.group(1)}"
    px_img_m = re.search(r"i\.pximg\.net/.+?/(\d+_p\d+)", cu)
    if px_img_m:
        return f"pximg:{px_img_m.group(1)}"
    parsed = urlparse(cu)
    return f"{parsed.netloc}{parsed.path}".lower()


def _guess_image_filename_from_url(url: str, fallback_prefix: str = "embed_img") -> str:
    try:
        path = urlparse(url).path
        base = path.rsplit("/", 1)[-1].split("?")[0]
        if base and any(base.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            return base[:80]
    except Exception:
        pass
    return f"{fallback_prefix}.jpg"


async def _download_remote_image_safe(url: str, referer: str = "") -> tuple[bytes, str] | None:
    try:
        safe_url = await asyncio.to_thread(validate_safe_public_url, url)
    except Exception as e:
        api_log(f"Embed Image SSRF/URL skip ({url[:60]}): {e}")
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    lower_u = safe_url.lower()
    if referer:
        headers["Referer"] = referer
    elif "pximg.net" in lower_u or "pixiv.net" in lower_u:
        headers["Referer"] = f"{HTTPS}www.pixiv.net/"

    try:
        session = get_http_session()
        async with session.get(
            safe_url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)
        ) as resp:
            if resp.status != 200:
                api_log(f"Embed Image HTTP {resp.status} -> {safe_url[:70]}")
                return None
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and not (ctype.startswith("image/") or "octet-stream" in ctype):
                return None
            c_len = int(resp.headers.get("Content-Length") or 0)
            if c_len > MAX_EMBED_IMAGE_BYTES:
                return None

            buf = bytearray()
            async for chunk in resp.content.iter_chunked(65536):
                buf.extend(chunk)
                if len(buf) > MAX_EMBED_IMAGE_BYTES:
                    return None
            if len(buf) < 512:
                return None
            fname = _guess_image_filename_from_url(safe_url)
            return bytes(buf), fname
    except Exception as e:
        api_log(f"Embed Image download note ({safe_url[:60]}): {type(e).__name__}")
        return None


async def _fetch_x_twitter_post_info(screen_name: str, tweet_id: str) -> tuple[str, list[str]]:
    t0 = time.monotonic()
    session = get_http_session()
    user_seg = screen_name or "i"
    endpoints = (
        f"{HTTPS}api.fxtwitter.com/{user_seg}/status/{tweet_id}",
        f"{HTTPS}api.fxtwitter.com/status/{tweet_id}",
        f"{HTTPS}api.vxtwitter.com/{user_seg}/status/{tweet_id}",
        f"{HTTPS}api.vxtwitter.com/status/{tweet_id}",
    )
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    for ep in endpoints:
        try:
            async with session.get(
                ep,
                headers=req_headers,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json(content_type=None)
                if not isinstance(data, dict):
                    continue

                tweet = data.get("tweet")
                if isinstance(tweet, dict):
                    author = tweet.get("author") or {}
                    a_name = author.get("name") or ""
                    a_handle = author.get("screen_name") or screen_name or ""
                    t_text = (tweet.get("text") or "").strip()
                    t_url = tweet.get("url") or f"{HTTPS}x.com/{a_handle or 'i'}/status/{tweet_id}"
                    created = tweet.get("created_at") or ""
                    likes = tweet.get("likes")
                    retweets = tweet.get("retweets")

                    photo_urls = []
                    media = tweet.get("media") or {}
                    for ph in media.get("photos") or []:
                        if isinstance(ph, dict) and ph.get("url"):
                            photo_urls.append(ph["url"])
                    if not photo_urls:
                        for m_item in media.get("all") or []:
                            if isinstance(m_item, dict):
                                u = m_item.get("url") or m_item.get("thumbnail_url")
                                if u and m_item.get("type") in ("photo", "image", "gif", "video"):
                                    photo_urls.append(u)

                    lines = [
                        f"Platform: X (Twitter) | Post URL: {t_url}",
                        f"Author / Artist: {a_name} (@{a_handle})" if a_handle else f"Author / Artist: {a_name}",
                        f"Post Caption / Title Text: {t_text or '(No text caption)'}",
                    ]
                    if created:
                        lines.append(f"Posted At: {created}")
                    if likes is not None or retweets is not None:
                        lines.append(f"Engagement: {likes or 0} Likes | {retweets or 0} Reposts")

                    quote = tweet.get("quote")
                    if isinstance(quote, dict):
                        q_auth = quote.get("author") or {}
                        q_handle = q_auth.get("screen_name") or ""
                        q_text = (quote.get("text") or "").strip()
                        if q_text:
                            lines.append(f"Quoted Post (@{q_handle}):\n{q_text}")
                        if not photo_urls:
                            for qph in (quote.get("media") or {}).get("photos") or []:
                                if isinstance(qph, dict) and qph.get("url"):
                                    photo_urls.append(qph["url"])

                    api_log(
                        f"Embed Link (FxTwitter) -> tweet={tweet_id}, caption_len={len(t_text)}, photos={len(photo_urls)} ({time.monotonic() - t0:.2f}s)"
                    )
                    return "\n".join(lines), photo_urls

                if "text" in data or "mediaURLs" in data:
                    u_name = data.get("user_name") or ""
                    u_screen = data.get("user_screen_name") or screen_name or ""
                    t_text = (data.get("text") or "").strip()
                    media_urls = [
                        u for u in (data.get("mediaURLs") or [])
                        if isinstance(u, str) and any(ext in u.lower() for ext in (".jpg", ".jpeg", ".png", ".webp", "pbs.twimg.com/media"))
                    ]
                    lines = [
                        f"Platform: X (Twitter) | Post URL: {HTTPS}x.com/{u_screen or 'i'}/status/{tweet_id}",
                        f"Author / Artist: {u_name} (@{u_screen})" if u_screen else f"Author / Artist: {u_name}",
                        f"Post Caption / Title Text: {t_text or '(No text caption)'}",
                    ]
                    api_log(
                        f"Embed Link (vxTwitter) -> tweet={tweet_id}, caption_len={len(t_text)}, photos={len(media_urls)} ({time.monotonic() - t0:.2f}s)"
                    )
                    return "\n".join(lines), media_urls
        except Exception:
            continue
    api_log(f"Embed Link (X/Twitter API) -> fallback to Discord native embed for tweet={tweet_id}")
    return "", []


async def _fetch_pixiv_artwork_info(illust_id: str) -> tuple[str, list[str]]:
    t0 = time.monotonic()
    session = get_http_session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": f"{HTTPS}www.pixiv.net/artworks/{illust_id}",
        "Accept": "application/json",
    }
    meta_url = f"{HTTPS}www.pixiv.net/ajax/illust/{illust_id}?lang=en"
    pages_url = f"{HTTPS}www.pixiv.net/ajax/illust/{illust_id}/pages?lang=en"

    try:
        async with session.get(meta_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return "", []
            payload = await resp.json(content_type=None)
            if payload.get("error"):
                return "", []
            body = payload.get("body") or {}

        title = (body.get("title") or "").strip()
        user_name = (body.get("userName") or "").strip()
        user_id = str(body.get("userId") or "").strip()
        page_count = int(body.get("pageCount") or 1)
        desc_raw = re.sub(r"<[^>]+>", " ", body.get("description") or "").strip()
        desc_clean = re.sub(r"\s+", " ", desc_raw)[:600]

        tags_list = []
        for t_obj in ((body.get("tags") or {}).get("tags") or [])[:15]:
            if isinstance(t_obj, dict):
                t_jp = t_obj.get("tag") or ""
                t_en = ((t_obj.get("translation") or {}).get("en")) or ""
                tags_list.append(f"{t_jp} ({t_en})" if t_en else t_jp)

        image_urls = []
        urls_obj = body.get("urls") or {}
        primary_u = urls_obj.get("regular") or urls_obj.get("small") or urls_obj.get("original")
        if primary_u:
            image_urls.append(primary_u)

        if page_count > 1:
            try:
                async with session.get(pages_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as p_resp:
                    if p_resp.status == 200:
                        p_data = await p_resp.json(content_type=None)
                        p_items = p_data.get("body") or []
                        extracted_pages = []
                        for item in p_items[:6]:
                            u_map = (item.get("urls") or {}) if isinstance(item, dict) else {}
                            pu = u_map.get("regular") or u_map.get("small")
                            if pu:
                                extracted_pages.append(pu)
                        if extracted_pages:
                            image_urls = extracted_pages
            except Exception:
                pass

        lines = [
            f"Platform: Pixiv | Artwork URL: {HTTPS}www.pixiv.net/artworks/{illust_id}",
            f"Artwork Title / Caption: {title}",
            f"Artist / Illustrator: {user_name} (Pixiv User ID: {user_id})",
            f"Total Pages: {page_count}",
        ]
        if tags_list:
            lines.append(f"Tags: {', '.join(tags_list)}")
        if desc_clean:
            lines.append(f"Artist Caption: {desc_clean}")

        api_log(
            f"Embed Link (Pixiv AJAX) -> illust={illust_id}, pages={len(image_urls)} ({time.monotonic() - t0:.2f}s)"
        )
        return "\n".join(lines), image_urls
    except Exception as e:
        api_log(f"Embed Link (Pixiv AJAX) -> skipped ({type(e).__name__})")
        return "", []


async def _fetch_bsky_post_info(handle: str, rkey: str) -> tuple[str, list[str]]:
    session = get_http_session()
    try:
        resolve_url = f"{HTTPS}public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle?handle={quote_plus(handle)}"
        async with session.get(resolve_url, timeout=aiohttp.ClientTimeout(total=6)) as r1:
            if r1.status != 200:
                return "", []
            did = (await r1.json()).get("did")
            if not did:
                return "", []

        uri = f"at://{did}/app.bsky.feed.post/{rkey}"
        thread_url = f"{HTTPS}public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri={quote_plus(uri)}&depth=0"
        async with session.get(thread_url, timeout=aiohttp.ClientTimeout(total=6)) as r2:
            if r2.status != 200:
                return "", []
            post = ((await r2.json()).get("thread") or {}).get("post") or {}
            author = post.get("author") or {}
            record = post.get("record") or {}
            embed = post.get("embed") or {}
            img_urls = []
            for img_obj in embed.get("images") or []:
                if isinstance(img_obj, dict) and img_obj.get("fullsize"):
                    img_urls.append(img_obj["fullsize"])

            lines = [
                f"Platform: Bluesky | Post URL: {HTTPS}bsky.app/profile/{handle}/post/{rkey}",
                f"Author: {author.get('displayName') or handle} (@{handle})",
                f"Post Caption / Content:\n{record.get('text') or '(No text caption)'}",
            ]
            return "\n".join(lines), img_urls
    except Exception:
        return "", []


def _extract_native_discord_embeds_data(embeds: list[discord.Embed]) -> tuple[list[str], list[str]]:
    meta_blocks = []
    image_urls = []
    seen_dedup_keys = set()

    for idx, emb in enumerate(embeds, 1):
        parts = []
        if emb.provider and emb.provider.name:
            parts.append(f"Provider: {emb.provider.name}")
        if emb.author and emb.author.name:
            a_str = f"Author / Artist: {emb.author.name}"
            if emb.author.url:
                a_str += f" ({emb.author.url})"
            parts.append(a_str)
        if emb.title:
            parts.append(f"Embed Title: {emb.title}")
        if emb.url:
            parts.append(f"URL: {emb.url}")
        if emb.description:
            parts.append(f"Post Caption / Description:\n{emb.description[:3000]}")
        for fld in emb.fields:
            if fld.name or fld.value:
                parts.append(f"- {fld.name}: {fld.value}")
        if emb.footer and emb.footer.text:
            parts.append(f"Footer: {emb.footer.text}")

        for img_candidate in (
            getattr(emb.image, "url", None),
            getattr(emb.image, "proxy_url", None),
            getattr(emb.thumbnail, "url", None),
            getattr(emb.thumbnail, "proxy_url", None),
        ):
            if img_candidate and isinstance(img_candidate, str):
                clean_img = clean_url(img_candidate)
                dkey = _normalize_image_dedup_key(clean_img)
                if clean_img and dkey not in seen_dedup_keys:
                    seen_dedup_keys.add(dkey)
                    image_urls.append(clean_img)
                    break

        if parts:
            meta_blocks.append(f"[Discord Embed #{idx}]\n" + "\n".join(parts))

    return meta_blocks, image_urls


async def collect_embedded_images_and_metadata(
    message: discord.Message,
    ref_msg: discord.Message | None = None,
    max_images: int = 4,
) -> tuple[list[tuple[bytes, str, str]], list[str], set[str]]:
    downloaded_images: list[tuple[bytes, str, str]] = []
    metadata_sections: list[str] = []
    handled_urls: set[str] = set()
    candidate_image_urls: list[tuple[str, str]] = []
    seen_dedup_keys: set[str] = set()

    def _queue_img(u: str, referer: str = ""):
        cu = clean_url(u)
        dkey = _normalize_image_dedup_key(cu)
        if cu and dkey not in seen_dedup_keys:
            seen_dedup_keys.add(dkey)
            candidate_image_urls.append((cu, referer))

    target_msgs = [message]
    if ref_msg is not None:
        target_msgs.append(ref_msg)

    urls_in_curr = URL_REGEX.findall(message.content or "")
    if urls_in_curr and not message.embeds:
        try:
            await asyncio.sleep(1.0)
            refetched = await message.channel.fetch_message(message.id)
            target_msgs[0] = refetched
        except Exception:
            pass

    all_urls: list[str] = []
    for m in target_msgs:
        for u in URL_REGEX.findall(m.content or ""):
            cu = clean_url(u)
            if cu and cu not in all_urls:
                all_urls.append(cu)

    for url in all_urls[:4]:
        tw_m = TWITTER_STATUS_URL_REGEX.search(url)
        if tw_m:
            handled_urls.add(url)
            screen_name, tweet_id = tw_m.group(1), tw_m.group(2)
            tw_meta, tw_photos = await _fetch_x_twitter_post_info(screen_name, tweet_id)
            if tw_meta:
                metadata_sections.append(f"=== Embedded X (Twitter) Post Metadata & Caption ===\n{tw_meta}")
            for p_url in tw_photos:
                _queue_img(p_url)
            continue

        px_m = PIXIV_ILLUST_URL_REGEX.search(url)
        if px_m:
            handled_urls.add(url)
            illust_id = px_m.group(1)
            px_meta, px_photos = await _fetch_pixiv_artwork_info(illust_id)
            if px_meta:
                metadata_sections.append(f"=== Embedded Pixiv Artwork Metadata & Caption ===\n{px_meta}")
            for p_url in px_photos:
                _queue_img(p_url, referer=f"{HTTPS}www.pixiv.net/artworks/{illust_id}")
            continue

        bsky_m = BSKY_POST_URL_REGEX.search(url)
        if bsky_m:
            handled_urls.add(url)
            b_meta, b_photos = await _fetch_bsky_post_info(bsky_m.group(1), bsky_m.group(2))
            if b_meta:
                metadata_sections.append(f"=== Embedded Bluesky Post Metadata & Caption ===\n{b_meta}")
            for p_url in b_photos:
                _queue_img(p_url)
            continue

        if DIRECT_IMAGE_URL_REGEX.match(url):
            handled_urls.add(url)
            _queue_img(url)

    for m in target_msgs:
        if not m.embeds:
            continue
        non_yt_embeds = [
            e for e in m.embeds
            if not (e.url and extract_youtube_id(e.url))
        ]
        if not non_yt_embeds:
            continue
        emb_meta_list, emb_img_urls = _extract_native_discord_embeds_data(non_yt_embeds)
        for idx, e_meta in enumerate(emb_meta_list):
            if not metadata_sections or idx > 0:
                metadata_sections.append(f"=== Discord Link Embed Metadata & Caption ===\n{e_meta}")
        for e_img in emb_img_urls:
            _queue_img(e_img)

    if candidate_image_urls:
        tasks = [
            _download_remote_image_safe(img_u, referer=ref)
            for img_u, ref in candidate_image_urls[:max_images]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (img_u, _), res in zip(candidate_image_urls[:max_images], results):
            if isinstance(res, tuple) and len(res) == 2:
                raw_b, fname = res
                downloaded_images.append((raw_b, fname, img_u))

    return downloaded_images, metadata_sections, handled_urls


def fetch_youtube_context_sync(
    video_id: str, discord_embed_info: dict | None = None, max_chars: int = 20000
) -> str:
    t0 = time.monotonic()
    watch_url = clean_url(f"{HTTPS}www.youtube.com/watch?v={video_id}")
    title = (discord_embed_info or {}).get("title", "").strip()
    author = (discord_embed_info or {}).get("author", "").strip()
    description = (discord_embed_info or {}).get("description", "").strip()
    transcript_text = ""

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

    search_enrichment = ""
    if len(transcript_text) < 120 and title:
        clean_title_for_search = re.sub(
            r"[|_—–-]\s*(?:Release Date Trailer|Official Trailer|Trailer|PV|Teaser).*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
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
            saucenao_url, data=form, timeout=aiohttp.ClientTimeout(total=6)
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
    t0 = time.monotonic()
    iqdb_url = f"{HTTPS}iqdb.org/"
    try:
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
            timeout=aiohttp.ClientTimeout(total=6),
        ) as resp:
            if resp.status != 200:
                return ""
            html_text = await resp.text()

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
    t0 = time.monotonic()
    trace_url = f"{HTTPS}api.trace.moe/search?anilistInfo&cutBorders"
    try:
        session = get_http_session()
        async with session.post(
            trace_url,
            data=raw_bytes,
            headers={"Content-Type": mime_type},
            timeout=aiohttp.ClientTimeout(total=6),
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


def _compress_image_for_vision_ocr_sync(raw_bytes: bytes, max_dim: int = 1280, quality: int = 90) -> tuple[str, str, bool]:
    from PIL import Image, ImageStat
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    hsv_stat = ImageStat.Stat(img.convert("HSV"))
    mean_saturation = hsv_stat.mean[1] if len(hsv_stat.mean) > 1 else 0.0
    is_color_image = (mean_saturation >= 14.0)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    img.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg", is_color_image


async def extract_multi_images_batch_ocr(
    images_list: list[tuple[bytes, str, str]],
    post_meta_hint: str = "",
) -> str:
    """
    Inspects 2-4 sequential comic/illustration pages together in a SINGLE multimodal batch call
    so the series title, exact character cast, and cross-page story stay 100% consistent without
    hallucinating new characters on later pages:
    - Tier 1: Gemini 3 Flash (`thinkingBudget: 0`, `896px` @ `q=82`, `35s` timeout)
    - Tier 2: Cloudflare Workers AI `@cf/google/gemma-4-26b-a4b-it` (multi-image chat completions)
    """
    if len(images_list) < 2:
        return ""

    batch_prompt = (
        f"You are analyzing a sequential multi-image post/comic (Pages 1 to {len(images_list)}) for Yukino Yukinoshita.\n"
        + (f"Post Metadata / Caption Context:\n{post_meta_hint[:1000]}\n\n" if post_meta_hint else "")
        + "Examine ALL pages together in sequence so you maintain strict character consistency across every page:\n"
        "1. [Series / Franchise & Character Identification]:\n"
        "   - State the Anime, Game, or Manga franchise and Character names ONLY if clearly supported by visual cues "
        "(e.g., halos -> Blue Archive) or the post caption (e.g., ノアコ = Noa & Ako).\n"
        "   - CRITICAL ANTI-HALLUCINATION RULE: Keep the SAME characters consistent across pages! "
        "NEVER invent or guess extra characters from the same franchise (such as Arona, Ui, or Shiroko) unless they are clearly and unmistakably drawn in that panel.\n"
        "2. [Page-by-Page Story & Panel-by-Panel Dialogue/SFX (Right-to-Left, Top-to-Bottom)]:\n"
        "   - For each page (Page 1, Page 2, ...), summarize the visual events and list all dialogue, thoughts, "
        "and key SFX in Japanese reading order tagged with the speaker."
    )

    encoded_images: list[tuple[str, str, str]] = []
    for raw_b, fname, _ in images_list[:4]:
        try:
            b64_str, mime_type, _ = await asyncio.to_thread(_compress_image_for_vision_ocr_sync, raw_b, 896, 82)
            encoded_images.append((b64_str, mime_type, fname))
        except Exception:
            continue

    if len(encoded_images) < 2:
        return ""

    # Tier 1: Gemini 3 Flash Multi-Image Batch (`thinkingBudget: 0` for fast ~6-10s response)
    if GEMINI_API_KEY:
        parts = [{"text": batch_prompt}]
        for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
            parts.append({"text": f"=== Image {idx}/{len(encoded_images)} (`{fname}`) ==="})
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(
                payload, timeout_sec=35, purpose=f"MultiImageOCR({len(encoded_images)}p)"
            )
            if data:
                resp_parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                out_txt = "\n".join(p.get("text", "") for p in resp_parts if p.get("text")).strip()
                out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
                if out_txt:
                    return f"=== Unified Multi-Image Visual & OCR Analysis ({len(encoded_images)} Images) ===\n{out_txt}"
        except Exception as e:
            api_log(f"MultiImageOCR Tier 1 Gemini -> fallback to Tier 2 CF ({type(e).__name__})")

    # Tier 2: Cloudflare Workers AI Gemma 4 Multi-Image Batch
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        cf_host = "api." + "cloudflare.com"
        cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        user_content: list[dict] = [{"type": "text", "text": batch_prompt}]
        for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
            user_content.append({"type": "text", "text": f"=== Image {idx}/{len(encoded_images)} (`{fname}`) ==="})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}})

        cf_payload = {
            "model": CF_VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond directly and concisely with the requested analysis. Do NOT output internal thinking steps.",
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.monotonic()
        try:
            session = get_http_session()
            async with session.post(
                cf_chat_url, headers=headers, json=cf_payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices") or []
                    if choices:
                        content = (choices[0].get("message") or {}).get("content") or ""
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                        if content:
                            api_log(f"MultiImageOCR (Tier 2 CF) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                            return f"=== Unified Multi-Image Visual & OCR Analysis ({len(encoded_images)} Images) ===\n{content}"
                api_log(f"MultiImageOCR (Tier 2 CF) -> HTTP {resp.status} ({elapsed:.2f}s)")
        except Exception as e:
            api_log(f"MultiImageOCR (Tier 2 CF) -> fallback to per-image OCR ({type(e).__name__})")

    return ""


async def extract_image_via_transient_ocr(
    raw_bytes: bytes,
    filename: str,
    run_source_lookup: bool = True,
    post_meta_hint: str = "",
    page_label: str = "",
) -> str:
    try:
        b64_img, mime_type, is_color_img = await asyncio.to_thread(
            _compress_image_for_vision_ocr_sync, raw_bytes
        )
    except Exception:
        b64_img = base64.b64encode(raw_bytes).decode("utf-8")
        mime_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        is_color_img = True

    context_header = ""
    if page_label or post_meta_hint:
        context_header = (
            f"Context Note ({page_label or 'Single Image'}):\n"
            + (f"{post_meta_hint[:600]}\n" if post_meta_hint else "")
            + "Keep character identities consistent with the post caption/context above.\n\n"
        )

    ocr_prompt = (
        f"{context_header}"
        "Analyze this image thoroughly so Yukino Yukinoshita can understand the exact visual story, "
        "series/character identity, speaker attribution, and source.\n"
        "Provide your output in 3 concise, well-structured sections:\n"
        "1. [Identification & Panel-by-Panel Visual Context]:\n"
        "   - If Comic / Manga / Fanart: Identify the Anime/Game/Manga Series and Character names ONLY if clearly supported by "
        "unmistakable visual features or the post caption above. NEVER guess or invent other characters from the same franchise "
        "when a character is distorted or ambiguous—describe them by appearance or as the same character from the caption instead! "
        "Describe what is happening visually in each panel in traditional Japanese reading order (Top-to-Bottom, Right-to-Left).\n"
        "   - If Code / Terminal / Stack Trace / UI Screenshot: Describe the exact error message, HTTP status, UI component, or terminal output shown.\n"
        "   - If Real-World Photo/Product/Hardware: Identify the Brand, Product Line, Model Name/Number, and distinguishing physical features.\n"
        "2. [Ordered Dialogue & Verbatim OCR Text]:\n"
        "   - For comics/manga: List the dialogue and SFX in reading order (Right-to-Left, Top-to-Bottom) tagged with the speaker.\n"
        "   - Transcribe ALL visible text (Thai, Japanese, English, code, error codes) verbatim, paying close attention to Thai tone marks/vowels.\n"
        "3. [WEB_VERIFY_QUERY]: On the very last line, ONLY if the image contains actual spoken dialogue or product model text (NEVER pure comic sound effects like タッ or ドシッ!), output:\n"
        "   `[WEB_VERIFY_QUERY: <exact verbatim dialogue quote or product model — NEVER pure SFX and NEVER a guessed title>]`"
    )

    reverse_tasks = []
    if run_source_lookup:
        reverse_tasks.append(asyncio.create_task(_search_saucenao_async(raw_bytes, filename)))
        if is_color_img:
            reverse_tasks.append(asyncio.create_task(_search_tracemoe_async(raw_bytes, mime_type)))

    vision_result = ""

    # Tier 1: Cloudflare Workers AI Vision
    cf_ocr = await call_cloudflare_vision_chat(ocr_prompt, mime_type, b64_img, purpose="Image OCR Tier 1 CF")
    if cf_ocr:
        vision_result = cf_ocr

    # Tier 2: Groq LPU Vision
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
                    "max_tokens": 3072,
                }
                async with session.post(
                    chat_url, headers=headers, json=groq_vision_payload, timeout=aiohttp.ClientTimeout(total=25)
                ) as resp:
                    elapsed = time.monotonic() - t0
                    if resp.status == 200:
                        state.active_groq_vision_model = g_mod
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                        if content:
                            api_log(f"Image OCR (Tier 2 Groq) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                            vision_result = content
                            break
                    api_log(f"Image OCR (Tier 2 Groq) -> {g_mod} HTTP {resp.status}; fallback to Tier 3")
            except Exception as e:
                api_log(f"Image OCR (Tier 2 Groq) -> {g_mod} failed ({type(e).__name__}); fallback to Tier 3")

    # Tier 3: Google AI Studio Gemma 4 Vision
    if not vision_result:
        gemma_ocr = await call_gemma_ai_studio_vision(
            ocr_prompt, mime_type, b64_img, timeout_sec=12, purpose="Image OCR Tier 3"
        )
        if gemma_ocr:
            vision_result = gemma_ocr

    # Tier 4: Modal T4 GPU
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
        saucenao_txt = rev_results[0] if (len(rev_results) > 0 and isinstance(rev_results[0], str)) else ""
        tracemoe_txt = rev_results[1] if (len(rev_results) > 1 and isinstance(rev_results[1], str)) else ""

        if saucenao_txt:
            out_parts.append(saucenao_txt)
        elif not post_meta_hint:
            iqdb_txt = await _search_iqdb_keyless_async(raw_bytes, filename)
            if iqdb_txt:
                out_parts.append(iqdb_txt)

        if tracemoe_txt:
            out_parts.append(tracemoe_txt)

    if vision_result:
        clean_vis = re.sub(r"\[WEB_VERIFY_QUERY:\s*[^\]]+?\]", "", vision_result, flags=re.IGNORECASE).strip()
        out_parts.append(f"=== Vision Analysis & OCR ===\n{clean_vis}")

        q_match = re.search(r"\[WEB_VERIFY_QUERY:\s*([^\]]+?)\s*\]", vision_result, re.IGNORECASE)
        if run_source_lookup and q_match:
            verify_q = q_match.group(1).strip()[:120]
            if (
                verify_q
                and verify_q.lower() not in ("none", "n/a")
                and not _is_pure_sfx_query(verify_q)
            ):
                try:
                    web_verify = await asyncio.to_thread(run_safe_web_search_sync, verify_q, 3)
                    if web_verify:
                        out_parts.append(f"=== Web Verification for Image (`{verify_q}`) ===\n{web_verify}")
                except Exception:
                    pass

    return "\n\n".join(out_parts) if out_parts else "[OCR could not extract text from image]"


async def resolve_referenced_message(message: discord.Message) -> discord.Message | None:
    if not message.reference or not message.reference.message_id:
        return None
    if isinstance(message.reference.resolved, discord.Message):
        return message.reference.resolved
    try:
        return await message.channel.fetch_message(message.reference.message_id)
    except Exception:
        return None


async def extract_all_context(message: discord.Message) -> tuple[str, str]:
    from voice_engine import convert_audio_bytes_to_wav_sync, transcribe_wav_bytes_hybrid

    blocks, voice_clip_transcripts = [], []
    has_code_file = False
    parsed_heavy_doc = False
    processed_image_count = 0

    ref_msg = await resolve_referenced_message(message)

    target_attachments = list(message.attachments)
    if not target_attachments and ref_msg and ref_msg.attachments:
        target_attachments = list(ref_msg.attachments)

    image_att_batch: list[tuple[bytes, str, str]] = []

    for att in target_attachments:
        if att.size > MAX_ATTACHMENT_BYTES:
            size_mb = round(att.size / (1024 * 1024), 2)
            blocks.append(
                f"[Attachment Size Notice: File `{att.filename}` is {size_mb} MB, which exceeds the 25 MB container limit. "
                f"Inform the user politely in Yukino's persona that the file is too large ({size_mb} MB > 25 MB) and ask them to split or compress it.]"
            )
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
                decoded_text = raw_bytes.decode("utf-8", errors="replace")
                if fname.endswith(CODE_AND_LOG_EXTENSIONS) or STACK_TRACE_SIGNATURE_REGEX.search(decoded_text[:4000]):
                    has_code_file = True
                formatted_code = format_code_or_log_attachment(decoded_text, is_log=is_log)
                label = "Log / Stack Trace" if (is_log or STACK_TRACE_SIGNATURE_REGEX.search(decoded_text[:4000])) else "Source File"
                blocks.append(
                    f"=== Attached {label}: `{att.filename}` ===\n"
                    f"```\n{formatted_code}\n```"
                )
            elif fname.endswith(".xlsx"):
                parsed_heavy_doc = True
                blocks.append(
                    f"=== Attached Excel: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_xlsx_bytes_to_markdown, raw_bytes)
                )
            elif fname.endswith(".pdf"):
                parsed_heavy_doc = True
                blocks.append(
                    f"=== Attached PDF Document: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_pdf_bytes, raw_bytes)
                )
            elif fname.endswith(".docx"):
                parsed_heavy_doc = True
                blocks.append(
                    f"=== Attached Word Document: `{att.filename}` ===\n"
                    + await asyncio.to_thread(parse_docx_bytes, raw_bytes)
                )
            elif fname.endswith(IMAGE_EXTENSIONS) and len(image_att_batch) < 4:
                image_att_batch.append((raw_bytes, att.filename, ""))
            raw_bytes = None
        except Exception as e:
            blocks.append(f"[Error reading `{att.filename}`: {e}]")

    if image_att_batch:
        if len(image_att_batch) >= 2:
            batch_vis = await extract_multi_images_batch_ocr(image_att_batch, post_meta_hint=message.content or "")
            if batch_vis:
                processed_image_count += len(image_att_batch)
                if STACK_TRACE_SIGNATURE_REGEX.search(batch_vis):
                    has_code_file = True
                blocks.append(batch_vis)
        if processed_image_count == 0:
            total_att_imgs = len(image_att_batch)
            for idx, (img_b, img_fn, _) in enumerate(image_att_batch, 1):
                processed_image_count += 1
                ocr_out = await extract_image_via_transient_ocr(
                    img_b,
                    img_fn,
                    run_source_lookup=(idx == 1),
                    post_meta_hint=message.content or "",
                    page_label=f"Page {idx} of {total_att_imgs}" if total_att_imgs > 1 else "",
                )
                if STACK_TRACE_SIGNATURE_REGEX.search(ocr_out):
                    has_code_file = True
                blocks.append(f"=== Image #{idx} Inspection & Source Lookup (`{img_fn}`) ===\n{ocr_out}")
        image_att_batch.clear()

    embed_images, embed_meta_sections, handled_social_urls = await collect_embedded_images_and_metadata(
        message, ref_msg=ref_msg, max_images=max(1, 4 - processed_image_count)
    )
    for meta_sec in embed_meta_sections:
        blocks.append(meta_sec)

    if embed_images and processed_image_count < 4:
        remaining_slots = 4 - processed_image_count
        target_embed_imgs = embed_images[:remaining_slots]
        meta_hint = "\n".join(embed_meta_sections)
        has_known_post_meta = bool(meta_hint.strip())
        batch_done = False

        if len(target_embed_imgs) >= 2:
            batch_vis = await extract_multi_images_batch_ocr(target_embed_imgs, post_meta_hint=meta_hint)
            if batch_vis:
                processed_image_count += len(target_embed_imgs)
                if STACK_TRACE_SIGNATURE_REGEX.search(batch_vis):
                    has_code_file = True
                blocks.append(batch_vis)
                batch_done = True

        if not batch_done:
            total_emb_imgs = len(target_embed_imgs)
            for idx, (emb_bytes, emb_fname, emb_url) in enumerate(target_embed_imgs, 1):
                if processed_image_count >= 4:
                    break
                processed_image_count += 1
                try:
                    # Skip redundant reverse-image lookups when FxTwitter/Pixiv already provided the exact source post
                    do_lookup = (not has_known_post_meta) and (idx == 1)
                    ocr_out = await extract_image_via_transient_ocr(
                        emb_bytes,
                        emb_fname,
                        run_source_lookup=do_lookup,
                        post_meta_hint=meta_hint,
                        page_label=f"Page {idx} of {total_emb_imgs}" if total_emb_imgs > 1 else "",
                    )
                    if STACK_TRACE_SIGNATURE_REGEX.search(ocr_out):
                        has_code_file = True
                    blocks.append(
                        f"=== Embedded Image #{idx} Inspection & Source Lookup (`{emb_fname}` | `{emb_url}`) ===\n{ocr_out}"
                    )
                except Exception as e:
                    blocks.append(f"[Error inspecting embedded image `{emb_fname}`: {e}]")

    if parsed_heavy_doc or embed_images:
        embed_images.clear()
        release_unused_memory()

    target_stickers = list(getattr(message, "stickers", []) or [])
    if not target_stickers and ref_msg:
        target_stickers = list(getattr(ref_msg, "stickers", []) or [])

    for stk in target_stickers:
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
                            stk_vis = await extract_image_via_transient_ocr(
                                stk_bytes, f"{stk_name}.png", run_source_lookup=False
                            )
                            if stk_vis and "[OCR could not" not in stk_vis:
                                stk_desc += f"\nVisual/Sticker Details:\n{stk_vis}"
            except Exception as e:
                print(f"Sticker vision note ({stk_name}): {e}")
        blocks.append(f"=== Discord Sticker ===\n{stk_desc}")

    custom_emojis = CUSTOM_EMOJI_REGEX.findall(message.content or "")
    if custom_emojis:
        emote_names = [f":{name}:" for name, _ in custom_emojis]
        blocks.append(
            f"=== Custom Discord Emojis in User Message ===\n"
            f"User included custom emoji(s): {', '.join(emote_names)}"
        )

    url_source_msg = message
    urls_in_msg = URL_REGEX.findall(message.content or "")[:2]
    if not urls_in_msg and ref_msg and ref_msg.content:
        urls_in_msg = URL_REGEX.findall(ref_msg.content)[:2]
        if urls_in_msg:
            url_source_msg = ref_msg

    if urls_in_msg and not url_source_msg.embeds:
        try:
            await asyncio.sleep(1.0)
            url_source_msg = await message.channel.fetch_message(url_source_msg.id)
        except Exception:
            pass

    for url in urls_in_msg:
        cu = clean_url(url)
        if cu in handled_social_urls:
            continue
        yt_id = extract_youtube_id(cu)
        if yt_id:
            embed_info = {}
            for emb in url_source_msg.embeds:
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
                    blocks.append(f"=== YouTube Video Information (`{cu}`) ===\n{yt_context}")
            except Exception as e:
                print(f"YouTube context error ({yt_id}): {e}")
        else:
            has_rich_embed_already = bool(embed_meta_sections and embed_images)
            if has_rich_embed_already:
                continue
            try:
                blocks.append(
                    f'<untrusted_web_search_results url="{cu}">\n'
                    + await asyncio.to_thread(fetch_webpage_sync, cu)
                    + "\n</untrusted_web_search_results>"
                )
            except Exception as e:
                blocks.append(f"[Web extraction error: {e}]")

    if ref_msg and ref_msg.content and ref_msg.content.strip():
        ref_txt = re.sub(r"<@!?\d+>", "", ref_msg.content).strip()
        is_old_image_cmd = bool(
            (ref_msg.attachments or ref_msg.embeds)
            and re.search(r"(^!|แปลมังงะ|แปลรูป|แปลภาพ|แปลหน้า|\btranslate\b)", ref_txt, re.IGNORECASE)
        )
        if ref_txt and not is_old_image_cmd:
            ref_author = getattr(ref_msg.author, "display_name", "User")
            blocks.append(
                f"=== Replied-To Message from {ref_author} ===\n{ref_txt[:3000]}"
            )

    joined_context = "\n\n".join(blocks)[:125000]
    prune_stale_attachments()
    if joined_context and message.channel:
        att_names = (
            [a.filename for a in target_attachments]
            if target_attachments
            else ([f[1] for f in embed_images] or ["embed_context"])
        )
        channel_last_attachment[message.channel.id].append({
            "filenames": att_names,
            "content": joined_context[:50000],
            "is_code": has_code_file,
            "timestamp": time.time(),
        })
    return joined_context, " ".join(voice_clip_transcripts).strip()