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
    discover_active_gemma_models,
    call_gemma_ai_studio_vision,
    call_cloudflare_vision_chat,
    call_gemini_generate_content_self_healing,
)

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024  # 25 MB cap (supports standard Discord PDF/doc uploads safely)
MAX_EMBED_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per embedded image cap for 512MB Render container
MAX_BATCH_IMAGES = 12  # 12-image cap across multi-part threads and batch requests (safe within 512MB RAM)

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

# Detects part/thread continuation markers on Part 1:
_X_PART_1_SIGNAL_REGEX = re.compile(
    r"(?:(?:^|[^\d０-９])[1１]\s*[/／]\s*(?:1[0-2]|１[０-２]|[2-9２-９])(?=$|[^\d０-９])|"
    r"(?:^|[^\d０-９])[1１]\s*[〜～~–—−\-ー]\s*(?:1[0-2]|１[０-２]|[2-9２-９])(?=$|[^\d０-９])|"
    r"[①❶⑴]|(?:^|[\s(【\[（第])[1１]\s*話|その\s*[1１]|part\s*1\b|pt\.?\s*1\b|"
    r"前編|上巻|つづく|続く|続きます|リプに続|ツリーに続|→\s*$)",
    re.IGNORECASE,
)

# Detects continuation markers on Part 2+:
_X_NEXT_PART_CAPTION_REGEX = re.compile(
    r"(?:(?:^|[^\d０-９])(?:1[0-2]|１[０-２]|[2-9２-９])\s*[/／]\s*(?:1[0-2]|１[０-２]|[2-9２-９])(?=$|[^\d０-９])|"
    r"(?:^|[^\d０-９])(?:1[0-2]|１[０-２]|[2-9２-９])\s*[〜～~–—−\-ー]\s*(?:1[0-9]|１[０-９]|[2-9２-９])(?=$|[^\d０-９])|"
    r"(?:^|[\s(【\[（第])(?:1[0-2]|１[０-２]|[2-9２-９])(?:\s*(?:枚目|p|P|話)|\s*[)】\]）])|"
    r"[②③④⑤⑥⑦⑧⑨⑩❷❸❹❺❻❼❽❾❿⑵⑶⑷⑸⑹⑺⑻⑼⑽]|"
    r"その\s*(?:1[0-2]|１[０-２]|[2-9２-９])|part\s*(?:[2-9]|1[0-2])\b|pt\.?\s*(?:[2-9]|1[0-2])\b|後編|中編|下巻|続き|続編)",
    re.IGNORECASE,
)

_X_PAGE_RANGE_EXTRACT_REGEX = re.compile(
    r"(?:^|[^\d０-９])([1-9１-９]|1[0-2]|１[０-２])\s*[〜～~–—−\-ー]\s*([1-9１-９]|1[0-9]|１[０-９])(?=$|[^\d０-９])"
)

_FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")


def _extract_x_page_range_bounds(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    m = _X_PAGE_RANGE_EXTRACT_REGEX.search(text)
    if not m:
        return None
    try:
        s_num = int(m.group(1).translate(_FULLWIDTH_DIGIT_TRANS))
        e_num = int(m.group(2).translate(_FULLWIDTH_DIGIT_TRANS))
        if 1 <= s_num <= e_num <= 20:
            return s_num, e_num
    except Exception:
        pass
    return None


def _is_image_only_or_minimal_reply(tweet_obj: dict) -> bool:
    if not tweet_obj.get("photos"):
        return False
    raw_txt = tweet_obj.get("text") or ""
    if re.search(r"https?://(?!t\.co/)\S+", raw_txt, re.IGNORECASE):
        return False
    cleaned = re.sub(r"https?://t\.co/\S+", " ", raw_txt, flags=re.IGNORECASE)
    cleaned = re.sub(r"@[A-Za-z0-9_]+", " ", cleaned)
    cleaned = re.sub(r"#\S+", " ", cleaned)
    cleaned = re.sub(r"[\s\u3000.…,、。!！?？・ー〜～~()（）【】\[\]]+", "", cleaned)
    return len(cleaned) <= 6


_SFX_TOKEN_REGEX = re.compile(
    r"^(?:タッ|ドシッ|ドッ|ドドン|ピュ+|ピシャ+ン|ズシ+|ズゥ+ン|ズーン|ヴォ+|ブル+|ムチ+|ムク+|"
    r"ぷくっ?|ぷる+|だぷっ?|ずり+|ガタッ|クッ|カッ|ボフッ|ハッ|ハァ|フゥ|ゴゴ+|ザワ+|ドキッ|PAN|[!?…・ーッっ]+)$",
    re.IGNORECASE,
)


def _is_pure_sfx_query(q: str) -> bool:
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
    import openpyxl

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
    from pypdf import PdfReader

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
    import docx

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


def _normalize_fxtwitter_tweet_obj(tweet: dict, fallback_screen_name: str = "", fallback_id: str = "") -> dict:
    if isinstance(tweet.get("status"), dict) and "id" not in tweet:
        tweet = tweet["status"]

    author = tweet.get("author") or {}
    if not isinstance(author, dict):
        author = {}
    a_name = str(author.get("name") or "").strip()
    a_handle = str(author.get("screen_name") or fallback_screen_name or "").strip()
    tid = str(tweet.get("id") or fallback_id or "").strip()
    t_text = str(tweet.get("text") or "").strip()
    t_url = str(tweet.get("url") or f"{HTTPS}x.com/{a_handle or 'i'}/status/{tid}").strip()

    raw_rep_to = tweet.get("replying_to")
    rep_to_handle = ""
    rep_to_status = str(
        tweet.get("replying_to_status")
        or tweet.get("replying_to_status_id")
        or tweet.get("in_reply_to_status_id")
        or tweet.get("in_reply_to_status_id_str")
        or tweet.get("parent_id")
        or ""
    ).strip()

    if isinstance(raw_rep_to, dict):
        rep_to_handle = str(raw_rep_to.get("screen_name") or raw_rep_to.get("handle") or "").strip()
        if not rep_to_status:
            rep_to_status = str(
                raw_rep_to.get("status") or raw_rep_to.get("status_id") or raw_rep_to.get("id") or ""
            ).strip()
    elif isinstance(raw_rep_to, str):
        rep_to_handle = raw_rep_to.strip()

    if not rep_to_handle:
        rep_to_handle = str(tweet.get("in_reply_to_screen_name") or "").strip()

    conv_id = str(tweet.get("conversation_id") or tweet.get("conversation_id_str") or "").strip()

    photo_urls = []
    media = tweet.get("media") or {}
    if isinstance(media, dict):
        for ph in media.get("photos") or []:
            if isinstance(ph, dict) and ph.get("url"):
                photo_urls.append(ph["url"])
        if not photo_urls:
            for m_item in media.get("all") or []:
                if isinstance(m_item, dict):
                    u = m_item.get("url") or m_item.get("thumbnail_url")
                    if u and m_item.get("type") in ("photo", "image", "gif", "video"):
                        photo_urls.append(u)
    elif isinstance(media, list):
        for m_item in media:
            if isinstance(m_item, dict):
                u = m_item.get("url") or m_item.get("thumbnail_url")
                if u:
                    photo_urls.append(u)

    quote_info = None
    quote = tweet.get("quote")
    if isinstance(quote, dict):
        q_auth = quote.get("author") or {}
        q_handle = str(q_auth.get("screen_name") or "").strip() if isinstance(q_auth, dict) else ""
        q_text = str(quote.get("text") or "").strip()
        q_photos = []
        q_media = quote.get("media") or {}
        if isinstance(q_media, dict):
            for qph in q_media.get("photos") or []:
                if isinstance(qph, dict) and qph.get("url"):
                    q_photos.append(qph["url"])
        quote_info = {"handle": q_handle, "text": q_text, "photos": q_photos}

    return {
        "id": tid,
        "conversation_id": conv_id,
        "author_name": a_name,
        "author_handle": a_handle,
        "text": t_text,
        "url": t_url,
        "created_at": str(tweet.get("created_at") or "").strip(),
        "created_timestamp": int(tweet.get("created_timestamp") or 0),
        "likes": tweet.get("likes"),
        "retweets": tweet.get("retweets"),
        "replies_count": int(tweet.get("replies") or 0),
        "replying_to": rep_to_handle,
        "replying_to_status": rep_to_status,
        "photos": photo_urls,
        "quote": quote_info,
    }


async def _fetch_single_x_tweet_normalized(screen_name: str, tweet_id: str) -> dict | None:
    session = get_http_session()
    user_seg = screen_name or "i"
    endpoints = (
        f"{HTTPS}api.fxtwitter.com/{user_seg}/status/{tweet_id}",
        f"{HTTPS}api.fxtwitter.com/2/status/{tweet_id}",
        f"{HTTPS}api.vxtwitter.com/{user_seg}/status/{tweet_id}",
    )
    req_headers = {
        "User-Agent": "YukinonDiscordBot/2.0 (FxEmbed Client)",
        "Accept": "application/json",
    }
    for ep in endpoints:
        try:
            async with session.get(
                ep,
                headers=req_headers,
                timeout=aiohttp.ClientTimeout(total=6),
            ) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json(content_type=None)
                if not isinstance(data, dict):
                    continue
                tweet = data.get("tweet") or data.get("status")
                if isinstance(tweet, dict):
                    return _normalize_fxtwitter_tweet_obj(
                        tweet, fallback_screen_name=screen_name, fallback_id=tweet_id
                    )
                if "text" in data or "mediaURLs" in data:
                    u_name = (data.get("user_name") or "").strip()
                    u_screen = (data.get("user_screen_name") or screen_name or "").strip()
                    t_text = (data.get("text") or "").strip()
                    media_urls = [
                        u for u in (data.get("mediaURLs") or [])
                        if isinstance(u, str) and any(
                            ext in u.lower() for ext in (".jpg", ".jpeg", ".png", ".webp", "pbs.twimg.com/media")
                        )
                    ]
                    return {
                        "id": str(tweet_id),
                        "conversation_id": str(data.get("conversationID") or ""),
                        "author_name": u_name,
                        "author_handle": u_screen,
                        "text": t_text,
                        "url": f"{HTTPS}x.com/{u_screen or 'i'}/status/{tweet_id}",
                        "created_at": "",
                        "created_timestamp": 0,
                        "likes": data.get("likes"),
                        "retweets": data.get("retweets"),
                        "replies_count": int(data.get("replies") or 0),
                        "replying_to": (data.get("replyingTo") or "").strip(),
                        "replying_to_status": str(data.get("replyingToID") or "").strip(),
                        "photos": media_urls,
                        "quote": None,
                    }
        except Exception:
            continue
    return None


def _collect_tweet_dicts_from_fxembed_v2_payload(payload, depth: int = 0) -> list[dict]:
    if depth > 6:
        return []
    found: list[dict] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(_collect_tweet_dicts_from_fxembed_v2_payload(item, depth + 1))
    elif isinstance(payload, dict):
        if isinstance(payload.get("status"), dict):
            found.append(payload["status"])
        elif isinstance(payload.get("tweet"), dict):
            found.append(payload["tweet"])
        elif "id" in payload and any(k in payload for k in ("text", "author", "media", "url")):
            found.append(payload)

        for key in ("thread", "statuses", "tweets", "conversation", "replies", "children", "items", "results", "data"):
            val = payload.get(key)
            if isinstance(val, (list, dict)):
                found.extend(_collect_tweet_dicts_from_fxembed_v2_payload(val, depth + 1))
    return found


async def _fetch_x_thread_candidates_from_fxembed_v2(tweet_id: str, author_handle: str) -> list[dict]:
    if not tweet_id:
        return []
    session = get_http_session()
    req_headers = {
        "User-Agent": "YukinonDiscordBot/2.0 (FxEmbed Client)",
        "Accept": "application/json",
    }
    endpoints = [
        f"{HTTPS}api.fxtwitter.com/2/thread/{tweet_id}",
        f"{HTTPS}api.fxtwitter.com/2/conversation/{tweet_id}",
    ]
    if author_handle:
        endpoints.append(f"{HTTPS}api.fxtwitter.com/2/profile/{author_handle}/statuses")
        endpoints.append(f"{HTTPS}api.fxtwitter.com/2/profile/{author_handle}/media")

    async def _fetch_ep(url: str) -> list[dict]:
        try:
            async with session.get(
                url, headers=req_headers, timeout=aiohttp.ClientTimeout(total=6)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                return _collect_tweet_dicts_from_fxembed_v2_payload(data)
        except Exception:
            return []

    batches = await asyncio.gather(*[_fetch_ep(ep) for ep in endpoints], return_exceptions=True)
    normalized_by_id: dict[str, dict] = {}
    for batch in batches:
        if not isinstance(batch, list):
            continue
        for rt in batch:
            try:
                nt = _normalize_fxtwitter_tweet_obj(rt, fallback_screen_name=author_handle)
                if nt["id"] and nt["id"] not in normalized_by_id:
                    normalized_by_id[nt["id"]] = nt
            except Exception:
                continue

    return list(normalized_by_id.values())


async def _fetch_x_twitter_post_info(screen_name: str, tweet_id: str) -> tuple[str, list[str]]:
    t0 = time.monotonic()
    anchor_tweet = await _fetch_single_x_tweet_normalized(screen_name, tweet_id)
    if not anchor_tweet:
        api_log(f"Embed Link (X/Twitter API) -> fallback to Discord native embed for tweet={tweet_id}")
        return "", []

    author_handle = anchor_tweet["author_handle"] or screen_name
    visited_ids: set[str] = {anchor_tweet["id"]}

    parent_chain: list[dict] = []
    curr = anchor_tweet
    for _ in range(4):
        p_id = curr.get("replying_to_status") or ""
        p_to = (curr.get("replying_to") or "").lower()
        if not p_id or p_id in visited_ids or (author_handle and p_to and p_to != author_handle.lower()):
            break
        parent_tweet = await _fetch_single_x_tweet_normalized(author_handle, p_id)
        if not parent_tweet:
            break
        if author_handle and parent_tweet["author_handle"].lower() != author_handle.lower():
            break
        visited_ids.add(parent_tweet["id"])
        parent_chain.append(parent_tweet)
        curr = parent_tweet

    parent_chain.reverse()
    ordered_tweets: list[dict] = parent_chain + [anchor_tweet]

    tail_tweet = ordered_tweets[-1]
    has_explicit_part_marker = any(_X_PART_1_SIGNAL_REGEX.search(t["text"]) for t in ordered_tweets)
    has_forward_signal = bool(
        has_explicit_part_marker
        or (tail_tweet["replies_count"] > 0 and len(tail_tweet["photos"]) >= 1)
    )

    if has_forward_signal and len(ordered_tweets) < 6:
        thread_pool = await _fetch_x_thread_candidates_from_fxembed_v2(tail_tweet["id"], author_handle)
        if ordered_tweets[0]["id"] != tail_tweet["id"] and len(thread_pool) <= 1:
            root_pool = await _fetch_x_thread_candidates_from_fxembed_v2(ordered_tweets[0]["id"], author_handle)
            existing_ids = {x["id"] for x in thread_pool}
            for item in root_pool:
                if item["id"] not in existing_ids:
                    thread_pool.append(item)

        root_conv_id = ordered_tweets[0].get("conversation_id") or ordered_tweets[0]["id"]

        for _ in range(5):
            if len(ordered_tweets) >= 6:
                break
            curr_tail = ordered_tweets[-1]
            curr_range = _extract_x_page_range_bounds(curr_tail["text"])

            direct_children = [
                cand for cand in thread_pool
                if cand["id"] not in visited_ids
                and (not author_handle or cand["author_handle"].lower() == author_handle.lower())
                and cand.get("replying_to_status") == curr_tail["id"]
                and (len(cand["photos"]) >= 1 or _X_NEXT_PART_CAPTION_REGEX.search(cand["text"]))
            ]

            if not direct_children:
                curr_ts = curr_tail.get("created_timestamp") or 0
                for cand in thread_pool:
                    if cand["id"] in visited_ids:
                        continue
                    if author_handle and cand["author_handle"].lower() != author_handle.lower():
                        continue
                    cand_parent = cand.get("replying_to_status") or ""
                    if cand_parent and cand_parent not in visited_ids:
                        continue

                    same_conv_self_reply = bool(
                        root_conv_id
                        and cand.get("conversation_id") == root_conv_id
                        and (not cand.get("replying_to") or cand.get("replying_to", "").lower() == author_handle.lower())
                        and (len(cand["photos"]) >= 1 or _X_NEXT_PART_CAPTION_REGEX.search(cand["text"]))
                    )
                    cand_has_marker = bool(_X_NEXT_PART_CAPTION_REGEX.search(cand["text"]))
                    cand_is_img_only = _is_image_only_or_minimal_reply(cand)

                    if not (same_conv_self_reply or cand_has_marker or cand_is_img_only):
                        continue

                    cand_ts = cand.get("created_timestamp") or 0
                    if curr_ts and cand_ts and not (0 <= cand_ts - curr_ts <= 86400):
                        continue
                    direct_children.append(cand)

            if not direct_children:
                break

            def _child_sort_key(c: dict):
                range_bonus = 1
                if curr_range:
                    c_range = _extract_x_page_range_bounds(c["text"])
                    if c_range and c_range[0] == curr_range[1] + 1:
                        range_bonus = 0
                img_bonus = 0 if c.get("photos") else 1
                return (
                    range_bonus,
                    img_bonus,
                    c.get("created_timestamp") or 0,
                    int(c["id"]) if c["id"].isdigit() else 0,
                )

            direct_children.sort(key=_child_sort_key)
            next_part = direct_children[0]
            visited_ids.add(next_part["id"])
            ordered_tweets.append(next_part)

    merged_photo_urls: list[str] = []
    seen_photo_keys: set[str] = set()
    for tw in ordered_tweets:
        p_list = list(tw["photos"])
        if not p_list and tw.get("quote") and tw["quote"].get("photos"):
            p_list = list(tw["quote"]["photos"])
        for pu in p_list:
            dkey = _normalize_image_dedup_key(pu)
            if pu and dkey not in seen_photo_keys and len(merged_photo_urls) < MAX_BATCH_IMAGES:
                seen_photo_keys.add(dkey)
                merged_photo_urls.append(pu)

    primary = ordered_tweets[0]
    a_disp = (
        f"{primary['author_name']} (@{primary['author_handle']})"
        if primary["author_handle"]
        else primary["author_name"]
    )

    if len(ordered_tweets) == 1:
        lines = [
            f"Platform: X (Twitter) | Post URL: {primary['url']}",
            f"Author / Artist: {a_disp}",
            f"Post Caption / Title Text: {primary['text'] or '(No text caption)'}",
        ]
        if primary["created_at"]:
            lines.append(f"Posted At: {primary['created_at']}")
        if primary["likes"] is not None or primary["retweets"] is not None:
            lines.append(f"Engagement: {primary['likes'] or 0} Likes | {primary['retweets'] or 0} Reposts")
        q_obj = primary.get("quote")
        if q_obj and q_obj.get("text"):
            lines.append(f"Quoted Post (@{q_obj['handle']}):\n{q_obj['text']}")
    else:
        lines = [
            f"Platform: X (Twitter) Multi-Part Thread ({len(ordered_tweets)} Connected Parts) | Root URL: {primary['url']}",
            f"Author / Artist: {a_disp}",
            f"Total Collected Images Across Thread: {len(merged_photo_urls)} (capped at {MAX_BATCH_IMAGES})",
        ]
        for idx, tw in enumerate(ordered_tweets, 1):
            cap_disp = tw["text"] or "(Image-only continuation reply — no text caption)"
            lines.append(
                f"\n[Part {idx}/{len(ordered_tweets)} — {tw['url']} ({len(tw['photos'])} photo(s))]\n"
                f"Caption: {cap_disp}"
            )
            q_obj = tw.get("quote")
            if q_obj and q_obj.get("text"):
                lines.append(f"Quoted Post (@{q_obj['handle']}): {q_obj['text']}")

    api_log(
        f"Embed Link (FxTwitter) -> tweet={tweet_id}, parts={len(ordered_tweets)}, "
        f"photos={len(merged_photo_urls)} ({time.monotonic() - t0:.2f}s)"
    )
    return "\n".join(lines), merged_photo_urls


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
                            for item in p_items[:MAX_BATCH_IMAGES]:
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
            return "\n".join(lines), image_urls[:MAX_BATCH_IMAGES]
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
    max_images: int = MAX_BATCH_IMAGES,
) -> tuple[list[tuple[bytes, str, str]], list[str], set[str]]:
    effective_cap = min(MAX_BATCH_IMAGES, max(1, max_images))
    downloaded_images: list[tuple[bytes, str, str]] = []
    metadata_sections: list[str] = []
    handled_urls: set[str] = set()
    candidate_image_urls: list[tuple[str, str]] = []
    seen_dedup_keys: set[str] = set()
    resolved_social_api_post = False

    def _queue_img(u: str, referer: str = ""):
        if len(candidate_image_urls) >= effective_cap:
            return
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
                resolved_social_api_post = True
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
                resolved_social_api_post = True
                metadata_sections.append(f"=== Embedded Pixiv Artwork Metadata & Caption ===\n{px_meta}")
                for p_url in px_photos:
                    _queue_img(p_url, referer=f"{HTTPS}www.pixiv.net/artworks/{illust_id}")
                continue

        bsky_m = BSKY_POST_URL_REGEX.search(url)
        if bsky_m:
            handled_urls.add(url)
            b_meta, b_photos = await _fetch_bsky_post_info(bsky_m.group(1), bsky_m.group(2))
            if b_meta:
                resolved_social_api_post = True
                metadata_sections.append(f"=== Embedded Bluesky Post Metadata & Caption ===\n{b_meta}")
                for p_url in b_photos:
                    _queue_img(p_url)
                continue

        if DIRECT_IMAGE_URL_REGEX.match(url):
            handled_urls.add(url)
            _queue_img(url)

    if not resolved_social_api_post or not candidate_image_urls:
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
            for img_u, ref in candidate_image_urls[:effective_cap]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (img_u, _), res in zip(candidate_image_urls[:effective_cap], results):
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


async def _call_gemini_multi_image_batch(
    encoded_images: list[tuple[str, str, str]],
    batch_prompt: str,
    timeout_sec: int = 24,
    max_output_tokens: int = 2560,
    page_offset: int = 0,
) -> str:
    if not GEMINI_API_KEY:
        return ""
    parts = [{"text": batch_prompt}]
    for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
        p_num = page_offset + idx
        parts.append({"text": f"=== Page {p_num} (`{fname}`) ==="})
        parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    try:
        data = await call_gemini_generate_content_self_healing(
            payload,
            timeout_sec=timeout_sec,
            purpose=f"MultiImageOCR({len(encoded_images)}p)",
            max_models=2,
            max_wall_sec=float(timeout_sec + 12),
        )
        if data:
            resp_parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            out_txt = "\n".join(p.get("text", "") for p in resp_parts if p.get("text") and not p.get("thought", False)).strip()
            out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
            if out_txt:
                return out_txt
    except asyncio.CancelledError:
        raise
    except Exception as e:
        api_log(f"MultiImageOCR Tier 1 Gemini -> error ({type(e).__name__})")
    return ""


async def _call_ai_studio_gemma_multi_image_batch(
    encoded_images: list[tuple[str, str, str]],
    batch_prompt: str,
    timeout_sec: int = 22,
    max_output_tokens: int = 2560,
    page_offset: int = 0,
) -> str:
    """Tier 2 Google AI Studio Gemma 4 multi-image batch backup (uses fast 26B MoE)."""
    if not GEMINI_API_KEY:
        return ""
    gemma_models = await discover_active_gemma_models(vision_required=True)
    candidates = [m for m in gemma_models if "26b" in m] or gemma_models[:1]
    if not candidates:
        return ""

    parts = [{"text": batch_prompt}]
    for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
        p_num = page_offset + idx
        parts.append({"text": f"=== Page {p_num} (`{fname}`) ==="})
        parts.append({"inlineData": {"mimeType": mime_type, "data": b64_img}})

    session = get_http_session()
    for model_id in candidates[:1]:
        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_output_tokens,
            },
        }
        t0 = time.monotonic()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    data = await resp.json()
                    resp_parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                    out_txt = "\n".join(
                        p.get("text", "") for p in resp_parts
                        if p.get("text") and not p.get("thought", False)
                    ).strip()
                    out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
                    if out_txt:
                        api_log(f"MultiImageOCR (Tier 2 AI Studio Gemma) -> {model_id} 200 OK ({elapsed:.2f}s)")
                        return out_txt
                api_log(f"MultiImageOCR (Tier 2 AI Studio Gemma) -> {model_id} HTTP {resp.status} ({elapsed:.2f}s)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            api_log(f"MultiImageOCR (Tier 2 AI Studio Gemma) -> {model_id} error ({type(e).__name__})")
    return ""


async def _call_cf_multi_image_batch(
    encoded_images: list[tuple[str, str, str]],
    batch_prompt: str,
    timeout_sec: int = 24,
    max_tokens: int = 2560,
    page_offset: int = 0,
) -> str:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return ""
    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    user_content: list[dict] = [{"type": "text", "text": batch_prompt}]
    for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
        p_num = page_offset + idx
        user_content.append({"type": "text", "text": f"=== Page {p_num} (`{fname}`) ==="})
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
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            cf_chat_url, headers=headers, json=cf_payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content") or ""
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    if content:
                        api_log(f"MultiImageOCR (Tier 3 CF) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            api_log(f"MultiImageOCR (Tier 3 CF) -> HTTP {resp.status} ({elapsed:.2f}s)")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        api_log(f"MultiImageOCR (Tier 3 CF) -> error ({type(e).__name__})")
    return ""


async def _process_single_encoded_chunk(
    encoded_chunk: list[tuple[str, str, str]],
    total_pages: int,
    page_offset: int,
    post_meta_hint: str,
    out_tokens: int,
) -> str:
    start_p = page_offset + 1
    end_p = page_offset + len(encoded_chunk)
    batch_prompt = (
        f"You are analyzing Pages {start_p} to {end_p} (of a {total_pages}-page sequential comic/post) for Yukino Yukinoshita.\n"
        + (f"Post Metadata / Caption Context:\n{post_meta_hint[:1400]}\n\n" if post_meta_hint else "")
        + "Examine these pages together in sequence so you maintain strict character & franchise consistency:\n"
        "1. [Series / Franchise & Character Identification — STRICT EVIDENCE RULE]:\n"
        "   - Verify concrete visual traits (hair/eye color, signature hair accessories/ornaments, outfits, presence or absence of floating halos) "
        "and cross-check with the post caption/tags before naming ANY franchise or character.\n"
        "   - NEVER mix characters from one franchise with the title/protagonist of a different franchise! "
        "(For example: do NOT label a post as 'Blue Archive' or call a male character 'Sensei' unless floating halos above characters' heads "
        "or explicit Blue Archive names are actually visible; if characters are Firefly/Hotaru and Caelus, the franchise is 'Honkai: Star Rail').\n"
        "   - If the franchise or character names are not 100% certain from the visuals or caption, write 'Unknown / Unconfirmed Series' "
        "and describe the characters strictly by appearance rather than guessing.\n"
        "2. [Page-by-Page Story & Panel-by-Panel Dialogue/SFX (Right-to-Left, Top-to-Bottom)]:\n"
        f"   - For each page (Page {start_p} .. Page {end_p}), summarize the visual events and list all dialogue, thoughts, "
        "and key SFX in Japanese reading order tagged with the speaker."
    )

    has_gemini = bool(GEMINI_API_KEY)
    has_cf = bool(CF_ACCOUNT_ID and CF_API_TOKEN)
    gemini_timeout = 24
    backup_timeout = 22

    async def _run_backup() -> str:
        if has_gemini:
            gemma_res = await _call_ai_studio_gemma_multi_image_batch(
                encoded_chunk,
                batch_prompt,
                timeout_sec=backup_timeout,
                max_output_tokens=out_tokens,
                page_offset=page_offset,
            )
            if gemma_res:
                return gemma_res
        if has_cf:
            return await _call_cf_multi_image_batch(
                encoded_chunk,
                batch_prompt,
                timeout_sec=backup_timeout,
                max_tokens=out_tokens,
                page_offset=page_offset,
            )
        return ""

    if not has_gemini:
        return await _run_backup()

    gemini_task = asyncio.create_task(
        _call_gemini_multi_image_batch(
            encoded_chunk,
            batch_prompt,
            timeout_sec=gemini_timeout,
            max_output_tokens=out_tokens,
            page_offset=page_offset,
        )
    )
    done, _ = await asyncio.wait({gemini_task}, timeout=12.0)
    if gemini_task in done:
        try:
            res = gemini_task.result()
            if res:
                return res
        except Exception:
            pass
        return await _run_backup()

    api_log(f"Hedge (MultiImageOCR p{start_p}-{end_p}) -> Gemini >12s; racing backup batch in parallel")
    backup_task = asyncio.create_task(_run_backup())
    tasks = {gemini_task, backup_task}
    try:
        while tasks:
            done_set, pending_set = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for finished in done_set:
                tasks.discard(finished)
                try:
                    out = finished.result()
                except Exception:
                    out = ""
                if out:
                    for p in pending_set:
                        p.cancel()
                    if pending_set:
                        await asyncio.gather(*pending_set, return_exceptions=True)
                    return out
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return ""


async def extract_multi_images_batch_ocr(
    images_list: list[tuple[bytes, str, str]],
    post_meta_hint: str = "",
) -> str:
    """
    Inspects 2 to 12 sequential comic/illustration pages using `gemini-3.5-flash-lite`
    (`thinkingLevel: "MINIMAL"`). Automatically splits 6..12 page threads into parallel
    <=5-page mini-batches so payload sizes stay small and finish in ~4-8 seconds.
    """
    target_list = images_list[:MAX_BATCH_IMAGES]
    if len(target_list) < 2:
        return ""

    n_imgs = len(target_list)
    if n_imgs >= 6:
        target_dim, target_q, out_tokens = 800, 78, 2560
    elif n_imgs >= 3:
        target_dim, target_q, out_tokens = 864, 82, 2048
    else:
        target_dim, target_q, out_tokens = 928, 85, 2048

    encoded_images: list[tuple[str, str, str]] = []
    for raw_b, fname, _ in target_list:
        try:
            b64_str, mime_type, _ = await asyncio.to_thread(
                _compress_image_for_vision_ocr_sync, raw_b, target_dim, target_q
            )
            encoded_images.append((b64_str, mime_type, fname))
        except Exception:
            continue

    if len(encoded_images) < 2:
        return ""

    try:
        total_p = len(encoded_images)
        # Split 6..12 images into parallel mini-batches of <=5 pages so payloads never stall
        if total_p <= 5:
            single_out = await _process_single_encoded_chunk(
                encoded_images, total_p, 0, post_meta_hint, out_tokens
            )
            if single_out:
                return f"=== Unified Multi-Image Visual & OCR Analysis ({total_p} Images) ===\n{single_out}"
            return ""

        chunk_size = 5
        chunks = [
            (offset, encoded_images[offset : offset + chunk_size])
            for offset in range(0, total_p, chunk_size)
        ]
        chunk_results = await asyncio.gather(
            *[
                _process_single_encoded_chunk(ch, total_p, offset, post_meta_hint, out_tokens)
                for offset, ch in chunks
            ]
        )
        valid_sections = [res for res in chunk_results if res]
        if len(valid_sections) == len(chunks):
            merged = "\n\n".join(valid_sections)
            return f"=== Unified Multi-Image Visual & OCR Analysis ({total_p} Images) ===\n{merged}"
        return ""
    finally:
        encoded_images.clear()


async def _call_gemini_single_image_vision(
    ocr_prompt: str, mime_type: str, b64_img: str, purpose: str = "Image OCR Tier 1 Gemini"
) -> str:
    if not GEMINI_API_KEY:
        return ""
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": ocr_prompt},
                    {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    data = await call_gemini_generate_content_self_healing(
        payload,
        timeout_sec=12,
        purpose=purpose,
        max_models=2,
        max_wall_sec=16.0,
    )
    if not data:
        return ""
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    out = "\n".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought", False)).strip()
    return re.sub(r"<think>[\s\S]*?</think>", "", out).strip()


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
        "unmistakable visual features (hair ornaments, outfits, halos/no halos) or the post caption above. NEVER mix character names "
        "from one franchise with a different franchise title, and NEVER invent nonsensical style names or guess when ambiguous—write 'Unknown / Unconfirmed Series' instead! "
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

    # Tier 1: Gemini 3.5 Flash-Lite -> Gemini 3.1 Flash-Lite
    if GEMINI_API_KEY:
        vision_result = await _call_gemini_single_image_vision(ocr_prompt, mime_type, b64_img)

    # Tier 2: Cloudflare Workers AI Vision (`@cf/google/gemma-4-26b-a4b-it`)
    if not vision_result:
        cf_ocr = await call_cloudflare_vision_chat(ocr_prompt, mime_type, b64_img, purpose="Image OCR Tier 2 CF")
        if cf_ocr:
            vision_result = cf_ocr

    # Tier 3: Google AI Studio Gemma 4 Vision (`gemma-4-26b-a4b-it`)
    if not vision_result:
        gemma_ocr = await call_gemma_ai_studio_vision(
            ocr_prompt, mime_type, b64_img, timeout_sec=12, purpose="Image OCR Tier 3 AI Studio"
        )
        if gemma_ocr:
            vision_result = gemma_ocr

    # Tier 4: Groq LPU Vision (`qwen/qwen3.8-27b`)
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
                    chat_url, headers=headers, json=groq_vision_payload, timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    elapsed = time.monotonic() - t0
                    if resp.status == 200:
                        state.active_groq_vision_model = g_mod
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                        if content:
                            api_log(f"Image OCR (Tier 4 Groq) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                            vision_result = content
                            break
                    api_log(f"Image OCR (Tier 4 Groq) -> {g_mod} HTTP {resp.status}; fallback to Tier 5")
            except Exception as e:
                api_log(f"Image OCR (Tier 4 Groq) -> {g_mod} failed ({type(e).__name__}); fallback to Tier 5")

    # Tier 5: Modal T4 GPU
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
                            api_log(f"Image OCR (Tier 5 Modal) -> {ACTIVE_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                            vision_result = content
        except Exception as e:
            api_log(f"Image OCR (Tier 5 Modal) -> {ACTIVE_VISION_MODEL} failed ({type(e).__name__})")

    out_parts = []
    if reverse_tasks:
        rev_results = await asyncio.gather(*reverse_tasks, return_exceptions=True)
        saucenao_txt = rev_results[0] if (len(rev_results) > 0 and isinstance(rev_results[0], str)) else ""
        tracemoe_txt = rev_results[1] if (len(rev_results) > 1 and isinstance(rev_results[1], str)) else ""

        if saucenao_txt:
            out_parts.append(saucenao_txt)
        else:
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


async def extract_all_context(
    message: discord.Message,
    preloaded_attachments: dict[int, bytes] | None = None,
) -> tuple[str, str]:
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
            if preloaded_attachments and att.id in preloaded_attachments:
                raw_bytes = preloaded_attachments[att.id]
            else:
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
            elif fname.endswith(IMAGE_EXTENSIONS) and len(image_att_batch) < MAX_BATCH_IMAGES:
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
            sem = asyncio.Semaphore(3)

            async def _ocr_att_item(idx: int, img_b: bytes, img_fn: str) -> tuple[int, str, str]:
                async with sem:
                    ocr_out = await extract_image_via_transient_ocr(
                        img_b,
                        img_fn,
                        run_source_lookup=(idx <= 2),
                        post_meta_hint=message.content or "",
                        page_label=f"Page {idx} of {total_att_imgs}" if total_att_imgs > 1 else "",
                    )
                    return idx, img_fn, ocr_out

            att_tasks = [
                _ocr_att_item(idx, img_b, img_fn)
                for idx, (img_b, img_fn, _) in enumerate(image_att_batch, 1)
            ]
            for idx, img_fn, ocr_out in await asyncio.gather(*att_tasks):
                processed_image_count += 1
                if STACK_TRACE_SIGNATURE_REGEX.search(ocr_out):
                    has_code_file = True
                blocks.append(f"=== Image #{idx} Inspection & Source Lookup (`{img_fn}`) ===\n{ocr_out}")
        image_att_batch.clear()

    embed_images, embed_meta_sections, handled_social_urls = await collect_embedded_images_and_metadata(
        message, ref_msg=ref_msg, max_images=max(1, MAX_BATCH_IMAGES - processed_image_count)
    )
    for meta_sec in embed_meta_sections:
        blocks.append(meta_sec)

    if embed_images and processed_image_count < MAX_BATCH_IMAGES:
        remaining_slots = MAX_BATCH_IMAGES - processed_image_count
        target_embed_imgs = embed_images[:remaining_slots]
        meta_hint = "\n".join(embed_meta_sections)
        has_authoritative_social_meta = any(s.startswith("=== Embedded ") for s in embed_meta_sections)
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
            sem = asyncio.Semaphore(3)

            async def _ocr_embed_item(idx: int, emb_bytes: bytes, emb_fname: str, emb_url: str) -> str:
                async with sem:
                    try:
                        do_lookup = (not has_authoritative_social_meta) and (idx <= 2)
                        ocr_out = await extract_image_via_transient_ocr(
                            emb_bytes,
                            emb_fname,
                            run_source_lookup=do_lookup,
                            post_meta_hint=meta_hint,
                            page_label=f"Page {idx} of {total_emb_imgs}" if total_emb_imgs > 1 else "",
                        )
                        return f"=== Embedded Image #{idx} Inspection & Source Lookup (`{emb_fname}` | `{emb_url}`) ===\n{ocr_out}"
                    except Exception as e:
                        return f"[Error inspecting embedded image `{emb_fname}`: {e}]"

            emb_tasks = [
                _ocr_embed_item(idx, emb_bytes, emb_fname, emb_url)
                for idx, (emb_bytes, emb_fname, emb_url) in enumerate(target_embed_imgs, 1)
            ]
            for res_block in await asyncio.gather(*emb_tasks):
                processed_image_count += 1
                if STACK_TRACE_SIGNATURE_REGEX.search(res_block):
                    has_code_file = True
                blocks.append(res_block)

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