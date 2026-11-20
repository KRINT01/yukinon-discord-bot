import re
import json
import time
import socket
import asyncio
import ipaddress
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, quote_plus

import aiohttp
import trafilatura
from ddgs import DDGS

from config import (
    HTTPS,
    DEFAULT_WEATHER_CITY,
    DEFAULT_WEATHER_LAT,
    DEFAULT_WEATHER_LON,
    THAI_LOCATION_COORDS,
    WMO_WEATHER_TH,
    CGNAT_NETWORK,
    CODE_AND_LOG_EXTENSIONS,
    CODE_INTENT_PATTERN,
    STACK_TRACE_SIGNATURE_REGEX,
    WEATHER_INTENT_PATTERN,
    WEB_SEARCH_INTENT_PATTERN,
    RECENCY_INTENT_REGEX,
    USER_CORRECTION_PATTERN,
    THAI_CHAR_REGEX,
    NEWS_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    GAME_QUERY_REGEX,
    URL_REGEX,
    TAVILY_API_KEY,
    DDGS_PROXY,
    GEMINI_API_KEY,
    channel_memory,
    channel_active_topic,
    channel_last_weather,
    state,
    clean_url,
    api_log,
    get_http_session,
    channel_has_recent_code_attachment,
)

ENGLISH_ENTITY_REGEX = re.compile(r"\b[A-Za-z][A-Za-z0-9_.:-]{1,25}(?:\s+[A-Za-z0-9_.:-]{1,25}){0,4}\b")

STOPWORD_ENTITIES = {
    "search", "google", "wikipedia", "wiki", "what", "when", "where", "which", "who", "how",
    "the", "and", "for", "with", "this", "that", "from", "are", "you", "can", "help", "not",
    "yes", "no", "ok", "okay", "pdf", "docx", "xlsx", "file", "http", "https",
    "web", "extraction", "error", "readable", "found", "embed",
    "markdown", "json", "html", "table", "tables", "bullet", "bullets", "points", "checklist",
    "items", "in-bubble", "bubble", "ocr", "sfx", "canon", "lore",
}
_MARKUP_NOISE_WORDS = {"markdown", "json", "html", "in-bubble", "ocr", "sfx", "error", "extraction", "readable"}

# Bounded 5-minute in-memory TTL cache for search results (eliminates redundant lookups on follow-up buttons)
_SEARCH_CACHE: dict[str, tuple[float, str]] = {}
_SEARCH_CACHE_TTL_SEC = 300.0
_SEARCH_CACHE_MAX_ENTRIES = 32


def _get_cached_search(cache_key: str) -> str:
    entry = _SEARCH_CACHE.get(cache_key)
    if not entry:
        return ""
    ts, val = entry
    if (time.time() - ts) < _SEARCH_CACHE_TTL_SEC:
        return val
    _SEARCH_CACHE.pop(cache_key, None)
    return ""


def _store_cached_search(cache_key: str, value: str):
    if not value:
        return
    now = time.time()
    if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX_ENTRIES:
        oldest_key = min(_SEARCH_CACHE.keys(), key=lambda k: _SEARCH_CACHE[k][0])
        _SEARCH_CACHE.pop(oldest_key, None)
    _SEARCH_CACHE[cache_key] = (now, value)


def _strip_context_and_system_blocks(text: str) -> str:
    """Strips appended context headers, speaker tags, and internal prompt directives before intent/entity scanning."""
    if not text:
        return ""
    head = re.split(
        r"\n\n(?:===|<untrusted_web_search_results|<mentioned_users_dossier>|\[Attachment |\[PDF Notice:|\[Web extraction error:|"
        r"\[กฎสำคัญ\]|\[บริบทหัวข้อก่อนหน้า\]|\[บริบทบทสนทนาล่าสุดในห้องเสียง\]|\[สรุปคำตอบก่อนหน้า|\[เนื้อหาอ้างอิงจากไฟล์|\[ข้อมูลบริบทระบบปัจจุบัน\])",
        text,
        maxsplit=1,
    )[0]
    head = re.sub(r"^\[ผู้พูด:[^\]]+\]\s*", "", head.strip())
    head = re.sub(
        r"\[(?:Web extraction error|Error reading|Attachment Size Notice|PDF Notice|กฎสำคัญ|SEARCH_HINT):[^\]]+\]",
        " ",
        head,
        flags=re.IGNORECASE,
    )
    head = re.sub(r"\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*?\)", " ", head)
    return head.strip()


def detect_coding_intent(
    text: str,
    attachments: list,
    channel_id: int = None,
    extracted_ctx: str = "",
) -> bool:
    for att in attachments:
        if att.filename.lower().endswith(CODE_AND_LOG_EXTENSIONS):
            return True

    clean_user_text = _strip_context_and_system_blocks(text)
    user_text_no_urls = URL_REGEX.sub(" ", clean_user_text).strip()

    if user_text_no_urls:
        if CODE_INTENT_PATTERN.search(user_text_no_urls) or STACK_TRACE_SIGNATURE_REGEX.search(user_text_no_urls):
            return True

    if extracted_ctx and STACK_TRACE_SIGNATURE_REGEX.search(extracted_ctx):
        return True

    has_non_code_fresh_media = bool(attachments) or bool(extracted_ctx) or bool(URL_REGEX.search(clean_user_text))
    if has_non_code_fresh_media:
        return False

    if channel_has_recent_code_attachment(channel_id):
        return True
    return False


def detect_web_search_intent(text: str, channel_id: int = None) -> bool:
    clean_q = _strip_context_and_system_blocks(text)
    if not clean_q:
        return False
    if (
        WEB_SEARCH_INTENT_PATTERN.search(clean_q)
        or RECENCY_INTENT_REGEX.search(clean_q)
        or NEWS_QUERY_REGEX.search(clean_q)
        or ANIME_MANGA_QUERY_REGEX.search(clean_q)
        or GAME_QUERY_REGEX.search(clean_q)
        or USER_CORRECTION_PATTERN.search(clean_q)
    ):
        return True
    if channel_id and channel_id in channel_active_topic:
        if time.time() - channel_active_topic[channel_id].get("timestamp", 0) < 1800:
            if re.search(r"(\?|ไหม|มั้ย|ยังไง|อะไร|ไหน|เมื่อไหร่|กี่ปี|ทำไม|หรอ|เหรอ|ลอง|สรุป|เช็ค)", clean_q):
                return True
    return False


def _is_valid_entity_phrase(phrase: str) -> bool:
    p = phrase.strip()
    if not p or p.lower() in STOPWORD_ENTITIES:
        return False
    words = [w.lower() for w in p.split()]
    if all(w in STOPWORD_ENTITIES for w in words):
        return False
    if any(w in _MARKUP_NOISE_WORDS for w in words):
        return False
    return True


def update_and_get_channel_topic(user_text: str, channel_id: int = None) -> str:
    if channel_id is None:
        return ""
    now = time.time()
    existing = channel_active_topic.get(channel_id)
    if existing and (now - existing.get("timestamp", 0) > 1800):
        channel_active_topic.pop(channel_id, None)
        existing = None

    yt_title_m = re.search(r"Video Title:\s*(.+)", user_text)
    if yt_title_m:
        yt_title = re.sub(
            r"\s*[|_—–-]\s*(?:Release Date Trailer|Official Trailer|Trailer|PV|Teaser).*",
            "",
            yt_title_m.group(1),
            flags=re.IGNORECASE,
        ).strip()[:70]
        if yt_title:
            channel_active_topic[channel_id] = {"topic": yt_title, "timestamp": now}
            return yt_title

    question_only = _strip_context_and_system_blocks(user_text)
    text_without_urls = re.sub(r"https?://\S+", " ", question_only)

    is_corr = bool(USER_CORRECTION_PATTERN.search(text_without_urls))
    en_matches = [
        m.strip() for m in ENGLISH_ENTITY_REGEX.findall(text_without_urls)
        if _is_valid_entity_phrase(m)
    ]

    if en_matches and not is_corr and len(en_matches) <= 3:
        candidate = en_matches[0]
        if existing:
            prev_topic = existing.get("topic", "")
            if candidate.lower() in prev_topic.lower() and len(prev_topic) > len(candidate):
                existing["timestamp"] = now
                return prev_topic
        channel_active_topic[channel_id] = {"topic": candidate, "timestamp": now}
        return candidate

    for m in en_matches:
        if len(m.split()) >= 3:
            prev_t = existing.get("topic", "") if existing else ""
            combined_t = f"{prev_t} ({m})".strip() if (prev_t and prev_t.lower() not in m.lower()) else m
            channel_active_topic[channel_id] = {"topic": combined_t[:80], "timestamp": now}
            return combined_t[:80]

    if existing:
        existing["timestamp"] = now
        return existing.get("topic", "")

    if channel_id in channel_memory:
        for msg in list(channel_memory[channel_id]):
            if msg.get("role") != "user":
                continue
            hist_clean = re.sub(r"https?://\S+", " ", _strip_context_and_system_blocks(msg.get("content", "")))
            first_ents = [
                m.strip() for m in ENGLISH_ENTITY_REGEX.findall(hist_clean)
                if _is_valid_entity_phrase(m)
            ]
            if first_ents:
                root_t = first_ents[0]
                channel_active_topic[channel_id] = {"topic": root_t, "timestamp": now}
                return root_t
    return ""


def build_contextual_search_query(user_text: str, channel_id: int = None) -> str:
    hint_m = re.search(r"\[SEARCH_HINT:\s*([^\]]+?)\s*\]", user_text or "", re.IGNORECASE)
    if hint_m:
        clean_hint = re.sub(r"\s+", " ", hint_m.group(1)).strip()
        if clean_hint:
            return clean_hint[:140]

    active_topic = update_and_get_channel_topic(user_text, channel_id=channel_id)

    question_only = _strip_context_and_system_blocks(user_text)
    raw = re.sub(r"https?://\S+", " ", question_only)
    raw = re.sub(r"[\r\n]+", " ", raw).strip()

    en_in_msg = [
        m.strip() for m in ENGLISH_ENTITY_REGEX.findall(raw)
        if _is_valid_entity_phrase(m)
    ]

    subject_prefix = active_topic
    if not subject_prefix and en_in_msg:
        subject_prefix = en_in_msg[0]

    if subject_prefix and ENGLISH_ENTITY_REGEX.search(subject_prefix):
        base_subj = re.sub(r"[()]", " ", subject_prefix).strip()
        facets = []
        if re.search(r"(เนื้อเรื่อง|สรุปเนื้อเรื่อง|ไปถึงไหนแล้ว|บทไหน|story|plot|lore|episode)", raw, re.IGNORECASE):
            facets.append("story plot episodes summary wiki")
        if re.search(r"(คลาส|อาชีพ|เล่นคลาสอะไร|classes|class)", raw, re.IGNORECASE):
            facets.append("all classes weapons list wiki")
        if re.search(r"(อาวุธ|เวท|ธนู|ดาบ|ปืน|weapon|weapons)", raw, re.IGNORECASE):
            facets.append("class weapons list wiki")
        if re.search(r"(ของค่ายไหน|ใครพัฒนา|พัฒนาโดย|เกมเป็นของ|มาจากไหนก่อน|developer|publisher|sega)", raw, re.IGNORECASE):
            facets.append("developer publisher release date")
        if re.search(r"(เปิดมากี่ปี|เปิดมาจะ|เปิดเมื่อไหร่|โหมดทดลอง|release\s+date|year)", raw, re.IGNORECASE):
            facets.append("release date year official")
        if re.search(r"(น่าเล่นไหม|สนุกไหม|รีวิว|ระบบเกม|เป็นยังไง|review|gameplay|worth\s+playing)", raw, re.IGNORECASE):
            facets.append("steam gameplay review features")

        extra_en = [
            e for e in en_in_msg
            if e.lower() not in base_subj.lower() and len(e) > 2
        ][:3]

        if facets or extra_en or USER_CORRECTION_PATTERN.search(raw):
            q_parts = [base_subj] + extra_en + facets
            return re.sub(r"\s+", " ", " ".join(q_parts)).strip()[:140]

    stripped = re.sub(
        r"(นี่ก่อนตอบมา|เธอได้ลอง|ดูบ้างไหม|หรือ\s*wikipedia\s*ก็ยังดี|เพราะข้อมูลที่เธอตอบมา.*|"
        r"ดูเป็นคำตอบที่ไม่ค่อยช่วยเท่าไหร่เลยนะ|นี่เธอพยายามหาข้อมูลมาสรุปแล้วจริงๆเหรอ|"
        r"จริงๆแล้วข้อมูล.*ที่ให้มาก็ยังผิดอยู่นะ|แถมสรุปเนื้อเรื่องก็ไม่ได้สาระอะไรเลย.*|"
        r"โหมดทดลองอะไรกัน|มาจากไหนก่อน|ลองสรุปมาให้หน่อย|ลองเช็คดูหน่อย)",
        " ",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    stripped = re.sub(r"\s+", " ", stripped).strip()

    if subject_prefix and subject_prefix.lower() not in stripped.lower():
        return f"{subject_prefix} {stripped}".strip()[:140]
    return (stripped or subject_prefix or raw)[:140]


def detect_weather_intent(text: str, channel_id: int = None) -> bool:
    clean_q = _strip_context_and_system_blocks(text)
    if not clean_q:
        return False
    if WEATHER_INTENT_PATTERN.search(clean_q):
        return True
    if channel_id and channel_id in channel_last_weather:
        if time.time() - channel_last_weather[channel_id].get("timestamp", 0) < 1200:
            if re.search(r"(ช่วงเช้า|ตอนเช้า|ช่วงบ่าย|ตอนบ่าย|ช่วงเย็น|ตอนเย็น|กลางคืน|ดึก|คืนนี้|พรุ่งนี้|กี่โมง)", clean_q):
                return True
    return False


async def fetch_live_weather_context(user_text: str, channel_id: int = None) -> str:
    lat = DEFAULT_WEATHER_LAT
    lon = DEFAULT_WEATHER_LON
    loc_display = DEFAULT_WEATHER_CITY
    wttr_query = f"{lat},{lon}"
    custom_loc_requested = False

    for th_key, (k_lat, k_lon, k_disp, k_wttr) in THAI_LOCATION_COORDS.items():
        if th_key in user_text:
            lat, lon, loc_display, wttr_query = k_lat, k_lon, k_disp, k_wttr
            custom_loc_requested = True
            break

    if not custom_loc_requested and channel_id and channel_id in channel_last_weather:
        cached_w = channel_last_weather[channel_id]
        if time.time() - cached_w.get("timestamp", 0) < 600 and cached_w.get("report"):
            api_log(f"Weather (Cached) -> {loc_display} (reused recent report)")
            return cached_w["report"]

    t0 = time.monotonic()
    try:
        om_url = clean_url(
            f"{HTTPS}api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
            "&hourly=temperature_2m,precipitation_probability,precipitation,weather_code"
            "&daily=precipitation_probability_max,temperature_2m_max,temperature_2m_min"
            "&timezone=Asia%2FBangkok&forecast_days=1"
        )
        session = get_http_session()
        async with session.get(om_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                om = await resp.json()
                api_log(f"Weather (Tier 1 Open-Meteo) -> {loc_display} ({lat},{lon}) 200 OK ({time.monotonic() - t0:.2f}s)")
                c = om.get("current") or {}
                d = om.get("daily") or {}
                h = om.get("hourly") or {}

                curr_code = int(c.get("weather_code", 0))
                curr_desc = WMO_WEATHER_TH.get(curr_code, "มีเมฆบางส่วน")
                rain_prob_max = (d.get("precipitation_probability_max") or [0])[0]
                t_max = (d.get("temperature_2m_max") or ["?"])[0]
                t_min = (d.get("temperature_2m_min") or ["?"])[0]

                h_times = h.get("time") or []
                h_temps = h.get("temperature_2m") or []
                h_probs = h.get("precipitation_probability") or []
                h_codes = h.get("weather_code") or []

                hourly_lines = []
                for idx in range(0, min(len(h_times), 24), 3):
                    time_label = h_times[idx].split("T")[-1] if "T" in str(h_times[idx]) else f"{idx:02d}:00"
                    temp_v = h_temps[idx] if idx < len(h_temps) else "?"
                    prob_v = h_probs[idx] if idx < len(h_probs) else 0
                    code_v = int(h_codes[idx]) if idx < len(h_codes) and h_codes[idx] is not None else 0
                    desc_v = WMO_WEATHER_TH.get(code_v, "ปกติ")
                    hourly_lines.append(
                        f"- เวลา {time_label} น.: {temp_v}°C | โอกาสฝนตก {prob_v}% ({desc_v})"
                    )

                report = (
                    f'<live_weather_report location="{loc_display}" coordinates="{lat},{lon}">\n'
                    f"พื้นที่พยากรณ์: {loc_display} (พิกัด {lat}, {lon})\n"
                    f"สภาพอากาศปัจจุบัน: {curr_desc} | อุณหภูมิ: {c.get('temperature_2m')}°C (รู้สึกเหมือน {c.get('apparent_temperature')}°C)\n"
                    f"ความชื้นสัมพัทธ์: {c.get('relative_humidity_2m')}% | ลม: {c.get('wind_speed_10m')} km/h | ปริมาณฝนขณะนี้: {c.get('precipitation')} มม.\n"
                    f"อุณหภูมิต่ำสุด-สูงสุดวันนี้: {t_min}°C ถึง {t_max}°C | โอกาสเกิดฝนตกสูงสุดของวันนี้: {rain_prob_max}%\n"
                    f"แนวโน้มรายชั่วโมงวันนี้ (เช้า/บ่าย/เย็น/กลางคืน):\n" + "\n".join(hourly_lines) + "\n"
                    f"</live_weather_report>"
                )
                if channel_id:
                    channel_last_weather[channel_id] = {"report": report, "timestamp": time.time()}
                return report
    except Exception as e:
        api_log(f"Weather (Tier 1 Open-Meteo) -> failed ({type(e).__name__}); falling back to wttr.in")

    t1 = time.monotonic()
    url = clean_url(f"{HTTPS}wttr.in/{wttr_query}?format=j1&lang=th")
    try:
        session = get_http_session()
        async with session.get(
            url, headers={"User-Agent": "curl/8.5.0"}, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            raw_text = await resp.text()
            if resp.status == 200 and raw_text.lstrip().startswith("{"):
                data = json.loads(raw_text)
                api_log(f"Weather (Tier 2 wttr.in) -> ({loc_display}) 200 OK ({time.monotonic() - t1:.2f}s)")
                curr = (data.get("current_condition") or [{}])[0]
                weather_days = data.get("weather") or [{}]
                today_w = weather_days[0] if weather_days else {}
                hourly = today_w.get("hourly") or []

                temp_c = curr.get("temp_C", "?")
                feels_c = curr.get("FeelsLikeC", "?")
                humidity = curr.get("humidity", "?")
                precip_mm = curr.get("precipMM", "0.0")
                lang_th = curr.get("lang_th") or curr.get("weatherDesc") or [{"value": "ไม่ระบุ"}]
                cond_desc = lang_th[0].get("value", "")

                rain_chances = [int(h.get("chanceofrain", 0)) for h in hourly if "chanceofrain" in h]
                max_rain_chance = max(rain_chances) if rain_chances else 0

                hourly_lines = []
                for h_item in hourly:
                    raw_tm = str(h_item.get("time", "0")).zfill(4)
                    hh_str = f"{raw_tm[:2]}:{raw_tm[2:]}"
                    h_temp = h_item.get("tempC", "?")
                    h_rain = h_item.get("chanceofrain", "0")
                    h_desc_list = h_item.get("lang_th") or h_item.get("weatherDesc") or [{"value": "ปกติ"}]
                    h_desc = h_desc_list[0].get("value", "ปกติ").strip()
                    hourly_lines.append(
                        f"- เวลา {hh_str} น.: {h_temp}°C | โอกาสฝนตก {h_rain}% ({h_desc})"
                    )

                report = (
                    f'<live_weather_report location="{loc_display}">\n'
                    f"สภาพอากาศปัจจุบัน ({loc_display}): {cond_desc}\n"
                    f"อุณหภูมิปัจจุบัน: {temp_c}°C (รู้สึกเหมือน {feels_c}°C) | ความชื้น: {humidity}% | น้ำฝน: {precip_mm} มม.\n"
                    f"โอกาสเกิดฝนตกสูงสุดของวันนี้: {max_rain_chance}%\n"
                    f"แนวโน้มรายชั่วโมงวันนี้ (เช้า/บ่าย/เย็น/กลางคืน):\n" + "\n".join(hourly_lines) + "\n"
                    f"</live_weather_report>"
                )
                if channel_id:
                    channel_last_weather[channel_id] = {"report": report, "timestamp": time.time()}
                return report
    except Exception as e:
        api_log(f"Weather (Tier 2 wttr.in) -> failed ({type(e).__name__})")
    return ""


# ==============================================================================
# SSRF-PROTECTED PUBLIC URL VALIDATION & ASYNC WEBPAGE SCRAPER
# ==============================================================================
def _check_resolved_addrs_safe(addr_info: list):
    for entry in addr_info:
        ip_obj = ipaddress.ip_address(entry[4][0])
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped
        if (
            not ip_obj.is_global or ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
            or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified
            or (isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in CGNAT_NETWORK)
        ):
            raise ValueError(f"SSRF Protection: Blocked non-public IP `{ip_obj}`.")


def _validate_url_syntax_and_host(url: str) -> tuple[str, str, int]:
    cleaned = clean_url(url)
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked unsafe URL scheme `{parsed.scheme}`.")
    hostname = parsed.hostname
    if not hostname or hostname.lower() in (
        "localhost", "localhost.localdomain", "metadata.google.internal", "169.254.169.254"
    ):
        raise ValueError(f"SSRF Protection: Blocked internal host `{hostname}`.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return cleaned, hostname, port


def validate_safe_public_url(url: str) -> str:
    cleaned, hostname, port = _validate_url_syntax_and_host(url)
    try:
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except Exception as e:
        raise ValueError(f"DNS resolution failed for `{hostname}`: {e}")
    _check_resolved_addrs_safe(addr_info)
    return cleaned


async def validate_safe_public_url_async(url: str) -> str:
    cleaned, hostname, port = _validate_url_syntax_and_host(url)
    loop = asyncio.get_running_loop()
    try:
        addr_info = await loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except Exception as e:
        raise ValueError(f"DNS resolution failed for `{hostname}`: {e}")
    _check_resolved_addrs_safe(addr_info)
    return cleaned


def fetch_webpage_sync(url: str, max_chars: int = 14000) -> str:
    safe_url = validate_safe_public_url(url)
    downloaded = trafilatura.fetch_url(safe_url)
    if not downloaded:
        raise RuntimeError("Could not download URL content.")
    extracted = trafilatura.extract(
        downloaded, include_tables=True, include_comments=False, favor_precision=True
    )
    if not extracted:
        raise RuntimeError("No readable article text found on page.")
    return extracted[:max_chars]


async def fetch_webpage_async(url: str, max_chars: int = 14000, timeout_sec: float = 6.0) -> str:
    """Async SSRF-validated webpage fetcher using pooled aiohttp + trafilatura HTML extraction."""
    safe_url = await validate_safe_public_url_async(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    session = get_http_session()
    async with session.get(
        safe_url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_sec)
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} downloading URL.")
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if ctype and not any(k in ctype for k in ("text/", "xml", "json", "xhtml")):
            raise RuntimeError(f"Non-text content type `{ctype}`.")
        # Cap downloaded HTML at 1.5 MB to protect 512MB RAM
        raw_chunks = bytearray()
        async for chunk in resp.content.iter_chunked(65536):
            raw_chunks.extend(chunk)
            if len(raw_chunks) >= 1536 * 1024:
                break
        html_text = raw_chunks.decode("utf-8", errors="replace")

    extracted = await asyncio.to_thread(
        trafilatura.extract,
        html_text,
        include_tables=True,
        include_comments=False,
        favor_precision=True,
    )
    if not extracted:
        raise RuntimeError("No readable article text found on page.")
    return extracted[:max_chars]


# ==============================================================================
# PARALLEL ASYNC MULTI-ENGINE SEARCH BACKENDS
# ==============================================================================
async def _search_google_ai_studio_grounded_async(query: str) -> list[dict]:
    """Fast async Gemini Google Search Grounding (auto-throttled for 30m on HTTP 429)."""
    if not GEMINI_API_KEY or time.time() < state.gemini_grounding_cooldown_until:
        return []
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"Search Google for the latest accurate facts about: {query}\n"
                            "Return a concise factual summary with concrete dates, numbers, and names."
                        )
                    }
                ]
            }
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    session = get_http_session()
    for model_id in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        t0 = time.monotonic()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    data = await resp.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        continue
                    cand = candidates[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    summary_text = "\n".join(
                        p.get("text", "") for p in parts if p.get("text") and not p.get("thought", False)
                    ).strip()
                    chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
                    src_urls = [
                        (ch.get("web") or {}).get("uri", "")
                        for ch in chunks[:4]
                        if (ch.get("web") or {}).get("uri")
                    ]
                    if summary_text:
                        api_log(f"Search (Gemini Grounding) -> {model_id} 200 OK ({elapsed:.2f}s)")
                        return [{
                            "title": "[Google Search Grounded Summary]",
                            "href": src_urls[0] if src_urls else "",
                            "body": summary_text[:2800] + (
                                ("\nVerified Sources: " + " | ".join(src_urls)) if src_urls else ""
                            ),
                            "skip_scrape": True,
                        }]
                elif resp.status == 429:
                    state.gemini_grounding_cooldown_until = time.time() + 1800.0
                    api_log(f"Search (Gemini Grounding) -> {model_id} HTTP 429 (cooldown 30m)")
                    return []
        except Exception as e:
            api_log(f"Search (Gemini Grounding) -> {model_id} skipped ({type(e).__name__})")
    return []


async def _search_google_news_rss_async(query: str, is_thai: bool, max_results: int = 4) -> list[dict]:
    hl = "th" if is_thai else "en-US"
    gl = "TH" if is_thai else "US"
    ceid = "TH:th" if is_thai else "US:en"
    rss_url = clean_url(f"{HTTPS}news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}")
    results = []
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.get(
            rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            if resp.status != 200:
                return []
            xml_bytes = await resp.read()
        root = ET.fromstring(xml_bytes)
        for item in root.findall("./channel/item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            results.append({
                "title": f"[{source}] {title}" if source else title,
                "href": link,
                "body": f"สำนักข่าว (Source): {source or 'Unknown'} | วันที่เผยแพร่: {pub_date} | พาดหัวข่าว: {title}",
                "skip_scrape": True,
            })
        if results:
            api_log(f"Search (Google News RSS) -> {len(results)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        api_log(f"Search (Google News RSS) -> failed ({type(e).__name__})")
    return results


async def _search_wikipedia_full_extract_async(query: str, is_thai: bool, max_results: int = 2) -> list[dict]:
    has_en_terms = bool(ENGLISH_ENTITY_REGEX.search(query))
    lang_hosts = ["en", "th"] if has_en_terms else (["th", "en"] if is_thai else ["en"])
    session = get_http_session()
    headers = {"User-Agent": "YukinonBot/2.0"}
    hits = []

    for lang_host in lang_hosts:
        if len(hits) >= max_results:
            break
        t0 = time.monotonic()
        q_for_wiki = query
        if lang_host == "en" and is_thai and has_en_terms:
            en_parts = [
                m.strip() for m in ENGLISH_ENTITY_REGEX.findall(query)
                if _is_valid_entity_phrase(m)
            ]
            if en_parts:
                q_for_wiki = " ".join(en_parts[:4])

        try:
            search_url = clean_url(
                f"{HTTPS}{lang_host}.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={quote_plus(q_for_wiki)}&utf8=&format=json&srlimit={max_results}"
            )
            async with session.get(
                search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json()
            search_items = ((data.get("query") or {}).get("search")) or []
            titles = [item.get("title", "") for item in search_items[:max_results] if item.get("title")]
            if not titles:
                continue

            # Batch-fetch all matched Wikipedia page extracts in ONE HTTP request (instead of N serial requests!)
            titles_param = "|".join(titles)
            ext_url = clean_url(
                f"{HTTPS}{lang_host}.wikipedia.org/w/api.php"
                f"?action=query&prop=extracts&explaintext=1&exchars=3800"
                f"&titles={quote_plus(titles_param)}&format=json"
            )
            async with session.get(
                ext_url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as e_resp:
                if e_resp.status != 200:
                    continue
                e_data = await e_resp.json()
            pages = ((e_data.get("query") or {}).get("pages")) or {}
            for _, p_obj in pages.items():
                title = p_obj.get("title") or ""
                extract = (p_obj.get("extract") or "").strip()
                if title and extract:
                    page_url = f"{HTTPS}{lang_host}.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                    hits.append({
                        "title": f"[Wikipedia {lang_host.upper()}] {title}",
                        "href": page_url,
                        "body": extract[:3800],
                        "skip_scrape": True,
                    })
            if hits:
                api_log(f"Search (Wikipedia:{lang_host} q='{q_for_wiki[:35]}') -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
        except Exception:
            pass
    return hits[:max_results]


async def _search_jikan_anime_manga_async(query: str, max_results: int = 2) -> list[dict]:
    clean_topic = re.sub(
        r"(อนิเมะ|มังงะ|ไลท์โนเวล|นิยาย|เรื่องย่อ|ตัวละคร|คืออะไร|สนุกไหม|คิดยังไงกับ|เรื่อง)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    if len(clean_topic) < 2:
        clean_topic = query
    is_manga = bool(re.search(r"(มังงะ|ไลท์โนเวล|นิยาย|\bmanga\b|\bnovel\b)", query, re.IGNORECASE))
    media_type = "manga" if is_manga else "anime"
    url = clean_url(f"{HTTPS}api.jikan.moe/v4/{media_type}?q={quote_plus(clean_topic)}&limit={max_results}")
    t0 = time.monotonic()
    hits = []
    try:
        session = get_http_session()
        async with session.get(
            url, headers={"User-Agent": "YukinonBot/2.0"}, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        for item in (data.get("data") or [])[:max_results]:
            title = item.get("title") or "Unknown"
            title_jp = item.get("title_japanese") or ""
            score = item.get("score") or "N/A"
            status = item.get("status") or "Unknown"
            episodes_or_ch = item.get("episodes") or item.get("chapters") or "?"
            synopsis = (item.get("synopsis") or "").strip()[:1200]
            mal_url = item.get("url") or ""
            hits.append({
                "title": f"[MyAnimeList {media_type.upper()}] {title} ({title_jp})",
                "href": mal_url,
                "body": (
                    f"Type: {item.get('type')} | Score: {score}/10 | Status: {status} | "
                    f"Episodes/Chapters: {episodes_or_ch}\nSynopsis: {synopsis}"
                ),
                "skip_scrape": True,
            })
        if hits:
            api_log(f"Search (Jikan MAL:{media_type}) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception:
        pass
    return hits


async def _search_steam_games_async(query: str, max_results: int = 2) -> list[dict]:
    clean_game = re.sub(
        r"(เกม|สตีม|ราคาเกม|สเปคเกม|สนุกไหม|คิดยังไงกับ|คือเกมอะไร|\bgame\b|\bsteam\b|\bwiki\b|\bclasses\b|\bweapons\b|\blist\b)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    if len(clean_game) < 2:
        return []
    url = clean_url(
        f"{HTTPS}store.steampowered.com/api/storesearch/?term={quote_plus(clean_game[:60])}&l=thai&cc=TH"
    )
    t0 = time.monotonic()
    hits = []
    try:
        session = get_http_session()
        async with session.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
        for item in (data.get("items") or [])[:max_results]:
            gid = item.get("id")
            name = item.get("name") or "Unknown Game"
            price_info = item.get("price") or {}
            final_thb = (price_info.get("final", 0) / 100.0) if price_info else 0
            price_str = f"{final_thb:,.2f} THB" if final_thb > 0 else "Free-to-Play / TBA"
            store_url = f"{HTTPS}store.steampowered.com/app/{gid}" if gid else ""
            hits.append({
                "title": f"[Steam Store TH] {name}",
                "href": store_url,
                "body": f"Game: {name} | Steam Thailand Price: {price_str} | AppID: {gid}",
                "skip_scrape": True,
            })
        if hits:
            api_log(f"Search (Steam Store) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception:
        pass
    return hits


async def _search_tavily_async(query: str, max_results: int = 5) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    t0 = time.monotonic()
    hits = []
    try:
        session = get_http_session()
        async with session.post(
            clean_url(f"{HTTPS}api.tavily.com/search"),
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
            timeout=aiohttp.ClientTimeout(total=7),
        ) as resp:
            if resp.status != 200:
                return []
            t_data = await resp.json()
        for r in t_data.get("results", []):
            hits.append({
                "title": r.get("title", ""),
                "href": r.get("url", ""),
                "body": r.get("content", ""),
            })
        if hits:
            api_log(f"Search (Tavily) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        api_log(f"Search (Tavily) -> failed ({type(e).__name__})")
    return hits


def _search_ddgs_sync(query: str, region_code: str, timelimit: str | None, max_results: int = 5) -> list[dict]:
    t0 = time.monotonic()
    hits = []
    try:
        with DDGS(proxy=DDGS_PROXY, timeout=8) as ddgs:
            timelimit_passes = [timelimit, None] if timelimit else [None]
            for tl in timelimit_passes:
                if hits:
                    break
                for backend_choice in ("auto", "brave", "bing", "google", "wikipedia", "yahoo"):
                    try:
                        ddg_hits = list(
                            ddgs.text(
                                query,
                                region=region_code,
                                safesearch="moderate",
                                timelimit=tl,
                                backend=backend_choice,
                                max_results=max_results,
                            )
                        )
                        if ddg_hits:
                            hits.extend(ddg_hits)
                            api_log(
                                f"Search (DDGS:{backend_choice} tl={tl} q='{query[:45]}') -> "
                                f"{len(ddg_hits)} hits ({time.monotonic() - t0:.2f}s)"
                            )
                            break
                    except Exception:
                        continue
    except Exception as e:
        api_log(f"Search (DDGS init) -> {type(e).__name__}")
    return hits


async def run_safe_web_search_async(query: str, max_results: int = 5) -> str:
    """
    High-speed parallel async multi-engine web search with 5-minute TTL caching + Phase 4 Decision-1 denoising:
      - Fans out domain-appropriate backends (Gemini Grounding, Google News RSS, Jikan MAL,
        Wikipedia batched extracts, Steam Store, Tavily, DDGS) concurrently via `asyncio.gather`.
      - Filters off-topic & clickbait/rumor snippets in ONE Decision-1 forward pass (~250ms)
        BEFORE deep-scraping, keeping only the cleanest, highest-relevance evidence for Gemma/Flash-Lite.
      - Deep-scrapes up to 2 surviving short-snippet web pages concurrently via `fetch_webpage_async`.
    """
    clean_q = re.sub(r"[\r\n]+", " ", query or "").strip()
    clean_q = re.sub(
        r"^(สวัสดี|ยูกิโนะ|ยูกิน่อน|ยูกิพีเดีย|yukipedia|ช่วยหา|ค้นหา|ช่วยบอก|อยากรู้ว่า|สรุป)\s*",
        "",
        clean_q,
        flags=re.IGNORECASE,
    ).strip()[:180]
    if not clean_q:
        return ""

    cache_key = f"{clean_q.lower()}|{max_results}"
    cached = _get_cached_search(cache_key)
    if cached:
        api_log(f"Search (Cache Hit) -> q='{clean_q[:45]}'")
        return cached

    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_year_ce = str(now_th.year)
    current_date_iso = now_th.strftime("%Y-%m-%d")

    is_thai = bool(THAI_CHAR_REGEX.search(clean_q))
    has_en_entity = bool(ENGLISH_ENTITY_REGEX.search(clean_q))
    region_code = "wt-wt" if has_en_entity else ("th-th" if is_thai else "wt-wt")

    is_news = bool(NEWS_QUERY_REGEX.search(clean_q))
    is_recency = bool(RECENCY_INTENT_REGEX.search(clean_q))
    is_anime_manga = bool(ANIME_MANGA_QUERY_REGEX.search(clean_q))
    is_game = bool(GAME_QUERY_REGEX.search(clean_q))

    dated_q = clean_q
    if is_news and not re.search(r"\b202\d\b|\b256\d\b", clean_q):
        dated_q = f"{clean_q} {current_year_ce}".strip()

    # Phase 1: Parallel fan-out across fast domain-matched async APIs
    phase1_tasks = []
    if is_news or is_recency:
        phase1_tasks.append(_search_google_ai_studio_grounded_async(dated_q))
    if is_news:
        phase1_tasks.append(_search_google_news_rss_async(dated_q, is_thai=is_thai, max_results=3))
    if is_anime_manga:
        phase1_tasks.append(_search_jikan_anime_manga_async(clean_q, max_results=2))
        phase1_tasks.append(_search_wikipedia_full_extract_async(clean_q, is_thai=is_thai, max_results=2))
    elif is_game:
        phase1_tasks.append(_search_wikipedia_full_extract_async(clean_q, is_thai=is_thai, max_results=2))
        phase1_tasks.append(_search_steam_games_async(clean_q, max_results=1))
    elif not is_news:
        phase1_tasks.append(_search_wikipedia_full_extract_async(clean_q, is_thai=is_thai, max_results=2))

    if TAVILY_API_KEY:
        phase1_tasks.append(_search_tavily_async(dated_q if is_news else clean_q, max_results=max_results))

    # Launch DDGS concurrently whenever we don't have Tavily or need broader web snippets
    preferred_tl = "m" if (is_news and not is_game and not is_anime_manga) else None
    phase1_tasks.append(
        asyncio.to_thread(_search_ddgs_sync, dated_q if is_news else clean_q, region_code, preferred_tl, max_results)
    )

    batches = await asyncio.gather(*phase1_tasks, return_exceptions=True)
    hits: list[dict] = []
    for batch in batches:
        if isinstance(batch, list):
            hits.extend(batch)

    # Last-resort fallback if everything above returned empty
    if not hits:
        fallback_batches = await asyncio.gather(
            _search_wikipedia_full_extract_async(clean_q, is_thai=is_thai, max_results=2),
            _search_google_news_rss_async(clean_q, is_thai=is_thai, max_results=max_results),
            return_exceptions=True,
        )
        for batch in fallback_batches:
            if isinstance(batch, list):
                hits.extend(batch)

    if not hits:
        return ""

    # Deduplicate up to 8 candidates so Decision-1 can rank and drop clickbait/irrelevant hits BEFORE scraping
    deduped_candidates: list[dict] = []
    seen_urls: set[str] = set()
    for item in hits:
        if len(deduped_candidates) >= max(8, max_results):
            break
        href = (item.get("href") or item.get("url") or "").strip()
        if href and href in seen_urls:
            continue
        if href:
            seen_urls.add(href)
        deduped_candidates.append(item)

    # Phase 4: Single-pass Decision-1 Snippet Relevance & Clickbait Filter (lazy import avoids circular import)
    if len(deduped_candidates) >= 2 and GEMINI_API_KEY:
        try:
            from llm import filter_search_snippets_with_decision_1
            selected_items = await filter_search_snippets_with_decision_1(
                clean_q, deduped_candidates, max_keep=max_results
            )
        except Exception as e:
            api_log(f"SearchFilter -> skipped ({type(e).__name__})")
            selected_items = deduped_candidates[:max_results]
    else:
        selected_items = deduped_candidates[:max_results]

    # Phase 2: Concurrently deep-scrape up to 2 surviving short-snippet web articles
    scrape_targets: list[tuple[int, str]] = []
    for idx, item in enumerate(selected_items):
        if len(scrape_targets) >= 2:
            break
        if item.get("skip_scrape"):
            continue
        href = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or "").strip()
        if (
            href
            and len(snippet) < 1200
            and not any(skip in href for skip in ("news.google.com", "myanimelist.net", "steampowered.com", "youtube.com", "wikipedia.org"))
        ):
            scrape_targets.append((idx, href))

    scraped_excerpts: dict[int, str] = {}
    if scrape_targets:
        async def _safe_scrape(idx_i: int, target_u: str) -> tuple[int, str]:
            try:
                txt = await fetch_webpage_async(target_u, max_chars=3200, timeout_sec=5.0)
                return idx_i, txt
            except Exception:
                return idx_i, ""

        scrape_results = await asyncio.gather(*[_safe_scrape(i, u) for i, u in scrape_targets])
        for i, txt in scrape_results:
            if txt:
                scraped_excerpts[i] = txt

    formatted_entries = []
    for idx, item in enumerate(selected_items):
        title = (item.get("title") or "Untitled").strip()
        href = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or "").strip()
        entry_text = f"[{idx + 1}] Title: {title}\nURL: {href}\nSnippet: {snippet}"
        if idx in scraped_excerpts:
            entry_text += f"\nFull Page Excerpt:\n{scraped_excerpts[idx]}"
        formatted_entries.append(entry_text)

    joined_results = "\n\n".join(formatted_entries)[:9000]
    output_block = (
        f'<untrusted_web_search_results query="{clean_q}" as_of_date="{current_date_iso}">\n'
        f"[ข้อมูลหลักฐานจากการค้นหาเว็บ ณ วันที่ {current_date_iso} (ปี ค.ศ. {current_year_ce}): "
        f"จงใช้เฉพาะข้อเท็จจริงที่มีอยู่ในข้อมูลนี้เท่านั้น ห้ามแต่งชื่ออาวุธ ชื่อคลาส ชื่อบริษัท หรือเนื้อเรื่องที่ไม่มีในข้อมูลนี้ขึ้นมาเองเด็ดขาด!]\n"
        f"{joined_results}\n"
        f"</untrusted_web_search_results>"
    )
    _store_cached_search(cache_key, output_block)
    return output_block


def run_safe_web_search_sync(query: str, max_results: int = 5) -> str:
    """Backward-compatible synchronous wrapper around `run_safe_web_search_async`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_safe_web_search_async(query, max_results=max_results))
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run_safe_web_search_async(query, max_results=max_results)).result(timeout=20)