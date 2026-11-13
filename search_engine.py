import re
import json
import time
import socket
import ipaddress
import urllib.request
import xml.etree.ElementTree as ET
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
    WEATHER_INTENT_PATTERN,
    WEB_SEARCH_INTENT_PATTERN,
    THAI_CHAR_REGEX,
    NEWS_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    GAME_QUERY_REGEX,
    TAVILY_API_KEY,
    DDGS_PROXY,
    GEMINI_API_KEY,
    channel_last_attachment,
    state,
    clean_url,
    api_log,
    get_http_session,
    prune_stale_attachments,
)


def detect_coding_intent(text: str, attachments: list, channel_id: int = None) -> bool:
    for att in attachments:
        if att.filename.lower().endswith(CODE_AND_LOG_EXTENSIONS):
            return True
    if bool(CODE_INTENT_PATTERN.search(text)):
        return True
    prune_stale_attachments()
    if channel_id and channel_id in channel_last_attachment:
        if channel_last_attachment[channel_id].get("is_code"):
            return True
    return False


def detect_weather_intent(text: str) -> bool:
    return bool(WEATHER_INTENT_PATTERN.search(text))


def detect_web_search_intent(text: str) -> bool:
    return bool(WEB_SEARCH_INTENT_PATTERN.search(text))


async def fetch_live_weather_context(user_text: str) -> str:
    lat = DEFAULT_WEATHER_LAT
    lon = DEFAULT_WEATHER_LON
    loc_display = DEFAULT_WEATHER_CITY
    wttr_query = f"{lat},{lon}"

    for th_key, (k_lat, k_lon, k_disp, k_wttr) in THAI_LOCATION_COORDS.items():
        if th_key in user_text:
            lat, lon, loc_display, wttr_query = k_lat, k_lon, k_disp, k_wttr
            break

    # Tier 1: Open-Meteo High-Resolution Forecast (Fast ~0.3s, No Rate Limits, Exact GPS Coords)
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
        async with session.get(om_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
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

                return (
                    f'<live_weather_report location="{loc_display}" coordinates="{lat},{lon}">\n'
                    f"พื้นที่พยากรณ์: {loc_display} (พิกัด {lat}, {lon})\n"
                    f"สภาพอากาศปัจจุบัน: {curr_desc} | อุณหภูมิ: {c.get('temperature_2m')}°C (รู้สึกเหมือน {c.get('apparent_temperature')}°C)\n"
                    f"ความชื้นสัมพัทธ์: {c.get('relative_humidity_2m')}% | ลม: {c.get('wind_speed_10m')} km/h | ปริมาณฝนขณะนี้: {c.get('precipitation')} มม.\n"
                    f"อุณหภูมิต่ำสุด-สูงสุดวันนี้: {t_min}°C ถึง {t_max}°C | โอกาสเกิดฝนตกสูงสุดของวันนี้: {rain_prob_max}%\n"
                    f"แนวโน้มรายชั่วโมงวันนี้:\n" + "\n".join(hourly_lines) + "\n"
                    f"</live_weather_report>"
                )
    except Exception as e:
        api_log(f"Weather (Tier 1 Open-Meteo) -> failed ({type(e).__name__}); falling back to wttr.in")

    # Tier 2 Fallback: wttr.in (with text guard so it never throws ContentTypeError)
    t1 = time.monotonic()
    url = clean_url(f"{HTTPS}wttr.in/{wttr_query}?format=j1&lang=th")
    try:
        session = get_http_session()
        async with session.get(
            url, headers={"User-Agent": "curl/8.5.0"}, timeout=aiohttp.ClientTimeout(total=8)
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
                return (
                    f'<live_weather_report location="{loc_display}">\n'
                    f"สภาพอากาศปัจจุบัน ({loc_display}): {cond_desc}\n"
                    f"อุณหภูมิปัจจุบัน: {temp_c}°C (รู้สึกเหมือน {feels_c}°C) | ความชื้น: {humidity}% | น้ำฝน: {precip_mm} มม.\n"
                    f"โอกาสเกิดฝนตกสูงสุดของวันนี้: {max_rain_chance}%\n"
                    f"</live_weather_report>"
                )
    except Exception as e:
        api_log(f"Weather (Tier 2 wttr.in) -> failed ({type(e).__name__})")
    return ""


def validate_safe_public_url(url: str) -> str:
    cleaned = clean_url(url)
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked unsafe URL scheme `{parsed.scheme}`.")
    hostname = parsed.hostname
    if not hostname or hostname.lower() in (
        "localhost", "localhost.localdomain", "metadata.google.internal", "169.254.169.254"
    ):
        raise ValueError(f"SSRF Protection: Blocked internal host `{hostname}`.")
    try:
        addr_info = socket.getaddrinfo(
            hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP
        )
    except Exception as e:
        raise ValueError(f"DNS resolution failed for `{hostname}`: {e}")
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


def _search_google_ai_studio_grounded_sync(query: str) -> str:
    if not GEMINI_API_KEY or time.time() < state.gemini_grounding_cooldown_until:
        return ""
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"Search Google for the latest accurate information about: {query}\n"
                            "Provide a concise factual summary with dates, numbers, and key facts."
                        )
                    }
                ]
            }
        ],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1},
    }
    model_id = state.active_gemini_model or "gemini-3-flash-preview"
    url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.monotonic() - t0
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        summary_text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()

        chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
        sources = []
        for idx, ch in enumerate(chunks[:5], 1):
            web_info = ch.get("web") or {}
            uri = web_info.get("uri", "")
            title = web_info.get("title", "Source")
            if uri:
                sources.append(f"[{idx}] {title}: {uri}")

        if summary_text:
            api_log(f"Search (Gemini Grounding) -> {model_id} 200 OK ({elapsed:.2f}s)")
            sources_block = "\n".join(sources)
            return (
                f'<untrusted_web_search_results query="{query}" engine="google_search_grounding">\n'
                f"Verified Google Search Summary:\n{summary_text}\n\n"
                f"Sources:\n{sources_block}\n"
                f"</untrusted_web_search_results>"
            )
    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            state.gemini_grounding_cooldown_until = time.time() + 600
            api_log(f"Search (Gemini Grounding) -> {model_id} HTTP 429 (cooldown 10m)")
        else:
            api_log(f"Search (Gemini Grounding) -> {model_id} skipped ({err_str})")
    return ""


def _search_google_news_rss_sync(query: str, is_thai: bool, max_results: int = 5) -> list[dict]:
    hl = "th" if is_thai else "en-US"
    gl = "TH" if is_thai else "US"
    ceid = "TH:th" if is_thai else "US:en"
    rss_url = clean_url(f"{HTTPS}news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}")
    results = []
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()
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
            })
        if results:
            api_log(f"Search (Tier 1 Google News RSS) -> {len(results)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        api_log(f"Search (Google News RSS) -> failed ({type(e).__name__})")
    return results


def _search_wikipedia_summary_sync(query: str, is_thai: bool, max_results: int = 2) -> list[dict]:
    lang_host = "th" if is_thai else "en"
    t0 = time.monotonic()
    hits = []
    try:
        search_url = clean_url(
            f"{HTTPS}{lang_host}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote_plus(query)}&utf8=&format=json&srlimit={max_results}"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "YukinonBot/2.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        search_items = ((data.get("query") or {}).get("search")) or []
        for item in search_items[:max_results]:
            title = item.get("title", "")
            if not title:
                continue
            sum_url = clean_url(
                f"{HTTPS}{lang_host}.wikipedia.org/api/rest_v1/page/summary/{quote_plus(title.replace(' ', '_'))}"
            )
            req_s = urllib.request.Request(sum_url, headers={"User-Agent": "YukinonBot/2.0"})
            with urllib.request.urlopen(req_s, timeout=6) as s_resp:
                s_data = json.loads(s_resp.read().decode("utf-8"))
            extract = (s_data.get("extract") or "").strip()
            page_url = ((s_data.get("content_urls") or {}).get("desktop") or {}).get("page") or sum_url
            if extract:
                hits.append({
                    "title": f"[Wikipedia {lang_host.upper()}] {title}",
                    "href": page_url,
                    "body": extract[:1500],
                })
        if hits:
            api_log(f"Search (Wikipedia:{lang_host}) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception:
        pass
    return hits


def _search_jikan_anime_manga_sync(query: str, max_results: int = 2) -> list[dict]:
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
        req = urllib.request.Request(url, headers={"User-Agent": "YukinonBot/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in (data.get("data") or [])[:max_results]:
            title = item.get("title") or "Unknown"
            title_jp = item.get("title_japanese") or ""
            score = item.get("score") or "N/A"
            status = item.get("status") or "Unknown"
            episodes_or_ch = item.get("episodes") or item.get("chapters") or "?"
            synopsis = (item.get("synopsis") or "").strip()[:1000]
            mal_url = item.get("url") or ""
            hits.append({
                "title": f"[MyAnimeList {media_type.upper()}] {title} ({title_jp})",
                "href": mal_url,
                "body": (
                    f"Type: {item.get('type')} | Score: {score}/10 | Status: {status} | "
                    f"Episodes/Chapters: {episodes_or_ch}\nSynopsis: {synopsis}"
                ),
            })
        if hits:
            api_log(f"Search (Jikan MAL:{media_type}) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception:
        pass
    return hits


def _search_steam_games_sync(query: str, max_results: int = 2) -> list[dict]:
    clean_game = re.sub(
        r"(เกม|สตีม|ราคาเกม|สเปคเกม|สนุกไหม|คิดยังไงกับ|คือเกมอะไร|\bgame\b|\bsteam\b)",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    if len(clean_game) < 2:
        return []
    url = clean_url(
        f"{HTTPS}store.steampowered.com/api/storesearch/?term={quote_plus(clean_game)}&l=thai&cc=TH"
    )
    t0 = time.monotonic()
    hits = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=7) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in (data.get("items") or [])[:max_results]:
            gid = item.get("id")
            name = item.get("name") or "Unknown Game"
            price_info = item.get("price") or {}
            final_thb = (price_info.get("final", 0) / 100.0) if price_info else 0
            price_str = f"{final_thb:,.2f} THB" if final_thb > 0 else "Free / TBA"
            store_url = f"{HTTPS}store.steampowered.com/app/{gid}" if gid else ""
            hits.append({
                "title": f"[Steam Store TH] {name}",
                "href": store_url,
                "body": f"Game: {name} | Steam Thailand Price: {price_str} | AppID: {gid}",
            })
        if hits:
            api_log(f"Search (Steam Store) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception:
        pass
    return hits


def run_safe_web_search_sync(query: str, max_results: int = 5) -> str:
    clean_q = re.sub(r"[\r\n]+", " ", query).strip()
    clean_q = re.sub(
        r"^(สวัสดี|ยูกิโนะ|ยูกิน่อน|ยูกิพีเดีย|yukipedia|ช่วยหา|ค้นหา|ช่วยบอก|อยากรู้ว่า|สรุป)\s*",
        "",
        clean_q,
        flags=re.IGNORECASE,
    ).strip()[:200]
    if not clean_q:
        return ""

    is_thai = bool(THAI_CHAR_REGEX.search(clean_q))
    region_code = "th-th" if is_thai else "wt-wt"
    is_news = bool(NEWS_QUERY_REGEX.search(clean_q))
    is_anime_manga = bool(ANIME_MANGA_QUERY_REGEX.search(clean_q))
    is_game = bool(GAME_QUERY_REGEX.search(clean_q))

    hits = []

    # 1. Domain-Specific Fast APIs (0.3s - 0.5s, Zero Rate Limits)
    if is_news:
        hits.extend(_search_google_news_rss_sync(clean_q, is_thai=is_thai, max_results=max_results))

    if is_anime_manga:
        hits.extend(_search_jikan_anime_manga_sync(clean_q, max_results=2))
        hits.extend(_search_wikipedia_summary_sync(clean_q, is_thai=is_thai, max_results=2))

    if is_game:
        hits.extend(_search_steam_games_sync(clean_q, max_results=2))
        hits.extend(_search_wikipedia_summary_sync(clean_q, is_thai=is_thai, max_results=2))

    # 2. Tavily API (if configured)
    if len(hits) < 2 and TAVILY_API_KEY:
        t0 = time.monotonic()
        try:
            t_req = urllib.request.Request(
                clean_url(f"{HTTPS}api.tavily.com/search"),
                data=json.dumps({
                    "api_key": TAVILY_API_KEY,
                    "query": clean_q,
                    "max_results": max_results,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(t_req, timeout=10) as resp:
                t_data = json.loads(resp.read().decode("utf-8"))
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

    # 3. DDGS v9+ Multi-Backend Metasearch (brave, bing, google, wikipedia, yahoo, auto)
    if len(hits) < 3:
        t0 = time.monotonic()
        try:
            with DDGS(proxy=DDGS_PROXY, timeout=10) as ddgs:
                if is_news and not hits:
                    try:
                        news_hits = list(ddgs.news(clean_q, region=region_code, max_results=max_results))
                        if news_hits:
                            hits.extend(news_hits)
                            api_log(f"Search (DDGS:news) -> {len(news_hits)} hits ({time.monotonic() - t0:.2f}s)")
                    except Exception:
                        pass

                if len(hits) < 3:
                    for backend_choice in ("auto", "brave", "bing", "google", "wikipedia", "yahoo"):
                        try:
                            ddg_hits = list(
                                ddgs.text(
                                    clean_q,
                                    region=region_code,
                                    safesearch="moderate",
                                    backend=backend_choice,
                                    max_results=max_results,
                                )
                            )
                            if ddg_hits:
                                hits.extend(ddg_hits)
                                api_log(
                                    f"Search (DDGS:{backend_choice}) -> {len(ddg_hits)} hits ({time.monotonic() - t0:.2f}s)"
                                )
                                break
                        except Exception:
                            continue
        except Exception as e:
            api_log(f"Search (DDGS init) -> {type(e).__name__}")

    # 4. Wikipedia + Google News RSS Fallbacks for General Queries
    if not hits:
        hits.extend(_search_wikipedia_summary_sync(clean_q, is_thai=is_thai, max_results=2))
    if not hits:
        hits.extend(_search_google_news_rss_sync(clean_q, is_thai=is_thai, max_results=max_results))

    # 5. Last-Resort Fallback: Gemini Grounded Search
    if not hits and GEMINI_API_KEY:
        grounded_res = _search_google_ai_studio_grounded_sync(clean_q)
        if grounded_res:
            return grounded_res

    if not hits:
        return ""

    formatted_entries = []
    deep_scraped_count = 0
    for idx, item in enumerate(hits[:max_results], 1):
        title = (item.get("title") or "Untitled").strip()
        href = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or "").strip()
        entry_text = f"[{idx}] Title: {title}\nURL: {href}\nSnippet: {snippet}"
        if (
            href
            and deep_scraped_count < 1
            and not any(skip in href for skip in ("news.google.com", "wikipedia.org", "myanimelist.net", "steampowered.com"))
        ):
            try:
                article_excerpt = fetch_webpage_sync(href, max_chars=2200)
                if article_excerpt:
                    entry_text += f"\nVerified Page Excerpt:\n{article_excerpt}"
                    deep_scraped_count += 1
            except Exception:
                pass
        formatted_entries.append(entry_text)
    joined_results = "\n\n".join(formatted_entries)[:6000]
    return (
        f'<untrusted_web_search_results query="{clean_q}">\n'
        f"[หมายเหตุสำคัญสำหรับยูกิโนะ: จงคัดกรองเฉพาะข้อเท็จจริงจากแหล่งข่าวที่น่าเชื่อถือ หากพาดหัวใดดูเป็นข่าวลือคลิกเบตหรือข่าวปลอมจากโซเชียล ให้ตัดทิ้งหรือเตือนผู้ใช้ทันที]\n"
        f"{joined_results}\n"
        f"</untrusted_web_search_results>"
    )