import os
import re
import io
import json
import time
import asyncio
import threading
import discord

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import (
    THAI_WEEKDAYS,
    THAI_MONTHS,
    GAME_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    CODE_INTENT_PATTERN,
    STACK_TRACE_SIGNATURE_REGEX,
    CAT_TOPIC_REGEX,
    WEATHER_INTENT_PATTERN,
    OWNER_NAME_STRIP_REGEX,
    URL_REGEX,
    channel_active_topic,
    state,
    api_log,
)
from persona import is_owner_user, get_current_thai_datetime_str

USER_PROFILES_FILE = os.path.join(os.path.dirname(__file__), "user_profiles.json")
MEMORY_BACKUP_CHANNEL_ID = int(os.getenv("MEMORY_BACKUP_CHANNEL_ID", "0") or "0")

# In-memory store: { "<user_id_str>": { ...profile_data... } }
user_profiles: dict[str, dict] = {}
_DIRTY_SINCE_LAST_BACKUP = False
_LAST_CLOUD_BACKUP_TS = 0.0
_CLOUD_BACKUP_LOCK: asyncio.Lock | None = None
_LOCAL_SAVE_THREAD_LOCK = threading.Lock()
_CONSOLIDATION_COOLDOWN: dict[str, float] = {}
_CONSOLIDATION_IN_PROGRESS: set[str] = set()
_TOPIC_SUMMARIZE_COOLDOWN: dict[str, float] = {}
_TOPIC_SUMMARIZE_IN_PROGRESS: set[str] = set()

USER_MENTION_REGEX = re.compile(r"<@!?(\d+)>")
AGENTIC_MEMO_TAG_REGEX = re.compile(r"\[MEMO:\s*([^\]]+?)\s*\]", re.IGNORECASE)

TOPIC_KEYWORDS_MAP = {
    "PSO2 NGS": re.compile(r"\b(pso2|ngs|halpha|arks)\b", re.IGNORECASE),
    "Gaming / Steam": GAME_QUERY_REGEX,
    "Anime & Manga": ANIME_MANGA_QUERY_REGEX,
    "Software & Coding": CODE_INTENT_PATTERN,
    "Cats & Pan-san": CAT_TOPIC_REGEX,
    "Weather Forecast": WEATHER_INTENT_PATTERN,
}

_USER_TAG_REGEX = re.compile(r"^user_\d+$", re.IGNORECASE)
_FAV_TOPIC_NOISE_TOKENS = frozenset({
    "partner", "owner", "partner (owner)", "member", "yukino", "yukinon", "yuki",
    "bot", "discord", "ot", "unknown", "user",
    "row", "col", "column", "header", "cell", "table", "error", "undefined", "null", "test",
})

# ==============================================================================
# MEMORY HYGIENE FILTERS (BLOCKS CODE/LOGS/TABLES/INTERNAL PROMPTS FROM POLLUTING MEMORY)
# ==============================================================================
_TABLE_OR_BOX_ART_REGEX = re.compile(
    r"(?:\+[-=]{3,}\+|\|\s*[-:]{2,}\s*\||[┏━┿┃│]{3,}|"
    r"\|\s*Header\s*\d|\bRow\s*\d+\s*Col\s*\d+|\bHeader\s*1\b.*\bHeader\s*2\b)",
    re.IGNORECASE,
)
_CODE_SYNTAX_OR_ERROR_REGEX = re.compile(
    r"(?:```|\b(?:TypeError|ReferenceError|SyntaxError|RangeError|NullPointerException|"
    r"IllegalArgumentException|IndexError|KeyError|ValueError|AttributeError|ImportError|"
    r"ORA-\d{4,5}|TNS-\d{4,5}|Traceback\s*\(most recent|CrashLoopBackOff|OOMKilled)\b|"
    r"can't\s+access\s+property|cannot\s+read\s+propert|\bis\s+undefined\b|\bis\s+not\s+a\s+function\b|"
    r"\.(?:tsx?|jsx?|py|java|sql|log|json|ya?ml|xml|css|html|sh|ps1|min\.js):\d+|"
    r"\b(?:const|let|var|function|return|import|export|async|await|interface|class)\s+[A-Za-z_$]|"
    r"=>|===|!==|\bscheduler\.production\.min\.js\b|\bwebpack\s+\d+\b)",
    re.IGNORECASE,
)
_DOTTED_CODE_IDENT_REGEX = re.compile(r"\b[a-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
_FILE_EXT_OR_LINE_REF_REGEX = re.compile(
    r"(?:\.(?:tsx?|jsx?|py|java|class|kt|go|rs|cpp|cs|php|rb|sql|pls|pkb|pks|log|json|ya?ml|xml|css|scss|html|sh|ps1|bat)\b|:\d+\b)",
    re.IGNORECASE,
)
_PLACEHOLDER_TABLE_TOKEN_REGEX = re.compile(
    r"^(?:(?:row|col|column|header|cell|field|item|test|sample|example)\s*\d+(?:\s+(?:row|col|column|header|cell|field|item)\s*\d+)*)$",
    re.IGNORECASE,
)
_INTERNAL_BOT_OR_BUTTON_PROMPT_REGEX = re.compile(
    r"(?:\(ผู้ใช้ส่งลิงก์นี้มาให้|ช่วยเจาะลึกองค์ประกอบภาพ|ช่วยค้นหาข้อมูลเพิ่มเติมเกี่ยวกับศิลปิน|"
    r"ช่วยอธิบายเจาะลึกเฉพาะความสัมพันธ์ของตัวละคร|ช่วยสร้าง\s*\*\*Code\s+Map|"
    r"ช่วย\s*\*\*อธิบายการทำงานและสาเหตุ|ช่วย\s*\*\*ปรับปรุงประสิทธิภาพและโครงสร้าง|"
    r"ช่วยวิเคราะห์ข้อดี\s+ข้อเสีย\s+ความคุ้มค่า|ช่วยสรุปใจความสำคัญและข้อสรุปหลัก|"
    r"ช่วยสรุปขั้นตอนที่ต้องทำ|จากสินค้า\s+โมเดล\s+หรือฟิกเกอร์ที่ระบุไว้|"
    r"ช่วยเล่าเหตุการณ์และบทสนทนาที่เห็นในภาพ|ช่วยตรวจสอบ\s+วิเคราะห์จุดบกพร่องหรือหา\s+Root\s+Cause|"
    r"ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด|\[กฎสำคัญ\]|\[บริบทหัวข้อก่อนหน้า\]|\[SEARCH_HINT:|\[สรุปคำตอบก่อนหน้า)",
    re.IGNORECASE,
)
_BARE_X_URL_EXTRACT_REGEX = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:x\.com|twitter\.com|fixupx\.com|fxtwitter\.com|vxtwitter\.com)/([A-Za-z0-9_]{2,30})/status/\d+",
    re.IGNORECASE,
)
_BARE_PIXIV_URL_EXTRACT_REGEX = re.compile(
    r"https?://(?:www\.)?(?:pixiv\.net|phixiv\.net|ppxiv\.net)/(?:[a-z]{2}/)?(?:artworks/|i/)(\d+)",
    re.IGNORECASE,
)
_BARE_BSKY_URL_EXTRACT_REGEX = re.compile(
    r"https?://(?:www\.)?(?:bsky\.app|fxbsky\.app)/profile/([^/\s]+)/post/[a-zA-Z0-9]+",
    re.IGNORECASE,
)
_BARE_YT_URL_EXTRACT_REGEX = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/\S+",
    re.IGNORECASE,
)
_ERROR_TYPE_EXTRACT_REGEX = re.compile(
    r"\b(TypeError|ReferenceError|SyntaxError|RangeError|NullPointerException|"
    r"IllegalArgumentException|KeyError|ValueError|AttributeError|ImportError|IndexError|"
    r"ORA-\d{4,5}|TNS-\d{4,5}|HTTP\s*[45]\d{2}|CrashLoopBackOff|OOMKilled)\b",
    re.IGNORECASE,
)
_FILENAME_EXTRACT_REGEX = re.compile(
    r"\b([A-Za-z0-9_.-]+\.(?:tsx?|jsx?|py|java|sql|ya?ml|xml|json|log|sh|ps1|css|html))\b",
    re.IGNORECASE,
)


def _looks_like_code_or_error_or_table(text: str) -> bool:
    """True when text contains raw stack traces, error logs, code syntax, or ASCII/Markdown tables."""
    if not text:
        return False
    s = str(text)
    if _TABLE_OR_BOX_ART_REGEX.search(s):
        return True
    if _CODE_SYNTAX_OR_ERROR_REGEX.search(s):
        return True
    if STACK_TRACE_SIGNATURE_REGEX.search(s):
        return True
    if s.count("|") >= 3 and (re.search(r"\|\s*[-:]{2,}\s*\|", s) or s.count("\n") >= 1 or "+---" in s):
        return True
    # High programming punctuation density (`{ } ; ( )` combined with `=` or `.`)
    if len(s) >= 20 and sum(s.count(ch) for ch in ("{", "}", ";")) >= 2:
        return True
    return False


def _looks_like_internal_bot_or_button_prompt(text: str) -> bool:
    """True when text is an internal bot wrapper prompt or UI follow-up button instruction."""
    if not text:
        return False
    return bool(_INTERNAL_BOT_OR_BUTTON_PROMPT_REGEX.search(str(text)))


def _is_blocked_favorite_topic(topic: str, user=None, profile: dict | None = None) -> bool:
    """
    Blocks Discord usernames, display names, owner names, @handles, URLs,
    code/stack-trace snippets, filenames/line refs, table placeholders (`Row 1 Col 1`),
    and internal bot prompts from polluting `favorite_topics`.
    """
    raw = str(topic or "")
    if "\n" in raw or "\r" in raw:
        return True
    t = re.sub(r"\s+", " ", raw).strip()
    if len(t) < 2 or len(t) > 50:
        return True
    tl = t.lower()
    if tl in _FAV_TOPIC_NOISE_TOKENS or tl.startswith(("@", "http://", "https://", "<@", "```", "|", "+-")):
        return True
    if _USER_TAG_REGEX.match(tl):
        return True
    if OWNER_NAME_STRIP_REGEX.search(t):
        return True
    if _PLACEHOLDER_TABLE_TOKEN_REGEX.match(t):
        return True
    if _FILE_EXT_OR_LINE_REF_REGEX.search(t):
        return True
    if _DOTTED_CODE_IDENT_REGEX.search(t):
        return True
    if _looks_like_code_or_error_or_table(t) or _looks_like_internal_bot_or_button_prompt(t):
        return True
    if any(ch in t for ch in ("`", "|", "{", "}", "[", "]", ";", "=")):
        return True

    identity_names: set[str] = set()
    if user is not None:
        for attr in ("name", "display_name", "global_name"):
            val = str(getattr(user, attr, "") or "").strip().lower()
            if len(val) >= 3:
                identity_names.add(val)
    if isinstance(profile, dict):
        for key in ("username", "display_name"):
            val = str(profile.get(key) or "").strip().lower()
            if len(val) >= 3 and val != "partner (owner)":
                identity_names.add(val)

    for id_n in identity_names:
        if tl == id_n or re.search(rf"(?<!\w){re.escape(id_n)}(?!\w)", tl):
            return True
    return False


def _is_valid_custom_note(note: str) -> bool:
    """
    Validates rolling `custom_notes` and auto-saved facts:
    Blocks raw code/stack traces/tables, URLs, internal bot/button prompts, and empty/trivial noise.
    """
    s = re.sub(r"\s+", " ", str(note or "")).strip()
    if len(s) < 5 or len(s) > 150:
        return False
    if _looks_like_code_or_error_or_table(s) or _looks_like_internal_bot_or_button_prompt(s):
        return False
    if _PLACEHOLDER_TABLE_TOKEN_REGEX.match(s):
        return False
    # Reject bare URLs or strings that are mostly a link
    if URL_REGEX.search(s) and len(URL_REGEX.sub("", s).strip()) < 10:
        return False
    return True


def _is_valid_recent_topic(topic: str) -> bool:
    """
    Validates entries in `recent_topics` (both newly recorded and during startup scrubbing):
    Rejects raw code blocks, ASCII/Markdown tables, raw error traces, raw social URLs,
    internal `(ผู้ใช้ส่งลิงก์นี้มาให้...)` wrappers, and internal UI button instructions.
    """
    s = re.sub(r"\s+", " ", str(topic or "")).strip()
    if len(s) < 4 or len(s) > 120:
        return False
    if _looks_like_internal_bot_or_button_prompt(s):
        return False
    if _TABLE_OR_BOX_ART_REGEX.search(s) or "```" in s:
        return False
    if _PLACEHOLDER_TABLE_TOKEN_REGEX.match(s):
        return False
    # Reject raw un-summarized URLs (e.g. `https://x.com/hibikamin/status/...`)
    if re.search(r"https?://\S+", s):
        return False
    # Reject raw un-summarized stack traces / property access errors
    if re.search(
        r"(?:can't\s+access\s+property|cannot\s+read\s+propert|Error\s+fetc?hing|Traceback\s*\(|"
        r"scheduler\.production\.min\.js|webpack\s+\d+|:\d+:\d+|e\.[a-zA-Z_]+\s+is\s+undefined)",
        s,
        re.IGNORECASE,
    ):
        return False
    # Reject raw code lines with braces/semicolons/arrows
    if sum(s.count(ch) for ch in ("{", "}", ";")) >= 2 or "=>" in s:
        return False
    return True


def _condense_recent_topic_sync(
    user_prompt: str,
    is_coding: bool = False,
    is_button_followup: bool = False,
) -> tuple[str, bool]:
    """
    Deterministically cleans and summarizes a user turn into a crisp `recent_topics` label
    (zero API latency). Returns `(condensed_topic, wants_bg_llm_summary)`.
    - Returns `("", False)` to skip noise (UI button clicks, pure test tables, empty/tiny text).
    - Summarizes raw stack traces/code dumps into crisp labels like `"ดีบัก TypeError (FormSearch.tsx)"`.
    - Summarizes bare X/Pixiv/Bluesky/YouTube links into clean labels like `"ดูโพสต์ X ของ @hibikamin"`.
    """
    if is_button_followup or not user_prompt:
        return "", False

    raw = str(user_prompt).strip()
    if not raw:
        return "", False

    # Strip internal bot prompt wrapper `(ผู้ใช้ส่งลิงก์นี้มาให้...)` BEFORE checking button prompts
    had_bare_link_wrapper = bool(re.search(r"\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*$", raw))
    stripped_wrapper = re.sub(r"\n*\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*$", "", raw).strip()
    stripped_wrapper = re.sub(r"^\[ผู้พูด:[^\]]+\]\s*", "", stripped_wrapper).strip()
    stripped_wrapper = re.sub(r"<@!?\d+>", "", stripped_wrapper).strip()

    if _looks_like_internal_bot_or_button_prompt(stripped_wrapper):
        return "", False

    # 1. Pure ASCII/Markdown test table -> skip completely (unrelated noise!)
    if _TABLE_OR_BOX_ART_REGEX.search(stripped_wrapper):
        no_table_text = re.sub(r"```[\s\S]*?```", " ", stripped_wrapper)
        no_table_text = "\n".join(
            ln for ln in no_table_text.splitlines()
            if not re.match(r"^\s*(?:[+|┃┏┗┠]|[|\s:-]{4,}|Row\s*\d+|Header\s*\d+)", ln, re.IGNORECASE)
        )
        no_table_text = re.sub(r"\s+", " ", no_table_text).strip()
        if len(no_table_text) < 12 or _PLACEHOLDER_TABLE_TOKEN_REGEX.match(no_table_text):
            return "", False
        stripped_wrapper = no_table_text

    # 2. Bare social/media link (X, Pixiv, Bluesky, YouTube) -> clean readable label + optional BG refine
    text_no_urls = URL_REGEX.sub(" ", stripped_wrapper).strip()
    if URL_REGEX.search(stripped_wrapper) and (had_bare_link_wrapper or len(text_no_urls) < 8):
        m_x = _BARE_X_URL_EXTRACT_REGEX.search(stripped_wrapper)
        if m_x:
            return f"ดูโพสต์ X ของ @{m_x.group(1)}", True
        m_px = _BARE_PIXIV_URL_EXTRACT_REGEX.search(stripped_wrapper)
        if m_px:
            return f"ดูภาพ Pixiv #{m_px.group(1)}", True
        m_bsky = _BARE_BSKY_URL_EXTRACT_REGEX.search(stripped_wrapper)
        if m_bsky:
            return f"ดูโพสต์ Bluesky ของ @{m_bsky.group(1)}", True
        if _BARE_YT_URL_EXTRACT_REGEX.search(stripped_wrapper):
            return "ดูวิดีโอ YouTube", True
        return "", False

    # 3. Raw stack trace / error log / code dump -> clean concise debug label (never raw trace!)
    if _looks_like_code_or_error_or_table(stripped_wrapper):
        err_m = _ERROR_TYPE_EXTRACT_REGEX.search(stripped_wrapper)
        fn_m = _FILENAME_EXTRACT_REGEX.search(stripped_wrapper)
        err_tok = err_m.group(1) if err_m else ""
        fn_tok = fn_m.group(1) if fn_m else ""
        # Check if there is also a short natural Thai/English question line outside code fences
        prose_only = re.sub(r"```[\s\S]*?```", " ", stripped_wrapper)
        prose_lines = [
            ln.strip() for ln in prose_only.splitlines()
            if ln.strip() and not _looks_like_code_or_error_or_table(ln) and not _FILE_EXT_OR_LINE_REF_REGEX.search(ln)
        ]
        prose_q = re.sub(r"\s+", " ", " ".join(prose_lines)).strip()
        if err_tok and fn_tok:
            return f"ดีบัก {err_tok} ({fn_tok})", True
        if err_tok:
            return f"ดีบัก {err_tok}", True
        if fn_tok:
            return f"ตรวจโค้ด ({fn_tok})", True
        if len(prose_q) >= 10:
            clean_p = prose_q[:72].rsplit(" ", 1)[0] if len(prose_q) > 72 and " " in prose_q[:72] else prose_q[:72]
            return clean_p.strip(), len(prose_q) > 65
        return "สอบถาม/ดีบักโค้ด", True

    # 4. Normal conversational query -> strip URLs, collapse whitespace, cut cleanly without chopping words
    clean_q = re.sub(r"\s+", " ", text_no_urls).strip()
    if len(clean_q) <= 6 or _PLACEHOLDER_TABLE_TOKEN_REGEX.match(clean_q):
        return "", False

    if len(clean_q) <= 75:
        return clean_q, (is_coding and len(clean_q) > 55)

    # Trim at a clean whitespace boundary if available
    cut = clean_q[:75]
    if " " in cut[35:]:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip(" ,.-:"), True


def _scrub_profile_favorite_topics(profile: dict, user=None) -> bool:
    """Removes polluted Discord names, code/error traces, and table noise from favorite_topics in-place."""
    raw_favs = list(profile.get("favorite_topics") or [])
    cleaned = [t for t in raw_favs if not _is_blocked_favorite_topic(t, user=user, profile=profile)]
    if cleaned != raw_favs:
        profile["favorite_topics"] = cleaned
        return True
    return False


def _scrub_profile_memory(profile: dict, user=None) -> bool:
    """
    Scrubs polluted entries across `favorite_topics`, `recent_topics`, AND `custom_notes`
    in-place so legacy or accidental code/error/table/button-prompt noise is automatically removed.
    Returns True if any field was modified.
    """
    changed = _scrub_profile_favorite_topics(profile, user=user)

    raw_recents = list(profile.get("recent_topics") or [])
    clean_recents: list[str] = []
    seen_r: set[str] = set()
    for r in raw_recents:
        rs = re.sub(r"\s+", " ", str(r or "")).strip()
        if rs and _is_valid_recent_topic(rs) and rs.lower() not in seen_r:
            seen_r.add(rs.lower())
            clean_recents.append(rs)
    if clean_recents != raw_recents:
        profile["recent_topics"] = clean_recents[-6:]
        changed = True

    raw_notes = list(profile.get("custom_notes") or [])
    clean_notes: list[str] = []
    seen_n: set[str] = set()
    for n in raw_notes:
        ns = re.sub(r"\s+", " ", str(n or "")).strip()
        if ns and _is_valid_custom_note(ns) and ns.lower() not in seen_n:
            seen_n.add(ns.lower())
            clean_notes.append(ns)
    if clean_notes != raw_notes:
        profile["custom_notes"] = clean_notes[-14:]
        changed = True

    return changed


def has_dirty_user_profiles() -> bool:
    """Returns True if local in-RAM user_profiles has changes not yet flushed to Discord cloud backup."""
    return bool(_DIRTY_SINCE_LAST_BACKUP)


def _get_cloud_backup_lock() -> asyncio.Lock:
    global _CLOUD_BACKUP_LOCK
    if _CLOUD_BACKUP_LOCK is None:
        _CLOUD_BACKUP_LOCK = asyncio.Lock()
    return _CLOUD_BACKUP_LOCK


def _read_user_profiles_local_sync() -> dict | None:
    """Reads user_profiles.json from local disk (safe to run in a worker thread; does not touch shared state)."""
    if not os.path.exists(USER_PROFILES_FILE):
        return None
    try:
        with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"[User Memory] Local load warning: {e}", flush=True)
        return None


def _apply_loaded_user_profiles(data: dict | None) -> None:
    """Mutates `user_profiles` in place so bot.py / ui_interactions.py references stay on the same dict object."""
    if data is None:
        return
    user_profiles.clear()
    user_profiles.update(data)
    print(f"[User Memory] Loaded {len(user_profiles)} user profile(s) from local disk.", flush=True)


def load_user_profiles_local():
    """Loads user_profiles.json from local disk if present."""
    _apply_loaded_user_profiles(_read_user_profiles_local_sync())


def _write_profiles_json_atomic_sync(serialized_json: str) -> None:
    """Thread-safe atomic disk flush via temp-file + os.replace so restarts never corrupt JSON."""
    tmp_path = f"{USER_PROFILES_FILE}.tmp"
    with _LOCAL_SAVE_THREAD_LOCK:
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(serialized_json)
            os.replace(tmp_path, USER_PROFILES_FILE)
        except Exception as e:
            print(f"[User Memory] Local save warning: {e}", flush=True)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass


def save_user_profiles_local():
    """Synchronous backward-compatible atomic save of user_profiles.json."""
    try:
        payload_str = json.dumps(user_profiles, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[User Memory] Local serialize warning: {e}", flush=True)
        return
    _write_profiles_json_atomic_sync(payload_str)


async def save_user_profiles_local_async():
    """
    Non-blocking atomic save of user_profiles.json:
    serializes an immutable JSON snapshot on the event loop (zero dict-mutation races),
    then offloads file write + `os.replace` to a worker thread so active voice streams never stall.
    """
    try:
        payload_str = json.dumps(user_profiles, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[User Memory] Local serialize warning: {e}", flush=True)
        return
    await asyncio.to_thread(_write_profiles_json_atomic_sync, payload_str)


def _find_backup_channel(bot: discord.Client):
    if MEMORY_BACKUP_CHANNEL_ID:
        ch = bot.get_channel(MEMORY_BACKUP_CHANNEL_ID)
        if ch:
            return ch
    for guild in bot.guilds:
        for ch in guild.text_channels:
            if ch.name.lower() in ("yukinon-memory", "bot-memory", "ai-memory"):
                return ch
    return None


async def restore_user_profiles_from_discord(bot: discord.Client):
    """
    Restores user_profiles.json from a private Discord channel (#yukinon-memory or MEMORY_BACKUP_CHANNEL_ID)
    on startup, auto-scrubs polluted entries from favorite_topics, recent_topics, and custom_notes,
    and scans #work channels to recover any OT logs so records survive Render redeploys.
    """
    global _DIRTY_SINCE_LAST_BACKUP
    # Read in a worker thread, mutate the shared dict on the event loop (no cross-thread dict mutation)
    _apply_loaded_user_profiles(await asyncio.to_thread(_read_user_profiles_local_sync))
    ch = _find_backup_channel(bot)
    if ch:
        try:
            restored = False
            async for msg in ch.history(limit=15):
                if msg.author.id == bot.user.id and msg.attachments:
                    for att in msg.attachments:
                        if att.filename == "user_profiles.json":
                            raw = await att.read()
                            data = json.loads(raw.decode("utf-8"))
                            if isinstance(data, dict) and data:
                                user_profiles.update(data)
                                await save_user_profiles_local_async()
                                print(
                                    f"[User Memory] Restored {len(user_profiles)} user profile(s) "
                                    f"from Discord channel #{ch.name}",
                                    flush=True,
                                )
                                restored = True
                                break
                if restored:
                    break
        except Exception as e:
            print(f"[User Memory] Cloud restore note: {e}", flush=True)

    await recover_ot_logs_from_work_channels(bot)

    scrubbed_any = False
    for prof in user_profiles.values():
        if isinstance(prof, dict) and _scrub_profile_memory(prof):
            scrubbed_any = True
    if scrubbed_any:
        _DIRTY_SINCE_LAST_BACKUP = True
        await save_user_profiles_local_async()
        print("[User Memory] Auto-scrubbed polluted code/error/table/prompt noise from user profiles.", flush=True)
        await backup_user_profiles_to_discord(force=True)


async def backup_user_profiles_to_discord(force: bool = False):
    """
    Serialized cloud backup of user_profiles.json to #yukinon-memory (throttled to every 3 mins unless force=True).
    Protected by `_CLOUD_BACKUP_LOCK` so concurrent edits never delete each other's latest backup!
    Only deletes previous `user_profiles.json` backup messages — NEVER normal bot replies,
    `/memory` cards, or deferred thinking placeholders in `#yukinon-memory`.
    """
    global _DIRTY_SINCE_LAST_BACKUP, _LAST_CLOUD_BACKUP_TS
    if not _DIRTY_SINCE_LAST_BACKUP and not force:
        return
    now = time.time()
    if not force and (now - _LAST_CLOUD_BACKUP_TS < 180.0):
        return
    bot = state.bot
    if not bot or not bot.is_ready():
        return
    ch = _find_backup_channel(bot)
    if not ch:
        return

    async with _get_cloud_backup_lock():
        now = time.time()
        if not force and (not _DIRTY_SINCE_LAST_BACKUP or (now - _LAST_CLOUD_BACKUP_TS < 180.0)):
            return
        try:
            _LAST_CLOUD_BACKUP_TS = now
            _DIRTY_SINCE_LAST_BACKUP = False
            payload_bytes = json.dumps(user_profiles, ensure_ascii=False, indent=2).encode("utf-8")

            # Upload new backup FIRST, then delete older `user_profiles.json` backup messages ONLY
            new_msg = await ch.send(
                content=f"💾 **[Yukinon Persistent User Memory Backup]** (`{len(user_profiles)}` users | `{get_current_thai_datetime_str()}`)",
                file=discord.File(io.BytesIO(payload_bytes), filename="user_profiles.json"),
            )

            async for old_msg in ch.history(limit=20):
                is_backup_msg = any(a.filename == "user_profiles.json" for a in (old_msg.attachments or []))
                if old_msg.author.id == bot.user.id and old_msg.id != new_msg.id and is_backup_msg:
                    try:
                        await old_msg.delete()
                    except Exception:
                        pass
        except Exception as e:
            _DIRTY_SINCE_LAST_BACKUP = True
            print(f"[User Memory] Cloud backup note: {e}", flush=True)


def ensure_user_profile_exists(
    uid_str: str,
    display_name: str = "",
    username: str = "",
    is_owner: bool = False,
) -> dict:
    """Returns an existing or newly initialized user profile dict with normalized structured fields."""
    now_str = get_current_thai_datetime_str()
    clean_disp = re.sub(r"[^\w\u0E00-\u0E7F\s().-]", "", display_name or "").strip()[:36]
    default_disp = "Partner (Owner)" if is_owner else (clean_disp or f"User_{uid_str[-4:]}")

    prof = user_profiles.get(uid_str)
    if not isinstance(prof, dict):
        prof = {
            "user_id": uid_str,
            "display_name": default_disp,
            "username": username,
            "is_owner": is_owner,
            "interaction_count": 1,
            "first_seen": now_str,
            "last_seen": now_str,
            "pinned_facts": [],
            "favorite_topics": [],
            "custom_notes": [],
            "recent_topics": [],
        }
        user_profiles[uid_str] = prof

    prof.setdefault("user_id", uid_str)
    if not prof.get("display_name"):
        prof["display_name"] = default_disp
    if username and not prof.get("username"):
        prof["username"] = username
    if is_owner:
        prof["is_owner"] = True
    prof.setdefault("interaction_count", 1)
    prof.setdefault("first_seen", now_str)
    prof.setdefault("last_seen", now_str)
    if not isinstance(prof.get("pinned_facts"), list):
        prof["pinned_facts"] = []
    if not isinstance(prof.get("favorite_topics"), list):
        prof["favorite_topics"] = []
    if not isinstance(prof.get("custom_notes"), list):
        prof["custom_notes"] = []
    if not isinstance(prof.get("recent_topics"), list):
        prof["recent_topics"] = []
    return prof


def _dedupe_clean_list(items: list[str], max_items: int = 12, max_item_len: int = 140) -> list[str]:
    out: list[str] = []
    seen_lower: set[str] = set()
    for raw in items:
        s = re.sub(r"\s+", " ", str(raw or "")).strip()[:max_item_len]
        if s and s.lower() not in seen_lower:
            seen_lower.add(s.lower())
            out.append(s)
            if len(out) >= max_items:
                break
    return out


async def _summarize_recent_topic_bg(
    uid_str: str,
    placeholder_topic: str,
    user_prompt: str,
    assistant_reply: str = "",
):
    """
    Non-blocking background topic summarizer (0ms user reply latency):
    Replaces a condensed `recent_topics` placeholder in-place with a crisp 1-phrase
    topic summary (<= 65 chars, no raw URLs, stack traces, or code syntax).
    """
    global _DIRTY_SINCE_LAST_BACKUP
    if not uid_str or not placeholder_topic or uid_str in _TOPIC_SUMMARIZE_IN_PROGRESS:
        return
    now = time.time()
    if now < _TOPIC_SUMMARIZE_COOLDOWN.get(uid_str, 0.0):
        return

    _TOPIC_SUMMARIZE_IN_PROGRESS.add(uid_str)
    _TOPIC_SUMMARIZE_COOLDOWN[uid_str] = now + 20.0
    try:
        from config import GROQ_API_KEY
        from llm import (
            call_groq_chat,
            get_groq_search_and_chat_candidates,
            call_gemma_ai_studio_chat,
            call_cloudflare_gemma_chat,
        )

        clean_u = re.sub(r"\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*$", "", str(user_prompt or "")).strip()[:350]
        clean_a = re.sub(r"\s+", " ", str(assistant_reply or "")).strip()[:350]
        sum_prompt = (
            "Summarize this chat exchange into ONE ultra-concise topic phrase (<= 55 characters, "
            "natural Thai or English matching the conversation) suitable for a `recent_topics` list.\n"
            "STRICT RULES:\n"
            "- NEVER include raw URLs, stack traces, raw code syntax, table borders, or internal prompt text.\n"
            "- If the message is just a test table, formatting test, or trivial filler, return {\"topic\": \"\"}.\n"
            "- Examples: \"ดีบัก TypeError ใน FormSearch.tsx\", \"ดูงานวาด อาโออิ เซนะ ของ @hibikamin\", "
            "\"ปรับสูตรชงเอสเปรสโซ่คั่วกลาง\", \"ถามเรื่องสเปคโมเดล Yamato 1/350\".\n"
            'Return ONLY valid JSON: {"topic": "..."}\n\n'
            f"User input: {clean_u}\n"
            + (f"Bot reply snippet: {clean_a}" if clean_a else "")
        )
        msgs = [{"role": "user", "content": sum_prompt}]
        raw_out = ""
        if GROQ_API_KEY:
            for g_mod in await get_groq_search_and_chat_candidates():
                try:
                    raw_out = await call_groq_chat(
                        msgs, g_mod, temperature=0.0, is_coding=False, purpose="TopicSummarize"
                    )
                    if raw_out:
                        break
                except Exception:
                    continue
        if not raw_out:
            raw_out = await call_gemma_ai_studio_chat(
                msgs, temperature=0.0, max_output_tokens=120, purpose="TopicSummarize"
            )
        if not raw_out:
            raw_out = await call_cloudflare_gemma_chat(
                msgs, temperature=0.0, max_tokens=120, purpose="TopicSummarize"
            )
        if not raw_out:
            return

        m_json = re.search(r"\{[\s\S]*\}", raw_out)
        if not m_json:
            return
        parsed = json.loads(m_json.group(0))
        summary = re.sub(r"\s+", " ", str(parsed.get("topic") or "")).strip()[:70]

        prof = user_profiles.get(uid_str)
        if not isinstance(prof, dict):
            return
        recents = list(prof.get("recent_topics") or [])
        if placeholder_topic not in recents:
            return
        idx = len(recents) - 1 - recents[::-1].index(placeholder_topic)
        if not summary:
            # Summarizer determined it was trivial/test noise -> remove placeholder cleanly
            recents.pop(idx)
        elif _is_valid_recent_topic(summary):
            recents[idx] = summary
            # Deduplicate consecutive/duplicate entries
            recents = _dedupe_clean_list(recents, max_items=6, max_item_len=80)
        else:
            return

        prof["recent_topics"] = recents[-6:]
        _DIRTY_SINCE_LAST_BACKUP = True
        await save_user_profiles_local_async()
    except Exception as e:
        api_log(f"TopicSummarize -> skipped ({type(e).__name__})")
    finally:
        _TOPIC_SUMMARIZE_IN_PROGRESS.discard(uid_str)


async def consolidate_user_memory_async(uid_str: str):
    """
    Background LLM memory consolidator: triggered automatically when rolling `custom_notes`
    exceeds 10 entries. Merges duplicates, promotes durable personal facts to `pinned_facts`,
    drops transient code/error/table/URL noise, and prunes outdated ephemeral notes without blocking replies.
    """
    global _DIRTY_SINCE_LAST_BACKUP
    now = time.time()
    if uid_str in _CONSOLIDATION_IN_PROGRESS:
        return
    if now < _CONSOLIDATION_COOLDOWN.get(uid_str, 0.0):
        return

    prof = user_profiles.get(uid_str)
    if not prof:
        return
    notes = list(prof.get("custom_notes") or [])
    pinned = list(prof.get("pinned_facts") or [])
    if len(notes) <= 10:
        return

    _CONSOLIDATION_IN_PROGRESS.add(uid_str)
    _CONSOLIDATION_COOLDOWN[uid_str] = now + 1800.0
    try:
        from config import GROQ_API_KEY
        from llm import (
            call_groq_chat,
            get_groq_search_and_chat_candidates,
            call_gemma_ai_studio_chat,
            call_cloudflare_gemma_chat,
        )

        consolidation_prompt = (
            "You are a memory consolidation module. Deduplicate, clean, and organize the user's memory entries.\n"
            "Rules:\n"
            "1. Keep ALL valid existing `pinned_facts` unless an exact duplicate or directly contradicted by a newer entry.\n"
            "2. Promote durable long-term facts (preferences, hardware, profession, pets, hobbies, recurring goals) "
            "from `custom_notes` into `pinned_facts` (max 15 items total).\n"
            "3. Keep only distinct, recent, meaningful short-term context in `custom_notes` (max 8 items total), merging duplicates.\n"
            "4. DISCARD any temporary error logs, stack traces, raw code snippets, Markdown/ASCII tables, raw URLs, "
            "one-off debugging errors, or internal bot prompts.\n"
            "5. Preserve original language (Thai/English) concisely.\n"
            '6. Return ONLY valid JSON: {"pinned_facts": [...], "custom_notes": [...]}\n\n'
            f"Current pinned_facts: {json.dumps(pinned, ensure_ascii=False)}\n"
            f"Current custom_notes: {json.dumps(notes, ensure_ascii=False)}"
        )
        msgs = [{"role": "user", "content": consolidation_prompt}]

        raw_out = ""
        if GROQ_API_KEY:
            for g_mod in await get_groq_search_and_chat_candidates():
                try:
                    raw_out = await call_groq_chat(
                        msgs, g_mod, temperature=0.0, is_coding=False, purpose="MemoryConsolidate"
                    )
                    if raw_out:
                        break
                except Exception:
                    continue
        if not raw_out:
            raw_out = await call_gemma_ai_studio_chat(msgs, temperature=0.0, max_output_tokens=768, purpose="MemoryConsolidate")
        if not raw_out:
            raw_out = await call_cloudflare_gemma_chat(msgs, temperature=0.0, max_tokens=768, purpose="MemoryConsolidate")

        if not raw_out:
            return
        m_json = re.search(r"\{[\s\S]*\}", raw_out)
        if not m_json:
            return
        parsed = json.loads(m_json.group(0))

        # The LLM await can take seconds; if the user (or a tool) changed memory meanwhile, writing
        # the stale snapshot would silently drop that change. Skip and let the next trigger retry.
        cur = user_profiles.get(uid_str)
        if (
            cur is not prof
            or list(cur.get("custom_notes") or []) != notes
            or list(cur.get("pinned_facts") or []) != pinned
        ):
            api_log(f"MemoryConsolidate -> uid={uid_str[-4:]} skipped (profile changed during consolidation)")
            _CONSOLIDATION_COOLDOWN.pop(uid_str, None)
            return

        new_pinned = parsed.get("pinned_facts")
        new_notes = parsed.get("custom_notes")
        if isinstance(new_pinned, list) and isinstance(new_notes, list):
            clean_pinned = [x for x in new_pinned if _is_valid_custom_note(x)]
            clean_notes = [x for x in new_notes if _is_valid_custom_note(x)]
            # Existing pinned facts are never removed: they come first and the cap only limits additions.
            # Ceiling: dedupe is exact (case/whitespace-insensitive), so a reworded duplicate of an existing
            # pinned fact can be added alongside the original. Upgrade path: fuzzy matching.
            prof["pinned_facts"] = _dedupe_clean_list(
                pinned + clean_pinned, max_items=max(15, len(pinned)), max_item_len=150
            )
            prof["custom_notes"] = _dedupe_clean_list(clean_notes, max_items=8, max_item_len=150)
            _DIRTY_SINCE_LAST_BACKUP = True
            await save_user_profiles_local_async()
            api_log(
                f"MemoryConsolidate -> uid={uid_str[-4:]} consolidated "
                f"(pinned={len(prof['pinned_facts'])}, rolling={len(prof['custom_notes'])})"
            )
            await backup_user_profiles_to_discord(force=True)
    except Exception as e:
        api_log(f"MemoryConsolidate -> skipped ({type(e).__name__})")
    finally:
        _CONSOLIDATION_IN_PROGRESS.discard(uid_str)


# Strong references to fire-and-forget tasks: asyncio only keeps weak refs, so unreferenced tasks can be GC'd mid-run
_BG_TASKS: set[asyncio.Task] = set()


def _schedule_bg_task(coro) -> bool:
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        if state.bot and getattr(state.bot, "loop", None) and state.bot.loop.is_running():
            task = state.bot.loop.create_task(coro)
        else:
            coro.close()
            return False
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)
    return True


def record_user_interaction(
    user,
    user_prompt: str,
    channel_id: int = None,
    learned_memo: str = "",
    is_button_followup: bool = False,
    is_coding: bool = False,
    assistant_reply: str = "",
):
    """
    Updates the persistent profile of the speaker with hygiene-filtered topics, interaction count,
    clean condensed/summarized recent topics, and validated LLM memos.
    """
    global _DIRTY_SINCE_LAST_BACKUP
    if user is None or getattr(user, "bot", False):
        return

    uid_str = str(user.id)
    now_str = get_current_thai_datetime_str()
    owner_flag = is_owner_user(user)
    raw_display = getattr(user, "display_name", "") or getattr(user, "name", "Unknown")
    clean_display = re.sub(r"[^\w\u0E00-\u0E7F\s.-]", "", raw_display).strip()[:32] or "Member"

    is_new = uid_str not in user_profiles
    profile = ensure_user_profile_exists(
        uid_str,
        display_name="Partner (Owner)" if owner_flag else clean_display,
        username=getattr(user, "name", ""),
        is_owner=owner_flag,
    )

    if not owner_flag:
        profile["display_name"] = clean_display
    profile["is_owner"] = owner_flag
    profile["interaction_count"] = 1 if is_new else (int(profile.get("interaction_count", 0)) + 1)
    profile["last_seen"] = now_str

    _scrub_profile_memory(profile, user=user)
    favs = list(profile.get("favorite_topics") or [])
    if not is_button_followup and channel_id and channel_id in channel_active_topic:
        act_t = channel_active_topic[channel_id].get("topic", "").strip()
        if act_t and act_t not in favs and not _is_blocked_favorite_topic(act_t, user=user, profile=profile):
            favs.append(act_t)

    if not is_button_followup and not _looks_like_internal_bot_or_button_prompt(user_prompt):
        for label, pattern in TOPIC_KEYWORDS_MAP.items():
            if pattern.search(user_prompt) and label not in favs:
                favs.append(label)

    profile["favorite_topics"] = favs[-12:]

    condensed_recent, wants_bg_summary = _condense_recent_topic_sync(
        user_prompt,
        is_coding=is_coding,
        is_button_followup=is_button_followup,
    )
    if condensed_recent and _is_valid_recent_topic(condensed_recent):
        recents = list(profile.get("recent_topics") or [])
        if not recents or recents[-1].lower() != condensed_recent.lower():
            recents.append(condensed_recent)
        profile["recent_topics"] = _dedupe_clean_list(recents, max_items=6, max_item_len=80)
        if wants_bg_summary:
            _schedule_bg_task(
                _summarize_recent_topic_bg(uid_str, condensed_recent, user_prompt, assistant_reply)
            )

    if learned_memo:
        clean_memo = re.sub(r"\s+", " ", learned_memo).strip()[:150]
        if _is_valid_custom_note(clean_memo):
            notes = list(profile.get("custom_notes") or [])
            if clean_memo.lower() not in {n.lower() for n in notes}:
                notes.append(clean_memo)
                profile["custom_notes"] = notes[-14:]

    user_profiles[uid_str] = profile
    _DIRTY_SINCE_LAST_BACKUP = True

    # Offload local disk flush to worker thread when inside an active event loop (zero voice jitter)
    if not _schedule_bg_task(save_user_profiles_local_async()):
        save_user_profiles_local()

    if len(profile.get("custom_notes") or []) > 10:
        _schedule_bg_task(consolidate_user_memory_async(uid_str))

    _schedule_bg_task(backup_user_profiles_to_discord(force=False))


def get_speaker_static_memory_context(user) -> str:
    """
    Returns ONLY permanent `pinned_facts` for Gemini implicit prefix caching so `systemInstruction`
    remains 100% byte-identical across turns even when auto-detected topics change.
    """
    if user is None:
        return ""
    uid_str = str(getattr(user, "id", ""))
    profile = user_profiles.get(uid_str)
    if not profile:
        return ""

    pinned = "; ".join(profile.get("pinned_facts") or [])
    if pinned:
        return f"- ข้อมูลสำคัญถาวรที่ปักหมุดไว้ (Pinned Facts): **{pinned}**"
    return ""


def get_speaker_volatile_memory_context(user) -> str:
    """
    Returns per-turn or auto-updating user profile fields (`interaction_count`, `last_seen`,
    `favorite_topics`, `custom_notes`, `recent_topics`) meant to be injected at the tail of
    the latest user turn so `systemInstruction` never invalidates Gemini implicit prefix caches.
    """
    if user is None:
        return ""
    uid_str = str(getattr(user, "id", ""))
    profile = user_profiles.get(uid_str)
    if not profile or int(profile.get("interaction_count", 0)) <= 1:
        if not is_owner_user(user):
            return "- ประวัติการคุย: นี่เป็นการพูดคุยครั้งแรกๆ กับสมาชิกคนนี้ในเซิร์ฟเวอร์"
        if not profile:
            return ""

    cnt = profile.get("interaction_count", 1)
    favs = ", ".join(profile.get("favorite_topics") or [])
    notes = "; ".join(profile.get("custom_notes") or [])
    recents = " | ".join(profile.get("recent_topics") or [])

    lines = [
        f"- ประวัติความคุ้นเคยกับผู้พูดคนนี้: เคยคุยกันมาแล้ว **{cnt} ครั้ง** (คุยล่าสุดเมื่อ: {profile.get('last_seen', 'ไม่นานมานี้')})",
    ]
    if favs:
        lines.append(f"- หัวข้อ/ความสนใจหลักของผู้พูดคนนี้: **{favs}**")
    if notes:
        lines.append(f"- บันทึกความจำระยะสั้น (Rolling Notes): **{notes}**")
    if recents:
        lines.append(f"- เรื่องล่าสุดที่เคยคุยกัน: `{recents}`")
    return "\n".join(lines)


def get_speaker_memory_context(user) -> str:
    """Returns full structured memory summary (static + volatile) for backward-compatible callers."""
    parts = [
        p for p in (
            get_speaker_static_memory_context(user),
            get_speaker_volatile_memory_context(user),
        )
        if p
    ]
    return "\n".join(parts)


def resolve_user_mentions_and_build_dossier(
    raw_text: str, guild: discord.Guild | None, bot_user_id: int, speaker_user=None
) -> tuple[str, str]:
    if not raw_text:
        return "", ""

    bot_id_str = str(bot_user_id) if bot_user_id else ""
    mentioned_ids = []
    for m_id in USER_MENTION_REGEX.findall(raw_text):
        if m_id != bot_id_str and m_id not in mentioned_ids:
            mentioned_ids.append(m_id)

    if not mentioned_ids:
        return raw_text, ""

    resolved_text = raw_text
    dossier_entries = []

    for uid_str in mentioned_ids[:4]:
        member = guild.get_member(int(uid_str)) if (guild and uid_str.isdigit()) else None
        prof = user_profiles.get(uid_str)
        is_target_owner = is_owner_user(member) if member else (prof.get("is_owner", False) if prof else False)

        if is_target_owner:
            readable_tag = "@[คู่หูชมรมบริการ / ผู้สร้างของฉัน]"
            resolved_text = re.sub(rf"<@!?{uid_str}>", readable_tag, resolved_text)
            favs = ", ".join((prof or {}).get("favorite_topics") or ["การเขียนโค้ด", "เกม", "เทคโนโลยี"])
            all_notes = list((prof or {}).get("pinned_facts") or []) + list((prof or {}).get("custom_notes") or [])
            notes = "; ".join(all_notes[:10])
            asking_about_self = (speaker_user and str(getattr(speaker_user, "id", "")) == uid_str)
            dossier_entries.append(
                f"- ข้อมูลของ {readable_tag}: เขาคือคู่หูคนสำคัญในชมรมบริการและผู้ดูแลของฉันเองค่ะ "
                f"{'(ซึ่งก็คือตัวนายที่กำลังถามฉันอยู่นี่แหละ!)' if asking_about_self else ''} "
                f"ความสนใจของเขาคือ: {favs}"
                + (f" | บันทึกเพิ่มเติม: {notes}" if notes else "")
                + "\n  [กฎเหล็ก: ห้ามเอ่ยหรือพิมพ์ชื่อจริงของเขาเด็ดขาด ให้พูดถึงเขาในฐานะ 'นาย' (หากเขาถามเอง) หรือ 'เขา / คู่หูชมรมบริการของฉัน' (หากคนอื่นถามถึง)]"
            )
            continue

        raw_name = (
            (member.display_name if member else None)
            or (prof.get("display_name") if prof else None)
            or f"User_{uid_str[-4:]}"
        )
        clean_name = re.sub(r"[^\w\u0E00-\u0E7F\s.-]", "", raw_name).strip()[:32] or f"User_{uid_str[-4:]}"
        resolved_text = re.sub(rf"<@!?{uid_str}>", f"@{clean_name}", resolved_text)

        roles_list = []
        if member and hasattr(member, "roles"):
            roles_list = [r.name for r in member.roles if r.name != "@everyone"][-4:]
        roles_str = ", ".join(roles_list) if roles_list else "สมาชิกในเซิร์ฟเวอร์"

        if prof and int(prof.get("interaction_count", 0)) > 0:
            cnt = prof.get("interaction_count", 1)
            last_t = prof.get("last_seen", "ไม่ระบุ")
            favs = ", ".join(prof.get("favorite_topics") or []) or "คุยเรื่องทั่วไป"
            all_notes = list(prof.get("pinned_facts") or []) + list(prof.get("custom_notes") or [])
            notes = "; ".join(all_notes[:10])
            recents = " / ".join(prof.get("recent_topics") or [])
            entry = (
                f"- ข้อมูลของสมาชิกชื่อ **{clean_name}** (ยศในเซิร์ฟเวอร์: {roles_str}):\n"
                f"  * สถานะความคุ้นเคย: **รู้จักและเคยคุยกับฉันมาแล้ว {cnt} ครั้ง** (คุยกันล่าสุดเมื่อ: {last_t})\n"
                f"  * ความสนใจ/หัวข้อที่ชอบคุย: {favs}\n"
                f"  * เรื่องที่เคยถามฉันล่าสุด: `{recents}`"
            )
            if notes:
                entry += f"\n  * บันทึกความจำเกี่ยวกับคนนี้: {notes}"
            dossier_entries.append(entry)
        else:
            dossier_entries.append(
                f"- ข้อมูลของสมาชิกชื่อ **{clean_name}** (ยศในเซิร์ฟเวอร์: {roles_str}):\n"
                f"  * สถานะความคุ้นเคย: **เห็นรายชื่ออยู่ในเซิร์ฟเวอร์นี้ แต่ยังไม่เคยเข้ามาพูดคุยกับฉันในชมรมบริการเลยสักครั้งค่ะ**"
            )

    dossier_block = (
        "<mentioned_users_dossier>\n"
        "### [ข้อมูลแฟ้มประวัติของสมาชิกที่ถูกกล่าวถึงในข้อความนี้ (ใช้ตอบเมื่อถูกถามว่ารู้จักไหม หรือพูดถึงคนนี้)]:\n"
        + "\n".join(dossier_entries)
        + "\n</mentioned_users_dossier>"
    )
    return resolved_text, dossier_block


# ==============================================================================
# EDITABLE PERSISTENT USER MEMORY (`user_profiles.json` in `#yukinon-memory`)
# ==============================================================================
MEMORY_COMMAND_REGEX = re.compile(
    r"^(?:"
    r"(?:ดู|เช็ค|เปิด|แสดง|สรุป|แก้|แก้ไข|อัปเดต|ลบ|ล้าง|รีเซ็ต)\s*(?:ข้อมูล)?(?:ความจำ|โปรไฟล์|บันทึกความจำ)\b|"
    r"(?:ความจำ|โปรไฟล์)(?:ของฉัน|ส่วนตัว)\b|"
    r"(?:จำไว้ว่า|ช่วยจำว่า|ปักหมุดความจำว่า?|บันทึกความจำว่า|เพิ่มบันทึก|เพิ่มโน้ต|เพิ่มหัวข้อที่ชอบ|เพิ่มความสนใจ)\b|"
    r"(?:แก้|แก้ไข|อัปเดต|ลบ|ล้าง)\s*(?:ข้อมูลถาวร|ปักหมุด|หัวข้อที่ชอบ|ความสนใจ|บันทึก|โน้ต|เรื่องล่าสุด|หัวข้อล่าสุด|ชื่อแสดงผล)\b|"
    r"(?:view|show|edit|update|clear|reset)\s+(?:my\s+)?(?:memory|profile)\b|"
    r"remember\s+that\b"
    r")",
    re.IGNORECASE,
)


def is_memory_command_text(text: str) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"<@!?\d+>", "", text).strip()
    return bool(MEMORY_COMMAND_REGEX.search(cleaned))


async def update_user_memory_fields(
    target_uid: str,
    display_name: str | None = None,
    pinned_facts: list[str] | None = None,
    favorite_topics: list[str] | None = None,
    custom_notes: list[str] | None = None,
    recent_topics: list[str] | None = None,
) -> dict:
    """Updates specified fields on `user_profiles[target_uid]`, saves atomically off-thread, and force-syncs `#yukinon-memory`."""
    global _DIRTY_SINCE_LAST_BACKUP
    prof = ensure_user_profile_exists(target_uid)
    if display_name is not None and display_name.strip():
        clean_disp = re.sub(r"[^\w\u0E00-\u0E7F\s().-]", "", display_name).strip()[:36]
        if clean_disp:
            prof["display_name"] = clean_disp
    if pinned_facts is not None:
        prof["pinned_facts"] = _dedupe_clean_list(pinned_facts, max_items=15, max_item_len=150)
    if favorite_topics is not None:
        prof["favorite_topics"] = _dedupe_clean_list(favorite_topics, max_items=12, max_item_len=80)
    if custom_notes is not None:
        prof["custom_notes"] = _dedupe_clean_list(custom_notes, max_items=14, max_item_len=150)
    if recent_topics is not None:
        prof["recent_topics"] = _dedupe_clean_list(recent_topics, max_items=8, max_item_len=120)

    prof["last_seen"] = get_current_thai_datetime_str()
    user_profiles[target_uid] = prof
    _DIRTY_SINCE_LAST_BACKUP = True
    await save_user_profiles_local_async()
    await backup_user_profiles_to_discord(force=True)
    return prof


def format_user_memory_card(target_uid: str, header_title: str = "ข้อมูลความจำ (User Memory)") -> str:
    prof = ensure_user_profile_exists(target_uid)
    disp = prof.get("display_name") or f"User_{target_uid[-4:]}"
    cnt = int(prof.get("interaction_count") or 1)
    last_seen = prof.get("last_seen") or "-"

    pinned = prof.get("pinned_facts") or []
    favs = prof.get("favorite_topics") or []
    notes = prof.get("custom_notes") or []
    recents = prof.get("recent_topics") or []

    pinned_lines = (
        "\n".join(f"  `P{i}.` {p}" for i, p in enumerate(pinned, 1))
        if pinned
        else "  *(ยังไม่มีข้อมูลถาวรที่ปักหมุดไว้)*"
    )
    fav_str = ", ".join(f"`{x}`" for x in favs) if favs else "*(ยังไม่มีข้อมูล)*"
    notes_lines = (
        "\n".join(f"  `{i}.` {n}" for i, n in enumerate(notes, 1))
        if notes
        else "  *(ยังไม่มีบันทึกระยะสั้น)*"
    )
    recent_lines = (
        "\n".join(f"  `{i}.` {r}" for i, r in enumerate(recents, 1))
        if recents
        else "  *(ยังไม่มีประวัติหัวข้อล่าสุด)*"
    )

    return (
        f"🧠 **[{header_title}]**\n"
        f"- **ชื่อที่บันทึก (Display Name):** `{disp}` (คุยกันแล้ว `{cnt}` ครั้ง | อัปเดต: `{last_seen}`)\n"
        f"- **ข้อมูลถาวรที่ปักหมุด (Pinned Facts):**\n{pinned_lines}\n"
        f"- **หัวข้อที่ชอบ (Favorite Topics):** {fav_str}\n"
        f"- **บันทึกระยะสั้น (Rolling Notes):**\n{notes_lines}\n"
        f"- **เรื่องที่คุยล่าสุด (Recent Topics):**\n{recent_lines}\n"
        f"-# [MEM_RECORD: uid={target_uid}]"
    )


async def apply_tool_memory_action(speaker_user, action: str, value: str = "") -> tuple[str, str]:
    """Native function-calling handler for `manage_user_memory` with strict noise hygiene."""
    uid_str = str(getattr(speaker_user, "id", ""))
    if not uid_str:
        return "[Memory Error: Missing user ID]", ""

    owner_flag = is_owner_user(speaker_user)
    raw_disp = getattr(speaker_user, "display_name", "") or getattr(speaker_user, "name", "")
    prof = ensure_user_profile_exists(
        uid_str,
        display_name="Partner (Owner)" if owner_flag else raw_disp,
        username=getattr(speaker_user, "name", ""),
        is_owner=owner_flag,
    )
    act = (action or "view").strip().lower()
    val = re.sub(r"\s+", " ", (value or "")).strip()

    if act == "view":
        return format_user_memory_card(uid_str), uid_str

    if act == "pin_fact" and val:
        if not _is_valid_custom_note(val):
            return "[Memory Notice: Skipped saving non-personal code/log/table/URL noise into pinned_facts.]", ""
        pinned = list(prof.get("pinned_facts") or [])
        pinned.append(val)
        await update_user_memory_fields(uid_str, pinned_facts=pinned)
        return f"Pinned permanent fact to user memory: '{val}'.\n" + format_user_memory_card(uid_str), uid_str

    if act == "add_note" and val:
        if not _is_valid_custom_note(val):
            return "[Memory Notice: Skipped saving non-personal code/log/table/URL noise into custom_notes.]", ""
        notes = list(prof.get("custom_notes") or [])
        notes.append(val)
        await update_user_memory_fields(uid_str, custom_notes=notes)
        if len(notes) > 10:
            _schedule_bg_task(consolidate_user_memory_async(uid_str))
        return f"Saved rolling note to user memory: '{val}'.", uid_str

    if act == "add_topic" and val:
        raw_items = [x.strip() for x in re.split(r"[,;،]+", val) if x.strip()]
        new_items = [
            x for x in raw_items
            if not _is_blocked_favorite_topic(x, user=speaker_user, profile=prof)
        ]
        if not new_items:
            return "[Memory Notice: Skipped adding blocked/code/table noise into favorite_topics.]", ""
        favs = list(prof.get("favorite_topics") or []) + new_items
        await update_user_memory_fields(uid_str, favorite_topics=favs)
        return f"Added favorite topic(s): {', '.join(new_items)}.", uid_str

    if act == "delete_fact" and val:
        pinned = list(prof.get("pinned_facts") or [])
        cleaned_idx = re.sub(r"^[pP]", "", val)
        if val.lower() in ("all", "ทั้งหมด"):
            pinned = []
        elif cleaned_idx.isdigit() and 1 <= int(cleaned_idx) <= len(pinned):
            pinned.pop(int(cleaned_idx) - 1)
        else:
            pinned = [p for p in pinned if val.lower() not in p.lower()]
        await update_user_memory_fields(uid_str, pinned_facts=pinned)
        return f"Updated pinned facts after deleting '{val}'.\n" + format_user_memory_card(uid_str), uid_str

    if act == "delete_note" and val:
        notes = list(prof.get("custom_notes") or [])
        if val.lower() in ("all", "ทั้งหมด"):
            notes = []
        elif val.isdigit() and 1 <= int(val) <= len(notes):
            notes.pop(int(val) - 1)
        else:
            notes = [n for n in notes if val.lower() not in n.lower()]
        await update_user_memory_fields(uid_str, custom_notes=notes)
        return f"Updated rolling notes after deleting '{val}'.\n" + format_user_memory_card(uid_str), uid_str

    return format_user_memory_card(uid_str), uid_str


async def process_memory_request(message, raw_text: str) -> tuple[str, str]:
    """
    Handles viewing, adding, pinning, editing, deleting, and resetting Persistent User Memory
    (`user_profiles.json` stored in `#yukinon-memory`), returning `(reply_text, target_uid)`.
    """
    speaker = message.author
    target_uid = str(speaker.id)
    target_member = speaker

    bot_id_str = str(state.bot.user.id) if (state.bot and state.bot.user) else ""
    if is_owner_user(speaker):
        for m_id in USER_MENTION_REGEX.findall(raw_text):
            if m_id != bot_id_str:
                target_uid = m_id
                if getattr(message, "guild", None) and m_id.isdigit():
                    found_m = message.guild.get_member(int(m_id))
                    if found_m:
                        target_member = found_m
                break

    owner_flag = is_owner_user(target_member) if target_member is not None else False
    raw_disp = getattr(target_member, "display_name", "") or getattr(target_member, "name", "")
    prof = ensure_user_profile_exists(
        target_uid,
        display_name="Partner (Owner)" if owner_flag else raw_disp,
        username=getattr(target_member, "name", ""),
        is_owner=owner_flag,
    )

    text = re.sub(r"<@!?\d+>", "", raw_text.strip()).strip()
    text = re.sub(r"^!(?:memory|memo|profile|ความจำ)\b\s*", "", text, flags=re.IGNORECASE).strip()

    if re.match(r"^(?:รีเซ็ต|ล้าง)\s*(?:ข้อมูล)?(?:ความจำ|โปรไฟล์)(?:ทั้งหมด|ของฉัน)?$", text, re.IGNORECASE) or text.lower() in ("reset", "clear", "reset all", "clear all"):
        await update_user_memory_fields(
            target_uid,
            pinned_facts=[],
            favorite_topics=[],
            custom_notes=[],
            recent_topics=[],
        )
        return (
            format_user_memory_card(target_uid, header_title="รีเซ็ตความจำถาวรเรียบร้อยแล้วค่ะ (คงข้อมูล OT ไว้ให้ครบ)"),
            target_uid,
        )

    if re.match(r"^(?:ลบ|ล้าง)\s*(?:เรื่องล่าสุด|หัวข้อล่าสุด)(?:ทั้งหมด)?$", text, re.IGNORECASE) or text.lower() in ("clear recent", "clear recent_topics"):
        await update_user_memory_fields(target_uid, recent_topics=[])
        return (
            format_user_memory_card(target_uid, header_title="ล้างรายการเรื่องที่คุยล่าสุดเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    # Explicit "จำไว้ว่า..." / "ปักหมุดความจำ..." writes to permanent `pinned_facts` so FIFO never evicts it!
    m_pin_fact = re.match(
        r"^(?:จำไว้ว่า|ช่วยจำว่า|ปักหมุดความจำว่า?|บันทึกความจำว่า|pin\s+fact|remember\s+that)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_pin_fact:
        new_fact = m_pin_fact.group(1).strip()
        pinned = list(prof.get("pinned_facts") or [])
        pinned.append(new_fact)
        await update_user_memory_fields(target_uid, pinned_facts=pinned)
        return (
            format_user_memory_card(target_uid, header_title="ปักหมุดข้อมูลสำคัญลงในความจำถาวรเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_add_note = re.match(
        r"^(?:เพิ่มบันทึก|เพิ่มโน้ต|add\s+note)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_add_note:
        new_note = m_add_note.group(1).strip()
        notes = list(prof.get("custom_notes") or [])
        notes.append(new_note)
        await update_user_memory_fields(target_uid, custom_notes=notes)
        return (
            format_user_memory_card(target_uid, header_title="เพิ่มบันทึกระยะสั้นให้เรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_add_fav = re.match(
        r"^(?:เพิ่มหัวข้อที่ชอบ|เพิ่มความสนใจ|add\s+topic)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_add_fav:
        raw_items = [x.strip() for x in re.split(r"[,;،]+", m_add_fav.group(1)) if x.strip()]
        favs = list(prof.get("favorite_topics") or []) + raw_items
        await update_user_memory_fields(target_uid, favorite_topics=favs)
        return (
            format_user_memory_card(target_uid, header_title="เพิ่มหัวข้อที่ชอบเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_edit_pinned = re.match(
        r"^(?:แก้|แก้ไข|อัปเดต)\s*(?:ข้อมูลถาวร|ปักหมุด)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_edit_pinned:
        val = m_edit_pinned.group(1).strip()
        new_pinned = [] if val in ("-", "ว่าง", "none", "clear") else [x.strip() for x in re.split(r"[;\n]+", val) if x.strip()]
        await update_user_memory_fields(target_uid, pinned_facts=new_pinned)
        return (
            format_user_memory_card(target_uid, header_title="อัปเดตข้อมูลถาวรที่ปักหมุดเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_edit_fav = re.match(
        r"^(?:แก้|แก้ไข|อัปเดต)\s*(?:หัวข้อที่ชอบ|ความสนใจ)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_edit_fav:
        val = m_edit_fav.group(1).strip()
        new_favs = [] if val in ("-", "ว่าง", "none", "clear") else [x.strip() for x in re.split(r"[,;،]+", val) if x.strip()]
        await update_user_memory_fields(target_uid, favorite_topics=new_favs)
        return (
            format_user_memory_card(target_uid, header_title="อัปเดตหัวข้อที่ชอบเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_edit_notes = re.match(
        r"^(?:แก้|แก้ไข|อัปเดต)\s*(?:บันทึกความจำ|บันทึก|โน้ต)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_edit_notes:
        val = m_edit_notes.group(1).strip()
        new_notes = [] if val in ("-", "ว่าง", "none", "clear") else [x.strip() for x in re.split(r"[;\n]+", val) if x.strip()]
        await update_user_memory_fields(target_uid, custom_notes=new_notes)
        return (
            format_user_memory_card(target_uid, header_title="อัปเดตบันทึกความจำเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_edit_recent = re.match(
        r"^(?:แก้|แก้ไข|อัปเดต)\s*(?:เรื่องล่าสุด|หัวข้อล่าสุด)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_edit_recent:
        val = m_edit_recent.group(1).strip()
        new_recents = [] if val in ("-", "ว่าง", "none", "clear") else [x.strip() for x in re.split(r"[;\n]+", val) if x.strip()]
        await update_user_memory_fields(target_uid, recent_topics=new_recents)
        return (
            format_user_memory_card(target_uid, header_title="อัปเดตเรื่องที่คุยล่าสุดเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_edit_name = re.match(
        r"^(?:แก้|แก้ไข|อัปเดต)\s*(?:ชื่อแสดงผล|ชื่อในความจำ)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_edit_name:
        await update_user_memory_fields(target_uid, display_name=m_edit_name.group(1).strip())
        return (
            format_user_memory_card(target_uid, header_title="อัปเดตชื่อที่แสดงในความจำเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_del_pinned = re.match(
        r"^(?:ลบ|ยกเลิก)\s*(?:ข้อมูลถาวร|ปักหมุด)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_del_pinned:
        target_tok = m_del_pinned.group(1).strip()
        pinned = list(prof.get("pinned_facts") or [])
        cleaned_idx = re.sub(r"^[pP]", "", target_tok)
        if target_tok.lower() in ("ทั้งหมด", "all"):
            pinned = []
        elif cleaned_idx.isdigit() and 1 <= int(cleaned_idx) <= len(pinned):
            pinned.pop(int(cleaned_idx) - 1)
        else:
            pinned = [p for p in pinned if target_tok.lower() not in p.lower()]
        await update_user_memory_fields(target_uid, pinned_facts=pinned)
        return (
            format_user_memory_card(target_uid, header_title="ลบรายการข้อมูลถาวรเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_del_note = re.match(
        r"^(?:ลบ|ยกเลิก)\s*(?:บันทึกความจำ|บันทึก|โน้ต)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_del_note:
        target_tok = m_del_note.group(1).strip()
        notes = list(prof.get("custom_notes") or [])
        if target_tok.lower() in ("ทั้งหมด", "all"):
            notes = []
        elif target_tok.isdigit() and 1 <= int(target_tok) <= len(notes):
            notes.pop(int(target_tok) - 1)
        else:
            notes = [n for n in notes if target_tok.lower() not in n.lower()]
        await update_user_memory_fields(target_uid, custom_notes=notes)
        return (
            format_user_memory_card(target_uid, header_title="ลบรายการบันทึกความจำเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    m_del_fav = re.match(
        r"^(?:ลบ|ยกเลิก)\s*(?:หัวข้อที่ชอบ|ความสนใจ)\s*[:：-]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if m_del_fav:
        target_tok = m_del_fav.group(1).strip()
        favs = list(prof.get("favorite_topics") or [])
        if target_tok.lower() in ("ทั้งหมด", "all"):
            favs = []
        elif target_tok.isdigit() and 1 <= int(target_tok) <= len(favs):
            favs.pop(int(target_tok) - 1)
        else:
            favs = [f for f in favs if target_tok.lower() not in f.lower()]
        await update_user_memory_fields(target_uid, favorite_topics=favs)
        return (
            format_user_memory_card(target_uid, header_title="ลบหัวข้อที่ชอบเรียบร้อยแล้วค่ะ"),
            target_uid,
        )

    return format_user_memory_card(target_uid), target_uid


# ==============================================================================
# OVERTIME (OT) TRACKER & MONTHLY TABLE SUMMARIZER
# ==============================================================================
OT_TRIGGER_REGEX = re.compile(
    r"^(?:จด|บันทึก|ลง|เพิ่ม|แก้|อัปเดต|สรุป|ดู|เช็ค|ตาราง|ยอด|ลบ|ยกเลิก)\s*(?:เวลา\s*)?(?:ot|โอที)\b",
    re.IGNORECASE,
)
OT_SUMMARY_REGEX = re.compile(
    r"^(?:สรุป|ดู|เช็ค|ตาราง|ยอด)\s*(?:เวลา\s*)?(?:ot|โอที)\b|(?:ot|โอที)\s*เดือน",
    re.IGNORECASE,
)
OT_DELETE_REGEX = re.compile(
    r"^(?:ลบ|ยกเลิก)\s*(?:เวลา\s*)?(?:ot|โอที)\b",
    re.IGNORECASE,
)
WEEKEND_TIME_REPLY_REGEX = re.compile(
    r"^(?:เริ่ม(?:\s*เวลา)?|ตั้งแต่(?:\s*เวลา)?)?\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:น\.?|โมง)?"
    r"(?:\s*(?:-|ถึง|to)\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:น\.?|โมง)?)?\s*(.*)$",
    re.IGNORECASE,
)

pending_weekend_ot: dict[tuple[int, int], dict] = {}

_OT_INVALID_TIME_REPLY = (
    "เวลาที่พิมพ์มาไม่ถูกต้องนะคะ ใส่เป็น `HH.MM` หรือ `HH:MM` เช่น `09.00` หรือ `18.00 - 20.00` ค่ะ"
)

THAI_MONTH_NAME_TO_NUM = {name: idx for idx, name in enumerate(THAI_MONTHS) if idx > 0}
THAI_SHORT_MONTHS = {
    "ม.ค.": 1, "ม.ค": 1, "มค": 1,
    "ก.พ.": 2, "ก.พ": 2, "กพ": 2,
    "มี.ค.": 3, "มี.ค": 3, "มีค": 3,
    "เม.ย.": 4, "เม.ย": 4, "เมย": 4,
    "พ.ค.": 5, "พ.ค": 5, "พค": 5,
    "มิ.ย.": 6, "มิ.ย": 6, "มิย": 6,
    "ก.ค.": 7, "ก.ค": 7, "กค": 7,
    "ส.ค.": 8, "ส.ค": 8, "สค": 8,
    "ก.ย.": 9, "ก.ย": 9, "กย": 9,
    "ต.ค.": 10, "ต.ค": 10, "ตค": 10,
    "พ.ย.": 11, "พ.ย": 11, "พย": 11,
    "ธ.ค.": 12, "ธ.ค": 12, "ธค": 12,
}
ALL_THAI_MONTH_TOKENS = {**THAI_MONTH_NAME_TO_NUM, **THAI_SHORT_MONTHS}
_MONTH_PATTERN_STR = "|".join(
    re.escape(k) for k in sorted(ALL_THAI_MONTH_TOKENS.keys(), key=len, reverse=True)
)
THAI_SHORT_WEEKDAYS = ("จ.", "อ.", "พ.", "พฤ.", "ศ.", "ส.", "อา.")

# Colloquial Thai spoken-time parser helpers (converts '6 โมงถึงทุ่มนึง' -> '18.00 ถึง 19.00')
_THAI_SPOKEN_NUM_MAP = {
    "หนึ่ง": 1, "นึง": 1, "เอ็ด": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5,
    "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10, "สิบเอ็ด": 11,
}
_THAI_NUM_TOKEN_PAT = r"(?:สิบเอ็ด|สิบ|หนึ่ง|นึง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|1[0-1]|[1-9])"
_THAI_MIN_SUFFIX_PAT = r"(?:\s*(ครึ่ง)|\s*(\d{1,2})\s*นาที|\s*ตรง)?"


def _parse_thai_small_num(tok: str | None) -> int:
    if not tok:
        return 1
    t = tok.strip()
    if t.isdigit():
        return int(t)
    return _THAI_SPOKEN_NUM_MAP.get(t, 1)


def _parse_thai_minutes_suffix(half_grp: str | None, min_grp: str | None) -> int:
    if half_grp:
        return 30
    if min_grp and min_grp.isdigit():
        return min(59, max(0, int(min_grp)))
    return 0


def _normalize_colloquial_thai_times_in_ot_text(text: str) -> str:
    """
    Converts spoken Thai time expressions into 24-hour `HH.MM` format so OT parsing works
    identically for voice commands and casual Thai text:
    - '6 โมงถึงทุ่มนึง' -> '18.00 ถึง 19.00'
    - 'หกโมงเย็นถึงสองทุ่มครึ่ง' -> '18.00 ถึง 20.30'
    - 'บ่ายโมงถึงสี่โมงเย็น' -> '13.00 ถึง 16.00'
    - '9 โมงเช้าถึงเที่ยงครึ่ง' -> '09.00 ถึง 12.30'
    """
    if not text:
        return text
    s = text

    # 1. เที่ยงคืน -> 00.MM
    s = re.sub(
        rf"เที่ยงคืน{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" 00.{_parse_thai_minutes_suffix(m.group(1), m.group(2)):02d} ",
        s,
    )
    # 2. เที่ยง / เที่ยงวัน -> 12.MM
    s = re.sub(
        rf"เที่ยง(?:วัน)?{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" 12.{_parse_thai_minutes_suffix(m.group(1), m.group(2)):02d} ",
        s,
    )
    # 3. ตี 1..5 -> 01..05.MM
    s = re.sub(
        rf"ตี\s*(หนึ่ง|สอง|สาม|สี่|ห้า|[1-5]){_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" {_parse_thai_small_num(m.group(1)):02d}.{_parse_thai_minutes_suffix(m.group(2), m.group(3)):02d} ",
        s,
    )
    # 4. บ่ายโมง (13:00) & บ่าย 1..5 โมง (13..17)
    s = re.sub(
        rf"บ่าย\s*โมง{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" 13.{_parse_thai_minutes_suffix(m.group(1), m.group(2)):02d} ",
        s,
    )
    s = re.sub(
        rf"บ่าย\s*(หนึ่ง|สอง|สาม|สี่|ห้า|[1-5])\s*(?:โมง)?{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" {12 + _parse_thai_small_num(m.group(1)):02d}.{_parse_thai_minutes_suffix(m.group(2), m.group(3)):02d} ",
        s,
    )
    # 5. X ทุ่ม (1..5 ทุ่ม -> 19..23)
    s = re.sub(
        rf"(หนึ่ง|สอง|สาม|สี่|ห้า|[1-5])\s*ทุ่ม{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" {18 + _parse_thai_small_num(m.group(1)):02d}.{_parse_thai_minutes_suffix(m.group(2), m.group(3)):02d} ",
        s,
    )
    # 6. ทุ่มนึง / ทุ่มหนึ่ง / ทุ่มครึ่ง / ทุ่ม XX นาที -> 19.MM
    s = re.sub(
        rf"ทุ่ม(?:\s*(?:นึง|หนึ่ง))?{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" 19.{_parse_thai_minutes_suffix(m.group(1), m.group(2)):02d} ",
        s,
    )
    # 7. X โมงเย็น (4..6 โมงเย็น -> 16..18)
    s = re.sub(
        rf"(สี่|ห้า|หก|[4-6])\s*โมง\s*เย็น{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" {12 + _parse_thai_small_num(m.group(1)):02d}.{_parse_thai_minutes_suffix(m.group(2), m.group(3)):02d} ",
        s,
    )
    # 8. X โมงเช้า (6..11 โมงเช้า -> 06..11)
    s = re.sub(
        rf"({_THAI_NUM_TOKEN_PAT})\s*โมง\s*เช้า{_THAI_MIN_SUFFIX_PAT}",
        lambda m: f" {_parse_thai_small_num(m.group(1)):02d}.{_parse_thai_minutes_suffix(m.group(2), m.group(3)):02d} ",
        s,
    )
    # 9. Bare 'X โมง' in OT context (1..6 โมง -> 13..18 evening/afternoon; 7..11 โมง -> 07..11 morning)
    def _bare_mong_repl(m):
        h_raw = _parse_thai_small_num(m.group(1))
        mm = _parse_thai_minutes_suffix(m.group(2), m.group(3))
        hh = (12 + h_raw) if 1 <= h_raw <= 6 else h_raw
        return f" {hh:02d}.{mm:02d} "

    s = re.sub(
        rf"({_THAI_NUM_TOKEN_PAT})\s*โมง{_THAI_MIN_SUFFIX_PAT}",
        _bare_mong_repl,
        s,
    )

    return re.sub(r"\s+", " ", s).strip()


def _normalize_ot_year_ce(yy_str: str | None, default_year_ce: int) -> int:
    if not yy_str:
        return default_year_ce
    y = int(yy_str)
    if y > 2400:
        return y - 543
    if y < 100:
        return (y + 2500 - 543) if y >= 60 else (2000 + y)
    return y


def _format_date_thai_ddmmyyyy(dt: datetime) -> str:
    return f"{dt.day:02d}/{dt.month:02d}/{dt.year + 543:04d}"


def _parse_stored_ot_date(date_str: str) -> tuple[int, int, int] | None:
    if not date_str:
        return None
    s = str(date_str).strip()
    m_dmy = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m_dmy:
        dd, mm, yyyy = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        year_ce = (yyyy - 543) if yyyy > 2400 else yyyy
        return (year_ce, mm, dd)
    m_iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m_iso:
        yyyy, mm, dd = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        year_ce = (yyyy - 543) if yyyy > 2400 else yyyy
        return (year_ce, mm, dd)
    return None


def _normalize_stored_date_to_thai_ddmmyyyy(date_str: str) -> str:
    parsed = _parse_stored_ot_date(date_str)
    if not parsed:
        return str(date_str)
    y_ce, mm, dd = parsed
    return f"{dd:02d}/{mm:02d}/{y_ce + 543:04d}"


def _format_time_dot(dt_or_str) -> str:
    if isinstance(dt_or_str, datetime):
        return dt_or_str.strftime("%H.%M")
    return str(dt_or_str).strip().replace(":", ".")


def _extract_custom_date_from_text(text: str, default_dt: datetime) -> tuple[datetime | None, str]:
    s = text.strip()

    m_yest = re.match(r"^(?:ของ)?(?:เมื่อวาน(?:นี้)?|วานนี้)\b\s*(.*)", s)
    if m_yest:
        return default_dt - timedelta(days=1), m_yest.group(1).strip()

    m_iso = re.match(r"^(?:วันที่\s*)?(\d{4})-(\d{1,2})-(\d{1,2})\b\s*(.*)", s)
    if m_iso:
        yy = _normalize_ot_year_ce(m_iso.group(1), default_dt.year)
        mm, dd, rem = int(m_iso.group(2)), int(m_iso.group(3)), m_iso.group(4)
        try:
            return default_dt.replace(year=yy, month=mm, day=dd), rem.strip()
        except ValueError:
            pass

    m_wan_duan = re.match(
        rf"^(?:ของ)?วันที่\s*(\d{{1,2}})\s*เดือน(?:ที่)?\s*(\d{{1,2}}|{_MONTH_PATTERN_STR})"
        r"(?:\s*(?:ปี|พ\.ศ\.?|ค\.ศ\.?)?\s*(\d{2,4})(?!\s*[:.]\d{2}))?\s*(.*)",
        s,
    )
    if m_wan_duan:
        dd = int(m_wan_duan.group(1))
        m_tok = m_wan_duan.group(2).strip()
        mm = int(m_tok) if m_tok.isdigit() else ALL_THAI_MONTH_TOKENS.get(m_tok, default_dt.month)
        yy = _normalize_ot_year_ce(m_wan_duan.group(3), default_dt.year)
        rem = m_wan_duan.group(4).strip()
        try:
            return default_dt.replace(year=yy, month=mm, day=dd), rem
        except ValueError:
            pass

    m_th_month = re.match(
        rf"^(?:(?:ของ)?วันที่\s*)?(\d{{1,2}})\s*({_MONTH_PATTERN_STR})"
        r"(?:\s*(\d{4}|\d{2})(?!\s*[:.]\d{2}))?\s*(.*)",
        s,
    )
    if m_th_month:
        dd = int(m_th_month.group(1))
        m_tok = m_th_month.group(2).strip()
        mm = ALL_THAI_MONTH_TOKENS.get(m_tok, default_dt.month)
        yy = _normalize_ot_year_ce(m_th_month.group(3), default_dt.year)
        rem = m_th_month.group(4).strip()
        try:
            return default_dt.replace(year=yy, month=mm, day=dd), rem
        except ValueError:
            pass

    m_slash = re.match(
        r"^(?:(?:ของ)?วันที่\s*)?(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b\s*(.*)",
        s,
    )
    if m_slash:
        dd, mm = int(m_slash.group(1)), int(m_slash.group(2))
        yy = _normalize_ot_year_ce(m_slash.group(3), default_dt.year)
        rem = m_slash.group(4).strip()
        try:
            return default_dt.replace(year=yy, month=mm, day=dd), rem
        except ValueError:
            pass

    m_day_only = re.match(r"^(?:ของ)?วันที่\s*(\d{1,2})(?!\s*[:.])\b\s*(.*)", s)
    if m_day_only:
        dd = int(m_day_only.group(1))
        rem = m_day_only.group(2).strip()
        try:
            return default_dt.replace(day=dd), rem
        except ValueError:
            pass

    return None, s


def _has_active_pending_weekend_ot(channel_id: int | None, user_id: int | None) -> bool:
    if not channel_id or not user_id:
        return False
    key = (int(channel_id), int(user_id))
    pending = pending_weekend_ot.get(key)
    if not pending:
        return False
    if time.time() - pending.get("timestamp", 0) > 600:
        pending_weekend_ot.pop(key, None)
        return False
    return True


def is_ot_command_text(text: str, channel_id: int = None, user_id: int = None) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"^<@!?\d+>\s*", "", text.strip()).strip()
    if OT_TRIGGER_REGEX.search(cleaned):
        return True
    if _has_active_pending_weekend_ot(channel_id, user_id):
        norm_c = _normalize_colloquial_thai_times_in_ot_text(cleaned)
        if WEEKEND_TIME_REPLY_REGEX.match(norm_c) or cleaned.lower() in ("ยกเลิก", "cancel"):
            return True
    return False


def _format_hours_human(hours_val: float) -> str:
    total_mins = int(round(hours_val * 60))
    h, m = divmod(total_mins, 60)
    if m == 0:
        return f"{h} ชม."
    if h == 0:
        return f"{m} นาที"
    return f"{h} ชม. {m} นาที ({hours_val:.2f} ชม.)"


async def recover_ot_logs_from_work_channels(bot: discord.Client):
    recovered_count = 0
    pattern = re.compile(
        r"\[OT_RECORD:\s*uid=(\d+)\s*\|\s*date=([0-9/\-]+)\s*\|\s*"
        r"slot=([0-9:.]+-[0-9:.]+)\s*\|\s*hours=([0-9.]+)\s*\|\s*detail=(.+?)\]"
    )
    for guild in bot.guilds:
        for ch in guild.text_channels:
            if not any(k in ch.name.lower() for k in ("work", "ot", "งาน")):
                continue
            try:
                async for msg in ch.history(limit=120):
                    if not bot.user or msg.author.id != bot.user.id or not msg.content:
                        continue
                    for m in pattern.finditer(msg.content):
                        uid_s, raw_d, slot_s, hrs_s, det_s = m.groups()
                        norm_date = _normalize_stored_date_to_thai_ddmmyyyy(raw_d)
                        parsed_key = _parse_stored_ot_date(norm_date)
                        if not parsed_key:
                            continue
                        start_s, end_s = (slot_s.split("-") + ["18.00", "21.00"])[:2]
                        start_s = _format_time_dot(start_s)
                        end_s = _format_time_dot(end_s)
                        prof = ensure_user_profile_exists(uid_s)
                        prof.setdefault("ot_channel_id", ch.id)
                        ot_list = prof.setdefault("ot_logs", [])
                        if not any(_parse_stored_ot_date(x.get("date", "")) == parsed_key for x in ot_list):
                            ot_list.append({
                                "date": norm_date,
                                "noted_start": start_s,
                                "noted_end": end_s,
                                "hours": float(hrs_s),
                                "detail": det_s.strip(),
                            })
                            recovered_count += 1
            except Exception:
                continue

    if recovered_count > 0:
        for prof in user_profiles.values():
            if "ot_logs" in prof:
                prof["ot_logs"].sort(key=lambda x: _parse_stored_ot_date(x.get("date", "")) or (0, 0, 0))
        await save_user_profiles_local_async()
        print(f"[OT Tracker] Recovered {recovered_count} OT record(s) from #work channel history.", flush=True)


async def _save_and_format_ot_entry(
    prof: dict,
    ot_logs: list[dict],
    uid_str: str,
    work_dt: datetime,
    noted_start_str: str,
    noted_end_str: str,
    ot_hours: float,
    clean_detail: str,
    has_new_detail: bool,
    time_note: str,
    channel_id: int | None = None,
) -> str:
    global _DIRTY_SINCE_LAST_BACKUP
    target_date_th = _format_date_thai_ddmmyyyy(work_dt)
    target_key = (work_dt.year, work_dt.month, work_dt.day)

    noted_start_str = _format_time_dot(noted_start_str)
    noted_end_str = _format_time_dot(noted_end_str)

    updated_existing = False
    for entry in ot_logs:
        if _parse_stored_ot_date(entry.get("date", "")) == target_key:
            entry["date"] = target_date_th
            entry["noted_start"] = noted_start_str
            entry["noted_end"] = noted_end_str
            entry["hours"] = ot_hours
            if has_new_detail:
                entry["detail"] = clean_detail
            else:
                clean_detail = entry.get("detail") or clean_detail
            updated_existing = True
            break

    if not updated_existing:
        ot_logs.append({
            "date": target_date_th,
            "noted_start": noted_start_str,
            "noted_end": noted_end_str,
            "hours": ot_hours,
            "detail": clean_detail,
        })

    for entry in ot_logs:
        entry["date"] = _normalize_stored_date_to_thai_ddmmyyyy(entry.get("date", ""))
        entry["noted_start"] = _format_time_dot(entry.get("noted_start", "18.00"))
        entry["noted_end"] = _format_time_dot(entry.get("noted_end", "-"))

    ot_logs.sort(key=lambda x: _parse_stored_ot_date(x.get("date", "")) or (0, 0, 0))
    prof["ot_logs"] = ot_logs
    if channel_id:
        prof["ot_channel_id"] = int(channel_id)
    _DIRTY_SINCE_LAST_BACKUP = True
    await save_user_profiles_local_async()
    await backup_user_profiles_to_discord(force=True)

    month_total = 0.0
    month_days = 0
    for e in ot_logs:
        pk = _parse_stored_ot_date(e.get("date", ""))
        if pk and pk[0] == work_dt.year and pk[1] == work_dt.month:
            month_total += float(e.get("hours") or 0.0)
            month_days += 1

    action_word = "อัปเดตบันทึกเวลาทำ OT" if updated_existing else "บันทึกเวลาทำ OT เรียบร้อยแล้วค่ะ"
    hrs_display = _format_hours_human(ot_hours)
    month_total_display = _format_hours_human(month_total)
    weekday_th = THAI_WEEKDAYS[work_dt.weekday()]

    return (
        f"⏱️ **[{action_word}]**\n"
        f"- **วันที่:** `{target_date_th}` ({weekday_th})\n"
        f"- **เวลาที่บันทึก (OT):** **`{noted_start_str} - {noted_end_str} น.`** {time_note}\n"
        f"- **จำนวนชั่วโมง:** **`{hrs_display}`**\n"
        f"- **รายละเอียดงาน:** `{clean_detail}`\n"
        f"- **สะสมเดือน{THAI_MONTHS[work_dt.month]} {work_dt.year + 543}:** `{month_days} วัน` | รวม **`{month_total_display}`**\n"
        f"-# [OT_RECORD: uid={uid_str} | date={target_date_th} | slot={noted_start_str}-{noted_end_str} | hours={ot_hours:.2f} | detail={clean_detail}]"
    )


def build_monthly_ot_summary_markdown(ot_logs: list[dict], target_year: int, target_month: int) -> str:
    month_entries = []
    for e in ot_logs:
        pk = _parse_stored_ot_date(e.get("date", ""))
        if pk and pk[0] == target_year and pk[1] == target_month:
            month_entries.append(e)

    month_entries.sort(key=lambda x: _parse_stored_ot_date(x.get("date", "")) or (0, 0, 0))
    month_th = THAI_MONTHS[target_month]
    year_be = target_year + 543

    if not month_entries:
        return (
            f"📊 **สรุปเวลาทำ OT ประจำเดือน{month_th} พ.ศ. {year_be}**\n"
            f"ยังไม่มีรายการบันทึก OT ในเดือน{month_th}เลยค่ะ หากวันไหนอยู่ทำงานล่วงเวลา พิมพ์ `จด OT <รายละเอียดงาน>` ทิ้งไว้ได้เลยนะคะ"
        )

    table_lines = [
        f"📊 **ตารางสรุปเวลาทำ OT ประจำเดือน{month_th} พ.ศ. {year_be} ({target_month:02d}/{year_be})**",
        "",
        "| ลำดับ | วันที่ | เวลาที่บันทึก (OT) | จำนวนชั่วโมง | รายละเอียดงาน |",
        "| :---: | :---: | :---: | :---: | :--- |",
    ]
    total_hours = 0.0
    for idx, entry in enumerate(month_entries, 1):
        d_th = _normalize_stored_date_to_thai_ddmmyyyy(entry.get("date", ""))
        n_start = _format_time_dot(entry.get("noted_start", "18.00"))
        n_end = _format_time_dot(entry.get("noted_end", "-"))
        hrs = float(entry.get("hours") or 0.0)
        total_hours += hrs
        hrs_label = f"{int(hrs)} ชม." if hrs.is_integer() else f"{hrs:.2f} ชม."
        det = (entry.get("detail") or "-").replace("|", "\\|")
        table_lines.append(
            f"| {idx} | {d_th} | {n_start} - {n_end} | {hrs_label} | {det} |"
        )

    days_count = len(month_entries)
    avg_hours = total_hours / days_count if days_count else 0.0
    total_human = _format_hours_human(total_hours)

    table_lines.extend([
        "",
        "**📌 สรุปยอดรวมประจำเดือน:**",
        f"- **จำนวนวันที่ทำ OT:** `{days_count} วัน`",
        f"- **รวมชั่วโมง OT ทั้งสิ้น:** **`{total_human}`** (เฉลี่ยวันละ `{avg_hours:.1f} ชม.`)",
        "-# อย่าหักโหมจนเสียงานวันพรุ่งนี้ก็แล้วกันนะคะ",
    ])
    return "\n".join(table_lines)


async def process_ot_request(message: discord.Message, raw_text: str) -> str:
    global _DIRTY_SINCE_LAST_BACKUP
    user = message.author
    uid_str = str(user.id)
    chan_id = int(getattr(getattr(message, "channel", None), "id", 0) or 0)
    pending_key = (chan_id, int(user.id))

    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    default_work_dt = (now_th - timedelta(days=1)) if now_th.hour < 5 else now_th

    text = re.sub(r"^<@!?\d+>\s*", "", raw_text.strip()).strip()
    text = re.sub(r"^!ot\b\s*", "", text, flags=re.IGNORECASE).strip()

    if uid_str not in user_profiles:
        record_user_interaction(user, text, channel_id=chan_id or None)
    prof = user_profiles[uid_str]
    ot_logs: list[dict] = list(prof.get("ot_logs") or [])

    if _has_active_pending_weekend_ot(chan_id, user.id) and not OT_TRIGGER_REGEX.search(text):
        if text.lower() in ("ยกเลิก", "cancel"):
            pending_weekend_ot.pop(pending_key, None)
            return "ยกเลิกการรอเวลาเริ่มทำ OT ให้แล้วค่ะ"

        norm_reply = _normalize_colloquial_thai_times_in_ot_text(text)
        tm_match = WEEKEND_TIME_REPLY_REGEX.match(norm_reply)
        if tm_match:
            pending_data = pending_weekend_ot.pop(pending_key)
            saved_work_dt: datetime = pending_data["work_dt"]
            saved_end_dt: datetime = pending_data["end_dt"]
            saved_detail: str = pending_data["detail"]
            is_backdated_pending: bool = pending_data.get("is_backdated", False)

            sh = int(tm_match.group(1))
            sm = int(tm_match.group(2) or 0)
            eh_str, em_str = tm_match.group(3), tm_match.group(4)
            extra_detail = (tm_match.group(5) or "").strip(" -:")

            if not (0 <= sh <= 23 and 0 <= sm <= 59) or int(em_str or 0) > 59:
                pending_weekend_ot[pending_key] = pending_data
                return _OT_INVALID_TIME_REPLY

            start_dt = saved_work_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
            if eh_str is not None:
                eh = int(eh_str)
                em = int(em_str or 0)
                end_dt = saved_work_dt.replace(hour=eh % 24, minute=em, second=0, microsecond=0)
                if end_dt <= start_dt:
                    end_dt += timedelta(days=1)
            else:
                end_dt = saved_work_dt.replace(
                    hour=saved_end_dt.hour, minute=saved_end_dt.minute, second=0, microsecond=0
                )
                if saved_end_dt.hour < 5:
                    end_dt += timedelta(days=1)

            diff_mins = int(round((end_dt - start_dt).total_seconds() / 60.0))
            if diff_mins <= 0:
                pending_weekend_ot[pending_key] = pending_data
                return (
                    f"เนื่องจากเวลาปัจจุบันคือ `{_format_time_dot(saved_end_dt)} น.` ซึ่งไม่ได้อยู่หลังเวลาเริ่ม `{sh:02d}.{sm:02d} น.` "
                    f"ใส่เป็น**ช่วงเวลาเริ่ม-เลิก**แทนนะคะ เช่น `{sh:02d}.{sm:02d} - 21.00` ค่ะ"
                )

            ot_hours = round(diff_mins / 60.0, 2)
            default_label = "ทำ OT ย้อนหลัง" if is_backdated_pending else "ทำ OT วันหยุด"
            final_detail = extra_detail or saved_detail or f"{default_label} (ไม่ได้ระบุรายละเอียด)"
            note_label = "*(บันทึกย้อนหลัง)*" if is_backdated_pending else "*(เวลาทำ OT วันหยุดตามจริง)*"
            return await _save_and_format_ot_entry(
                prof,
                ot_logs,
                uid_str,
                saved_work_dt,
                f"{sh:02d}.{sm:02d}",
                _format_time_dot(end_dt),
                ot_hours,
                final_detail,
                has_new_detail=bool(extra_detail or saved_detail),
                time_note=note_label,
                channel_id=chan_id or None,
            )

    if OT_DELETE_REGEX.search(text) or text.lower().startswith(("delete ", "remove ", "ลบ")):
        pending_weekend_ot.pop(pending_key, None)
        del_body = re.sub(r"^(?:ลบ|ยกเลิก|delete|remove)\s*(?:เวลา\s*)?(?:ot|โอที)?\s*", "", text, flags=re.IGNORECASE).strip()
        custom_del_dt, _ = _extract_custom_date_from_text(del_body, default_work_dt)
        target_dt = custom_del_dt or default_work_dt
        target_key = (target_dt.year, target_dt.month, target_dt.day)
        target_date_th = _format_date_thai_ddmmyyyy(target_dt)

        before_len = len(ot_logs)
        ot_logs = [
            entry for entry in ot_logs
            if _parse_stored_ot_date(entry.get("date", "")) != target_key
        ]
        prof["ot_logs"] = ot_logs
        _DIRTY_SINCE_LAST_BACKUP = True
        await save_user_profiles_local_async()
        await backup_user_profiles_to_discord(force=True)

        if len(ot_logs) < before_len:
            return f"🗑️ ลบรายการบันทึก OT ของวันที่ **`{target_date_th}`** ออกให้เรียบร้อยแล้วค่ะ"
        return f"ไม่พบรายการบันทึก OT ของวันที่ **`{target_date_th}`** ในระบบค่ะ"

    if not text or OT_SUMMARY_REGEX.search(text) or text.lower().startswith(("summary", "list", "table", "สรุป", "ดู", "ตาราง")):
        target_year = default_work_dt.year
        target_month = default_work_dt.month

        ym_m = re.search(r"(\d{4})-(\d{1,2})", text)
        m_slash_ym = re.search(r"\b(\d{1,2})/(\d{4})\b", text)
        m_num_m = re.search(r"เดือน(?:ที่)?\s*(\d{1,2})", text)
        if ym_m:
            target_year = _normalize_ot_year_ce(ym_m.group(1), default_work_dt.year)
            target_month = int(ym_m.group(2))
        elif m_slash_ym:
            target_month = int(m_slash_ym.group(1))
            target_year = _normalize_ot_year_ce(m_slash_ym.group(2), default_work_dt.year)
        elif m_num_m and 1 <= int(m_num_m.group(1)) <= 12:
            target_month = int(m_num_m.group(1))
        else:
            for m_name, m_idx in THAI_MONTH_NAME_TO_NUM.items():
                if m_name in text:
                    target_month = m_idx
                    break
            else:
                for s_name, m_idx in THAI_SHORT_MONTHS.items():
                    if s_name in text:
                        target_month = m_idx
                        break

        return build_monthly_ot_summary_markdown(ot_logs, target_year, target_month)

    detail_str = re.sub(
        r"^(?:จด|บันทึก|ลง|เพิ่ม|แก้|อัปเดต)\s*(?:เวลา\s*)?(?:ot|โอที)\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    custom_dt, detail_str = _extract_custom_date_from_text(detail_str, default_work_dt)
    is_custom_date = (custom_dt is not None)
    work_dt = custom_dt if is_custom_date else default_work_dt
    target_date_th = _format_date_thai_ddmmyyyy(work_dt)
    is_weekend = (work_dt.weekday() >= 5)

    # Convert colloquial spoken Thai times ('6 โมงถึงทุ่มนึง' -> '18.00 ถึง 19.00') before range/until matching
    detail_str = _normalize_colloquial_thai_times_in_ot_text(detail_str)

    range_time_m = re.match(
        r"^(?:ตั้งแต่(?:\s*เวลา)?|เริ่ม(?:\s*เวลา)?)?\s*(\d{1,2})[:.](\d{2})\s*(?:น\.?)?\s*(?:-|ถึง|to)\s*(\d{1,2})[:.](\d{2})(?:\s*น\.?)?\s*(.*)",
        detail_str,
        flags=re.IGNORECASE,
    )
    if range_time_m:
        sh, sm = int(range_time_m.group(1)), int(range_time_m.group(2))
        eh, em = int(range_time_m.group(3)), int(range_time_m.group(4))
        if sm > 59 or em > 59:
            return _OT_INVALID_TIME_REPLY
        pending_weekend_ot.pop(pending_key, None)
        rem_detail = range_time_m.group(5).strip(" -:")
        start_dt = work_dt.replace(hour=sh % 24, minute=sm, second=0, microsecond=0)
        end_dt = work_dt.replace(hour=eh % 24, minute=em, second=0, microsecond=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        diff_mins = max(15, int(round((end_dt - start_dt).total_seconds() / 60.0)))
        ot_hours = round(diff_mins / 60.0, 2)
        clean_detail = rem_detail or "ทำ OT (ไม่ได้ระบุรายละเอียด)"
        note_str = "*(บันทึกย้อนหลัง)*" if is_custom_date else "*(ระบุช่วงเวลาโดยตรง)*"
        return await _save_and_format_ot_entry(
            prof,
            ot_logs,
            uid_str,
            work_dt,
            f"{sh:02d}.{sm:02d}",
            f"{eh:02d}.{em:02d}",
            ot_hours,
            clean_detail,
            has_new_detail=bool(rem_detail),
            time_note=note_str,
            channel_id=chan_id or None,
        )

    if is_weekend or is_custom_date:
        wknd_start_m = re.match(
            r"^(?:เริ่ม(?:\s*เวลา)?|ตั้งแต่(?:\s*เวลา)?)\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:น\.?|โมง)?\s*(.*)",
            detail_str,
            flags=re.IGNORECASE,
        ) or re.match(
            r"^(\d{1,2})[:.](\d{2})(?:\s*น\.?)?(?:\s+|$)(.*)",
            detail_str,
        )

        if wknd_start_m:
            sh = int(wknd_start_m.group(1))
            sm = int(wknd_start_m.group(2) or 0)
            if sm > 59:
                return _OT_INVALID_TIME_REPLY
            pending_weekend_ot.pop(pending_key, None)
            rem_detail = (wknd_start_m.group(3) or "").strip(" -:")
            start_dt = work_dt.replace(hour=sh % 24, minute=sm, second=0, microsecond=0)

            dur_m = re.match(
                r"^(\d+(?:\.\d+)?)\s*(?:ชั่วโมง|ชม\.?|hrs?\.?|hours?|h)\b\s*(.*)",
                rem_detail,
                flags=re.IGNORECASE,
            )
            if dur_m:
                ot_hours = round(float(dur_m.group(1)), 2)
                rem_detail = dur_m.group(2).strip(" -:")
                end_dt = start_dt + timedelta(minutes=int(round(ot_hours * 60)))
            else:
                end_dt = work_dt.replace(
                    hour=now_th.hour, minute=now_th.minute, second=0, microsecond=0
                )
                if now_th.hour < 5:
                    end_dt += timedelta(days=1)
                diff_mins = int(round((end_dt - start_dt).total_seconds() / 60.0))
                if diff_mins <= 0:
                    return (
                        f"เวลาเริ่ม (`{sh:02d}.{sm:02d} น.`) ต้องมาก่อนเวลาสิ้นสุด (`{_format_time_dot(end_dt)} น.`) นะคะ "
                        f"ถ้าจดย้อนหลัง ใส่เป็นช่วงเวลาแทนนะคะ เช่น `จด OT {target_date_th} {sh:02d}.{sm:02d} - 21.00` ค่ะ"
                    )
                ot_hours = round(diff_mins / 60.0, 2)

            default_label = "ทำ OT ย้อนหลัง" if is_custom_date else "ทำ OT วันหยุด"
            clean_detail = rem_detail or f"{default_label} (ไม่ได้ระบุรายละเอียด)"
            note_str = "*(บันทึกย้อนหลัง)*" if is_custom_date else "*(เวลาทำ OT วันหยุดตามจริง)*"
            return await _save_and_format_ot_entry(
                prof,
                ot_logs,
                uid_str,
                work_dt,
                f"{sh:02d}.{sm:02d}",
                _format_time_dot(end_dt),
                ot_hours,
                clean_detail,
                has_new_detail=bool(rem_detail),
                time_note=note_str,
                channel_id=chan_id or None,
            )

        saved_detail = detail_str.strip(" -:")
        pending_weekend_ot[pending_key] = {
            "work_dt": work_dt,
            "end_dt": now_th,
            "detail": saved_detail,
            "is_backdated": is_custom_date,
            "timestamp": time.time(),
        }
        weekday_th = THAI_WEEKDAYS[work_dt.weekday()]
        detail_note = f" (งาน: `{saved_detail}`)" if saved_detail else ""

        addr_subj = "นาย" if is_owner_user(user) else ""
        if is_custom_date:
            return (
                f"🗓️ บันทึก OT ย้อนหลังของวันที่ **`{target_date_th}`** ({weekday_th}){detail_note}\n"
                f"วันนั้น{addr_subj}เริ่มทำ OT กี่โมงถึงกี่โมงคะ? **(พิมพ์ตอบแค่ช่วงเวลา เช่น `18.00 - 20.00` หรือเวลาเริ่ม เช่น `18.00` ได้เลยค่ะ)**"
            )
        return (
            f"🗓️ วันนี้เป็น**{weekday_th}** (`{target_date_th}` — วันหยุดสุดสัปดาห์) จะไม่ได้เริ่มนับจาก 17.00 น. เหมือนวันทำงานปกตินะคะ{detail_note}\n"
            f"{addr_subj}เริ่มทำ OT ตั้งแต่กี่โมงคะ? **(พิมพ์ตอบแค่เวลาเริ่ม เช่น `09.00` หรือช่วงเวลา `09.00 - 17.00` ได้เลยค่ะ เดี๋ยวฉันบันทึกให้ทันทีค่ะ)**"
        )

    explicit_hrs_m = re.match(
        r"^(\d+(?:\.\d+)?)\s*(?:ชั่วโมง|ชม\.?|hrs?\.?|hours?|h)\b\s*(.*)",
        detail_str,
        flags=re.IGNORECASE,
    )
    until_time_m = re.match(
        r"^(?:ถึง(?:\s*เวลา)?\s*)?(\d{1,2})[:.](\d{2})(?:\s*น\.?)?\s*(.*)",
        detail_str,
    )

    if explicit_hrs_m:
        ot_hours = round(float(explicit_hrs_m.group(1)), 2)
        detail_str = explicit_hrs_m.group(2).strip()
        end_dt = work_dt.replace(hour=18, minute=0, second=0, microsecond=0) + timedelta(
            minutes=int(round(ot_hours * 60))
        )
        noted_end_str = _format_time_dot(end_dt)
        actual_finish_str = _format_time_dot(end_dt - timedelta(hours=1))
    elif until_time_m:
        fin_h, fin_m = int(until_time_m.group(1)), int(until_time_m.group(2))
        if fin_m > 59:
            return _OT_INVALID_TIME_REPLY
        detail_str = until_time_m.group(3).strip()
        finish_dt = work_dt.replace(hour=fin_h % 24, minute=fin_m, second=0, microsecond=0)
        if fin_h < 5:
            finish_dt += timedelta(days=1)
        start_actual = work_dt.replace(hour=17, minute=0, second=0, microsecond=0)
        diff_mins = max(30, int(round((finish_dt - start_actual).total_seconds() / 60.0)))
        ot_hours = round(diff_mins / 60.0, 2)
        noted_end_dt = finish_dt + timedelta(hours=1)
        noted_end_str = _format_time_dot(noted_end_dt)
        actual_finish_str = _format_time_dot(finish_dt)
    else:
        start_actual = work_dt.replace(hour=17, minute=0, second=0, microsecond=0)
        raw_diff_mins = (now_th - start_actual).total_seconds() / 60.0
        if raw_diff_mins <= 0:
            return (
                f"ตอนนี้เพิ่งเวลา **{_format_time_dot(now_th)} น.** เองนะคะ (ยังไม่หลังเวลาเริ่ม OT วันทำงาน 17.00 น. เลยค่ะ) "
                "หรือหากต้องการจดย้อนหลัง สามารถระบุวันที่และช่วงเวลาได้นะคะ เช่น `จด OT 5 พ.ย. 18.00 - 20.00 แก้โปรแกรม Gen PDF` ค่ะ"
            )
        diff_mins = int(round(raw_diff_mins))
        ot_hours = round(diff_mins / 60.0, 2)
        noted_end_dt = now_th + timedelta(hours=1)
        noted_end_str = _format_time_dot(noted_end_dt)
        actual_finish_str = _format_time_dot(now_th)

    clean_detail = detail_str.strip(" -:") or "ทำ OT (ไม่ได้ระบุรายละเอียด)"
    return await _save_and_format_ot_entry(
        prof,
        ot_logs,
        uid_str,
        work_dt,
        "18.00",
        noted_end_str,
        ot_hours,
        clean_detail,
        has_new_detail=bool(detail_str.strip()),
        time_note=f"*(เวลาเลิกจริง {actual_finish_str} น. | คิดจาก 17.00 -> เลื่อน +1 ชม. เป็น 18.00 - {noted_end_str} น.)*",
        channel_id=chan_id or None,
    )