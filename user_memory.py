import os
import re
import io
import json
import time
import asyncio
import discord

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config import THAI_WEEKDAYS, THAI_MONTHS

from config import (
    GAME_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    CODE_INTENT_PATTERN,
    CAT_TOPIC_REGEX,
    WEATHER_INTENT_PATTERN,
    channel_active_topic,
    state,
)
from persona import is_owner_user, get_current_thai_datetime_str

USER_PROFILES_FILE = os.path.join(os.path.dirname(__file__), "user_profiles.json")
MEMORY_BACKUP_CHANNEL_ID = int(os.getenv("MEMORY_BACKUP_CHANNEL_ID", "0") or "0")

# In-memory store: { "<user_id_str>": { ...profile_data... } }
user_profiles: dict[str, dict] = {}
_DIRTY_SINCE_LAST_BACKUP = False
_LAST_CLOUD_BACKUP_TS = 0.0

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


def load_user_profiles_local():
    """Loads user_profiles.json from local disk if present."""
    global user_profiles
    if os.path.exists(USER_PROFILES_FILE):
        try:
            with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    user_profiles = data
                    print(f"[User Memory] Loaded {len(user_profiles)} user profile(s) from local disk.", flush=True)
        except Exception as e:
            print(f"[User Memory] Local load warning: {e}", flush=True)


def save_user_profiles_local():
    """Saves user_profiles.json to local disk."""
    try:
        with open(USER_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(user_profiles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[User Memory] Local save warning: {e}", flush=True)


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
    on startup, and also scans #work channels to recover any OT logs so records survive Render redeploys.
    """
    global user_profiles
    load_user_profiles_local()
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
                                save_user_profiles_local()
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

    # Always scan #work channels on startup to recover any OT entries
    await recover_ot_logs_from_work_channels(bot)


async def backup_user_profiles_to_discord(force: bool = False):
    """Backs up user_profiles.json to the private Discord backup channel (throttled to at most once every 3 mins)."""
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

    try:
        _LAST_CLOUD_BACKUP_TS = now
        _DIRTY_SINCE_LAST_BACKUP = False
        payload_bytes = json.dumps(user_profiles, ensure_ascii=False, indent=2).encode("utf-8")

        async for old_msg in ch.history(limit=10):
            if old_msg.author.id == bot.user.id:
                try:
                    await old_msg.delete()
                except Exception:
                    pass

        await ch.send(
            content=f"💾 **[Yukinon Persistent User Memory Backup]** (`{len(user_profiles)}` users | `{get_current_thai_datetime_str()}`)",
            file=discord.File(io.BytesIO(payload_bytes), filename="user_profiles.json"),
        )
    except Exception as e:
        print(f"[User Memory] Cloud backup note: {e}", flush=True)


def record_user_interaction(
    user,
    user_prompt: str,
    channel_id: int = None,
    learned_memo: str = "",
):
    """Updates the persistent profile of the speaker with their topics, interaction count, and optional LLM memo."""
    global _DIRTY_SINCE_LAST_BACKUP
    if user is None or getattr(user, "bot", False):
        return

    uid_str = str(user.id)
    now_str = get_current_thai_datetime_str()
    owner_flag = is_owner_user(user)
    raw_display = getattr(user, "display_name", "") or getattr(user, "name", "Unknown")
    clean_display = re.sub(r"[^\w\u0E00-\u0E7F\s.-]", "", raw_display).strip()[:32] or "Member"

    profile = user_profiles.get(uid_str) or {
        "user_id": uid_str,
        "display_name": "Partner (Owner)" if owner_flag else clean_display,
        "username": getattr(user, "name", ""),
        "is_owner": owner_flag,
        "interaction_count": 0,
        "first_seen": now_str,
        "last_seen": now_str,
        "favorite_topics": [],
        "custom_notes": [],
        "recent_topics": [],
    }

    profile["display_name"] = "Partner (Owner)" if owner_flag else clean_display
    profile["is_owner"] = owner_flag
    profile["interaction_count"] = int(profile.get("interaction_count", 0)) + 1
    profile["last_seen"] = now_str

    favs = list(profile.get("favorite_topics") or [])
    if channel_id and channel_id in channel_active_topic:
        act_t = channel_active_topic[channel_id].get("topic", "").strip()
        if act_t and act_t not in favs:
            favs.append(act_t)

    for label, pattern in TOPIC_KEYWORDS_MAP.items():
        if pattern.search(user_prompt) and label not in favs:
            favs.append(label)

    profile["favorite_topics"] = favs[-8:]

    clean_q = re.sub(r"\s+", " ", user_prompt).strip()
    if len(clean_q) > 6:
        recents = list(profile.get("recent_topics") or [])
        snippet = clean_q[:90]
        if not recents or recents[-1] != snippet:
            recents.append(snippet)
        profile["recent_topics"] = recents[-4:]

    if learned_memo:
        notes = list(profile.get("custom_notes") or [])
        clean_memo = learned_memo.strip()[:120]
        if clean_memo and clean_memo not in notes:
            notes.append(clean_memo)
        profile["custom_notes"] = notes[-6:]

    user_profiles[uid_str] = profile
    _DIRTY_SINCE_LAST_BACKUP = True
    save_user_profiles_local()

    if state.bot and state.bot.loop:
        state.bot.loop.create_task(backup_user_profiles_to_discord(force=False))


def get_speaker_memory_context(user) -> str:
    """Returns a concise memory summary for the current speaker so Yukino remembers their history & preferences."""
    if user is None:
        return ""
    uid_str = str(getattr(user, "id", ""))
    profile = user_profiles.get(uid_str)
    if not profile or int(profile.get("interaction_count", 0)) <= 1:
        if not is_owner_user(user):
            return "- ประวัติการคุย: นี่เป็นการพูดคุยครั้งแรกๆ กับสมาชิกคนนี้ในเซิร์ฟเวอร์"
        return ""

    cnt = profile.get("interaction_count", 1)
    favs = ", ".join(profile.get("favorite_topics") or []) or "ทั่วไป"
    notes = "; ".join(profile.get("custom_notes") or [])
    recents = " | ".join(profile.get("recent_topics") or [])

    lines = [
        f"- ประวัติความคุ้นเคยกับผู้พูดคนนี้: เคยคุยกันมาแล้ว **{cnt} ครั้ง** (คุยล่าสุดเมื่อ: {profile.get('last_seen', 'ไม่นานมานี้')})",
        f"- หัวข้อ/ความสนใจที่ผู้พูดคนนี้ชอบถามบ่อย: **{favs}**",
    ]
    if notes:
        lines.append(f"- ข้อมูลจำเพาะ/ความชอบส่วนตัวที่เคยจำไว้: **{notes}**")
    if recents:
        lines.append(f"- เรื่องล่าสุดที่เคยคุยกัน: `{recents}`")
    return "\n".join(lines)


def resolve_user_mentions_and_build_dossier(
    raw_text: str, guild: discord.Guild | None, bot_user_id: int, speaker_user=None
) -> tuple[str, str]:
    """
    1) Replaces raw `<@123456789>` mentions in the user's prompt with readable `@DisplayName`
       so the LLM knows who is being referenced when asked 'รู้จัก @<User> ไหม'.
    2) Builds a `<mentioned_users_dossier>` block with everything Yukino knows about that user.
    """
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
            notes = "; ".join((prof or {}).get("custom_notes") or [])
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
            notes = "; ".join(prof.get("custom_notes") or [])
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
# OVERTIME (OT) TRACKER & MONTHLY TABLE SUMMARIZER
# - Date Format: DD/MM/YYYY (Thai Buddhist Year, e.g., 16/11/2569)
# - Time Format: Accepts HH.MM or HH:MM, records & displays as HH.MM (e.g., 18.00 - 21.00)
# - Today (Mon-Fri): 17.00 -> Now shifted +1h to 18.00 -> (Now + 1h)
# - Weekend (Sat-Sun) OR Backdated Date: Requires Start Time (asks if omitted!)
# ==============================================================================
OT_TRIGGER_REGEX = re.compile(
    r"^(?:จด|บันทึก|ลง|เพิ่ม|แก้|อัปเดต|สรุป|ดู|เช็ค|ตาราง|ยอด|ลบ|ยกเลิก)\s*(?:เวลา\s*)?ot\b",
    re.IGNORECASE,
)
OT_SUMMARY_REGEX = re.compile(
    r"^(?:สรุป|ดู|เช็ค|ตาราง|ยอด)\s*(?:เวลา\s*)?ot\b|ot\s*เดือน",
    re.IGNORECASE,
)
OT_DELETE_REGEX = re.compile(
    r"^(?:ลบ|ยกเลิก)\s*(?:เวลา\s*)?ot\b",
    re.IGNORECASE,
)
# Matches a standalone time reply when Yukino is waiting for a weekend/backdated time
# Supports both HH.MM and HH:MM: '09.00', '09:00', 'เริ่ม 9.30', '18.00 - 20.00', '18:00 - 20:00'
WEEKEND_TIME_REPLY_REGEX = re.compile(
    r"^(?:เริ่ม(?:\s*เวลา)?|ตั้งแต่(?:\s*เวลา)?)?\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:น\.?|โมง)?"
    r"(?:\s*(?:-|ถึง|to)\s*(\d{1,2})(?:[:.](\d{2}))?\s*(?:น\.?|โมง)?)?\s*(.*)$",
    re.IGNORECASE,
)

# Stores pending weekend/backdated OT requests waiting for a start/range time: {(channel_id, user_id): { ... }}
pending_weekend_ot: dict[tuple[int, int], dict] = {}

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


def _normalize_ot_year_ce(yy_str: str | None, default_year_ce: int) -> int:
    """Converts a 2-digit or 4-digit Thai/CE year string into a 4-digit CE year."""
    if not yy_str:
        return default_year_ce
    y = int(yy_str)
    if y > 2400:
        return y - 543
    if y < 100:
        return (y + 2500 - 543) if y >= 60 else (2000 + y)
    return y


def _format_date_thai_ddmmyyyy(dt: datetime) -> str:
    """Formats a datetime as DD/MM/YYYY in Thai Buddhist Year (e.g., 05/11/2569)."""
    return f"{dt.day:02d}/{dt.month:02d}/{dt.year + 543:04d}"


def _parse_stored_ot_date(date_str: str) -> tuple[int, int, int] | None:
    """
    Parses a stored OT date string ('DD/MM/YYYY' Thai/CE year or legacy 'YYYY-MM-DD')
    into a sortable (year_ce, month, day) tuple.
    """
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
    """Formats a datetime or time string as HH.MM (e.g., 18.00)."""
    if isinstance(dt_or_str, datetime):
        return dt_or_str.strftime("%H.%M")
    return str(dt_or_str).strip().replace(":", ".")


def _extract_custom_date_from_text(text: str, default_dt: datetime) -> tuple[datetime | None, str]:
    """
    Detects if the user specified a custom/backdated date at the start of the OT command, such as:
    - 'วันที่ 5 เดือน 11', 'วันที่ 5 เดือนพฤศจิกายน', 'วันที่ 5'
    - '5 พ.ย.', '5 พย', 'วันที่ 5 พ.ย. 2569', '5 พฤศจิกายน 2569'
    - '05/11/2569', '5/11', '05/11/2026', '5/11/69', '2026-11-05'
    - 'เมื่อวาน', 'วานนี้'
    Returns (parsed_datetime_or_None, remaining_text_without_date).
    """
    s = text.strip()

    # 1. 'เมื่อวาน' / 'วานนี้'
    m_yest = re.match(r"^(?:ของ)?(?:เมื่อวาน(?:นี้)?|วานนี้)\b\s*(.*)", s)
    if m_yest:
        return default_dt - timedelta(days=1), m_yest.group(1).strip()

    # 2. ISO date: '2026-11-05' (with optional leading 'วันที่')
    m_iso = re.match(r"^(?:วันที่\s*)?(\d{4})-(\d{1,2})-(\d{1,2})\b\s*(.*)", s)
    if m_iso:
        yy = _normalize_ot_year_ce(m_iso.group(1), default_dt.year)
        mm, dd, rem = int(m_iso.group(2)), int(m_iso.group(3)), m_iso.group(4)
        try:
            return default_dt.replace(year=yy, month=mm, day=dd), rem.strip()
        except ValueError:
            pass

    # 3. 'วันที่ 5 เดือน 11 [ปี 2569]' or 'วันที่ 5 เดือนพฤศจิกายน'
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

    # 4. Day + Thai Month name/abbreviation: '5 พ.ย.', 'วันที่ 5 พ.ย. 2569', '5 พฤศจิกายน'
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

    # 5. Slash date: '05/11/2569', '5/11', '05/11/2026', '5/11/69'
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

    # 6. 'วันที่ 5' (explicit 'วันที่ <day>' in current month, not followed by ':' or '.' time)
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
        if WEEKEND_TIME_REPLY_REGEX.match(cleaned) or cleaned.lower() in ("ยกเลิก", "cancel"):
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
    """
    Self-healing fallback: scans #work channels on startup for previous Yukinon OT confirmation
    messages so OT records survive Render redeploys even without a #yukinon-memory channel.
    """
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
                        prof = user_profiles.setdefault(uid_s, {
                            "user_id": uid_s,
                            "display_name": "Partner (Owner)",
                            "is_owner": True,
                            "interaction_count": 1,
                            "ot_logs": [],
                        })
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
        save_user_profiles_local()
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

    # Normalize all entries to DD/MM/YYYY (Thai Year) and HH.MM, sorted chronologically
    for entry in ot_logs:
        entry["date"] = _normalize_stored_date_to_thai_ddmmyyyy(entry.get("date", ""))
        entry["noted_start"] = _format_time_dot(entry.get("noted_start", "18.00"))
        entry["noted_end"] = _format_time_dot(entry.get("noted_end", "-"))

    ot_logs.sort(key=lambda x: _parse_stored_ot_date(x.get("date", "")) or (0, 0, 0))
    prof["ot_logs"] = ot_logs
    _DIRTY_SINCE_LAST_BACKUP = True
    save_user_profiles_local()
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


async def process_ot_request(message: discord.Message, raw_text: str) -> str:
    """
    Handles logging, backdating, deleting, and summarizing Monthly Overtime (OT):
    - Date Format: DD/MM/YYYY in Thai Buddhist Year (e.g., 16/11/2569)
    - Time Format: Accepts HH.MM or HH:MM; records & displays as HH.MM (e.g., 18.00 - 21.00)
    """
    global _DIRTY_SINCE_LAST_BACKUP
    user = message.author
    uid_str = str(user.id)
    chan_id = int(message.channel.id)
    pending_key = (chan_id, int(user.id))

    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    default_work_dt = (now_th - timedelta(days=1)) if now_th.hour < 5 else now_th

    text = re.sub(r"^<@!?\d+>\s*", "", raw_text.strip()).strip()
    text = re.sub(r"^!ot\b\s*", "", text, flags=re.IGNORECASE).strip()

    if uid_str not in user_profiles:
        record_user_interaction(user, text, channel_id=chan_id)
    prof = user_profiles[uid_str]
    ot_logs: list[dict] = list(prof.get("ot_logs") or [])

    # ------------------------------------------------------------------
    # 0. HANDLE REPLY TO PENDING WEEKEND / BACKDATED START-TIME PROMPT
    # ------------------------------------------------------------------
    if _has_active_pending_weekend_ot(chan_id, user.id) and not OT_TRIGGER_REGEX.search(text):
        if text.lower() in ("ยกเลิก", "cancel"):
            pending_weekend_ot.pop(pending_key, None)
            return "ยกเลิกการรอเวลาเริ่มทำ OT ให้แล้วค่ะ"

        tm_match = WEEKEND_TIME_REPLY_REGEX.match(text)
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

            if not (0 <= sh <= 23 and 0 <= sm <= 59):
                return "รูปแบบเวลาไม่ถูกต้องค่ะ กรุณาระบุเวลาเป็น `HH.MM` หรือ `HH:MM` เช่น `09.00` หรือ `18.00 - 20.00` นะคะ"

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
                    f"กรุณาระบุเป็น**ช่วงเวลาเริ่ม-เลิก**แทนนะคะ เช่น `{sh:02d}.{sm:02d} - 21.00` ค่ะ"
                )

            ot_hours = round(diff_mins / 60.0, 2)
            default_label = "ปฏิบัติงานล่วงเวลาย้อนหลัง" if is_backdated_pending else "ปฏิบัติงานล่วงเวลาวันหยุด"
            final_detail = extra_detail or saved_detail or f"{default_label} (ไม่ระบุรายละเอียด)"
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
            )

    # ------------------------------------------------------------------
    # 1. DELETE OT ENTRY ('ลบ OT วันนี้', 'ลบ OT วันที่ 5 เดือน 11', 'ลบ OT 05/11/2569')
    # ------------------------------------------------------------------
    if OT_DELETE_REGEX.search(text) or text.lower().startswith(("delete ", "remove ", "ลบ")):
        pending_weekend_ot.pop(pending_key, None)
        del_body = re.sub(r"^(?:ลบ|ยกเลิก|delete|remove)\s*(?:เวลา\s*)?(?:ot)?\s*", "", text, flags=re.IGNORECASE).strip()
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
        save_user_profiles_local()
        await backup_user_profiles_to_discord(force=True)

        if len(ot_logs) < before_len:
            return f"🗑️ ลบรายการบันทึก OT ของวันที่ **`{target_date_th}`** ออกให้เรียบร้อยแล้วค่ะ"
        return f"ไม่พบรายการบันทึก OT ของวันที่ **`{target_date_th}`** ในระบบค่ะ"

    # ------------------------------------------------------------------
    # 2. SUMMARIZE OT ('สรุป OT', 'ดู OT', 'ตาราง OT', or bare '!ot')
    # ------------------------------------------------------------------
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
            "-# พักผ่อนให้เพียงพอด้วยนะคะ อย่าโหมงานหนักจนเสียสุขภาพล่ะ",
        ])
        return "\n".join(table_lines)

    # ------------------------------------------------------------------
    # 3. LOG / UPDATE OT ENTRY ('จด OT [วันที่] [เวลา] [รายละเอียด]')
    # ------------------------------------------------------------------
    detail_str = re.sub(
        r"^(?:จด|บันทึก|ลง|เพิ่ม|แก้|อัปเดต)\s*(?:เวลา\s*)?ot\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    custom_dt, detail_str = _extract_custom_date_from_text(detail_str, default_work_dt)
    is_custom_date = (custom_dt is not None)
    work_dt = custom_dt if is_custom_date else default_work_dt
    target_date_th = _format_date_thai_ddmmyyyy(work_dt)
    is_weekend = (work_dt.weekday() >= 5)

    # Pattern A: Explicit start-end range in HH.MM or HH:MM (e.g. '18.00 - 20.00 แก้โปรแกรม' or '18:00 - 20:00')
    range_time_m = re.match(
        r"^(?:ตั้งแต่(?:\s*เวลา)?|เริ่ม(?:\s*เวลา)?)?\s*(\d{1,2})[:.](\d{2})\s*(?:น\.?)?\s*(?:-|ถึง|to)\s*(\d{1,2})[:.](\d{2})(?:\s*น\.?)?\s*(.*)",
        detail_str,
        flags=re.IGNORECASE,
    )
    if range_time_m:
        pending_weekend_ot.pop(pending_key, None)
        sh, sm = int(range_time_m.group(1)), int(range_time_m.group(2))
        eh, em = int(range_time_m.group(3)), int(range_time_m.group(4))
        rem_detail = range_time_m.group(5).strip(" -:")
        start_dt = work_dt.replace(hour=sh % 24, minute=sm, second=0, microsecond=0)
        end_dt = work_dt.replace(hour=eh % 24, minute=em, second=0, microsecond=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        diff_mins = max(15, int(round((end_dt - start_dt).total_seconds() / 60.0)))
        ot_hours = round(diff_mins / 60.0, 2)
        clean_detail = rem_detail or "ปฏิบัติงานล่วงเวลา (ไม่ระบุรายละเอียด)"
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
        )

    # ------------------------------------------------------------------
    # BRANCH 3A: WEEKEND OR CUSTOM BACKDATED DATE (Requires Start Time!)
    # ------------------------------------------------------------------
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
            pending_weekend_ot.pop(pending_key, None)
            sh = int(wknd_start_m.group(1))
            sm = int(wknd_start_m.group(2) or 0)
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
                        f"สำหรับการจดย้อนหลัง กรุณาระบุเป็นช่วงเวลา เช่น `จด OT {target_date_th} {sh:02d}.{sm:02d} - 21.00` ค่ะ"
                    )
                ot_hours = round(diff_mins / 60.0, 2)

            default_label = "ปฏิบัติงานล่วงเวลาย้อนหลัง" if is_custom_date else "ปฏิบัติงานล่วงเวลาวันหยุด"
            clean_detail = rem_detail or f"{default_label} (ไม่ระบุรายละเอียด)"
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

        if is_custom_date:
            return (
                f"🗓️ บันทึก OT ย้อนหลังของวันที่ **`{target_date_th}`** ({weekday_th}){detail_note}\n"
                f"วันนั้นนายเริ่มทำ OT กี่โมงถึงกี่โมงคะ? **(พิมพ์ตอบแค่ช่วงเวลา เช่น `18.00 - 20.00` หรือเวลาเริ่ม เช่น `18.00` ได้เลยค่ะ)**"
            )
        return (
            f"🗓️ วันนี้เป็น**{weekday_th}** (`{target_date_th}` — วันหยุดสุดสัปดาห์) จะไม่ได้เริ่มนับจาก 17.00 น. เหมือนวันทำงานปกตินะคะ{detail_note}\n"
            f"นายเริ่มทำ OT ตั้งแต่กี่โมงคะ? **(พิมพ์ตอบแค่เวลาเริ่ม เช่น `09.00` หรือช่วงเวลา `09.00 - 17.00` ได้เลยค่ะ เดี๋ยวฉันบันทึกให้ทันทีค่ะ)**"
        )

    # ------------------------------------------------------------------
    # BRANCH 3B: CURRENT WORKDAY RULE (Mon-Fri — 17.00 -> Now shifted +1h to 18.00)
    # ------------------------------------------------------------------
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

    clean_detail = detail_str.strip(" -:") or "ปฏิบัติงานล่วงเวลา (ไม่ระบุรายละเอียด)"
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
        time_note=f"*(เวลาเลิกจริง {actual_finish_str} น. | คิดจาก 17.00 น.)*",
    )