import os
import re
import io
import json
import time
import asyncio
import discord

from config import (
    ENGLISH_ENTITY_REGEX if hasattr(__import__("config"), "ENGLISH_ENTITY_REGEX") else None,
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
    on startup so user memories survive Render container rebuilds/redeploys without an external DB.
    """
    global user_profiles
    load_user_profiles_local()
    ch = _find_backup_channel(bot)
    if not ch:
        return
    try:
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
                            return
    except Exception as e:
        print(f"[User Memory] Cloud restore note: {e}", flush=True)


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

        # Delete previous backup messages from the bot in that channel to keep it clean
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

    # 1. Detect general & active topics
    favs = list(profile.get("favorite_topics") or [])
    if channel_id and channel_id in channel_active_topic:
        act_t = channel_active_topic[channel_id].get("topic", "").strip()
        if act_t and act_t not in favs:
            favs.append(act_t)

    for label, pattern in TOPIC_KEYWORDS_MAP.items():
        if pattern.search(user_prompt) and label not in favs:
            favs.append(label)

    profile["favorite_topics"] = favs[-8:]

    # 2. Save short snippet of what they talked about (excluding pure greetings)
    clean_q = re.sub(r"\s+", " ", user_prompt).strip()
    if len(clean_q) > 6:
        recents = list(profile.get("recent_topics") or [])
        snippet = clean_q[:90]
        if not recents or recents[-1] != snippet:
            recents.append(snippet)
        profile["recent_topics"] = recents[-4:]

    # 3. Save any explicit personal preference/fact extracted by the LLM via [MEMO: ...]
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
            # Never put the owner's actual name into the prompt so Yukino never speaks his name!
            readable_tag = "@[คู่หูชมรมบริการ / ผู้สร้างของฉัน]"
            resolved_text = re.sub(rf"<@!?{uid_str}>", readable_tag, resolved_text)
            favs = ", ".join((prof or {}).get("favorite_topics") or ["การเขียนโค้ด", "เกม", "เทคโนโลยี"])
            notes = "; ".join((prof or {}).get("custom_notes") or [])
            Asking_about_self = (speaker_user and str(getattr(speaker_user, "id", "")) == uid_str)
            dossier_entries.append(
                f"- ข้อมูลของ {readable_tag}: เขาคือคู่หูคนสำคัญในชมรมบริการและผู้ดูแลของฉันเองค่ะ "
                f"{'(ซึ่งก็คือตัวนายที่กำลังถามฉันอยู่นี่แหละ!)' if Asking_about_self else ''} "
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