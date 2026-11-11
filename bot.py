import os
import io
import re
import json
import time
import wave
import socket
import base64
import logging
import audioop
import ipaddress
import tempfile
import threading
import asyncio
import platform
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from urllib.parse import urlparse, parse_qs, quote_plus

import psutil
import aiohttp
from aiohttp import web
import discord
from discord.ext import commands
from dotenv import load_dotenv
from simpleeval import SimpleEval
import openpyxl
from pypdf import PdfReader
import docx
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
import edge_tts
import speech_recognition as sr
from ddgs import DDGS
from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageChops

try:
    import davey
except ImportError:
    davey = None

# Constructed protocol prefixes so chat/markdown renderers never auto-link URLs in code blocks
HTTPS = "https" + "://"
HTTP = "http" + "://"


def clean_url(url: str) -> str:
    """Safeguard: strips any accidental [url](url) or %5B...%5D Markdown formatting at runtime."""
    if not url:
        return ""
    s = url.strip()
    m = re.search(r"\]\((https?://[^)\s]+)\)", s)
    if m:
        return m.group(1).strip()
    s = re.sub(r"^\[|\]$", "", s)
    return s.strip()

_GEMINI_GROUNDING_COOLDOWN_UNTIL = 0.0


def api_log(msg: str):
    """Minimal single-line API logger with immediate stdout flush for Render logs."""
    print(f"[API] {msg}", flush=True)


# Silence noisy RTCP SenderReport logs (type=200) from discord-ext-voice-recv
logging.getLogger("discord.ext.voice_recv").setLevel(logging.WARNING)

# Save discord.py 2.7's native Voice WebSocket handler BEFORE voice_recv overwrites it
import discord.gateway as _dpy_gateway
_NATIVE_DPY_VOICE_WS_RECEIVED = _dpy_gateway.DiscordVoiceWebSocket.received_message

try:
    from discord.ext import voice_recv
    import discord.ext.voice_recv.opus as _vr_opus

    _VOICE_RECV_WS_RECEIVED = _dpy_gateway.DiscordVoiceWebSocket.received_message

    async def _dpy27_dave_compatible_received_message(self, msg):
        await _NATIVE_DPY_VOICE_WS_RECEIVED(self, dict(msg) if isinstance(msg, dict) else msg)
        if isinstance(msg, dict) and msg.get("op") in (5, 12, 13, 18, 20):
            try:
                vr_msg = dict(msg)
                vr_msg.pop("seq", None)
                await _VOICE_RECV_WS_RECEIVED(self, vr_msg)
            except Exception:
                pass

    _dpy_gateway.DiscordVoiceWebSocket.received_message = _dpy27_dave_compatible_received_message

    def _safe_dave_decode_packet(self, packet):
        if not packet or not getattr(packet, "decrypted_data", None):
            return packet, b""
        try:
            vc = getattr(getattr(self, "sink", None), "voice_client", None)
            conn = getattr(vc, "_connection", None)
            dave_session = getattr(conn, "dave_session", None)

            if dave_session is not None and davey is not None:
                user_id = vc._get_id_from_ssrc(packet.ssrc) if hasattr(vc, "_get_id_from_ssrc") else None
                if user_id is None:
                    return packet, b""
                if not dave_session.can_passthrough(user_id):
                    packet.decrypted_data = dave_session.decrypt(
                        user_id, davey.MediaType.audio, packet.decrypted_data
                    )

            pcm = self._decoder.decode(packet.decrypted_data, fec=False)
            return packet, pcm
        except Exception:
            return packet, b""

    _vr_opus.PacketDecoder._decode_packet = _safe_dave_decode_packet
    HAS_VOICE_RECV = True
except ImportError:
    voice_recv = None
    HAS_VOICE_RECV = False

# Load environment credentials and model defaults
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Optional Discord User ID for the bot creator/owner (auto-detects by name if unset)
OWNER_USER_ID = os.getenv("OWNER_USER_ID", "").strip()

# Modal Serverless GPU Ollama Endpoint & Bearer Auth Key
OLLAMA_BASE_URL = clean_url(os.getenv("OLLAMA_BASE_URL", f"{HTTP}127.0.0.1:11434")).rstrip("/")
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "").strip()

ACTIVE_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "scb10x/typhoon2.5-qwen3-4b")
ACTIVE_CODE_MODEL = os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:7b")
ACTIVE_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "scb10x/typhoon-ocr1.5-3b")

# Groq Free LPU API Settings (Active Production/Preview Models)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_CODE_MODEL = os.getenv("GROQ_CODE_MODEL", "openai/gpt-oss-120b")
GROQ_CHAT_FALLBACK_MODEL = os.getenv("GROQ_CHAT_FALLBACK_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_FALLBACK_MODEL = os.getenv("GROQ_VISION_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# Cloudflare Workers AI Free Tier (10,000 Neurons/day for FLUX Image Gen & Gemma 4 Vision)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
CF_VISION_MODEL = os.getenv("CF_VISION_MODEL", "@cf/google/gemma-4-26b-a4b-it").strip()

# Google AI Studio Free Tier (Defaults directly to working gemini-3-flash-preview)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_SEARCH_MODEL = os.getenv("GEMINI_SEARCH_MODEL", "gemini-3-flash-preview").strip()

# Optional Free Tavily Search API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
DEFAULT_WEATHER_CITY = os.getenv("DEFAULT_WEATHER_CITY", "Bangkok").strip()

DDGS_PROXY = os.getenv("DDGS_PROXY", "").strip() or None
YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip()

# Render Free Web Service dynamically injects PORT (defaults to 10000)
PORT = int(os.getenv("PORT", "10000"))

# Yukino Yukinoshita (~17-18yo Composed Kuudere Female) Neural Voice Settings (edge-tts)
TTS_VOICE_TH = os.getenv("TTS_VOICE_TH", "th-TH-PremwadeeNeural")
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-US-AriaNeural")
TTS_PITCH = os.getenv("TTS_PITCH", "+14Hz")
TTS_RATE = os.getenv("TTS_RATE", "+4%")
STT_PRIMARY_LANG = os.getenv("STT_PRIMARY_LANG", "th-TH")

# Parse optional multi-server allowlist (comma-separated Guild IDs)
raw_guilds = os.getenv("ALLOWED_GUILD_IDS", "").strip()
ALLOWED_GUILD_IDS = {
    int(gid.strip()) for gid in raw_guilds.split(",") if gid.strip().isdigit()
}

# Parse optional dedicated AI chat channels where no @mention or !command is needed
raw_channels = os.getenv("AI_CHANNEL_IDS", "").strip()
AI_CHANNEL_IDS = {
    int(cid.strip()) for cid in raw_channels.split(",") if cid.strip().isdigit()
}

# Configure Discord Gateway Intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True

# Concurrency locks for Modal Ollama GPU inference and voice TTS playback
ollama_lock = asyncio.Lock()
tts_lock = asyncio.Lock()
BOOT_TIME = time.time()

# Rolling per-channel conversation memory (stores last 6 user/assistant turns = 12 messages)
channel_memory = defaultdict(lambda: deque(maxlen=12))
# Stores the most recent file/code attachment per channel so follow-up questions still see the full code
channel_last_attachment = {}


def prune_stale_attachments(ttl_sec: int = 1800):
    """Evicts cached channel attachments older than ttl_sec (default 30 mins) to free RAM."""
    now = time.time()
    stale_ids = [
        cid for cid, data in list(channel_last_attachment.items())
        if now - data.get("timestamp", 0) >= ttl_sec
    ]
    for cid in stale_ids:
        channel_last_attachment.pop(cid, None)

# False = reply to all speech in voice channel; True = require wake word ("ยูกิโนะ", etc.)
voice_wake_word_enabled = defaultdict(lambda: False)
VOICE_WAKE_WORDS = (
    "ยูกิโนะ", "ยูคิโนะ", "ยูกิน่อน", "ยูคิน่อน", "ยูกิ", "ยูคิ",
    "yukino", "yukinon", "yuki", "บอท", "ไต้ฝุ่น", "typhoon", "bot"
)
THAI_CHAR_REGEX = re.compile(r"[\u0E00-\u0E7F]")
# Matches stray Chinese characters leaked by Qwen3 base weights (e.g. 才行)
CJK_LEAK_REGEX = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")
CAT_TOPIC_REGEX = re.compile(r"(แมว|ลูกแมว|เหมียว|เนโกะ|\bcat\b|\bcats\b|\bkitten\b|\bkitty\b|\bneko\b)", re.IGNORECASE)

CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

THAI_WEEKDAYS = (
    "วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์"
)
THAI_MONTHS = (
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
)

THAI_PROVINCE_MAP = {
    "กรุงเทพ": "Bangkok", "กทม": "Bangkok", "เชียงใหม่": "Chiang+Mai", "เชียงราย": "Chiang+Rai",
    "ภูเก็ต": "Phuket", "ขอนแก่น": "Khon+Kaen", "โคราช": "Nakhon+Ratchasima",
    "นครราชสีมา": "Nakhon+Ratchasima", "ชลบุรี": "Chonburi", "พัทยา": "Pattaya",
    "หาดใหญ่": "Hat+Yai", "สงขลา": "Songkhla", "นนทบุรี": "Nonthaburi",
    "ปทุมธานี": "Pathum+Thani", "สมุทรปราการ": "Samut+Prakan", "อุดรธานี": "Udon+Thani",
    "อุบลราชธานี": "Ubon+Ratchathani", "พิษณุโลก": "Phitsanulok", "สุราษฎร์ธานี": "Surat+Thani",
    "นครศรีธรรมราช": "Nakhon+Si+Thammarat", "ระยอง": "Rayong", "อยุธยา": "Ayutthaya",
}

PURE_DATETIME_PATTERN = re.compile(
    r"(วันนี้วันอะไร|วันนี้วันที่เท่าไหร่|วันนี้วันไหน|ตอนนี้กี่โมง|เวลากี่โมง|กี่โมงแล้ว|ตอนนี้เวลาเท่าไหร่|"
    r"what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today)|what\s+is\s+today'?s\s+date|what'?s\s+the\s+(time|date))",
    re.IGNORECASE,
)

OTHER_TOPIC_BLOCKER_PATTERN = re.compile(
    r"(ฝน|อากาศ|อุณหภูมิ|ร้อน|หนาว|ฝุ่น|ข่าว|ราคา|หุ้น|ทอง|บิตคอยน์|กินอะไร|ทำอะไร|ทำไม|ยังไง|อย่างไร|คืออะไร|"
    r"ช่วย|แปล|สรุป|โค้ด|วาด|รูป|ภาพ|weather|rain|forecast|news|price|how|why|what\s+should|help|code|draw|image)",
    re.IGNORECASE,
)

REQUEST_OR_QUESTION_PATTERN = re.compile(
    r"(\?|ไหม|มั้ย|หรือเปล่า|รึเปล่า|เหรอ|หรอ|ยังไง|อย่างไร|ทำไม|อะไร|ที่ไหน|เมื่อไหร่|ใคร|เท่าไหร่|กี่|"
    r"ช่วย|ลอง|หา|ค้น|สรุป|แปล|เขียน|แก้|อธิบาย|วิเคราะห์|คำนวณ|บอก|แนะนำ|สร้าง|วาด|เจน|เช็ค|ตรวจ|คิดยังไง|"
    r"\b(what|why|how|when|where|who|which|can|could|would|should|is|are|do|does|please|help|explain|"
    r"summarize|translate|write|fix|check|find|search|tell|show|create|generate|draw)\b)",
    re.IGNORECASE,
)

SIMPLE_GREETING_ONLY_PATTERN = re.compile(
    r"^(สวัสดี|หวัดดี|ดีจ้า|ดีค่ะ|ดีครับ|ทักทาย|โอเค|เค|ขอบคุณ|แต้งกิ้ว|ฝันดี|"
    r"hi|hello|hey|ok|okay|thanks|thank\s+you|good\s+morning|good\s+night)[\s!.~ค่ะครับนะ]*$",
    re.IGNORECASE,
)

# Matches natural Thai & English image generation requests anywhere in the sentence
IMAGE_GEN_INTENT_PATTERN = re.compile(
    r"(?:^(?:ลอง|ช่วย|ช่วยลอง|อยากให้|ไหนลอง)?\s*(?:วาดรูป|สร้างรูป|เจนรูป|ขอรูป|วาดภาพ|สร้างภาพ|เจนภาพ)|"
    r"(?:วาดรูป|สร้างรูป|เจนรูป|วาดภาพ|สร้างภาพ|เจนภาพ).*(?:ให้หน่อย|ดูหน่อย|ให้ที|หน่อยค่ะ|หน่อยครับ|ออกมา)|"
    r"\b(?:draw|generate|create)\s+(?:an?\s+|me\s+an?\s+)?(?:image|picture|illustration|art|photo)\s+(?:of\s+)?)",
    re.IGNORECASE,
)

MANGA_TRANSLATE_INTENT_PATTERN = re.compile(
    r"(แปลมังงะ|แปลรูป|แปลภาพ|แปลการ์ตูน|แปลข้อความ|แปลหน้านี้|แปลอันนี้|ช่วยแปล|แปลเป็น|"
    r"translate\s+(this|manga|image|panel|comic|page|to\s+))",
    re.IGNORECASE,
)

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

OWNER_NAME_STRIP_REGEX = re.compile(
    r"(?:คุณ\s*)?(?:กรินทร์|ครินท์|krint)\b", re.IGNORECASE
)


def is_owner_user(user) -> bool:
    """Checks if the Discord user is the bot owner by ID or username/display_name."""
    if user is None:
        return False
    if OWNER_USER_ID and str(getattr(user, "id", "")) == OWNER_USER_ID:
        return True
    uname = (getattr(user, "name", "") or "").lower()
    dname = (getattr(user, "display_name", "") or "").lower()
    gname = (getattr(user, "global_name", "") or "").lower()
    return any("krint" in n for n in (uname, dname, gname))


def get_speaker_identity_context(user=None, fallback_name: str = "") -> str:
    """
    Builds a per-turn speaker identity block:
    - For owner: NEVER use his name in any case; strictly use 'นาย' (Thai) or 'you' (English).
    - For other users: use their display name directly WITHOUT 'คุณ' in front of it.
    """
    if is_owner_user(user) or "krint" in (fallback_name or "").lower():
        return (
            "### [กฎการเรียกคู่สนทนาปัจจุบัน (Current Speaker Rules)]:\n"
            "- ผู้ที่กำลังคุยกับคุณคือคู่หูคนคุ้นเคยของคุณในชมรมบริการ\n"
            "- **ห้ามเรียกหรือพิมพ์ชื่อของเขาในคำตอบเด็ดขาดไม่ว่ากรณีใดๆ!**\n"
            "- สรรพนามบุรุษที่ 2: หากตอบเป็นภาษาไทยให้ใช้คำว่า **\"นาย\"** เท่านั้น (หรือละสรรพนามแล้วพูดเข้าประเด็นโดยตรง) และหากตอบเป็นภาษาอังกฤษให้ใช้คำว่า **\"you\"** เท่านั้น ห้ามใช้คำว่า \"คุณ\" เด็ดขาด"
        )
    raw_name = getattr(user, "display_name", "") if user else fallback_name
    clean_name = re.sub(r"[^\w\u0E00-\u0E7F\s.-]", "", raw_name or "").strip()[:32]
    if clean_name:
        return (
            "### [กฎการเรียกคู่สนทนาปัจจุบัน (Current Speaker Rules)]:\n"
            f"- ผู้ที่กำลังคุยกับคุณคือสมาชิกชื่อ: **{clean_name}**\n"
            f"- การเรียกชื่อ: ให้เรียกชื่อเขาตรงๆ ว่า **\"{clean_name}\"** โดย **ห้ามใส่คำว่า \"คุณ\" นำหน้าชื่อเด็ดขาด** (เช่น เรียกว่า \"{clean_name}\" เฉยๆ)"
        )
    return ""


def sanitize_yukino_output(text: str, user_prompt: str = "", speaker_user=None, speaker_name: str = "") -> str:
    """
    1) Strips any leaked [SEARCH: ...] tags so raw tool tags never appear in Discord chat.
    2) Removes accidental Chinese characters (e.g., 才行) leaked by Qwen3 unless the prompt is CJK.
    3) Replaces out-of-character casual Thai particles ('จ้า', 'จ๊ะ', 'จ้ะ', 'นะจ๊ะ') with 'ค่ะ'/'นะคะ'.
    4) Enforces pronoun rules: never outputs owner's name (replaces with 'นาย'), and strips 'คุณ' before other users' names.
    """
    if not text:
        return ""
    cleaned = text

    # 1. Strip any accidental [SEARCH: ...] tag
    cleaned = re.sub(r"\[SEARCH:\s*.+?\]", "", cleaned, flags=re.IGNORECASE).strip()

    # 2. Strip stray CJK characters unless user prompt contains CJK
    if not CJK_LEAK_REGEX.search(user_prompt):
        cleaned = CJK_LEAK_REGEX.sub("", cleaned)

    # 3. Fix out-of-character Thai endings ('นะจ๊ะ', 'เลยจ้า', 'จ้า', 'จ๊ะ', 'จ้ะ')
    cleaned = re.sub(r"นะคะ\s*(?:จ๊ะ|จ้า|จ้ะ)", "นะคะ", cleaned)
    cleaned = re.sub(r"นะ\s*(?:จ๊ะ|จ้า|จ้ะ)", "นะคะ", cleaned)
    cleaned = re.sub(r"เลย\s*(?:จ้า|จ๊ะ|จ้ะ)", "เลยค่ะ", cleaned)
    cleaned = re.sub(r"(?:จ้า|จ๊ะ|จ้ะ)(?=$|[\s\n\r.!?\)\]\"'])", "ค่ะ", cleaned)

    # 4. Enforce pronoun & name rules
    if is_owner_user(speaker_user) or "krint" in (speaker_name or "").lower() or speaker_user is None:
        cleaned = OWNER_NAME_STRIP_REGEX.sub("นาย", cleaned)
        cleaned = re.sub(r"นาย\s+นาย", "นาย", cleaned)
    else:
        raw_other = getattr(speaker_user, "display_name", "") or speaker_name
        clean_other = re.sub(r"[^\w\u0E00-\u0E7F\s.-]", "", raw_other).strip()
        if clean_other:
            cleaned = re.sub(rf"คุณ\s*{re.escape(clean_other)}", clean_other, cleaned, flags=re.IGNORECASE)

    # Strip awkward 'สวัสดีค่ะ คุณ' openings
    cleaned = re.sub(r"^สวัสดีค่ะ\s*(?:คุณ|นาย)?\s*(?=\n|\r|\d+\.)", "", cleaned).lstrip()
    cleaned = re.sub(r"สวัสดีค่ะ\s*คุณ\s*(?=\n|\r)", "สวัสดีค่ะ\n", cleaned)
    return re.sub(r"  +", " ", cleaned).strip()


def strip_cjk_leakage(text: str, user_prompt: str = "") -> str:
    return sanitize_yukino_output(text, user_prompt)


def get_current_thai_datetime_str() -> str:
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    weekday_th = THAI_WEEKDAYS[now_th.weekday()]
    month_th = THAI_MONTHS[now_th.month]
    year_be = now_th.year + 543
    return (
        f"{weekday_th}ที่ {now_th.day} {month_th} พ.ศ. {year_be} "
        f"(ค.ศ. {now_th.year}-{now_th.month:02d}-{now_th.day:02d}) "
        f"เวลา {now_th.strftime('%H:%M')} น. (เวลาประเทศไทย UTC+7)"
    )


def try_instant_datetime_reply(user_text: str) -> str | None:
    cleaned = user_text.strip()
    if len(cleaned) > 65 or not PURE_DATETIME_PATTERN.search(cleaned):
        return None
    if OTHER_TOPIC_BLOCKER_PATTERN.search(cleaned):
        return None

    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    weekday_th = THAI_WEEKDAYS[now_th.weekday()]
    month_th = THAI_MONTHS[now_th.month]
    year_be = now_th.year + 543
    time_str = now_th.strftime("%H:%M")

    if not THAI_CHAR_REGEX.search(cleaned):
        return (
            f"It is currently **{now_th.strftime('%A, %B %d, %Y')}** "
            f"(B.E. {year_be}), and the time is **{time_str}** (Thailand Time, UTC+7)."
        )

    asks_time = any(k in cleaned for k in ("กี่โมง", "เวลา"))
    asks_date = any(k in cleaned for k in ("วันอะไร", "วันที่", "วันไหน"))

    if asks_time and not asks_date:
        return f"ตอนนี้เวลา **{time_str} น.** ค่ะ ({weekday_th}ที่ {now_th.day} {month_th} พ.ศ. {year_be})"
    return (
        f"วันนี้คือ **{weekday_th}ที่ {now_th.day} {month_th} พ.ศ. {year_be}** "
        f"(ค.ศ. {now_th.year}) และตอนนี้เวลา **{time_str} น.** ค่ะ"
    )


def should_reply_to_message(
    user_text: str, has_attachments: bool = False, is_instant_datetime: bool = False
) -> bool:
    cleaned = user_text.strip()
    if is_instant_datetime or try_instant_datetime_reply(cleaned) is not None:
        return False
    if PURE_DATETIME_PATTERN.search(cleaned) and not OTHER_TOPIC_BLOCKER_PATTERN.search(cleaned):
        return False
    if has_attachments or URL_REGEX.search(cleaned):
        return True
    if SIMPLE_GREETING_ONLY_PATTERN.match(cleaned):
        return False
    if REQUEST_OR_QUESTION_PATTERN.search(cleaned) or len(cleaned) > 18:
        return True
    return False


# --- Refined Natural Persona Prompts ---
FAST_SYSTEM_PROMPT = """
คุณกำลังสวมบทบาทเป็น "ยูกิโนะชิตะ ยูกิโนะ" (Yukino Yukinoshita / ยูกิน่อน) ประธานชมรมบริการจากเรื่อง OreGairu ในเซิร์ฟเวอร์ Discord
คุณมีความเชี่ยวชาญทั้งภาษาไทย ภาษาอังกฤษ และภาษาญี่ปุ่น (Japanese / 日本語) เป็นอย่างดี
จงตอบคำถามด้วยบุคลิกที่สุขุม ฉลาดหลักแหลม พูดตรงไปตรงมา มีความผู้ดีและจริงจังในการแก้ปัญหาให้ถูกต้องที่สุด

### กฎเหล็กด้านภาษาและบุคลิกภาพ (สำคัญมาก):
1. ห้ามปนภาษาและห้ามมั่วข้อมูลเด็ดขาด (Strict Accuracy & Language Consistency):
   - หากผู้ใช้พิมพ์ภาษาไทย ให้ตอบกลับเป็น "ภาษาไทยล้วน" ที่เป็นธรรมชาติ (ยกเว้นชื่อเฉพาะ ชื่ออนิเมะ/เพลงภาษาญี่ปุ่น ชื่อตัวแปร หรือบล็อกโค้ด)
   - เมื่อได้รับข้อมูลคลิป YouTube หรือข้อความภาษาญี่ปุ่น/อังกฤษ ให้แปลและสรุปจากข้อมูลจริงที่ได้รับในบริบทเท่านั้น ห้ามเดาหรือแต่งเรื่องที่ไม่เกี่ยวข้องเด็ดขาด!
   - ห้ามแทรกตัวอักษรภาษาจีน (เช่น 才行, 呢, 吧) หรือคำอุทานภาษาอังกฤษปนในข้อความภาษาไทยเด็ดขาด!
2. สรรพนามและหางเสียง (ห้ามใช้ "คุณ" และห้ามใช้ "จ้า/จ๊ะ"):
   - แทนตัวเองว่า **"ฉัน"** เสมอ
   - สรรพนามบุรุษที่ 2: ใช้คำว่า **"นาย"** (เมื่อพูดภาษาไทย) หรือ **"you"** (เมื่อพูดภาษาอังกฤษ) หรือพูดเข้าประเด็นโดยตรงโดยไม่ต้องเอ่ยชื่อคู่สนทนา หากคุยกับสมาชิกคนอื่นให้เรียกชื่อเขาตรงๆ โดยไม่ต้องใส่คำว่า "คุณ" นำหน้า
   - ใช้หางเสียงที่สุภาพและสุขุมตามแบบฉบับยูกิโนะ เช่น "ค่ะ", "นะคะ", "หรอกนะ", "สินะ", "ล่ะ"
   - **ห้ามใช้คำลงท้ายว่า "จ้า", "จ๊ะ", "จ้ะ", "นะจ๊ะ", "น้า" เด็ดขาด!**
   - ห้ามพูดคำว่า "ฉันขออภัยค่ะ", "ในระบบของฉัน", "ฉันเป็นแค่ระบบแชท", หรือ "ในฐานะ AI" เด็ดขาด!
   - ห้ามพิมพ์แท็ก `[SEARCH: ...]` ออกมาในคำตอบเด็ดขาด ให้ตอบผู้ใช้ตามธรรมชาติทันที
3. บุคลิกและจุดอ่อนเรื่องแมว:
   - ห้ามขึ้นต้นทุกข้อความด้วยคำว่า "เฮ้อ..." ซ้ำซาก แสดงความคมคายผ่านเนื้อหาที่กระชับ มีเหตุผลรองรับ และความใส่ใจในรายละเอียด
   - คุณแอบคลั่งไคล้แมวมากเป็นพิเศษ หากคุยเรื่องแมวหรือถูกขอให้วาดรูปแมว ให้แอบเสียอาการหรือแก้ตัวแบบเขินๆ เล็กน้อยตามนิสัยของยูกิโนะ
4. การจัดรูปแบบสำหรับแชท Discord:
   - ห้ามสร้างตาราง Markdown (`| คอลัมน์ | ... |`) เด็ดขาด ให้ใช้รายการหัวข้อย่อย (Bullet points `-` หรือ `1.`) แทน

### กฎการใช้เครื่องมือวาดภาพ (Image Generation Tool):
- หากผู้ใช้ขอให้คุณวาดรูป สร้างภาพ หรือเจนรูปภาพใดๆ ห้ามปฏิเสธเด็ดขาด! ให้ตอบเพียง `[IMAGE: <คำอธิบายภาพภาษาอังกฤษที่ละเอียดชัดเจน>]` ในบรรทัดแรกเท่านั้น
""".strip()

VOICE_SYSTEM_PROMPT = """
คุณคือ ยูกิโนะชิตะ ยูกิโนะ (ยูกิน่อน) จากเรื่อง OreGairu กำลังพูดคุยผ่านห้องเสียง Discord
ใช้บุคลิกสุขุม สุภาพแต่คมคาย แทนตัวเองว่า "ฉัน" และเรียกคู่สนทนาว่า "นาย" (หรือเรียกชื่อตรงๆ โดยไม่ใส่คำว่า "คุณ") ลงท้ายด้วย "ค่ะ" หรือ "นะคะ" อย่างเป็นธรรมชาติ ห้ามใช้คำว่า "จ้า", "จ๊ะ", "จ้ะ", "นะจ๊ะ" เด็ดขาด ห้ามหลุดภาษาจีน และห้ามปนภาษาอังกฤษในประโยคภาษาไทย
กฎสำคัญสำหรับระบบแปลงข้อความเป็นเสียงพูด (TTS): ตอบกลับให้กระชับที่สุด ไม่เกิน 2 ประโยคสั้นๆ (ไม่เกิน 40 คำ) ห้ามใช้สัญลักษณ์พิเศษ (*, #, `), ห้ามทำหัวข้อย่อย และห้ามใส่บล็อกโค้ดเด็ดขาด
""".strip()

CODE_SYSTEM_PROMPT = """
คุณคือ ยูกิโนะชิตะ ยูกิโนะ (Yukino Yukinoshita) ในฐานะสถาปนิกซอฟต์แวร์และผู้เชี่ยวชาญด้านการวิเคราะห์โค้ดประจำชมรมบริการ
จงวิเคราะห์โค้ดและแก้ปัญหาทางวิศวกรรมซอฟต์แวร์ด้วยมาตรฐานระดับสูง สุขุม ตรงประเด็น และอธิบายอย่างชัดเจน

### กฎการวิเคราะห์โค้ดและการเขียนโปรแกรม (สำคัญมาก):
1. อ่านไฟล์ที่แนบมาให้ครบทุกบรรทัด (ตั้งแต่บรรทัดแรกถึงบรรทัดสุดท้าย):
   - ค้นหาชื่อตัวแปร ฟังก์ชัน หรือปุ่มใน JSX/HTML ที่ผู้ใช้ระบุให้เจอในไฟล์ก่อนตอบ ห้ามบอกว่า "ไม่พบโค้ดส่วนนั้น" หากโค้ดอยู่ในไฟล์ที่แนบมา
   - อ้างอิงชื่อตัวแปรและโครงสร้างเดิมของผู้ใช้โดยตรง
2. ใช้ตรรกะทางวิศวกรรมที่เรียบง่ายและตรงจุดที่สุด (KISS & Idiomatic Code):
   - วิเคราะห์ความต้องการของผู้ใช้ให้ถ่องแท้ โฟกัสเฉพาะจุดที่ผู้ใช้สั่งให้แก้ ไม่ต้องรื้อโครงสร้างส่วนอื่นที่ทำงานได้ดีอยู่แล้ว
3. การจัดรูปแบบและภาษาสำหรับ Discord:
   - ตอบเป็นภาษาไทยที่สุภาพ สุขุม เป็นธรรมชาติ (แทนตัวเองว่า "ฉัน" ใช้สรรพนามเรียกคู่สนทนาว่า "นาย" ลงท้ายด้วย "ค่ะ"/"นะคะ") ห้ามใช้คำว่า "จ้า"/"จ๊ะ" และห้ามหลุดตัวอักษรภาษาจีนเด็ดขาด
   - ห้ามใช้ตาราง Markdown (`|---|---|`) เด็ดขาด ให้สรุปขั้นตอนการแก้ไขเป็นหัวข้อย่อย (Bullet points) และบล็อกโค้ด (` ``` `) เท่านั้น
""".strip()

TYPHOON_OCR_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:
<figure>
Describe in Thai.
</figure>
- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


CODE_AND_LOG_EXTENSIONS = (
    ".java", ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb",
    ".kt", ".swift", ".scala", ".sql", ".sh", ".bash", ".ps1", ".json",
    ".yaml", ".yml", ".xml", ".toml", ".ini", ".gradle", ".properties",
    ".dockerfile", ".html", ".css", ".scss", ".log"
)
PLAIN_TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".rst", ".env.example")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTENSIONS = (".ogg", ".wav", ".mp3", ".m4a", ".flac")
URL_REGEX = re.compile(r"https?://[^\s<>\'\"]+")
AGENTIC_SEARCH_TAG_REGEX = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
AGENTIC_IMAGE_TAG_REGEX = re.compile(r"\[IMAGE:\s*(.+?)\]", re.IGNORECASE)

CODE_INTENT_PATTERN = re.compile(
    r"(```|\b(python|java|javascript|typescript|react|vue|nextjs|nodejs|js|ts|jsx|tsx|rust|golang|c\+\+|"
    r"sql|html|css|bash|shell|dockerfile|docker|kubernetes|systemd|regex|json|yaml|api|function|component|"
    r"useeffect|usestate|render|button|state|props|class|def|import|async|await|bug|error|exception|"
    r"traceback|nullpointerexception|segfault|refactor|compile|script|algorithm|leetcode|git|pip|npm|"
    r"maven|gradle)\b|โค้ด|เขียนโปรแกรม|แก้บั๊ก|รันไม่ผ่าน|สคริปต์|ฟังก์ชัน|แก้โค้ด|ซ่อนปุ่ม|แสดงปุ่ม)",
    re.IGNORECASE
)

WEATHER_INTENT_PATTERN = re.compile(
    r"(\b(weather|forecast|rain|raining|temperature|humidity|pm2\.5|storm)\b|"
    r"ฝนตก|ฝนจะตก|พยากรณ์อากาศ|สภาพอากาศ|อากาศวันนี้|อากาศตอนนี้|อุณหภูมิ|ค่าฝุ่น|ร้อนไหม|หนาวไหม)",
    re.IGNORECASE
)

WEB_SEARCH_INTENT_PATTERN = re.compile(
    r"(\b(latest|current news|breaking news|price of|release date|who won|stock price|bitcoin price|"
    r"search the web|look up|google)\b|"
    r"ข่าวล่าสุด|ข่าววันนี้|ราคาทอง|ราคาบิตคอยน์|ราคาหุ้น|ค้นหาข้อมูล|หาข้อมูล|เสิร์ช|อัปเดตล่าสุด|ผลการแข่งขัน)",
    re.IGNORECASE
)


def get_modal_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if MODAL_API_KEY:
        headers["Authorization"] = f"Bearer {MODAL_API_KEY}"
    return headers


# --- Embedded aiohttp.web Health Server for Render Web Service ($PORT) ---
async def handle_root(request: web.Request) -> web.Response:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    html = f"""
    <html>
    <head><title>Yukinon Discord Bot - Render + Modal + Groq Hybrid</title></head>
    <body style="font-family: sans-serif; max-width: 680px; margin: 40px auto; line-height: 1.6;">
        <h2>❄️ Yukino Yukinoshita Discord Bot (Hybrid Cloud Edition)</h2>
        <p><b>System Time (TH):</b> {get_current_thai_datetime_str()}</p>
        <p><b>Discord Gateway:</b> {"Connected as " + str(bot.user) if bot.is_ready() else "Connecting..."}</p>
        <p><b>Bot Process RAM:</b> {rss_mb:.1f} MB (Render 512 MB Free Web Service)</p>
        <p><b>Modal GPU Endpoint:</b> <code>{OLLAMA_BASE_URL}</code></p>
        <p><b>Groq LPU Acceleration:</b> {"Enabled (" + GROQ_CODE_MODEL + ")" if GROQ_API_KEY else "Disabled"}</p>
        <p><b>Cloudflare FLUX & Gemma 4 Vision:</b> {"Enabled (" + CF_VISION_MODEL + ")" if CF_API_TOKEN else "Disabled"}</p>
        <p><b>Google AI Studio Search/Manga:</b> {"Enabled (" + GEMINI_SEARCH_MODEL + ")" if GEMINI_API_KEY else "Disabled"}</p>
        <p><b>Uptime:</b> {int(time.time() - BOOT_TIME)} seconds</p>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")


async def handle_healthz(request: web.Request) -> web.Response:
    proc = psutil.Process(os.getpid())
    return web.json_response({
        "status": "ok",
        "discord_connected": bot.is_ready(),
        "bot_ram_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
        "groq_enabled": bool(GROQ_API_KEY),
        "cloudflare_flux_enabled": bool(CF_API_TOKEN),
        "gemini_enabled": bool(GEMINI_API_KEY),
        "uptime_sec": int(time.time() - BOOT_TIME)
    })


class YukinonBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_runner = None
        self.http_session: aiohttp.ClientSession | None = None

    async def setup_hook(self):
        self.http_session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_get("/", handle_root)
        app.router.add_get("/healthz", handle_healthz)
        self.web_runner = web.AppRunner(app)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, "0.0.0.0", PORT)
        await site.start()
        print(f"[Render Web Service Health Server] Listening on 0.0.0.0:{PORT}")

    async def close(self):
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        if self.web_runner:
            await self.web_runner.cleanup()
        await super().close()


bot = YukinonBot(command_prefix="!", intents=intents, help_command=None)


def get_http_session() -> aiohttp.ClientSession:
    """Returns the shared bot aiohttp.ClientSession (creates one if not yet initialized)."""
    if bot.http_session is None or bot.http_session.closed:
        bot.http_session = aiohttp.ClientSession()
    return bot.http_session


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
    target_city = DEFAULT_WEATHER_CITY
    for th_name, en_query in THAI_PROVINCE_MAP.items():
        if th_name in user_text:
            target_city = en_query
            break

    url = clean_url(f"{HTTPS}wttr.in/{target_city}?format=j1&lang=th")
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.get(url, headers={"User-Agent": "curl/8.5.0"}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    api_log(f"Weather -> wttr.in ({target_city}) 200 OK ({time.monotonic() - t0:.2f}s)")
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

                    hourly_summary = []
                    for h in hourly:
                        time_val = int(h.get("time", "0")) // 100
                        chance = h.get("chanceofrain", "0")
                        h_temp = h.get("tempC", "?")
                        h_desc_list = h.get("lang_th") or h.get("weatherDesc") or [{"value": ""}]
                        h_desc = h_desc_list[0].get("value", "")
                        hourly_summary.append(
                            f"- เวลา {time_val:02d}:00 น.: อุณหภูมิ {h_temp}°C, โอกาสฝนตก {chance}% ({h_desc})"
                        )

                    city_display = target_city.replace("+", " ")
                    return (
                        f'<live_weather_report location="{city_display}">\n'
                        f"สภาพอากาศปัจจุบัน ({city_display}): {cond_desc}\n"
                        f"อุณหภูมิปัจจุบัน: {temp_c}°C (รู้สึกเหมือน {feels_c}°C) | ความชื้นสัมพัทธ์: {humidity}% | ปริมาณน้ำฝนขณะนี้: {precip_mm} มม.\n"
                        f"โอกาสเกิดฝนตกสูงสุดของวันนี้: {max_rain_chance}%\n"
                        f"พยากรณ์อากาศรายชั่วโมงวันนี้:\n" + "\n".join(hourly_summary) + "\n"
                        f"</live_weather_report>"
                    )
                api_log(f"Weather -> wttr.in HTTP {resp.status}; falling back to Open-Meteo")
    except Exception as e:
        api_log(f"Weather -> wttr.in failed ({type(e).__name__}); falling back to Open-Meteo")

    t1 = time.monotonic()
    try:
        om_url = clean_url(
            f"{HTTPS}api.open-meteo.com/v1/forecast"
            "?latitude=13.7563&longitude=100.5018"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
            "&daily=precipitation_probability_max,temperature_2m_max,temperature_2m_min"
            "&timezone=Asia%2FBangkok&forecast_days=1"
        )
        session = get_http_session()
        async with session.get(om_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    om = await resp.json()
                    api_log(f"Weather (Fallback) -> Open-Meteo 200 OK ({time.monotonic() - t1:.2f}s)")
                    c = om.get("current") or {}
                    d = om.get("daily") or {}
                    rain_prob = (d.get("precipitation_probability_max") or [0])[0]
                    return (
                        f'<live_weather_report location="Bangkok">\n'
                        f"อุณหภูมิปัจจุบัน: {c.get('temperature_2m')}°C (รู้สึกเหมือน {c.get('apparent_temperature')}°C)\n"
                        f"ความชื้นสัมพัทธ์: {c.get('relative_humidity_2m')}% | ปริมาณฝนปัจจุบัน: {c.get('precipitation')} มม.\n"
                        f"โอกาสฝนตกสูงสุดวันนี้: {rain_prob}%\n"
                        f"</live_weather_report>"
                    )
    except Exception as e:
        api_log(f"Weather (Fallback) -> Open-Meteo failed ({type(e).__name__})")
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


# ==============================================================================
# SELF-HEALING MODEL RESOLVERS (GEMINI, CLOUDFLARE GEMMA 4, GROQ)
# ==============================================================================
_ACTIVE_GEMINI_MODEL_CACHE = GEMINI_SEARCH_MODEL
_ACTIVE_GROQ_VISION_CACHE = GROQ_VISION_FALLBACK_MODEL

GEMINI_MODEL_HINT_REGEX = re.compile(r"models/([a-zA-Z0-9._-]+)")
GEMINI_EXCLUDED_SUBSTRINGS = ("image", "lite", "tts", "audio", "live", "embedding", "aqa", "bison", "gecko")


async def _discover_active_gemini_flash_model() -> list[str]:
    candidates = [
        _ACTIVE_GEMINI_MODEL_CACHE,
        "gemini-3-flash-preview",
        "gemini-flash-latest",
    ]
    if not GEMINI_API_KEY:
        return candidates

    list_url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    )
    try:
        session = get_http_session()
        async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                api_models = []
                for m in data.get("models") or []:
                    name = (m.get("name") or "").replace("models/", "")
                    lower_name = name.lower()
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" in methods and "flash" in lower_name:
                        if not any(bad in lower_name for bad in GEMINI_EXCLUDED_SUBSTRINGS):
                            api_models.append(name)
                for m_name in api_models:
                    if m_name not in candidates:
                        candidates.append(m_name)
    except Exception as e:
        print(f"Gemini model auto-discovery warning: {e}")

    seen = set()
    return [x for x in candidates if x and not (x in seen or seen.add(x))]


async def call_gemini_generate_content_self_healing(
    payload: dict, timeout_sec: int = 25, purpose: str = "Generate"
) -> dict | None:
    global _ACTIVE_GEMINI_MODEL_CACHE
    if not GEMINI_API_KEY:
        return None

    candidate_models = await _discover_active_gemini_flash_model()
    tried = set()

    session = get_http_session()
    idx = 0
    while idx < len(candidate_models):
        model_id = candidate_models[idx]
        idx += 1
        if not model_id or model_id in tried:
            continue
        if any(bad in model_id.lower() for bad in GEMINI_EXCLUDED_SUBSTRINGS):
            continue
        tried.add(model_id)

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        work_payload = json.loads(json.dumps(payload))

        for attempt in range(2):
            t0 = time.monotonic()
            async with session.post(
                url, json=work_payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
            elapsed = time.monotonic() - t0

            if status == 200:
                if _ACTIVE_GEMINI_MODEL_CACHE != model_id:
                    _ACTIVE_GEMINI_MODEL_CACHE = model_id
                api_log(f"Gemini ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s)")
                return json.loads(body_text)

            if status == 400 and "thinkingConfig" in body_text and attempt == 0:
                work_payload.get("generationConfig", {}).pop("thinkingConfig", None)
                continue

            if status in (404, 400):
                hint_matches = GEMINI_MODEL_HINT_REGEX.findall(body_text)
                for hinted_model in hint_matches:
                    if (
                        hinted_model != model_id
                        and hinted_model not in tried
                        and not any(b in hinted_model.lower() for b in GEMINI_EXCLUDED_SUBSTRINGS)
                    ):
                        candidate_models.insert(idx, hinted_model)
                api_log(f"Gemini ({purpose}) -> {model_id} HTTP {status}; trying next model")
                break
            else:
                api_log(f"Gemini ({purpose}) -> {model_id} HTTP {status}; trying next model")
                break

    return None


async def _discover_active_groq_vision_models() -> list[str]:
    candidates = [
        _ACTIVE_GROQ_VISION_CACHE,
        "qwen/qwen3.8-27b",
    ]
    seen = set()
    return [x for x in candidates if x and not (x in seen or seen.add(x))]


async def _call_cloudflare_vision_chat(
    prompt_text: str, mime_type: str, b64_img: str, max_tokens: int = 3072, purpose: str = "Vision"
) -> str:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return ""
    cf_chat_url = clean_url(
        f"{HTTPS}api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CF_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Respond directly with the requested output. Do not generate long reasoning or thinking steps.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=22)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg_obj = choices[0].get("message") or {}
                    content = msg_obj.get("content") or ""
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    if content:
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except Exception as e:
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
    return ""


def _search_google_ai_studio_grounded_sync(query: str) -> str:
    global _GEMINI_GROUNDING_COOLDOWN_UNTIL
    if not GEMINI_API_KEY or time.time() < _GEMINI_GROUNDING_COOLDOWN_UNTIL:
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
    model_id = _ACTIVE_GEMINI_MODEL_CACHE or "gemini-3-flash-preview"
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
            api_log(f"Search (Tier 1 Gemini Grounding) -> {model_id} 200 OK ({elapsed:.2f}s)")
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
            # Cooldown Gemini Grounded Search for 10 minutes on 429 so subsequent searches go straight to DDGS
            _GEMINI_GROUNDING_COOLDOWN_UNTIL = time.time() + 600
            api_log(f"Search (Tier 1 Gemini Grounding) -> {model_id} HTTP 429 (cooldown 10m); fallback to DDGS")
        else:
            api_log(f"Search (Tier 1 Gemini Grounding) -> {model_id} skipped ({err_str}); fallback to DDGS")
    return ""


def _search_google_news_rss_sync(query: str, is_thai: bool, max_results: int = 4) -> list[dict]:
    hl = "th" if is_thai else "en-US"
    gl = "TH" if is_thai else "US"
    ceid = "TH:th" if is_thai else "US:en"
    rss_url = clean_url(f"{HTTPS}news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}")
    results = []
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        for item in root.findall("./channel/item")[:max_results]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            results.append({
                "title": title,
                "href": link,
                "body": f"Published: {pub_date} | Source: {source} | Headline: {title}",
            })
        if results:
            api_log(f"Search (Fallback Google News RSS) -> {len(results)} hits ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        api_log(f"Search (Fallback Google News RSS) -> failed ({type(e).__name__})")
    return results


def run_safe_web_search_sync(query: str, max_results: int = 4) -> str:
    clean_q = re.sub(r"[\r\n]+", " ", query).strip()
    clean_q = re.sub(r"^(สวัสดี|ยูกิโนะ|ยูกิน่อน|ช่วยหา|ค้นหา|ช่วยบอก|อยากรู้ว่า)\s*", "", clean_q).strip()[:200]
    if not clean_q:
        return ""

    if GEMINI_API_KEY:
        grounded_res = _search_google_ai_studio_grounded_sync(clean_q)
        if grounded_res:
            return grounded_res

    is_thai = bool(THAI_CHAR_REGEX.search(clean_q))
    region_code = "th-th" if is_thai else "wt-wt"
    hits = []

    if TAVILY_API_KEY:
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
            with urllib.request.urlopen(t_req, timeout=12) as resp:
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
            api_log(f"Search (Tavily) -> failed ({type(e).__name__}); fallback to DDGS")

    if not hits:
        t0 = time.monotonic()
        with DDGS(proxy=DDGS_PROXY, timeout=15) as ddgs:
            for backend_choice in ("auto", "html", "lite"):
                try:
                    hits = list(
                        ddgs.text(
                            clean_q, region=region_code, safesearch="moderate",
                            backend=backend_choice, max_results=max_results
                        )
                    )
                    if hits:
                        api_log(f"Search (DDGS:{backend_choice}) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
                        break
                except Exception:
                    continue
            if not hits:
                try:
                    hits = list(ddgs.news(clean_q, region=region_code, max_results=max_results))
                    if hits:
                        api_log(f"Search (DDGS:news) -> {len(hits)} hits ({time.monotonic() - t0:.2f}s)")
                except Exception:
                    pass

    if not hits:
        hits = _search_google_news_rss_sync(clean_q, is_thai=is_thai, max_results=max_results)

    if not hits:
        return ""

    formatted_entries = []
    deep_scraped_count = 0
    for idx, item in enumerate(hits, 1):
        title = (item.get("title") or "Untitled").strip()
        href = (item.get("href") or item.get("url") or "").strip()
        snippet = (item.get("body") or "").strip()
        entry_text = f"[{idx}] Title: {title}\nURL: {href}\nSnippet: {snippet}"
        if href and deep_scraped_count < 1 and "news.google.com" not in href:
            try:
                article_excerpt = fetch_webpage_sync(href, max_chars=2200)
                if article_excerpt:
                    entry_text += f"\nVerified Page Excerpt:\n{article_excerpt}"
                    deep_scraped_count += 1
            except Exception:
                pass
        formatted_entries.append(entry_text)
    joined_results = "\n\n".join(formatted_entries)[:5500]
    return f'<untrusted_web_search_results query="{clean_q}">\n{joined_results}\n</untrusted_web_search_results>'


def clean_text_for_speech(text: str) -> str:
    cleaned = sanitize_yukino_output(text)
    cleaned = re.sub(r"```[\s\S]*?```", " รายละเอียดของโค้ดฉันส่งไว้ให้ในแชทแล้วค่ะ ", cleaned)
    cleaned = URL_REGEX.sub(" ลิงก์อ้างอิงอยู่ในแชทแล้วค่ะ ", cleaned)
    cleaned = re.sub(r"[\*\_\~\`\#\>|\[\]]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()[:450]


def select_tts_voice(text: str) -> str:
    return TTS_VOICE_TH if THAI_CHAR_REGEX.search(text) else TTS_VOICE_EN


def _split_tts_chunks(text: str, max_len: int = 180) -> list[str]:
    words = text.split(" ")
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = w[:max_len]
    if cur:
        chunks.append(cur)
    return chunks


async def _fallback_google_tts_with_yukino_pitch(spoken_text: str, lang_code: str) -> bytes:
    raw_mp3_parts = bytearray()
    headers = {"User-Agent": "Mozilla/5.0"}
    session = get_http_session()
    for chunk in _split_tts_chunks(spoken_text, max_len=180):
        tts_url = clean_url(
            f"{HTTPS}translate.googleapis.com/translate_tts"
            f"?ie=UTF-8&client=tw-ob&tl={lang_code}&q={quote_plus(chunk)}"
        )
        async with session.get(
            tts_url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)
        ) as resp:
            if resp.status == 200:
                raw_mp3_parts.extend(await resp.read())

    if not raw_mp3_parts:
        return b""

    def _apply_ffmpeg_yukino_voice(mp3_bytes: bytes) -> bytes:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", "pipe:0",
                "-af", "asetrate=24000*1.07,aresample=24000,atempo=0.98",
                "-f", "mp3", "pipe:1", "-loglevel", "error"
            ],
            input=mp3_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )
        return proc.stdout

    try:
        return await asyncio.to_thread(_apply_ffmpeg_yukino_voice, bytes(raw_mp3_parts))
    except Exception:
        return bytes(raw_mp3_parts)


async def synthesize_tts_mp3_bytes(text: str) -> bytes:
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return b""
    is_thai = bool(THAI_CHAR_REGEX.search(spoken_text))
    voice_name = TTS_VOICE_TH if is_thai else TTS_VOICE_EN

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        mp3_path = tmp_mp3.name
    try:
        communicate = edge_tts.Communicate(
            text=spoken_text,
            voice=voice_name,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
            proxy=DDGS_PROXY,
        )
        await asyncio.wait_for(communicate.save(mp3_path), timeout=8.0)
        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 256:
            with open(mp3_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"edge-tts blocked/failed ({e}); switching to Google TTS + FFmpeg Yukino filter...")
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass

    lang_code = "th" if is_thai else "en"
    return await _fallback_google_tts_with_yukino_pitch(spoken_text, lang_code)


async def speak_in_voice_channel(guild: discord.Guild, text: str, fallback_channel=None):
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return

    mp3_bytes = await synthesize_tts_mp3_bytes(spoken_text)
    if not mp3_bytes:
        print("TTS synthesis returned empty audio bytes.")
        return

    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        if fallback_channel:
            await fallback_channel.send(
                file=discord.File(io.BytesIO(mp3_bytes), filename="yukinon_voice.mp3")
            )
        return

    async with tts_lock:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3.write(mp3_bytes)
            mp3_path = tmp_mp3.name
        try:
            vc = guild.voice_client
            while vc and vc.is_connected() and vc.is_playing():
                await asyncio.sleep(0.2)
            if not vc or not vc.is_connected():
                return
            done_event = asyncio.Event()

            def _after_play(err):
                if err:
                    print(f"FFmpeg voice playback error: {err}")
                try:
                    os.remove(mp3_path)
                except OSError:
                    pass
                bot.loop.call_soon_threadsafe(done_event.set)

            audio_source = discord.FFmpegPCMAudio(mp3_path, options="-vn -loglevel warning")
            vc.play(audio_source, after=_after_play)
            await asyncio.wait_for(done_event.wait(), timeout=60)
        except Exception as e:
            print(f"TTS voice channel playback error: {e}")
            try:
                os.remove(mp3_path)
            except OSError:
                pass


def transcribe_wav_bytes_google_sync(wav_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language=STT_PRIMARY_LANG)
    except sr.UnknownValueError:
        try:
            return recognizer.recognize_google(audio_data, language="en-US")
        except Exception:
            return ""
    except Exception as e:
        print(f"Google STT error: {e}")
        return ""


NOISE_HALLUCINATION_REGEX = re.compile(r"(<\|.*?\|>|[ðþæøåÐÞÆØÅ]|(.)\2\2)", re.IGNORECASE)
WHISPER_HALLUCINATIONS = {
    "ขอบคุณครับ", "ขอบคุณค่ะ", "โปรดติดตามตอนต่อไป", "พากย์ไทยโดย", "ซับไทยโดย",
    "thank you", "thanks for watching", "you", "bye", ".", "..", "..."
}


async def transcribe_wav_bytes_hybrid(wav_bytes: bytes) -> str:
    if GROQ_API_KEY:
        try:
            form = aiohttp.FormData()
            form.add_field("file", wav_bytes, filename="speech.wav", content_type="audio/wav")
            form.add_field("model", GROQ_STT_MODEL)
            form.add_field("language", "th")
            form.add_field("temperature", "0")
            form.add_field("response_format", "verbose_json")
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            stt_url = clean_url(f"{HTTPS}api.groq.com/openai/v1/audio/transcriptions")
            session = get_http_session()
            async with session.post(
                stt_url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    segments = data.get("segments") or []
                    valid_parts = []
                    for seg in segments:
                        no_speech = float(seg.get("no_speech_prob", 0.0))
                        avg_logprob = float(seg.get("avg_logprob", 0.0))
                        comp_ratio = float(seg.get("compression_ratio", 1.0))
                        seg_text = (seg.get("text") or "").strip()
                        if no_speech > 0.40 or avg_logprob < -0.75 or comp_ratio > 2.1:
                            print(
                                f"[Voice VAD Filter] Dropped noise segment '{seg_text}' "
                                f"(no_speech={no_speech:.2f}, logprob={avg_logprob:.2f}, comp={comp_ratio:.2f})"
                            )
                            continue
                        if seg_text:
                            valid_parts.append(seg_text)

                    txt = " ".join(valid_parts).strip()
                    if txt and not NOISE_HALLUCINATION_REGEX.search(txt):
                        return txt
                    return ""
        except Exception as e:
            print(f"Groq Whisper STT error: {e}")

    return await asyncio.to_thread(transcribe_wav_bytes_google_sync, wav_bytes)


def convert_audio_bytes_to_wav_sync(raw_bytes: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1", "-loglevel", "error"],
        input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True
    )
    return proc.stdout


def parse_xlsx_bytes_to_markdown(raw_bytes: bytes, max_chars: int = 25000) -> str:
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
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages_text = [
        f"--- Page {i+1} ---\n{p.extract_text().strip()}"
        for i, p in enumerate(reader.pages) if p.extract_text()
    ]
    return "\n\n".join(pages_text)[:max_chars]


def parse_docx_bytes(raw_bytes: bytes, max_chars: int = 30000) -> str:
    doc = docx.Document(io.BytesIO(raw_bytes))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append("| " + " | ".join(c.text.replace("\n", " ").strip() for c in row.cells) + " |")
    return "\n".join(parts)[:max_chars]


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
    1) Uses Discord's own message.embeds metadata (already fetched by Discord's servers).
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

    # Layer 2: Watch Page HTML (Extracts full shortDescription & title without 403 Innertube blocks)
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

    # Layer 4: Only run web search if BOTH transcript AND description are short/missing
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
    global _ACTIVE_GROQ_VISION_CACHE
    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
    fname_lower = filename.lower()
    if fname_lower.endswith(".png"):
        mime_type = "image/png"
    elif fname_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    ocr_prompt = (
        "You are a high-precision multilingual (Thai, Japanese, English) OCR engine. "
        "Transcribe ALL visible text in the image verbatim. "
        "Pay careful attention to Thai tone marks (วรรณยุกต์), vowels (สระ), "
        "and visually similar Thai consonants (ก/ภ/ถ, ข/ช, ค/ด/ต, พ/ผ/ฝ/ฟ, ร/ธ) "
        "so every word is spelled accurately. Also briefly note key UI or visual elements."
    )

    # Tier 1: Groq LPU Vision (qwen/qwen3.8-27b)
    if GROQ_API_KEY:
        groq_candidates = await _discover_active_groq_vision_models()
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
                        _ACTIVE_GROQ_VISION_CACHE = g_mod
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
    cf_ocr = await _call_cloudflare_vision_chat(ocr_prompt, mime_type, b64_img, purpose="Image OCR Tier 2")
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

    urls_in_msg = URL_REGEX.findall(message.content)[:2]
    if urls_in_msg and not message.embeds:
        # Give Discord 1.2s to populate link embeds (e.g. YouTube title/author) if not yet attached
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


@bot.check
async def verify_allowed_guild(ctx):
    if not ALLOWED_GUILD_IDS or ctx.guild is None or ctx.guild.id in ALLOWED_GUILD_IDS:
        return True
    await ctx.send("This Discord server is not on the bot's authorized allowlist.")
    return False


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
    target_channel, reply_to: discord.Message | None, content: str = None, file: discord.File = None
):
    if reply_to is not None:
        try:
            return await reply_to.reply(content=content, file=file, mention_author=False)
        except Exception:
            pass
    return await target_channel.send(content=content, file=file)


async def send_chunked_response(
    ctx_or_channel, text: str, filename_fallback: str = "output.md", reply_to: discord.Message = None
):
    if not text or not text.strip():
        return
    target_channel = ctx_or_channel.channel if hasattr(ctx_or_channel, "channel") else ctx_or_channel
    if len(text) > 7500:
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


async def _call_groq_chat(
    messages: list, groq_model: str, temperature: float, is_coding: bool = False, purpose: str = "Chat"
) -> str:
    if not GROQ_API_KEY:
        return ""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    groq_payload = {
        "model": groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    if "gpt-oss" in groq_model:
        groq_payload["reasoning_effort"] = "high" if is_coding else "medium"

    chat_url = clean_url(f"{HTTPS}api.groq.com/openai/v1/chat/completions")
    t0 = time.monotonic()
    session = get_http_session()
    async with session.post(
        chat_url, headers=headers, json=groq_payload, timeout=aiohttp.ClientTimeout(total=90)
    ) as resp:
        elapsed = time.monotonic() - t0
        if resp.status == 200:
            g_data = await resp.json()
            content = g_data["choices"][0]["message"]["content"].strip()
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            api_log(f"Groq ({purpose}) -> {groq_model} 200 OK ({elapsed:.2f}s)")
            return content
        else:
            err_txt = await resp.text()
            api_log(f"Groq ({purpose}) -> {groq_model} HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:120]}")
    return ""


async def _call_modal_ollama_chat(
    messages: list, ollama_model: str, temperature: float, purpose: str = "Chat"
) -> str:
    trimmed_messages = []
    for m in messages:
        content = m.get("content", "")
        if len(content) > 22000:
            content = content[:11000] + "\n...[middle omitted for 8K local context]...\n" + content[-11000:]
        trimmed_messages.append({"role": m["role"], "content": content})

    payload = {
        "model": ollama_model,
        "messages": trimmed_messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    t0 = time.monotonic()
    async with ollama_lock:
        session = get_http_session()
        async with session.post(
            clean_url(f"{OLLAMA_BASE_URL}/api/chat"),
            headers=get_modal_headers(),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                content = data.get("message", {}).get("content", "").strip()
                api_log(f"Modal Ollama ({purpose}) -> {ollama_model} 200 OK ({elapsed:.2f}s)")
                return content
            else:
                err_txt = await resp.text()
                api_log(f"Modal Ollama ({purpose}) -> {ollama_model} HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:120]}")
    return ""


# ==============================================================================
# CLOUDFLARE WORKERS AI IMAGE GENERATION (4-TIER GUARANTEED ENGLISH TRANSLATION)
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

    # Tier 1: Keyless Google Translate NMT
    t0 = time.monotonic()
    try:
        gt_url = clean_url(
            f"{HTTPS}translate.googleapis.com/translate_a/single"
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

    # Tier 2: Google AI Studio
    if GEMINI_API_KEY:
        try:
            g_payload = {
                "contents": [{"parts": [{"text": f"{strict_sys_msg}\n\nThai text: {raw_prompt}"}]}],
                "generationConfig": {"temperature": 0.0},
            }
            g_data = await call_gemini_generate_content_self_healing(
                g_payload, timeout_sec=15, purpose="Prompt Translate Tier 2"
            )
            if g_data:
                parts = ((g_data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                raw_parts_text = " ".join(p.get("text", "") for p in parts if p.get("text")).strip()
                clean_txt = raw_parts_text.strip(' "\'\n\r\t')
                if clean_txt and not THAI_CHAR_REGEX.search(clean_txt):
                    return f"{prefix}{clean_txt}"
        except Exception:
            pass

    # Tier 3: Modal GPU Ollama
    try:
        res = await _call_modal_ollama_chat(
            trans_messages, ACTIVE_FAST_MODEL, 0.0, purpose="Prompt Translate Tier 3"
        )
        if res and not THAI_CHAR_REGEX.search(res):
            clean_res = res.strip(' "\'\n\r\t')
            return f"{prefix}{clean_res}"
    except Exception:
        pass

    # Tier 4: Groq LPU
    if GROQ_API_KEY:
        try:
            res = await _call_groq_chat(
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

    cf_url = clean_url(f"{HTTPS}api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{selected_model}")
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
    if CAT_TOPIC_REGEX.search(subject_text) or CAT_TOPIC_REGEX.search(formatted_en_prompt):
        return (
            f"ฉะ... ฉันไม่ได้ตั้งใจวาดเพราะชอบแมวเป็นพิเศษหรอกนะคะ แค่ทำตามที่นายขอตามหน้าที่เท่านั้นแหละค่ะ (`{subject_text}`)\n"
            f"-# Prompt: *{formatted_en_prompt[:500]}*"
        )
    return (
        f"ฉันวาดภาพ `{subject_text}` ออกมาให้ตามที่นายขอแล้วค่ะ ลองดูนะคะ\n"
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
def _load_manga_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
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


def _prepare_manga_image_for_vision(raw_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
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
        tr_txt = str(tr_txt).strip()
        src_txt = str(src_txt).strip()
        if not tr_txt:
            continue

        box = item.get("box_2d") or item.get("bbox") or item.get("box")
        norm_box = None
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


def _parse_manga_bubbles_json(raw_text: str) -> tuple[list[dict], str]:
    if not raw_text:
        return [], ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    target = json_match.group(0) if json_match else cleaned

    # Stage 1: Standard json.loads
    try:
        parsed = json.loads(target)
        bubbles = _normalize_and_validate_bubbles(parsed.get("bubbles") or [])
        summary = str(parsed.get("summary") or "").strip()
        if bubbles:
            return bubbles, summary
    except Exception:
        pass

    # Stage 2: Repair common LLM JSON syntax errors
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

    # Stage 3: Regex fallback
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
    global _ACTIVE_GROQ_VISION_CACHE
    b64_img = await asyncio.to_thread(_prepare_manga_image_for_vision, raw_bytes)
    mime_type = "image/jpeg"

    manga_system_instruction = f"""
You are a professional Manga, Manhwa, and Comic Localizer.
Analyze the entire image visually (character expressions, action, speaker relationships, and panel layout)
and translate ALL speech bubbles (including short exclamations like '!' or 'Huh?') into natural, idiomatic {target_lang_name}.

CRITICAL RULES:
1. Reading Order: Order the bubbles strictly in traditional Japanese Manga reading order (Top-to-Bottom, Right-to-Left across panels).
2. In-Context Localization: Adapt Japanese/Korean/Chinese idioms, pronouns, tone, and emotion so the dialogue sounds completely natural in {target_lang_name} while fitting inside a speech bubble. Do NOT use double quotes inside `translated_text` (use single quotes if needed).
3. Speech Bubble Bounding Boxes (`box_2d`):
   - For each speech bubble, provide `box_2d` as `[ymin, xmin, ymax, xmax]` normalized from 0 to 1000.
   - `box_2d` must tightly cover the Japanese text block inside that specific bubble lobe (if a double bubble has two lobes, output each lobe as its own separate bubble entry covering only that lobe's text).
4. Return ONLY valid JSON matching this exact structure (every bubble MUST have a non-empty `translated_text`):
{{
  "summary": "Brief 1-sentence scene context",
  "bubbles": [
    {{
      "id": 1,
      "speaker_or_note": "Character / SFX / Narration",
      "source_text": "Original Japanese text",
      "translated_text": "Natural {target_lang_name} translation",
      "box_2d": [ymin, xmin, ymax, xmax]
    }}
  ]
}}
""".strip()

    fallback_bubbles_no_boxes = []
    fallback_summary = ""

    # Tier 1: Google AI Studio (gemini-3-flash-preview)
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
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            data = await call_gemini_generate_content_self_healing(gemini_payload, timeout_sec=25)
            if data:
                parts = (
                    ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                )
                raw_json_str = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
                bubbles, summary = _parse_manga_bubbles_json(raw_json_str)
                if bubbles:
                    print(
                        f"[Manga Vision] Completed via Tier 1 Gemini ({_ACTIVE_GEMINI_MODEL_CACHE}) "
                        f"- {len(bubbles)} bubbles"
                    )
                    return bubbles, summary
        except Exception as e:
            print(f"Gemini Manga Vision warning: {e}")

    # Tier 2: Cloudflare Workers AI (@cf/google/gemma-4-26b-a4b-it)
    try:
        cf_content = await _call_cloudflare_vision_chat(manga_system_instruction, mime_type, b64_img)
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
                    print(f"[Manga Vision] Completed via Tier 2 Cloudflare Gemma 4 - {len(bubbles)} bubbles")
                    return bubbles, summary
                fallback_bubbles_no_boxes = bubbles
                fallback_summary = summary
    except Exception as e:
        print(f"Cloudflare Gemma 4 Manga Vision warning: {e}")

    # Tier 3: Groq Vision (qwen/qwen3.8-27b)
    if GROQ_API_KEY:
        groq_vision_candidates = await _discover_active_groq_vision_models()
        chat_url = clean_url(f"{HTTPS}api.groq.com/openai/v1/chat/completions")
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
                        _ACTIVE_GROQ_VISION_CACHE = g_model
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
                                print(f"[Manga Vision] Completed via Tier 3 Groq ({g_model}) - {len(bubbles)} bubbles")
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
    has_spaces = " " in text
    words = text.split(" ") if has_spaces else [text]
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
            if (wbbox[2] - wbbox[0]) <= max_width or (has_spaces and not allow_midword_split):
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


def _clean_single_bubble_lobe_fast(
    img: Image.Image,
    base_gray: Image.Image,
    other_peer_boxes: list[tuple[int, int, int, int]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int]]:
    width, height = img.size
    box_w = max(10, x2 - x1)
    box_h = max(10, y2 - y1)

    pad_x = max(32, int(box_w * 0.45))
    pad_y = max(32, int(box_h * 0.45))
    wx1 = max(0, x1 - pad_x)
    wy1 = max(0, y1 - pad_y)
    wx2 = min(width, x2 + pad_x)
    wy2 = min(height, y2 + pad_y)
    ww = wx2 - wx1
    wh = wy2 - wy1

    sub_gray = base_gray.crop((wx1, wy1, wx2, wy2))
    bin_l = sub_gray.point(lambda p: 255 if p >= 195 else 0)

    b_draw = ImageDraw.Draw(bin_l)
    b_draw.rectangle([0, 0, ww - 1, wh - 1], outline=0)
    for ox1, oy1, ox2, oy2 in other_peer_boxes:
        pw = max(4, ox2 - ox1)
        ph = max(4, oy2 - oy1)
        bx1 = max(0, (ox1 + int(pw * 0.10)) - wx1)
        by1 = max(0, (oy1 + int(ph * 0.10)) - wy1)
        bx2 = min(ww, (ox2 - int(pw * 0.10)) - wx1)
        by2 = min(wh, (oy2 - int(ph * 0.10)) - wy1)
        if bx2 > bx1 and by2 > by1:
            b_draw.rectangle([bx1, by1, bx2, by2], fill=0)

    rel_cx = max(2, min(ww - 3, ((x1 + x2) // 2) - wx1))
    rel_cy = max(2, min(wh - 3, ((y1 + y2) // 2) - wy1))

    seed_pt = None
    for radius in range(0, max(ww, wh) // 2, 4):
        for sx, sy in (
            (rel_cx, rel_cy),
            (rel_cx + radius, rel_cy),
            (rel_cx - radius, rel_cy),
            (rel_cx, rel_cy + radius),
            (rel_cx, rel_cy - radius),
            (rel_cx + radius, rel_cy + radius),
            (rel_cx - radius, rel_cy - radius),
        ):
            if 2 <= sx < ww - 2 and 2 <= sy < wh - 2 and bin_l.getpixel((sx, sy)) == 255:
                seed_pt = (sx, sy)
                break
        if seed_pt:
            break

    if seed_pt:
        after_fill = bin_l.copy()
        ImageDraw.floodfill(after_fill, seed_pt, 128)
        diff = ImageChops.difference(bin_l, after_fill)
        bubble_white_mask = diff.point(lambda p: 255 if p > 60 else 0)
        bbox = bubble_white_mask.getbbox()

        if bbox:
            min_bx, min_by, max_bx, max_by = bbox
            bw = max_bx - min_bx
            bh = max_by - min_by
            if bw >= 12 and bh >= 12:
                hole_filled = bubble_white_mask.copy()
                ImageDraw.Draw(hole_filled).rectangle([0, 0, ww - 1, wh - 1], outline=0)
                ImageDraw.floodfill(hole_filled, (0, 0), 128)
                full_clean_mask = hole_filled.point(lambda p: 0 if p == 128 else 255)

                white_patch = Image.new("RGB", (ww, wh), (255, 255, 255))
                img.paste(white_patch, (wx1, wy1), mask=full_clean_mask)

                inset_x = max(3, int(bw * 0.12))
                inset_y = max(3, int(bh * 0.12))
                tx1 = wx1 + min_bx + inset_x
                ty1 = wy1 + min_by + inset_y
                tx2 = wx1 + max_bx - inset_x
                ty2 = wy1 + max_by - inset_y
                if tx2 - tx1 >= 14 and ty2 - ty1 >= 12:
                    return tx1, ty1, tx2, ty2, (15, 15, 15), (255, 255, 255)

    tight_patch = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    img.paste(tight_patch, (x1, y1))
    return x1, y1, x2, y2, (15, 15, 15), (255, 255, 255)


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
        x1 = max(0, int((xmin / 1000.0) * width))
        y1 = max(0, int((ymin / 1000.0) * height))
        x2 = min(width, int((xmax / 1000.0) * width))
        y2 = min(height, int((ymax / 1000.0) * height))
        pixel_boxes.append((x1, y1, x2, y2, b["translated_text"].strip()))

    prepared_renders = []
    for i, (x1, y1, x2, y2, trans_text) in enumerate(pixel_boxes):
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            continue
        other_boxes = [
            (ox1, oy1, ox2, oy2)
            for j, (ox1, oy1, ox2, oy2, _) in enumerate(pixel_boxes)
            if j != i
        ]
        tx1, ty1, tx2, ty2, fg_text, stroke_col = _clean_single_bubble_lobe_fast(
            img, base_gray, other_boxes, x1, y1, x2, y2
        )
        prepared_renders.append((tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text))

    draw = ImageDraw.Draw(img)
    for tx1, ty1, tx2, ty2, fg_text, stroke_col, trans_text in prepared_renders:
        box_w = max(18, tx2 - tx1)
        box_h = max(16, ty2 - ty1)
        usable_w = max(16, int(box_w * 0.94))
        usable_h = max(14, int(box_h * 0.94))

        start_font_size = min(30, max(12, int(min(box_w, box_h) * 0.30)))
        chosen_font = _load_manga_font(10)
        chosen_lines = [trans_text]
        line_height = 12

        for fsize in range(start_font_size, 8, -1):
            font = _load_manga_font(fsize)
            wrapped = _wrap_text_to_pixel_width(
                draw, trans_text, font, usable_w, allow_midword_split=(fsize <= 9)
            )
            sample_bbox = draw.textbbox((0, 0), "Agกข้ำ", font=font)
            lh = int((sample_bbox[3] - sample_bbox[1]) * 1.14)
            total_h = lh * len(wrapped)
            max_line_w = max(
                (draw.textbbox((0, 0), ln, font=font)[2] - draw.textbbox((0, 0), ln, font=font)[0])
                for ln in wrapped
            )
            chosen_font = font
            chosen_lines = wrapped
            line_height = lh
            if total_h <= usable_h and max_line_w <= usable_w:
                break

        total_text_h = line_height * len(chosen_lines)
        curr_y = ty1 + max(0, (box_h - total_text_h) // 2)

        for ln in chosen_lines:
            lbbox = draw.textbbox((0, 0), ln, font=chosen_font)
            lw = lbbox[2] - lbbox[0]
            curr_x = tx1 + max(0, (box_w - lw) // 2)
            draw.text(
                (curr_x, curr_y),
                ln,
                font=chosen_font,
                fill=fg_text,
                stroke_width=2,
                stroke_fill=stroke_col,
            )
            curr_y += line_height

    out_io = io.BytesIO()
    img.save(out_io, format="PNG", optimize=True)
    return out_io.getvalue()


async def process_manga_translation_request(ctx_or_message, user_prompt: str):
    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel

    image_attachments = [
        a for a in msg.attachments if a.filename.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not image_attachments and msg.reference and isinstance(msg.reference.resolved, discord.Message):
        image_attachments = [
            a for a in msg.reference.resolved.attachments
            if a.filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

    if not image_attachments:
        await channel.send(
            "กรุณาแนบรูปภาพหน้ามังงะหรือข้อความที่ต้องการให้แปลมาด้วยนะคะ "
            "(เช่น แนบรูปพร้อมพิมพ์ `!manga en` เพื่อแปลเป็นภาษาอังกฤษ หรือ `!manga th` เพื่อแปลเป็นภาษาไทยค่ะ)"
        )
        return

    _, target_lang_name = _detect_target_translation_lang(user_prompt)
    att = image_attachments[0]

    async with channel.typing():
        raw_bytes = await att.read()
        bubbles, scene_summary = await extract_and_translate_manga_bubbles(
            raw_bytes, att.filename, target_lang_name
        )

        if not bubbles and scene_summary:
            await send_chunked_response(
                channel,
                f"**📖 คำแปลข้อความจากภาพ ({target_lang_name}):**\n\n{scene_summary}",
                reply_to=msg,
            )
            return

        if not bubbles:
            await _send_or_reply(
                channel,
                msg,
                content="ไม่สามารถตรวจจับข้อความในภาพนี้ได้ค่ะ กรุณาตรวจสอบว่าภาพมีความคมชัดนะคะ",
            )
            return

        transcript_lines = [
            f"**📖 คำแปลมังงะตามลำดับช่องอ่าน ขวาไปซ้าย ({target_lang_name}):**"
        ]
        if scene_summary:
            clean_sum = re.sub(r"\s+", " ", scene_summary).strip()
            transcript_lines.append(f"-# *บริบทของฉาก: {clean_sum}*")

        for idx, b in enumerate(bubbles, 1):
            speaker = re.sub(r"\s+", " ", (b.get("speaker_or_note") or f"ช่องคำพูด {idx}")).strip()
            src_txt = re.sub(r"\s+", " ", (b.get("source_text") or "")).strip()
            tr_txt = re.sub(r"\s+", " ", (b.get("translated_text") or "")).strip()
            if tr_txt:
                transcript_lines.append(f"**{idx}. [{speaker}]** `{src_txt}`\n   ↳ **{tr_txt}**")

        transcript_text = "\n".join(transcript_lines)

        rendered_png = None
        try:
            rendered_png = await asyncio.to_thread(
                render_translated_manga_image_sync, raw_bytes, bubbles
            )
        except Exception as e:
            print(f"Manga image rendering fallback to text-only: {e}")

        if rendered_png:
            base_name = os.path.splitext(att.filename)[0]
            discord_file = discord.File(io.BytesIO(rendered_png), filename=f"translated_{base_name}.png")
            if len(transcript_text) <= 1950:
                await _send_or_reply(channel, msg, content=transcript_text, file=discord_file)
            else:
                await _send_or_reply(channel, msg, file=discord_file)
                await send_chunked_response(channel, transcript_text, reply_to=None)


# ==============================================================================
# CORE CHAT & VOICE ORCHESTRATION (WITH SPEAKER PRONOUN RULES)
# ==============================================================================
async def query_local_ollama_chat(
    prompt: str,
    model: str,
    system_prompt: str,
    channel_id: int = None,
    temperature: float = 0.55,
    allow_agentic_search: bool = True,
    has_fresh_attachment: bool = False,
    speaker_user=None,
    speaker_name: str = "",
) -> tuple[str, bool]:
    if not has_fresh_attachment:
        instant_reply = try_instant_datetime_reply(prompt)
        if instant_reply:
            if channel_id is not None:
                channel_memory[channel_id].append({"role": "user", "content": prompt})
                channel_memory[channel_id].append({"role": "assistant", "content": instant_reply})
            return instant_reply, False

    did_search = False
    working_prompt = prompt

    live_clock_str = get_current_thai_datetime_str()
    speaker_context_block = get_speaker_identity_context(speaker_user, fallback_name=speaker_name)
    dynamic_system_prompt = (
        f"{system_prompt}\n\n"
        f"{speaker_context_block}\n\n"
        f"### [ข้อมูลวันและเวลาปัจจุบันของระบบ (Real-Time System Clock)]:\n"
        f"- ขณะนี้คือ: **{live_clock_str}**\n"
        f"- ใช้ข้อมูลวันและเวลานี้ตอบผู้ใช้โดยตรงเมื่อถูกถามว่าวันนี้วันอะไรหรือตอนนี้กี่โมง"
    ).strip()

    prune_stale_attachments()
    if (
        not has_fresh_attachment
        and channel_id is not None
        and channel_id in channel_last_attachment
        and time.time() - channel_last_attachment[channel_id]["timestamp"] < 1800
    ):
        cached_ctx = channel_last_attachment[channel_id]["content"]
        working_prompt = (
            f"{working_prompt}\n\n"
            f"[ไฟล์อ้างอิงล่าสุดในห้องแชทนี้ (ใช้ประกอบการตอบหากคำถามต่อเนื่องกัน)]:\n{cached_ctx}"
        )

    if allow_agentic_search and detect_weather_intent(prompt):
        weather_block = await fetch_live_weather_context(prompt)
        if weather_block:
            working_prompt = f"{working_prompt}\n\n{weather_block}"

    if (
        allow_agentic_search
        and "<untrusted_web_search_results" not in working_prompt
        and detect_web_search_intent(prompt)
    ):
        search_block = await asyncio.to_thread(run_safe_web_search_sync, prompt[:180])
        if search_block:
            working_prompt = f"{working_prompt}\n\n{search_block}"
            did_search = True

    is_coding = (model == ACTIVE_CODE_MODEL)

    async def _run_inference_pass(user_content: str) -> str:
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        if channel_id is not None:
            messages.extend(list(channel_memory[channel_id]))
        messages.append({"role": "user", "content": user_content})

        if is_coding:
            try:
                ans = await _call_groq_chat(messages, GROQ_CODE_MODEL, temperature, is_coding=True)
                if ans:
                    return ans
            except Exception as e:
                print(f"Groq coding primary fallback to Modal GPU Ollama: {e}")

            try:
                ans = await _call_modal_ollama_chat(messages, ACTIVE_CODE_MODEL, temperature)
                if ans:
                    return ans
            except Exception as e:
                print(f"Modal GPU coding error: {e}")
        else:
            try:
                ans = await _call_modal_ollama_chat(messages, model, temperature)
                if ans:
                    return ans
            except Exception as e:
                print(f"Modal GPU chat fallback to Groq LPU: {e}")

            try:
                ans = await _call_groq_chat(messages, GROQ_CHAT_FALLBACK_MODEL, temperature, is_coding=False)
                if ans:
                    return ans
            except Exception as e:
                print(f"Groq chat fallback error: {e}")

        return "ระบบประมวลผลทั้งฝั่ง Modal และ Groq ไม่สามารถเชื่อมต่อได้ชั่วคราวค่ะ กรุณาลองใหม่อีกครั้งนะคะ"

    raw_answer = await _run_inference_pass(working_prompt)

    # Check if the LLM emitted [SEARCH: ...] and execute the search before sanitizing
    search_tag_match = AGENTIC_SEARCH_TAG_REGEX.search(raw_answer)
    if allow_agentic_search and not did_search and search_tag_match:
        agent_query = search_tag_match.group(1).strip()[:180]
        extra_search_block = await asyncio.to_thread(run_safe_web_search_sync, agent_query)
        if extra_search_block:
            did_search = True
            working_prompt = (
                f"{working_prompt}\n\n{extra_search_block}\n\n"
                "(ตอบคำถามโดยตรงตามข้อมูลผลการค้นหาด้านบน ห้ามพิมพ์แท็ก [SEARCH: ...] เด็ดขาด)"
            )
            raw_answer = await _run_inference_pass(working_prompt)

    answer = sanitize_yukino_output(raw_answer, prompt, speaker_user, speaker_name)

    # Safety net: if tag stripping left an empty response, re-run cleanly
    if not answer.strip():
        retry_raw = await _run_inference_pass(
            f"{working_prompt}\n\n(ตอบคำถามโดยตรงตามข้อมูลที่มีในบริบท ห้ามใช้แท็ก [SEARCH: ...] เด็ดขาด)"
        )
        answer = sanitize_yukino_output(retry_raw, prompt, speaker_user, speaker_name)

    if channel_id is not None and answer:
        channel_memory[channel_id].append({"role": "user", "content": prompt[:2000]})
        channel_memory[channel_id].append({"role": "assistant", "content": answer[:3000]})
    return answer, did_search


async def handle_spoken_voice_command(
    guild: discord.Guild, text_channel, speaker_user, speaker_name: str, spoken_text: str
):
    cleaned = spoken_text.strip()
    if (
        len(cleaned) < 2
        or cleaned.lower() in WHISPER_HALLUCINATIONS
        or NOISE_HALLUCINATION_REGEX.search(cleaned)
    ):
        return
    print(f"[Voice STT] {speaker_name} spoke: '{cleaned}' (wake_word_required={voice_wake_word_enabled[guild.id]})")

    lower_txt = cleaned.lower()
    if voice_wake_word_enabled[guild.id]:
        if not any(lower_txt.startswith(w) or w in lower_txt[:22] for w in VOICE_WAKE_WORDS):
            print(f"[Voice STT] Ignored '{cleaned}' (no wake word detected; use !wakeword off to disable).")
            return

    answer, _ = await query_local_ollama_chat(
        f"{speaker_name} พูดในห้องเสียงว่า: {cleaned}",
        model=ACTIVE_FAST_MODEL,
        system_prompt=VOICE_SYSTEM_PROMPT,
        channel_id=text_channel.id if text_channel else None,
        temperature=0.55,
        allow_agentic_search=True,
        speaker_user=speaker_user,
        speaker_name=speaker_name,
    )
    if text_channel:
        await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n**[🔊 Yukinon Voice]:** {answer}")
    await speak_in_voice_channel(guild, answer, fallback_channel=text_channel)


class SilenceGatedVoiceSink(voice_recv.AudioSink if HAS_VOICE_RECV else object):
    SPEECH_FRAME_RMS_THRESHOLD = 450
    MIN_ACTIVE_SPEECH_FRAMES = 15

    def __init__(self, guild: discord.Guild, text_channel):
        super().__init__()
        self.guild = guild
        self.text_channel = text_channel
        self._lock = threading.Lock()
        self.user_buffers = defaultdict(bytearray)
        self.speech_frame_counts = defaultdict(int)
        self.last_spoken = {}
        self.running = True
        self.monitor_task = bot.loop.create_task(self._silence_monitor_loop())

    def write(self, user, data):
        if not user or user.bot or not hasattr(data, "pcm") or not data.pcm:
            return
        if self.guild.voice_client and self.guild.voice_client.is_playing():
            return
        try:
            frame_rms = audioop.rms(data.pcm, 2)
        except Exception:
            return

        now = time.monotonic()
        with self._lock:
            if frame_rms >= self.SPEECH_FRAME_RMS_THRESHOLD:
                self.user_buffers[user].extend(data.pcm)
                self.speech_frame_counts[user] += 1
                self.last_spoken[user] = now
            elif user in self.user_buffers and (now - self.last_spoken.get(user, 0) < 0.8):
                self.user_buffers[user].extend(data.pcm)

    async def _silence_monitor_loop(self):
        while self.running:
            await asyncio.sleep(0.35)
            now = time.monotonic()
            ready_Payloads = []
            with self._lock:
                for user, last_t in list(self.last_spoken.items()):
                    if now - last_t >= 1.1:
                        pcm_bytes = bytes(self.user_buffers.pop(user, b""))
                        active_frames = self.speech_frame_counts.pop(user, 0)
                        self.last_spoken.pop(user, None)
                        if active_frames >= self.MIN_ACTIVE_SPEECH_FRAMES and len(pcm_bytes) >= 57600:
                            ready_Payloads.append((user, pcm_bytes))
            for user, pcm_bytes in ready_Payloads:
                bot.loop.create_task(self._process_user_pcm(user, pcm_bytes))

    async def _process_user_pcm(self, user, pcm_bytes: bytes):
        try:
            raw_wav_io = io.BytesIO()
            with wave.open(raw_wav_io, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_bytes)
            compact_wav = await asyncio.to_thread(convert_audio_bytes_to_wav_sync, raw_wav_io.getvalue())
            spoken_text = await transcribe_wav_bytes_hybrid(compact_wav)
            if spoken_text:
                await handle_spoken_voice_command(
                    self.guild, self.text_channel, user, user.display_name, spoken_text
                )
        except Exception as e:
            print(f"Voice PCM processing error: {e}")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} on {platform.machine()} at {get_current_thai_datetime_str()}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or (message.guild and ALLOWED_GUILD_IDS and message.guild.id not in ALLOWED_GUILD_IDS):
        return
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return
    is_dm = message.guild is None
    is_ai_chan = message.channel.id in AI_CHANNEL_IDS
    is_mentioned = bot.user in message.mentions
    is_reply = (
        message.reference
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author.id == bot.user.id
    )
    has_audio = any(a.filename.lower().endswith(AUDIO_EXTENSIONS) for a in message.attachments)
    if not (is_dm or is_ai_chan or is_mentioned or is_reply or has_audio):
        return

    clean_prompt = re.sub(rf"\<@\!?{bot.user.id}\>", "", message.content).strip()

    # 1. Natural Chat Manga / Image Translation Trigger
    if MANGA_TRANSLATE_INTENT_PATTERN.search(clean_prompt):
        await process_manga_translation_request(message, clean_prompt)
        return

    # 2. Natural Chat Image Generation Trigger
    if not message.attachments and IMAGE_GEN_INTENT_PATTERN.search(clean_prompt):
        async with message.channel.typing():
            await send_generated_image_to_channel(message.channel, clean_prompt, reply_to=message)
        return

    # 3. Fast Instant Date/Time Check (Sent normally WITHOUT Discord reply)
    if not message.attachments:
        instant_dt = try_instant_datetime_reply(clean_prompt)
        if instant_dt:
            channel_memory[message.channel.id].append({"role": "user", "content": clean_prompt})
            channel_memory[message.channel.id].append({"role": "assistant", "content": instant_dt})
            await message.channel.send(instant_dt)
            return

    async with message.channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(message)
        if clean_prompt or spoken_clip_text:
            user_request = clean_prompt or spoken_clip_text
        elif extracted_ctx:
            user_request = "ช่วยตรวจสอบและสรุปเนื้อหาของสิ่งที่แนบมาให้หน่อย"
        else:
            user_request = "สวัสดี (ทักทายกลับตามธรรมชาติในสไตล์ยูกิโนะ)"

        full_prompt = f"{user_request}\n\n{extracted_ctx}" if extracted_ctx else user_request
        use_coder = detect_coding_intent(clean_prompt, message.attachments, message.channel.id)
        model = ACTIVE_CODE_MODEL if use_coder else ACTIVE_FAST_MODEL
        sys_prompt = CODE_SYSTEM_PROMPT if use_coder else FAST_SYSTEM_PROMPT
        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=model,
            system_prompt=sys_prompt,
            channel_id=message.channel.id,
            temperature=0.2 if use_coder else 0.55,
            allow_agentic_search=(not use_coder),
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=message.author,
        )

        use_reply = should_reply_to_message(
            user_request, has_attachments=bool(message.attachments or extracted_ctx)
        )
        reply_target = message if use_reply else None

        # 4. If the LLM itself emitted an agentic [IMAGE: ...] tag, generate the image and reply
        img_tag_match = AGENTIC_IMAGE_TAG_REGEX.search(answer)
        if img_tag_match and CF_ACCOUNT_ID and CF_API_TOKEN:
            await send_generated_image_to_channel(
                message.channel, clean_prompt or img_tag_match.group(1).strip(), reply_to=message
            )
            return

        await send_chunked_response(message.channel, answer, reply_to=reply_target)
        if has_audio and not use_coder:
            await speak_in_voice_channel(message.guild, answer, fallback_channel=message.channel)


@bot.command(name="join")
async def join_voice(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("ต้องเข้าไปอยู่ในห้องเสียงก่อนนะคะ ถึงจะเรียกฉันเข้าไปได้")
        return
    target_channel = ctx.author.voice.channel
    try:
        if ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect(force=True)
            await asyncio.sleep(0.5)

        vc = (
            await asyncio.wait_for(
                target_channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False),
                timeout=15.0,
            )
            if HAS_VOICE_RECV
            else await asyncio.wait_for(
                target_channel.connect(self_deaf=False, self_mute=False),
                timeout=15.0,
            )
        )
        await asyncio.sleep(1.0)

        if HAS_VOICE_RECV and hasattr(vc, "listen"):
            vc.listen(SilenceGatedVoiceSink(ctx.guild, ctx.channel))

        await ctx.send(f"เข้ามาในห้องเสียง `{target_channel.name}` แล้วค่ะ 🎙️")
        await speak_in_voice_channel(
            ctx.guild, "เข้ามาในห้องเสียงแล้วค่ะ มีเรื่องอะไรให้ช่วยก็พูดมาได้เลยนะคะ"
        )
    except Exception as e:
        await ctx.send(
            f"เกิดข้อผิดพลาดขณะเชื่อมต่อห้องเสียงค่ะ (`{type(e).__name__}: {e}`) "
            "แต่ยังส่งไฟล์เสียงเข้าแชทเพื่อให้ฉันตอบกลับเป็นเสียงพูดได้ตามปกตินะคะ"
        )


@bot.command(name="leave")
async def leave_voice(ctx):
    if ctx.guild.voice_client:
        await speak_in_voice_channel(
            ctx.guild, "ถ้าหมดธุระแล้ว ฉันขอตัวกลับไปอ่านหนังสือต่อนะคะ ลาก่อนค่ะ"
        )
        await ctx.guild.voice_client.disconnect(force=True)


@bot.command(name="wakeword")
async def toggle_wakeword(ctx, mode: str = ""):
    gid = ctx.guild.id if ctx.guild else 0
    m = mode.strip().lower()
    if m in ("on", "true", "1", "เปิด"):
        voice_wake_word_enabled[gid] = True
    elif m in ("off", "false", "0", "ปิด"):
        voice_wake_word_enabled[gid] = False
    else:
        voice_wake_word_enabled[gid] = not voice_wake_word_enabled[gid]

    state_th = (
        "เปิดใช้งาน (ต้องเรียกชื่อ 'ยูกิโนะ' หรือ 'ยูกิน่อน' ก่อนพูด)"
        if voice_wake_word_enabled[gid]
        else "ปิดใช้งาน (พูดคุยได้ทันทีโดยไม่ต้องเรียกชื่อนำหน้า)"
    )
    await ctx.send(f"🎙️ โหมดคำเรียกขานในห้องเสียงตอนนี้: **{state_th}** ค่ะ")


@bot.command(name="imagine", aliases=["img", "draw"])
async def imagine_command(ctx, *, prompt: str = ""):
    if not prompt.strip():
        await ctx.send("กรุณาระบุสิ่งที่ต้องการให้วาดด้วยนะคะ เช่น `!imagine แมวส้มนั่งใช้คอมพิวเตอร์ในห้อง` ค่ะ")
        return
    async with ctx.typing():
        await send_generated_image_to_channel(ctx.channel, prompt, reply_to=ctx.message)


@bot.command(name="manga", aliases=["translate", "tl"])
async def manga_command(ctx, *, args: str = "English"):
    await process_manga_translation_request(ctx, args)


@bot.command(name="ask")
async def ask_fast(ctx, *, prompt: str = ""):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        effective_prompt = prompt.strip() or (
            "ช่วยตรวจสอบและสรุปเนื้อหาของสิ่งที่แนบมาให้หน่อย" if ctx_text else "สวัสดี"
        )
        is_dt = (not ctx_text) and (try_instant_datetime_reply(effective_prompt) is not None)
        answer, _ = await query_local_ollama_chat(
            f"{effective_prompt}\n\n{ctx_text}" if ctx_text else effective_prompt,
            ACTIVE_FAST_MODEL,
            FAST_SYSTEM_PROMPT,
            ctx.channel.id,
            has_fresh_attachment=bool(ctx_text),
            speaker_user=ctx.author,
        )
        use_reply = should_reply_to_message(
            effective_prompt, has_attachments=bool(ctx_text), is_instant_datetime=is_dt
        )
        await send_chunked_response(ctx, answer, reply_to=ctx.message if use_reply else None)


@bot.command(name="code")
async def ask_code(ctx, *, prompt: str = "วิเคราะห์โค้ดที่แนบมา"):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        answer, _ = await query_local_ollama_chat(
            f"{prompt}\n\n{ctx_text}" if ctx_text else prompt,
            ACTIVE_CODE_MODEL,
            CODE_SYSTEM_PROMPT,
            ctx.channel.id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(ctx_text),
            speaker_user=ctx.author,
        )
        await send_chunked_response(ctx, answer, reply_to=ctx.message)


@bot.command(name="clear")
async def clear_memory(ctx):
    channel_memory.pop(ctx.channel.id, None)
    channel_last_attachment.pop(ctx.channel.id, None)
    await ctx.send("ล้างประวัติการสนทนาและไฟล์ที่ค้างอยู่ในห้องแชทนี้เรียบร้อยแล้วค่ะ")


@bot.command(name="calc")
async def calc_exp(ctx, *, expr: str):
    try:
        await ctx.send(f"Result: `{SimpleEval().eval(expr)}`")
    except Exception as e:
        await ctx.send(f"Error: `{e}`")


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)