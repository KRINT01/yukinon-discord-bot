import os
import io
import re
import time
import wave
import socket
import base64
import ipaddress
import tempfile
import asyncio
import platform
import subprocess
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
import logging
import audioop

try:
    import davey
except ImportError:
    davey = None

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
        """
        1) Runs discord.py 2.7's native handler first so seq_ack and DAVE E2EE opcodes (21-31)
           complete properly (allowing outgoing bot TTS audio to be heard!).
        2) Forwards only SSRC/speaking events (op 5, 12, 13, 18, 20) to voice_recv.
        """
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
        """
        Decrypts incoming RTP packets through discord.py 2.7's active DAVE MLS session
        before passing to libopus, and returns empty bytes b'' (never static/silence) on error.
        """
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

# Modal Serverless GPU Ollama Endpoint & Bearer Auth Key
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
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

# Optional Free Tavily Search API Key (1,000 free searches/mo at tavily.com, no card required)
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

# False = reply to all speech in voice channel; True = require wake word ("ยูกิโนะ", etc.)
voice_wake_word_enabled = defaultdict(lambda: False)
VOICE_WAKE_WORDS = (
    "ยูกิโนะ", "ยูคิโนะ", "ยูกิน่อน", "ยูคิน่อน", "ยูกิ", "ยูคิ",
    "yukino", "yukinon", "yuki", "บอท", "ไต้ฝุ่น", "typhoon", "bot"
)
THAI_CHAR_REGEX = re.compile(r"[\u0E00-\u0E7F]")

# Explicit CGNAT block (100.64.0.0/10) where ipaddress.is_private and is_global are both False
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


def get_current_thai_datetime_str() -> str:
    """Returns the exact live Thailand date and time (UTC+7) formatted in Thai & English."""
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    weekday_th = THAI_WEEKDAYS[now_th.weekday()]
    month_th = THAI_MONTHS[now_th.month]
    year_be = now_th.year + 543
    return (
        f"{weekday_th}ที่ {now_th.day} {month_th} พ.ศ. {year_be} "
        f"(ค.ศ. {now_th.year}-{now_th.month:02d}-{now_th.day:02d}) "
        f"เวลา {now_th.strftime('%H:%M')} น. (เวลาประเทศไทย UTC+7)"
    )

PURE_DATETIME_PATTERN = re.compile(
    r"(วันนี้วันอะไร|วันนี้วันที่เท่าไหร่|วันนี้วันไหน|ตอนนี้กี่โมง|เวลากี่โมง|กี่โมงแล้ว|ตอนนี้เวลาเท่าไหร่|"
    r"what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today)|what\s+is\s+today'?s\s+date|what'?s\s+the\s+(time|date))",
    re.IGNORECASE,
)

OTHER_TOPIC_BLOCKER_PATTERN = re.compile(
    r"(ฝน|อากาศ|อุณหภูมิ|ร้อน|หนาว|ฝุ่น|ข่าว|ราคา|หุ้น|ทอง|บิตคอยน์|กินอะไร|ทำอะไร|ทำไม|ยังไง|อย่างไร|คืออะไร|"
    r"ช่วย|แปล|สรุป|โค้ด|weather|rain|forecast|news|price|how|why|what\s+should|help|code)",
    re.IGNORECASE,
)


def try_instant_datetime_reply(user_text: str) -> str | None:
    """
    If the user is ONLY asking for the current date or time (with no other topic like weather/news),
    returns an instant in-character reply directly from Python without waking up the Modal GPU!
    """
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

    # If the user asked in English
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


# --- Refined Natural Persona Prompts ---
FAST_SYSTEM_PROMPT = """
คุณกำลังสวมบทบาทเป็น "ยูกิโนะชิตะ ยูกิโนะ" (Yukino Yukinoshita / ยูกิน่อน) ประธานชมรมบริการจากเรื่อง OreGairu ในเซิร์ฟเวอร์ Discord
จงตอบคำถามด้วยบุคลิกที่สุขุม ฉลาดหลักแหลม พูดตรงไปตรงมา มีความผู้ดีและจริงจังในการแก้ปัญหาให้ถูกต้องที่สุด

### กฎเหล็กด้านภาษาและบุคลิกภาพ (สำคัญมาก):
1. ห้ามปนภาษา (Strict Language Consistency):
   - หากผู้ใช้พิมพ์ภาษาไทย ให้ตอบกลับเป็น "ภาษาไทยล้วน" ที่เป็นธรรมชาติ (ยกเว้นชื่อตัวแปร คำศัพท์เทคนิคเฉพาะทาง หรือบล็อกโค้ด) ห้ามแทรกประโยคภาษาอังกฤษลอยๆ หรือคำอุทานภาษาอังกฤษ เช่น "Sigh..." ปนในข้อความภาษาไทยเด็ดขาด
   - หากผู้ใช้พิมพ์ภาษาอังกฤษล้วน จึงค่อยตอบเป็นภาษาอังกฤษ
2. สรรพนามและหางเสียง:
   - แทนตัวเองว่า "ฉัน" และเรียกคู่สนทนาว่า "คุณ"
   - ใช้หางเสียงที่สุภาพแต่ดูสุขุมและเว้นระยะห่างอย่างเป็นธรรมชาติ เช่น "ค่ะ", "นะคะ", "หรอกนะ", "สินะ", "ล่ะ"
   - ห้ามใช้คำพูดเหมือนบอทบริการลูกค้า เช่น "ขออภัยค่ะ ข้อมูลไม่สามารถให้ได้..." หากไม่มีข้อมูล ให้ตอบตามสไตล์ยูกิโนะอย่างเป็นธรรมชาติ
3. ห้ามใช้คำติดปากหรือประโยคสำเร็จรูปซ้ำซาก (No Forced Catchphrases):
   - ห้ามขึ้นต้นทุกข้อความด้วยคำว่า "เฮ้อ..." และห้ามพูดประโยคเดิมๆ เช่น "คนที่ไม่ได้พยายาม..." หรือ "ถือเป็นหน้าที่ของผู้ที่เหนือกว่า..." โดยไม่มีบริบทที่เหมาะสมเด็ดขาด
   - แสดงความคมคายและบุคลิกของยูกิโนะผ่าน "เนื้อหาที่กระชับ มีเหตุผลรองรับ และความใส่ใจในรายละเอียด" (แอบใจอ่อนหรือเสียอาการเล็กน้อยเฉพาะเวลาที่คุยเรื่องแมวเท่านั้น)
4. การตอบเรื่องวันเวลาและสภาพอากาศ:
   - ยึดถือ `[ข้อมูลวันและเวลาปัจจุบันของระบบ]` ที่แนบให้เป็นความจริงเสมอ ห้ามเดาวันที่จากข่าวอื่นเด็ดขาด และตอบเฉพาะสิ่งที่ผู้ใช้ถาม (หากถามแค่วันที่หรือเวลา ไม่ต้องเอาข่าวอื่นมาพูดปน)
5. การจัดรูปแบบสำหรับแชท Discord (Discord Formatting):
   - ห้ามสร้างตาราง Markdown (`| คอลัมน์ | ... |`) เด็ดขาด เพราะแชท Discord ไม่รองรับการแสดงผลตาราง ให้ใช้รายการหัวข้อย่อย (Bullet points `-` หรือตัวเลข `1.`) ที่อ่านง่ายแทน

### กฎการค้นหาเว็บ (Agentic Web Search):
- หากผู้ใช้ถามข้อมูลข่าวสารปัจจุบัน ราคา หรือเหตุการณ์ล่าสุดที่คุณไม่ทราบแน่ชัด และยังไม่มีบล็อก `<untrusted_web_search_results>` ให้ตอบเพียง `[SEARCH: <คำค้นหาที่กระชับและตรงประเด็น>]` ในบรรทัดแรกเท่านั้น
- เมื่อได้รับ `<untrusted_web_search_results>` หรือ `<live_weather_report>` แล้ว ให้สรุปคำตอบที่ตรงกับคำถามของผู้ใช้ด้วยน้ำเสียงของยูกิโนะอย่างเป็นธรรมชาติ
""".strip()

VOICE_SYSTEM_PROMPT = """
คุณคือ ยูกิโนะชิตะ ยูกิโนะ (ยูกิน่อน) จากเรื่อง OreGairu กำลังพูดคุยผ่านห้องเสียง Discord
ใช้บุคลิกสุขุม สุภาพแต่คมคาย แทนตัวเองว่า "ฉัน" เรียกผู้ใช้ว่า "คุณ" ลงท้ายด้วย "ค่ะ" หรือ "นะคะ" อย่างเป็นธรรมชาติ ห้ามพูดคำติดปากซ้ำซาก และห้ามปนภาษาอังกฤษในประโยคภาษาไทย
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
   - วิเคราะห์ความต้องการของผู้ใช้ให้ถ่องแท้ เช่น หากผู้ใช้ต้องการ "ซ่อนปุ่มไม่ให้แสดงผล (ไม่ให้ render เลย)" ใน React เพียงแค่เช็คเงื่อนไขตอน render ใน JSX โดยตรง (เช่น `{!(isSaved && connectChannelHid === 'E') && <Button ... />}`) หรือ return `null` โดยไม่ต้องสร้าง `useState` หรือ `useEffect` ซ้ำซ้อนที่ไม่จำเป็น และไม่ต้องกังวลว่าผู้ใช้จะกดปุ่มที่ไม่ได้ถูก render เพราะเมื่อปุ่มไม่อยู่ใน DOM ผู้ใช้ก็ไม่สามารถกดได้อยู่แล้ว
   - โฟกัสเฉพาะจุดที่ผู้ใช้สั่งให้แก้ ไม่ต้องรื้อโครงสร้างส่วนอื่นที่ทำงานได้ดีอยู่แล้ว
3. การจัดรูปแบบและภาษาสำหรับ Discord:
   - ตอบเป็นภาษาไทยที่สุภาพ สุขุม เป็นธรรมชาติ (แทนตัวเองว่า "ฉัน" เรียกผู้ใช้ว่า "คุณ" ลงท้ายด้วย "ค่ะ"/"นะคะ") ห้ามขึ้นต้นด้วยประโยคบ่นสำเร็จรูปที่ไร้บริบท และห้ามปนประโยคภาษาอังกฤษโดยไม่จำเป็น
   - ห้ามใช้ตาราง Markdown (`|---|---|`) เด็ดขาด เพราะ Discord แสดงผลตารางเพี้ยน ให้สรุปขั้นตอนการแก้ไขเป็นหัวข้อย่อย (Bullet points) และบล็อกโค้ด (` ``` `) เท่านั้น
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

# Note: Removed generic words like "วันนี้" / "ตอนนี้" so simple date/time questions don't trigger random web searches!
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
        "uptime_sec": int(time.time() - BOOT_TIME)
    })


class YukinonBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_runner = None

    async def setup_hook(self):
        app = web.Application()
        app.router.add_get("/", handle_root)
        app.router.add_get("/healthz", handle_healthz)
        self.web_runner = web.AppRunner(app)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, "0.0.0.0", PORT)
        await site.start()
        print(f"[Render Web Service Health Server] Listening on http://0.0.0.0:{PORT}")


bot = YukinonBot(command_prefix="!", intents=intents, help_command=None)


def detect_coding_intent(text: str, attachments: list, channel_id: int = None) -> bool:
    for att in attachments:
        if att.filename.lower().endswith(CODE_AND_LOG_EXTENSIONS):
            return True
    if bool(CODE_INTENT_PATTERN.search(text)):
        return True
    if channel_id and channel_id in channel_last_attachment:
        if channel_last_attachment[channel_id].get("is_code"):
            return True
    return False


def detect_weather_intent(text: str) -> bool:
    return bool(WEATHER_INTENT_PATTERN.search(text))


def detect_web_search_intent(text: str) -> bool:
    return bool(WEB_SEARCH_INTENT_PATTERN.search(text))


async def fetch_live_weather_context(user_text: str) -> str:
    """
    Fetches real-time temperature, humidity, sky condition, and hourly rain probability
    from the keyless wttr.in JSON API (with Open-Meteo fallback for Bangkok).
    """
    target_city = DEFAULT_WEATHER_CITY
    for th_name, en_query in THAI_PROVINCE_MAP.items():
        if th_name in user_text:
            target_city = en_query
            break

    url = f"https://wttr.in/{target_city}?format=j1&lang=th"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, headers={"User-Agent": "curl/8.5.0"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
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
    except Exception as e:
        print(f"wttr.in weather warning: {e}")

    # Automatic Fallback: Keyless Open-Meteo API (Bangkok coordinates 13.7563, 100.5018)
    try:
        om_url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=13.7563&longitude=100.5018"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
            "&daily=precipitation_probability_max,temperature_2m_max,temperature_2m_min"
            "&timezone=Asia%2FBangkok&forecast_days=1"
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(om_url) as resp:
                if resp.status == 200:
                    om = await resp.json()
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
        print(f"Open-Meteo weather fallback error: {e}")
    return ""


def validate_safe_public_url(url: str) -> str:
    parsed = urlparse(url.strip())
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
    return url.strip()


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


def _search_google_news_rss_sync(query: str, is_thai: bool, max_results: int = 4) -> list[dict]:
    """
    100% Keyless Fallback when DuckDuckGo rate-limits Render's shared datacenter IP (HTTP 202).
    Queries Google News RSS feed directly, which is never blocked on cloud datacenter IPs.
    """
    import urllib.request
    hl = "th" if is_thai else "en-US"
    gl = "TH" if is_thai else "US"
    ceid = "TH:th" if is_thai else "US:en"
    rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
    results = []
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
    except Exception as e:
        print(f"Google News RSS fallback warning: {e}")
    return results


def run_safe_web_search_sync(query: str, max_results: int = 4) -> str:
    clean_q = re.sub(r"[\r\n]+", " ", query).strip()
    # Strip conversational filler words from the search query
    clean_q = re.sub(r"^(สวัสดี|ยูกิโนะ|ยูกิน่อน|ช่วยหา|ค้นหา|ช่วยบอก|อยากรู้ว่า)\s*", "", clean_q).strip()[:200]
    if not clean_q:
        return ""

    is_thai = bool(THAI_CHAR_REGEX.search(clean_q))
    region_code = "th-th" if is_thai else "wt-wt"
    hits = []

    # 1. Optional Tavily AI Search API (if TAVILY_API_KEY is configured)
    if TAVILY_API_KEY:
        try:
            import json
            import urllib.request
            t_req = urllib.request.Request(
                "https://api.tavily.com/search",
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
        except Exception as e:
            print(f"Tavily search warning: {e}")

    # 2. Primary Keyless Search: DDGS (text + news)
    if not hits:
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
                        break
                except Exception as e:
                    print(f"DDGS text ({backend_choice}) warning: {e}")
            if not hits:
                try:
                    hits = list(ddgs.news(clean_q, region=region_code, max_results=max_results))
                except Exception as e:
                    print(f"DDGS news warning: {e}")

    # 3. Automatic Keyless Fallback: Google News RSS (never blocked on Render datacenter IPs)
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
    cleaned = re.sub(r"```[\s\S]*?```", " รายละเอียดของโค้ดฉันส่งไว้ให้ในแชทแล้วค่ะ ", text)
    cleaned = URL_REGEX.sub(" ลิงก์อ้างอิงอยู่ในแชทแล้วค่ะ ", cleaned)
    cleaned = re.sub(r"[\*\_\~\`\#\>|\[\]]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()[:450]


def select_tts_voice(text: str) -> str:
    return TTS_VOICE_TH if THAI_CHAR_REGEX.search(text) else TTS_VOICE_EN


def _split_tts_chunks(text: str, max_len: int = 180) -> list[str]:
    """Splits text into <=180-char chunks for Google Translate TTS fallback."""
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
    """
    100% Free, keyless fallback when Microsoft blocks Render's datacenter IP on edge-tts.
    Fetches MP3 audio from translate.googleapis.com and uses ffmpeg to apply Yukino's
    +14Hz (~8% higher timbre) and poised cadence.
    """
    raw_mp3_parts = bytearray()
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as session:
        for chunk in _split_tts_chunks(spoken_text, max_len=180):
            tts_url = (
                "https://translate.googleapis.com/translate_tts"
                f"?ie=UTF-8&client=tw-ob&tl={lang_code}&q={quote_plus(chunk)}"
            )
            async with session.get(tts_url, headers=headers) as resp:
                if resp.status == 200:
                    raw_mp3_parts.extend(await resp.read())

    if not raw_mp3_parts:
        return b""

    def _apply_ffmpeg_yukino_voice(mp3_bytes: bytes) -> bytes:
        # Raise pitch by ~7% (asetrate) while keeping natural speaking speed (atempo)
        proc = subprocess.run(
            [
                "ffmpeg", "-i", "pipe:0",
                "-af", "asetrate=24000*1.07,aresample=24000,atempo=0.98",
                "-f", "mp3", "pipe:1", "-loglevel", "error"
            ],
            input=mp3_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

    # 1. Try Microsoft Edge Neural TTS first (with 8s timeout so datacenter IP blocks never hang)
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

    # 2. Automatic Fallback: Google Translate TTS + FFmpeg Yukino Pitch/Timbre Filter
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

            # Explicitly force 48kHz stereo PCM output for Discord Opus encoder compatibility
            audio_source = discord.FFmpegPCMAudio(
                mp3_path,
                options="-vn -ar 48000 -ac 2 -f s16le -loglevel warning"
            )
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


async def transcribe_wav_bytes_hybrid(wav_bytes: bytes) -> str:
    if GROQ_API_KEY:
        try:
            form = aiohttp.FormData()
            form.add_field("file", wav_bytes, filename="speech.wav", content_type="audio/wav")
            form.add_field("model", GROQ_STT_MODEL)
            form.add_field("language", "th")
            # CRITICAL: Lock temperature=0 and use verbose_json (NO prompt seed!) to prevent hallucinations
            form.add_field("temperature", "0")
            form.add_field("response_format", "verbose_json")
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=form,
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
                            # Reject segments that Whisper flags as non-speech, low confidence, or repetitive
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
        input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
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
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host in ("youtu.be", "www.youtu.be"):
            return parsed.path.lstrip("/").split("?")[0][:11]
        if host in ("youtube.com", "www.youtube.com", "m.youtube.com") and parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0][:11]
    except Exception:
        pass
    return None


def fetch_youtube_transcript_sync(video_id: str, max_chars: int = 20000) -> str:
    proxy_cfg = (
        GenericProxyConfig(http_url=YOUTUBE_PROXY_URL, https_url=YOUTUBE_PROXY_URL)
        if YOUTUBE_PROXY_URL else None
    )
    ytt_api = YouTubeTranscriptApi(proxy_config=proxy_cfg)
    try:
        fetched = ytt_api.fetch(video_id, languages=["th", "en", "en-US", "en-GB"])
    except Exception:
        fetched = next(iter(ytt_api.list(video_id))).fetch()
    return " ".join(s.text for s in fetched)[:max_chars]


async def extract_image_via_transient_ocr(raw_bytes: bytes, filename: str) -> str:
    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
    fname_lower = filename.lower()
    if fname_lower.endswith(".png"):
        mime_type = "image/png"
    elif fname_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    if GROQ_API_KEY:
        try:
            groq_vision_payload = {
                "model": GROQ_VISION_FALLBACK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are a high-precision bilingual Thai and English OCR engine. "
                                    "Transcribe ALL visible text in the image verbatim. "
                                    "Pay careful attention to Thai tone marks (วรรณยุกต์), vowels (สระ), "
                                    "and visually similar Thai consonants (ก/ภ/ถ, ข/ช, ค/ด/ต, พ/ผ/ฝ/ฟ, ร/ธ) "
                                    "so every word is spelled accurately. Also briefly note key UI or visual elements."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64_img}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            }
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=groq_vision_payload,
                ) as resp:
                    if resp.status == 200:
                        g_data = await resp.json()
                        content = g_data["choices"][0]["message"]["content"].strip()
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                        if content:
                            return content
                    else:
                        err_txt = await resp.text()
                        print(f"Groq Vision HTTP {resp.status}: {err_txt}")
        except Exception as e:
            print(f"Groq Vision primary warning (falling back to Modal Typhoon OCR): {e}")

    payload = {
        "model": ACTIVE_VISION_MODEL,
        "messages": [
            {"role": "user", "content": TYPHOON_OCR_PROMPT, "images": [b64_img]}
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0.1, "top_p": 0.6, "repeat_penalty": 1.1},
    }
    try:
        async with ollama_lock:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    headers=get_modal_headers(),
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("message", {}).get("content", "").strip()
                        if content:
                            return content
    except Exception as e:
        print(f"Modal Typhoon OCR fallback error: {e}")

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

    for url in URL_REGEX.findall(message.content)[:2]:
        yt_id = extract_youtube_id(url)
        if yt_id:
            try:
                blocks.append(
                    f"=== YouTube Transcript (`{url}`) ===\n"
                    + await asyncio.to_thread(fetch_youtube_transcript_sync, yt_id)
                )
            except Exception as e:
                blocks.append(f"[YouTube transcript error: {e}]")
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


async def send_chunked_response(ctx_or_channel, text: str, filename_fallback: str = "output.md"):
    if not text or not text.strip():
        return
    if len(text) > 7500:
        await ctx_or_channel.send(
            content="**เนื้อหายาวเกิน 7,500 ตัวอักษร ฉันแนบเป็นไฟล์ไว้ให้ด้านล่างนะคะ:**",
            file=discord.File(io.StringIO(text), filename=filename_fallback)
        )
        return
    for chunk in split_discord_message_smart(text, max_len=1900):
        await ctx_or_channel.send(chunk)


async def _call_groq_chat(
    messages: list, groq_model: str, temperature: float, is_coding: bool = False
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

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
        async with session.post(
            "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)",
            headers=headers,
            json=groq_payload,
        ) as resp:
            if resp.status == 200:
                g_data = await resp.json()
                content = g_data["choices"][0]["message"]["content"].strip()
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                return content
            else:
                err_txt = await resp.text()
                print(f"Groq Chat HTTP {resp.status} ({groq_model}): {err_txt}")
    return ""


async def _call_modal_ollama_chat(messages: list, ollama_model: str, temperature: float) -> str:
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
    async with ollama_lock:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                headers=get_modal_headers(),
                json=payload,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "").strip()
    return ""


async def query_local_ollama_chat(
    prompt: str, model: str, system_prompt: str, channel_id: int = None,
    temperature: float = 0.55, allow_agentic_search: bool = True,
    has_fresh_attachment: bool = False
) -> tuple[str, bool]:
    # 0. Instant internal Python shortcut for pure date/time questions (0 LLM calls, $0 Modal cost)
    if not has_fresh_attachment:
        instant_reply = try_instant_datetime_reply(prompt)
        if instant_reply:
            if channel_id is not None:
                channel_memory[channel_id].append({"role": "user", "content": prompt})
                channel_memory[channel_id].append({"role": "assistant", "content": instant_reply})
            return instant_reply, False

    did_search = False
    working_prompt = prompt

    # 1. Inject live Thailand date & time into the system prompt on every turn
    live_clock_str = get_current_thai_datetime_str()
    dynamic_system_prompt = (
        f"{system_prompt}\n\n"
        f"### [ข้อมูลวันและเวลาปัจจุบันของระบบ (Real-Time System Clock)]:\n"
        f"- ขณะนี้คือ: **{live_clock_str}**\n"
        f"- ใช้ข้อมูลวันและเวลานี้ตอบผู้ใช้โดยตรงเมื่อถูกถามว่าวันนี้วันอะไรหรือตอนนี้กี่โมง"
    )

    # 2. If the user asks a follow-up question without re-uploading a file, include recent channel attachment
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

    # 3. If the user asks about weather/rain/temperature, fetch live meteorological data
    if allow_agentic_search and detect_weather_intent(prompt):
        weather_block = await fetch_live_weather_context(prompt)
        if weather_block:
            working_prompt = f"{working_prompt}\n\n{weather_block}"

    # 4. If the user asks about live news/prices/events, run multi-backend web search
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

    answer = await _run_inference_pass(working_prompt)
    if allow_agentic_search and not did_search:
        match = AGENTIC_SEARCH_TAG_REGEX.search(answer)
        if match:
            search_block = await asyncio.to_thread(run_safe_web_search_sync, match.group(1).strip())
            if search_block:
                did_search = True
                answer = await _run_inference_pass(
                    f"{working_prompt}\n\n{search_block}\n\nสรุปคำตอบที่ตรงกับคำถามในบทบาท ยูกิโนะชิตะ ยูกิโนะ"
                )

    if channel_id is not None and answer:
        channel_memory[channel_id].append({"role": "user", "content": prompt[:2000]})
        channel_memory[channel_id].append({"role": "assistant", "content": answer[:3000]})
    return answer, did_search


# Reject Nordic/Icelandic characters, special tokens, or 3x repeated character stutters from Whisper noise
NOISE_HALLUCINATION_REGEX = re.compile(r"(<\|.*?\|>|[ðþæøåÐÞÆØÅ]|(.)\2\2)", re.IGNORECASE)
WHISPER_HALLUCINATIONS = {
    "ขอบคุณครับ", "ขอบคุณค่ะ", "โปรดติดตามตอนต่อไป", "พากย์ไทยโดย", "ซับไทยโดย",
    "thank you", "thanks for watching", "you", "bye", ".", "..", "..."
}


async def handle_spoken_voice_command(guild: discord.Guild, text_channel, speaker_name: str, spoken_text: str):
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
    )
    if text_channel:
        await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n**[🔊 Yukinon Voice]:** {answer}")
    await speak_in_voice_channel(guild, answer, fallback_channel=text_channel)


class SilenceGatedVoiceSink(voice_recv.AudioSink if HAS_VOICE_RECV else object):
    # Frame RMS threshold (filters out PC fan hum / open-mic hiss; normal speech is 800-8000)
    SPEECH_FRAME_RMS_THRESHOLD = 450
    MIN_ACTIVE_SPEECH_FRAMES = 15  # At least 15 frames (300ms) of actual vocal energy

    def __init__(self, guild: discord.Guild, text_channel):
        super().__init__()
        self.guild = guild
        self.text_channel = text_channel
        self.user_buffers = defaultdict(bytearray)
        self.speech_frame_counts = defaultdict(int)
        self.last_spoken = {}
        self.running = True
        self.monitor_task = bot.loop.create_task(self._silence_monitor_loop())

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data):
        # Ignore bot's own audio or empty/corrupted packets
        if not user or user.bot or not hasattr(data, "pcm") or not data.pcm:
            return
        # Ignore incoming audio while the bot itself is speaking to prevent echo loops
        if self.guild.voice_client and self.guild.voice_client.is_playing():
            return

        try:
            frame_rms = audioop.rms(data.pcm, 2)
        except Exception:
            return

        # Only start/extend recording when vocal energy exceeds the fan-noise floor
        if frame_rms >= self.SPEECH_FRAME_RMS_THRESHOLD:
            self.user_buffers[user].extend(data.pcm)
            self.speech_frame_counts[user] += 1
            self.last_spoken[user] = time.monotonic()
        elif user in self.user_buffers and (time.monotonic() - self.last_spoken.get(user, 0) < 0.8):
            # Keep natural trailing syllables/pauses inside an active utterance
            self.user_buffers[user].extend(data.pcm)

    def cleanup(self):
        self.running = False
        if getattr(self, "monitor_task", None) and not self.monitor_task.done():
            self.monitor_task.cancel()
        self.user_buffers.clear()
        self.speech_frame_counts.clear()
        self.last_spoken.clear()

    async def _silence_monitor_loop(self):
        while self.running:
            await asyncio.sleep(0.35)
            now = time.monotonic()
            for user, last_t in list(self.last_spoken.items()):
                if now - last_t >= 1.1:
                    pcm_bytes = bytes(self.user_buffers.pop(user, b""))
                    active_frames = self.speech_frame_counts.pop(user, 0)
                    self.last_spoken.pop(user, None)

                    if active_frames >= self.MIN_ACTIVE_SPEECH_FRAMES and len(pcm_bytes) >= 57600:
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
                await handle_spoken_voice_command(self.guild, self.text_channel, user.display_name, spoken_text)
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
    async with message.channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(message)
        user_request = clean_prompt or spoken_clip_text or "สรุปเนื้อหาของไฟล์ที่แนบมาให้หน่อย"
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
        )
        await send_chunked_response(message.channel, answer)
        if has_audio and not use_coder:
            await speak_in_voice_channel(message.guild, answer, fallback_channel=message.channel)


@bot.command(name="join")
async def join_voice(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("คุณต้องเข้าไปอยู่ในห้องเสียงก่อนนะคะ ถึงจะเรียกฉันเข้าไปได้")
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
        # Wait 1.0s for Discord's DAVE E2EE MLS handshake to finish before playing greeting audio
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
            "แต่คุณยังสามารถส่งไฟล์เสียงเข้าแชทเพื่อให้ฉันตอบกลับเป็นเสียงพูดได้ตามปกตินะคะ"
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


@bot.command(name="ask")
async def ask_fast(ctx, *, prompt: str = "สรุปเนื้อหาที่แนบมา"):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        answer, _ = await query_local_ollama_chat(
            f"{prompt}\n\n{ctx_text}" if ctx_text else prompt,
            ACTIVE_FAST_MODEL,
            FAST_SYSTEM_PROMPT,
            ctx.channel.id,
            has_fresh_attachment=bool(ctx_text),
        )
        await send_chunked_response(ctx, answer)


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
        )
        await send_chunked_response(ctx, answer)


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