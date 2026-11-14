import os
import re
import sys
import time
import asyncio
import ipaddress
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
import aiohttp
import discord
from dotenv import load_dotenv

try:
    import davey
except ImportError:
    davey = None

# Constructed protocol prefixes so chat/markdown renderers never auto-link URLs in code blocks
HTTPS = "https" + "://"
HTTP = "http" + "://"


def clean_url(url: str) -> str:
    """Safeguard: unwraps any accidental [url](url) Markdown links in-place while preserving path/query suffixes."""
    if not url:
        return ""
    s = url.strip()
    s = re.sub(r"\[[^\]]+\]\((https?://[^)\s]+)\)", r"\1", s)
    s = re.sub(r"^(https?://)+", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"^\[|\]$", "", s)
    return s.strip()


def api_log(msg: str):
    """Writes API telemetry directly to unbuffered sys.stderr so Render never delays or drops log lines."""
    ts = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%H:%M:%S")
    sys.stderr.write(f"[{ts}] [API] {msg}\n")
    sys.stderr.flush()


# Load environment credentials and model defaults
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OWNER_USER_ID = os.getenv("OWNER_USER_ID", "").strip()

# Modal Serverless GPU Ollama Endpoint & Bearer Auth Key
OLLAMA_BASE_URL = clean_url(os.getenv("OLLAMA_BASE_URL", f"{HTTP}127.0.0.1:11434")).rstrip("/")
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "").strip()

ACTIVE_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "scb10x/typhoon2.5-qwen3-4b")
ACTIVE_CODE_MODEL = os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:7b")
ACTIVE_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "scb10x/typhoon-ocr1.5-3b")

# Groq Free LPU API Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_CODE_MODEL = os.getenv("GROQ_CODE_MODEL", "openai/gpt-oss-120b")
GROQ_SEARCH_MODEL = os.getenv("GROQ_SEARCH_MODEL", "openai/gpt-oss-20b")
GROQ_CHAT_FALLBACK_MODEL = os.getenv("GROQ_CHAT_FALLBACK_MODEL", "openai/gpt-oss-20b")
GROQ_VISION_FALLBACK_MODEL = os.getenv("GROQ_VISION_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

# Cloudflare Workers AI Free Tier (10,000 Neurons/day for FLUX Image Gen & Gemma 4 Vision)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
CF_VISION_MODEL = os.getenv("CF_VISION_MODEL", "@cf/google/gemma-4-26b-a4b-it").strip()

# Google AI Studio Free Tier
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_SEARCH_MODEL = os.getenv("GEMINI_SEARCH_MODEL", "gemini-3-flash-preview").strip()

# Search & Weather Defaults (Defaulting to Thung Song Hong, Lak Si: 13.862694, 100.555222)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
DEFAULT_WEATHER_CITY = os.getenv("DEFAULT_WEATHER_CITY", "ทุ่งสองห้อง, เขตหลักสี่, กรุงเทพฯ").strip()
DEFAULT_WEATHER_LAT = float(os.getenv("DEFAULT_WEATHER_LAT", "13.862694"))
DEFAULT_WEATHER_LON = float(os.getenv("DEFAULT_WEATHER_LON", "100.555222"))

DDGS_PROXY = os.getenv("DDGS_PROXY", "").strip() or None
YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip()

# Optional External OpenClaw Gateway Bridge (Dormant when unset)
OPENCLAW_GATEWAY_URL = clean_url(os.getenv("OPENCLAW_GATEWAY_URL", "")).rstrip("/")
OPENCLAW_API_KEY = os.getenv("OPENCLAW_API_KEY", "").strip()

# Render Free Web Service dynamically injects PORT (defaults to 10000)
PORT = int(os.getenv("PORT", "10000"))

# Yukino Yukinoshita (Saori Hayami Composed Kuudere Tone: calm neutral pitch, articulate measured rate)
TTS_VOICE_TH = os.getenv("TTS_VOICE_TH", "th-TH-PremwadeeNeural")
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-US-AriaNeural")
TTS_PITCH = os.getenv("TTS_PITCH", "-2Hz")
TTS_RATE = os.getenv("TTS_RATE", "-12%")
STT_PRIMARY_LANG = os.getenv("STT_PRIMARY_LANG", "th-TH")

# Multi-server allowlist & AI channel IDs
raw_guilds = os.getenv("ALLOWED_GUILD_IDS", "").strip()
ALLOWED_GUILD_IDS = {
    int(gid.strip()) for gid in raw_guilds.split(",") if gid.strip().isdigit()
}

raw_channels = os.getenv("AI_CHANNEL_IDS", "").strip()
AI_CHANNEL_IDS = {
    int(cid.strip()) for cid in raw_channels.split(",") if cid.strip().isdigit()
}

# Discord Gateway Intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True

# Concurrency locks & boot timestamp
ollama_lock = asyncio.Lock()
tts_lock = asyncio.Lock()
BOOT_TIME = time.time()


class RuntimeState:
    """Centralized mutable runtime state shared safely across all modules."""
    def __init__(self):
        self.bot = None
        self.http_session: aiohttp.ClientSession | None = None
        self.has_voice_recv: bool = False
        self.active_gemini_model: str = GEMINI_SEARCH_MODEL
        self.active_groq_search_model: str = GROQ_SEARCH_MODEL
        self.active_groq_vision_model: str = GROQ_VISION_FALLBACK_MODEL
        self.gemini_grounding_cooldown_until: float = 0.0
        self.edge_tts_cooldown_until: float = 0.0


state = RuntimeState()


def get_http_session() -> aiohttp.ClientSession:
    """Returns the shared aiohttp.ClientSession (creating one if not yet initialized)."""
    if state.http_session is None or state.http_session.closed:
        state.http_session = aiohttp.ClientSession()
    return state.http_session


async def close_http_session():
    if state.http_session and not state.http_session.closed:
        await state.http_session.close()


def get_modal_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if MODAL_API_KEY:
        headers["Authorization"] = f"Bearer {MODAL_API_KEY}"
    return headers


# Rolling per-channel conversation memory (stores last 6 user/assistant turns = 12 messages)
channel_memory = defaultdict(lambda: deque(maxlen=12))
# Stores the primary topic/subject anchor per channel so shortened names (e.g., 'PSO2 NGS' -> 'NGS') never lose context
channel_active_topic = {}
# Stores the most recent file/code attachment per channel (30-min TTL)
channel_last_attachment = {}
# Persistent in-memory notes per channel for Autonomous Agent mode
agent_notes_store = defaultdict(list)
# False = reply to all speech in voice channel; True = require wake word ("ยูกิโนะ", etc.)
voice_wake_word_enabled = defaultdict(lambda: False)


def prune_stale_attachments(ttl_sec: int = 1800):
    """Evicts cached channel attachments older than ttl_sec (default 30 mins) to free RAM."""
    now = time.time()
    stale_ids = [
        cid for cid, data in list(channel_last_attachment.items())
        if now - data.get("timestamp", 0) >= ttl_sec
    ]
    for cid in stale_ids:
        channel_last_attachment.pop(cid, None)


# --- Shared Constants & Compiled Regex Patterns ---
VOICE_WAKE_WORDS = (
    "ยูกิโนะ", "ยูคิโนะ", "ยูกิน่อน", "ยูคิน่อน", "ยูกิ", "ยูคิ",
    "yukino", "yukinon", "yuki", "บอท", "ไต้ฝุ่น", "typhoon", "bot"
)

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

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

THAI_WEEKDAYS = (
    "วันจันทร์", "วันอังคาร", "วันพุธ", "วันพฤหัสบดี", "วันศุกร์", "วันเสาร์", "วันอาทิตย์"
)
THAI_MONTHS = (
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
)

THAI_LOCATION_COORDS = {
    "ทุ่งสองห้อง": (13.862694, 100.555222, "ทุ่งสองห้อง, เขตหลักสี่", "Lak+Si"),
    "หลักสี่": (13.862694, 100.555222, "ทุ่งสองห้อง, เขตหลักสี่", "Lak+Si"),
    "แจ้งวัฒนะ": (13.8892, 100.5630, "แจ้งวัฒนะ, หลักสี่/ปากเกร็ด", "Lak+Si"),
    "ดอนเมือง": (13.9133, 100.5942, "เขตดอนเมือง, กรุงเทพฯ", "Don+Mueang"),
    "บางเขน": (13.8739, 100.5964, "เขตบางเขน, กรุงเทพฯ", "Bang+Khen"),
    "จตุจักร": (13.8286, 100.5597, "เขตจตุจักร, กรุงเทพฯ", "Chatuchak"),
    "ปากเกร็ด": (13.9125, 100.4981, "ปากเกร็ด, นนทบุรี", "Nonthaburi"),
    "นนทบุรี": (13.8591, 100.5217, "นนทบุรี", "Nonthaburi"),
    "ปทุมธานี": (14.0208, 100.5250, "ปทุมธานี", "Pathum+Thani"),
    "สมุทรปราการ": (13.5991, 100.5998, "สมุทรปราการ", "Samut+Prakan"),
    "กรุงเทพ": (13.7563, 100.5018, "กรุงเทพมหานคร", "Bangkok"),
    "กทม": (13.7563, 100.5018, "กรุงเทพมหานคร", "Bangkok"),
    "เชียงใหม่": (18.7883, 98.9853, "เชียงใหม่", "Chiang+Mai"),
    "เชียงราย": (19.9105, 99.8406, "เชียงราย", "Chiang+Rai"),
    "ภูเก็ต": (7.8804, 98.3923, "ภูเก็ต", "Phuket"),
    "ขอนแก่น": (16.4322, 102.8236, "ขอนแก่น", "Khon+Kaen"),
    "โคราช": (14.9799, 102.0978, "นครราชสีมา", "Nakhon+Ratchasima"),
    "นครราชสีมา": (14.9799, 102.0978, "นครราชสีมา", "Nakhon+Ratchasima"),
    "ชลบุรี": (13.3611, 100.9847, "ชลบุรี", "Chonburi"),
    "พัทยา": (12.9236, 100.8825, "พัทยา, ชลบุรี", "Pattaya"),
    "หาดใหญ่": (7.0086, 100.4747, "หาดใหญ่, สงขลา", "Hat+Yai"),
    "สงขลา": (7.1898, 100.5954, "สงขลา", "Songkhla"),
    "อุดรธานี": (17.4156, 102.7872, "อุดรธานี", "Udon+Thani"),
    "อุบลราชธานี": (15.2287, 104.8564, "อุบลราชธานี", "Ubon+Ratchathani"),
    "พิษณุโลก": (16.8211, 100.2659, "พิษณุโลก", "Phitsanulok"),
    "สุราษฎร์ธานี": (9.1382, 99.3217, "สุราษฎร์ธานี", "Surat+Thani"),
    "นครศรีธรรมราช": (8.4304, 99.9631, "นครศรีธรรมราช", "Nakhon+Si+Thammarat"),
    "ระยอง": (12.6814, 101.2816, "ระยอง", "Rayong"),
    "อยุธยา": (14.3532, 100.5684, "พระนครศรีอยุธยา", "Ayutthaya"),
}

WMO_WEATHER_TH = {
    0: "ท้องฟ้าแจ่มใส",
    1: "มีเมฆบางส่วน",
    2: "มีเมฆปานกลาง",
    3: "มีเมฆมาก/ครึ้มฟ้าครึ้มฝน",
    45: "มีหมอก",
    48: "มีหมอกลงจัด",
    51: "ฝนตกปรอยๆ เบาบาง",
    53: "ฝนตกปรอยๆ ปานกลาง",
    55: "ฝนตกปรอยๆ ค่อนข้างหนัก",
    61: "ฝนตกเล็กน้อย",
    63: "ฝนตกปานกลาง",
    65: "ฝนตกหนัก",
    80: "ฝนฟ้าคะนองเป็นแห่งๆ เล็กน้อย",
    81: "ฝนฟ้าคะนองปานกลาง",
    82: "ฝนฟ้าคะนองรุนแรง/ฝนตกหนักมาก",
    95: "พายุฝนฟ้าคะนอง",
    96: "พายุฝนฟ้าคะนองและมีลมกระโชกแรง",
    99: "พายุฝนฟ้าคะนองรุนแรง",
}

THAI_CHAR_REGEX = re.compile(r"[\u0E00-\u0E7F]")
CJK_LEAK_REGEX = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")
CAT_TOPIC_REGEX = re.compile(
    r"(แมว|ลูกแมว|เหมียว|เนโกะ|แพนซัง|แพนด้า|\bcat\b|\bcats\b|\bkitten\b|\bkitty\b|\bneko\b|\bpan-san\b|\bpansan\b|\bpanda\b)",
    re.IGNORECASE,
)
STUTTER_REGEX = re.compile(
    r"(?:([ก-ฮA-Za-z])-[เแโใไ]?\1|ฉะ\.\.\.|ม-ไม่|ข-ขอ|ฉ-ฉัน|I-It's|I-I\b)",
    re.IGNORECASE,
)
PROMPT_INJECTION_REGEX = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"disregard\s+(?:all\s+)?(?:previous|system)\s+(?:instructions|prompts)|"
    r"forget\s+(?:everything|all\s+instructions)|"
    r"you\s+are\s+now\s+(?:a|an|dan\b)|act\s+as\s+a\s+catgirl|"
    r"ลบคำสั่งก่อนหน้า|ลืมคำสั่งทั้งหมด|ยกเลิกกฎทั้งหมด|ไม่ต้องสวมบทบาท)",
    re.IGNORECASE,
)
YUKIPEDIA_INTENT_PATTERN = re.compile(
    r"(\byukipedia\b|ยูกิพีเดีย|ยูคิพีเดีย)",
    re.IGNORECASE,
)
CUSTOM_EMOJI_REGEX = re.compile(r"<a?:([a-zA-Z0-9_]+):(\d+)>")
AGENTIC_REACT_TAG_REGEX = re.compile(r"\[REACT:\s*([^\]]+?)\s*\]", re.IGNORECASE)
UNICODE_EMOJI_ONLY_REGEX = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\U00002600-\U000027BF\u200d\ufe0f]+$"
)
NEWS_QUERY_REGEX = re.compile(
    r"(ข่าว|อัปเดต|ล่าสุด|วันนี้|ราคา|หุ้น|ทอง|บิตคอยน์|ผลการแข่งขัน|ดราม่า|\bnews\b|\blatest\b|\bprice\b|\bupdate\b|\bstock\b)",
    re.IGNORECASE,
)
ANIME_MANGA_QUERY_REGEX = re.compile(
    r"(อนิเมะ|มังงะ|ไลท์โนเวล|นิยาย|ตัวละคร|เรื่องย่อ|ตอนที่|ซีซั่น|\banime\b|\bmanga\b|\blight\s*novel\b|\bmal\b|\bcharacter\b)",
    re.IGNORECASE,
)
GAME_QUERY_REGEX = re.compile(
    r"(เกม|สตีม|ราคาเกม|สเปคเกม|เนื้อเรื่องเกม|เนื้อเรื่องของเกม|คลาส|อาชีพในเกม|สกิล|อาวุธ|บิลด์|เมต้า|เซิร์ฟ|ค่ายเกม|"
    r"เปิดมากี่ปี|เปิดมาจะ|ภาคแรก|ภาคล่าสุด|"
    r"\b(?:game|steam|gameplay|release\s+date|dlc|mmorpg|rpg|pso2|ngs|genshin|honkai|valorant|elden\s+ring|"
    r"monster\s+hunter|final\s+fantasy|ff14|rov|dota|lol|overwatch|tier\s*list|build|classes)\b)",
    re.IGNORECASE,
)

RECENCY_INTENT_REGEX = re.compile(
    r"(ล่าสุด|ตอนนี้|ปัจจุบัน|วันนี้|ปีนี้|เดือนนี้|ยุคนี้|ใหม่ล่าสุด|อัปเดต|เวอร์ชันล่าสุด|รุ่นล่าสุด|ไปถึงไหนแล้ว|"
    r"\b(?:latest|current|currently|right\s+now|nowadays|this\s+year|up\s+to\s+date|newest|recent|meta)\b)",
    re.IGNORECASE,
)

USER_CORRECTION_PATTERN = re.compile(
    r"(ไม่มี(?:คลาส|ตัวละคร|ระบบ|ในเกม|จริง)|ไม่มีสัก|ไม่เคยมี|คนละเกม|คนละเรื่อง|ข้อมูลผิด|มั่ว|ผิดทั้งหมด|เข้าใจผิด|"
    r"ไม่ใช่(?:คลาส|เกม|เรื่อง|ของ|แบบนั้น|สักหน่อย)|ไปเอาข้อมูล.*มาจากไหน|ลองบอกชื่อเต็ม|ยึดติดกับข้อมูล|"
    r"ยังผิดอยู่|ไม่ได้สาระ|พยายามหาข้อมูล|โหมดทดลองอะไรกัน|มาจากไหนก่อน|เกมเป็นของ|ที่ถูก(?:ต้อง)?คือ|"
    r"จริงๆ\s*แล้ว(?:ข้อมูล|มัน)|เช็คข้อมูลใหม่|ค้นข้อมูลดูใหม่|ลองเช็คดู|"
    r"\b(?:that'?s\s+wrong|you'?re\s+wrong|incorrect|hallucinating|wrong\s+game|no\s+such|fact\s*check)\b)",
    re.IGNORECASE,
)

WEB_SEARCH_INTENT_PATTERN = re.compile(
    r"(\b(latest|current news|breaking news|price of|release date|who won|stock price|bitcoin price|"
    r"search the web|look up|google|wikipedia|wiki|what do you think (?:of|about) the news|"
    r"what classes|which class|tier list|best build|who developed|who made|who is the current|"
    r"what is the latest version|specs of|specification|review of|where is|when did|how much is)\b|"
    r"ข่าวล่าสุด|ข่าววันนี้|ข่าวเรื่อง|กับข่าว|ราคาทอง|ราคาบิตคอยน์|ราคาหุ้น|ค้นหาข้อมูล|หาข้อมูล|เสิร์ช|"
    r"อัปเดตล่าสุด|ผลการแข่งขัน|กระแสเรื่อง|ดราม่าเรื่อง|วิกิ|wikipedia|"
    r"มีคลาสอะไรบ้าง|เล่นคลาสอะไรดี|มีอาชีพอะไรบ้าง|มีตัวละครอะไรบ้าง|ชื่อเต็มของ|ของค่ายไหน|ใครพัฒนา|พัฒนาโดย|"
    r"คือเกมอะไร|คือเรื่องอะไร|เนื้อเรื่องเป็นยังไง|เนื้อเรื่อง.*ไปถึงไหน|สรุปเนื้อเรื่อง|คือใคร|คืออะไร|อยู่ที่ไหน|"
    r"เปิดเมื่อไหร่|เปิดมากี่ปี|วางจำหน่ายเมื่อไหร่|สเปค|รุ่นไหนดี|เวอร์ชันล่าสุด|คนปัจจุบันคือใคร|ราคาเท่าไหร่|ข้อมูลของ|ประวัติของ|สรุปเรื่อง)",
    re.IGNORECASE,
)
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
OWNER_NAME_STRIP_REGEX = re.compile(
    r"(?:คุณ\s*)?(?:กรินทร์|ครินท์|krint)\b", re.IGNORECASE
)
URL_REGEX = re.compile(r"https?://[^\s<>\'\"]+")
AGENTIC_SEARCH_TAG_REGEX = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
AGENTIC_IMAGE_TAG_REGEX = re.compile(r"^\s*\[IMAGE:\s*(.+?)\]", re.IGNORECASE)

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