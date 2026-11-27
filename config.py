import os
import re
import gc
import sys
import json
import time
import shutil
import ctypes
import asyncio
import tempfile
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


_GLIBC_LIBC = None


def _configure_glibc_low_ram_allocator():
    """
    Locks Linux glibc `M_TRIM_THRESHOLD` (-1) and `M_MMAP_THRESHOLD` (-3) to 128 KB and caps
    `M_ARENA_MAX` (-8) (default 2, override via `MALLOC_ARENA_MAX`) so multi-megabyte Pillow/OpenCV
    buffers allocated inside `asyncio.to_thread()` workers use `mmap()` and return cleanly to the OS
    instead of fragmenting per-thread `brk` arenas on 512 MB Render containers.
    """
    global _GLIBC_LIBC
    try:
        libc = ctypes.CDLL("libc.so.6")
        _GLIBC_LIBC = libc
        arena_max = max(1, int(os.getenv("MALLOC_ARENA_MAX", "2") or "2"))
        libc.mallopt(-1, 131072)     # M_TRIM_THRESHOLD = 128 KB
        libc.mallopt(-3, 131072)     # M_MMAP_THRESHOLD = 128 KB
        libc.mallopt(-8, arena_max)  # M_ARENA_MAX = 2 (caps worker-thread arenas)
        libc.malloc_trim(0)
    except Exception:
        pass


_configure_glibc_low_ram_allocator()

# Constructed protocol prefix so chat/markdown renderers never auto-link URLs in code blocks
HTTPS = "https" + "://"


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

# Hardware & Concurrency Scaling (defaults tuned for Render 512 MB / 0.1 vCPU; scale via env on VPS/ARM64 hosts)
CONTAINER_RAM_LIMIT_MB = max(128, int(os.getenv("CONTAINER_RAM_LIMIT_MB", "512") or "512"))
OPENCV_THREADS = max(1, int(os.getenv("OPENCV_THREADS", "1") or "1"))
MANGA_CONCURRENCY = max(1, int(os.getenv("MANGA_CONCURRENCY", "1") or "1"))
VOICE_STT_CONCURRENCY = max(1, int(os.getenv("VOICE_STT_CONCURRENCY", "2") or "2"))
MAX_CACHED_CHANNELS = max(8, int(os.getenv("MAX_CACHED_CHANNELS", "64") or "64"))

# Groq Free LPU API Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_CODE_MODEL = os.getenv("GROQ_CODE_MODEL", "openai/gpt-oss-120b")
GROQ_SEARCH_MODEL = os.getenv("GROQ_SEARCH_MODEL", "openai/gpt-oss-20b")
GROQ_CHAT_FALLBACK_MODEL = os.getenv("GROQ_CHAT_FALLBACK_MODEL", "openai/gpt-oss-20b")
GROQ_VISION_FALLBACK_MODEL = os.getenv("GROQ_VISION_FALLBACK_MODEL", "qwen/qwen3.8-27b")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

# Cloudflare Workers AI Free Tier (10,000 Neurons/day: FLUX Image, Gemma 4 Vision/Chat, GPT-OSS 120B/20B)
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell").strip()
CF_VISION_MODEL = os.getenv("CF_VISION_MODEL", "@cf/google/gemma-4-26b-a4b-it").strip()
CF_VISION_FALLBACK_MODEL = os.getenv("CF_VISION_FALLBACK_MODEL", "@cf/qwen/qwen3.8-27b").strip()
CF_DECISION_MODEL = os.getenv("CF_DECISION_MODEL", "@cf/cloudflare/clef-flash").strip()
CF_CODE_MODEL = os.getenv("CF_CODE_MODEL", "@cf/openai/gpt-oss-120b").strip()
CF_CHAT_FAST_MODEL = os.getenv("CF_CHAT_FAST_MODEL", "@cf/zai-org/glm-4.7-flash").strip()

# Google AI Studio Free Tier (Decisions API + Gemini 3.5 Flash-Lite -> 3.1 Flash-Lite -> 3 Flash Preview + Hydrogen + Gemma 4 + Live STT)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_DECISION_MODEL = os.getenv("GEMINI_DECISION_MODEL", "gemini-decision-1").strip()
GEMINI_HEAVY_MODEL = os.getenv("GEMINI_HEAVY_MODEL", "gemini-4-hydrogen").strip()
GEMINI_SEARCH_MODEL = os.getenv("GEMINI_SEARCH_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip()
GEMINI_PREVIEW_MODEL = os.getenv("GEMINI_PREVIEW_MODEL", "gemini-3-flash-preview").strip()
GEMINI_TRANSCRIBE_LIVE_MODEL = os.getenv("GEMINI_TRANSCRIBE_LIVE_MODEL", "gemini-3.5-transcribe-live").strip()
GEMMA_CHAT_MODEL = os.getenv("GEMMA_CHAT_MODEL", "gemma-4-31b-it").strip()
GEMMA_VISION_MODEL = os.getenv("GEMMA_VISION_MODEL", "gemma-4-26b-a4b-it").strip()

# OpenRouter Free Tier (Emergency Multi-Page MangaBatch Vision Fallback when Google AI Studio is down)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MANGA_MODEL = os.getenv("OPENROUTER_MANGA_MODEL", "nvidia/nemotron-3.5-super-vl:free").strip()

# Search & Weather Defaults (Defaulting to Thung Song Hong, Lak Si: 13.862694, 100.555222)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
DEFAULT_WEATHER_CITY = os.getenv("DEFAULT_WEATHER_CITY", "ทุ่งสองห้อง, เขตหลักสี่, กรุงเทพฯ").strip()
DEFAULT_WEATHER_LAT = float(os.getenv("DEFAULT_WEATHER_LAT", "13.862694"))
DEFAULT_WEATHER_LON = float(os.getenv("DEFAULT_WEATHER_LON", "100.555222"))

DDGS_PROXY = os.getenv("DDGS_PROXY", "").strip() or None
YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip()
SAUCENAO_API_KEY = os.getenv("SAUCENAO_API_KEY", "").strip()

# Render Free Web Service dynamically injects PORT (defaults to 10000)
PORT = int(os.getenv("PORT", "10000"))

# Yukino Yukinoshita Voice Profile (Cool, calm, crisp articulation)
# TTS_PRIMARY_ENGINE: "paradee" (offline MakHamTTS Paradee 1.1 INT8 for Thai + Google TTS for EN/fallback) or "google"
TTS_PRIMARY_ENGINE = os.getenv("TTS_PRIMARY_ENGINE", "paradee").strip().lower()
TTS_PARADEE_SPEED = float(os.getenv("TTS_PARADEE_SPEED", "1.05"))
TTS_PARADEE_PITCH = float(os.getenv("TTS_PARADEE_PITCH", "0.94"))
TTS_PARADEE_MODEL_DIR = os.getenv("TTS_PARADEE_MODEL_DIR", "/app/models/paradee_thai").strip()
TTS_PARADEE_IDLE_UNLOAD_SEC = max(30, int(os.getenv("TTS_PARADEE_IDLE_UNLOAD_SEC", "180") or "180"))
TTS_PARADEE_RAM_GUARD_MB = max(220, int(os.getenv("TTS_PARADEE_RAM_GUARD_MB", "340") or "340"))
TTS_PARADEE_TIMEOUT_SEC = max(3.0, float(os.getenv("TTS_PARADEE_TIMEOUT_SEC", "12.0") or "12.0"))
TTS_GOOGLE_PITCH = float(os.getenv("TTS_GOOGLE_PITCH", "0.97"))
TTS_GOOGLE_SPEED = float(os.getenv("TTS_GOOGLE_SPEED", "1.24"))
TTS_SILENCE_TRIM = os.getenv("TTS_SILENCE_TRIM", "1").strip().lower() not in ("0", "false", "no", "off")
STT_PRIMARY_LANG = os.getenv("STT_PRIMARY_LANG", "th-TH")

# Passive Multi-Speaker Voice Scribe Guardrails (Option 1: In-RAM only, auto-cleared on finish/leave)
VOICE_SCRIBE_MAX_ENTRIES = int(os.getenv("VOICE_SCRIBE_MAX_ENTRIES", "2000"))
VOICE_SCRIBE_MAX_DURATION_SEC = int(os.getenv("VOICE_SCRIBE_MAX_DURATION_SEC", "10800"))

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
tts_lock = asyncio.Lock()
# Global OOM guard: defaults to Semaphore(1) on 512 MB containers; scales via MANGA_CONCURRENCY on multi-core VPS hosts
manga_semaphore = asyncio.Semaphore(MANGA_CONCURRENCY)
# Caps concurrent multi-speaker STT tasks; scales via VOICE_STT_CONCURRENCY
voice_stt_semaphore = asyncio.Semaphore(VOICE_STT_CONCURRENCY)
BOOT_TIME = time.time()


class BoundedChannelDict(dict):
    """
    LRU-bounded per-channel dictionary capped at `MAX_CACHED_CHANNELS` (default 64)
    so multi-server channel caches never grow unbounded over long uptimes.
    Uses Python 3.7+ insertion-ordered `dict` (zero `OrderedDict.pop` quirks);
    read-only `.get()` avoids mutating order during iteration, while `__getitem__`
    and `__setitem__` refresh recency and evict oldest entries beyond `maxsize`.
    """

    def __init__(self, maxsize: int = MAX_CACHED_CHANNELS):
        super().__init__()
        self._maxsize = max(1, int(maxsize))

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if len(self) > 1:
            super().__delitem__(key)
            super().__setitem__(key, val)
        return val

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            super().__delitem__(next(iter(self)))

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = default
        return default

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class BoundedChannelDefaultDict(BoundedChannelDict):
    """Drop-in LRU-bounded replacement for `defaultdict(factory)` keyed by Discord channel_id."""

    def __init__(self, default_factory, maxsize: int = MAX_CACHED_CHANNELS):
        super().__init__(maxsize=maxsize)
        self._default_factory = default_factory

    def __missing__(self, key):
        if self._default_factory is None:
            raise KeyError(key)
        val = self._default_factory()
        self[key] = val
        return val


class RuntimeState:
    """Centralized mutable runtime state and lightweight performance telemetry shared across all modules."""

    def __init__(self):
        self.bot = None
        self.http_session: aiohttp.ClientSession | None = None
        self.has_voice_recv: bool = False
        self.profiles_restored: bool = False
        self.active_gemini_model: str = "gemini-3.5-flash-lite"
        self.active_gemma_model: str = GEMMA_CHAT_MODEL
        self.active_groq_search_model: str = GROQ_SEARCH_MODEL
        self.active_groq_vision_model: str = GROQ_VISION_FALLBACK_MODEL
        self.gemini_grounding_cooldown_until: float = 0.0
        self.gemini_decision_cooldown_until: float = 0.0
        self.cf_decision_cooldown_until: float = 0.0
        self.gemini_hydrogen_cooldown_until: float = 0.0
        self.gemini_transcribe_live_cooldown_until: float = 0.0
        self.gemini_hydrogen_call_timestamps: deque = deque(maxlen=5)
        self.cf_gpt_oss_cooldown_until: float = 0.0
        self.openrouter_manga_cooldown_until: float = 0.0
        self.paradee_tts_cooldown_until: float = 0.0
        self.paradee_tts_loaded: bool = False
        self.active_tts_engine: str = TTS_PRIMARY_ENGINE

        # Lightweight zero-RAM runtime telemetry counters
        self.decision_calls: int = 0
        self.decision_errors: int = 0
        self.decision_total_sec: float = 0.0
        self.search_filter_kept: int = 0
        self.search_filter_dropped: int = 0
        self.gemini_cache_hits: int = 0
        self.gemini_cached_tokens_total: int = 0
        self.provider_calls: dict[str, int] = {
            "gemini_flash": 0,
            "gemini_hydrogen": 0,
            "gemma_ai_studio": 0,
            "groq": 0,
            "cloudflare": 0,
            "openrouter": 0,
        }

    def record_decision_stat(self, elapsed_sec: float, ok: bool = True) -> None:
        self.decision_calls += 1
        if ok:
            self.decision_total_sec += max(0.0, float(elapsed_sec))
        else:
            self.decision_errors += 1

    def record_provider_call(self, provider_key: str) -> None:
        if provider_key in self.provider_calls:
            self.provider_calls[provider_key] += 1
        else:
            self.provider_calls[provider_key] = 1

    def record_gemini_cache_hit(self, cached_tokens: int) -> None:
        if cached_tokens > 0:
            self.gemini_cache_hits += 1
            self.gemini_cached_tokens_total += int(cached_tokens)

    def record_search_filter_stats(self, kept: int, dropped: int) -> None:
        self.search_filter_kept += max(0, int(kept))
        self.search_filter_dropped += max(0, int(dropped))

    def get_telemetry_summary(self) -> dict:
        ok_dec = max(0, self.decision_calls - self.decision_errors)
        avg_dec_ms = round((self.decision_total_sec / ok_dec) * 1000.0, 1) if ok_dec > 0 else 0.0
        total_sf = self.search_filter_kept + self.search_filter_dropped
        sf_drop_pct = round((self.search_filter_dropped / total_sf) * 100.0, 1) if total_sf > 0 else 0.0
        return {
            "decision_calls": self.decision_calls,
            "decision_errors": self.decision_errors,
            "decision_avg_ms": avg_dec_ms,
            "search_filter_kept": self.search_filter_kept,
            "search_filter_dropped": self.search_filter_dropped,
            "search_filter_drop_pct": sf_drop_pct,
            "gemini_cache_hits": self.gemini_cache_hits,
            "gemini_cached_tokens_total": self.gemini_cached_tokens_total,
            "provider_calls": dict(self.provider_calls),
        }


state = RuntimeState()


def get_http_session() -> aiohttp.ClientSession:
    """
    Returns the shared pooled aiohttp.ClientSession with 5-minute DNS caching and
    keep-alive connection reuse across Google AI Studio, Cloudflare, Groq, FxTwitter, and Pixiv.
    """
    if state.http_session is None or state.http_session.closed:
        connector = aiohttp.TCPConnector(
            limit=40,
            limit_per_host=12,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        state.http_session = aiohttp.ClientSession(connector=connector)
    return state.http_session


async def close_http_session():
    if state.http_session and not state.http_session.closed:
        await state.http_session.close()


# Rolling per-channel conversation memory (LRU-capped at MAX_CACHED_CHANNELS; 12 messages per channel)
channel_memory = BoundedChannelDefaultDict(lambda: deque(maxlen=12), maxsize=MAX_CACHED_CHANNELS)
# Primary topic/subject anchor per channel (LRU-capped at MAX_CACHED_CHANNELS)
channel_active_topic = BoundedChannelDict(maxsize=MAX_CACHED_CHANNELS)
# Multi-file rolling attachment buffer per channel (LRU-capped; up to 4 recent files/logs within 30-min TTL)
channel_last_attachment: BoundedChannelDefaultDict = BoundedChannelDefaultDict(
    lambda: deque(maxlen=4), maxsize=MAX_CACHED_CHANNELS
)
# Most recent live weather report per channel (LRU-capped; 30-min TTL) for follow-up questions
channel_last_weather = BoundedChannelDict(maxsize=MAX_CACHED_CHANNELS)
# False = reply to all speech in voice channel; True = require wake word ("ยูกิโนะ", etc.)
voice_wake_word_enabled = defaultdict(lambda: False)
# Active in-RAM passive Voice Scribe sessions keyed by guild_id (cleared on stop/leave/disconnect)
voice_scribe_sessions: dict[int, dict] = {}

# ==============================================================================
# ZERO-RAM /tmp IMAGE CACHE (5-min TTL, 30 MB cap — speeds up In-Bubble follow-ups)
# ==============================================================================
TEMP_IMG_CACHE_DIR = os.path.join(tempfile.gettempdir(), "yukinon_img_cache")
TEMP_IMG_CACHE_TTL_SEC = 300
TEMP_IMG_CACHE_MAX_BYTES = max(10, int(os.getenv("TEMP_IMG_CACHE_MAX_MB", "30") or "30")) * 1024 * 1024


def _dir_size_bytes(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def prune_temp_image_cache(ttl_sec: int = TEMP_IMG_CACHE_TTL_SEC):
    """Evicts expired (>ttl_sec) or oldest /tmp image cache buckets beyond TEMP_IMG_CACHE_MAX_BYTES."""
    if not os.path.isdir(TEMP_IMG_CACHE_DIR):
        return
    now = time.time()
    buckets: list[tuple[float, int, str]] = []
    try:
        for entry in os.scandir(TEMP_IMG_CACHE_DIR):
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if (now - mtime) >= ttl_sec:
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                buckets.append((mtime, _dir_size_bytes(entry.path), entry.path))
    except OSError:
        return

    total_sz = sum(sz for _, sz, _ in buckets)
    if total_sz > TEMP_IMG_CACHE_MAX_BYTES:
        buckets.sort(key=lambda x: x[0])
        for _, sz, b_path in buckets:
            if total_sz <= TEMP_IMG_CACHE_MAX_BYTES:
                break
            shutil.rmtree(b_path, ignore_errors=True)
            total_sz -= sz


def save_temp_image_cache(
    msg_id: int,
    images_list: list,
    meta_sections: list[str] | None = None,
) -> None:
    """
    Spills downloaded embed/attachment image bytes to `/tmp/yukinon_img_cache/<msg_id>/`
    so clicking 'แปลลงภาพ (In-Bubble)' reuses them instantly with zero RAM footprint.
    """
    if not msg_id or not images_list:
        return
    try:
        os.makedirs(TEMP_IMG_CACHE_DIR, exist_ok=True)
        bucket_dir = os.path.join(TEMP_IMG_CACHE_DIR, str(int(msg_id)))
        tmp_dir = f"{bucket_dir}.tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        manifest_files = []
        for idx, item in enumerate(images_list[:12]):
            raw_b = item[0]
            orig_fname = str(item[1] or f"img_{idx + 1}.jpg")
            if not raw_b:
                continue
            safe_ext = os.path.splitext(orig_fname)[1].lower()
            if safe_ext not in (".jpg", ".jpeg", ".png", ".webp"):
                safe_ext = ".jpg"
            disk_name = f"p{idx + 1:02d}{safe_ext}"
            with open(os.path.join(tmp_dir, disk_name), "wb") as f:
                f.write(raw_b)
            manifest_files.append({"disk": disk_name, "filename": orig_fname})

        if not manifest_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        manifest = {
            "ts": time.time(),
            "files": manifest_files,
            "meta_sections": list(meta_sections or []),
        }
        with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False)

        shutil.rmtree(bucket_dir, ignore_errors=True)
        os.replace(tmp_dir, bucket_dir)
        prune_temp_image_cache()
    except Exception:
        pass


def load_temp_image_cache(
    msg_id: int, ttl_sec: int = TEMP_IMG_CACHE_TTL_SEC
) -> tuple[list[tuple[bytes, str]], list[str]] | None:
    """Loads cached `(pages_input, meta_sections)` for `msg_id` from `/tmp` if fresh."""
    if not msg_id:
        return None
    bucket_dir = os.path.join(TEMP_IMG_CACHE_DIR, str(int(msg_id)))
    meta_path = os.path.join(bucket_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as mf:
            manifest = json.load(mf)
        if (time.time() - float(manifest.get("ts") or 0.0)) >= ttl_sec:
            shutil.rmtree(bucket_dir, ignore_errors=True)
            return None
        pages: list[tuple[bytes, str]] = []
        for entry in manifest.get("files") or []:
            fpath = os.path.join(bucket_dir, entry["disk"])
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    pages.append((f.read(), entry.get("filename") or entry["disk"]))
        if not pages:
            return None
        return pages, list(manifest.get("meta_sections") or [])
    except Exception:
        return None


def prune_stale_attachments(ttl_sec: int = 1800):
    """Evicts cached channel attachments older than ttl_sec (default 30 mins) and stale /tmp image caches."""
    now = time.time()
    empty_cids = []
    for cid, entries in list(channel_last_attachment.items()):
        while entries and (now - entries[0].get("timestamp", 0) >= ttl_sec):
            entries.popleft()
        if not entries:
            empty_cids.append(cid)
    for cid in empty_cids:
        channel_last_attachment.pop(cid, None)
    prune_temp_image_cache()


def get_cached_channel_attachments(channel_id: int | None, ttl_sec: int = 1800, max_total_chars: int = 85000) -> str:
    """Returns combined recent attachments for a channel in chronological order within RAM/token bounds."""
    if channel_id is None:
        return ""
    prune_stale_attachments(ttl_sec)
    entries = list(channel_last_attachment.get(channel_id) or ())
    if not entries:
        return ""

    selected = []
    used_chars = 0
    for item in reversed(entries):
        c = item.get("content") or ""
        if not c:
            continue
        if used_chars + len(c) > max_total_chars and selected:
            break
        selected.append(c[:max_total_chars])
        used_chars += len(c[:max_total_chars])

    selected.reverse()
    return "\n\n".join(selected)


def channel_has_recent_code_attachment(channel_id: int | None, ttl_sec: int = 1800) -> bool:
    if channel_id is None:
        return False
    prune_stale_attachments(ttl_sec)
    return any(item.get("is_code") for item in channel_last_attachment.get(channel_id, ()))


def release_unused_memory() -> tuple[float, float]:
    """
    Forces Python's full generation-2 garbage collector and Linux glibc (`mallopt` + `malloc_trim`)
    to immediately return freed heap memory (such as Pillow manga buffers) back to the OS kernel.
    Returns (before_mb, after_mb).
    """
    import psutil

    proc = psutil.Process(os.getpid())
    before_mb = round(proc.memory_info().rss / (1024 ** 2), 1)

    prune_stale_attachments()
    gc.collect(2)

    try:
        libc = _GLIBC_LIBC or ctypes.CDLL("libc.so.6")
        libc.mallopt(-1, 131072)
        libc.mallopt(-3, 131072)
        libc.malloc_trim(0)
    except Exception:
        pass

    after_mb = round(proc.memory_info().rss / (1024 ** 2), 1)
    return before_mb, after_mb


# --- Shared Constants & Compiled Regex Patterns ---
VOICE_WAKE_WORDS = (
    "ยูกิโนะ", "ยูคิโนะ", "ยูกิน่อน", "ยูคิน่อน", "ยูกิ", "ยูคิ",
    "yukino", "yukinon", "yuki", "บอท", "ไต้ฝุ่น", "typhoon", "bot"
)

CODE_AND_LOG_EXTENSIONS = (
    ".java", ".py", ".js", ".ts", ".tsx", ".jsx", ".vue", ".svelte",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb",
    ".kt", ".swift", ".scala", ".sql", ".pls", ".plb", ".pks", ".pkb",
    ".sh", ".bash", ".ps1", ".json", ".yaml", ".yml", ".xml", ".toml",
    ".ini", ".gradle", ".properties", ".dockerfile", ".html", ".css",
    ".scss", ".log", ".diff", ".patch"
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
IMAGE_SOURCE_INTENT_PATTERN = re.compile(
    r"(ขอวาร์ป|วาร์ป|หาที่มา|ที่มาของรูป|รูปนี้มาจาก|ภาพนี้มาจาก|มาจากเรื่องอะไร|เรื่องอะไร|ตอนที่เท่าไหร่|"
    r"ใครวาด|คนวาด|ศิลปิน|ชื่อภาพ|รุ่นอะไร|ยี่ห้ออะไร|คือรุ่นไหน|คือตัวอะไร|คืออะไร|"
    r"\b(?:source|sauce|pixiv|artist|what\s+anime|what\s+manga|which\s+chapter|what\s+episode|"
    r"what\s+model|what\s+phone|identify\s+this|what\s+is\s+this)\b)",
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
    r"(แปลมังงะ|แปลรูป|แปลภาพ|แปลการ์ตูน|แปลข้อความ|แปลหน้า|แปลอันนี้|แปลพวกนี้|ช่วยแปล|แปลเป็น|"
    r"\btranslate\b.*?\b(?:this|these|all|manga|manhwa|image|images|panel|panels|comic|page|pages|bubble|bubbles)\b|"
    r"\btranslate\s+to\s+)",
    re.IGNORECASE,
)
OWNER_NAME_STRIP_REGEX = re.compile(
    r"(?:คุณ\s*)?(?:กรินทร์|ครินท์|krint)\b", re.IGNORECASE
)
URL_REGEX = re.compile(r"https?://[^\s<>'\"]+")
AGENTIC_SEARCH_TAG_REGEX = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
AGENTIC_IMAGE_TAG_REGEX = re.compile(r"^\s*\[IMAGE:\s*(.+?)\]", re.IGNORECASE)

# Canonical Compound-Intent & Native Tool Routing Regexes (shared by bot.py & chat_pipeline.py)
COMPOUND_CONNECTOR_REGEX = re.compile(
    r"(แล้ว(?:ก็)?|พร้อม(?:กับ|ทั้ง)?|และ(?:ช่วย)?|กับ|แถม|\band\s+(?:also|then)?\b|\bplus\b|,)",
    re.IGNORECASE,
)
_MATH_EXPR_CUE_PAT = (
    r"\d+\s*[\+\-\*/x×^%]\s*\d+|คำนวณ|แปลงค่าเงิน|กี่บาท|กี่เยน|กี่ดอลลาร์|\bcalculate\b|\bconvert\b"
)
_MEMORY_INLINE_CUE_PAT = (
    r"จำไว้ว่า|ช่วยจำว่า|ปักหมุดความจำ|บันทึกความจำ|\bremember\s+that\b"
)
_OT_INLINE_CUE_PAT = (
    r"(?:จด|บันทึก|สรุป|ลบ)\s*(?:เวลา\s*)?ot\b"
)
MATH_EXPR_CUE_REGEX = re.compile(rf"({_MATH_EXPR_CUE_PAT})", re.IGNORECASE)
MEMORY_INLINE_CUE_REGEX = re.compile(rf"({_MEMORY_INLINE_CUE_PAT})", re.IGNORECASE)
OT_INLINE_CUE_REGEX = re.compile(rf"({_OT_INLINE_CUE_PAT})", re.IGNORECASE)
NATIVE_TOOL_CUE_REGEX = re.compile(
    rf"({_MATH_EXPR_CUE_PAT}|{_MEMORY_INLINE_CUE_PAT}|{_OT_INLINE_CUE_PAT}|"
    r"ช่วยจำ|ความจำ|บันทึกไว้|\bot\b|โอที|"
    r"วาดรูป|สร้างภาพ|เจนรูป|draw\s+a|generate\s+an?\s+image|"
    r"อากาศ|ฝนตก|พยากรณ์|\bweather\b|\bforecast\b)",
    re.IGNORECASE,
)

# Enterprise & Full-Stack Engineering Intent Pattern + Stack Trace Signatures
CODE_INTENT_PATTERN = re.compile(
    r"(```|\b(python|java|maven|pom\.xml|spring|springboot|spring\s*boot|hibernate|jpa|jpql|bean|autowired|"
    r"javascript|typescript|react|bootstrap|vue|nextjs|nodejs|node\.js|express|js|ts|jsx|tsx|"
    r"oracle|pl/?sql|sql|stored\s*procedure|package|trigger|explain\s*plan|tablespace|index|ddl|dml|"
    r"rust|golang|c\+\+|html|css|scss|bash|shell|dockerfile|docker|docker-compose|kubernetes|k8s|kubectl|"
    r"helm|ingress|pod|deployment|configmap|secret|microservice|microservices|rabbitmq|kafka|redis|"
    r"nginx|weblogic|tomcat|systemd|regex|json|yaml|yml|xml|api|rest|graphql|jwt|oauth|cors|"
    r"function|component|useeffect|usestate|usememo|usecallback|render|button|state|props|class|def|import|"
    r"async|await|promise|stream|optional|completablefuture|thread|deadlock|memory\s*leak|oomkilled|"
    r"crashloopbackoff|bug|error|exception|traceback|stack\s*trace|stacktrace|caused\s*by|"
    r"nullpointerexception|classnotfoundexception|sqlexception|beancreationexception|"
    r"segfault|refactor|compile|build|script|algorithm|leetcode|git|diff|patch|pip|npm|yarn|gradle)\b|"
    r"ORA-\d{4,5}|TNS-\d{4,5}|HTTP\s*[45]\d{2}|"
    r"at\s+[a-zA-Z0-9_$.]+\([A-Za-z0-9_]+\.java:\d+\)|"
    r"Caused\s+by:\s+[a-zA-Z0-9_.$/-]+|"
    r"โค้ด|เขียนโปรแกรม|แก้บั๊ก|รันไม่ผ่าน|สคริปต์|ฟังก์ชัน|แก้โค้ด|ซ่อนปุ่ม|แสดงปุ่ม|คิวรี|ฐานข้อมูล|คอนเทนเนอร์)",
    re.IGNORECASE,
)

STACK_TRACE_SIGNATURE_REGEX = re.compile(
    r"(Caused\s+by:\s+[a-zA-Z0-9_.$/-]+|"
    r"Exception\s+in\s+thread\s+\"[^\"]+\"|"
    r"at\s+[a-zA-Z0-9_$.]+\([A-Za-z0-9_]+\.(?:java|kt|scala):\d+\)|"
    r"org\.springframework\.[a-zA-Z0-9_.]+\.Exception|"
    r"java\.(?:lang|sql|io|net|util)\.[a-zA-Z0-9_.]+\.Exception|"
    r"ORA-\d{4,5}:|TNS-\d{4,5}:|"
    r"Traceback\s+\(most\s+recent\s+call\s+last\):|"
    r"UnhandledPromiseRejection|TypeError:|ReferenceError:|SyntaxError:|"
    r"CrashLoopBackOff|OOMKilled|ImagePullBackOff|ErrImagePull|"
    r"Back-off\s+restarting\s+failed\s+container)",
    re.IGNORECASE,
)

WEATHER_INTENT_PATTERN = re.compile(
    r"(\b(weather|forecast|rain|raining|temperature|humidity|pm2\.5|storm)\b|"
    r"ฝนตก|ฝนจะตก|จะตกไหม|จะตกมั้ย|ตกไหม|ตกมั้ย|ตกกี่โมง|หยุดตก|พกร่ม|ฟ้าครึ้ม|แดดออก|"
    r"พยากรณ์อากาศ|สภาพอากาศ|อากาศวันนี้|อากาศตอนนี้|อากาศช่วง|อุณหภูมิ|ค่าฝุ่น|ร้อนไหม|หนาวไหม)",
    re.IGNORECASE,
)