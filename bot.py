import os
import re
import math
import time
import signal
import asyncio
import platform
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil
from aiohttp import web
import discord
from discord.ext import commands, tasks
import simpleeval
from simpleeval import SimpleEval

simpleeval.MAX_POWER = 1000

import config
from config import (
    DISCORD_TOKEN,
    PORT,
    BOOT_TIME,
    CONTAINER_RAM_LIMIT_MB,
    OPENCV_THREADS,
    MANGA_CONCURRENCY,
    VOICE_STT_CONCURRENCY,
    MAX_CACHED_CHANNELS,
    ALLOWED_GUILD_IDS,
    AI_CHANNEL_IDS,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    CF_API_TOKEN,
    CF_ACCOUNT_ID,
    CF_IMAGE_MODEL,
    CF_VISION_MODEL,
    CF_VISION_FALLBACK_MODEL,
    CF_DECISION_MODEL,
    CF_CODE_MODEL,
    CF_CHAT_FAST_MODEL,
    GEMINI_API_KEY,
    GEMINI_DECISION_MODEL,
    GEMINI_HEAVY_MODEL,
    GEMINI_TRANSCRIBE_LIVE_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MANGA_MODEL,
    TTS_PRIMARY_ENGINE,
    TTS_PARADEE_SPEED,
    TTS_PARADEE_PITCH,
    TTS_GOOGLE_PITCH,
    TTS_GOOGLE_SPEED,
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    CUSTOM_EMOJI_REGEX,
    YUKIPEDIA_INTENT_PATTERN,
    MANGA_TRANSLATE_INTENT_PATTERN,
    IMAGE_GEN_INTENT_PATTERN,
    AGENTIC_IMAGE_TAG_REGEX,
    WEATHER_INTENT_PATTERN,
    COMPOUND_CONNECTOR_REGEX,
    MATH_EXPR_CUE_REGEX,
    MEMORY_INLINE_CUE_REGEX,
    OT_INLINE_CUE_REGEX,
    URL_REGEX,
    CODE_AND_LOG_EXTENSIONS,
    intents,
    channel_memory,
    channel_last_attachment,
    voice_wake_word_enabled,
    voice_scribe_sessions,
    state,
    davey,
    api_log,
    get_http_session,
    release_unused_memory,
    prune_stale_attachments,
    close_http_session,
)
from persona import (
    is_owner_user,
    get_current_thai_datetime_str,
    try_instant_datetime_reply,
    should_reply_to_message,
    maybe_react_to_discord_message,
    FAST_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    YUKIPEDIA_SYSTEM_PROMPT,
)
from user_memory import (
    restore_user_profiles_from_discord,
    backup_user_profiles_to_discord,
    has_dirty_user_profiles,
    resolve_user_mentions_and_build_dossier,
    is_ot_command_text,
    process_ot_request,
    is_memory_command_text,
    process_memory_request,
    build_monthly_ot_summary_markdown,
    _parse_stored_ot_date,
    user_profiles,
    META_KEY,
    user_profile_count,
    get_last_ot_report_ym,
    set_last_ot_report_ym,
    is_ot_work_channel_name,
)
from search_engine import detect_coding_intent, run_safe_web_search_async
from llm import (
    query_ai_chat,
    orchestrate_message_with_decision_1,
)
from media_context import (
    MAX_BATCH_IMAGES,
    TWITTER_STATUS_URL_REGEX,
    PIXIV_ILLUST_URL_REGEX,
    BSKY_POST_URL_REGEX,
    DIRECT_IMAGE_URL_REGEX,
    extract_all_context,
    resolve_referenced_message,
    get_batch_ocr_encode_params,
    encode_ocr_and_decision_thumb_sync,
)
from image_engine import (
    _send_or_reply,
    send_chunked_response,
    send_generated_image_to_channel,
    process_manga_translation_request,
)
from voice_engine import (
    HAS_VOICE_RECV,
    voice_recv,
    SilenceGatedVoiceSink,
    speak_in_voice_channel,
    is_voice_scribe_active,
    start_voice_scribe_session,
    stop_voice_scribe_session,
    clear_voice_scribe_session,
    register_voice_sink,
    unregister_voice_sink,
    prewarm_paradee_tts_bg,
    unload_paradee_tts,
    has_baked_paradee_model,
)
from ui_interactions import (
    CodeReviewActionView,
    OTActionView,
    MemoryActionView,
    build_contextual_suggestion_view,
    parse_ot_record_metadata,
    get_previous_month_info,
    register_app_commands,
    should_attach_interactive_view,
    is_failure_or_refusal_reply,
)
from tools import ToolExecutionContext, TOOL_SPECS
from discord_tables import format_markdown_tables_for_discord

OT_REPORT_CHANNEL_ID = int(os.getenv("OT_REPORT_CHANNEL_ID", "0") or "0")


class _HeavyStatusTracker:
    """
    Bundle 2 progressive status message manager for heavy `gemini-4-hydrogen` calls:
    - Posts a temporary status message ONLY if Hydrogen is actually invoked.
    - Edits in place on slow-thinking heartbeat updates (throttled to >= 2.0s).
    - Auto-deletes cleanly before the final response is posted (or on exception).
    """

    def __init__(self, channel, reply_to: discord.Message | None = None):
        self.channel = channel
        self.reply_to = reply_to
        self._msg: discord.Message | None = None
        self._last_edit_ts: float = 0.0

    async def update(self, text: str) -> None:
        if not text or not self.channel:
            return
        now = time.monotonic()
        if self._msg is not None and (now - self._last_edit_ts) < 2.0:
            return
        try:
            if self._msg is None:
                self._msg = await _send_or_reply(self.channel, self.reply_to, content=text)
            else:
                await self._msg.edit(content=text)
            self._last_edit_ts = now
        except Exception:
            pass

    async def cleanup(self) -> None:
        msg = self._msg
        self._msg = None
        if msg is None:
            return
        try:
            await msg.delete()
        except Exception:
            pass


def _looks_like_compound_request(text: str) -> bool:
    """
    Detects multi-intent compound prompts (e.g., weather + math + memory, or search + image gen)
    so they bypass single-intent fast-paths and execute all actions in parallel via Native Function Calling!
    """
    if not text or len(text.strip()) < 14:
        return False
    if not COMPOUND_CONNECTOR_REGEX.search(text):
        return False

    hit_domains = 0
    if WEATHER_INTENT_PATTERN.search(text):
        hit_domains += 1
    if MATH_EXPR_CUE_REGEX.search(text):
        hit_domains += 1
    if MEMORY_INLINE_CUE_REGEX.search(text):
        hit_domains += 1
    if OT_INLINE_CUE_REGEX.search(text):
        hit_domains += 1
    if IMAGE_GEN_INTENT_PATTERN.search(text):
        hit_domains += 1
    if re.search(r"(ข่าว|ราคา|ค้นหา|สรุป|เช็คข้อมูล|\bsearch\b|\bnews\b|\bprice\b)", text, re.IGNORECASE):
        hit_domains += 1
    return hit_domains >= 2


MASS_PING_COOLDOWN_SEC = max(0, int(os.getenv("MASS_PING_COOLDOWN_SEC", "600") or "600"))
_MASS_PING_REQUEST_REGEX = re.compile(
    r"(@everyone|@here|<@&\d+>|แท็กทุกคน|เรียกทุกคน|ปิงทุกคน|"
    r"\b(?:ping|tag|mention)\s+(?:everyone|all|here)\b)",
    re.IGNORECASE,
)
_last_mass_ping_at: dict[int, float] = {}


def _resolve_mass_ping(message: discord.Message, user_text: str, answer: str) -> tuple[discord.AllowedMentions | None, str]:
    """
    Opt-in @everyone/@here/role pings for ONE reply when the user explicitly asked for a ping.
    Mirrors Discord's own rules (author needs `mention_everyone`, or the role must be mentionable)
    so members cannot use the bot to ping beyond their own permissions, and enforces a per-guild
    cooldown (`MASS_PING_COOLDOWN_SEC`, default 600s) against spam.
    Returns `(allowed_mentions_override_or_None, answer_with_optional_note)`.
    """
    guild = message.guild
    if guild is None or not answer or not _MASS_PING_REQUEST_REGEX.search(user_text or ""):
        return None, answer
    wants_everyone = bool(re.search(r"@(?:everyone|here)\b", answer))
    role_ids = {int(r) for r in re.findall(r"<@&(\d+)>", answer)}
    # Override is only applied on single-message replies, so multi-chunk answers never multi-ping
    if (not wants_everyone and not role_ids) or len(answer) > 1900:
        return None, answer

    perms = message.channel.permissions_for(message.author)
    allow_everyone = wants_everyone and perms.mention_everyone
    roles = [
        r for r in (guild.get_role(rid) for rid in role_ids)
        if r is not None and (r.mentionable or perms.mention_everyone)
    ]
    if not allow_everyone and not roles:
        return None, f"{answer}\n-# (ผู้สั่งไม่มีสิทธิ์แท็กทุกคน/แท็กยศนี้ในห้องนี้ ครั้งนี้เลยไม่ได้ส่งแจ้งเตือนนะคะ)"

    now = time.monotonic()
    last_ts = _last_mass_ping_at.get(guild.id)
    if last_ts is not None and (now - last_ts) < MASS_PING_COOLDOWN_SEC:
        wait_min = max(1, math.ceil((MASS_PING_COOLDOWN_SEC - (now - last_ts)) / 60))
        return None, f"{answer}\n-# (การแท็กทุกคน/แท็กยศติดคูลดาวน์อีกประมาณ {wait_min} นาที ครั้งนี้เลยไม่ได้ส่งแจ้งเตือนนะคะ)"

    _last_mass_ping_at[guild.id] = now
    return discord.AllowedMentions(everyone=allow_everyone, roles=roles, users=True), answer


def _format_uptime(seconds: int) -> str:
    days, rem = divmod(max(0, int(seconds)), 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m {secs}s"
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


async def _preload_image_attachments_and_thumb(
    all_msg_atts: list[discord.Attachment],
) -> tuple[dict[int, bytes], list[tuple[str, str]], dict[int, tuple[str, str, bool, int, int]]]:
    """
    Preloads up to 4 image attachments and decodes the first image ONCE via
    `encode_ocr_and_decision_thumb_sync` to produce BOTH the 512px PreRoute thumbnail
    and the OCR-ready JPEG reused directly by `extract_all_context` (zero duplicate decode).
    Supports both `gemini-decision-1` AND Cloudflare `@cf/cloudflare/clef-flash` fallback.
    """
    preloaded: dict[int, bytes] = {}
    all_valid_imgs = [
        a for a in all_msg_atts
        if a.filename.lower().endswith(IMAGE_EXTENSIONS) and a.size <= 8 * 1024 * 1024
    ]
    total_img_count = min(MAX_BATCH_IMAGES, len(all_valid_imgs))
    img_atts = all_valid_imgs[:4]
    for att in img_atts:
        try:
            preloaded[att.id] = await att.read()
        except Exception:
            continue

    low_res_imgs: list[tuple[str, str]] = []
    preencoded_ocr: dict[int, tuple[str, str, bool, int, int]] = {}
    now = time.time()
    has_decision_backend = bool(
        (GEMINI_API_KEY and GEMINI_DECISION_MODEL and now >= state.gemini_decision_cooldown_until)
        or (
            CF_ACCOUNT_ID
            and CF_API_TOKEN
            and CF_DECISION_MODEL
            and now >= getattr(state, "cf_decision_cooldown_until", 0.0)
        )
    )
    if preloaded and has_decision_backend:
        try:
            first_att_id = next(iter(preloaded))
            first_bytes = preloaded[first_att_id]
            ocr_dim, ocr_q, _ = get_batch_ocr_encode_params(max(1, total_img_count))
            ocr_b64, ocr_mime, is_col, b64_thumb = await asyncio.to_thread(
                encode_ocr_and_decision_thumb_sync,
                first_bytes,
                ocr_dim,
                ocr_q,
                512,
                72,
            )
            low_res_imgs.append((b64_thumb, "image/jpeg"))
            preencoded_ocr[first_att_id] = (ocr_b64, ocr_mime, is_col, ocr_dim, ocr_q)
        except Exception:
            pass

    return preloaded, low_res_imgs, preencoded_ocr


def _collect_health_metrics() -> dict:
    prune_stale_attachments()
    now = time.time()
    proc = psutil.Process(os.getpid())
    rss_mb = round(proc.memory_info().rss / (1024 ** 2), 1)
    ram_limit_f = float(CONTAINER_RAM_LIMIT_MB)
    ram_pct = round((rss_mb / ram_limit_f) * 100.0, 1)
    uptime_sec = int(now - BOOT_TIME)

    ws_ping_ms = (
        round(bot.latency * 1000, 1)
        if bot.is_ready() and bot.latency == bot.latency
        else None
    )
    gemini_cd = max(0, int(state.gemini_grounding_cooldown_until - now))
    decision_cd = max(0, int(state.gemini_decision_cooldown_until - now))
    cf_decision_cd = max(0, int(getattr(state, "cf_decision_cooldown_until", 0.0) - now))
    openrouter_cd = max(0, int(getattr(state, "openrouter_manga_cooldown_until", 0.0) - now))
    hydrogen_cd = max(0, int(state.gemini_hydrogen_cooldown_until - now))
    live_stt_cd = max(0, int(state.gemini_transcribe_live_cooldown_until - now))
    edge_cd = max(0, int(state.edge_tts_cooldown_until - now))
    paradee_cd = max(0, int(state.paradee_tts_cooldown_until - now))
    cached_files_count = sum(len(q) for q in channel_last_attachment.values())
    active_scribe_count = sum(1 for s in voice_scribe_sessions.values() if s.get("active"))
    telemetry = state.get_telemetry_summary()

    return {
        "status": "ok" if rss_mb < (ram_limit_f * 0.90) else "warning_high_memory",
        "system_time_th": get_current_thai_datetime_str(),
        "uptime_sec": uptime_sec,
        "uptime_human": _format_uptime(uptime_sec),
        "discord": {
            "connected": bot.is_ready(),
            "user": str(bot.user) if bot.user else None,
            "gateway_ping_ms": ws_ping_ms,
            "guilds_count": len(bot.guilds) if bot.is_ready() else 0,
            "active_voice_sessions": len(bot.voice_clients),
            "active_scribe_sessions": active_scribe_count,
            "dave_e2ee_ready": bool(davey is not None and HAS_VOICE_RECV),
        },
        "memory": {
            "bot_ram_mb": rss_mb,
            "ram_limit_mb": CONTAINER_RAM_LIMIT_MB,
            "render_limit_mb": CONTAINER_RAM_LIMIT_MB,
            "ram_usage_pct": ram_pct,
            "active_chat_channels": len(channel_memory),
            "max_cached_channels": MAX_CACHED_CHANNELS,
            "cached_attachments": cached_files_count,
            "stored_user_profiles": user_profile_count(),
        },
        "runtime_limits": {
            "ram_limit_mb": CONTAINER_RAM_LIMIT_MB,
            "opencv_threads": OPENCV_THREADS,
            "manga_concurrency": MANGA_CONCURRENCY,
            "voice_stt_concurrency": VOICE_STT_CONCURRENCY,
            "max_cached_channels": MAX_CACHED_CHANNELS,
        },
        "telemetry": telemetry,
        "ai_backends": {
            "native_function_calling": True,
            "registered_tools": [s["name"] for s in TOOL_SPECS],
            "code_model": GROQ_CODE_MODEL if GROQ_API_KEY else CF_CODE_MODEL,
            "groq_enabled": bool(GROQ_API_KEY),
            "groq_search_model": state.active_groq_search_model if GROQ_API_KEY else None,
            "groq_vision_model": state.active_groq_vision_model if GROQ_API_KEY else None,
            "openrouter_enabled": bool(OPENROUTER_API_KEY),
            "openrouter_manga_model": OPENROUTER_MANGA_MODEL if OPENROUTER_API_KEY else None,
            "openrouter_manga_cooldown_sec": openrouter_cd,
            "cloudflare_enabled": bool(CF_API_TOKEN and CF_ACCOUNT_ID),
            "cloudflare_decision_model": CF_DECISION_MODEL if CF_API_TOKEN else None,
            "cloudflare_decision_cooldown_sec": cf_decision_cd,
            "cloudflare_image_model": CF_IMAGE_MODEL if CF_API_TOKEN else None,
            "cloudflare_gemma_model": CF_VISION_MODEL if CF_API_TOKEN else None,
            "cloudflare_vision_fallback_model": CF_VISION_FALLBACK_MODEL if CF_API_TOKEN else None,
            "cloudflare_gpt_oss_models": [CF_CODE_MODEL, CF_CHAT_FAST_MODEL] if CF_API_TOKEN else [],
            "gemini_enabled": bool(GEMINI_API_KEY),
            "gemini_decision_orchestrator": GEMINI_DECISION_MODEL if GEMINI_API_KEY else None,
            "gemini_decision_cooldown_sec": decision_cd,
            "gemini_heavy_specialist": GEMINI_HEAVY_MODEL if GEMINI_API_KEY else None,
            "gemini_hydrogen_cooldown_sec": hydrogen_cd,
            "gemini_transcribe_live_model": GEMINI_TRANSCRIBE_LIVE_MODEL if GEMINI_API_KEY else None,
            "gemini_transcribe_live_cooldown_sec": live_stt_cd,
            "gemma_active_model": getattr(state, "active_gemma_model", None) if GEMINI_API_KEY else None,
            "gemini_active_model": state.active_gemini_model if GEMINI_API_KEY else None,
            "gemini_grounding_cooldown_sec": gemini_cd,
            "tts_primary_engine": TTS_PRIMARY_ENGINE,
            "tts_active_engine": state.active_tts_engine,
            "tts_paradee_speed": TTS_PARADEE_SPEED,
            "tts_paradee_pitch": TTS_PARADEE_PITCH,
            "tts_paradee_baked": has_baked_paradee_model(),
            "tts_paradee_loaded": state.paradee_tts_loaded,
            "tts_paradee_cooldown_sec": paradee_cd,
            "tts_google_pitch": TTS_GOOGLE_PITCH,
            "tts_google_speed": TTS_GOOGLE_SPEED,
            "edge_tts_cooldown_sec": edge_cd,
        },
    }


async def handle_root(request: web.Request) -> web.Response:
    m = _collect_health_metrics()
    d = m["discord"]
    mem = m["memory"]
    rl = m["runtime_limits"]
    ai = m["ai_backends"]
    tel = m["telemetry"]
    pc = tel["provider_calls"]

    ping_str = f"{d['gateway_ping_ms']} ms" if d["gateway_ping_ms"] is not None else "N/A"
    gw_status = (
        f"🟢 Connected as <b>{d['user']}</b> ({d['guilds_count']} guilds | Ping: {ping_str})"
        if d["connected"]
        else "🟡 Connecting to Discord Gateway..."
    )
    gemini_cd_note = f" (Grounding Cooldown: {ai['gemini_grounding_cooldown_sec']}s)" if ai["gemini_grounding_cooldown_sec"] else ""
    if ai["tts_primary_engine"] == "paradee":
        tts_note = (
            f" | TTS: MakHamTTS Paradee INT8 (pitch={ai['tts_paradee_pitch']}, speed={ai['tts_paradee_speed']}x, "
            f"baked={ai['tts_paradee_baked']}, loaded={ai['tts_paradee_loaded']})"
        )
    elif ai["tts_primary_engine"] == "edge":
        tts_note = " | TTS: Edge-TTS Neural"
    else:
        tts_note = f" | TTS: Google Yukino (pitch={ai['tts_google_pitch']}, speed={ai['tts_google_speed']}x)"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Yukinon Discord Bot - System Telemetry</title>
    </head>
    <body style="font-family: system-ui, sans-serif; max-width: 740px; margin: 36px auto; line-height: 1.6; color: #1f2937; padding: 0 16px;">
        <h2>❄️ Yukino Yukinoshita Discord Bot (Decision-1 Orchestrated, Native Tools &amp; Voice Scribe Edition)</h2>
        <p><b>System Time (TH):</b> {m['system_time_th']}</p>
        <p><b>Discord Gateway:</b> {gw_status}</p>
        <p><b>Native Tools:</b> {len(ai['registered_tools'])} active (<code>{', '.join(ai['registered_tools'])}</code>)</p>
        <p><b>Voice &amp; Scribe:</b> {d['active_voice_sessions']} voice session(s), {d['active_scribe_sessions']} active scribe(s) | DAVE Ready: {d['dave_e2ee_ready']} | Live STT: <code>{ai['gemini_transcribe_live_model']}</code>{tts_note}</p>
        <p><b>Container RAM:</b> {mem['bot_ram_mb']} MB / {mem['ram_limit_mb']} MB ({mem['ram_usage_pct']}%) |
           <b>Cached Contexts:</b> {mem['active_chat_channels']}/{mem['max_cached_channels']} chat(s), {mem['cached_attachments']} file(s), {mem['stored_user_profiles']} profile(s)</p>
        <p><b>Runtime Limits:</b> OpenCV <code>{rl['opencv_threads']}</code> thread(s) | Manga Concurrency <code>{rl['manga_concurrency']}</code> | Voice STT Concurrency <code>{rl['voice_stt_concurrency']}</code> | Channel LRU Cap <code>{rl['max_cached_channels']}</code></p>
        <p><b>Runtime Telemetry:</b> Decision-1 <code>{tel['decision_calls']}</code> calls (avg <code>{tel['decision_avg_ms']}ms</code>, errs <code>{tel['decision_errors']}</code>) |
           SearchFilter dropped <code>{tel['search_filter_dropped']}</code> (<code>{tel['search_filter_drop_pct']}%</code>) |
           Gemini Implicit Cache: <code>{tel['gemini_cache_hits']}</code> hit(s), <code>{tel['gemini_cached_tokens_total']:,}</code> tokens</p>
        <p><b>Provider Calls:</b> Flash <code>{pc.get('gemini_flash', 0)}</code> | Hydrogen <code>{pc.get('gemini_hydrogen', 0)}</code> | Gemma <code>{pc.get('gemma_ai_studio', 0)}</code> | Groq <code>{pc.get('groq', 0)}</code> | CF <code>{pc.get('cloudflare', 0)}</code> | OR <code>{pc.get('openrouter', 0)}</code></p>
        <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;">
        <p><b>Google AI Studio:</b> {"Enabled (Orchestrator: <code>" + str(ai['gemini_decision_orchestrator']) + "</code> | Heavy Specialist: <code>" + str(ai['gemini_heavy_specialist']) + "</code> | Live STT: <code>" + str(ai['gemini_transcribe_live_model']) + "</code> | Gemma 4: <code>" + str(ai['gemma_active_model']) + "</code> | Flash-Lite: <code>" + str(ai['gemini_active_model']) + "</code>)" + gemini_cd_note if ai['gemini_enabled'] else "Disabled"}</p>
        <p><b>Cloudflare Workers AI:</b> {"Enabled (Decision Fallback: <code>" + str(ai['cloudflare_decision_model']) + "</code> | Code: <code>" + CF_CODE_MODEL + "</code> | Fast: <code>" + CF_CHAT_FAST_MODEL + "</code> | Vision/Gemma: <code>" + str(ai['cloudflare_gemma_model']) + "</code> | Vision Fallback: <code>" + str(ai['cloudflare_vision_fallback_model']) + "</code> | Image: <code>" + str(ai['cloudflare_image_model']) + "</code>)" if ai['cloudflare_enabled'] else "Disabled"}</p>
        <p><b>Groq LPU Acceleration:</b> {"Enabled (Code: <code>" + GROQ_CODE_MODEL + "</code> | Search: <code>" + str(ai['groq_search_model']) + "</code> | Vision: <code>" + str(ai['groq_vision_model']) + "</code>)" if ai['groq_enabled'] else "Disabled"}</p>
        <p><b>OpenRouter Fallback:</b> {"Enabled (MangaBatch: <code>" + str(ai['openrouter_manga_model']) + "</code>)" if ai['openrouter_enabled'] else "Disabled"}</p>
        <p><b>Uptime:</b> {m['uptime_human']} ({m['uptime_sec']}s)</p>
    </body>
    </html>
    """
    return web.Response(
        text=html,
        content_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


async def handle_healthz(request: web.Request) -> web.Response:
    metrics = _collect_health_metrics()
    return web.json_response(
        metrics,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _resolve_ot_report_channel(prof: dict):
    """Scopes OT monthly reports to OT_REPORT_CHANNEL_ID -> user's saved ot_channel_id -> #work/#ot fallback."""
    if OT_REPORT_CHANNEL_ID:
        ch = bot.get_channel(OT_REPORT_CHANNEL_ID)
        if ch:
            return ch
    saved_cid = prof.get("ot_channel_id")
    if saved_cid:
        ch = bot.get_channel(int(saved_cid))
        if ch:
            return ch
    for guild in bot.guilds:
        for ch in guild.text_channels:
            if is_ot_work_channel_name(ch.name):
                return ch
    return None


@tasks.loop(minutes=15)
async def monthly_ot_auto_report_task():
    if not bot.is_ready():
        return
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    prev_year, prev_month, prev_month_label = get_previous_month_info(now_th)
    ym_key = f"{prev_year:04d}-{prev_month:02d}"

    if now_th.day != 1 or now_th.hour < 9 or get_last_ot_report_ym() == ym_key:
        return

    # At-most-once: persisted before sending so a crash mid-send never causes a duplicate report on restart.
    await set_last_ot_report_ym(ym_key)

    for uid_str, prof in list(user_profiles.items()):
        if uid_str == META_KEY:
            continue
        ot_logs = prof.get("ot_logs") or []
        has_prev_month_data = any(
            (pk := _parse_stored_ot_date(e.get("date", ""))) is not None
            and pk[0] == prev_year
            and pk[1] == prev_month
            for e in ot_logs
        )
        if has_prev_month_data:
            target_channel = _resolve_ot_report_channel(prof)
            if not target_channel:
                continue
            summary_md = build_monthly_ot_summary_markdown(ot_logs, prev_year, prev_month)
            report_text = f"📅 **[สรุปยอด OT ประจำเดือน{prev_month_label} — รายงานอัตโนมัติวันที่ 1 ของเดือนใหม่]**\n{summary_md}"
            if len(report_text) <= 7500:
                report_text = format_markdown_tables_for_discord(report_text)
            await send_chunked_response(target_channel, report_text)


class YukinonBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_runner = None
        self._shutdown_flushed = False

    async def _flush_state_before_shutdown(self, reason: str = "Shutdown") -> None:
        """
        Gracefully finalizes any active in-RAM Voice Scribe sessions and force-flushes
        dirty `user_profiles.json` to `#yukinon-memory` before Render stops the container.
        """
        if self._shutdown_flushed:
            return
        self._shutdown_flushed = True
        api_log(f"Graceful Shutdown ({reason}) -> flushing active Scribe sessions & dirty user profiles...")

        # 1. Finalize active Voice Scribe sessions (bounded to 10s so SIGTERM completes cleanly)
        for gid in list(voice_scribe_sessions.keys()):
            if is_voice_scribe_active(gid):
                guild = self.get_guild(gid)
                if guild is not None:
                    try:
                        await asyncio.wait_for(
                            stop_voice_scribe_session(
                                guild,
                                triggered_by=reason,
                                speak_ack=False,
                                reason=f"รีสตาร์ทระบบอัตโนมัติ ({reason})",
                            ),
                            timeout=10.0,
                        )
                    except Exception as e:
                        api_log(f"Graceful Shutdown Scribe note (guild={gid}): {type(e).__name__}")
                        clear_voice_scribe_session(gid)
                else:
                    clear_voice_scribe_session(gid)

        # 2. Force-flush any dirty user_profiles.json edits to #yukinon-memory
        if has_dirty_user_profiles() and self.is_ready():
            try:
                await asyncio.wait_for(backup_user_profiles_to_discord(force=True), timeout=6.0)
                api_log("Graceful Shutdown -> flushed dirty user_profiles.json to Discord cloud backup.")
            except Exception as e:
                api_log(f"Graceful Shutdown profile flush note: {type(e).__name__}")

    async def setup_hook(self):
        register_app_commands(self)
        get_http_session()
        app = web.Application()
        app.router.add_get("/", handle_root)
        app.router.add_get("/healthz", handle_healthz)
        self.web_runner = web.AppRunner(app, access_log=None)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, "0.0.0.0", PORT)
        await site.start()
        print(f"[Render Web Service Health Server] Listening on 0.0.0.0:{PORT}", flush=True)

        # Register POSIX SIGTERM handler for clean Render redeploy / restart flushes
        try:
            loop = asyncio.get_running_loop()

            def _on_sigterm():
                api_log("Received SIGTERM from host — initiating graceful flush & shutdown...")
                # Keep a strong reference so the shutdown task cannot be garbage-collected mid-flush
                self._sigterm_close_task = loop.create_task(self.close())

            loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    async def close(self):
        try:
            await self._flush_state_before_shutdown(reason="SIGTERM/Close")
        except Exception:
            pass
        for gid in list(voice_scribe_sessions.keys()):
            clear_voice_scribe_session(gid)
        unload_paradee_tts(reason="shutdown")
        if monthly_ot_auto_report_task.is_running():
            monthly_ot_auto_report_task.cancel()
        await close_http_session()
        if self.web_runner:
            await self.web_runner.cleanup()
        await super().close()


# Bot-wide safe default: LLM output / echoed user text can never ping @everyone, @here, or roles.
# Explicit user-requested mass pings are opted in per message by `_resolve_mass_ping()` (permission + cooldown).
bot = YukinonBot(
    command_prefix="!",
    intents=intents,
    help_command=None,
    allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True),
)
state.bot = bot


@bot.check
async def verify_allowed_guild(ctx):
    if not ALLOWED_GUILD_IDS or ctx.guild is None or ctx.guild.id in ALLOWED_GUILD_IDS:
        return True
    await ctx.send("เซิร์ฟเวอร์นี้ยังไม่ได้เปิดสิทธิ์ให้ฉันใช้งานนะคะ")
    return False


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, (commands.CommandNotFound, commands.CheckFailure)):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"ยังพิมพ์ข้อมูลไม่ครบนะคะ (ขาด `{error.param.name}`)")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"รูปแบบข้อมูลไม่ถูกต้องนะคะ (`{error}`)")
        return
    orig = getattr(error, "original", error)
    cmd_name = getattr(getattr(ctx, "command", None), "name", "unknown")
    api_log(f"PrefixCommand Error (!{cmd_name}) -> {type(orig).__name__}: {str(orig)[:180]}")
    try:
        await ctx.send(f"คำสั่ง `!{cmd_name}` ติดขัดนิดหน่อยค่ะ (`{type(orig).__name__}`) ลองใหม่อีกทีนะคะ")
    except Exception:
        pass


@bot.event
async def on_ready():
    # Guard against Discord gateway reconnects so unsaved in-RAM profile updates are never overwritten!
    if not state.profiles_restored:
        await restore_user_profiles_from_discord(bot)
        state.profiles_restored = True
    else:
        api_log("Discord Gateway reconnected (on_ready) -> preserved in-RAM user profiles.")
    if not monthly_ot_auto_report_task.is_running():
        monthly_ot_auto_report_task.start()
    print(f"Logged in as {bot.user} on {platform.machine()} at {get_current_thai_datetime_str()}", flush=True)


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    """
    1) Registers mid-call joiners into DAVE `_KNOWN_VOICE_USER_IDS` for multi-speaker decryption.
    2) Auto-finalizes & clears in-RAM Voice Scribe sessions if the bot is disconnected OR all humans leave.
    """
    guild = member.guild
    if not guild:
        return
    vc = guild.voice_client

    # Case A: Bot itself was disconnected/kicked from a voice channel
    if bot.user and member.id == bot.user.id:
        if before.channel is not None and after.channel is None:
            if is_voice_scribe_active(guild.id):
                await stop_voice_scribe_session(
                    guild,
                    triggered_by="Disconnect",
                    speak_ack=False,
                    reason="ตัดการเชื่อมต่อจากห้องเสียง",
                )
            else:
                clear_voice_scribe_session(guild.id)
            unregister_voice_sink(guild.id)
        return

    if not vc or not vc.is_connected() or not getattr(vc, "channel", None):
        return

    # Case B: Another human joins the bot's active voice channel -> register for DAVE multi-speaker E2EE
    if not member.bot and after.channel is not None and after.channel.id == vc.channel.id:
        from voice_engine import _KNOWN_VOICE_USER_IDS
        _KNOWN_VOICE_USER_IDS.add(int(member.id))
        return

    # Case C: A human left the bot's active voice channel -> check if room is now empty
    if not member.bot and before.channel is not None and before.channel.id == vc.channel.id:
        if after.channel is None or after.channel.id != vc.channel.id:
            remaining_humans = [m for m in vc.channel.members if not m.bot]
            if not remaining_humans:
                if is_voice_scribe_active(guild.id):
                    await stop_voice_scribe_session(
                        guild,
                        triggered_by="Auto-Leave",
                        speak_ack=False,
                        reason="สมาชิกออกจากห้องเสียงครบทุกคนแล้ว",
                    )
                else:
                    clear_voice_scribe_session(guild.id)
                if hasattr(vc, "stop_listening"):
                    try:
                        vc.stop_listening()
                    except Exception:
                        pass
                unregister_voice_sink(guild.id)
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass


async def _send_response_with_interactive_view(
    channel: discord.abc.Messageable,
    answer: str,
    view: discord.ui.View | None,
    reply_to: discord.Message | None = None,
    allowed_mentions: discord.AllowedMentions | None = None,
):
    if not answer or not answer.strip():
        return
    display_answer = format_markdown_tables_for_discord(answer) if len(answer) <= 7500 else answer
    if len(display_answer) <= 1900:
        # `allowed_mentions=None` keeps the bot-wide safe default (no @everyone/@here/role pings)
        sent_msg = None
        if reply_to is not None:
            try:
                sent_msg = await reply_to.reply(
                    display_answer, view=view, mention_author=False, allowed_mentions=allowed_mentions
                )
            except Exception:
                sent_msg = None
        if sent_msg is None:
            sent_msg = await channel.send(display_answer, view=view, allowed_mentions=allowed_mentions)
        if view is not None and hasattr(view, "message"):
            view.message = sent_msg
            view.delete_message_on_timeout = False
        return

    # If formatting expanded an inline message past 7500, let send_chunked_response attach raw Markdown (.md)
    chunk_payload = display_answer if len(display_answer) <= 7500 else answer
    await send_chunked_response(channel, chunk_payload, reply_to=reply_to)
    if view is not None:
        footer_msg = await channel.send("-# ตัวเลือกเพิ่มเติม:", view=view)
        if hasattr(view, "message"):
            view.message = footer_msg
            view.delete_message_on_timeout = True


async def _dispatch_post_tool_side_effects(
    channel,
    message: discord.Message,
    answer: str,
    user_request: str,
    tool_ctx: ToolExecutionContext,
):
    """Executes queued reactions and FLUX image generations emitted by Native Function Calling."""
    await maybe_react_to_discord_message(
        message,
        raw_llm_output=answer,
        user_text=user_request,
        queued_reactions=tool_ctx.reactions,
    )
    for img_prompt in tool_ctx.generated_images[:2]:
        await send_generated_image_to_channel(channel, img_prompt, reply_to=message)


async def process_yukipedia_request(ctx_or_message, raw_prompt: str = ""):
    """
    Unified 'Yukipedia Mode' dispatcher:
    1) If image/link + manga translation requested -> runs Manga Localization.
    2) If image/document/reply attached -> runs OCR/Document extraction + Yukipedia analysis.
    3) Otherwise -> runs Web Search & Decision-1 in parallel for faster Yukipedia answers.
    """
    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel
    author = msg.author

    clean_q = re.sub(
        r"^(?:เปิดโหมด|โหมด|ถาม|ค้นหา|ช่วยค้น|เรียก)?\s*(?:yukipedia|ยูกิพีเดีย|ยูคิพีเดีย)\s*[:,-]?\s*",
        "",
        raw_prompt.strip(),
        flags=re.IGNORECASE,
    ).strip()

    ref_msg = await resolve_referenced_message(msg) if msg.reference else None
    all_atts = list(msg.attachments) + (list(ref_msg.attachments) if ref_msg else [])
    has_images = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in all_atts)

    if has_images and MANGA_TRANSLATE_INTENT_PATTERN.search(clean_q):
        await process_manga_translation_request(msg, clean_q)
        return

    async with channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(msg)
        effective_query = clean_q or spoken_clip_text

        if not extracted_ctx and ref_msg and ref_msg.content and ref_msg.content.strip():
            extracted_ctx = f"=== ข้อความอ้างอิงที่ต้องการให้ Yukipedia ตรวจสอบ/แปล ===\n{ref_msg.content.strip()}"

        if extracted_ctx and not effective_query:
            effective_query = (
                "อ่านข้อความ แปล และวิเคราะห์รายละเอียดทั้งหมดในฐานะ Yukipedia "
                "พร้อมช่วยดูคำผิดหรือจุดบกพร่องในต้นฉบับให้ด้วย"
            )
        elif not effective_query and not extracted_ctx:
            await _send_or_reply(
                channel,
                msg,
                content=(
                    "หากจะพึ่งพาคลังความรู้ **Yukipedia** ของฉัน ก็บอกหัวข้อที่จะให้ค้นหา "
                    "หรือแนบรูป/เอกสารที่อยากให้ช่วยอ่านและแปลเข้ามาด้วยสิคะ "
                    "(เช่น `!yukipedia สรุปข่าว...` หรือแนบรูปพร้อมพิมพ์ `!yukipedia แปลและตรวจคำผิด` ค่ะ)"
                ),
            )
            return

        # Run Web Search and Decision-1 concurrently when no attachment context is present
        if effective_query and not extracted_ctx:
            search_block, dec_plan = await asyncio.gather(
                run_safe_web_search_async(effective_query[:180], max_results=5),
                orchestrate_message_with_decision_1(
                    effective_query,
                    extracted_ctx="",
                    purpose="Yukipedia",
                    channel_id=channel.id,
                ),
            )
        else:
            search_block = ""
            dec_plan = await orchestrate_message_with_decision_1(
                effective_query,
                extracted_ctx=extracted_ctx,
                purpose="Yukipedia",
                channel_id=channel.id,
            )

        composed_prompt = effective_query
        if extracted_ctx:
            composed_prompt = f"{composed_prompt}\n\n{extracted_ctx}"
        if search_block:
            composed_prompt = f"{composed_prompt}\n\n{search_block}"

        tool_ctx = ToolExecutionContext(
            channel_id=channel.id,
            guild=msg.guild,
            channel=channel,
            speaker_user=author,
            message=msg,
            mode="general",
        )
        heavy_tracker = _HeavyStatusTracker(channel, reply_to=msg)
        try:
            answer, did_search = await query_ai_chat(
                composed_prompt,
                system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
                mode="general",
                channel_id=channel.id,
                temperature=0.3,
                allow_agentic_search=True,
                has_fresh_attachment=bool(extracted_ctx or search_block),
                speaker_user=author,
                decision_plan=dec_plan,
                tool_ctx=tool_ctx,
                on_heavy_status=heavy_tracker.update,
            )
        finally:
            await heavy_tracker.cleanup()

        view = (
            build_contextual_suggestion_view(
                channel.id,
                effective_query,
                answer,
                extracted_ctx=extracted_ctx,
                source_message=msg,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                effective_query,
                answer,
                did_search=bool(did_search or search_block),
                extracted_ctx=extracted_ctx,
                decision_plan=dec_plan,
            )
            else None
        )
        await _send_response_with_interactive_view(channel, answer, view, reply_to=msg)
        await _dispatch_post_tool_side_effects(channel, msg, answer, effective_query, tool_ctx)


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
    is_ot_trigger = is_ot_command_text(
        message.content,
        channel_id=message.channel.id,
        user_id=message.author.id,
    )
    is_mem_trigger = is_memory_command_text(message.content)

    ref_msg = await resolve_referenced_message(message) if message.reference else None
    is_reply = bool(ref_msg and bot.user and ref_msg.author.id == bot.user.id)
    has_audio = any(a.filename.lower().endswith(AUDIO_EXTENSIONS) for a in message.attachments)
    if not (is_dm or is_ai_chan or is_mentioned or is_reply or has_audio or is_ot_trigger or is_mem_trigger):
        return

    clean_prompt = re.sub(rf"<@!?{bot.user.id}>", "", message.content).strip()
    clean_prompt = CUSTOM_EMOJI_REGEX.sub(r":\1:", clean_prompt).strip()

    if not clean_prompt and ref_msg and ref_msg.content and (not bot.user or ref_msg.author.id != bot.user.id):
        inherited = re.sub(rf"<@!?{bot.user.id}>", "", ref_msg.content).strip()
        inherited = CUSTOM_EMOJI_REGEX.sub(r":\1:", inherited).strip()
        if re.match(r"^!(?:manga|translate|tl)\b", inherited, re.IGNORECASE):
            lang_arg = re.sub(r"^!(?:manga|translate|tl)\s*", "", inherited, flags=re.IGNORECASE).strip()
            clean_prompt = f"translate manga {lang_arg}".strip()
        elif re.match(r"^!(?:source|sauce|whatis|วาร์ป)\b", inherited, re.IGNORECASE):
            clean_prompt = (
                re.sub(r"^!(?:source|sauce|whatis|วาร์ป)\s*", "", inherited, flags=re.IGNORECASE).strip()
                or "ช่วยระบุที่มาของภาพนี้อย่างละเอียด"
            )
        else:
            clean_prompt = inherited

    is_compound_prompt = _looks_like_compound_request(clean_prompt)

    # 0A. Pure Single-Intent Persistent User Memory Viewer & Editor (Zero API cost)
    if is_memory_command_text(clean_prompt) and not is_compound_prompt:
        mem_reply, target_uid = await process_memory_request(message, clean_prompt)
        mem_view = MemoryActionView(requester_id=message.author.id, target_uid=target_uid)
        await _send_response_with_interactive_view(message.channel, mem_reply, mem_view, reply_to=message)
        return

    clean_prompt, mentioned_dossier = resolve_user_mentions_and_build_dossier(
        clean_prompt,
        message.guild,
        bot.user.id if bot.user else 0,
        speaker_user=message.author,
    )

    all_msg_atts = list(message.attachments) + (list(ref_msg.attachments) if ref_msg else [])
    has_image_att = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in all_msg_atts)
    has_code_att = any(a.filename.lower().endswith(CODE_AND_LOG_EXTENSIONS) for a in all_msg_atts)
    has_doc_att = any(a.filename.lower().endswith((".pdf", ".docx", ".xlsx")) for a in all_msg_atts)
    has_embedded_media_or_link = (
        bool(message.embeds)
        or bool(ref_msg and ref_msg.embeds)
        or bool(URL_REGEX.search(clean_prompt))
        or bool(ref_msg and URL_REGEX.search(ref_msg.content or ""))
    )

    # 0B. Fast Instant Date/Time Check (Zero API cost)
    if not message.attachments and not message.embeds and not is_compound_prompt:
        instant_dt = try_instant_datetime_reply(clean_prompt)
        if instant_dt:
            channel_memory[message.channel.id].append({"role": "user", "content": clean_prompt})
            channel_memory[message.channel.id].append({"role": "assistant", "content": instant_dt})
            await message.channel.send(instant_dt)
            return

    # 0C. Deterministic Single-Intent Fast-Path Triggers (skipped on compound prompts so Native Tools handle all intents!)
    if not is_compound_prompt:
        if is_ot_command_text(clean_prompt, channel_id=message.channel.id, user_id=message.author.id):
            ot_reply = await process_ot_request(message, clean_prompt)
            rec_date, rec_slot, rec_detail = parse_ot_record_metadata(ot_reply)
            ot_view = (
                OTActionView(message.author.id, rec_date, rec_slot, rec_detail)
                if rec_date and not is_failure_or_refusal_reply(ot_reply)
                else None
            )
            await _send_response_with_interactive_view(message.channel, ot_reply, ot_view, reply_to=message)
            return

        if YUKIPEDIA_INTENT_PATTERN.search(clean_prompt):
            await process_yukipedia_request(message, clean_prompt)
            return

        combined_link_text = f"{clean_prompt} {getattr(ref_msg, 'content', '') or ''}"
        has_visual_social_or_img_url = bool(
            TWITTER_STATUS_URL_REGEX.search(combined_link_text)
            or PIXIV_ILLUST_URL_REGEX.search(combined_link_text)
            or BSKY_POST_URL_REGEX.search(combined_link_text)
            or DIRECT_IMAGE_URL_REGEX.search(combined_link_text)
        )
        has_visual_media_for_manga = bool(
            has_image_att
            or has_visual_social_or_img_url
            or any(
                getattr(e, "image", None) and getattr(e.image, "url", None)
                for e in (list(message.embeds) + (list(ref_msg.embeds) if ref_msg else []))
            )
        )
        has_explicit_manga_phrase = bool(
            re.search(
                r"(แปลมังงะ|แปลรูป|แปลภาพ|แปลการ์ตูน|แปลหน้า|\btranslate\b.*?\b(?:manga|manhwa|comic|panel|panels|bubble|bubbles)\b)",
                clean_prompt,
                re.IGNORECASE,
            )
        )
        if has_explicit_manga_phrase or (
            has_visual_media_for_manga
            and (
                MANGA_TRANSLATE_INTENT_PATTERN.search(clean_prompt)
                or re.search(r"(แปล|\btranslate\b|\btl\b)", clean_prompt, re.IGNORECASE)
            )
        ):
            await process_manga_translation_request(message, clean_prompt)
            return

        if not message.attachments and IMAGE_GEN_INTENT_PATTERN.search(clean_prompt):
            async with message.channel.typing():
                await send_generated_image_to_channel(message.channel, clean_prompt, reply_to=message)
            return

    # 1. Single-Pass vs PreRoute Decision-1 Orchestration (context-aware with last 2 turns of channel_memory!)
    # Bundle 2: When an image is uploaded, PreRoute also evaluates ImageTriage in the SAME pass AND
    # decodes the first image ONCE to produce both the 512px PreRoute thumb & OCR JPEG!
    text_without_urls = URL_REGEX.sub("", clean_prompt).strip()
    is_bare_link_only = bool(has_embedded_media_or_link and not text_without_urls and not all_msg_atts)
    has_ambiguous_action_cue = bool(
        re.search(r"(แปล|วาด|สร้างภาพ|รูป|\bot\b|โอที|translate|draw)", clean_prompt, re.IGNORECASE)
    )
    skip_preroute = (
        is_bare_link_only
        or is_compound_prompt
        or has_code_att
        or has_doc_att
        or (has_embedded_media_or_link and not has_ambiguous_action_cue)
    )

    preloaded_att_bytes, low_res_imgs, preencoded_ocr = (
        await _preload_image_attachments_and_thumb(all_msg_atts)
        if (has_image_att and not skip_preroute)
        else ({}, [], {})
    )
    pre_plan = (
        None
        if skip_preroute
        else await orchestrate_message_with_decision_1(
            clean_prompt,
            has_code_attachment=has_code_att,
            inline_images_b64=low_res_imgs,
            purpose="PreRoute",
            channel_id=message.channel.id,
        )
    )
    pre_route = (pre_plan or {}).get("route", "")
    pre_conf = float((pre_plan or {}).get("route_confidence") or 0.0)
    pre_image_triage = (pre_plan or {}).get("image_triage")

    if pre_route == "ot_log" and pre_conf >= 0.82:
        preloaded_att_bytes.clear()
        preencoded_ocr.clear()
        ot_reply = await process_ot_request(message, clean_prompt)
        rec_date, rec_slot, rec_detail = parse_ot_record_metadata(ot_reply)
        ot_view = (
            OTActionView(message.author.id, rec_date, rec_slot, rec_detail)
            if rec_date and not is_failure_or_refusal_reply(ot_reply)
            else None
        )
        await _send_response_with_interactive_view(message.channel, ot_reply, ot_view, reply_to=message)
        return

    if (has_image_att or has_embedded_media_or_link) and pre_route == "manga_translate" and pre_conf >= 0.78:
        preloaded_att_bytes.clear()
        preencoded_ocr.clear()
        await process_manga_translation_request(message, clean_prompt)
        return

    if not message.attachments and pre_route == "image_gen" and pre_conf >= 0.82:
        preloaded_att_bytes.clear()
        preencoded_ocr.clear()
        async with message.channel.typing():
            await send_generated_image_to_channel(message.channel, clean_prompt, reply_to=message)
        return

    async with message.channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(
            message,
            preloaded_attachments=preloaded_att_bytes,
            pre_triage=pre_image_triage,
            preencoded_ocr=preencoded_ocr,
        )
        preloaded_att_bytes.clear()
        preencoded_ocr.clear()

        if is_compound_prompt:
            # Still run Decision-1 once for complexity, tone, guardrails, and passive memory (forcing compound route)
            decision_plan = await orchestrate_message_with_decision_1(
                clean_prompt or spoken_clip_text,
                extracted_ctx=extracted_ctx,
                has_code_attachment=has_code_att,
                purpose="CompoundRoute",
                channel_id=message.channel.id,
            ) or {}
            decision_plan["route"] = "compound"
            decision_plan["route_confidence"] = 0.95
            decision_plan["needs_web_search"] = False
            decision_plan["needs_tool_use"] = True
            decision_plan["tool_confidence"] = 0.99
        else:
            needs_refined_route = (
                pre_plan is None
                or pre_conf < 0.85
                or bool(spoken_clip_text and not clean_prompt)
            )
            if needs_refined_route and (extracted_ctx or spoken_clip_text or skip_preroute):
                decision_plan = await orchestrate_message_with_decision_1(
                    clean_prompt or spoken_clip_text,
                    extracted_ctx=extracted_ctx,
                    has_code_attachment=has_code_att,
                    purpose="Route",
                    channel_id=message.channel.id,
                ) or pre_plan
            else:
                decision_plan = pre_plan

        regex_coder = detect_coding_intent(
            clean_prompt,
            message.attachments,
            message.channel.id,
            extracted_ctx=extracted_ctx,
        )
        if decision_plan and decision_plan.get("route") == "code" and decision_plan.get("route_confidence", 0.0) >= 0.72:
            use_coder = True
        elif decision_plan and decision_plan.get("route") == "compound":
            use_coder = False
        else:
            use_coder = regex_coder

        if clean_prompt and not text_without_urls and extracted_ctx:
            if isinstance(decision_plan, dict):
                decision_plan = dict(decision_plan)
                decision_plan["no_tables"] = True
                decision_plan["suppress_social_embeds"] = True
            else:
                decision_plan = {"no_tables": True, "suppress_social_embeds": True}
            user_request = (
                f"{clean_prompt}\n"
                "(ผู้ใช้ส่งลิงก์นี้มาให้ เล่าเหตุการณ์และบริบทของโพสต์โดยเชื่อมโยงแคปชัน ชื่อผู้วาด และบทสนทนาในภาพเป็นย่อหน้าสนทนาต่อเนื่อง "
                "แปลบทพูดสำคัญให้เป็นธรรมชาติ คงชื่อผู้วาด/ศิลปินตามต้นฉบับทุกตัวอักษรโดยไม่แปลหรือทับศัพท์เป็นไทย (เช่น `ヤッペン (@yatt83)`) "
                "ใช้ชื่อตัวละครตาม Canonical Names ไม่ต้องวางลิงก์โพสต์ซ้ำ แล้วแทรกความเห็นส่วนตัวสั้นๆ ลงไปอย่างเป็นธรรมชาติโดยไม่ตั้งหัวข้อและไม่จบด้วยคำถามชวนคุย)"
            )
        elif clean_prompt or spoken_clip_text:
            user_request = clean_prompt or spoken_clip_text
        elif message.stickers:
            stk_names = ", ".join(s.name for s in message.stickers)
            user_request = f"(ผู้ใช้ส่งสติกเกอร์ `{stk_names}` มาให้ ช่วยตอบโต้หรือแสดงความเห็นสั้นๆ อย่างเป็นธรรมชาติ)"
        elif extracted_ctx:
            if use_coder:
                user_request = "ช่วยตรวจสอบ วิเคราะห์จุดบกพร่องหรือหา Root Cause และสรุปแนวทางแก้ไขสำหรับสิ่งที่แนบมาให้หน่อย"
            else:
                user_request = (
                    "เล่าเหตุการณ์และบทสนทนาที่เห็นในภาพที่แนบมาเป็นย่อหน้าสนทนาต่อเนื่อง แปลบทพูดสำคัญให้เป็นธรรมชาติ "
                    "และแทรกความรู้สึกส่วนตัวต่อฉากนี้ลงไปตรงๆ โดยไม่ตั้งหัวข้อ"
                )
        else:
            user_request = "สวัสดี (ทักทายกลับสั้นๆ อย่างเป็นธรรมชาติ)"

        extra_blocks = "\n\n".join(b for b in (mentioned_dossier, extracted_ctx) if b)
        full_prompt = f"{user_request}\n\n{extra_blocks}" if extra_blocks else user_request
        chat_mode = "code" if use_coder else "general"
        sys_prompt = CODE_SYSTEM_PROMPT if use_coder else FAST_SYSTEM_PROMPT

        tool_ctx = ToolExecutionContext(
            channel_id=message.channel.id,
            guild=message.guild,
            channel=message.channel,
            speaker_user=message.author,
            message=message,
            mode=chat_mode,
        )
        heavy_tracker = _HeavyStatusTracker(message.channel, reply_to=message)
        try:
            answer, did_search = await query_ai_chat(
                full_prompt,
                system_prompt=sys_prompt,
                mode=chat_mode,
                channel_id=message.channel.id,
                temperature=0.2 if use_coder else 0.55,
                allow_agentic_search=(not use_coder),
                has_fresh_attachment=bool(extracted_ctx),
                speaker_user=message.author,
                decision_plan=decision_plan,
                tool_ctx=tool_ctx,
                on_heavy_status=heavy_tracker.update,
            )
        finally:
            await heavy_tracker.cleanup()

        use_reply = should_reply_to_message(
            user_request, has_attachments=bool(message.attachments or extracted_ctx or tool_ctx.tools_used)
        )
        reply_target = message if use_reply else None

        # Legacy [IMAGE: ...] tag safeguard (for fallback models without native tools)
        img_tag_match = AGENTIC_IMAGE_TAG_REGEX.search(answer)
        if (
            img_tag_match
            and not tool_ctx.generated_images
            and CF_ACCOUNT_ID
            and CF_API_TOKEN
            and re.search(r"(รูป|ภาพ|วาด|เจน|สร้าง|image|picture|draw|art|photo)", clean_prompt, re.IGNORECASE)
        ):
            await send_generated_image_to_channel(
                message.channel, img_tag_match.group(1).strip(), reply_to=message
            )
            return

        action_view = None
        if tool_ctx.ot_reply_meta and not is_failure_or_refusal_reply(answer):
            rec_d, rec_s, rec_det = tool_ctx.ot_reply_meta
            action_view = OTActionView(message.author.id, rec_d, rec_s, rec_det)
        elif tool_ctx.memory_updated_uid and not is_failure_or_refusal_reply(answer) and not tool_ctx.did_search and not use_coder:
            action_view = MemoryActionView(requester_id=message.author.id, target_uid=tool_ctx.memory_updated_uid)
        elif should_attach_interactive_view(
            user_request,
            answer,
            use_coder=use_coder,
            did_search=did_search,
            extracted_ctx=extracted_ctx,
            decision_plan=decision_plan,
        ):
            if use_coder:
                action_view = CodeReviewActionView(message.author.id, message.channel.id, user_request, answer)
            else:
                action_view = build_contextual_suggestion_view(
                    message.channel.id,
                    clean_prompt or user_request,
                    answer,
                    extracted_ctx=extracted_ctx,
                    source_message=message,
                    decision_plan=decision_plan,
                )

        mass_ping_mentions, send_answer = _resolve_mass_ping(message, clean_prompt, answer)
        await _send_response_with_interactive_view(
            message.channel, send_answer, action_view, reply_to=reply_target, allowed_mentions=mass_ping_mentions
        )
        await _dispatch_post_tool_side_effects(message.channel, message, answer, user_request, tool_ctx)
        if has_audio and not use_coder:
            await speak_in_voice_channel(message.guild, answer, fallback_channel=message.channel)


@bot.command(name="sync")
async def sync_app_commands(ctx, scope: str = "guild"):
    if not is_owner_user(ctx.author):
        await ctx.send("คำสั่งนี้ให้เฉพาะเจ้าของบอทใช้เท่านั้นนะคะ")
        return
    try:
        if scope.strip().lower() == "global":
            synced = await bot.tree.sync()
            await ctx.send(f"❄️ ซิงค์ Slash Commands & Context Menus แบบ **Global** เรียบร้อยแล้วค่ะ (`{len(synced)}` รายการ — อาจใช้เวลาสูงสุด 1 ชม. ในการอัปเดตทุกเซิร์ฟเวอร์)")
        elif ctx.guild:
            bot.tree.copy_global_to(guild=ctx.guild)
            synced = await bot.tree.sync(guild=ctx.guild)
            await ctx.send(f"❄️ ซิงค์ Slash Commands & Context Menus เข้าสู่เซิร์ฟเวอร์ **`{ctx.guild.name}`** ทันทีเรียบร้อยแล้วค่ะ (`{len(synced)}` รายการ)")
        else:
            synced = await bot.tree.sync()
            await ctx.send(f"❄️ ซิงค์ Global Commands เรียบร้อยแล้วค่ะ (`{len(synced)}` รายการ)")
    except Exception as e:
        await ctx.send(f"ซิงค์คำสั่งติดขัดนิดหน่อยค่ะ (`{type(e).__name__}: {e}`)")


async def _connect_bot_to_voice_channel(ctx, target_channel, greet: bool = True):
    """Shared voice connection helper with DAVE E2EE + registered SilenceGatedVoiceSink."""
    from voice_engine import _KNOWN_VOICE_USER_IDS
    for m in target_channel.members:
        _KNOWN_VOICE_USER_IDS.add(int(m.id))
    if bot.user:
        _KNOWN_VOICE_USER_IDS.add(int(bot.user.id))

    if ctx.guild.voice_client:
        if is_voice_scribe_active(ctx.guild.id):
            await stop_voice_scribe_session(
                ctx.guild,
                fallback_text_channel=ctx.channel,
                triggered_by=ctx.author.display_name,
                speak_ack=False,
                reason="ย้ายห้องเสียง",
            )
        unregister_voice_sink(ctx.guild.id)
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

    prewarm_paradee_tts_bg()
    await asyncio.sleep(1.5)

    if HAS_VOICE_RECV and hasattr(vc, "listen"):
        sink = SilenceGatedVoiceSink(ctx.guild, ctx.channel, vc=vc)
        register_voice_sink(ctx.guild.id, sink)
        vc.listen(sink)

    conn = getattr(vc, "_connection", None)
    dave_sess = getattr(conn, "dave_session", None)
    print(
        f"[Voice Join] Connected to '{target_channel.name}' "
        f"(discord.py={discord.__version__}, davey={davey is not None}, "
        f"dave_session={dave_sess is not None})",
        flush=True,
    )

    if greet:
        await ctx.send(f"เข้ามาในห้องเสียง `{target_channel.name}` แล้วค่ะ 🎙️")
        await speak_in_voice_channel(
            ctx.guild, "เข้ามาในห้องเสียงแล้วค่ะ ถ้ามีธุระอะไรก็พูดมาได้เลยนะคะ"
        )
    return vc


@bot.command(name="join")
async def join_voice(ctx):
    if not ctx.guild:
        await ctx.send("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์ที่มีห้องเสียงค่ะ")
        return
    if not getattr(ctx.author, "voice", None) or not ctx.author.voice.channel:
        await ctx.send("ต้องเข้าไปอยู่ในห้องเสียงก่อนนะคะ ถึงจะเรียกฉันเข้าไปได้")
        return
    target_channel = ctx.author.voice.channel
    print(f"[Voice Join] Command invoked by {ctx.author} for channel '{target_channel.name}'", flush=True)
    try:
        await _connect_bot_to_voice_channel(ctx, target_channel, greet=True)
    except Exception as e:
        print(f"[Voice Join Error] {type(e).__name__}: {e}", flush=True)
        await ctx.send(
            f"เข้าห้องเสียงไม่ได้ชั่วคราวค่ะ (`{type(e).__name__}: {e}`) "
            "แต่ยังส่งไฟล์เสียงในแชทให้ฉันตอบกลับได้ตามปกตินะคะ"
        )


@bot.command(name="scribe", aliases=["จดประชุม", "จดบันทึก", "สรุปเสียง", "meeting", "notes"])
async def scribe_command(ctx, *, args: str = ""):
    """
    Starts (or stops if arg is 'stop'/'หยุด'/'สรุป') passive multi-speaker Voice Scribe mode.
    Auto-joins the caller's voice channel if the bot is not connected yet.
    """
    if not ctx.guild:
        await ctx.send("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์ที่มีห้องเสียงค่ะ")
        return

    arg_clean = args.strip()
    if arg_clean.lower() in ("stop", "end", "finish", "หยุด", "หยุดจด", "หยุดบันทึก", "สรุป", "สรุปประชุม"):
        if not is_voice_scribe_active(ctx.guild.id):
            await ctx.send("ตอนนี้ยังไม่ได้เปิดโหมดจดบันทึกในห้องเสียงค่ะ")
            return
        await stop_voice_scribe_session(
            ctx.guild,
            fallback_text_channel=ctx.channel,
            triggered_by=ctx.author.display_name,
            speak_ack=True,
        )
        return

    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        if not getattr(ctx.author, "voice", None) or not ctx.author.voice.channel:
            await ctx.send("เข้าไปในห้องเสียงก่อนนะคะ แล้วค่อยพิมพ์ `!scribe` หรือ `!join` ค่ะ")
            return
        try:
            await _connect_bot_to_voice_channel(ctx, ctx.author.voice.channel, greet=False)
        except Exception as e:
            await ctx.send(f"เข้าห้องเสียงเพื่อเริ่มจดให้ไม่ได้ชั่วคราวค่ะ (`{type(e).__name__}: {e}`)")
            return

    await start_voice_scribe_session(
        ctx.guild,
        ctx.channel,
        started_by=ctx.author.display_name,
        topic=arg_clean,
        speak_ack=True,
    )


@bot.command(name="stopscribe", aliases=["หยุดจด", "หยุดบันทึก", "สรุปประชุม", "endscribe"])
async def stop_scribe_command(ctx):
    """Stops active Voice Scribe mode, summarizes everyone's discussion, and clears in-RAM transcript."""
    if not ctx.guild or not is_voice_scribe_active(ctx.guild.id):
        await ctx.send("ตอนนี้ยังไม่มีโหมดจดบันทึกที่กำลังทำงานอยู่ค่ะ")
        return
    await stop_voice_scribe_session(
        ctx.guild,
        fallback_text_channel=ctx.channel,
        triggered_by=ctx.author.display_name,
        speak_ack=True,
    )


@bot.command(name="leave")
async def leave_voice(ctx):
    if not ctx.guild:
        await ctx.send("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์ที่มีห้องเสียงค่ะ")
        return
    if is_voice_scribe_active(ctx.guild.id):
        await stop_voice_scribe_session(
            ctx.guild,
            fallback_text_channel=ctx.channel,
            triggered_by=ctx.author.display_name,
            speak_ack=False,
            reason="ออกจากห้องเสียง (!leave)",
        )
    else:
        clear_voice_scribe_session(ctx.guild.id)

    vc = ctx.guild.voice_client
    if vc:
        if hasattr(vc, "stop_listening"):
            try:
                vc.stop_listening()
            except Exception:
                pass
        await speak_in_voice_channel(
            ctx.guild, "ถ้าหมดธุระแล้ว ฉันขอตัวกลับไปอ่านหนังสือต่อนะคะ ลาก่อนค่ะ"
        )
        await vc.disconnect(force=True)
        unregister_voice_sink(ctx.guild.id)
    else:
        await ctx.send("ตอนนี้ฉันไม่ได้อยู่ในห้องเสียงค่ะ")


@bot.command(name="wakeword")
async def toggle_wakeword(ctx, mode: str = ""):
    if not ctx.guild:
        await ctx.send("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์ที่มีห้องเสียงค่ะ")
        return
    gid = ctx.guild.id
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
    await ctx.send(f"🎙️ โหมดเรียกชื่อก่อนพูดในห้องเสียงตอนนี้: **{state_th}** ค่ะ")


@bot.command(name="vstatus")
async def voice_status_command(ctx):
    if not ctx.guild:
        await ctx.send("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์ที่มีห้องเสียงค่ะ")
        return
    vc = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        await ctx.send("ตอนนี้ฉันไม่ได้อยู่ในห้องเสียงค่ะ")
        return
    conn = getattr(vc, "_connection", None)
    dave_s = getattr(conn, "dave_session", None) if conn else None
    status = getattr(dave_s, "status", "None") if dave_s else "None"
    ready = getattr(dave_s, "ready", False) if dave_s else False
    can_enc = getattr(dave_s, "can_encrypt", False) if dave_s else False
    uids = dave_s.get_user_ids() if (dave_s and hasattr(dave_s, "get_user_ids")) else []
    scribe_sess = voice_scribe_sessions.get(ctx.guild.id)
    scribe_line = (
        f"กำลังจดบันทึก (`{len(scribe_sess.get('entries') or [])}` ประโยค)"
        if (scribe_sess and scribe_sess.get("active"))
        else "ปิดอยู่ (พิมพ์ `!scribe` เพื่อเริ่มจดประชุม)"
    )
    await ctx.send(
        f"🎙️ **สถานะห้องเสียง (`{vc.channel.name}`):**\n"
        f"- **Voice Scribe:** {scribe_line}\n"
        f"- **Primary Live STT:** `{GEMINI_TRANSCRIBE_LIVE_MODEL}` (Fallback: Groq Whisper-v3 + Google STT)\n"
        + (
            f"- **TTS Engine:** `MakHamTTS Paradee INT8` (`pitch={TTS_PARADEE_PITCH}`, `speed={TTS_PARADEE_SPEED}x`, "
            f"`baked={has_baked_paradee_model()}`, `loaded={state.paradee_tts_loaded}`)\n"
            if TTS_PRIMARY_ENGINE == "paradee"
            else f"- **TTS Engine:** `{TTS_PRIMARY_ENGINE}` (`pitch={TTS_GOOGLE_PITCH}`, `speed={TTS_GOOGLE_SPEED}x`)\n"
        )
        + f"- **DAVE E2EE Status:** `{status}` | **Ready:** `{ready}` | **Can Encrypt:** `{can_enc}`\n"
        f"- **MLS Group User IDs:** `{uids}`"
    )


@bot.command(name="imagine", aliases=["img", "draw"])
async def imagine_command(ctx, *, prompt: str = ""):
    if not prompt.strip():
        await ctx.send("ระบุมาด้วยนะคะว่าจะให้วาดอะไร เช่น `!imagine แมวส้มนั่งใช้คอมพิวเตอร์ในห้อง` ค่ะ")
        return
    async with ctx.typing():
        await send_generated_image_to_channel(ctx.channel, prompt, reply_to=ctx.message)


@bot.command(name="manga", aliases=["translate", "tl"])
async def manga_command(ctx, *, args: str = "English"):
    await process_manga_translation_request(ctx, args)


@bot.command(name="yukipedia", aliases=["yp", "wiki", "ยูกิพีเดีย"])
async def yukipedia_command(ctx, *, query: str = ""):
    await process_yukipedia_request(ctx, query)


@bot.command(name="ask")
async def ask_fast(ctx, *, prompt: str = ""):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        effective_prompt = prompt.strip() or (
            "ช่วยตรวจสอบ สรุปเนื้อหา และบอกความรู้สึกหรือความคิดเห็นของเธอต่อสิ่งที่แนบมาให้หน่อย" if ctx_text else "สวัสดี"
        )
        is_dt = (not ctx_text) and (try_instant_datetime_reply(effective_prompt) is not None)
        dec_plan = None if is_dt else await orchestrate_message_with_decision_1(
            effective_prompt,
            extracted_ctx=ctx_text,
            purpose="AskCmd",
            channel_id=ctx.channel.id,
        )
        tool_ctx = ToolExecutionContext(
            channel_id=ctx.channel.id,
            guild=ctx.guild,
            channel=ctx.channel,
            speaker_user=ctx.author,
            message=ctx.message,
            mode="general",
        )
        heavy_tracker = _HeavyStatusTracker(ctx.channel, reply_to=ctx.message)
        try:
            answer, did_search = await query_ai_chat(
                f"{effective_prompt}\n\n{ctx_text}" if ctx_text else effective_prompt,
                system_prompt=FAST_SYSTEM_PROMPT,
                mode="general",
                channel_id=ctx.channel.id,
                has_fresh_attachment=bool(ctx_text),
                speaker_user=ctx.author,
                decision_plan=dec_plan,
                tool_ctx=tool_ctx,
                on_heavy_status=heavy_tracker.update,
            )
        finally:
            await heavy_tracker.cleanup()

        use_reply = should_reply_to_message(
            effective_prompt, has_attachments=bool(ctx_text or tool_ctx.tools_used), is_instant_datetime=is_dt
        )
        view = (
            build_contextual_suggestion_view(
                ctx.channel.id,
                effective_prompt,
                answer,
                extracted_ctx=ctx_text,
                source_message=ctx.message,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                effective_prompt,
                answer,
                did_search=did_search,
                extracted_ctx=ctx_text,
                is_instant_dt=is_dt,
                decision_plan=dec_plan,
            )
            else None
        )
        await _send_response_with_interactive_view(ctx.channel, answer, view, reply_to=ctx.message if use_reply else None)
        await _dispatch_post_tool_side_effects(ctx.channel, ctx.message, answer, effective_prompt, tool_ctx)


@bot.command(name="code", aliases=["review", "diff", "trace"])
async def ask_code(ctx, *, prompt: str = "วิเคราะห์โค้ดหรือ Stack Trace ที่แนบมาและระบุ Root Cause พร้อมวิธีแก้ไข"):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        tool_ctx = ToolExecutionContext(
            channel_id=ctx.channel.id,
            guild=ctx.guild,
            channel=ctx.channel,
            speaker_user=ctx.author,
            message=ctx.message,
            mode="code",
        )
        heavy_tracker = _HeavyStatusTracker(ctx.channel, reply_to=ctx.message)
        try:
            answer, _ = await query_ai_chat(
                f"{prompt}\n\n{ctx_text}" if ctx_text else prompt,
                system_prompt=CODE_SYSTEM_PROMPT,
                mode="code",
                channel_id=ctx.channel.id,
                temperature=0.2,
                allow_agentic_search=False,
                has_fresh_attachment=bool(ctx_text),
                speaker_user=ctx.author,
                decision_plan={"route": "code", "route_confidence": 0.99, "needs_web_search": False, "search_confidence": 0.99},
                tool_ctx=tool_ctx,
                on_heavy_status=heavy_tracker.update,
            )
        finally:
            await heavy_tracker.cleanup()

        view = (
            CodeReviewActionView(ctx.author.id, ctx.channel.id, prompt, answer)
            if should_attach_interactive_view(prompt, answer, use_coder=True, extracted_ctx=ctx_text)
            else None
        )
        await _send_response_with_interactive_view(ctx.channel, answer, view, reply_to=ctx.message)
        await _dispatch_post_tool_side_effects(ctx.channel, ctx.message, answer, prompt, tool_ctx)


@bot.command(name="source", aliases=["sauce", "whatis", "วาร์ป"])
async def source_command(ctx, *, question: str = ""):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        if not ctx_text:
            await ctx.send(
                "ถ้าไม่แนบรูป วางลิงก์ (X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) "
                "หรือกด Reply ข้อความที่มีรูปมาด้วย ฉันก็ช่วยเช็คที่มาให้ไม่ได้นะคะ"
            )
            return
        dec_plan = await orchestrate_message_with_decision_1(
            question.strip() or ctx.message.content,
            extracted_ctx=ctx_text,
            purpose="SourceCmd",
            channel_id=ctx.channel.id,
        ) or {}
        dec_plan = dict(dec_plan)
        dec_plan["suppress_social_embeds"] = False
        user_q = question.strip() or (
            "ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด: "
            "หากเป็นภาพวาด/อนิเมะ/มังงะ ให้บอกชื่อเรื่อง ชื่อตัวละคร ตอนที่/Chapter ชื่อผู้วาด และแนบลิงก์ที่มา (เช่น Pixiv/X/MyAnimeList) ถ้ามี "
            "หากเป็นภาพถ่ายสิ่งของหรืออุปกรณ์ ให้ระบุยี่ห้อ รุ่น และรายละเอียดสำคัญของสิ่งนั้นค่ะ"
        )
        tool_ctx = ToolExecutionContext(
            channel_id=ctx.channel.id,
            guild=ctx.guild,
            channel=ctx.channel,
            speaker_user=ctx.author,
            message=ctx.message,
            mode="general",
        )
        heavy_tracker = _HeavyStatusTracker(ctx.channel, reply_to=ctx.message)
        try:
            answer, did_search = await query_ai_chat(
                f"{user_q}\n\n{ctx_text}",
                system_prompt=FAST_SYSTEM_PROMPT,
                mode="general",
                channel_id=ctx.channel.id,
                temperature=0.2,
                allow_agentic_search=True,
                has_fresh_attachment=True,
                speaker_user=ctx.author,
                decision_plan=dec_plan,
                tool_ctx=tool_ctx,
                on_heavy_status=heavy_tracker.update,
            )
        finally:
            await heavy_tracker.cleanup()

        view = (
            build_contextual_suggestion_view(
                ctx.channel.id,
                question.strip() or ctx.message.content,
                answer,
                extracted_ctx=ctx_text,
                source_message=ctx.message,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                user_q, answer, did_search=did_search, extracted_ctx=ctx_text, decision_plan=dec_plan
            )
            else None
        )
        await _send_response_with_interactive_view(ctx.channel, answer, view, reply_to=ctx.message)
        await _dispatch_post_tool_side_effects(ctx.channel, ctx.message, answer, user_q, tool_ctx)


@bot.command(name="status", aliases=["stat", "ping", "ram"])
async def status_command(ctx):
    m = _collect_health_metrics()
    d = m["discord"]
    mem = m["memory"]
    rl = m["runtime_limits"]
    ai = m["ai_backends"]
    tel = m["telemetry"]
    pc = tel["provider_calls"]

    ping_str = f"{d['gateway_ping_ms']} ms" if d["gateway_ping_ms"] is not None else "N/A"
    active_topics_count = len(getattr(config, "channel_active_topic", {}))

    await ctx.send(
        f"❄️ **สถานะระบบ (Yukinon System Status)**\n"
        f"- **เวลาปัจจุบัน:** `{m['system_time_th']}`\n"
        f"- **Uptime:** `{m['uptime_human']}` | **Ping:** `{ping_str}`\n"
        f"- **Voice & Scribe:** `{d['active_voice_sessions']}` ห้องเสียง | `{d['active_scribe_sessions']}` จดบันทึก | "
        + (
            f"TTS `paradee {ai['tts_paradee_pitch']}p / {ai['tts_paradee_speed']}x` (`loaded={ai['tts_paradee_loaded']}`)\n"
            if ai["tts_primary_engine"] == "paradee"
            else f"TTS `{ai['tts_primary_engine']} {ai['tts_google_pitch']}p / {ai['tts_google_speed']}x`\n"
        )
        + f"- **Native Tools:** `{len(ai['registered_tools'])}` เครื่องมือ (`{', '.join(ai['registered_tools'])}`)\n"
        f"- **Container RAM:** `{mem['bot_ram_mb']} MB / {mem['ram_limit_mb']} MB ({mem['ram_usage_pct']}%)`\n"
        f"- **Runtime Limits:** OpenCV `{rl['opencv_threads']}t` | Manga `{rl['manga_concurrency']}` | STT `{rl['voice_stt_concurrency']}` | Channel LRU `{rl['max_cached_channels']}`\n"
        f"- **Cached Contexts:** `{mem['active_chat_channels']}/{mem['max_cached_channels']}` แชท | "
        f"`{mem['cached_attachments']}` ไฟล์แนบ | "
        f"`{active_topics_count}` หัวข้อสนทนา | "
        f"`{mem['stored_user_profiles']}` โปรไฟล์ผู้ใช้\n"
        f"- **Telemetry:** Decision-1 `{tel['decision_calls']}` calls (`{tel['decision_avg_ms']}ms`, err `{tel['decision_errors']}`) | "
        f"SearchFilter drop `{tel['search_filter_dropped']}` (`{tel['search_filter_drop_pct']}%`) | "
        f"Cache `{tel['gemini_cache_hits']}` hits (`{tel['gemini_cached_tokens_total']:,}` tok)\n"
        f"- **Provider Calls:** Flash `{pc.get('gemini_flash', 0)}` | H2 `{pc.get('gemini_hydrogen', 0)}` | "
        f"Gemma `{pc.get('gemma_ai_studio', 0)}` | Groq `{pc.get('groq', 0)}` | CF `{pc.get('cloudflare', 0)}` | OR `{pc.get('openrouter', 0)}`"
    )


@bot.command(name="clear")
async def clear_memory(ctx, scope: str = ""):
    if scope.strip().lower() == "all":
        channel_memory.clear()
        channel_last_attachment.clear()
        if hasattr(config, "channel_active_topic"):
            config.channel_active_topic.clear()
        scope_label = "ทุกห้องแชท"
    else:
        channel_memory.pop(ctx.channel.id, None)
        channel_last_attachment.pop(ctx.channel.id, None)
        if hasattr(config, "channel_active_topic"):
            config.channel_active_topic.pop(ctx.channel.id, None)
        scope_label = "ห้องแชทนี้"

    before_mb, after_mb = release_unused_memory()
    freed_mb = round(max(0.0, before_mb - after_mb), 1)
    ram_note = (
        f"คืนพื้นที่ RAM `{before_mb} MB` -> `{after_mb} MB` (ลดลง `{freed_mb} MB`)"
        if freed_mb > 0
        else f"RAM ปัจจุบัน: `{after_mb} MB / {CONTAINER_RAM_LIMIT_MB} MB`"
    )
    await ctx.send(
        f"🧹 ล้างประวัติคุยกับไฟล์แนบของ**{scope_label}**ให้เรียบร้อยแล้วค่ะ ({ram_note})"
    )


@bot.command(name="calc")
async def calc_exp(ctx, *, expr: str = ""):
    clean_expr = (expr or "").strip()
    if not clean_expr:
        await ctx.send("ใส่โจทย์ที่จะให้คำนวณมาด้วยนะคะ เช่น `!calc (1250 * 1.07) / 3` ค่ะ")
        return
    if len(clean_expr) > 200:
        await ctx.send("โจทย์ยาวเกินไปค่ะ (ไม่เกิน 200 ตัวอักษรนะคะ)")
        return
    try:
        res = SimpleEval().eval(clean_expr)
        res_str = str(res)
        if len(res_str) > 1500:
            res_str = res_str[:1500] + "... (truncated)"
        await ctx.send(f"ได้ `{res_str}` ค่ะ")
    except Exception as e:
        await ctx.send(f"คิดเลขไม่ได้ค่ะ (`{e}`)")


@bot.command(name="ot")
async def ot_command(ctx, *, args: str = ""):
    query = args.strip()
    if query and not is_ot_command_text(query, channel_id=ctx.channel.id, user_id=ctx.author.id):
        query = f"จด OT {query}"
    reply = await process_ot_request(ctx.message, query or "สรุป OT")
    rec_date, rec_slot, rec_detail = parse_ot_record_metadata(reply)
    ot_view = (
        OTActionView(ctx.author.id, rec_date, rec_slot, rec_detail)
        if rec_date and not is_failure_or_refusal_reply(reply)
        else None
    )
    await _send_response_with_interactive_view(ctx.channel, reply, ot_view, reply_to=ctx.message)


@bot.command(name="memory", aliases=["memo", "profile", "ความจำ"])
async def memory_command(ctx, *, args: str = ""):
    query = args.strip() or "ดูความจำ"
    reply, target_uid = await process_memory_request(ctx.message, query)
    mem_view = MemoryActionView(requester_id=ctx.author.id, target_uid=target_uid)
    await _send_response_with_interactive_view(ctx.channel, reply, mem_view, reply_to=ctx.message)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)