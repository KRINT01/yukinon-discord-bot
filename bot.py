import os
import io
import re
import time
import base64
import asyncio
import platform
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil
from aiohttp import web
import discord
from discord.ext import commands, tasks
from simpleeval import SimpleEval

import config
from config import (
    DISCORD_TOKEN,
    PORT,
    BOOT_TIME,
    ALLOWED_GUILD_IDS,
    AI_CHANNEL_IDS,
    OLLAMA_BASE_URL,
    ACTIVE_FAST_MODEL,
    ACTIVE_CODE_MODEL,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    CF_API_TOKEN,
    CF_ACCOUNT_ID,
    CF_IMAGE_MODEL,
    CF_VISION_MODEL,
    CF_CODE_MODEL,
    CF_CHAT_FAST_MODEL,
    GEMINI_API_KEY,
    GEMINI_DECISION_MODEL,
    GEMINI_HEAVY_MODEL,
    GEMINI_TRANSCRIBE_LIVE_MODEL,
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
    URL_REGEX,
    CODE_AND_LOG_EXTENSIONS,
    intents,
    channel_memory,
    channel_last_attachment,
    voice_wake_word_enabled,
    voice_scribe_sessions,
    state,
    davey,
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
    resolve_user_mentions_and_build_dossier,
    is_ot_command_text,
    process_ot_request,
    is_memory_command_text,
    process_memory_request,
    build_monthly_ot_summary_markdown,
    _parse_stored_ot_date,
    user_profiles,
)
from search_engine import detect_coding_intent, run_safe_web_search_async
from llm import (
    query_local_ollama_chat,
    orchestrate_message_with_decision_1,
)
from media_context import extract_all_context, resolve_referenced_message
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

OT_REPORT_CHANNEL_ID = int(os.getenv("OT_REPORT_CHANNEL_ID", "0") or "0")

_COMPOUND_CONNECTOR_REGEX = re.compile(
    r"(แล้ว(?:ก็)?|พร้อม(?:กับ|ทั้ง)?|และ(?:ช่วย)?|กับ|แถม|\band\s+(?:also|then)?\b|\bplus\b|,)",
    re.IGNORECASE,
)
_MATH_EXPR_CUE_REGEX = re.compile(
    r"(\d+\s*[\+\-\*/x×^%]\s*\d+|คำนวณ|แปลงค่าเงิน|กี่บาท|กี่เยน|กี่ดอลลาร์|\bcalculate\b|\bconvert\b)",
    re.IGNORECASE,
)
_MEMORY_INLINE_CUE_REGEX = re.compile(
    r"(จำไว้ว่า|ช่วยจำว่า|ปักหมุดความจำ|บันทึกความจำ|\bremember\s+that\b)",
    re.IGNORECASE,
)
_OT_INLINE_CUE_REGEX = re.compile(
    r"((?:จด|บันทึก|สรุป|ลบ)\s*(?:เวลา\s*)?ot\b)",
    re.IGNORECASE,
)


def _looks_like_compound_request(text: str) -> bool:
    """
    Detects multi-intent compound prompts (e.g., weather + math + memory, or search + image gen)
    so they bypass single-intent fast-paths and execute all actions in parallel via Native Function Calling!
    """
    if not text or len(text.strip()) < 14:
        return False
    if not _COMPOUND_CONNECTOR_REGEX.search(text):
        return False

    hit_domains = 0
    if WEATHER_INTENT_PATTERN.search(text):
        hit_domains += 1
    if _MATH_EXPR_CUE_REGEX.search(text):
        hit_domains += 1
    if _MEMORY_INLINE_CUE_REGEX.search(text):
        hit_domains += 1
    if _OT_INLINE_CUE_REGEX.search(text):
        hit_domains += 1
    if IMAGE_GEN_INTENT_PATTERN.search(text):
        hit_domains += 1
    if re.search(r"(ข่าว|ราคา|ค้นหา|สรุป|เช็คข้อมูล|\bsearch\b|\bnews\b|\bprice\b)", text, re.IGNORECASE):
        hit_domains += 1
    return hit_domains >= 2


def _format_uptime(seconds: int) -> str:
    days, rem = divmod(max(0, int(seconds)), 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m {secs}s"
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def _encode_low_res_decision_thumb_sync(raw_bytes: bytes) -> str:
    from PIL import Image
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    img.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _preload_image_attachments_and_thumb(
    all_msg_atts: list[discord.Attachment],
) -> tuple[dict[int, bytes], list[tuple[str, str]]]:
    preloaded: dict[int, bytes] = {}
    img_atts = [
        a for a in all_msg_atts
        if a.filename.lower().endswith(IMAGE_EXTENSIONS) and a.size <= 8 * 1024 * 1024
    ][:4]
    for att in img_atts:
        try:
            preloaded[att.id] = await att.read()
        except Exception:
            continue

    low_res_imgs: list[tuple[str, str]] = []
    if preloaded and GEMINI_API_KEY and time.time() >= state.gemini_decision_cooldown_until:
        try:
            first_bytes = next(iter(preloaded.values()))
            b64_thumb = await asyncio.to_thread(_encode_low_res_decision_thumb_sync, first_bytes)
            low_res_imgs.append((b64_thumb, "image/jpeg"))
        except Exception:
            pass

    return preloaded, low_res_imgs


def _collect_health_metrics() -> dict:
    prune_stale_attachments()
    now = time.time()
    proc = psutil.Process(os.getpid())
    rss_mb = round(proc.memory_info().rss / (1024 ** 2), 1)
    ram_pct = round((rss_mb / 512.0) * 100, 1)
    uptime_sec = int(now - BOOT_TIME)

    ws_ping_ms = (
        round(bot.latency * 1000, 1)
        if bot.is_ready() and bot.latency == bot.latency
        else None
    )
    gemini_cd = max(0, int(state.gemini_grounding_cooldown_until - now))
    decision_cd = max(0, int(state.gemini_decision_cooldown_until - now))
    hydrogen_cd = max(0, int(state.gemini_hydrogen_cooldown_until - now))
    live_stt_cd = max(0, int(state.gemini_transcribe_live_cooldown_until - now))
    edge_cd = max(0, int(state.edge_tts_cooldown_until - now))
    cached_files_count = sum(len(q) for q in channel_last_attachment.values())
    active_scribe_count = sum(1 for s in voice_scribe_sessions.values() if s.get("active"))

    return {
        "status": "ok" if rss_mb < 480 else "warning_high_memory",
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
            "render_limit_mb": 512,
            "ram_usage_pct": ram_pct,
            "active_chat_channels": len(channel_memory),
            "cached_attachments": cached_files_count,
            "stored_user_profiles": len(user_profiles),
        },
        "ai_backends": {
            "native_function_calling": True,
            "registered_tools": [s["name"] for s in TOOL_SPECS],
            "modal_ollama_configured": bool(OLLAMA_BASE_URL),
            "fast_model": ACTIVE_FAST_MODEL,
            "code_model": GROQ_CODE_MODEL if GROQ_API_KEY else ACTIVE_CODE_MODEL,
            "groq_enabled": bool(GROQ_API_KEY),
            "groq_search_model": state.active_groq_search_model if GROQ_API_KEY else None,
            "groq_vision_model": state.active_groq_vision_model if GROQ_API_KEY else None,
            "cloudflare_enabled": bool(CF_API_TOKEN and CF_ACCOUNT_ID),
            "cloudflare_image_model": CF_IMAGE_MODEL if CF_API_TOKEN else None,
            "cloudflare_gemma_model": CF_VISION_MODEL if CF_API_TOKEN else None,
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
            "tts_google_pitch": TTS_GOOGLE_PITCH,
            "tts_google_speed": TTS_GOOGLE_SPEED,
            "edge_tts_cooldown_sec": edge_cd,
        },
    }


async def handle_root(request: web.Request) -> web.Response:
    m = _collect_health_metrics()
    d = m["discord"]
    mem = m["memory"]
    ai = m["ai_backends"]

    ping_str = f"{d['gateway_ping_ms']} ms" if d["gateway_ping_ms"] is not None else "N/A"
    gw_status = (
        f"🟢 Connected as <b>{d['user']}</b> ({d['guilds_count']} guilds | Ping: {ping_str})"
        if d["connected"]
        else "🟡 Connecting to Discord Gateway..."
    )
    gemini_cd_note = f" (Grounding Cooldown: {ai['gemini_grounding_cooldown_sec']}s)" if ai["gemini_grounding_cooldown_sec"] else ""
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
        <p><b>Container RAM:</b> {mem['bot_ram_mb']} MB / {mem['render_limit_mb']} MB ({mem['ram_usage_pct']}%) |
           <b>Cached Contexts:</b> {mem['active_chat_channels']} chat(s), {mem['cached_attachments']} file(s), {mem['stored_user_profiles']} profile(s)</p>
        <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;">
        <p><b>Google AI Studio:</b> {"Enabled (Orchestrator: <code>" + str(ai['gemini_decision_orchestrator']) + "</code> | Heavy Specialist: <code>" + str(ai['gemini_heavy_specialist']) + "</code> | Live STT: <code>" + str(ai['gemini_transcribe_live_model']) + "</code> | Gemma 4: <code>" + str(ai['gemma_active_model']) + "</code> | Flash-Lite: <code>" + str(ai['gemini_active_model']) + "</code>)" + gemini_cd_note if ai['gemini_enabled'] else "Disabled"}</p>
        <p><b>Cloudflare Workers AI:</b> {"Enabled (Code: <code>" + CF_CODE_MODEL + "</code> | Fast: <code>" + CF_CHAT_FAST_MODEL + "</code> | Vision/Gemma: <code>" + str(ai['cloudflare_gemma_model']) + "</code> | Image: <code>" + str(ai['cloudflare_image_model']) + "</code>)" if ai['cloudflare_enabled'] else "Disabled"}</p>
        <p><b>Groq LPU Acceleration:</b> {"Enabled (Code: <code>" + GROQ_CODE_MODEL + "</code> | Search: <code>" + str(ai['groq_search_model']) + "</code> | Vision: <code>" + str(ai['groq_vision_model']) + "</code>)" if ai['groq_enabled'] else "Disabled"}</p>
        <p><b>Modal GPU Ollama:</b> {"Configured" if ai['modal_ollama_configured'] else "Not Configured"} (Fallback: <code>{ai['fast_model']}</code>)</p>
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
            if any(k in ch.name.lower() for k in ("work", "ot", "งาน")):
                return ch
    return None


@tasks.loop(minutes=15)
async def monthly_ot_auto_report_task():
    if not bot.is_ready():
        return
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    prev_year, prev_month, prev_month_label = get_previous_month_info(now_th)
    ym_key = f"{prev_year:04d}-{prev_month:02d}"

    if now_th.day != 1 or now_th.hour < 9 or state.last_ot_auto_report_ym == ym_key:
        return

    state.last_ot_auto_report_ym = ym_key

    for uid_str, prof in list(user_profiles.items()):
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
            await send_chunked_response(
                target_channel,
                f"📅 **[สรุปยอด OT ประจำเดือน{prev_month_label} — รายงานอัตโนมัติวันที่ 1 ของเดือนใหม่]**\n{summary_md}",
            )


class YukinonBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_runner = None

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

    async def close(self):
        for gid in list(voice_scribe_sessions.keys()):
            clear_voice_scribe_session(gid)
        if monthly_ot_auto_report_task.is_running():
            monthly_ot_auto_report_task.cancel()
        await close_http_session()
        if self.web_runner:
            await self.web_runner.cleanup()
        await super().close()


bot = YukinonBot(command_prefix="!", intents=intents, help_command=None)
state.bot = bot


@bot.check
async def verify_allowed_guild(ctx):
    if not ALLOWED_GUILD_IDS or ctx.guild is None or ctx.guild.id in ALLOWED_GUILD_IDS:
        return True
    await ctx.send("This Discord server is not on the bot's authorized allowlist.")
    return False


@bot.event
async def on_ready():
    await restore_user_profiles_from_discord(bot)
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
):
    if not answer or not answer.strip():
        return
    if len(answer) <= 1900:
        sent_msg = None
        if reply_to is not None:
            try:
                sent_msg = await reply_to.reply(answer, view=view, mention_author=False)
            except Exception:
                sent_msg = None
        if sent_msg is None:
            sent_msg = await channel.send(answer, view=view)
        if view is not None and hasattr(view, "message"):
            view.message = sent_msg
            view.delete_message_on_timeout = False
        return

    await send_chunked_response(channel, answer, reply_to=reply_to)
    if view is not None:
        footer_msg = await channel.send("-# ตัวเลือกดำเนินการเพิ่มเติม:", view=view)
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
                "สกัดข้อความ แปล และวิเคราะห์รายละเอียดทั้งหมดในฐานะ Yukipedia "
                "พร้อมตรวจสอบคำผิดหรือจุดบกพร่องในต้นฉบับให้ด้วย"
            )
        elif not effective_query and not extracted_ctx:
            await _send_or_reply(
                channel,
                msg,
                content=(
                    "หากจะพึ่งพาคลังความรู้ **Yukipedia** ของฉัน ก็ช่วยระบุหัวข้อที่ต้องการค้นหา "
                    "หรือแนบภาพ/เอกสารที่ต้องการให้ฉันสกัดข้อความ (OCR) และแปลภาษาเข้ามาด้วยสิคะ "
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
        answer, did_search = await query_local_ollama_chat(
            composed_prompt,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=channel.id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=bool(extracted_ctx or search_block),
            speaker_user=author,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
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

        if MANGA_TRANSLATE_INTENT_PATTERN.search(clean_prompt) or (
            (has_image_att or has_embedded_media_or_link)
            and re.search(r"(แปล|\btranslate\b|\btl\b)", clean_prompt, re.IGNORECASE)
        ):
            await process_manga_translation_request(message, clean_prompt)
            return

        if not message.attachments and IMAGE_GEN_INTENT_PATTERN.search(clean_prompt):
            async with message.channel.typing():
                await send_generated_image_to_channel(message.channel, clean_prompt, reply_to=message)
            return

    # 1. Single-Pass vs PreRoute Decision-1 Orchestration (context-aware with last 2 turns of channel_memory!)
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

    preloaded_att_bytes, low_res_imgs = (
        await _preload_image_attachments_and_thumb(all_msg_atts)
        if (has_image_att and not skip_preroute)
        else ({}, [])
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

    if pre_route == "ot_log" and pre_conf >= 0.82:
        preloaded_att_bytes.clear()
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
        await process_manga_translation_request(message, clean_prompt)
        return

    if not message.attachments and pre_route == "image_gen" and pre_conf >= 0.82:
        preloaded_att_bytes.clear()
        async with message.channel.typing():
            await send_generated_image_to_channel(message.channel, clean_prompt, reply_to=message)
        return

    async with message.channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(
            message, preloaded_attachments=preloaded_att_bytes
        )
        preloaded_att_bytes.clear()

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
            user_request = (
                f"{clean_prompt}\n"
                "(ผู้ใช้ส่งลิงก์นี้มาให้ ช่วยเล่าเหตุการณ์และบริบทของโพสต์นี้โดยเชื่อมโยงแคปชัน ชื่อผู้วาด และบทสนทนาในภาพตามลำดับช่องอ่าน "
                "แปลบทพูดสำคัญเป็นภาษาไทยอย่างเป็นธรรมชาติ และแสดงความรู้สึกส่วนตัวของเธอต่อฉากนี้โดยตรงเป็นย่อหน้าสนทนาต่อเนื่อง "
                "ห้ามตั้งหัวข้อเมตาเช่น 'เนื้อหาทางภาพ', 'เหตุการณ์และบริบทของเรื่องราว', 'ความรู้สึกของฉัน' หรือ 'ความคิดเห็นในสไตล์ยูกิโนะ' และห้ามสร้างตาราง Markdown เด็ดขาด)"
            )
        elif clean_prompt or spoken_clip_text:
            user_request = clean_prompt or spoken_clip_text
        elif message.stickers:
            stk_names = ", ".join(s.name for s in message.stickers)
            user_request = f"(ผู้ใช้ส่งสติกเกอร์ `{stk_names}` มาให้ ช่วยตอบโต้หรือแสดงความเห็นต่อสติกเกอร์นี้ในสไตล์ยูกิโนะ)"
        elif extracted_ctx:
            if use_coder:
                user_request = "ช่วยตรวจสอบ วิเคราะห์จุดบกพร่องหรือหา Root Cause และสรุปแนวทางแก้ไขสำหรับสิ่งที่แนบมาให้หน่อย"
            else:
                user_request = (
                    "ช่วยเล่าเหตุการณ์และบทสนทนาที่เห็นในภาพที่แนบมาตามลำดับช่องอ่าน แปลบทพูดสำคัญเป็นภาษาไทยอย่างเป็นธรรมชาติ "
                    "และบอกความรู้สึกส่วนตัวของเธอต่อฉากนี้โดยตรงเป็นย่อหน้าสนทนาต่อเนื่อง (ห้ามตั้งหัวข้อเมตาเช่น 'เนื้อหาทางภาพ' หรือ 'ความรู้สึกของฉัน' เด็ดขาด)"
                )
        else:
            user_request = "สวัสดี (ทักทายกลับตามธรรมชาติในสไตล์ยูกิโนะ)"

        extra_blocks = "\n\n".join(b for b in (mentioned_dossier, extracted_ctx) if b)
        full_prompt = f"{user_request}\n\n{extra_blocks}" if extra_blocks else user_request
        model = ACTIVE_CODE_MODEL if use_coder else ACTIVE_FAST_MODEL
        sys_prompt = CODE_SYSTEM_PROMPT if use_coder else FAST_SYSTEM_PROMPT

        tool_ctx = ToolExecutionContext(
            channel_id=message.channel.id,
            guild=message.guild,
            channel=message.channel,
            speaker_user=message.author,
            message=message,
            mode="code" if use_coder else "general",
        )
        answer, did_search = await query_local_ollama_chat(
            full_prompt,
            model=model,
            system_prompt=sys_prompt,
            channel_id=message.channel.id,
            temperature=0.2 if use_coder else 0.55,
            allow_agentic_search=(not use_coder),
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=message.author,
            decision_plan=decision_plan,
            tool_ctx=tool_ctx,
        )

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

        await _send_response_with_interactive_view(message.channel, answer, action_view, reply_to=reply_target)
        await _dispatch_post_tool_side_effects(message.channel, message, answer, user_request, tool_ctx)
        if has_audio and not use_coder:
            await speak_in_voice_channel(message.guild, answer, fallback_channel=message.channel)


@bot.command(name="sync")
async def sync_app_commands(ctx, scope: str = "guild"):
    if not is_owner_user(ctx.author):
        await ctx.send("คำสั่งนี้สงวนไว้สำหรับเจ้าของบอทเท่านั้นนะคะ")
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
        await ctx.send(f"เกิดข้อผิดพลาดระหว่างซิงค์คำสั่งค่ะ (`{type(e).__name__}: {e}`)")


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
            ctx.guild, "เข้ามาในห้องเสียงแล้วค่ะ มีเรื่องอะไรให้ช่วยก็พูดมาได้เลยนะคะ"
        )
    return vc


@bot.command(name="join")
async def join_voice(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("ต้องเข้าไปอยู่ในห้องเสียงก่อนนะคะ ถึงจะเรียกฉันเข้าไปได้")
        return
    target_channel = ctx.author.voice.channel
    print(f"[Voice Join] Command invoked by {ctx.author} for channel '{target_channel.name}'", flush=True)
    try:
        await _connect_bot_to_voice_channel(ctx, target_channel, greet=True)
    except Exception as e:
        print(f"[Voice Join Error] {type(e).__name__}: {e}", flush=True)
        await ctx.send(
            f"เกิดข้อผิดพลาดขณะเชื่อมต่อห้องเสียงค่ะ (`{type(e).__name__}: {e}`) "
            "แต่ยังส่งไฟล์เสียงเข้าแชทเพื่อให้ฉันตอบกลับเป็นเสียงพูดได้ตามปกตินะคะ"
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
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("กรุณาเข้าไปในห้องเสียงก่อนนะคะ แล้วค่อยพิมพ์ `!scribe` หรือ `!join` ค่ะ")
            return
        try:
            await _connect_bot_to_voice_channel(ctx, ctx.author.voice.channel, greet=False)
        except Exception as e:
            await ctx.send(f"ไม่สามารถเข้าห้องเสียงเพื่อเริ่มจดบันทึกได้ค่ะ (`{type(e).__name__}: {e}`)")
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
    if ctx.guild and is_voice_scribe_active(ctx.guild.id):
        await stop_voice_scribe_session(
            ctx.guild,
            fallback_text_channel=ctx.channel,
            triggered_by=ctx.author.display_name,
            speak_ack=False,
            reason="ออกจากห้องเสียง (!leave)",
        )
    elif ctx.guild:
        clear_voice_scribe_session(ctx.guild.id)

    vc = ctx.guild.voice_client if ctx.guild else None
    if vc:
        if hasattr(vc, "stop_listening"):
            try:
                vc.stop_listening()
            except Exception:
                pass
        if ctx.guild:
            unregister_voice_sink(ctx.guild.id)
        await speak_in_voice_channel(
            ctx.guild, "ถ้าหมดธุระแล้ว ฉันขอตัวกลับไปอ่านหนังสือต่อนะคะ ลาก่อนค่ะ"
        )
        await vc.disconnect(force=True)


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


@bot.command(name="vstatus")
async def voice_status_command(ctx):
    vc = ctx.guild.voice_client if ctx.guild else None
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
        f"- **TTS Yukino Filter:** `pitch={TTS_GOOGLE_PITCH}`, `speed={TTS_GOOGLE_SPEED}x`\n"
        f"- **DAVE E2EE Status:** `{status}` | **Ready:** `{ready}` | **Can Encrypt:** `{can_enc}`\n"
        f"- **MLS Group User IDs:** `{uids}`"
    )


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
        answer, did_search = await query_local_ollama_chat(
            f"{effective_prompt}\n\n{ctx_text}" if ctx_text else effective_prompt,
            ACTIVE_FAST_MODEL,
            FAST_SYSTEM_PROMPT,
            ctx.channel.id,
            has_fresh_attachment=bool(ctx_text),
            speaker_user=ctx.author,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
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
        answer, _ = await query_local_ollama_chat(
            f"{prompt}\n\n{ctx_text}" if ctx_text else prompt,
            ACTIVE_CODE_MODEL,
            CODE_SYSTEM_PROMPT,
            ctx.channel.id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(ctx_text),
            speaker_user=ctx.author,
            decision_plan={"route": "code", "route_confidence": 0.99, "needs_web_search": False, "search_confidence": 0.99},
            tool_ctx=tool_ctx,
        )
        view = (
            CodeReviewActionView(ctx.author.id, ctx.channel.id, prompt, answer)
            if should_attach_interactive_view(prompt, answer, use_coder=True, extracted_ctx=ctx_text)
            else None
        )
        await _send_response_with_interactive_view(ctx.channel, answer, view, reply_to=ctx.message)


@bot.command(name="source", aliases=["sauce", "whatis", "วาร์ป"])
async def source_command(ctx, *, question: str = ""):
    async with ctx.typing():
        ctx_text, _ = await extract_all_context(ctx.message)
        if not ctx_text:
            await ctx.send(
                "กรุณาแนบรูปภาพ วางลิงก์ (เช่น X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) "
                "หรือกด Reply ที่ข้อความที่มีรูป/ลิงก์ที่ต้องการให้ฉันช่วยตรวจสอบที่มาด้วยนะคะ"
            )
            return
        dec_plan = await orchestrate_message_with_decision_1(
            question.strip() or ctx.message.content,
            extracted_ctx=ctx_text,
            purpose="SourceCmd",
            channel_id=ctx.channel.id,
        )
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
        answer, did_search = await query_local_ollama_chat(
            f"{user_q}\n\n{ctx_text}",
            model=ACTIVE_FAST_MODEL,
            system_prompt=FAST_SYSTEM_PROMPT,
            channel_id=ctx.channel.id,
            temperature=0.2,
            allow_agentic_search=True,
            has_fresh_attachment=True,
            speaker_user=ctx.author,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
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


@bot.command(name="status", aliases=["stat", "ping", "ram"])
async def status_command(ctx):
    m = _collect_health_metrics()
    d = m["discord"]
    mem = m["memory"]
    ai = m["ai_backends"]

    ping_str = f"{d['gateway_ping_ms']} ms" if d["gateway_ping_ms"] is not None else "N/A"
    active_topics_count = len(getattr(config, "channel_active_topic", {}))

    await ctx.send(
        f"❄️ **สถานะระบบ (Yukinon System Status)**\n"
        f"- **เวลาปัจจุบัน:** `{m['system_time_th']}`\n"
        f"- **Uptime:** `{m['uptime_human']}` | **Ping:** `{ping_str}`\n"
        f"- **Voice & Scribe:** `{d['active_voice_sessions']}` ห้องเสียง | `{d['active_scribe_sessions']}` จดบันทึก | TTS `{ai['tts_google_pitch']}p / {ai['tts_google_speed']}x`\n"
        f"- **Native Tools:** `{len(ai['registered_tools'])}` เครื่องมือ (`{', '.join(ai['registered_tools'])}`)\n"
        f"- **Container RAM:** `{mem['bot_ram_mb']} MB / {mem['render_limit_mb']} MB ({mem['ram_usage_pct']}%)`\n"
        f"- **Cached Contexts:** `{mem['active_chat_channels']}` แชท | "
        f"`{mem['cached_attachments']}` ไฟล์แนบ | "
        f"`{active_topics_count}` หัวข้อสนทนา | "
        f"`{mem['stored_user_profiles']}` โปรไฟล์ผู้ใช้"
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
        else f"RAM ปัจจุบัน: `{after_mb} MB / 512 MB`"
    )
    await ctx.send(
        f"🧹 ล้างประวัติการสนทนา ไฟล์แนบ และคืนค่าหน่วยความจำของ**{scope_label}**เรียบร้อยแล้วค่ะ ({ram_note})"
    )


@bot.command(name="calc")
async def calc_exp(ctx, *, expr: str):
    try:
        await ctx.send(f"Result: `{SimpleEval().eval(expr)}`")
    except Exception as e:
        await ctx.send(f"Error: `{e}`")


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