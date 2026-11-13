import os
import re
import time
import asyncio
import platform

import psutil
import aiohttp
from aiohttp import web
import discord
from discord.ext import commands
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
    GEMINI_API_KEY,
    OPENCLAW_GATEWAY_URL,
    AUDIO_EXTENSIONS,
    CUSTOM_EMOJI_REGEX,
    YUKIPEDIA_INTENT_PATTERN,
    MANGA_TRANSLATE_INTENT_PATTERN,
    IMAGE_GEN_INTENT_PATTERN,
    AGENTIC_IMAGE_TAG_REGEX,
    intents,
    channel_memory,
    channel_last_attachment,
    voice_wake_word_enabled,
    state,
    davey,
    prune_stale_attachments,
    close_http_session,
)
from persona import (
    get_current_thai_datetime_str,
    try_instant_datetime_reply,
    should_reply_to_message,
    maybe_react_to_discord_message,
    FAST_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
)
from search_engine import detect_coding_intent
from llm import query_local_ollama_chat
from media_context import extract_all_context
from image_engine import (
    send_chunked_response,
    send_generated_image_to_channel,
    process_manga_translation_request,
)
from voice_engine import (
    HAS_VOICE_RECV,
    voice_recv,
    SilenceGatedVoiceSink,
    speak_in_voice_channel,
)
from agent import (
    run_autonomous_agent_loop,
    process_yukipedia_request,
)


def _format_uptime(seconds: int) -> str:
    days, rem = divmod(max(0, int(seconds)), 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m {secs}s"
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def _collect_health_metrics() -> dict:
    """Collects zero-cost in-memory telemetry and prunes expired attachment caches."""
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
    edge_cd = max(0, int(state.edge_tts_cooldown_until - now))

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
            "dave_e2ee_ready": bool(davey is not None and HAS_VOICE_RECV),
        },
        "memory": {
            "bot_ram_mb": rss_mb,
            "render_limit_mb": 512,
            "ram_usage_pct": ram_pct,
            "active_chat_channels": len(channel_memory),
            "cached_attachments": len(channel_last_attachment),
        },
        "ai_backends": {
            "modal_ollama_configured": bool(OLLAMA_BASE_URL),
            "fast_model": ACTIVE_FAST_MODEL,
            "code_model": GROQ_CODE_MODEL if GROQ_API_KEY else ACTIVE_CODE_MODEL,
            "groq_enabled": bool(GROQ_API_KEY),
            "groq_vision_model": state.active_groq_vision_model if GROQ_API_KEY else None,
            "cloudflare_enabled": bool(CF_API_TOKEN and CF_ACCOUNT_ID),
            "cloudflare_image_model": CF_IMAGE_MODEL if CF_API_TOKEN else None,
            "cloudflare_vision_model": CF_VISION_MODEL if CF_API_TOKEN else None,
            "gemini_enabled": bool(GEMINI_API_KEY),
            "gemini_active_model": state.active_gemini_model if GEMINI_API_KEY else None,
            "gemini_grounding_cooldown_sec": gemini_cd,
            "edge_tts_cooldown_sec": edge_cd,
            "openclaw_gateway_ready": bool(OPENCLAW_GATEWAY_URL),
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
    edge_cd_note = f" (Fallback Google TTS active: {ai['edge_tts_cooldown_sec']}s left)" if ai["edge_tts_cooldown_sec"] else " (Edge-TTS Primary)"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Yukinon Discord Bot - System Telemetry</title>
    </head>
    <body style="font-family: system-ui, sans-serif; max-width: 720px; margin: 36px auto; line-height: 1.6; color: #1f2937; padding: 0 16px;">
        <h2>❄️ Yukino Yukinoshita Discord Bot (Hybrid Cloud Edition)</h2>
        <p><b>System Time (TH):</b> {m['system_time_th']}</p>
        <p><b>Discord Gateway:</b> {gw_status}</p>
        <p><b>Voice & DAVE E2EE:</b> {d['active_voice_sessions']} active session(s) | DAVE Ready: {d['dave_e2ee_ready']}{edge_cd_note}</p>
        <p><b>Container RAM:</b> {mem['bot_ram_mb']} MB / {mem['render_limit_mb']} MB ({mem['ram_usage_pct']}%) |
           <b>Cached Contexts:</b> {mem['active_chat_channels']} chat(s), {mem['cached_attachments']} file(s)</p>
        <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;">
        <p><b>Autonomous Agent / OpenClaw:</b> {"Connected" if OPENCLAW_GATEWAY_URL else "Internal 8-Skill Agent Active (OpenClaw Gateway on Standby)"}</p>
        <p><b>Modal GPU Ollama:</b> {"Configured" if ai['modal_ollama_configured'] else "Not Configured"} (Fast: <code>{ai['fast_model']}</code>)</p>
        <p><b>Groq LPU Acceleration:</b> {"Enabled (Code: <code>" + GROQ_CODE_MODEL + "</code> | Vision: <code>" + str(ai['groq_vision_model']) + "</code>)" if ai['groq_enabled'] else "Disabled"}</p>
        <p><b>Cloudflare Workers AI:</b> {"Enabled (Image: <code>" + str(ai['cloudflare_image_model']) + "</code> | Vision: <code>" + str(ai['cloudflare_vision_model']) + "</code>)" if ai['cloudflare_enabled'] else "Disabled"}</p>
        <p><b>Google AI Studio:</b> {"Enabled (<code>" + str(ai['gemini_active_model']) + "</code>)" + gemini_cd_note if ai['gemini_enabled'] else "Disabled"}</p>
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


class YukinonBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_runner = None

    async def setup_hook(self):
        state.http_session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_get("/", handle_root)
        app.router.add_get("/healthz", handle_healthz)
        self.web_runner = web.AppRunner(app, access_log=None)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, "0.0.0.0", PORT)
        await site.start()
        print(f"[Render Web Service Health Server] Listening on 0.0.0.0:{PORT}", flush=True)

    async def close(self):
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
    print(f"Logged in as {bot.user} on {platform.machine()} at {get_current_thai_datetime_str()}", flush=True)


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
    clean_prompt = CUSTOM_EMOJI_REGEX.sub(r":\1:", clean_prompt).strip()

    # 0. Natural Chat "Yukipedia Mode" Trigger (Search / OCR / Translation + Critique)
    if YUKIPEDIA_INTENT_PATTERN.search(clean_prompt):
        await process_yukipedia_request(message, clean_prompt)
        return

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
        elif message.stickers:
            stk_names = ", ".join(s.name for s in message.stickers)
            user_request = f"(ผู้ใช้ส่งสติกเกอร์ `{stk_names}` มาให้ ช่วยตอบโต้หรือแสดงความเห็นต่อสติกเกอร์นี้ในสไตล์ยูกิโนะ)"
        elif extracted_ctx:
            user_request = "ช่วยตรวจสอบ สรุปเนื้อหา และบอกความรู้สึกหรือความคิดเห็นของเธอต่อสิ่งที่แนบมาให้หน่อย"
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
        await maybe_react_to_discord_message(message, raw_llm_output=answer, user_text=user_request)
        if has_audio and not use_coder:
            await speak_in_voice_channel(message.guild, answer, fallback_channel=message.channel)


@bot.command(name="join")
async def join_voice(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("ต้องเข้าไปอยู่ในห้องเสียงก่อนนะคะ ถึงจะเรียกฉันเข้าไปได้")
        return
    target_channel = ctx.author.voice.channel
    print(f"[Voice Join] Command invoked by {ctx.author} for channel '{target_channel.name}'", flush=True)
    try:
        from voice_engine import _KNOWN_VOICE_USER_IDS
        for m in target_channel.members:
            _KNOWN_VOICE_USER_IDS.add(int(m.id))
        if bot.user:
            _KNOWN_VOICE_USER_IDS.add(int(bot.user.id))

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

        # Wait 1.5s for binary DAVE MLS key exchange (Opcodes 25-30 & 22) to complete
        await asyncio.sleep(1.5)

        if HAS_VOICE_RECV and hasattr(vc, "listen"):
            vc.listen(SilenceGatedVoiceSink(ctx.guild, ctx.channel, vc=vc))

        conn = getattr(vc, "_connection", None)
        dave_sess = getattr(conn, "dave_session", None)
        print(
            f"[Voice Join] Connected to '{target_channel.name}' "
            f"(discord.py={discord.__version__}, davey={davey is not None}, "
            f"dave_session={dave_sess is not None})",
            flush=True,
        )

        await ctx.send(f"เข้ามาในห้องเสียง `{target_channel.name}` แล้วค่ะ 🎙️")
        await speak_in_voice_channel(
            ctx.guild, "เข้ามาในห้องเสียงแล้วค่ะ มีเรื่องอะไรให้ช่วยก็พูดมาได้เลยนะคะ"
        )
    except Exception as e:
        print(f"[Voice Join Error] {type(e).__name__}: {e}", flush=True)
        await ctx.send(
            f"เกิดข้อผิดพลาดขณะเชื่อมต่อห้องเสียงค่ะ (`{type(e).__name__}: {e}`) "
            "แต่ยังส่งไฟล์เสียงเข้าแชทเพื่อให้ฉันตอบกลับเป็นเสียงพูดได้ตามปกตินะคะ"
        )


@bot.command(name="leave")
async def leave_voice(ctx):
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
    uids = dave_s.get_user_ids() if (dave_s and hasattr(dave_s, "get_user_ids")) else []
    await ctx.send(
        f"🎙️ **สถานะห้องเสียง (`{vc.channel.name}`):**\n"
        f"- **DAVE E2EE Status:** `{status}` | **Ready:** `{ready}`\n"
        f"- **MLS Group User IDs:** `{uids}`\n"
        f"- **ลองพูดในไมค์ว่า:** `\"ถ้าได้ยินแล้วพิมพ์ตอบในแชท\"` หรือ `\"เทสเสียง\"` เพื่อทดสอบไมค์ได้ทันทีค่ะ"
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


@bot.command(name="agent", aliases=["claw", "auto"])
async def agent_command(ctx, *, goal: str = ""):
    if not goal.strip() and not ctx.message.attachments:
        await ctx.send(
            "หากต้องการให้ฉันทำงานแบบ **Autonomous Multi-Tool Agent** (วางแผน ค้นหา คำนวณ อ่านเว็บ และสร้างภาพต่อเนื่องกัน) "
            "กรุณาระบุภารกิจที่ต้องการด้วยนะคะ เช่น `!agent ค้นหาราคาบิตคอยน์ล่าสุดเป็นบาท คำนวณมูลค่า 0.25 BTC แล้ววาดรูปแมวนั่งนับเงิน` ค่ะ"
        )
        return
    await run_autonomous_agent_loop(ctx, goal)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)