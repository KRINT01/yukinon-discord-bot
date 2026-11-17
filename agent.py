import re
import json
import asyncio
import aiohttp
import discord
from simpleeval import SimpleEval

from config import (
    OPENCLAW_GATEWAY_URL,
    OPENCLAW_API_KEY,
    DEFAULT_WEATHER_CITY,
    GROQ_API_KEY,
    GEMINI_API_KEY,
    ACTIVE_FAST_MODEL,
    CF_CHAT_FAST_MODEL,
    IMAGE_EXTENSIONS,
    MANGA_TRANSLATE_INTENT_PATTERN,
    agent_notes_store,
    get_http_session,
)
from persona import (
    get_current_thai_datetime_str,
    FAST_SYSTEM_PROMPT,
    YUKIPEDIA_SYSTEM_PROMPT,
)
from search_engine import (
    run_safe_web_search_sync,
    fetch_webpage_sync,
    fetch_live_weather_context,
)
from llm import (
    call_gemini_decision_orchestrator,
    call_groq_chat,
    call_cloudflare_gpt_oss,
    call_gemma_ai_studio_chat,
    call_cloudflare_gemma_chat,
    call_modal_ollama_chat,
    query_local_ollama_chat,
    get_groq_search_and_chat_candidates,
)
from media_context import (
    extract_youtube_id,
    fetch_youtube_context_sync,
    extract_all_context,
)
from image_engine import (
    _send_or_reply,
    send_chunked_response,
    send_generated_image_to_channel,
    process_manga_translation_request,
)

BASE_AGENT_SKILLS_SPEC = """
Available Tools (Skills):
1. `web_search`: Search Google/DuckDuckGo/News for real-time information, facts, prices, or current events.
   Args: {"query": "<search keywords>"}
2. `read_webpage`: Scrape and read the full text of a public URL found during search or provided by the user.
   Args: {"url": "<https://...>"}
3. `get_weather`: Fetch real-time weather and rain forecast for a city/province.
   Args: {"city_or_text": "<city or Thai province name>"}
4. `youtube_inspect`: Extract title, description, and subtitles/transcript from a YouTube video URL.
   Args: {"url": "<youtube url>"}
5. `calculate`: Evaluate a mathematical expression accurately (e.g., currency conversion, percentages, unit math).
   Args: {"expression": "<math expression like 42500 * 3.5 * 1.07>"}
6. `save_note`: Save a reminder, todo item, or key fact for this channel.
   Args: {"note": "<text to remember>"}
7. `list_notes`: Retrieve all saved notes/todos for this channel.
   Args: {}
8. `generate_image`: Queue an image generation task via Cloudflare FLUX when the user's goal includes drawing/creating a picture.
   Args: {"prompt": "<detailed description of image to generate>"}
""".strip()

AGENT_STEP_LABELS = {
    "web_search": "🔍 กำลังค้นหาข้อมูลบนเว็บ",
    "read_webpage": "📄 กำลังอ่านเนื้อหาเว็บไซต์",
    "get_weather": "🌤️ กำลังตรวจสอบสภาพอากาศ",
    "youtube_inspect": "▶️ กำลังดึงข้อมูลคลิป YouTube",
    "calculate": "🧮 กำลังคำนวณตัวเลข",
    "save_note": "📝 กำลังบันทึกข้อมูล",
    "list_notes": "🗂️ กำลังเปิดดูบันทึก",
    "generate_image": "🎨 กำลังเตรียมวาดภาพประกอบ",
    "openclaw_task": "🦞 กำลังส่งคำสั่งไปยัง OpenClaw Gateway",
}


def get_agent_skills_prompt_spec() -> str:
    """Only exposes `openclaw_task` to the LLM planner when OPENCLAW_GATEWAY_URL is configured."""
    if OPENCLAW_GATEWAY_URL:
        return (
            f"{BASE_AGENT_SKILLS_SPEC}\n"
            "9. `openclaw_task`: Delegate OS, browser, or external workflow automation to the connected OpenClaw Gateway.\n"
            '   Args: {"instruction": "<task instruction for OpenClaw>"}'
        )
    return BASE_AGENT_SKILLS_SPEC


async def execute_agent_skill(
    tool_name: str, args: dict, channel_id: int = None, pending_images: list = None
) -> str:
    """Executes a single agent skill safely and returns the observation string."""
    t_name = (tool_name or "").strip().lower()
    args = args if isinstance(args, dict) else {}

    try:
        if t_name == "web_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return "[Tool Error: Missing 'query']"
            res = await asyncio.to_thread(run_safe_web_search_sync, q, 4)
            return res or f"[No web search results found for '{q}']"

        elif t_name == "read_webpage":
            url = str(args.get("url") or "").strip()
            if not url:
                return "[Tool Error: Missing 'url']"
            text = await asyncio.to_thread(fetch_webpage_sync, url, 6000)
            return f'<webpage_content url="{url}">\n{text}\n</webpage_content>'

        elif t_name == "get_weather":
            loc = str(args.get("city_or_text") or DEFAULT_WEATHER_CITY).strip()
            res = await fetch_live_weather_context(loc)
            return res or f"[Could not fetch weather for '{loc}']"

        elif t_name == "youtube_inspect":
            url = str(args.get("url") or "").strip()
            vid = extract_youtube_id(url) or url[:11]
            res = await asyncio.to_thread(fetch_youtube_context_sync, vid, None, 10000)
            return res or f"[Could not extract YouTube video '{url}']"

        elif t_name == "calculate":
            expr = str(args.get("expression") or "").strip()
            expr_clean = re.sub(r"[^0-9+\-*/().%\s^]", "", expr).replace("^", "**")
            val = SimpleEval().eval(expr_clean)
            return f"Calculation Result ({expr_clean}) = {val}"

        elif t_name == "save_note":
            note_txt = str(args.get("note") or "").strip()
            if note_txt and channel_id is not None:
                ts = get_current_thai_datetime_str()
                agent_notes_store[channel_id].append(f"[{ts}] {note_txt}")
                return f"Saved note: {note_txt}"
            return "[Tool Error: Empty note]"

        elif t_name == "list_notes":
            notes = agent_notes_store.get(channel_id, [])
            return "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes)) if notes else "(No saved notes in this channel)"

        elif t_name == "generate_image":
            img_prompt = str(args.get("prompt") or "").strip()
            if img_prompt and pending_images is not None:
                pending_images.append(img_prompt)
                return f"[Image generation queued for prompt: '{img_prompt}'. It will be attached automatically with your final answer.]"
            return "[Tool Error: Missing 'prompt']"

        elif t_name == "openclaw_task":
            if not OPENCLAW_GATEWAY_URL:
                return "[OpenClaw Gateway is currently on standby (not configured); use internal tools instead.]"
            instruction = str(args.get("instruction") or "").strip()
            headers = {"Content-Type": "application/json"}
            if OPENCLAW_API_KEY:
                headers["Authorization"] = f"Bearer {OPENCLAW_API_KEY}"
            session = get_http_session()
            async with session.post(
                f"{OPENCLAW_GATEWAY_URL}/api/agent/exec",
                headers=headers,
                json={"prompt": instruction},
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                body = await resp.text()
                return f"OpenClaw Gateway Response (HTTP {resp.status}):\n{body[:4000]}"

        return f"[Unknown tool '{t_name}']"
    except Exception as e:
        return f"[Tool Execution Error in `{t_name}`: {type(e).__name__}: {e}]"


async def _pick_next_agent_tool_with_decision_1(
    effective_goal: str,
    extracted_ctx: str,
    observations_log: list[str],
) -> tuple[str | None, float]:
    """
    Uses `gemini-decision-1` (~200ms, 0 output tokens) to select the exact next tool
    (or `final_ready`) in the autonomous agent loop. Returns (action, confidence) or (None, 0.0).
    """
    criteria = {
        "web_search": "Search Google/DuckDuckGo/News for real-time information, prices, or current facts not yet gathered.",
        "read_webpage": "Scrape and read the full text of a specific public URL provided or found in search results.",
        "get_weather": "Fetch real-time weather and rain forecast for a city or Thai province.",
        "youtube_inspect": "Extract title, description, and subtitles/transcript from a YouTube URL.",
        "calculate": "Evaluate a math expression, unit conversion, or percentage accurately.",
        "save_note": "Save a reminder, todo item, or key fact requested by the user.",
        "list_notes": "Retrieve previously saved notes/todos for this channel.",
        "generate_image": "Generate an image/illustration via Cloudflare FLUX when requested by the goal.",
        "final_ready": "All required facts, calculations, and actions for the user's goal have been gathered and we are ready to write the final answer.",
    }
    if OPENCLAW_GATEWAY_URL:
        criteria["openclaw_task"] = "Delegate OS, browser, or external workflow automation to the connected OpenClaw Gateway."

    history_str = "\n\n".join(observations_log[-4:]) if observations_log else "(No tools executed yet)"
    shared_input = (
        f"User Goal: {effective_goal}\n\n"
        f"Attached Context Preview:\n{(extracted_ctx or '(None)')[:3000]}\n\n"
        f"Completed Tool Steps So Far:\n{history_str[:5000]}"
    )
    questions = {
        "next_action": {
            "type": "choice",
            "instructions": "Select the single best next tool to call to fulfill the User Goal, or select 'final_ready' if all necessary data/actions are complete.",
            "criteria": criteria,
        }
    }
    answers = await call_gemini_decision_orchestrator(shared_input, questions, timeout_sec=4.0, purpose="AgentStep")
    if not answers:
        return None, 0.0
    ans = answers.get("next_action") or {}
    return ans.get("choice"), float(ans.get("confidence") or 0.0)


async def _call_agent_planner_llm(messages: list) -> str:
    """
    Generates structured JSON tool action + arguments:
    1) Groq (gpt-oss-20b / qwen3.8-27b)
    2) Cloudflare Workers AI (@cf/openai/gpt-oss-20b)
    3) AI Studio Gemma 4
    4) Cloudflare Gemma 4
    5) Modal Ollama
    """
    if GROQ_API_KEY:
        for g_mod in await get_groq_search_and_chat_candidates():
            try:
                res = await call_groq_chat(
                    messages, g_mod, temperature=0.1, is_coding=False, purpose="AgentPlanner"
                )
                if res:
                    return res
            except Exception:
                continue

    res = await call_cloudflare_gpt_oss(
        messages, cf_model=CF_CHAT_FAST_MODEL, temperature=0.1, is_coding=False, purpose="AgentPlanner"
    )
    if res:
        return res

    res = await call_gemma_ai_studio_chat(messages, temperature=0.1, max_output_tokens=1024, purpose="AgentPlanner")
    if res:
        return res

    res = await call_cloudflare_gemma_chat(messages, temperature=0.1, max_tokens=1024, purpose="AgentPlanner")
    if res:
        return res

    return await call_modal_ollama_chat(messages, ACTIVE_FAST_MODEL, temperature=0.1, purpose="AgentPlanner")


async def run_autonomous_agent_loop(
    ctx_or_message,
    user_goal: str,
    max_steps: int = 4,
):
    msg = ctx_or_message.message if hasattr(ctx_or_message, "message") else ctx_or_message
    channel = ctx_or_message.channel
    author = msg.author

    status_msg = await _send_or_reply(
        channel, msg, content="❄️ **[Yukinon Autonomous Agent]** กำลังวางแผนและวิเคราะห์ลำดับงานค่ะ..."
    )

    extracted_ctx, spoken_clip_text = await extract_all_context(msg)
    effective_goal = user_goal.strip() or spoken_clip_text or "วิเคราะห์และจัดการสิ่งที่แนบมา"

    planner_system = f"""
You are the internal task-planning engine for Yukino Yukinoshita's autonomous Discord agent.
Current System Time: {get_current_thai_datetime_str()}

{get_agent_skills_prompt_spec()}

INSTRUCTIONS:
Break down the user's goal into logical steps. At each turn, output ONLY valid JSON in one of these two formats:
1) To call a tool:
{{"thought": "Brief reason for calling this tool", "action": "<tool_name>", "args": {{...}}}}
2) When you have gathered all necessary facts/calculations/actions to answer the user's goal completely:
{{"thought": "All steps complete", "action": "final_ready", "summary": "Key facts and numbers gathered"}}
""".strip()

    planner_messages = [
        {"role": "system", "content": planner_system},
        {
            "role": "user",
            "content": f"User Goal: {effective_goal}\n\nAttached Context:\n{extracted_ctx[:15000]}"
            if extracted_ctx
            else f"User Goal: {effective_goal}",
        },
    ]

    observations_log = []
    pending_images = []

    for step_idx in range(1, max_steps + 1):
        # Fast discriminative orchestrator check via `gemini-decision-1`
        decided_action, dec_conf = await _pick_next_agent_tool_with_decision_1(
            effective_goal, extracted_ctx, observations_log
        )

        if decided_action == "final_ready" and dec_conf >= 0.65:
            break

        if decided_action == "list_notes" and dec_conf >= 0.65:
            plan_obj = {
                "thought": "Retrieving saved channel notes",
                "action": "list_notes",
                "args": {},
            }
        else:
            step_messages = list(planner_messages)
            if decided_action and dec_conf >= 0.65:
                step_messages.append({
                    "role": "system",
                    "content": (
                        f'[Orchestrator Hint]: The selected tool for this step is `{decided_action}`. '
                        f'Output ONLY the JSON object with `"action": "{decided_action}"` and its exact `"args"`.'
                    ),
                })
            raw_plan = await _call_agent_planner_llm(step_messages)
            json_m = re.search(r"\{[\s\S]*\}", raw_plan or "")
            if not json_m:
                break
            try:
                plan_obj = json.loads(json_m.group(0))
            except Exception:
                break

        action = str(plan_obj.get("action") or "").strip().lower()
        thought = str(plan_obj.get("thought") or "").strip()
        args = plan_obj.get("args") or {}

        if not action or action in ("final_ready", "finish", "done", "none"):
            break

        label = AGENT_STEP_LABELS.get(action, f"⚙️ กำลังรัน `{action}`")
        try:
            await status_msg.edit(
                content=f"❄️ **[Yukinon Autonomous Agent — ขั้นตอนที่ {step_idx}/{max_steps}]** {label}..."
            )
        except Exception:
            pass

        obs = await execute_agent_skill(action, args, channel_id=channel.id, pending_images=pending_images)
        observations_log.append(
            f"--- Step {step_idx}: Tool `{action}` (Args: {json.dumps(args, ensure_ascii=False)}) ---\n"
            f"Planner Thought: {thought}\n"
            f"Tool Output:\n{obs}"
        )

        planner_messages.append({"role": "assistant", "content": json.dumps(plan_obj, ensure_ascii=False)})
        planner_messages.append({"role": "user", "content": f"Observation from `{action}`:\n{obs}"})

    try:
        await status_msg.edit(content="❄️ **[Yukinon Autonomous Agent]** รวบรวมข้อมูลเสร็จสิ้น กำลังสรุปคำตอบค่ะ...")
    except Exception:
        pass

    trace_block = "\n\n".join(observations_log)
    composed_prompt = effective_goal
    if extracted_ctx:
        composed_prompt += f"\n\n[Attached Context]:\n{extracted_ctx}"
    if trace_block:
        composed_prompt += (
            f"\n\n<agent_execution_trace>\n"
            f"ผลลัพธ์จากการใช้เครื่องมืออัตโนมัติหลายขั้นตอนของคุณ (จงใช้ข้อมูลจริงด้านล่างนี้ตอบผู้ใช้ในสไตล์ยูกิโนะชิตะ ยูกิโนะ "
            f"ราวกับคุณลงมือค้นคว้า คำนวณ และจัดการให้ด้วยตัวเอง):\n{trace_block}\n"
            f"</agent_execution_trace>"
        )

    answer, _ = await query_local_ollama_chat(
        composed_prompt,
        model=ACTIVE_FAST_MODEL,
        system_prompt=FAST_SYSTEM_PROMPT,
        channel_id=channel.id,
        temperature=0.45,
        allow_agentic_search=False,
        has_fresh_attachment=bool(extracted_ctx or trace_block),
        speaker_user=author,
    )

    try:
        await status_msg.delete()
    except Exception:
        pass

    await send_chunked_response(channel, answer, reply_to=msg)

    for img_prompt in pending_images[:2]:
        await send_generated_image_to_channel(channel, img_prompt, reply_to=msg)


async def process_yukipedia_request(ctx_or_message, raw_prompt: str = ""):
    """
    Unified 'Yukipedia Mode' dispatcher:
    1) If an image is attached/replied to AND manga translation is requested -> runs Manga Localization.
    2) If an image/document is attached/replied to -> runs OCR/Document extraction + Yukipedia proofreading/translation.
    3) Otherwise -> forces a live multi-tier Web Search + Weather lookup and answers in Yukipedia Information Broker mode.
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

    target_msg_for_ctx = msg
    if not msg.attachments and msg.reference and isinstance(msg.reference.resolved, discord.Message):
        if msg.reference.resolved.attachments:
            target_msg_for_ctx = msg.reference.resolved

    has_images = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in target_msg_for_ctx.attachments)
    if has_images and MANGA_TRANSLATE_INTENT_PATTERN.search(clean_q):
        await process_manga_translation_request(msg, clean_q)
        return

    async with channel.typing():
        extracted_ctx, spoken_clip_text = await extract_all_context(target_msg_for_ctx)
        effective_query = clean_q or spoken_clip_text

        if (
            not extracted_ctx
            and msg.reference
            and isinstance(msg.reference.resolved, discord.Message)
            and msg.reference.resolved.content
        ):
            ref_text = msg.reference.resolved.content.strip()
            extracted_ctx = f"=== ข้อความอ้างอิงที่ต้องการให้ Yukipedia ตรวจสอบ/แปล ===\n{ref_text}"

        search_block = ""
        if effective_query and not extracted_ctx:
            search_block = await asyncio.to_thread(run_safe_web_search_sync, effective_query[:180], 5)

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

        composed_prompt = effective_query
        if extracted_ctx:
            composed_prompt = f"{composed_prompt}\n\n{extracted_ctx}"
        if search_block:
            composed_prompt = f"{composed_prompt}\n\n{search_block}"

        answer, _ = await query_local_ollama_chat(
            composed_prompt,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=channel.id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=bool(extracted_ctx or search_block),
            speaker_user=author,
        )
        await send_chunked_response(channel, answer, reply_to=msg)