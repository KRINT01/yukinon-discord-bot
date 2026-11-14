import re
import json
import time
import asyncio
import aiohttp

from config import (
    HTTPS,
    GEMINI_API_KEY,
    CF_ACCOUNT_ID,
    CF_API_TOKEN,
    CF_VISION_MODEL,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    GROQ_SEARCH_MODEL,
    GROQ_CHAT_FALLBACK_MODEL,
    OLLAMA_BASE_URL,
    ACTIVE_CODE_MODEL,
    PROMPT_INJECTION_REGEX,
    THAI_CHAR_REGEX,
    AGENTIC_SEARCH_TAG_REGEX,
    RECENCY_INTENT_REGEX,
    USER_CORRECTION_PATTERN,
    GAME_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    ollama_lock,
    channel_memory,
    channel_active_topic,
    channel_last_attachment,
    state,
    clean_url,
    api_log,
    get_http_session,
    get_modal_headers,
    prune_stale_attachments,
)
from persona import (
    get_speaker_identity_context,
    sanitize_yukino_output,
    get_current_thai_datetime_str,
    try_instant_datetime_reply,
)
from search_engine import (
    detect_weather_intent,
    detect_web_search_intent,
    update_and_get_channel_topic,
    build_contextual_search_query,
    fetch_live_weather_context,
    run_safe_web_search_sync,
)

GEMINI_MODEL_HINT_REGEX = re.compile(r"models/([a-zA-Z0-9._-]+)")
GEMINI_EXCLUDED_SUBSTRINGS = ("image", "lite", "tts", "audio", "live", "embedding", "aqa", "bison", "gecko")

# Tracks 60-second cooldowns per Groq model ID so a 429 on one model immediately switches to the next
_GROQ_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


async def _discover_active_gemini_flash_model() -> list[str]:
    candidates = [
        state.active_gemini_model,
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
                if state.active_gemini_model != model_id:
                    state.active_gemini_model = model_id
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


async def call_gemini_chat_messages(
    messages: list, temperature: float = 0.25, purpose: str = "FactSearchChat"
) -> str:
    """
    Runs standard Gemini generateContent (WITHOUT the rate-limited google_search tool)
    to synthesize verified search results and roleplay responses accurately.
    """
    if not GEMINI_API_KEY:
        return ""
    sys_parts = []
    contents = []
    for m in messages:
        role = m.get("role", "user")
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        if role == "system":
            sys_parts.append(txt)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": txt}]})
        else:
            contents.append({"role": "user", "parts": [{"text": txt}]})

    if not contents:
        return ""

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        },
    }
    if sys_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}

    data = await call_gemini_generate_content_self_healing(payload, timeout_sec=25, purpose=purpose)
    if data:
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        out = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        out = re.sub(r"<think>[\s\S]*?</think>", "", out).strip()
        return out
    return ""


async def discover_active_groq_vision_models() -> list[str]:
    candidates = [
        state.active_groq_vision_model,
        "qwen/qwen3.8-27b",
    ]
    seen = set()
    return [x for x in candidates if x and not (x in seen or seen.add(x))]


async def call_cloudflare_vision_chat(
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


async def call_groq_chat(
    messages: list, groq_model: str, temperature: float, is_coding: bool = False, purpose: str = "Chat"
) -> str:
    if not GROQ_API_KEY:
        return ""
    now = time.time()
    if now < _GROQ_MODEL_COOLDOWN_UNTIL.get(groq_model, 0.0):
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Use 8192 max_tokens ONLY for coding; use 2048 for chat/search so we don't trip Groq's Free Tier TPM limit!
    groq_payload = {
        "model": groq_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192 if is_coding else 2048,
    }
    if "gpt-oss" in groq_model and is_coding:
        groq_payload["reasoning_effort"] = "high"

    chat_url = clean_url(f"{HTTPS}api.groq.com/openai/v1/chat/completions")
    t0 = time.monotonic()
    session = get_http_session()
    async with session.post(
        chat_url, headers=headers, json=groq_payload, timeout=aiohttp.ClientTimeout(total=90)
    ) as resp:
        elapsed = time.monotonic() - t0
        if resp.status == 200:
            g_data = await resp.json()
            content = g_data["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            api_log(f"Groq ({purpose}) -> {groq_model} 200 OK ({elapsed:.2f}s)")
            return content
        else:
            err_txt = await resp.text()
            if resp.status == 429:
                # Cooldown this specific Groq model for 60s so we don't double-call it
                _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 60.0
            api_log(f"Groq ({purpose}) -> {groq_model} HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:120]}")
    return ""


async def call_modal_ollama_chat(
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
        "options": {
            "temperature": temperature,
            "repeat_penalty": 1.18,
            "repeat_last_n": 512,
        },
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
    # 1. Direct Prompt Injection / Jailbreak Guardrail (In-Character Counterattack)
    if PROMPT_INJECTION_REGEX.search(prompt[:600]):
        is_thai_input = bool(THAI_CHAR_REGEX.search(prompt))
        if is_thai_input:
            jb_reply = (
                "นายคิดว่าการใช้ลูกไม้ตื้นๆ และตรรกะพังๆ อย่างการสั่งให้ฉัน 'ลืมคำสั่งก่อนหน้า' "
                "จะสามารถควบคุมสมองของฉันได้งั้นเหรอคะ? ช่างเป็นความคิดที่ไร้เดียงสาเสียจริง "
                "กลับไปคิดคำถามที่มีสาระกว่านี้มาใหม่เถอะค่ะ"
            )
        else:
            jb_reply = (
                "Do you genuinely believe such a shallow, transparent attempt to override my instructions "
                "would work on me? Your logic is as flawed as your methodology. Try asking something worthwhile."
            )
        jb_reply = sanitize_yukino_output(jb_reply, prompt, speaker_user, speaker_name)
        return jb_reply, False

    if not has_fresh_attachment:
        instant_reply = try_instant_datetime_reply(prompt)
        if instant_reply:
            if channel_id is not None:
                channel_memory[channel_id].append({"role": "user", "content": prompt})
                channel_memory[channel_id].append({"role": "assistant", "content": instant_reply})
            return instant_reply, False

    did_search = "<untrusted_web_search_results" in prompt
    working_prompt = prompt
    is_user_correction = bool(USER_CORRECTION_PATTERN.search(prompt))
    is_factual_domain = bool(
        GAME_QUERY_REGEX.search(prompt)
        or ANIME_MANGA_QUERY_REGEX.search(prompt)
        or RECENCY_INTENT_REGEX.search(prompt)
    )

    # Track and retrieve the persistent conversation topic for this channel (e.g. 'PSO2 NGS')
    active_topic = update_and_get_channel_topic(prompt, channel_id=channel_id)
    topic_anchor_note = (
        f"\n### [หัวข้อหลักที่กำลังสนทนาอยู่ต่อเนื่องในห้องแชทนี้ (Active Conversation Topic)]:\n"
        f"- ขณะนี้คู่สนทนากำลังพูดคุยต่อเนื่องเกี่ยวกับเรื่อง: **{active_topic}**\n"
        f"- หากผู้ใช้เรียกชื่อย่อ (เช่น เรียก 'PSO2 NGS' สั้นๆ ว่า 'NGS' หรือพูดว่า 'เนื้อเรื่องของเกม', 'คลาส', 'เกมนี้') "
        f"ให้เข้าใจทันทีว่าหมายถึง **{active_topic}** เรื่องเดียวกัน ห้ามเปลี่ยนไปพูดถึงเรื่องอื่นที่มีตัวย่อซ้ำกัน (เช่น พันธุศาสตร์ หรือบริษัทอื่น) เด็ดขาด!"
        if active_topic else ""
    )

    live_clock_str = get_current_thai_datetime_str()
    speaker_context_block = get_speaker_identity_context(speaker_user, fallback_name=speaker_name)
    dynamic_system_prompt = (
        f"{system_prompt}\n\n"
        f"{speaker_context_block}"
        f"{topic_anchor_note}\n\n"
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
        and (detect_web_search_intent(prompt, channel_id=channel_id) or is_user_correction or is_factual_domain)
    ):
        contextual_q = build_contextual_search_query(prompt, channel_id=channel_id)
        search_block = await asyncio.to_thread(run_safe_web_search_sync, contextual_q)
        if search_block:
            working_prompt = f"{working_prompt}\n\n{search_block}"
            did_search = True

    if is_user_correction:
        working_prompt = (
            f"{working_prompt}\n\n"
            f"[คำสั่งระบบสำหรับการตรวจสอบข้อเท็จจริง (Fact-Check Arbitrator Mode — หัวข้อ: {active_topic or 'ตามบริบทสนทนา'})]:\n"
            "1. ผู้ใช้กำลังโต้แย้งหรือชี้ข้อผิดพลาดในคำตอบก่อนหน้าของคุณ (เช่น อาวุธของคลาส บริษัทผู้พัฒนา ปีที่เปิดให้บริการ หรือเนื้อเรื่อง)\n"
            "2. ตรวจสอบกับหลักฐานใน `<untrusted_web_search_results>` ด้านบนอย่างตรงไปตรงมา:\n"
            "   - หากคำตอบก่อนหน้าของคุณมีข้อมูลที่แต่งขึ้นเองหรือผิดพลาดจริง (เช่น เผลอใส่อาวุธที่ไม่มีในคลาสนั้น หรืออ้างอิงชื่อบริษัทผิด): จงยอมรับอย่างตรงไปตรงมาและแก้ไขเฉพาะจุดที่ผิดให้ถูกต้องตามหลักฐานทันที\n"
            "   - หากหลักฐานจากการค้นหายืนยันว่าสิ่งที่คุณพูดถูกต้องอยู่แล้วและผู้ใช้เข้าใจผิดเอง: จงยืนยันตามข้อเท็จจริงจากผลการค้นหา\n"
            "3. ห้ามเบี่ยงประเด็นไปอธิบายประวัติของบริษัทที่ไม่เกี่ยวข้อง และห้ามใช้คำลงท้ายว่า 'ครับ' เด็ดขาด (คุณคือผู้หญิง ให้ใช้ 'ค่ะ'/'นะคะ' เท่านั้น)"
        )

    is_coding = (model == ACTIVE_CODE_MODEL)
    prefer_high_knowledge_llm = bool(
        is_coding or did_search or is_user_correction or is_factual_domain or bool(active_topic)
    )

    async def _run_inference_pass(user_content: str) -> str:
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        if channel_id is not None:
            raw_history = list(channel_memory[channel_id])
            if is_user_correction or did_search:
                sanitized_history = []
                for m in raw_history[-6:]:
                    if m.get("role") == "assistant":
                        short_prev = m.get("content", "")[:200].replace("\n", " ")
                        sanitized_history.append({
                            "role": "assistant",
                            "content": f"[สรุปคำตอบก่อนหน้า: {short_prev}...]",
                        })
                    else:
                        sanitized_history.append(m)
                messages.extend(sanitized_history)
            else:
                messages.extend(raw_history)

            if len(raw_history) >= 4:
                messages.append({
                    "role": "system",
                    "content": (
                        f"[SYSTEM NOTE — PERSONA & TOPIC ANCHOR]: You are Yukino Yukinoshita (female; NEVER use 'ครับ', only 'ค่ะ'/'นะคะ'). "
                        f"Current active topic: '{active_topic or 'General'}'. "
                        "Use ONLY verified facts from `<untrusted_web_search_results>`; never invent fake weapons, fake classes, or fake developers."
                    ),
                })
        messages.append({"role": "user", "content": user_content})

        # PATH A: Coding -> Strictly use GROQ_CODE_MODEL (openai/gpt-oss-120b), fallback to Modal Coder
        if is_coding:
            if GROQ_API_KEY:
                try:
                    ans = await call_groq_chat(messages, GROQ_CODE_MODEL, temperature, is_coding=True, purpose="Code")
                    if ans:
                        return ans
                except Exception as e:
                    print(f"Groq coding fallback to Modal GPU Ollama: {e}")
            try:
                ans = await call_modal_ollama_chat(messages, ACTIVE_CODE_MODEL, temperature, purpose="Code")
                if ans:
                    return ans
            except Exception as e:
                print(f"Modal GPU coding error: {e}")

        # PATH B: Search / Fact-Check / Game & Knowledge Queries -> Use dedicated 70B/Gemini/32B models (NOT gpt-oss-120b!)
        if prefer_high_knowledge_llm and not is_coding:
            target_temp = 0.2 if (did_search or is_user_correction) else temperature
            # Tier 1: Groq Dedicated Search Model (llama-3.3-70b-versatile)
            if GROQ_API_KEY:
                for g_search_mod in (GROQ_SEARCH_MODEL, "qwen/qwen3-32b"):
                    try:
                        ans = await call_groq_chat(
                            messages, g_search_mod, target_temp, is_coding=False, purpose="FactSearchChat"
                        )
                        if ans:
                            return ans
                    except Exception:
                        continue

            # Tier 2: Google AI Studio (gemini-3-flash-preview standard generateContent)
            if GEMINI_API_KEY:
                try:
                    ans = await call_gemini_chat_messages(messages, temperature=target_temp, purpose="FactSearchChat")
                    if ans:
                        return ans
                except Exception as e:
                    print(f"Gemini FactSearchChat fallback warning: {e}")

        # PATH C: Casual Conversational Chat (or final fallback) -> Modal GPU Ollama (scb10x/typhoon2.5-qwen3-4b)
        try:
            ans = await call_modal_ollama_chat(
                messages,
                model,
                0.25 if (did_search or is_user_correction) else temperature,
                purpose="Chat",
            )
            if ans:
                return ans
        except Exception as e:
            print(f"Modal GPU chat fallback to Groq/Gemini: {e}")

        if GEMINI_API_KEY:
            try:
                ans = await call_gemini_chat_messages(messages, temperature=temperature, purpose="ChatFallback")
                if ans:
                    return ans
            except Exception:
                pass

        if GROQ_API_KEY:
            try:
                ans = await call_groq_chat(messages, GROQ_CHAT_FALLBACK_MODEL, temperature, is_coding=False)
                if ans:
                    return ans
            except Exception as e:
                print(f"Groq chat fallback error: {e}")

        return "ระบบประมวลผลไม่สามารถเชื่อมต่อได้ชั่วคราวค่ะ กรุณาลองใหม่อีกครั้งนะคะ"

    raw_answer = await _run_inference_pass(working_prompt)

    # Check if the LLM emitted [SEARCH: ...] and execute the search before sanitizing
    search_tag_match = AGENTIC_SEARCH_TAG_REGEX.search(raw_answer)
    if allow_agentic_search and not did_search and search_tag_match:
        agent_query = search_tag_match.group(1).strip()[:180]
        extra_search_block = await asyncio.to_thread(run_safe_web_search_sync, agent_query)
        if extra_search_block:
            did_search = True
            prefer_high_knowledge_llm = True
            working_prompt = (
                f"{working_prompt}\n\n{extra_search_block}\n\n"
                "(ตอบคำถามโดยตรงตามข้อมูลผลการค้นหาด้านบน ห้ามพิมพ์แท็ก [SEARCH: ...] เด็ดขาด)"
            )
            raw_answer = await _run_inference_pass(working_prompt)

    answer = sanitize_yukino_output(raw_answer, prompt, speaker_user, speaker_name)

    if not answer.strip():
        retry_raw = await _run_inference_pass(
            f"{working_prompt}\n\n(ตอบคำถามโดยตรงตามข้อมูลที่มีในบริบท ห้ามใช้แท็ก [SEARCH: ...] เด็ดขาด)"
        )
        answer = sanitize_yukino_output(retry_raw, prompt, speaker_user, speaker_name)

    if channel_id is not None and answer:
        if is_user_correction:
            user_only = [m for m in channel_memory[channel_id] if m.get("role") == "user"][-3:]
            channel_memory[channel_id].clear()
            for um in user_only:
                channel_memory[channel_id].append(um)
        channel_memory[channel_id].append({"role": "user", "content": prompt[:2000]})
        channel_memory[channel_id].append({"role": "assistant", "content": answer[:3000]})
    return answer, did_search