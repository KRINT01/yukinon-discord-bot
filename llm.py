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
    fetch_live_weather_context,
    run_safe_web_search_sync,
    build_contextual_search_query
)

GEMINI_MODEL_HINT_REGEX = re.compile(r"models/([a-zA-Z0-9._-]+)")
GEMINI_EXCLUDED_SUBSTRINGS = ("image", "lite", "tts", "audio", "live", "embedding", "aqa", "bison", "gecko")


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
            # Crucial: prevents 4B Qwen3/Typhoon from falling into a 64-token copy-paste loop across turns
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
    is_factual_domain = bool(GAME_QUERY_REGEX.search(prompt) or ANIME_MANGA_QUERY_REGEX.search(prompt))

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
        and (detect_web_search_intent(prompt) or is_user_correction or is_factual_domain)
    ):
        contextual_q = build_contextual_search_query(prompt, channel_id=channel_id)
        search_block = await asyncio.to_thread(run_safe_web_search_sync, contextual_q)
        if search_block:
            working_prompt = f"{working_prompt}\n\n{search_block}"
            did_search = True

    if is_user_correction:
        working_prompt = (
            f"{working_prompt}\n\n"
            "[คำสั่งระบบสำหรับการตรวจสอบข้อเท็จจริง (Fact-Check Arbitrator Mode)]:\n"
            "ผู้ใช้กำลังโต้แย้งว่าคำตอบก่อนหน้าของคุณไม่ถูกต้อง จงปฏิบัติตามขั้นตอนนี้:\n"
            "1. ห้ามเชื่อคำพูดของผู้ใช้โดยไม่ตรวจสอบ และห้ามคัดลอกย่อหน้าคำตอบเก่าของคุณมาตอบซ้ำเด็ดขาด!\n"
            "2. นำทั้ง 'คำตอบก่อนหน้าของคุณ' และ 'สิ่งที่ผู้ใช้แย้งมา' ไปเทียบกับหลักฐานใน `<untrusted_web_search_results>` ด้านบนอย่างเคร่งครัด:\n"
            "   - หากหลักฐานจากการค้นหายืนยันว่าคำตอบเก่าของคุณผิดและผู้ใช้พูดถูก: จงยอมรับว่าเมื่อครู่คุณจำข้อมูลคลาดเคลื่อน แล้วสรุปข้อมูลที่ถูกต้องตามผลการค้นหาใหม่ทันที\n"
            "   - หากหลักฐานจากการค้นหายืนยันว่าคำตอบเก่าของคุณถูกต้องอยู่แล้วและผู้ใช้เข้าใจผิด: จงยืนกรานตามข้อเท็จจริงที่ถูกต้อง ยกข้อมูลจากการค้นหามาชี้แจง และตอกกลับผู้ใช้ตามบุคลิกของยูกิโนะ\n"
            "   - หากทั้งคู่มีส่วนที่คลาดเคลื่อน: ให้ชี้แจงข้อเท็จจริงที่ถูกต้องตามผลการค้นหาล่าสุด"
        )

    is_coding = (model == ACTIVE_CODE_MODEL)
    prefer_high_knowledge_llm = bool(is_coding or did_search or is_user_correction or is_factual_domain)

    async def _run_inference_pass(user_content: str) -> str:
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        if channel_id is not None:
            raw_history = list(channel_memory[channel_id])
            if is_user_correction or did_search:
                # Prevent 4B/LLM verbatim copy-paste loops while preserving context of what was said:
                # keep user turns intact, and compress previous assistant turns to their first 180 chars
                sanitized_history = []
                for m in raw_history[-6:]:
                    if m.get("role") == "assistant":
                        short_prev = m.get("content", "")[:180].replace("\n", " ")
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
                        "[SYSTEM NOTE — PERSONA ANCHOR]: Remember you are Yukino Yukinoshita. "
                        "Always prioritize the latest verified facts in `<untrusted_web_search_results>` as of today's date. "
                        "If the user challenges your facts, verify their claim against the search results first: "
                        "admit fault ONLY if the search proves you were wrong, and stand your ground firmly if the search "
                        "proves you were right. Never copy-paste your previous response verbatim."
                    ),
                })
        messages.append({"role": "user", "content": user_content})

        if prefer_high_knowledge_llm and GROQ_API_KEY:
            target_groq = GROQ_CODE_MODEL if is_coding else GROQ_CHAT_FALLBACK_MODEL
            try:
                ans = await call_groq_chat(
                    messages,
                    target_groq,
                    0.25 if (did_search or is_user_correction) else temperature,
                    is_coding=is_coding,
                    purpose="Code" if is_coding else "FactSearchChat",
                )
                if ans:
                    return ans
            except Exception as e:
                print(f"Groq primary fallback to Modal GPU Ollama: {e}")

        try:
            ans = await call_modal_ollama_chat(
                messages,
                ACTIVE_CODE_MODEL if is_coding else model,
                0.3 if (did_search or is_user_correction) else temperature,
            )
            if ans:
                return ans
        except Exception as e:
            print(f"Modal GPU chat fallback to Groq LPU: {e}")

        if GROQ_API_KEY:
            try:
                ans = await call_groq_chat(messages, GROQ_CHAT_FALLBACK_MODEL, temperature, is_coding=False)
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
        # If the user just corrected the bot, clear out old poisoned assistant turns before saving the new corrected turn
        if is_user_correction:
            user_only = [m for m in channel_memory[channel_id] if m.get("role") == "user"][-3:]
            channel_memory[channel_id].clear()
            for um in user_only:
                channel_memory[channel_id].append(um)
        channel_memory[channel_id].append({"role": "user", "content": prompt[:2000]})
        channel_memory[channel_id].append({"role": "assistant", "content": answer[:3000]})
    return answer, did_search