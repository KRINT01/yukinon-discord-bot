import re
import json
import time
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
import aiohttp

from config import (
    HTTPS,
    GEMINI_API_KEY,
    GEMINI_DECISION_MODEL,
    GEMINI_HEAVY_MODEL,
    GEMMA_CHAT_MODEL,
    GEMMA_VISION_MODEL,
    CF_ACCOUNT_ID,
    CF_API_TOKEN,
    CF_VISION_MODEL,
    CF_CODE_MODEL,
    CF_CHAT_FAST_MODEL,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    GROQ_SEARCH_MODEL,
    GROQ_CHAT_FALLBACK_MODEL,
    GROQ_VISION_FALLBACK_MODEL,
    OLLAMA_BASE_URL,
    ACTIVE_CODE_MODEL,
    PROMPT_INJECTION_REGEX,
    THAI_CHAR_REGEX,
    AGENTIC_SEARCH_TAG_REGEX,
    RECENCY_INTENT_REGEX,
    USER_CORRECTION_PATTERN,
    GAME_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    PURE_DATETIME_PATTERN,
    URL_REGEX,
    ollama_lock,
    channel_memory,
    channel_active_topic,
    state,
    clean_url,
    api_log,
    get_http_session,
    get_modal_headers,
    get_cached_channel_attachments,
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
GEMINI_EXCLUDED_SUBSTRINGS = (
    "image", "lite", "tts", "audio", "live", "embedding", "aqa", "bison", "gecko", "decision", "hydrogen"
)

GROQ_EXCLUDED_SUBSTRINGS = (
    "whisper", "orpheus", "guard", "safeguard", "tts", "audio", "embed", "moderation", "playai", "allam"
)

_GROQ_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_GROQ_AVAILABLE_MODELS_CACHE: list[str] = []
_GROQ_MODELS_LAST_FETCH_TS: float = 0.0

_AI_STUDIO_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_GEMMA_AVAILABLE_MODELS_CACHE: list[str] = []
_GEMMA_MODELS_LAST_FETCH_TS: float = 0.0
_CF_GEMMA_COOLDOWN_UNTIL: float = 0.0


def _estimate_messages_tokens(messages: list) -> int:
    """
    Conservative token estimator (~2.2 chars/token for mixed Thai/Japanese/English)
    so AI Studio Gemma 4 (16K TPM) and Groq (6K-8K TPM) never exceed per-minute token caps.
    """
    total_chars = sum(len(m.get("content") or "") for m in messages)
    return int(total_chars / 2.2)


def _extract_raw_user_turn(prompt: str) -> str:
    if not prompt:
        return ""
    head = re.split(
        r"\n\n(?:===|<untrusted_web_search_results|<mentioned_users_dossier>|\[Attachment |\[PDF Notice:|\[Web extraction error:)",
        prompt,
        maxsplit=1,
    )[0]
    return head.strip()


def _compact_messages_for_token_budget(messages: list, max_est_tokens: int = 4500) -> list:
    if _estimate_messages_tokens(messages) <= max_est_tokens:
        return messages

    compacted = [dict(m) for m in messages]
    max_total_chars = int(max_est_tokens * 2.2)
    non_user_chars = sum(len(m.get("content") or "") for m in compacted[:-1])
    allowed_user_chars = max(2400, max_total_chars - non_user_chars)

    last_content = compacted[-1].get("content") or ""
    if len(last_content) > allowed_user_chars:
        keep_head = min(1800, allowed_user_chars // 3)
        keep_tail = max(800, allowed_user_chars - keep_head - 80)
        compacted[-1]["content"] = (
            f"{last_content[:keep_head]}\n\n"
            f"... [ trimmed middle context to fit model token limit ] ...\n\n"
            f"{last_content[-keep_tail:]}"
        )
    return compacted


# ==============================================================================
# GOOGLE GEMINI 4 HYDROGEN SPECIALIST (`gemini-4-hydrogen` — 5 RPM, 250K TPM, 20 RPD)
# ==============================================================================
def _can_use_gemini_hydrogen() -> bool:
    """
    Enforces `gemini-4-hydrogen`'s strict 5 RPM / 20 RPD envelope with a 2-request
    daily safety reserve (max 18 RPD) and sliding 60s RPM window so it never wastes
    quota or stalls on 429s, falling back cleanly when saturated.
    """
    if not GEMINI_API_KEY or not GEMINI_HEAVY_MODEL:
        return False
    now = time.time()
    if now < state.gemini_hydrogen_cooldown_until:
        return False

    today_th = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d")
    if state.gemini_hydrogen_rpd_date != today_th:
        state.gemini_hydrogen_rpd_date = today_th
        state.gemini_hydrogen_rpd_count = 0

    if state.gemini_hydrogen_rpd_count >= 18:
        return False

    ts_q = state.gemini_hydrogen_call_timestamps
    if len(ts_q) >= 5 and (now - ts_q[0] < 61.0):
        return False

    return True


def _record_gemini_hydrogen_call():
    now = time.time()
    today_th = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d")
    if state.gemini_hydrogen_rpd_date != today_th:
        state.gemini_hydrogen_rpd_date = today_th
        state.gemini_hydrogen_rpd_count = 0
    state.gemini_hydrogen_rpd_count += 1
    state.gemini_hydrogen_call_timestamps.append(now)


async def call_gemini_hydrogen_generate(
    payload: dict, timeout_sec: int = 75, purpose: str = "HydrogenHeavy"
) -> dict | None:
    """
    Executes a high-context multimodal or text request on `gemini-4-hydrogen` (250K TPM).
    Automatically updates cooldowns/RPD counters and returns None on any error so callers
    seamlessly fall back to Gemini Flash / Groq / Gemma 4.
    """
    if not _can_use_gemini_hydrogen():
        return None

    model_id = GEMINI_HEAVY_MODEL
    url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    work_payload = json.loads(json.dumps(payload))
    session = get_http_session()
    _record_gemini_hydrogen_call()

    for attempt in range(2):
        t0 = time.monotonic()
        try:
            async with session.post(
                url, json=work_payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0

                if status == 200:
                    api_log(
                        f"Gemini Hydrogen ({purpose}) -> {model_id} 200 OK "
                        f"({elapsed:.2f}s | RPD {state.gemini_hydrogen_rpd_count}/18)"
                    )
                    return json.loads(body_text)

                if status == 400 and "thinkingConfig" in body_text and attempt == 0:
                    work_payload.get("generationConfig", {}).pop("thinkingConfig", None)
                    continue

                if status == 429:
                    if "per day" in body_text.lower() or "rpd" in body_text.lower():
                        state.gemini_hydrogen_rpd_count = 20
                        state.gemini_hydrogen_cooldown_until = time.time() + 3600.0
                    else:
                        state.gemini_hydrogen_cooldown_until = time.time() + 75.0
                elif status in (404, 400, 500, 503):
                    state.gemini_hydrogen_cooldown_until = time.time() + 300.0

                api_log(
                    f"Gemini Hydrogen ({purpose}) -> {model_id} HTTP {status} "
                    f"({elapsed:.2f}s); falling back"
                )
                return None
        except Exception as e:
            state.gemini_hydrogen_cooldown_until = time.time() + 180.0
            api_log(
                f"Gemini Hydrogen ({purpose}) -> {model_id} failed ({type(e).__name__}); falling back"
            )
            return None

    return None


async def call_gemini_hydrogen_chat(
    messages: list,
    temperature: float = 0.25,
    max_output_tokens: int = 4096,
    purpose: str = "HydrogenChat",
) -> str:
    """
    Sends full untrimmed conversation + large document/code context (~up to 120K chars)
    to `gemini-4-hydrogen` (250K TPM), returning the clean text output or `""` on fallback.
    """
    if not _can_use_gemini_hydrogen():
        return ""

    sys_parts = []
    raw_turns = []
    for m in messages:
        role = m.get("role", "user")
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        if role == "system":
            sys_parts.append(txt)
        elif role == "assistant":
            raw_turns.append(("model", txt))
        else:
            raw_turns.append(("user", txt))

    if not raw_turns:
        return ""

    contents = []
    for r_role, r_txt in raw_turns:
        if contents and contents[-1]["role"] == r_role:
            contents[-1]["parts"][0]["text"] += f"\n\n{r_txt}"
        else:
            contents.append({"role": r_role, "parts": [{"text": r_txt[:110000]}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingBudget": 512},
        },
    }
    if sys_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}

    data = await call_gemini_hydrogen_generate(payload, timeout_sec=60, purpose=purpose)
    if not data:
        return ""

    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text_parts = [
        p.get("text", "") for p in parts
        if p.get("text") and not p.get("thought", False)
    ]
    out = "\n".join(text_parts).strip()
    out = re.sub(r"<think>[\s\S]*?</think>", "", out).strip()
    out = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", out).strip()
    return out


# ==============================================================================
# GOOGLE GEMINI DECISIONS API ORCHESTRATOR (`gemini-decision-1` — 200 RPM, 50K TPM, 10K RPD)
# ==============================================================================
async def call_gemini_decision_orchestrator(
    shared_input: str,
    questions: dict,
    timeout_sec: float = 4.0,
    purpose: str = "Orchestrator",
) -> dict:
    if not GEMINI_API_KEY or not GEMINI_DECISION_MODEL:
        return {}
    if time.time() < state.gemini_decision_cooldown_until:
        return {}

    url = clean_url(f"{HTTPS}generativelanguage.googleapis.com/v1beta/decisions")
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": GEMINI_DECISION_MODEL,
        "input": shared_input[:16000],
        "questions": questions,
    }

    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                answers = data.get("answers") or {}
                if isinstance(answers, dict) and answers:
                    api_log(f"Decision API ({purpose}) -> {GEMINI_DECISION_MODEL} 200 OK ({elapsed:.2f}s)")
                    return answers
            elif resp.status == 429:
                state.gemini_decision_cooldown_until = time.time() + 60.0
                api_log(f"Decision API ({purpose}) -> HTTP 429 (cooldown 60s)")
            else:
                err_txt = await resp.text()
                state.gemini_decision_cooldown_until = time.time() + 180.0
                api_log(f"Decision API ({purpose}) -> HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:100]}")
    except Exception as e:
        state.gemini_decision_cooldown_until = time.time() + 120.0
        api_log(f"Decision API ({purpose}) -> skipped ({type(e).__name__})")
    return {}


async def orchestrate_message_with_decision_1(
    user_text: str,
    extracted_ctx: str = "",
    has_code_attachment: bool = False,
) -> dict | None:
    clean_u = _extract_raw_user_turn(user_text)
    if not clean_u and not extracted_ctx:
        return None
    if len(clean_u) <= 65 and PURE_DATETIME_PATTERN.search(clean_u) and not extracted_ctx:
        return None

    ctx_preview = (extracted_ctx or "")[:2500]
    shared_input = (
        f"User Message:\n{clean_u or '(No text; user sent attachment/link)'}\n\n"
        f"Has Code File Attachment: {has_code_attachment}\n"
        f"Attached / Embedded Context Preview:\n{ctx_preview or '(None)'}"
    )

    questions = {
        "route": {
            "type": "choice",
            "instructions": "Select the single best execution route for this user message.",
            "criteria": {
                "code": "Programming, debugging, stack traces, Java, Maven, Spring Boot, Oracle SQL, PL/SQL, Node.js, React, Bootstrap, Docker, Kubernetes, or config/log files.",
                "weather": "Asking about weather forecast, rain, temperature, humidity, or whether to bring an umbrella.",
                "web_search": "Asking for real-time news, live prices, release dates, game/anime lore facts, or disputing/fact-checking a previous statement.",
                "chat": "General conversation, greetings, personal advice, or summarizing/translating already-provided attachments, PDFs, or social media embeds (X/Twitter, Pixiv, YouTube).",
            },
        },
        "needs_web_search": {
            "type": "noul",
            "instructions": (
                "Answer YES only if answering accurately requires searching the live web because the user message "
                "and attached/embedded context do NOT already contain the needed facts or text. "
                "Answer NO if the attached PDF, code, or X/Pixiv embed + OCR already provides the full content to summarize or translate."
            ),
        },
        "attach_buttons": {
            "type": "noul",
            "instructions": (
                "Answer YES if this is a substantive technical, coding, document, search, or media post analysis "
                "that benefits from follow-up action buttons. Answer NO for simple greetings, casual chat, date/time checks, or error notices."
            ),
        },
        "button_domain": {
            "type": "choice",
            "instructions": "Select the primary content domain for follow-up action buttons.",
            "criteria": {
                "code": "Source code, stack trace, SQL query, or DevOps/Docker/K8s config.",
                "document": "Attached PDF, Word, Excel document, web article, or YouTube video.",
                "social_art": "Embedded X/Twitter, Pixiv, Bluesky post, illustration, fanart, or anime/manga page.",
                "game": "Video games, character builds, classes, or Steam store.",
                "news": "Breaking news, stock market, gold/crypto prices, or current events.",
                "general": "Hardware/product comparison, coffee/tech specs, or general advice.",
            },
        },
    }

    answers = await call_gemini_decision_orchestrator(shared_input, questions, timeout_sec=4.0, purpose="Route")
    if not answers:
        return None

    route_ans = answers.get("route") or {}
    search_ans = answers.get("needs_web_search") or {}
    btn_ans = answers.get("attach_buttons") or {}
    domain_ans = answers.get("button_domain") or {}

    chosen_route = route_ans.get("choice") or "chat"
    route_conf = float(route_ans.get("confidence") or 0.0)
    needs_search = float(search_ans.get("noul") or 0.0) >= 0.55
    attach_btns = float(btn_ans.get("noul") or 0.0) >= 0.50
    btn_domain = domain_ans.get("choice") or "general"

    if has_code_attachment:
        chosen_route = "code"
        btn_domain = "code"
        needs_search = False

    api_log(
        f"Orchestrator Plan -> route={chosen_route} ({route_conf:.2f}), "
        f"search={needs_search}, buttons={attach_btns} ({btn_domain})"
    )
    return {
        "route": chosen_route,
        "route_confidence": route_conf,
        "needs_web_search": needs_search,
        "attach_buttons": attach_btns,
        "button_domain": btn_domain,
    }


# ==============================================================================
# GROQ & GOOGLE AI STUDIO MODEL DISCOVERY
# ==============================================================================
async def discover_available_groq_models(force_refresh: bool = False) -> list[str]:
    global _GROQ_AVAILABLE_MODELS_CACHE, _GROQ_MODELS_LAST_FETCH_TS
    if not GROQ_API_KEY:
        return []

    now = time.time()
    if (
        not force_refresh
        and _GROQ_AVAILABLE_MODELS_CACHE
        and (now - _GROQ_MODELS_LAST_FETCH_TS < 3600.0)
    ):
        return _GROQ_AVAILABLE_MODELS_CACHE

    groq_host = "api." + "groq.com"
    models_url = f"{HTTPS}{groq_host}/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        session = get_http_session()
        async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                payload = await resp.json()
                discovered = []
                for item in payload.get("data") or []:
                    if not isinstance(item, dict) or item.get("active") is False:
                        continue
                    m_id = str(item.get("id") or "").strip()
                    if not m_id or any(bad in m_id.lower() for bad in GROQ_EXCLUDED_SUBSTRINGS):
                        continue
                    discovered.append(m_id)
                if discovered:
                    _GROQ_AVAILABLE_MODELS_CACHE = discovered
                    _GROQ_MODELS_LAST_FETCH_TS = now
                    api_log(f"Groq (Models Discovery) -> {len(discovered)} active chat models: {discovered}")
                    return discovered
    except Exception as e:
        print(f"Groq model auto-discovery warning: {e}")

    return _GROQ_AVAILABLE_MODELS_CACHE


async def get_groq_search_and_chat_candidates() -> list[str]:
    live_models = await discover_available_groq_models()
    preferred_order = [
        state.active_groq_search_model,
        GROQ_SEARCH_MODEL,
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        GROQ_CHAT_FALLBACK_MODEL,
    ]
    candidates = []
    for mod in preferred_order:
        if not mod or mod == GROQ_CODE_MODEL:
            continue
        if (not live_models or mod in live_models) and mod not in candidates:
            candidates.append(mod)

    for mod in live_models:
        if mod != GROQ_CODE_MODEL and mod not in candidates:
            candidates.append(mod)

    return candidates


async def discover_active_groq_vision_models() -> list[str]:
    live_models = await discover_available_groq_models()
    preferred = [
        state.active_groq_vision_model,
        GROQ_VISION_FALLBACK_MODEL,
        "qwen/qwen3.8-27b",
    ]
    out = []
    for mod in preferred:
        if mod and (not live_models or mod in live_models) and mod not in out:
            out.append(mod)
    return out or ["qwen/qwen3.8-27b"]


async def discover_active_gemma_models(vision_required: bool = False) -> list[str]:
    global _GEMMA_AVAILABLE_MODELS_CACHE, _GEMMA_MODELS_LAST_FETCH_TS
    default_list = (
        [GEMMA_VISION_MODEL, "gemma-4-26b-a4b-it", "gemma-4-31b-it"]
        if vision_required
        else [state.active_gemma_model, GEMMA_CHAT_MODEL, "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    )
    if not GEMINI_API_KEY:
        return list(dict.fromkeys(x for x in default_list if x))

    now = time.time()
    if not _GEMMA_AVAILABLE_MODELS_CACHE or (now - _GEMMA_MODELS_LAST_FETCH_TS >= 3600.0):
        list_url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        )
        try:
            session = get_http_session()
            async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    found = []
                    for m in data.get("models") or []:
                        name = (m.get("name") or "").replace("models/", "")
                        lower_name = name.lower()
                        methods = m.get("supportedGenerationMethods") or []
                        if "generateContent" in methods and "gemma-4" in lower_name:
                            found.append(name)
                    if found:
                        _GEMMA_AVAILABLE_MODELS_CACHE = found
                        _GEMMA_MODELS_LAST_FETCH_TS = now
                        api_log(f"Google AI Studio (Gemma Discovery) -> {found}")
        except Exception:
            pass

    candidates = []
    for mod in default_list + _GEMMA_AVAILABLE_MODELS_CACHE:
        if mod and mod not in candidates:
            if not _GEMMA_AVAILABLE_MODELS_CACHE or mod in _GEMMA_AVAILABLE_MODELS_CACHE:
                candidates.append(mod)
    return candidates or list(dict.fromkeys(x for x in default_list if x))


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


# ==============================================================================
# GOOGLE AI STUDIO GEMMA 4 (WITH 4.5K TOKEN COMPACTION & FAST FAILOVER)
# ==============================================================================
async def call_gemma_ai_studio_chat(
    messages: list, temperature: float = 0.4, max_output_tokens: int = 1024, purpose: str = "GemmaChat"
) -> str:
    if not GEMINI_API_KEY:
        return ""

    safe_messages = _compact_messages_for_token_budget(messages, max_est_tokens=4500)

    sys_parts = []
    raw_turns = []
    for m in safe_messages:
        role = m.get("role", "user")
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        if role == "system":
            sys_parts.append(txt)
        elif role == "assistant":
            raw_turns.append(("model", txt))
        else:
            raw_turns.append(("user", txt))

    if not raw_turns:
        return ""

    contents = []
    for r_role, r_txt in raw_turns:
        if contents and contents[-1]["role"] == r_role:
            contents[-1]["parts"][0]["text"] += f"\n\n{r_txt}"
        else:
            contents.append({"role": r_role, "parts": [{"text": r_txt}]})

    if sys_parts:
        sys_joined = "\n\n".join(sys_parts)
        if contents[0]["role"] == "user":
            contents[0]["parts"][0]["text"] = (
                f"[System Instructions — Follow Strictly]:\n{sys_joined}\n\n"
                f"[User Message]:\n{contents[0]['parts'][0]['text']}"
            )
        else:
            contents.insert(0, {"role": "user", "parts": [{"text": sys_joined}]})

    discovered = await discover_active_gemma_models(vision_required=False)
    candidates = sorted(discovered, key=lambda x: (0 if "26b" in x else 1))
    session = get_http_session()

    for model_id in candidates:
        now = time.time()
        if now < _AI_STUDIO_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0):
            continue

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": json.loads(json.dumps(contents)),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        for attempt in range(2):
            t0 = time.monotonic()
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=26)
                ) as resp:
                    status = resp.status
                    body_text = await resp.text()
                    elapsed = time.monotonic() - t0

                    if status == 200:
                        data = json.loads(body_text)
                        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                        text_parts = [
                            p.get("text", "") for p in parts
                            if p.get("text") and not p.get("thought", False)
                        ]
                        out = "\n".join(text_parts).strip()
                        out = re.sub(r"<think>[\s\S]*?</think>", "", out).strip()
                        out = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", out).strip()
                        if out:
                            state.active_gemma_model = model_id
                            api_log(f"Gemma AI Studio ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s)")
                            return out

                    if status in (400, 500) and "thinkingConfig" in payload.get("generationConfig", {}) and attempt == 0:
                        payload["generationConfig"].pop("thinkingConfig", None)
                        continue

                    if status == 429:
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 60.0
                    elif status in (404, 500):
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 300.0
                    api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP {status} ({elapsed:.2f}s)")
                    break
            except Exception as e:
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 180.0
                api_log(f"Gemma AI Studio ({purpose}) -> {model_id} failed ({type(e).__name__})")
                if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                    return ""
                break

    return ""


async def call_gemma_ai_studio_vision(
    prompt_text: str,
    mime_type: str,
    b64_img: str,
    response_mime_type: str = None,
    timeout_sec: int = 22,
    purpose: str = "GemmaVision",
) -> str:
    if not GEMINI_API_KEY:
        return ""

    discovered = await discover_active_gemma_models(vision_required=True)
    candidates = sorted(discovered, key=lambda x: (0 if "26b" in x else 1))
    session = get_http_session()

    for model_id in candidates:
        if time.time() < _AI_STUDIO_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0):
            continue

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        gen_cfg = {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if response_mime_type:
            gen_cfg["responseMimeType"] = response_mime_type

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                    ]
                }
            ],
            "generationConfig": gen_cfg,
        }

        for attempt in range(2):
            t0 = time.monotonic()
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
                ) as resp:
                    status = resp.status
                    body_text = await resp.text()
                    elapsed = time.monotonic() - t0

                    if status == 200:
                        data = json.loads(body_text)
                        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
                        text_parts = [
                            p.get("text", "") for p in parts
                            if p.get("text") and not p.get("thought", False)
                        ]
                        out = "\n".join(text_parts).strip()
                        out = re.sub(r"<think>[\s\S]*?</think>", "", out).strip()
                        out = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", out).strip()
                        if out:
                            api_log(f"Gemma AI Studio ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s)")
                            return out

                    if status in (400, 500) and attempt == 0:
                        payload["generationConfig"].pop("thinkingConfig", None)
                        payload["generationConfig"].pop("responseMimeType", None)
                        continue

                    if status == 429:
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 60.0
                    elif status in (404, 500):
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 180.0
                    api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP {status} ({elapsed:.2f}s)")
                    break
            except Exception as e:
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 120.0
                api_log(f"Gemma AI Studio ({purpose}) -> {model_id} failed ({type(e).__name__})")
                break

    return ""


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
            "maxOutputTokens": 4096,
            "thinkingConfig": {"thinkingBudget": 512},
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


# ==============================================================================
# CLOUDFLARE WORKERS AI (GEMMA 4 + OPENAI GPT-OSS-120B / GPT-OSS-20B)
# ==============================================================================
async def call_cloudflare_gemma_chat(
    messages: list, temperature: float = 0.4, max_tokens: int = 1024, purpose: str = "CFGemmaChat"
) -> str:
    global _CF_GEMMA_COOLDOWN_UNTIL
    if not CF_ACCOUNT_ID or not CF_API_TOKEN or time.time() < _CF_GEMMA_COOLDOWN_UNTIL:
        return ""

    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    cf_messages = []
    for idx, m in enumerate(messages):
        if idx == 0 and m.get("role") == "system":
            cf_messages.append({
                "role": "system",
                "content": m.get("content", "") + "\n\n[CRITICAL: Answer directly and concisely. Do NOT output internal thinking steps.]",
            })
        else:
            cf_messages.append(m)

    payload = {
        "model": CF_VISION_MODEL,
        "messages": cf_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
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
            elif resp.status == 429:
                _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 300.0
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except Exception as e:
        _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 120.0
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
    return ""


async def call_cloudflare_gpt_oss(
    messages: list,
    cf_model: str = None,
    temperature: float = 0.25,
    is_coding: bool = False,
    purpose: str = "CFGPTOSS",
) -> str:
    target_model = cf_model or (CF_CODE_MODEL if is_coding else CF_CHAT_FAST_MODEL)
    if not CF_ACCOUNT_ID or not CF_API_TOKEN or not target_model:
        return ""
    if time.time() < state.cf_gpt_oss_cooldown_until:
        return ""

    cf_host = "api." + "cloudflare.com"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    session = get_http_session()
    req_timeout = 45 if is_coding else 24

    sys_parts = []
    convo_lines = []
    for m in messages:
        role = m.get("role", "user")
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        if role == "system":
            sys_parts.append(txt)
        elif role == "assistant":
            convo_lines.append(f"Assistant: {txt}")
        else:
            convo_lines.append(f"User: {txt}")

    responses_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/responses"
    responses_payload = {
        "model": target_model,
        "instructions": "\n\n".join(sys_parts),
        "input": "\n\n".join(convo_lines),
        "reasoning": {"effort": "high" if is_coding else "low"},
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            responses_url, headers=headers, json=responses_payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                out_txt = (data.get("output_text") or "").strip()
                if not out_txt:
                    extracted_chunks = []
                    for item in data.get("output") or []:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "message":
                            for c_part in item.get("content") or []:
                                if isinstance(c_part, dict) and c_part.get("text"):
                                    extracted_chunks.append(c_part["text"])
                        elif item.get("text"):
                            extracted_chunks.append(item["text"])
                    out_txt = "\n".join(extracted_chunks).strip()
                out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
                if out_txt:
                    api_log(f"Cloudflare Responses ({purpose}) -> {target_model} 200 OK ({elapsed:.2f}s)")
                    return out_txt
            elif resp.status == 429:
                state.cf_gpt_oss_cooldown_until = time.time() + 180.0
                api_log(f"Cloudflare Responses ({purpose}) -> {target_model} HTTP 429 (cooldown 3m)")
                return ""
    except Exception as e:
        api_log(f"Cloudflare Responses ({purpose}) -> {target_model} note ({type(e).__name__}); trying chat/completions")

    chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    chat_payload = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096 if is_coding else 2048,
    }
    t1 = time.monotonic()
    try:
        async with session.post(
            chat_url, headers=headers, json=chat_payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
        ) as resp:
            elapsed = time.monotonic() - t1
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content") or ""
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    if content:
                        api_log(f"Cloudflare Chat ({purpose}) -> {target_model} 200 OK ({elapsed:.2f}s)")
                        return content
            elif resp.status == 429:
                state.cf_gpt_oss_cooldown_until = time.time() + 180.0
            api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP {resp.status} ({elapsed:.2f}s)")
    except Exception as e:
        api_log(f"Cloudflare Chat ({purpose}) -> {target_model} failed ({type(e).__name__})")
    return ""


async def call_cloudflare_vision_chat(
    prompt_text: str, mime_type: str, b64_img: str, max_tokens: int = 2048, purpose: str = "Vision"
) -> str:
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return ""
    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CF_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Respond directly and concisely with the requested output. Do NOT output internal thinking steps.",
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=18)
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
    if not GROQ_API_KEY or not groq_model:
        return ""
    now = time.time()
    if now < _GROQ_MODEL_COOLDOWN_UNTIL.get(groq_model, 0.0):
        return ""

    live_models = await discover_available_groq_models()
    if live_models and groq_model not in live_models:
        _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = now + 3600.0
        return ""

    token_budget = 6000 if is_coding else 4800
    safe_messages = _compact_messages_for_token_budget(messages, max_est_tokens=token_budget)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    groq_payload = {
        "model": groq_model,
        "messages": safe_messages,
        "temperature": temperature,
        "max_tokens": 8192 if is_coding else 2048,
    }
    if "gpt-oss" in groq_model:
        groq_payload["reasoning_effort"] = "high" if is_coding else "low"

    groq_host = "api." + "groq.com"
    chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
    t0 = time.monotonic()
    session = get_http_session()
    req_timeout = 55 if is_coding else 28
    async with session.post(
        chat_url, headers=headers, json=groq_payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
    ) as resp:
        elapsed = time.monotonic() - t0
        if resp.status == 200:
            g_data = await resp.json()
            content = g_data["choices"][0]["message"]["content"] or ""
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if not is_coding and groq_model != GROQ_CODE_MODEL:
                state.active_groq_search_model = groq_model
            api_log(f"Groq ({purpose}) -> {groq_model} 200 OK ({elapsed:.2f}s)")
            return content
        else:
            err_txt = await resp.text()
            if resp.status == 429:
                _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 60.0
            elif resp.status in (404, 400):
                _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 3600.0
                await discover_available_groq_models(force_refresh=True)
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


# ==============================================================================
# UNIFIED MULTI-CLOUD CHAT & AGENTIC SEARCH ENGINE
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
    decision_plan: dict | None = None,
) -> tuple[str, bool]:
    raw_user_query = _extract_raw_user_turn(prompt)

    if PROMPT_INJECTION_REGEX.search(raw_user_query[:600]):
        is_thai_input = bool(THAI_CHAR_REGEX.search(raw_user_query))
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
        jb_reply = sanitize_yukino_output(jb_reply, raw_user_query, speaker_user, speaker_name)
        return jb_reply, False

    if not has_fresh_attachment:
        instant_reply = try_instant_datetime_reply(raw_user_query)
        if instant_reply:
            if channel_id is not None:
                channel_memory[channel_id].append({"role": "user", "content": raw_user_query})
                channel_memory[channel_id].append({"role": "assistant", "content": instant_reply})
            return instant_reply, False

    did_search = "<untrusted_web_search_results" in prompt
    working_prompt = prompt
    is_user_correction = bool(USER_CORRECTION_PATTERN.search(raw_user_query))
    is_factual_domain = bool(
        GAME_QUERY_REGEX.search(raw_user_query)
        or ANIME_MANGA_QUERY_REGEX.search(raw_user_query)
        or RECENCY_INTENT_REGEX.search(raw_user_query)
    )
    is_japanese_or_social_media = bool(
        re.search(r"=== (?:Embedded (?:X|Pixiv|Bluesky)|Image Inspection|Unified Multi-Image)", prompt)
        or re.search(r"[ぁ-んァ-ヶ一-龯]", prompt)
    )

    active_topic = update_and_get_channel_topic(raw_user_query, channel_id=channel_id)
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

    if not has_fresh_attachment and channel_id is not None:
        cached_ctx = get_cached_channel_attachments(channel_id)
        if cached_ctx:
            working_prompt = (
                f"{working_prompt}\n\n"
                f"[ไฟล์อ้างอิงล่าสุดในห้องแชทนี้ (เรียงตามลำดับเวลา สามารถเชื่อมโยงข้ามไฟล์ได้)]:\n{cached_ctx}"
            )

    dec_route = (decision_plan or {}).get("route")
    dec_conf = float((decision_plan or {}).get("route_confidence") or 0.0)
    dec_needs_search = (decision_plan or {}).get("needs_web_search")

    if dec_route == "weather" and dec_conf >= 0.65:
        is_weather_query = allow_agentic_search
    else:
        is_weather_query = allow_agentic_search and detect_weather_intent(raw_user_query, channel_id=channel_id)

    if is_weather_query:
        weather_block = await fetch_live_weather_context(raw_user_query, channel_id=channel_id)
        if weather_block:
            working_prompt = (
                f"{working_prompt}\n\n{weather_block}\n"
                "[หมายเหตุ: ใช้ข้อมูลพยากรณ์อากาศและแนวโน้มรายชั่วโมงด้านบนตอบโดยตรง (พกร่ม/เสื้อกันฝน) ห้ามแต่งตัวเลขหรือทิศทางลมขึ้นมาเอง]"
            )

    if dec_needs_search is not None and dec_conf >= 0.65:
        should_run_web_search = bool(dec_needs_search or is_user_correction)
    else:
        should_run_web_search = bool(
            detect_web_search_intent(raw_user_query, channel_id=channel_id) or is_user_correction or is_factual_domain
        )

    if (
        allow_agentic_search
        and not is_weather_query
        and "<untrusted_web_search_results" not in working_prompt
        and should_run_web_search
    ):
        contextual_q = build_contextual_search_query(raw_user_query, channel_id=channel_id)
        search_block = await asyncio.to_thread(run_safe_web_search_sync, contextual_q)
        if search_block:
            working_prompt = f"{working_prompt}\n\n{search_block}"
            did_search = True

    if is_user_correction:
        working_prompt = (
            f"{working_prompt}\n\n"
            f"[คำสั่งระบบสำหรับการตรวจสอบข้อเท็จจริง (Fact-Check Arbitrator Mode — หัวข้อ: {active_topic or 'ตามบริบทสนทนา'})]:\n"
            "1. ผู้ใช้กำลังโต้แย้งหรือชี้ข้อผิดพลาดในคำตอบก่อนหน้าของคุณ\n"
            "2. ตรวจสอบกับหลักฐานใน `<untrusted_web_search_results>` ด้านบนอย่างตรงไปตรงมา:\n"
            "   - หากคำตอบก่อนหน้าของคุณมีข้อมูลที่แต่งขึ้นเองหรือผิดพลาดจริง: จงยอมรับอย่างตรงไปตรงมาและแก้ไขเฉพาะจุดที่ผิดให้ถูกต้องตามหลักฐานทันที\n"
            "   - หากหลักฐานจากการค้นหายืนยันว่าสิ่งที่คุณพูดถูกต้องอยู่แล้วและผู้ใช้เข้าใจผิดเอง: จงยืนยันตามข้อเท็จจริงจากผลการค้นหา\n"
            "3. ห้ามเบี่ยงประเด็นไปอธิบายประวัติของบริษัทที่ไม่เกี่ยวข้อง และห้ามใช้คำลงท้ายว่า 'ครับ' เด็ดขาด"
        )

    is_coding = (model == ACTIVE_CODE_MODEL) or (dec_route == "code" and dec_conf >= 0.70)
    prefer_high_knowledge_llm = bool(
        is_coding or did_search or is_user_correction or is_factual_domain or is_japanese_or_social_media or bool(active_topic)
    )

    async def _run_inference_pass(user_content: str) -> str:
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        raw_history = []
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
                        "Speak naturally in first person ('ฉัน'); NEVER write a meta-heading like 'ความคิดเห็นในสไตล์ยูกิโนะ'."
                    ),
                })
        messages.append({"role": "user", "content": user_content})

        # PATH A: Coding (routes >13K-char multi-file/large stack traces to `gemini-4-hydrogen` 250K TPM first!)
        if is_coding:
            if len(user_content) > 13000:
                ans = await call_gemini_hydrogen_chat(
                    messages, temperature=temperature, max_output_tokens=4096, purpose="HeavyCode"
                )
                if ans:
                    return ans

            if GROQ_API_KEY:
                code_candidates = [GROQ_CODE_MODEL] + await get_groq_search_and_chat_candidates()
                for c_mod in code_candidates:
                    try:
                        ans = await call_groq_chat(messages, c_mod, temperature, is_coding=True, purpose="Code")
                        if ans:
                            return ans
                    except Exception as e:
                        print(f"Groq coding ({c_mod}) warning: {e}")

            ans = await call_cloudflare_gpt_oss(
                messages, cf_model=CF_CODE_MODEL, temperature=temperature, is_coding=True, purpose="Code"
            )
            if ans:
                return ans

            ans = await call_gemma_ai_studio_chat(messages, temperature=temperature, max_output_tokens=4096, purpose="Code")
            if ans:
                return ans

            ans = await call_cloudflare_gemma_chat(messages, temperature=temperature, max_tokens=4096, purpose="Code")
            if ans:
                return ans

            try:
                ans = await call_modal_ollama_chat(messages, ACTIVE_CODE_MODEL, temperature, purpose="Code")
                if ans:
                    return ans
            except Exception as e:
                print(f"Modal GPU coding error: {e}")

        # HEAVY DOCUMENT SPECIALIST (>12.5K chars: full PDFs, Word docs, Excel sheets -> `gemini-4-hydrogen` 250K TPM)
        if not is_coding and len(user_content) > 12500:
            ans = await call_gemini_hydrogen_chat(
                messages, temperature=0.25, max_output_tokens=3072, purpose="HeavyDoc"
            )
            if ans:
                return ans

        # PATH B: Search / Fact-Check / Japanese Manga & Social Media
        if prefer_high_knowledge_llm and not is_coding:
            target_temp = 0.2 if (did_search or is_user_correction) else temperature

            if is_japanese_or_social_media:
                ans = await call_gemma_ai_studio_chat(
                    messages, temperature=target_temp, max_output_tokens=1800, purpose="MangaSocialChat"
                )
                if ans:
                    return ans

                ans = await call_cloudflare_gemma_chat(
                    messages, temperature=target_temp, max_tokens=1800, purpose="MangaSocialChat"
                )
                if ans:
                    return ans

            if GROQ_API_KEY:
                groq_search_candidates = await get_groq_search_and_chat_candidates()
                for g_mod in groq_search_candidates:
                    try:
                        ans = await call_groq_chat(
                            messages, g_mod, target_temp, is_coding=False, purpose="FactSearchChat"
                        )
                        if ans:
                            return ans
                    except Exception:
                        continue

            ans = await call_cloudflare_gpt_oss(
                messages, cf_model=CF_CHAT_FAST_MODEL, temperature=target_temp, is_coding=False, purpose="FactSearchChat"
            )
            if ans:
                return ans

            if not is_japanese_or_social_media:
                ans = await call_gemma_ai_studio_chat(
                    messages, temperature=target_temp, max_output_tokens=1800, purpose="FactSearchChat"
                )
                if ans:
                    return ans

                ans = await call_cloudflare_gemma_chat(
                    messages, temperature=target_temp, max_tokens=1800, purpose="FactSearchChat"
                )
                if ans:
                    return ans

            if GEMINI_API_KEY:
                try:
                    ans = await call_gemini_chat_messages(messages, temperature=target_temp, purpose="FactSearchChat")
                    if ans:
                        return ans
                except Exception as e:
                    print(f"Gemini FactSearchChat fallback warning: {e}")

        # PATH C: Casual Chat, Weather & Voice Chat
        chat_temp = 0.25 if (did_search or is_user_correction or is_weather_query) else temperature

        ans = await call_gemma_ai_studio_chat(
            messages, temperature=chat_temp, max_output_tokens=1024, purpose="Chat"
        )
        if ans:
            return ans

        ans = await call_cloudflare_gemma_chat(
            messages, temperature=chat_temp, max_tokens=1024, purpose="Chat"
        )
        if ans:
            return ans

        if GROQ_API_KEY:
            for g_mod in await get_groq_search_and_chat_candidates():
                try:
                    ans = await call_groq_chat(messages, g_mod, chat_temp, is_coding=False, purpose="ChatFallback")
                    if ans:
                        return ans
                except Exception:
                    continue

        ans = await call_cloudflare_gpt_oss(
            messages, cf_model=CF_CHAT_FAST_MODEL, temperature=chat_temp, is_coding=False, purpose="ChatFallback"
        )
        if ans:
            return ans

        try:
            ans = await call_modal_ollama_chat(messages, model, chat_temp, purpose="Chat")
            if ans:
                return ans
        except Exception as e:
            print(f"Modal GPU chat error: {e}")

        return "ระบบประมวลผลไม่สามารถเชื่อมต่อได้ชั่วคราวค่ะ กรุณาลองใหม่อีกครั้งนะคะ"

    raw_answer = await _run_inference_pass(working_prompt)

    allow_tag_search = allow_agentic_search and not (
        dec_needs_search is False and dec_conf >= 0.65 and not is_user_correction
    )
    search_tag_match = AGENTIC_SEARCH_TAG_REGEX.search(raw_answer) if allow_tag_search else None
    if search_tag_match:
        agent_query = search_tag_match.group(1).strip()[:180]
        extra_search_block = await asyncio.to_thread(run_safe_web_search_sync, agent_query)
        if extra_search_block:
            did_search = True
            prefer_high_knowledge_llm = True
            working_prompt = (
                f"{working_prompt}\n\n{extra_search_block}\n\n"
                "(ได้รับข้อมูลค้นหาเพิ่มเติมครบถ้วนแล้ว จงตอบคำถามของผู้ใช้โดยตรงตามข้อมูลด้านบน ห้ามพิมพ์แท็ก [SEARCH: ...] หรือ [IMAGE: ...] อีกเด็ดขาด)"
            )
            raw_answer = await _run_inference_pass(working_prompt)

    from user_memory import AGENTIC_MEMO_TAG_REGEX, record_user_interaction

    memo_match = AGENTIC_MEMO_TAG_REGEX.search(raw_answer or "")
    learned_memo = memo_match.group(1).strip() if memo_match else ""

    answer = sanitize_yukino_output(raw_answer, raw_user_query, speaker_user, speaker_name)

    if not answer.strip():
        retry_raw = await _run_inference_pass(
            f"{working_prompt}\n\n(ตอบคำถามโดยตรงตามข้อมูลที่มีในบริบท ห้ามใช้แท็ก [SEARCH: ...] เด็ดขาด)"
        )
        answer = sanitize_yukino_output(retry_raw, raw_user_query, speaker_user, speaker_name)

    if speaker_user is not None:
        record_user_interaction(
            speaker_user,
            raw_user_query[:300],
            channel_id=channel_id,
            learned_memo=learned_memo,
        )

    if channel_id is not None and answer:
        if is_user_correction:
            user_only = [m for m in channel_memory[channel_id] if m.get("role") == "user"][-3:]
            channel_memory[channel_id].clear()
            for um in user_only:
                channel_memory[channel_id].append(um)

        from persona import is_owner_user
        if is_owner_user(speaker_user):
            speaker_tag = "[ผู้พูด: คู่หูชมรมบริการ (Owner)]"
        else:
            s_name = getattr(speaker_user, "display_name", "") or speaker_name or "Member"
            speaker_tag = f"[ผู้พูด: {s_name}]"

        channel_memory[channel_id].append({"role": "user", "content": f"{speaker_tag} {raw_user_query[:1800]}"})
        channel_memory[channel_id].append({"role": "assistant", "content": answer[:3000]})
    return answer, did_search