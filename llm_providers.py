import re
import json
import time
import asyncio
import aiohttp

from config import (
    HTTPS,
    GEMINI_API_KEY,
    GEMINI_HEAVY_MODEL,
    GEMINI_SEARCH_MODEL,
    GEMINI_FALLBACK_MODEL,
    GEMINI_PREVIEW_MODEL,
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
    state,
    clean_url,
    api_log,
    get_http_session,
)
from tools import (
    GEMINI_TOOLS_BY_MODE,
    OPENAI_TOOLS_BY_MODE,
    ToolExecutionContext,
    execute_tool_calls_parallel,
)

GEMINI_MODEL_HINT_REGEX = re.compile(r"models/([a-zA-Z0-9._-]+)")
# Allow `gemini-3-flash-preview` while excluding legacy 1.5/2.0/2.5/omni/latest aliases and non-text endpoints
GEMINI_EXCLUDED_SUBSTRINGS = (
    "image", "tts", "audio", "live", "embedding", "aqa", "bison", "gecko",
    "decision", "hydrogen", "omni", "2.5", "2.0", "1.5", "latest", "3.7", "3.8"
)

GROQ_EXCLUDED_SUBSTRINGS = (
    "whisper", "orpheus", "guard", "safeguard", "tts", "audio", "embed", "moderation", "playai", "allam"
)

_GROQ_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_GROQ_AVAILABLE_MODELS_CACHE: list[str] = []
_GROQ_MODELS_LAST_FETCH_TS: float = 0.0

_GEMINI_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_AI_STUDIO_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

# Single shared Google AI Studio `/v1beta/models` catalog cache (eliminates redundant discovery HTTP calls)
_GOOGLE_ALL_MODELS_CACHE: list[str] = []
_GOOGLE_ALL_MODELS_TS: float = 0.0

_CF_GEMMA_COOLDOWN_UNTIL: float = 0.0
_CF_RESPONSES_UNSUPPORTED_UNTIL: float = 0.0


def get_google_model_cooldown(model_id: str) -> float:
    return _GEMINI_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0)


def set_google_model_cooldown(model_id: str, duration_sec: float):
    _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + max(1.0, float(duration_sec))


def _record_gemini_usage_cache(data: dict) -> str:
    """
    Safely inspects `usageMetadata.cachedContentTokenCount` (omitted by REST API when 0)
    and records implicit cache hits on `state`. Returns a compact log suffix when > 0.
    """
    usage = data.get("usageMetadata") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return ""
    cached_cnt = int(usage.get("cachedContentTokenCount") or 0)
    if cached_cnt <= 0:
        return ""
    prompt_cnt = int(usage.get("promptTokenCount") or 0)
    state.record_gemini_cache_hit(cached_cnt)
    if prompt_cnt > 0:
        return f", cache_hit={cached_cnt}/{prompt_cnt}"
    return f", cache_hit={cached_cnt}"


def _estimate_messages_tokens(messages: list) -> int:
    total_chars = sum(len(str(m.get("content") or "")) for m in messages)
    return int(total_chars / 2.2)


def _compact_messages_for_token_budget(messages: list, max_est_tokens: int = 4800) -> list:
    if _estimate_messages_tokens(messages) <= max_est_tokens:
        return messages

    compacted = [dict(m) for m in messages]
    max_total_chars = int(max_est_tokens * 2.2)
    non_user_chars = sum(len(str(m.get("content") or "")) for m in compacted[:-1])
    allowed_user_chars = max(2400, max_total_chars - non_user_chars)

    last_content = str(compacted[-1].get("content") or "")
    if len(last_content) > allowed_user_chars:
        keep_head = min(1800, allowed_user_chars // 3)
        keep_tail = max(800, allowed_user_chars - keep_head - 80)
        compacted[-1]["content"] = (
            f"{last_content[:keep_head]}\n\n"
            f"... [ trimmed middle context to fit model token limit ] ...\n\n"
            f"{last_content[-keep_tail:]}"
        )
    return compacted


def _convert_messages_to_gemini_contents(
    messages: list,
    embed_system_in_first_user: bool = False,
    max_turn_chars: int = 110000,
) -> tuple[list[str], list[dict]]:
    """
    Converts OpenAI-style chat messages to Gemini `contents` safely:
    merges consecutive same-role turns (`user`/`user` or `model`/`model`) so Gemini/Gemma
    never rejects turns after correction history pruning.
    """
    sys_parts: list[str] = []
    raw_turns: list[tuple[str, str]] = []
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
        return sys_parts, []

    contents: list[dict] = []
    for r_role, r_txt in raw_turns:
        trimmed = r_txt[:max_turn_chars]
        if contents and contents[-1]["role"] == r_role:
            contents[-1]["parts"][0]["text"] += f"\n\n{trimmed}"
        else:
            contents.append({"role": r_role, "parts": [{"text": trimmed}]})

    if embed_system_in_first_user and sys_parts:
        sys_joined = "\n\n".join(sys_parts)
        if contents[0]["role"] == "user":
            contents[0]["parts"][0]["text"] = (
                f"[System Instructions — Follow Strictly]:\n{sys_joined}\n\n"
                f"[User Message]:\n{contents[0]['parts'][0]['text']}"
            )
        else:
            contents.insert(0, {"role": "user", "parts": [{"text": sys_joined}]})

    if contents[-1]["role"] != "user":
        contents.append({"role": "user", "parts": [{"text": "Please respond to the above."}]})

    return sys_parts, contents


def _adapt_gemini_thinking_config_for_model(payload: dict, model_id: str) -> dict:
    """
    Normalizes `generationConfig` per official Google AI Studio specs using shallow copies
    so large inline base64 PDFs/images in `contents` are NEVER deep-copied on the event loop:
    1) `gemini-4-hydrogen` / `gemini-4-*`:
       - Strips `temperature`, `topP`, `topK`.
       - Raises `maxOutputTokens` to `65536`.
       - Uses `thinkingLevel` (`LOW`, `MEDIUM`, `HIGH`, `XHIGH`; rejects `MINIMAL`).
    2) `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-3.6-flash`:
       - Uses `thinkingLevel` (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`) and removes legacy `thinkingBudget`.
    3) `gemini-3.8-flash` / `gemini-3.7-flash`:
       - Uses `thinkingLevel` (`LOW`, `MEDIUM`, `HIGH`; never `MINIMAL`).
    """
    work_payload = dict(payload)
    orig_gen_cfg = work_payload.get("generationConfig")
    if not isinstance(orig_gen_cfg, dict):
        return work_payload

    gen_cfg = dict(orig_gen_cfg)
    work_payload["generationConfig"] = gen_cfg

    lower_m = (model_id or "").lower()
    is_gemini_4 = lower_m.startswith("gemini-4")
    no_minimal_models = is_gemini_4 or lower_m.startswith(("gemini-3.8", "gemini-3.7"))
    minimal_capable_models = lower_m.startswith(("gemini-3.6", "gemini-3.5", "gemini-3.1", "gemini-3-flash"))

    if is_gemini_4:
        gen_cfg.pop("temperature", None)
        gen_cfg.pop("topP", None)
        gen_cfg.pop("topK", None)
        cur_max = int(gen_cfg.get("maxOutputTokens") or 0)
        if cur_max < 65536:
            gen_cfg["maxOutputTokens"] = 65536

    orig_t_cfg = gen_cfg.get("thinkingConfig")
    if isinstance(orig_t_cfg, dict):
        t_cfg = dict(orig_t_cfg)
        gen_cfg["thinkingConfig"] = t_cfg
        if "thinkingBudget" in t_cfg:
            budget = int(t_cfg.pop("thinkingBudget") or 0)
            if "thinkingLevel" not in t_cfg:
                if no_minimal_models:
                    t_cfg["thinkingLevel"] = "LOW"
                else:
                    t_cfg["thinkingLevel"] = "MINIMAL" if budget <= 64 else "LOW"

        if no_minimal_models:
            cur_level = str(t_cfg.get("thinkingLevel") or "").upper()
            if not cur_level or cur_level == "MINIMAL":
                t_cfg["thinkingLevel"] = "LOW"
        elif minimal_capable_models:
            if "thinkingLevel" not in t_cfg:
                t_cfg["thinkingLevel"] = "MINIMAL"

    return work_payload


def _extract_gemini_text_and_calls(data: dict) -> tuple[str, list[tuple[str, dict]], list[dict]]:
    """
    Extracts `(text_output, [(fn_name, fn_args), ...], raw_model_parts)` from a Gemini API response.
    Preserves `raw_model_parts` verbatim (including any `thoughtSignature`) for multi-turn tool calls,
    and concatenates adjacent text `parts` with `""` (never `"\n"`) so multi-part responses never
    split a Thai word or combining vowel/tone mark across lines!
    """
    cand = (data.get("candidates") or [{}])[0]
    raw_parts = (cand.get("content") or {}).get("parts") or []
    text_chunks: list[str] = []
    fn_calls: list[tuple[str, dict]] = []

    for part in raw_parts:
        if not isinstance(part, dict):
            continue
        if part.get("functionCall"):
            fc = part["functionCall"]
            fn_name = str(fc.get("name") or "").strip()
            fn_args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
            if fn_name:
                fn_calls.append((fn_name, fn_args))
        elif part.get("text") and not part.get("thought", False):
            text_chunks.append(part["text"])

    out_txt = "".join(text_chunks).strip()
    out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
    out_txt = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", out_txt).strip()
    return out_txt, fn_calls, raw_parts


# ==============================================================================
# GOOGLE GEMINI 4 HYDROGEN SPECIALIST (`gemini-4-hydrogen` — 5 RPM, 250K TPM)
# ==============================================================================
def _can_use_gemini_hydrogen() -> bool:
    if not GEMINI_API_KEY or not GEMINI_HEAVY_MODEL:
        return False
    now = time.time()
    if now < state.gemini_hydrogen_cooldown_until:
        return False

    ts_q = state.gemini_hydrogen_call_timestamps
    if len(ts_q) >= 5 and (now - ts_q[0] < 61.0):
        return False

    return True


def _record_gemini_hydrogen_call():
    state.gemini_hydrogen_call_timestamps.append(time.time())


async def call_gemini_hydrogen_generate(
    payload: dict,
    timeout_sec: int = 75,
    purpose: str = "HydrogenHeavy",
    on_heavy_status=None,
) -> dict | None:
    if not _can_use_gemini_hydrogen():
        return None

    model_id = GEMINI_HEAVY_MODEL
    url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    work_payload = _adapt_gemini_thinking_config_for_model(payload, model_id)
    session = get_http_session()

    for attempt in range(2):
        t0 = time.monotonic()
        hb_task = None
        if on_heavy_status is not None:
            async def _hydrogen_slow_tick():
                try:
                    await asyncio.sleep(16.0)
                    await on_heavy_status("❄️ เนื้อหาค่อนข้างละเอียด รออีกนิดนะคะ...")
                except asyncio.CancelledError:
                    return
                except Exception:
                    return

            hb_task = asyncio.create_task(_hydrogen_slow_tick())

        try:
            async with session.post(
                url, json=work_payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0
        except Exception as e:
            state.gemini_hydrogen_cooldown_until = time.time() + 120.0
            api_log(
                f"Gemini Hydrogen ({purpose}) -> {model_id} failed ({type(e).__name__}); falling back"
            )
            return None
        finally:
            if hb_task is not None:
                hb_task.cancel()
                try:
                    await hb_task
                except Exception:
                    pass

        if status == 200:
            _record_gemini_hydrogen_call()
            state.record_provider_call("gemini_hydrogen")
            parsed = json.loads(body_text)
            cache_note = _record_gemini_usage_cache(parsed)
            api_log(f"Gemini Hydrogen ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s{cache_note})")
            return parsed

        if status == 400 and attempt == 0:
            changed = False
            if "thinkingConfig" in body_text and "thinkingConfig" in work_payload.get("generationConfig", {}):
                work_payload["generationConfig"].pop("thinkingConfig", None)
                changed = True
            if ("tools" in body_text.lower() or "function" in body_text.lower()) and "tools" in work_payload:
                work_payload.pop("tools", None)
                work_payload.pop("toolConfig", None)
                changed = True
            if changed:
                continue

        if status == 429:
            state.gemini_hydrogen_cooldown_until = time.time() + 75.0
        elif status in (404, 400, 500, 503):
            state.gemini_hydrogen_cooldown_until = time.time() + 180.0

        api_log(
            f"Gemini Hydrogen ({purpose}) -> {model_id} HTTP {status} "
            f"({elapsed:.2f}s); falling back"
        )
        return None

    return None


async def call_gemini_hydrogen_chat(
    messages: list,
    thinking_level: str = "LOW",
    purpose: str = "HydrogenChat",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
    on_heavy_status=None,
) -> str:
    if not _can_use_gemini_hydrogen():
        return ""

    sys_parts, contents = _convert_messages_to_gemini_contents(
        messages, embed_system_in_first_user=False, max_turn_chars=110000
    )
    if not contents:
        return ""

    safe_level = thinking_level.upper() if thinking_level.upper() in ("LOW", "MEDIUM", "HIGH", "XHIGH") else "LOW"
    tool_decls = GEMINI_TOOLS_BY_MODE.get(tool_ctx.mode, []) if tool_ctx is not None else []
    rounds = (max_tool_rounds + 1) if tool_decls else 1

    for r_idx in range(rounds):
        allow_tools_now = bool(tool_decls) and (r_idx < max_tool_rounds)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 65536,
                "thinkingConfig": {"thinkingLevel": safe_level},
            },
        }
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}
        if allow_tools_now:
            payload["tools"] = [{"functionDeclarations": tool_decls}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        data = await call_gemini_hydrogen_generate(
            payload,
            timeout_sec=60,
            purpose=purpose,
            on_heavy_status=(on_heavy_status if r_idx == 0 else None),
        )
        if not data:
            return ""

        out_txt, fn_calls, raw_parts = _extract_gemini_text_and_calls(data)
        if fn_calls and allow_tools_now and tool_ctx is not None:
            contents.append({"role": "model", "parts": raw_parts})
            obs_list = await execute_tool_calls_parallel(fn_calls, tool_ctx)
            fn_resp_parts = [
                {
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": obs_text},
                    }
                }
                for (fn_name, _), obs_text in zip(fn_calls, obs_list)
            ]
            contents.append({"role": "user", "parts": fn_resp_parts})
            continue

        if out_txt:
            return out_txt
        break

    return ""


# ==============================================================================
# GROQ & SHARED GOOGLE AI STUDIO MODEL DISCOVERY
# ==============================================================================
async def _fetch_google_ai_studio_model_names(force_remote: bool = False) -> list[str]:
    """Single shared 1-hour cached fetch of Google AI Studio `generateContent`-capable model IDs."""
    global _GOOGLE_ALL_MODELS_CACHE, _GOOGLE_ALL_MODELS_TS
    if not GEMINI_API_KEY:
        return []
    now = time.time()
    if not force_remote and _GOOGLE_ALL_MODELS_CACHE and (now - _GOOGLE_ALL_MODELS_TS < 3600.0):
        return _GOOGLE_ALL_MODELS_CACHE

    list_url = clean_url(
        f"{HTTPS}generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    )
    try:
        session = get_http_session()
        async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                names = []
                for m in data.get("models") or []:
                    methods = m.get("supportedGenerationMethods") or []
                    if "generateContent" in methods:
                        name = (m.get("name") or "").replace("models/", "").strip()
                        if name:
                            names.append(name)
                if names:
                    _GOOGLE_ALL_MODELS_CACHE = names
                    _GOOGLE_ALL_MODELS_TS = now
    except Exception as e:
        print(f"Google AI Studio model catalog warning: {e}")
    return _GOOGLE_ALL_MODELS_CACHE


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
    default_list = (
        [GEMMA_VISION_MODEL, "gemma-4-26b-a4b-it"]
        if vision_required
        else [state.active_gemma_model, GEMMA_CHAT_MODEL, "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    )
    if not GEMINI_API_KEY:
        return list(dict.fromkeys(x for x in default_list if x))

    all_models = await _fetch_google_ai_studio_model_names()
    gemma_live = [m for m in all_models if "gemma-4" in m.lower()]

    candidates = []
    for mod in default_list + gemma_live:
        if mod and mod not in candidates:
            if not gemma_live or mod in gemma_live:
                candidates.append(mod)
    return candidates or list(dict.fromkeys(x for x in default_list if x))


async def _discover_active_gemini_flash_model(force_remote: bool = False) -> list[str]:
    """
    Deterministic priority order for Gemini Flash models:
    1. `gemini-3.5-flash-lite`
    2. `gemini-3.1-flash-lite`
    3. `gemini-3-flash-preview`
    4. `gemini-3.6-flash`
    """
    preferred_seed = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.6-flash",
        GEMINI_SEARCH_MODEL,
        GEMINI_FALLBACK_MODEL,
        GEMINI_PREVIEW_MODEL,
    ]
    seen = set()
    deduped = [x for x in preferred_seed if x and not (x in seen or seen.add(x))]

    all_models = await _fetch_google_ai_studio_model_names(force_remote=force_remote)
    for name in all_models:
        low = name.lower()
        if "flash" in low and name not in seen and not any(bad in low for bad in GEMINI_EXCLUDED_SUBSTRINGS):
            seen.add(name)
            deduped.append(name)

    return deduped


# ==============================================================================
# GOOGLE AI STUDIO GEMMA 4 & GEMINI FLASH-LITE (SELF-HEALING + NATIVE TOOLS)
# ==============================================================================
async def call_gemma_ai_studio_chat(
    messages: list, temperature: float = 0.4, max_output_tokens: int = 1024, purpose: str = "GemmaChat"
) -> str:
    if not GEMINI_API_KEY:
        return ""

    safe_messages = _compact_messages_for_token_budget(messages, max_est_tokens=4500)
    _, contents = _convert_messages_to_gemini_contents(
        safe_messages, embed_system_in_first_user=True, max_turn_chars=18000
    )
    if not contents:
        return ""

    discovered = await discover_active_gemma_models(vision_required=False)
    candidates = sorted(discovered, key=lambda x: (0 if "26b" in x else 1))
    session = get_http_session()

    for model_id in candidates:
        if time.time() < _AI_STUDIO_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0):
            continue

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        t0 = time.monotonic()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0

                if status == 200:
                    data = json.loads(body_text)
                    out, _, _ = _extract_gemini_text_and_calls(data)
                    if out:
                        state.active_gemma_model = model_id
                        state.record_provider_call("gemma_ai_studio")
                        api_log(f"Gemma AI Studio ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s)")
                        return out

                if status == 429:
                    _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 60.0
                elif status in (404, 500, 503):
                    _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 180.0
                api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP {status} ({elapsed:.2f}s)")
        except Exception as e:
            _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 120.0
            api_log(f"Gemma AI Studio ({purpose}) -> {model_id} failed ({type(e).__name__})")
            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                return ""

    return ""


async def call_gemma_ai_studio_vision(
    prompt_text: str,
    mime_type: str,
    b64_img: str,
    response_mime_type: str = None,
    timeout_sec: int = 16,
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
                        out, _, _ = _extract_gemini_text_and_calls(data)
                        if out:
                            state.record_provider_call("gemma_ai_studio")
                            api_log(f"Gemma AI Studio ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s)")
                            return out

                    if status in (400, 500) and "responseMimeType" in payload.get("generationConfig", {}) and attempt == 0:
                        payload["generationConfig"].pop("responseMimeType", None)
                        continue

                    if status == 429:
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 60.0
                    elif status in (404, 500, 503):
                        _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 180.0
                    api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP {status} ({elapsed:.2f}s)")
                    break
            except Exception as e:
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 120.0
                api_log(f"Gemma AI Studio ({purpose}) -> {model_id} failed ({type(e).__name__})")
                break

    return ""


async def call_gemini_generate_content_self_healing(
    payload: dict,
    timeout_sec: int = 22,
    purpose: str = "Generate",
    max_models: int = 3,
    max_wall_sec: float | None = None,
    preferred_models: list[str] | None = None,
) -> dict | None:
    if not GEMINI_API_KEY:
        return None

    base_candidates = await _discover_active_gemini_flash_model(force_remote=False)
    if preferred_models:
        seen_p = set()
        candidate_models = []
        for m in list(preferred_models) + base_candidates:
            if m and m not in seen_p:
                seen_p.add(m)
                candidate_models.append(m)
    else:
        candidate_models = base_candidates

    tried = set()
    wall_start = time.monotonic()
    wall_budget = max_wall_sec if max_wall_sec is not None else float(timeout_sec * max_models + 4)

    session = get_http_session()
    idx = 0
    while idx < len(candidate_models) and len(tried) < max_models:
        elapsed_wall = time.monotonic() - wall_start
        remaining_wall = wall_budget - elapsed_wall
        if remaining_wall <= 2.0:
            break

        model_id = candidate_models[idx]
        idx += 1
        if not model_id or model_id in tried:
            continue
        if any(bad in model_id.lower() for bad in GEMINI_EXCLUDED_SUBSTRINGS):
            continue
        if time.time() < _GEMINI_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0):
            continue
        tried.add(model_id)

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        work_payload = _adapt_gemini_thinking_config_for_model(payload, model_id)
        per_call_timeout = min(float(timeout_sec), max(4.0, remaining_wall))

        for attempt in range(2):
            t0 = time.monotonic()
            try:
                async with session.post(
                    url, json=work_payload, timeout=aiohttp.ClientTimeout(total=per_call_timeout)
                ) as resp:
                    status = resp.status
                    body_text = await resp.text()
                    elapsed = time.monotonic() - t0

                    if status == 200:
                        if not preferred_models and state.active_gemini_model != model_id:
                            state.active_gemini_model = model_id
                        state.record_provider_call("gemini_flash")
                        parsed = json.loads(body_text)
                        cache_note = _record_gemini_usage_cache(parsed)
                        api_log(f"Gemini ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s{cache_note})")
                        return parsed

                    if status == 400 and attempt == 0:
                        changed = False
                        if "thinkingConfig" in body_text and "thinkingConfig" in work_payload.get("generationConfig", {}):
                            work_payload["generationConfig"].pop("thinkingConfig", None)
                            changed = True
                        if ("tools" in body_text.lower() or "function" in body_text.lower()) and "tools" in work_payload:
                            work_payload.pop("tools", None)
                            work_payload.pop("toolConfig", None)
                            changed = True
                        if changed:
                            continue

                    if status == 429:
                        _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 45.0
                    elif status in (500, 503):
                        _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 30.0
                    elif status == 404:
                        _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 3600.0

                    if status in (404, 400):
                        hint_matches = GEMINI_MODEL_HINT_REGEX.findall(body_text)
                        for hinted_model in hint_matches:
                            if (
                                hinted_model != model_id
                                and hinted_model not in tried
                                and not any(b in hinted_model.lower() for b in GEMINI_EXCLUDED_SUBSTRINGS)
                            ):
                                candidate_models.insert(idx, hinted_model)
                        api_log(f"Gemini ({purpose}) -> {model_id} HTTP {status}: {body_text[:110]}; trying next model")
                        break
                    else:
                        api_log(f"Gemini ({purpose}) -> {model_id} HTTP {status}; trying next model")
                        break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 30.0
                api_log(f"Gemini ({purpose}) -> {model_id} failed ({type(e).__name__}); trying next model")
                break

    return None


async def call_gemini_chat_messages(
    messages: list,
    temperature: float = 0.25,
    purpose: str = "FactSearchChat",
    thinking_level: str = "LOW",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
) -> str:
    """
    Gemini Flash-Lite/Preview chat with native multi-round parallel function calling
    (preserving `thoughtSignature` across turns!).
    """
    if not GEMINI_API_KEY:
        return ""
    sys_parts, contents = _convert_messages_to_gemini_contents(
        messages, embed_system_in_first_user=False, max_turn_chars=90000
    )
    if not contents:
        return ""

    tool_decls = GEMINI_TOOLS_BY_MODE.get(tool_ctx.mode, []) if tool_ctx is not None else []
    rounds = (max_tool_rounds + 1) if tool_decls else 1

    for r_idx in range(rounds):
        allow_tools_now = bool(tool_decls) and (r_idx < max_tool_rounds)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 4096,
                "thinkingConfig": {"thinkingLevel": thinking_level.upper()},
            },
        }
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}
        if allow_tools_now:
            payload["tools"] = [{"functionDeclarations": tool_decls}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        data = await call_gemini_generate_content_self_healing(
            payload,
            timeout_sec=18,
            purpose=f"{purpose}[r{r_idx+1}]" if r_idx > 0 else purpose,
            max_models=3,
            max_wall_sec=36.0,
        )
        if not data:
            return ""

        out_txt, fn_calls, raw_parts = _extract_gemini_text_and_calls(data)
        if fn_calls and allow_tools_now and tool_ctx is not None:
            contents.append({"role": "model", "parts": raw_parts})
            obs_list = await execute_tool_calls_parallel(fn_calls, tool_ctx)
            fn_resp_parts = [
                {
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": obs_text},
                    }
                }
                for (fn_name, _), obs_text in zip(fn_calls, obs_list)
            ]
            contents.append({"role": "user", "parts": fn_resp_parts})
            continue

        if out_txt:
            return out_txt
        break

    return ""


# ==============================================================================
# CLOUDFLARE WORKERS AI (GEMMA 4 + OPENAI GPT-OSS-120B / GPT-OSS-20B) + HEDGING
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
            cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=28)
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
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            elif resp.status == 429:
                _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 300.0
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 120.0
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
    return ""


async def _run_fast_groq_hedge(
    messages: list,
    temperature: float,
    purpose: str,
    tool_ctx: ToolExecutionContext | None = None,
) -> str:
    """Fast ~1.2s Groq LPU backup launched if primary Cloudflare Gemma 4 exceeds the hedge window."""
    if not GROQ_API_KEY:
        return ""
    for g_mod in await get_groq_search_and_chat_candidates():
        try:
            res = await call_groq_chat(
                messages, g_mod, temperature, is_coding=False, purpose=purpose, tool_ctx=tool_ctx
            )
            if res:
                return res
        except asyncio.CancelledError:
            raise
        except Exception:
            continue
    return ""


async def _call_cf_gemma_hedged_with_groq(
    messages: list,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    purpose: str = "Chat",
    hedge_delay_sec: float = 6.5,
) -> str:
    cf_available = bool(CF_ACCOUNT_ID and CF_API_TOKEN and time.time() >= _CF_GEMMA_COOLDOWN_UNTIL)
    if not cf_available:
        return ""

    cf_task = asyncio.create_task(
        call_cloudflare_gemma_chat(messages, temperature=temperature, max_tokens=max_tokens, purpose=purpose)
    )

    if not GROQ_API_KEY:
        return await cf_task

    done, _ = await asyncio.wait({cf_task}, timeout=hedge_delay_sec)
    if cf_task in done:
        try:
            return cf_task.result() or ""
        except Exception:
            return ""

    api_log(f"Hedge ({purpose}) -> CF >{hedge_delay_sec:.1f}s; racing Groq backup in parallel")
    groq_task = asyncio.create_task(
        _run_fast_groq_hedge(messages, temperature=temperature, purpose=f"{purpose}-Hedge")
    )
    tasks = {cf_task, groq_task}
    try:
        while tasks:
            done_set, pending_set = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for finished in done_set:
                tasks.discard(finished)
                try:
                    ans = finished.result()
                except Exception:
                    ans = ""
                if ans:
                    for p in pending_set:
                        p.cancel()
                    if pending_set:
                        await asyncio.gather(*pending_set, return_exceptions=True)
                    return ans
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return ""


async def call_cloudflare_gpt_oss(
    messages: list,
    cf_model: str = None,
    temperature: float = 0.25,
    is_coding: bool = False,
    purpose: str = "CFGPTOSS",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
) -> str:
    global _CF_RESPONSES_UNSUPPORTED_UNTIL
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
    req_timeout = 40 if is_coding else 22
    tool_schemas = OPENAI_TOOLS_BY_MODE.get(tool_ctx.mode, []) if tool_ctx is not None else []

    # Try Cloudflare `/ai/v1/responses` ONLY when not using native OpenAI tool_calls loop
    if not tool_schemas and time.time() >= _CF_RESPONSES_UNSUPPORTED_UNTIL:
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
                        out_txt = "".join(extracted_chunks).strip()
                    out_txt = re.sub(r"<think>[\s\S]*?</think>", "", out_txt).strip()
                    if out_txt:
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare Responses ({purpose}) -> {target_model} 200 OK ({elapsed:.2f}s)")
                        return out_txt
                elif resp.status == 429:
                    state.cf_gpt_oss_cooldown_until = time.time() + 180.0
                    api_log(f"Cloudflare Responses ({purpose}) -> {target_model} HTTP 429 (cooldown 3m)")
                    return ""
                elif resp.status in (400, 404):
                    _CF_RESPONSES_UNSUPPORTED_UNTIL = time.time() + 1800.0
                    api_log(f"Cloudflare Responses ({purpose}) -> HTTP {resp.status}; switching to chat/completions for 30m")
        except Exception as e:
            api_log(f"Cloudflare Responses ({purpose}) -> {target_model} note ({type(e).__name__}); trying chat/completions")

    chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    work_msgs = [dict(m) for m in messages]
    rounds = (max_tool_rounds + 1) if tool_schemas else 1
    tools_enabled = bool(tool_schemas)

    for r_idx in range(rounds):
        allow_tools_now = tools_enabled and (r_idx < max_tool_rounds)
        chat_payload: dict = {
            "model": target_model,
            "messages": work_msgs,
            "temperature": temperature,
            "max_tokens": 4096 if is_coding else 2048,
        }
        if allow_tools_now:
            chat_payload["tools"] = tool_schemas
            chat_payload["tool_choice"] = "auto"

        t1 = time.monotonic()
        try:
            async with session.post(
                chat_url, headers=headers, json=chat_payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
            ) as resp:
                elapsed = time.monotonic() - t1
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        return ""
                    msg_obj = choices[0].get("message") or {}
                    tool_calls = msg_obj.get("tool_calls") or []
                    if tool_calls and allow_tools_now and tool_ctx is not None:
                        work_msgs.append({
                            "role": "assistant",
                            "content": msg_obj.get("content") or "",
                            "tool_calls": tool_calls,
                        })
                        parsed_calls = [
                            (
                                (tc.get("function") or {}).get("name", ""),
                                (tc.get("function") or {}).get("arguments", "{}"),
                            )
                            for tc in tool_calls
                        ]
                        obs_list = await execute_tool_calls_parallel(parsed_calls, tool_ctx)
                        for tc, obs_text in zip(tool_calls, obs_list):
                            work_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id") or f"call_{r_idx}",
                                "name": (tc.get("function") or {}).get("name", ""),
                                "content": obs_text,
                            })
                        continue

                    content = msg_obj.get("content") or ""
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    if content:
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare Chat ({purpose}) -> {target_model} 200 OK ({elapsed:.2f}s)")
                        return content
                    return ""
                elif resp.status == 400 and allow_tools_now:
                    tools_enabled = False
                    api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP 400 on tools; retrying plain chat")
                    continue
                elif resp.status == 429:
                    state.cf_gpt_oss_cooldown_until = time.time() + 180.0
                    api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP 429 ({elapsed:.2f}s)")
                    return ""
                else:
                    api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP {resp.status} ({elapsed:.2f}s)")
                    return ""
        except Exception as e:
            api_log(f"Cloudflare Chat ({purpose}) -> {target_model} failed ({type(e).__name__})")
            return ""
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
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except Exception as e:
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
    return ""


# ==============================================================================
# GROQ LPU ACCELERATION (AUTO-DISCOVERY + NATIVE TOOL CALLING)
# ==============================================================================
async def call_groq_chat(
    messages: list,
    groq_model: str,
    temperature: float,
    is_coding: bool = False,
    purpose: str = "Chat",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
) -> str:
    """Groq LPU chat with native multi-round parallel OpenAI tool calling & graceful 400 fallback."""
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
    work_msgs = _compact_messages_for_token_budget(messages, max_est_tokens=token_budget)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    groq_host = "api." + "groq.com"
    chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
    session = get_http_session()
    req_timeout = 55 if is_coding else 28

    tool_schemas = OPENAI_TOOLS_BY_MODE.get(tool_ctx.mode, []) if tool_ctx is not None else []
    tools_enabled = bool(tool_schemas)
    rounds = (max_tool_rounds + 1) if tools_enabled else 1

    for r_idx in range(rounds):
        allow_tools_now = tools_enabled and (r_idx < max_tool_rounds)
        groq_payload: dict = {
            "model": groq_model,
            "messages": work_msgs,
            "temperature": temperature,
            "max_tokens": 8192 if is_coding else 2048,
        }
        if "gpt-oss" in groq_model:
            groq_payload["reasoning_effort"] = "high" if is_coding else "low"
        if allow_tools_now:
            groq_payload["tools"] = tool_schemas
            groq_payload["tool_choice"] = "auto"

        t0 = time.monotonic()
        async with session.post(
            chat_url, headers=headers, json=groq_payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                g_data = await resp.json()
                msg_obj = g_data["choices"][0]["message"] or {}
                tool_calls = msg_obj.get("tool_calls") or []

                if tool_calls and allow_tools_now and tool_ctx is not None:
                    work_msgs.append({
                        "role": "assistant",
                        "content": msg_obj.get("content") or "",
                        "tool_calls": tool_calls,
                    })
                    parsed_calls = [
                        (
                            (tc.get("function") or {}).get("name", ""),
                            (tc.get("function") or {}).get("arguments", "{}"),
                        )
                        for tc in tool_calls
                    ]
                    obs_list = await execute_tool_calls_parallel(parsed_calls, tool_ctx)
                    for tc, obs_text in zip(tool_calls, obs_list):
                        work_msgs.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id") or f"call_{r_idx}",
                            "name": (tc.get("function") or {}).get("name", ""),
                            "content": obs_text,
                        })
                    continue

                content = msg_obj.get("content") or ""
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                if not is_coding and groq_model != GROQ_CODE_MODEL:
                    state.active_groq_search_model = groq_model
                state.record_provider_call("groq")
                api_log(
                    f"Groq ({purpose}) -> {groq_model} 200 OK ({elapsed:.2f}s"
                    + (f", tools={len(tool_ctx.tools_used)}" if (tool_ctx and tool_ctx.tools_used) else "")
                    + ")"
                )
                return content
            else:
                err_txt = await resp.text()
                if resp.status == 400 and allow_tools_now and ("tool" in err_txt.lower() or "function" in err_txt.lower()):
                    tools_enabled = False
                    api_log(f"Groq ({purpose}) -> {groq_model} HTTP 400 tool_use note; retrying without tools")
                    continue
                if resp.status == 429:
                    _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 60.0
                elif resp.status in (404, 400):
                    _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 3600.0
                    await discover_available_groq_models(force_refresh=True)
                api_log(f"Groq ({purpose}) -> {groq_model} HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:120]}")
                return ""
    return ""