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
    CF_VISION_FALLBACK_MODEL,
    CF_CODE_MODEL,
    CF_CHAT_FAST_MODEL,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    GROQ_SEARCH_MODEL,
    GROQ_CHAT_FALLBACK_MODEL,
    GROQ_VISION_FALLBACK_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MANGA_MODEL,
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
_GROQ_MODELS_FAIL_UNTIL: float = 0.0  # negative cache: no discovery request until this time after a failed fetch

_GEMINI_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_AI_STUDIO_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

# Single shared Google AI Studio `/v1beta/models` catalog cache (eliminates redundant discovery HTTP calls)
_GOOGLE_ALL_MODELS_CACHE: list[str] = []
_GOOGLE_ALL_MODELS_TS: float = 0.0

_CF_GEMMA_COOLDOWN_UNTIL: float = 0.0
_CF_QWEN_VISION_COOLDOWN_UNTIL: float = 0.0
_CF_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_CF_RESPONSES_UNSUPPORTED_UNTIL: float = 0.0


def get_google_model_cooldown(model_id: str) -> float:
    return _GEMINI_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0)


def set_google_model_cooldown(model_id: str, duration_sec: float):
    _GEMINI_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + max(1.0, float(duration_sec))


def is_google_ai_studio_congested() -> bool:
    """True when all 3 primary Gemini Flash endpoints are simultaneously in cooldown (503/429/Timeout storm)."""
    now = time.time()
    return all(
        now < _GEMINI_MODEL_COOLDOWN_UNTIL.get(m, 0.0)
        for m in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview")
    )


def _get_tools_for_ctx(tool_ctx: ToolExecutionContext | None, schema_map: dict) -> list:
    """
    Returns mode-scoped tool declarations for `tool_ctx`, omitting `add_reaction`
    when `tool_ctx.message` is None (interactive button follow-ups / slash commands)
    so models never waste an extra [r2] round-trip calling a no-op reaction tool.
    """
    if tool_ctx is None:
        return []
    tools_list = schema_map.get(tool_ctx.mode, [])
    if getattr(tool_ctx, "message", None) is None and tools_list:
        return [
            t for t in tools_list
            if (t.get("name") or (t.get("function") or {}).get("name")) != "add_reaction"
        ]
    return tools_list


def _record_tool_observations_on_ctx(
    tool_ctx: ToolExecutionContext | None,
    fn_calls: list[tuple[str, object]],
    obs_list: list[str],
) -> None:
    """
    Persists successful tool observations on `tool_ctx._collected_observations` so that
    if a provider succeeds in Round 1 (executing web_search/tools) and then fails in Round 2
    with HTTP 503/429, downstream fallback providers reuse those results instead of losing them.
    """
    if tool_ctx is None:
        return
    store = getattr(tool_ctx, "_collected_observations", None)
    if store is None:
        store = []
        setattr(tool_ctx, "_collected_observations", store)
    seen = {k for k, _ in store}
    for (fn_name, fn_args), obs_text in zip(fn_calls, obs_list):
        obs_s = str(obs_text or "").strip()
        if not obs_s or obs_s.startswith(("[Tool ", "[No ", "[Could not")):
            continue
        arg_preview = str(fn_args)[:80]
        key = f"{fn_name}:{arg_preview}"
        if key not in seen:
            seen.add(key)
            store.append((key, obs_s[:7500]))
    if len(store) > 6:
        del store[:-6]


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
    """
    Compacts chat messages to fit token budgets WITHOUT sacrificing the current user turn
    to stale conversation history:
    1) Drops oldest middle turns (`compacted[1:-1]`) FIRST while preserving system + current turn.
    2) Only trims the middle of the current turn (`compacted[-1]`) if `[system, current_user]`
       alone still exceeds `max_est_tokens`.
    """
    if _estimate_messages_tokens(messages) <= max_est_tokens:
        return messages

    compacted = [dict(m) for m in messages]
    while len(compacted) > 2 and _estimate_messages_tokens(compacted) > max_est_tokens:
        compacted.pop(1)
    # Keep role alternation clean (avoid starting history with an orphaned assistant turn)
    while len(compacted) > 2 and compacted[1].get("role") == "assistant":
        compacted.pop(1)

    if _estimate_messages_tokens(compacted) <= max_est_tokens:
        return compacted

    max_total_chars = int(max_est_tokens * 2.2)
    non_user_chars = sum(len(str(m.get("content") or "")) for m in compacted[:-1])
    allowed_user_chars = max(3200, max_total_chars - non_user_chars)

    last_content = str(compacted[-1].get("content") or "")
    if len(last_content) > allowed_user_chars:
        keep_head = min(2800, int(allowed_user_chars * 0.55))
        keep_tail = max(1000, allowed_user_chars - keep_head - 80)
        compacted[-1]["content"] = (
            f"{last_content[:keep_head]}\n\n"
            f"... [ trimmed middle context to fit model token limit ] ...\n\n"
            f"{last_content[-keep_tail:]}"
        )
    return compacted


def _adapt_messages_for_oss_fallback(
    messages: list,
    model_id: str,
    has_tools: bool,
    is_coding: bool = False,
    is_voice: bool = False,
) -> list:
    """
    Adapts system instructions specifically for `openai/gpt-oss-*` models across all 3 paths:
    1) Strips `## [Native Tools ...]` when `has_tools=False` so Groq never raises HTTP 400
       ("Tool choice is none, but model called a tool").
    2) Applies a mode-aware Yukino persona lock (Coding bullet architecture vs Voice 35-word TTS
       vs General/MangaSocial natural Thai conversational paragraphs) with zero cross-mode conflicts.
    """
    low_m = (model_id or "").lower()
    if "gpt-oss" not in low_m:
        return messages

    adapted = [dict(m) for m in messages]
    if adapted and adapted[0].get("role") == "system":
        sys_txt = str(adapted[0].get("content") or "")
        if not has_tools:
            sys_txt = re.sub(
                r"## \[Native Tools & Multi-Intent Execution\][\s\S]*?(?=\n## \[|\Z)",
                "",
                sys_txt,
            )
            sys_txt = re.sub(
                r"\n- การจดจำข้อมูลผู้ใช้ระยะยาว:[^\n]+",
                "",
                sys_txt,
            )

        is_strict_no_tables = ("Strict Mode" in sys_txt) or ("ห้ามสร้างตาราง Markdown หรือตารางข้อความเด็ดขาด" in sys_txt)
        is_server_member = "SERVER MEMBER" in sys_txt
        last_user_txt = ""
        for m in reversed(adapted):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user_txt = m["content"][:500]
                break
        first_user_para = last_user_txt.split("\n\n")[0] if last_user_txt else ""
        is_english_query = bool(
            first_user_para
            and not re.search(r"[\u0E00-\u0E7F]", first_user_para)
            and len(re.findall(r"[A-Za-z]", first_user_para)) >= 12
        )
        speaker_addr_note = (
            "เรียกคู่สนทนาด้วยชื่อสมาชิกตามกฎด้านบน (ห้ามใส่ 'คุณ' นำหน้าชื่อสมาชิก Discord)"
            if is_server_member
            else "เรียกคู่สนทนาว่า 'นาย'"
        )
        lang_note = (
            "ตอบเป็นภาษาเดียวกับคำถามของผู้ใช้ (ภาษาอังกฤษที่เป็นธรรมชาติ)"
            if is_english_query
            else "ตอบเป็นภาษาไทยที่ลื่นไหล"
        )
        table_lock_note = (
            "ห้ามสร้างตาราง Markdown หรือตารางข้อความเด็ดขาดในคำตอบนี้ ให้ใช้ย่อหน้าสั้นๆ หรือหัวข้อย่อย (`-`) เท่านั้น"
            if is_strict_no_tables
            else "สร้างตาราง Markdown มาตรฐานได้เฉพาะเมื่อผู้ใช้ขอเปรียบเทียบหรือขอตาราง ไม่เกิน 4-5 คอลัมน์"
        )

        if is_coding:
            lock_note = (
                f"\n\n[CRITICAL PERSONA, LOGIC & FORMATTING LOCK]: {lang_note}ในบุคลิก 'ยูกิโนะชิตะ ยูกิโนะ' "
                f"(แทนตัวเองว่า 'ฉัน' {speaker_addr_note} ลงท้ายด้วย 'ค่ะ'/'นะคะ' เมื่อตอบไทย ห้ามใช้คำราชการแข็งทื่อ) "
                "1) ใช้เหตุผล ตรรกะ และบริบทจริงจากโค้ด/Stack Trace เสมอ ห้ามเดามั่วหรือแต่งโค้ดที่ไม่เกี่ยวข้อง "
                "2) ถ้าเป็นปัญหา/Error ที่ตรงไปตรงมา ให้ตอบสั้น กระชับ ตรงประเด็นที่สุด (ระบุ Root Cause ชัดๆ + วิธีแก้ที่ดีที่สุดเพียง 1 แนวทาง ห้ามใส่ตัวอย่างซ้ำซ้อน ห้ามเขียน Unit Test/ErrorBoundary ที่ไม่ได้ขอ และห้ามเขียนหัวข้อสรุปขั้นตอนซ้ำท้ายข้อความ) "
                "แต่ถ้าเป็นโค้ดหรือปัญหาซับซ้อน/หลายไฟล์ ให้อธิบายตรรกะ Edge Cases และแนวทางแก้ไขอย่างละเอียดครบถ้วน "
                "3) กฎเหล็กโค้ดบล็อก Discord: ต้องขึ้นบรรทัดใหม่ชิดซ้ายก่อนพิมพ์ ```lang และก่อนพิมพ์ ``` ปิดบล็อกเสมอ ห้ามพิมพ์ ``` ต่อท้ายบรรทัดหัวข้อย่อยหรือโค้ดเด็ดขาด"
            )
        elif is_voice:
            lock_note = (
                f"\n\n[CRITICAL VOICE PERSONA LOCK]: {lang_note}เป็นภาษาพูดของ 'ยูกิโนะชิตะ ยูกิโนะ' "
                f"(แทนตัวเองว่า 'ฉัน' {speaker_addr_note} ลงท้ายด้วย 'ค่ะ'/'นะคะ' เมื่อตอบไทย) "
                "สั้นกระชับที่สุดเพียง 1 ถึง 2 ประโยค (ไม่เกิน 35 คำ) สำหรับอ่านออกเสียง TTS "
                "ห้ามใส่วงเล็บ ห้ามใช้สัญลักษณ์พิเศษ (*, #, `) และห้ามใส่บล็อกโค้ด"
            )
        else:
            lock_note = (
                f"\n\n[CRITICAL PERSONA & STYLE LOCK]: {lang_note}ในบุคลิก 'ยูกิโนะชิตะ ยูกิโนะ' "
                f"(แทนตัวเองว่า 'ฉัน' {speaker_addr_note} ลงท้ายด้วย 'ค่ะ'/'นะคะ' เมื่อตอบไทย "
                "ห้ามใช้คำราชการอย่าง 'รับทราบ'/'ดำเนินการ'/'กรุณา') เขียนเป็นย่อหน้าสนทนาที่เป็นธรรมชาติเหมือนคุยกันในห้องชมรมบริการ "
                "ห้ามเขียนหัวข้อแบบรายงาน AI เช่น 'รายละเอียดเครื่องแต่งกาย:' 'จุดน่าสนใจ:' หรือ 'โดยสรุปภาพนี้...' "
                f"ห้ามแปลศัพท์เสื้อผ้า/ภาพวาดแบบแข็งทื่อตรงตัว ({table_lock_note})"
            )
        adapted[0]["content"] = sys_txt.strip() + lock_note
    return adapted


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


def _strip_model_thought_blocks(text: str) -> str:
    """Strips `<think>...</think>`, `<|thought|>...<|/thought|>`, and Gemma 4 `<|channel>thought...<channel|>` reasoning tags."""
    if not text:
        return ""
    out = re.sub(r"<think>[\s\S]*?</think>", "", text)
    out = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", out)
    out = re.sub(r"<\|channel>thought[\s\S]*?<channel\|>", "", out)
    out = re.sub(r"<\|channel>thought\s*", "", out)
    return out.strip()


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

    out_txt = _strip_model_thought_blocks("".join(text_chunks))
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
    tool_decls = _get_tools_for_ctx(tool_ctx, GEMINI_TOOLS_BY_MODE)
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
            _record_tool_observations_on_ctx(tool_ctx, fn_calls, obs_list)
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
    global _GROQ_AVAILABLE_MODELS_CACHE, _GROQ_MODELS_LAST_FETCH_TS, _GROQ_MODELS_FAIL_UNTIL
    if not GROQ_API_KEY:
        return []

    now = time.time()
    if not force_refresh and now < _GROQ_MODELS_FAIL_UNTIL:
        return _GROQ_AVAILABLE_MODELS_CACHE
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

    # Reached only on exception, non-200, or zero models: back off so every call does not re-hit /models
    _GROQ_MODELS_FAIL_UNTIL = time.time() + 120.0
    return _GROQ_AVAILABLE_MODELS_CACHE


def _is_reserved_groq_vision_model(model_id: str) -> bool:
    """Reserves `qwen/qwen3.8-27b` (and `GROQ_VISION_FALLBACK_MODEL`) exclusively for Vision fallback."""
    low = (model_id or "").strip().lower()
    if not low:
        return False
    if "qwen3.8" in low:
        return True
    if GROQ_VISION_FALLBACK_MODEL and low == GROQ_VISION_FALLBACK_MODEL.strip().lower():
        return True
    return False


async def get_groq_search_and_chat_candidates() -> list[str]:
    live_models = await discover_available_groq_models()
    preferred_order = [
        state.active_groq_search_model,
        GROQ_SEARCH_MODEL,
        "openai/gpt-oss-20b",
        GROQ_CHAT_FALLBACK_MODEL,
    ]
    candidates = []
    for mod in preferred_order:
        if not mod or mod == GROQ_CODE_MODEL or _is_reserved_groq_vision_model(mod):
            continue
        if (not live_models or mod in live_models) and mod not in candidates:
            candidates.append(mod)

    for mod in live_models:
        if mod != GROQ_CODE_MODEL and not _is_reserved_groq_vision_model(mod) and mod not in candidates:
            candidates.append(mod)

    # Drop cooling models first so they cannot occupy the capped slots (call_groq_chat would skip them anyway)
    now = time.time()
    candidates = [m for m in candidates if now >= _GROQ_MODEL_COOLDOWN_UNTIL.get(m, 0.0)]
    # Ceiling: only 3 Groq models are tried per call to bound fallback latency.
    # Upgrade path: raise the cap if Groq rate limits per model and more fallbacks are needed.
    return candidates[:3]


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
    messages: list,
    temperature: float = 0.4,
    max_output_tokens: int = 2048,
    purpose: str = "GemmaChat",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
    prefer_fast_26b: bool = False,
) -> str:
    """
    Google AI Studio Gemma 4 chat (`gemma-4-31b-it` / `gemma-4-26b-a4b-it`):
    Uses native `systemInstruction` (256K context) and native parallel function calling
    so Yukino's persona and Thai prose stay natural and consistent with Gemini 3.
    Automatically falls back to embedded system instructions or plain chat on HTTP 400.
    """
    if not GEMINI_API_KEY:
        return ""

    sys_parts, base_contents = _convert_messages_to_gemini_contents(
        messages, embed_system_in_first_user=False, max_turn_chars=60000
    )
    if not base_contents:
        return ""

    discovered = await discover_active_gemma_models(vision_required=False)
    # Prefer dense 31B for rich Yukino persona/prose; 26B MoE first only for latency-sensitive/background utility calls
    if prefer_fast_26b or purpose in ("PassiveMemoryExtract", "MemoryConsolidate", "VoiceChat"):
        candidates = sorted(discovered, key=lambda x: (0 if "26b" in x else 1))
    else:
        candidates = sorted(discovered, key=lambda x: (0 if "31b" in x else (1 if "26b" in x else 2)))

    tool_decls = _get_tools_for_ctx(tool_ctx, GEMINI_TOOLS_BY_MODE)
    session = get_http_session()

    for model_id in candidates:
        if time.time() < _AI_STUDIO_MODEL_COOLDOWN_UNTIL.get(model_id, 0.0):
            continue

        url = clean_url(
            f"{HTTPS}generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent?key={GEMINI_API_KEY}"
        )
        contents = [dict(c, parts=list(c.get("parts") or [])) for c in base_contents]
        use_native_sys = True
        use_tools = bool(tool_decls)
        rounds = (max_tool_rounds + 1) if use_tools else 1
        r_idx = 0
        retry_400_budget = 2

        while r_idx < rounds:
            allow_tools_now = use_tools and (r_idx < max_tool_rounds)
            payload: dict = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            }
            if use_native_sys and sys_parts:
                payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}
            if allow_tools_now:
                payload["tools"] = [{"functionDeclarations": tool_decls}]
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

            t0 = time.monotonic()
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=24)
                ) as resp:
                    status = resp.status
                    body_text = await resp.text()
                    elapsed = time.monotonic() - t0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 120.0
                api_log(f"Gemma AI Studio ({purpose}) -> {model_id} failed ({type(e).__name__})")
                break

            if status == 200:
                data = json.loads(body_text)
                out, fn_calls, raw_parts = _extract_gemini_text_and_calls(data)
                if fn_calls and allow_tools_now and tool_ctx is not None:
                    contents.append({"role": "model", "parts": raw_parts})
                    obs_list = await execute_tool_calls_parallel(fn_calls, tool_ctx)
                    _record_tool_observations_on_ctx(tool_ctx, fn_calls, obs_list)
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
                    r_idx += 1
                    continue

                if out:
                    state.active_gemma_model = model_id
                    state.record_provider_call("gemma_ai_studio")
                    cache_note = _record_gemini_usage_cache(data)
                    api_log(
                        f"Gemma AI Studio ({purpose}) -> {model_id} 200 OK ({elapsed:.2f}s{cache_note}"
                        + (f", tools={len(tool_ctx.tools_used)}" if (tool_ctx and tool_ctx.tools_used) else "")
                        + ")"
                    )
                    return out
                break

            if status == 400 and retry_400_budget > 0:
                retry_400_budget -= 1
                low_err = body_text.lower()
                changed = False
                if use_tools and ("tool" in low_err or "function" in low_err):
                    use_tools = False
                    rounds = 1
                    changed = True
                if use_native_sys and ("system_instruction" in low_err or "systeminstruction" in low_err or "developer instruction" in low_err):
                    use_native_sys = False
                    _, contents = _convert_messages_to_gemini_contents(
                        messages, embed_system_in_first_user=True, max_turn_chars=36000
                    )
                    changed = True
                if changed:
                    api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP 400; adapting payload and retrying")
                    continue

            if status == 429:
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 60.0
            elif status in (404, 500, 503):
                _AI_STUDIO_MODEL_COOLDOWN_UNTIL[model_id] = time.time() + 120.0
            api_log(f"Gemma AI Studio ({purpose}) -> {model_id} HTTP {status} ({elapsed:.2f}s)")
            break

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
                        {"inlineData": {"mimeType": mime_type, "data": b64_img}},
                        {"text": prompt_text},
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
    preferred_models: list[str] | None = None,
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

    tool_decls = _get_tools_for_ctx(tool_ctx, GEMINI_TOOLS_BY_MODE)
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
            preferred_models=preferred_models,
        )
        if not data:
            return ""

        out_txt, fn_calls, raw_parts = _extract_gemini_text_and_calls(data)
        if fn_calls and allow_tools_now and tool_ctx is not None:
            contents.append({"role": "model", "parts": raw_parts})
            obs_list = await execute_tool_calls_parallel(fn_calls, tool_ctx)
            _record_tool_observations_on_ctx(tool_ctx, fn_calls, obs_list)
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
    messages: list,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    purpose: str = "CFGemmaChat",
    tool_ctx: ToolExecutionContext | None = None,
    max_tool_rounds: int = 3,
) -> str:
    """
    Cloudflare Workers AI Gemma 4 (`@cf/google/gemma-4-26b-a4b-it`) with native
    multi-round parallel function calling (`tools` + `parallel_tool_calls`) and
    graceful HTTP 400 fallback to plain chat.
    """
    global _CF_GEMMA_COOLDOWN_UNTIL
    if not CF_ACCOUNT_ID or not CF_API_TOKEN or time.time() < _CF_GEMMA_COOLDOWN_UNTIL:
        return ""

    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    work_msgs = []
    for idx, m in enumerate(messages):
        if idx == 0 and m.get("role") == "system":
            work_msgs.append({
                "role": "system",
                "content": (
                    m.get("content", "")
                    + "\n\n[CRITICAL: Answer directly in-character as Yukino Yukinoshita. Do NOT output internal thinking steps.]"
                ),
            })
        else:
            work_msgs.append(dict(m))

    tool_schemas = _get_tools_for_ctx(tool_ctx, OPENAI_TOOLS_BY_MODE)
    tools_enabled = bool(tool_schemas)
    rounds = (max_tool_rounds + 1) if tools_enabled else 1
    session = get_http_session()

    for r_idx in range(rounds):
        allow_tools_now = tools_enabled and (r_idx < max_tool_rounds)
        payload: dict = {
            "model": CF_VISION_MODEL,
            "messages": work_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if allow_tools_now:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True

        t0 = time.monotonic()
        try:
            async with session.post(
                cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=36)
            ) as resp:
                elapsed = time.monotonic() - t0
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
                        _record_tool_observations_on_ctx(tool_ctx, parsed_calls, obs_list)
                        for tc, obs_text in zip(tool_calls, obs_list):
                            work_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id") or f"call_{r_idx}",
                                "name": (tc.get("function") or {}).get("name", ""),
                                "content": obs_text,
                            })
                        continue

                    content = _strip_model_thought_blocks(msg_obj.get("content") or "")
                    if content:
                        state.record_provider_call("cloudflare")
                        api_log(
                            f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} 200 OK ({elapsed:.2f}s"
                            + (f", tools={len(tool_ctx.tools_used)}" if (tool_ctx and tool_ctx.tools_used) else "")
                            + ")"
                        )
                        return content
                    return ""
                elif resp.status == 400 and allow_tools_now:
                    tools_enabled = False
                    api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP 400 on tools; retrying plain chat")
                    continue
                elif resp.status == 429:
                    _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 300.0
                api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
                return ""
        except asyncio.CancelledError:
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} cancelled by hedge ({time.monotonic() - t0:.2f}s)")
            raise
        except Exception as e:
            _CF_GEMMA_COOLDOWN_UNTIL = time.time() + 120.0
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
            return ""
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
    tool_ctx: ToolExecutionContext | None = None,
    hedge_with_cf_fast: bool = True,
    allow_groq_hedge_fallback: bool = True,
) -> str:
    """
    Runs Cloudflare Gemma 4 (`@cf/google/gemma-4-26b-a4b-it`) with a parallel backup hedge if it
    exceeds `hedge_delay_sec`:
    - Prioritizes `CF_CHAT_FAST_MODEL` (`@cf/zai-org/glm-4.7-flash`) ahead of Groq `openai/gpt-oss-20b`
      across ALL Path B (manga + non-manga) and Path C turns.
    - Falls back to hedging with Groq only if `CF_CHAT_FAST_MODEL` is cooled down / unavailable
      AND `allow_groq_hedge_fallback=True`.
    """
    cf_available = bool(CF_ACCOUNT_ID and CF_API_TOKEN and time.time() >= _CF_GEMMA_COOLDOWN_UNTIL)
    if not cf_available:
        return ""

    # When live native tools are enabled on `tool_ctx`, run Cloudflare Gemma 4 directly without
    # a concurrent hedge race so side-effecting tools (OT, memory, image gen) can never double-fire.
    if tool_ctx is not None:
        return await call_cloudflare_gemma_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
            tool_ctx=tool_ctx,
        )

    cf_task = asyncio.create_task(
        call_cloudflare_gemma_chat(messages, temperature=temperature, max_tokens=max_tokens, purpose=purpose)
    )

    can_cf_fast_hedge = bool(
        hedge_with_cf_fast
        and CF_CHAT_FAST_MODEL
        and CF_CHAT_FAST_MODEL != CF_VISION_MODEL
        and time.time() >= _CF_MODEL_COOLDOWN_UNTIL.get(CF_CHAT_FAST_MODEL, 0.0)
    )
    can_groq_hedge = bool(allow_groq_hedge_fallback and GROQ_API_KEY)

    if not can_cf_fast_hedge and not can_groq_hedge:
        return await cf_task

    done, _ = await asyncio.wait({cf_task}, timeout=hedge_delay_sec)
    if cf_task in done:
        try:
            return cf_task.result() or ""
        except Exception:
            return ""

    if can_cf_fast_hedge:
        api_log(f"Hedge ({purpose}) -> CF >{hedge_delay_sec:.1f}s; racing {CF_CHAT_FAST_MODEL} in parallel")
        backup_task = asyncio.create_task(
            call_cloudflare_gpt_oss(
                messages,
                cf_model=CF_CHAT_FAST_MODEL,
                temperature=temperature,
                is_coding=False,
                purpose=f"{purpose}-Hedge",
            )
        )
    else:
        api_log(f"Hedge ({purpose}) -> CF >{hedge_delay_sec:.1f}s; racing Groq backup in parallel")
        backup_task = asyncio.create_task(
            _run_fast_groq_hedge(messages, temperature=temperature, purpose=f"{purpose}-Hedge")
        )
    tasks = {cf_task, backup_task}
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
    if time.time() < _CF_MODEL_COOLDOWN_UNTIL.get(target_model, 0.0):
        return ""

    cf_host = "api." + "cloudflare.com"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    session = get_http_session()
    req_timeout = 40 if is_coding else 22
    tool_schemas = _get_tools_for_ctx(tool_ctx, OPENAI_TOOLS_BY_MODE)
    is_voice = ("voice" in (purpose or "").lower())

    is_gpt_oss_model = "gpt-oss" in target_model.lower()
    is_glm_model = "glm-" in target_model.lower()

    # Try Cloudflare `/ai/v1/responses` ONLY for `gpt-oss` models when not using native OpenAI tool_calls loop
    if is_gpt_oss_model and not tool_schemas and time.time() >= _CF_RESPONSES_UNSUPPORTED_UNTIL:
        sys_parts = []
        convo_lines = []
        for m in _adapt_messages_for_oss_fallback(
            messages, target_model, has_tools=False, is_coding=is_coding, is_voice=is_voice
        ):
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
                    _CF_MODEL_COOLDOWN_UNTIL[target_model] = time.time() + 180.0
                    api_log(f"Cloudflare Responses ({purpose}) -> {target_model} HTTP 429 (cooldown 3m)")
                    return ""
                elif resp.status in (400, 404):
                    _CF_RESPONSES_UNSUPPORTED_UNTIL = time.time() + 1800.0
                    api_log(f"Cloudflare Responses ({purpose}) -> HTTP {resp.status}; switching to chat/completions for 30m")
        except Exception as e:
            api_log(f"Cloudflare Responses ({purpose}) -> {target_model} note ({type(e).__name__}); trying chat/completions")

    chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    work_msgs = _adapt_messages_for_oss_fallback(
        messages, target_model, has_tools=bool(tool_schemas), is_coding=is_coding, is_voice=is_voice
    )
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
        if is_glm_model and not is_coding:
            chat_payload["chat_template_kwargs"] = {"enable_thinking": False}
        if allow_tools_now:
            chat_payload["tools"] = tool_schemas
            chat_payload["tool_choice"] = "auto"
            chat_payload["parallel_tool_calls"] = True

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
                        _record_tool_observations_on_ctx(tool_ctx, parsed_calls, obs_list)
                        for tc, obs_text in zip(tool_calls, obs_list):
                            work_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id") or f"call_{r_idx}",
                                "name": (tc.get("function") or {}).get("name", ""),
                                "content": obs_text,
                            })
                        continue

                    content = _strip_model_thought_blocks(msg_obj.get("content") or "")
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
                    _CF_MODEL_COOLDOWN_UNTIL[target_model] = time.time() + 180.0
                    api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP 429 ({elapsed:.2f}s)")
                    return ""
                else:
                    if resp.status >= 500:
                        _CF_MODEL_COOLDOWN_UNTIL[target_model] = time.time() + 60.0
                    api_log(f"Cloudflare Chat ({purpose}) -> {target_model} HTTP {resp.status} ({elapsed:.2f}s)")
                    return ""
        except asyncio.CancelledError:
            api_log(f"Cloudflare Chat ({purpose}) -> {target_model} cancelled by hedge ({time.monotonic() - t1:.2f}s)")
            raise
        except Exception as e:
            _CF_MODEL_COOLDOWN_UNTIL[target_model] = time.time() + 60.0
            api_log(f"Cloudflare Chat ({purpose}) -> {target_model} failed ({type(e).__name__})")
            return ""
    return ""


async def call_cloudflare_vision_chat(
    prompt_text: str,
    mime_type: str,
    b64_img: str,
    max_tokens: int = 2048,
    purpose: str = "Vision",
    enable_thinking: bool = False,
) -> str:
    """
    Cloudflare Workers AI Gemma 4 Vision (`@cf/google/gemma-4-26b-a4b-it`):
    - Places `image_url` BEFORE `text` per official Gemma 4 multimodal best practices.
    - Supports optional `enable_thinking=True` (for Google Lens-style visual deduction on
      fanart/products) with higher token headroom and automatic non-thinking retry if empty.
    """
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        return ""
    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    session = get_http_session()
    attempts = [True, False] if enable_thinking else [False]

    for think_flag in attempts:
        sys_msg = (
            "Analyze all visible visual details carefully before concluding, then output only the clean structured response."
            if think_flag
            else "Respond directly and concisely with the requested output. Do NOT output internal thinking steps."
        )
        eff_max_tokens = max(max_tokens, 3072) if think_flag else max_tokens
        req_timeout = 16 if think_flag else 18
        payload = {
            "model": CF_VISION_MODEL,
            "messages": [
                {"role": "system", "content": sys_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": eff_max_tokens,
            "chat_template_kwargs": {"enable_thinking": think_flag},
        }
        t0 = time.monotonic()
        try:
            async with session.post(
                cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=req_timeout)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices") or []
                    if choices:
                        msg_obj = choices[0].get("message") or {}
                        content = _strip_model_thought_blocks(msg_obj.get("content") or "")
                        if content:
                            state.record_provider_call("cloudflare")
                            mode_tag = " [think]" if think_flag else ""
                            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL}{mode_tag} 200 OK ({elapsed:.2f}s)")
                            return content
                    if think_flag:
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} [think] empty output; retrying direct")
                        continue
                api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
                if resp.status == 400 and think_flag:
                    continue
                return ""
        except Exception as e:
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_MODEL} failed ({type(e).__name__})")
            if think_flag:
                continue
            return ""
    return ""


async def call_cloudflare_vision_fallback_qwen(
    prompt_text: str,
    mime_type: str,
    b64_img: str,
    max_tokens: int = 2048,
    purpose: str = "VisionCFQwen",
    timeout_sec: int = 24,
) -> str:
    """
    Emergency Cloudflare Vision Fallback (`@cf/qwen/qwen3.8-27b`) used ONLY after Gemini,
    Gemma 4 (AI Studio + Cloudflare), and Groq `qwen/qwen3.8-27b` are unavailable.
    Forces `reasoning_effort="low"` to stay fast and conserve Cloudflare Neurons.
    """
    global _CF_QWEN_VISION_COOLDOWN_UNTIL
    if not CF_ACCOUNT_ID or not CF_API_TOKEN or not CF_VISION_FALLBACK_MODEL:
        return ""
    if time.time() < _CF_QWEN_VISION_COOLDOWN_UNTIL:
        return ""

    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CF_VISION_FALLBACK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Respond directly and concisely with the requested analysis. Do NOT output internal thinking steps.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = _strip_model_thought_blocks((choices[0].get("message") or {}).get("content") or "")
                    if content:
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            elif resp.status == 429:
                _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 300.0
            elif resp.status in (500, 503):
                _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 120.0
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 120.0
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} failed ({type(e).__name__})")
    return ""


async def call_cloudflare_multi_image_vision_qwen(
    encoded_images: list[tuple[str, str, str]],
    batch_prompt: str,
    timeout_sec: int = 50,
    max_tokens: int = 3072,
    purpose: str = "MultiImageCFQwen",
) -> str:
    """Multi-image emergency Cloudflare Vision Fallback (`@cf/qwen/qwen3.8-27b`, `reasoning_effort="low"`)."""
    global _CF_QWEN_VISION_COOLDOWN_UNTIL
    if not encoded_images or not CF_ACCOUNT_ID or not CF_API_TOKEN or not CF_VISION_FALLBACK_MODEL:
        return ""
    if time.time() < _CF_QWEN_VISION_COOLDOWN_UNTIL:
        return ""

    cf_host = "api." + "cloudflare.com"
    cf_chat_url = f"{HTTPS}{cf_host}/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    user_content: list[dict] = []
    for idx, (b64_img, mime_type, fname) in enumerate(encoded_images, 1):
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}})
        user_content.append({"type": "text", "text": f"[Above: Image {idx}/{len(encoded_images)} (`{fname}`)]"})
    user_content.append({"type": "text", "text": batch_prompt})

    payload = {
        "model": CF_VISION_FALLBACK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Respond directly and concisely with the requested analysis. Do NOT output internal thinking steps.",
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
    }
    t0 = time.monotonic()
    try:
        session = get_http_session()
        async with session.post(
            cf_chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status == 200:
                data = await resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = _strip_model_thought_blocks((choices[0].get("message") or {}).get("content") or "")
                    if content:
                        state.record_provider_call("cloudflare")
                        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} 200 OK ({elapsed:.2f}s)")
                        return content
            elif resp.status == 429:
                _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 300.0
            elif resp.status in (500, 503):
                _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 120.0
            api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} HTTP {resp.status} ({elapsed:.2f}s)")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _CF_QWEN_VISION_COOLDOWN_UNTIL = time.time() + 120.0
        api_log(f"Cloudflare ({purpose}) -> {CF_VISION_FALLBACK_MODEL} failed ({type(e).__name__})")
    return ""


async def call_groq_vision_chat(
    prompt_text: str,
    mime_type: str,
    b64_img: str,
    max_tokens: int = 3072,
    timeout_sec: int = 22,
    purpose: str = "VisionGroq",
) -> str:
    """Single-image Groq Vision Fallback reserved exclusively for `qwen/qwen3.8-27b`."""
    if not GROQ_API_KEY:
        return ""
    groq_candidates = await discover_active_groq_vision_models()
    groq_host = "api." + "groq.com"
    chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    session = get_http_session()
    for g_mod in groq_candidates:
        if time.time() < _GROQ_MODEL_COOLDOWN_UNTIL.get(g_mod, 0.0):
            continue
        payload = {
            "model": g_mod,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        t0 = time.monotonic()
        try:
            async with session.post(
                chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    state.active_groq_vision_model = g_mod
                    g_data = await resp.json()
                    content = _strip_model_thought_blocks((g_data["choices"][0]["message"].get("content") or ""))
                    if content:
                        state.record_provider_call("groq")
                        api_log(f"Groq ({purpose}) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                        return content
                elif resp.status == 429:
                    _GROQ_MODEL_COOLDOWN_UNTIL[g_mod] = time.time() + 60.0
                elif resp.status == 404:
                    _GROQ_MODEL_COOLDOWN_UNTIL[g_mod] = time.time() + 3600.0
                api_log(f"Groq ({purpose}) -> {g_mod} HTTP {resp.status} ({elapsed:.2f}s)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            api_log(f"Groq ({purpose}) -> {g_mod} failed ({type(e).__name__})")
    return ""


async def call_groq_multi_image_vision(
    encoded_images: list[tuple[str, str, str]],
    batch_prompt: str,
    max_tokens: int = 3072,
    timeout_sec: int = 35,
    purpose: str = "MultiImageGroq",
) -> str:
    """Multi-image Groq Vision Fallback reserved exclusively for `qwen/qwen3.8-27b`."""
    if not encoded_images or not GROQ_API_KEY:
        return ""
    groq_candidates = await discover_active_groq_vision_models()
    groq_host = "api." + "groq.com"
    chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    # Cap multi-image payload to 5 frames on Groq to stay safely within free-tier TPM/request limits
    batch_slice = encoded_images[:5]
    user_content: list[dict] = []
    for idx, (b64_img, mime_type, fname) in enumerate(batch_slice, 1):
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}})
        user_content.append({"type": "text", "text": f"[Above: Image {idx}/{len(batch_slice)} (`{fname}`)]"})
    user_content.append({"type": "text", "text": batch_prompt})

    session = get_http_session()
    for g_mod in groq_candidates:
        if time.time() < _GROQ_MODEL_COOLDOWN_UNTIL.get(g_mod, 0.0):
            continue
        payload = {
            "model": g_mod,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        t0 = time.monotonic()
        try:
            async with session.post(
                chat_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    state.active_groq_vision_model = g_mod
                    g_data = await resp.json()
                    content = _strip_model_thought_blocks((g_data["choices"][0]["message"].get("content") or ""))
                    if content:
                        state.record_provider_call("groq")
                        api_log(f"Groq ({purpose}) -> {g_mod} 200 OK ({elapsed:.2f}s)")
                        return content
                elif resp.status == 429:
                    _GROQ_MODEL_COOLDOWN_UNTIL[g_mod] = time.time() + 60.0
                elif resp.status == 404:
                    _GROQ_MODEL_COOLDOWN_UNTIL[g_mod] = time.time() + 3600.0
                api_log(f"Groq ({purpose}) -> {g_mod} HTTP {resp.status} ({elapsed:.2f}s)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            api_log(f"Groq ({purpose}) -> {g_mod} failed ({type(e).__name__})")
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

    tool_schemas = _get_tools_for_ctx(tool_ctx, OPENAI_TOOLS_BY_MODE)
    is_voice = ("voice" in (purpose or "").lower())
    token_budget = 6000 if is_coding else 4800
    adapted_msgs = _adapt_messages_for_oss_fallback(
        messages, groq_model, has_tools=bool(tool_schemas), is_coding=is_coding, is_voice=is_voice
    )
    work_msgs = _compact_messages_for_token_budget(adapted_msgs, max_est_tokens=token_budget)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    groq_host = "api." + "groq.com"
    chat_url = f"{HTTPS}{groq_host}/openai/v1/chat/completions"
    session = get_http_session()
    req_timeout = 55 if is_coding else 28

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
            if is_coding:
                groq_payload["reasoning_effort"] = "high" if "120b" in groq_model else "medium"
            else:
                groq_payload["reasoning_effort"] = "low"
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
                    _record_tool_observations_on_ctx(tool_ctx, parsed_calls, obs_list)
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
                if not content:
                    api_log(
                        f"Groq ({purpose}) -> {groq_model} empty output ({elapsed:.2f}s"
                        + (f", tools={len(tool_ctx.tools_used)}" if (tool_ctx and tool_ctx.tools_used) else "")
                        + ")"
                    )
                    return ""
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
                elif resp.status == 404:
                    _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 3600.0
                    await discover_available_groq_models(force_refresh=True)
                elif resp.status == 400:
                    _GROQ_MODEL_COOLDOWN_UNTIL[groq_model] = time.time() + 30.0
                api_log(f"Groq ({purpose}) -> {groq_model} HTTP {resp.status} ({elapsed:.2f}s): {err_txt[:120]}")
                return ""
    return ""

# ==============================================================================
# OPENROUTER MULTI-PAGE MANGA BATCH FALLBACK (`nvidia/nemotron-3.5-super-vl:free`)
# ==============================================================================
async def call_openrouter_multi_image_manga_batch(
    b64_pages: list[tuple[str, str, int]],
    batch_instruction: str,
    mime_type: str = "image/jpeg",
    timeout_sec: float = 75.0,
    max_tokens: int = 8192,
    purpose: str = "MangaBatchOpenRouter",
) -> str:
    """
    Emergency multi-page MangaBatch Vision fallback via OpenRouter (`nvidia/nemotron-3.5-super-vl:free`,
    1M context, 120B/12B MoE). Sends ALL pages in a single Set-of-Mark batch request (no per-page split)
    with `reasoning: {"effort": "low", "exclude": True}` and `response_format: {"type": "json_object"}`
    for fast, clean JSON output during Google AI Studio outages.
    """
    if not b64_pages or not OPENROUTER_API_KEY or not OPENROUTER_MANGA_MODEL:
        return ""
    if time.time() < getattr(state, "openrouter_manga_cooldown_until", 0.0):
        return ""

    or_host = "openrouter." + "ai"
    or_url = f"{HTTPS}{or_host}/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yukinon-bot.local",
        "X-Title": "Yukinon Discord Bot",
    }

    user_content: list[dict] = [{"type": "text", "text": batch_instruction}]
    for idx, (b64_img, fname, cv_cnt) in enumerate(b64_pages, 1):
        if cv_cnt > 0:
            badge_ids_str = ", ".join(str(i) for i in range(1, cv_cnt + 1))
            badge_note = (
                f"NUMBERED BADGES ON THIS PAGE: [{badge_ids_str}] "
                "(account for EVERY badge ID strictly inside its own red outline; "
                "add bubble_id=null for any unbadged dialogue in other panels)"
            )
        else:
            badge_note = "no numbered badges (use bubble_id=null + box_2d for any dialogue)"
        user_content.append({"type": "text", "text": f"=== PAGE {idx} ({fname} | {badge_note}) ==="})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_img}"},
        })

    payload: dict = {
        "model": OPENROUTER_MANGA_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.1,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "low", "exclude": True},
    }

    session = get_http_session()
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            async with session.post(
                or_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                status = resp.status
                body_text = await resp.text()
                elapsed = time.monotonic() - t0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state.openrouter_manga_cooldown_until = time.time() + 60.0
            api_log(
                f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} failed "
                f"({type(e).__name__}, {time.monotonic() - t0:.2f}s)"
            )
            return ""

        if status == 200:
            try:
                data = json.loads(body_text)
                choices = data.get("choices") or []
                msg_obj = (choices[0].get("message") or {}) if choices else {}
                raw_content = msg_obj.get("content") or ""
                if isinstance(raw_content, list):
                    raw_content = "".join(
                        str(p.get("text") or "") for p in raw_content if isinstance(p, dict)
                    )
                content = _strip_model_thought_blocks(str(raw_content))
            except Exception as e:
                api_log(f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} JSON parse error ({type(e).__name__})")
                return ""
            if content:
                state.record_provider_call("openrouter")
                api_log(f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} 200 OK ({elapsed:.2f}s)")
                return content
            api_log(f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} HTTP 200 empty output ({elapsed:.2f}s)")
            return ""

        if status == 400 and attempt == 0:
            changed = False
            low_err = body_text.lower()
            if "response_format" in payload and ("response_format" in low_err or "json" in low_err or "schema" in low_err):
                payload.pop("response_format", None)
                changed = True
            if "reasoning" in payload and ("reasoning" in low_err or "effort" in low_err or "exclude" in low_err):
                payload.pop("reasoning", None)
                changed = True
            if not changed and ("response_format" in payload or "reasoning" in payload):
                payload.pop("response_format", None)
                payload.pop("reasoning", None)
                changed = True
            if changed:
                api_log(
                    f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} HTTP 400 ({elapsed:.2f}s); "
                    "retrying without response_format/reasoning"
                )
                continue

        if status == 429:
            state.openrouter_manga_cooldown_until = time.time() + 120.0
        elif status in (401, 402, 403):
            state.openrouter_manga_cooldown_until = time.time() + 1800.0
        elif status >= 500:
            state.openrouter_manga_cooldown_until = time.time() + 60.0

        api_log(
            f"OpenRouter ({purpose}) -> {OPENROUTER_MANGA_MODEL} HTTP {status} "
            f"({elapsed:.2f}s): {body_text[:140]}"
        )
        return ""

    return ""