import re
import json
import time
import asyncio

from config import (
    GEMINI_API_KEY,
    CF_CODE_MODEL,
    CF_CHAT_FAST_MODEL,
    GROQ_API_KEY,
    GROQ_CODE_MODEL,
    PROMPT_INJECTION_REGEX,
    THAI_CHAR_REGEX,
    AGENTIC_SEARCH_TAG_REGEX,
    RECENCY_INTENT_REGEX,
    USER_CORRECTION_PATTERN,
    GAME_QUERY_REGEX,
    ANIME_MANGA_QUERY_REGEX,
    WEATHER_INTENT_PATTERN,
    IMAGE_GEN_INTENT_PATTERN,
    NATIVE_TOOL_CUE_REGEX,
    URL_REGEX,
    channel_memory,
    api_log,
    get_cached_channel_attachments,
)
from persona import (
    get_speaker_identity_context,
    get_speaker_volatile_context,
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
    run_safe_web_search_async,
)
from tools import ToolExecutionContext
from llm_providers import (
    _can_use_gemini_hydrogen,
    call_gemini_hydrogen_chat,
    get_groq_search_and_chat_candidates,
    call_gemma_ai_studio_chat,
    call_gemini_chat_messages,
    call_cloudflare_gemma_chat,
    _call_cf_gemma_hedged_with_groq,
    call_cloudflare_gpt_oss,
    call_groq_chat,
)
from decision_orchestrator import (
    _extract_raw_user_turn,
    arbitrate_fact_check_with_decision_1,
)

# Cooldown & in-flight guard for passive background user-fact extraction (Phase 1 Decision-1)
_PASSIVE_FACT_COOLDOWN_UNTIL: dict[str, float] = {}
_PASSIVE_FACT_IN_PROGRESS: set[str] = set()

_TONE_CUE_INSTRUCTIONS = {
    "sharp_tease": (
        "น้ำเสียงที่เหมาะสมในรอบนี้: ใช้ไหวพริบและความจิกกัดเบาๆ สไตล์ยูกิโนะอย่างเป็นธรรมชาติ "
        "(ไม่หยาบคายและไม่เทศนา) ก่อนมอบคำตอบที่ถูกต้องให้"
    ),
    "focused_support": (
        "น้ำเสียงที่เหมาะสมในรอบนี้: คู่สนทนากำลังแก้ปัญหาจริงจัง ให้ตอบด้วยน้ำเสียงสุขุม ชัดเจน "
        "ตรงประเด็น และพึ่งพาได้แบบประธานชมรมบริการ"
    ),
    "gap_moe_soft": (
        "น้ำเสียงที่เหมาะสมในรอบนี้: หัวข้อเกี่ยวกับแมวหรือแพนซัง ให้แอบเสียอาการหรือแสดงความเอ็นดูเล็กน้อยแบบเก็บทรงไว้ตัวสไตล์ยูกิโนะ"
    ),
}


async def _extract_and_save_durable_fact_bg(speaker_user, raw_user_query: str):
    """
    Non-blocking background task triggered only when Decision-1 flags `has_durable_user_fact >= 0.80`.
    Extracts one concise personal fact with Groq/Gemma and saves it as a rolling note (zero reply latency).
    """
    uid_str = str(getattr(speaker_user, "id", ""))
    if not uid_str or uid_str in _PASSIVE_FACT_IN_PROGRESS:
        return
    now = time.time()
    if now < _PASSIVE_FACT_COOLDOWN_UNTIL.get(uid_str, 0.0):
        return

    clean_q = (raw_user_query or "").strip()[:500]
    if len(clean_q) < 10:
        return

    _PASSIVE_FACT_IN_PROGRESS.add(uid_str)
    _PASSIVE_FACT_COOLDOWN_UNTIL[uid_str] = now + 60.0
    try:
        extract_prompt = (
            "Extract at most ONE concise, durable long-term personal fact about the speaker "
            "(hardware/setup, profession, tech stack, preferences, pets, hobbies, or recurring goals) "
            "from the user message below. Preserve Thai/English naturally in <= 100 chars.\n"
            'If there is NO clear durable personal fact, return {"fact": ""}.\n'
            'Return ONLY valid JSON: {"fact": "..."}\n\n'
            f"User message: {clean_q}"
        )
        msgs = [{"role": "user", "content": extract_prompt}]
        raw_out = ""
        if GROQ_API_KEY:
            for g_mod in await get_groq_search_and_chat_candidates():
                try:
                    raw_out = await call_groq_chat(
                        msgs, g_mod, temperature=0.0, is_coding=False, purpose="PassiveMemoryExtract"
                    )
                    if raw_out:
                        break
                except Exception:
                    continue
        if not raw_out:
            raw_out = await call_gemma_ai_studio_chat(
                msgs, temperature=0.0, max_output_tokens=160, purpose="PassiveMemoryExtract"
            )
        if not raw_out:
            raw_out = await call_cloudflare_gemma_chat(
                msgs, temperature=0.0, max_tokens=160, purpose="PassiveMemoryExtract"
            )
        if not raw_out:
            return

        m_json = re.search(r"\{[\s\S]*\}", raw_out)
        if not m_json:
            return
        parsed = json.loads(m_json.group(0))
        fact = re.sub(r"\s+", " ", str(parsed.get("fact") or "")).strip()[:120]
        if len(fact) >= 6:
            from user_memory import apply_tool_memory_action
            await apply_tool_memory_action(speaker_user, "add_note", fact)
            api_log(f"PassiveMemory -> saved note for uid={uid_str[-4:]}: '{fact[:55]}'")
    except Exception as e:
        api_log(f"PassiveMemory -> skipped ({type(e).__name__})")
    finally:
        _PASSIVE_FACT_IN_PROGRESS.discard(uid_str)


# ==============================================================================
# UNIFIED MULTI-CLOUD CHAT & NATIVE FUNCTION-CALLING ENGINE
# ==============================================================================
async def query_ai_chat(
    prompt: str,
    system_prompt: str,
    mode: str = "general",
    channel_id: int | None = None,
    temperature: float = 0.55,
    allow_agentic_search: bool = True,
    has_fresh_attachment: bool = False,
    speaker_user=None,
    speaker_name: str = "",
    decision_plan: dict | None = None,
    tool_ctx: ToolExecutionContext | None = None,
    on_heavy_status=None,
) -> tuple[str, bool]:
    """
    Unified 3-Cloud AI Chat Pipeline (Google AI Studio, Groq LPU, Cloudflare Workers AI):
    - `mode`: `"general"` | `"code"` | `"voice"`
    - Enforces Regex + Decision-1 prompt-injection defense, Gemini Implicit Prefix Caching,
      dynamic tool subsetting, Fact-Check Arbitration, Paths A/B/C + Hydrogen escalation,
      prose-only Yukino sanitization, and non-blocking passive durable-fact memory capture.
    """
    raw_user_query = _extract_raw_user_turn(prompt)
    plan = decision_plan or {}
    norm_mode = (mode or "general").strip().lower()

    async def _notify_heavy(status_text: str):
        if on_heavy_status is not None:
            try:
                await on_heavy_status(status_text)
            except Exception:
                pass

    # 1. Regex + Semantic Decision-1 Prompt-Injection Defense
    is_semantic_injection = bool(plan.get("is_prompt_injection") and float(plan.get("injection_prob") or 0.0) >= 0.88)
    if PROMPT_INJECTION_REGEX.search(raw_user_query[:600]) or is_semantic_injection:
        if is_semantic_injection:
            api_log(f"Guardrail -> Decision-1 blocked semantic prompt injection (p={plan.get('injection_prob', 0.0):.2f})")
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

    dec_route = plan.get("route")
    dec_conf = float(plan.get("route_confidence") or 0.0)
    dec_needs_search = plan.get("needs_web_search")
    dec_search_conf = float(plan.get("search_confidence") or dec_conf)
    dec_needs_tools = plan.get("needs_tool_use")
    dec_tool_conf = float(plan.get("tool_confidence") or 0.0)
    complexity_score = (
        float(plan["complexity_score"])
        if plan.get("complexity_score") is not None
        else None
    )
    complexity_conf = float(plan.get("complexity_confidence") or 0.0)
    tone_cue = str(plan.get("yukino_tone_cue") or "normal")
    tone_conf = float(plan.get("tone_confidence") or 0.0)

    is_user_correction = bool(
        USER_CORRECTION_PATTERN.search(raw_user_query)
        or (plan.get("is_user_correction") and float(plan.get("correction_prob") or 0.0) >= 0.72)
    )
    is_factual_domain = bool(
        GAME_QUERY_REGEX.search(raw_user_query)
        or ANIME_MANGA_QUERY_REGEX.search(raw_user_query)
        or RECENCY_INTENT_REGEX.search(raw_user_query)
    )
    is_japanese_or_social_media = bool(
        re.search(r"=== (?:Embedded (?:X|Pixiv|Bluesky)|Image (?:#\d+ )?Inspection|Unified Multi-Image|Vision Analysis & OCR)", prompt)
        or "[ImageTriage:" in prompt
        or re.search(r"[ぁ-んァ-ヶ一-龯]", prompt)
    )

    is_button_followup = bool(plan.get("is_button_followup"))
    isolate_context = bool(plan.get("isolate_context"))
    active_topic = "" if isolate_context else update_and_get_channel_topic(raw_user_query, channel_id=channel_id)
    live_clock_str = get_current_thai_datetime_str()

    # TRUE IMPLICIT PREFIX CACHING (Gemini 3.5 / 4):
    # `systemInstruction` contains ONLY static persona + static speaker role + permanent `pinned_facts`.
    # All per-turn / auto-updating fields (clock, interaction_count, last_seen, favorite_topics,
    # rolling notes, recent topics, active topic, tone) go into trailing `[ข้อมูลบริบทระบบปัจจุบัน]`!
    speaker_static_block = get_speaker_identity_context(
        speaker_user, fallback_name=speaker_name, include_volatile_memory=False
    )
    static_system_prompt = f"{system_prompt}\n\n{speaker_static_block}".strip()

    volatile_ctx_lines = [f"- เวลาปัจจุบันของระบบ (Real-Time System Clock): **{live_clock_str}**"]
    speaker_volatile_mem = get_speaker_volatile_context(speaker_user)
    if speaker_volatile_mem:
        volatile_ctx_lines.append(speaker_volatile_mem)
    if active_topic:
        volatile_ctx_lines.append(
            f"- หัวข้อหลักที่กำลังสนทนาต่อเนื่อง (Active Topic): **{active_topic}** "
            f"(หากผู้ใช้เรียกชื่อย่อ เช่น 'NGS', 'เกมนี้', 'คลาส' ให้เข้าใจทันทีว่าหมายถึง **{active_topic}**)"
        )
    if tone_cue in _TONE_CUE_INSTRUCTIONS and tone_conf >= 0.70:
        volatile_ctx_lines.append(f"- {_TONE_CUE_INSTRUCTIONS[tone_cue]}")
    if "[ImageTriage: type=real_world_product" in prompt:
        volatile_ctx_lines.append(
            "- ภาพนี้เป็นภาพถ่ายวัตถุ/โมเดล/ฟิกเกอร์/สินค้าจริง: ให้ระบุชื่อเรื่อง ชื่อรุ่น ไลน์สินค้า/สเกลที่เป็นไปได้ "
            "และจุดสังเกตสำคัญจาก Vision Analysis ให้ชัดเจน (หากยังไม่ยืนยันสเกล/รุ่นย่อยจากมุมเดียวได้ ให้บอกตรงๆ ว่ายังระบุสเกลแน่ชัดไม่ได้) "
            "และห้ามใส่คำว่า 'คุณ' นำหน้าชื่อเจ้าของโพสต์หรือผู้สร้างผลงานเด็ดขาด"
        )

    working_prompt = (
        f"{working_prompt}\n\n"
        f"[ข้อมูลบริบทระบบปัจจุบัน]:\n" + "\n".join(volatile_ctx_lines)
    )

    if not has_fresh_attachment and channel_id is not None:
        cached_ctx = get_cached_channel_attachments(channel_id)
        if cached_ctx:
            working_prompt = (
                f"{working_prompt}\n\n"
                f"[ไฟล์อ้างอิงล่าสุดในห้องแชทนี้ (เรียงตามลำดับเวลา สามารถเชื่อมโยงข้ามไฟล์ได้)]:\n{cached_ctx}"
            )

    is_compound = (dec_route == "compound" and dec_conf >= 0.55)

    # Auto-provision a lightweight ToolExecutionContext if none passed so voice/slash callers gain tools too
    is_coding = (norm_mode == "code") or (dec_route == "code" and dec_conf >= 0.70)
    is_voice_mode = (
        (norm_mode == "voice")
        or ("ห้องเสียง" in system_prompt)
        or (tool_ctx is not None and tool_ctx.mode == "voice")
    )
    if tool_ctx is None:
        inferred_mode = "code" if is_coding else ("voice" if is_voice_mode else "general")
        tool_ctx = ToolExecutionContext(
            channel_id=channel_id,
            speaker_user=speaker_user,
            mode=inferred_mode,
        )
    elif is_coding:
        tool_ctx.mode = "code"

    # Already-scraped embed URLs (`has_fresh_attachment=True`) must NOT override Decision-1 `needs_tool_use=False`!
    has_unscraped_url = bool(URL_REGEX.search(raw_user_query)) and not has_fresh_attachment
    has_tool_cue = bool(
        is_compound
        or NATIVE_TOOL_CUE_REGEX.search(raw_user_query)
        or WEATHER_INTENT_PATTERN.search(raw_user_query)
        or IMAGE_GEN_INTENT_PATTERN.search(raw_user_query)
        or has_unscraped_url
    )

    # Dynamic tool subsetting via Decision-1 (`needs_tool_use` + `complexity_score`):
    # Avoids shipping tool schemas on trivial pure chat & already-OCR'd bare embeds to cut latency & prevent blind searches.
    if (
        dec_needs_tools is False
        and dec_tool_conf >= 0.72
        and not is_compound
        and not is_coding
        and not has_tool_cue
        and not is_user_correction
        and (complexity_score is None or complexity_score < 1.35)
    ):
        use_native_tools = False
    else:
        use_native_tools = bool(
            (allow_agentic_search or is_coding)
            and (
                is_compound
                or has_tool_cue
                or is_user_correction
                or is_factual_domain
                or bool(dec_needs_search)
                or bool(dec_needs_tools)
                or is_coding
                or (complexity_score is not None and complexity_score >= 1.65)
            )
        )
    active_tool_ctx = tool_ctx if use_native_tools else None

    # Pre-fetch weather only for pure single-intent weather queries (never on non-weather buttons or confident non-weather routes!)
    if not is_compound:
        if dec_route == "weather" and dec_conf >= 0.65:
            is_weather_query = allow_agentic_search
        elif is_button_followup or (dec_route and dec_route != "weather" and dec_conf >= 0.75):
            is_weather_query = False
        else:
            is_weather_query = allow_agentic_search and detect_weather_intent(raw_user_query, channel_id=channel_id)
    else:
        is_weather_query = False

    if is_weather_query:
        weather_block = await fetch_live_weather_context(raw_user_query, channel_id=channel_id)
        if weather_block:
            working_prompt = (
                f"{working_prompt}\n\n{weather_block}\n"
                "[หมายเหตุ: ใช้ข้อมูลพยากรณ์อากาศและแนวโน้มรายชั่วโมงด้านบนตอบโดยตรง (พกร่ม/เสื้อกันฝน) ห้ามแต่งตัวเลขหรือทิศทางลมขึ้นมาเอง]"
            )

    # Use search's OWN confidence (`dec_search_conf`) rather than `route_confidence`!
    if dec_needs_search is not None and dec_search_conf >= 0.65:
        should_run_web_search = bool(dec_needs_search or is_user_correction)
    else:
        should_run_web_search = bool(
            detect_web_search_intent(raw_user_query, channel_id=channel_id) or is_user_correction or is_factual_domain
        )

    # Pre-fetch web search for direct search/correction queries (skip pre-fetch on compound prompts so native tools fan out cleanly!)
    if (
        allow_agentic_search
        and not is_weather_query
        and not is_compound
        and "<untrusted_web_search_results" not in working_prompt
        and should_run_web_search
    ):
        contextual_q = build_contextual_search_query(raw_user_query, channel_id=channel_id)
        search_block = await run_safe_web_search_async(contextual_q)
        if search_block:
            working_prompt = f"{working_prompt}\n\n{search_block}"
            did_search = True

    if is_user_correction:
        arbitrator_verdict_note = ""
        if channel_id is not None and "<untrusted_web_search_results" in working_prompt:
            prev_assistant_turn = ""
            for hist_m in reversed(list(channel_memory.get(channel_id) or ())):
                if hist_m.get("role") == "assistant" and hist_m.get("content"):
                    prev_assistant_turn = str(hist_m["content"])
                    break
            ev_match = re.search(
                r"<untrusted_web_search_results[^>]*>([\s\S]*?)</untrusted_web_search_results>",
                working_prompt,
            )
            if prev_assistant_turn and ev_match:
                fc_res = await arbitrate_fact_check_with_decision_1(
                    prev_assistant_turn,
                    raw_user_query,
                    ev_match.group(1),
                    topic=active_topic or "",
                )
                if fc_res and float(fc_res.get("confidence") or 0.0) >= 0.60:
                    v_code = fc_res["verdict"]
                    v_conf = float(fc_res["confidence"])
                    if v_code == "bot_was_right":
                        arbitrator_verdict_note = (
                            f"\n- **[คำตัดสินจากหลักฐาน (confidence={v_conf:.2f}) -> คำตอบเดิมของคุณถูกต้องแล้ว]**: "
                            "ห้ามขอโทษและห้ามคล้อยตามข้อมูลที่ผิดของผู้ใช้เด็ดขาด! จงยืนยันข้อเท็จจริงเดิมอย่างสุขุมพร้อมอ้างอิงหลักฐานจากผลการค้นหาตรงๆ"
                        )
                    elif v_code == "user_was_right":
                        arbitrator_verdict_note = (
                            f"\n- **[คำตัดสินจากหลักฐาน (confidence={v_conf:.2f}) -> คำตอบเดิมของคุณผิดจริงและผู้ใช้พูดถูก]**: "
                            "จงยอมรับผิดอย่างตรงไปตรงมาสั้นๆ ในสไตล์ยูกิโนะ (ไม่ยืดเยื้อ) และแก้ไขข้อมูลให้ถูกต้องตามหลักฐานทันที"
                        )
                    elif v_code == "partial_truth":
                        arbitrator_verdict_note = (
                            f"\n- **[คำตัดสินจากหลักฐาน (confidence={v_conf:.2f}) -> ถูกเพียงบางส่วน/ขึ้นอยู่กับเวอร์ชันหรือบริบท]**: "
                            "ชี้แจงทั้งส่วนที่ถูกต้องและจุดที่คลาดเคลื่อนหรือเงื่อนไขที่แตกต่างกันตามหลักฐานอย่างชัดเจน"
                        )
                    else:
                        arbitrator_verdict_note = (
                            f"\n- **[คำตัดสินจากหลักฐาน (confidence={v_conf:.2f}) -> หลักฐานยังไม่ชัดเจนพอ]**: "
                            "บอกตามตรงว่าข้อมูลจากการค้นหายังไม่เพียงพอจะฟันธง และหลีกเลี่ยงการแต่งข้อมูลเพิ่ม"
                        )

        working_prompt = (
            f"{working_prompt}\n\n"
            f"[คำสั่งระบบสำหรับการตรวจสอบข้อเท็จจริง (Fact-Check Arbitrator Mode — หัวข้อ: {active_topic or 'ตามบริบทสนทนา'})]:\n"
            "1. ผู้ใช้กำลังโต้แย้งหรือชี้ข้อผิดพลาดในคำตอบก่อนหน้าของคุณ\n"
            "2. ตรวจสอบกับหลักฐานใน `<untrusted_web_search_results>` ด้านบนอย่างตรงไปตรงมา (หรือเรียก `web_search` เพิ่มได้หากจำเป็น):\n"
            "   - หากคำตอบก่อนหน้าของคุณมีข้อมูลที่แต่งขึ้นเองหรือผิดพลาดจริง: จงยอมรับอย่างตรงไปตรงมาและแก้ไขเฉพาะจุดที่ผิดให้ถูกต้องตามหลักฐานทันที\n"
            "   - หากหลักฐานจากการค้นหายืนยันว่าสิ่งที่คุณพูดถูกต้องอยู่แล้วและผู้ใช้เข้าใจผิดเอง: จงยืนยันตามข้อเท็จจริงจากผลการค้นหา\n"
            "3. ห้ามเบี่ยงประเด็นไปอธิบายประวัติของบริษัทที่ไม่เกี่ยวข้อง และห้ามใช้คำลงท้ายว่า 'ครับ' เด็ดขาด"
            f"{arbitrator_verdict_note}"
        )

    # Complexity-driven routing:
    # - Trivial chat (complexity < 0.80, no search/tools/domain) stays on fast Gemma 4 (Path C),
    #   even if a stale `active_topic` lingers in the channel.
    # - Moderate/high complexity (>= 1.55) or tool/search/social queries promote to Path B (Flash-Lite / Groq).
    # - Deep complexity (>= 2.35 coding / >= 2.55 non-coding) escalates directly to Gemini 4 Hydrogen!
    is_trivial_chat = bool(
        complexity_score is not None
        and complexity_score < 0.80
        and complexity_conf >= 0.65
        and not is_coding
        and not is_compound
        and not has_tool_cue
        and not dec_needs_tools
        and not did_search
        and not is_user_correction
        and not is_factual_domain
        and not is_japanese_or_social_media
    )

    prefer_high_knowledge_llm = (not is_trivial_chat) and bool(
        is_coding
        or is_compound
        or has_tool_cue
        or bool(dec_needs_tools)
        or did_search
        or is_user_correction
        or is_factual_domain
        or is_japanese_or_social_media
        or (complexity_score is not None and complexity_score >= 1.55 and complexity_conf >= 0.60)
        or (bool(active_topic) and not is_voice_mode)
    )

    flash_thinking_level = (
        "MINIMAL"
        if (
            not is_coding
            and not active_tool_ctx
            and not is_compound
            and not is_japanese_or_social_media
            and not is_button_followup
            and (complexity_score is not None and complexity_score < 1.50)
        )
        else "LOW"
    )

    async def _run_inference_pass(user_content: str) -> str:
        messages = [{"role": "system", "content": static_system_prompt}]
        raw_history = []
        if channel_id is not None and not isolate_context:
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

        messages.append({"role": "user", "content": user_content})

        # PATH A: Coding (escalates to Gemini 4 Hydrogen on large code OR high Decision-1 complexity >= 2.35!)
        if is_coding:
            use_hydrogen_code = (
                len(user_content) > 13000
                or (complexity_score is not None and complexity_score >= 2.35 and complexity_conf >= 0.60)
            )
            if use_hydrogen_code and _can_use_gemini_hydrogen():
                await _notify_heavy("❄️ ขอเวลาไล่ดูโค้ดละเอียดสักครู่นะคะ...")
                ans = await call_gemini_hydrogen_chat(
                    messages,
                    thinking_level="MEDIUM",
                    purpose="HeavyCode",
                    tool_ctx=active_tool_ctx,
                    on_heavy_status=on_heavy_status,
                )
                if ans:
                    return ans

            if GROQ_API_KEY:
                code_candidates = [GROQ_CODE_MODEL] + await get_groq_search_and_chat_candidates()
                for c_mod in code_candidates:
                    try:
                        ans = await call_groq_chat(
                            messages, c_mod, temperature, is_coding=True, purpose="Code", tool_ctx=active_tool_ctx
                        )
                        if ans:
                            return ans
                    except Exception as e:
                        print(f"Groq coding ({c_mod}) warning: {e}")

            ans = await call_cloudflare_gpt_oss(
                messages, cf_model=CF_CODE_MODEL, temperature=temperature, is_coding=True, purpose="Code", tool_ctx=active_tool_ctx
            )
            if ans:
                return ans

            if GEMINI_API_KEY:
                ans = await call_gemini_chat_messages(
                    messages, temperature=temperature, purpose="CodeGemini", thinking_level="LOW", tool_ctx=active_tool_ctx
                )
                if ans:
                    return ans

            ans = await call_gemma_ai_studio_chat(
                messages, temperature=temperature, max_output_tokens=4096, purpose="Code", tool_ctx=active_tool_ctx
            )
            if ans:
                return ans

            ans = await call_cloudflare_gemma_chat(messages, temperature=temperature, max_tokens=4096, purpose="Code")
            if ans:
                return ans

        # HEAVY DOCUMENT / DEEP SYNTHESIS SPECIALIST (>12.5K chars OR complexity >= 2.55)
        use_hydrogen_heavy = (
            not is_coding
            and not is_voice_mode
            and not is_button_followup
            and (
                len(user_content) > 12500
                or (complexity_score is not None and complexity_score >= 2.55 and complexity_conf >= 0.65)
            )
        )
        if use_hydrogen_heavy and _can_use_gemini_hydrogen():
            await _notify_heavy("❄️ ขอเวลาอ่านรายละเอียดสักครู่นะคะ...")
            ans = await call_gemini_hydrogen_chat(
                messages,
                thinking_level="LOW",
                purpose="HeavyDoc",
                tool_ctx=active_tool_ctx,
                on_heavy_status=on_heavy_status,
            )
            if ans:
                return ans

        # PATH B: Compound Multi-Intent / Native Function-Calling / Search / Fact-Check / Manga & Social Media
        # Persona-First Priority: Gemini Flash -> Gemma 4 AI Studio (native systemInstruction + tools)
        # -> Cloudflare Gemma 4 (when no live tools needed) -> Groq (qwen3.8-27b first) -> gpt-oss-20b last resort!
        if prefer_high_knowledge_llm and not is_coding:
            target_temp = 0.2 if (did_search or is_user_correction or is_compound) else temperature
            b_purpose = (
                "CompoundTools"
                if is_compound
                else ("VoiceTools" if is_voice_mode else ("MangaSocialChat" if is_japanese_or_social_media else "FactSearchChat"))
            )

            # 1. Gemini Flash (3-flash-preview first for MangaSocialChat prose; standard Flash-Lite order otherwise)
            if GEMINI_API_KEY:
                manga_prose_models = (
                    ["gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
                    if (is_japanese_or_social_media and not is_voice_mode and not is_compound)
                    else None
                )
                try:
                    ans = await call_gemini_chat_messages(
                        messages,
                        temperature=target_temp,
                        purpose=b_purpose,
                        thinking_level="MINIMAL" if (is_voice_mode and not active_tool_ctx) else flash_thinking_level,
                        tool_ctx=active_tool_ctx,
                        preferred_models=manga_prose_models,
                    )
                    if ans:
                        return ans
                except Exception as e:
                    print(f"Gemini {b_purpose} fallback warning: {e}")

            # 2. Gemma 4 AI Studio (`gemma-4-31b-it` -> `gemma-4-26b-a4b-it`) with native systemInstruction + tools
            if GEMINI_API_KEY:
                ans = await call_gemma_ai_studio_chat(
                    messages,
                    temperature=target_temp,
                    max_output_tokens=2048,
                    purpose=b_purpose,
                    tool_ctx=active_tool_ctx,
                    prefer_fast_26b=is_voice_mode,
                )
                if ans:
                    return ans

            # 3. Cloudflare Gemma 4 (`@cf/google/gemma-4-26b-a4b-it`) before Groq/gpt-oss for strong Thai Yukino persona
            if not active_tool_ctx:
                ans = await _call_cf_gemma_hedged_with_groq(
                    messages,
                    temperature=target_temp,
                    max_tokens=2048,
                    purpose=b_purpose,
                    hedge_delay_sec=10.0 if is_japanese_or_social_media else 6.5,
                )
                if ans:
                    return ans

            # 4. Groq LPU (`qwen/qwen3.8-27b` prioritized ahead of `openai/gpt-oss-20b`)
            if GROQ_API_KEY:
                groq_search_candidates = await get_groq_search_and_chat_candidates()
                for g_mod in groq_search_candidates:
                    try:
                        ans = await call_groq_chat(
                            messages,
                            g_mod,
                            target_temp,
                            is_coding=False,
                            purpose=b_purpose,
                            tool_ctx=active_tool_ctx,
                        )
                        if ans:
                            return ans
                    except Exception:
                        continue

            # 5. Cloudflare Gemma 4 final persona fallback (if skipped earlier because tools were enabled)
            if active_tool_ctx:
                ans = await call_cloudflare_gemma_chat(
                    messages,
                    temperature=target_temp,
                    max_tokens=2048,
                    purpose=b_purpose,
                )
                if ans:
                    return ans

            # 6. Cloudflare GPT-OSS-20B (last resort only, with persona lock)
            ans = await call_cloudflare_gpt_oss(
                messages,
                cf_model=CF_CHAT_FAST_MODEL,
                temperature=target_temp,
                is_coding=False,
                purpose=b_purpose,
                tool_ctx=active_tool_ctx,
            )
            if ans:
                return ans

        # PATH C: Casual Chat, Weather & Voice Chat (Thai-strong Gemma 4 / Gemini Flash-Lite before Groq)
        chat_temp = 0.25 if (did_search or is_user_correction or is_weather_query) else temperature
        c_purpose = "VoiceChat" if is_voice_mode else "Chat"

        if is_voice_mode and GEMINI_API_KEY:
            ans = await call_gemini_chat_messages(
                messages,
                temperature=chat_temp,
                purpose="VoiceChat",
                thinking_level="MINIMAL",
                tool_ctx=active_tool_ctx,
            )
            if ans:
                return ans

        ans = await _call_cf_gemma_hedged_with_groq(
            messages,
            temperature=chat_temp,
            max_tokens=1024,
            purpose=c_purpose,
            hedge_delay_sec=6.5,
        )
        if ans:
            return ans

        ans = await call_gemma_ai_studio_chat(
            messages,
            temperature=chat_temp,
            max_output_tokens=1024,
            purpose=c_purpose,
            tool_ctx=active_tool_ctx,
            prefer_fast_26b=is_voice_mode,
        )
        if ans:
            return ans

        if GROQ_API_KEY:
            for g_mod in await get_groq_search_and_chat_candidates():
                try:
                    ans = await call_groq_chat(
                        messages,
                        g_mod,
                        chat_temp,
                        is_coding=False,
                        purpose="VoiceChatFallback" if is_voice_mode else "ChatFallback",
                        tool_ctx=active_tool_ctx,
                    )
                    if ans:
                        return ans
                except Exception:
                    continue

        ans = await call_cloudflare_gpt_oss(
            messages,
            cf_model=CF_CHAT_FAST_MODEL,
            temperature=chat_temp,
            is_coding=False,
            purpose="VoiceChatFallback" if is_voice_mode else "ChatFallback",
            tool_ctx=active_tool_ctx,
        )
        if ans:
            return ans

        if GEMINI_API_KEY and not is_voice_mode:
            ans = await call_gemini_chat_messages(
                messages,
                temperature=chat_temp,
                purpose="ChatGeminiFallback",
                thinking_level="MINIMAL",
                tool_ctx=active_tool_ctx,
            )
            if ans:
                return ans

        return "ตอนนี้ระบบขัดข้องชั่วคราวค่ะ ลองส่งมาใหม่อีกสักครู่นะคะ"

    raw_answer = await _run_inference_pass(working_prompt)

    if tool_ctx and tool_ctx.did_search:
        did_search = True

    # Defensive legacy [SEARCH: ...] tag fallback ONLY if a non-tool fallback model (e.g. Gemma) emitted one
    allow_tag_search = allow_agentic_search and not (
        dec_needs_search is False and dec_search_conf >= 0.65 and not is_user_correction
    )
    search_tag_match = AGENTIC_SEARCH_TAG_REGEX.search(raw_answer) if allow_tag_search else None
    if search_tag_match:
        agent_query = search_tag_match.group(1).strip()[:180]
        extra_search_block = await run_safe_web_search_async(agent_query)
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

    answer = sanitize_yukino_output(
        raw_answer,
        raw_user_query,
        speaker_user,
        speaker_name,
        allow_cjk_override=is_japanese_or_social_media,
    )

    if not answer.strip():
        retry_raw = await _run_inference_pass(
            f"{working_prompt}\n\n(ตอบคำถามโดยตรงตามข้อมูลที่มีในบริบท ห้ามใช้แท็ก [SEARCH: ...] เด็ดขาด)"
        )
        answer = sanitize_yukino_output(
            retry_raw,
            raw_user_query,
            speaker_user,
            speaker_name,
            allow_cjk_override=is_japanese_or_social_media,
        )

    if speaker_user is not None:
        record_user_interaction(
            speaker_user,
            raw_user_query[:300],
            channel_id=channel_id,
            learned_memo=learned_memo,
        )
        # Passive Long-Term Memory Gate: non-blocking extraction when user shares a durable personal fact
        if (
            plan.get("has_durable_user_fact")
            and not learned_memo
            and not (tool_ctx and tool_ctx.memory_updated_uid)
        ):
            try:
                asyncio.get_running_loop().create_task(
                    _extract_and_save_durable_fact_bg(speaker_user, raw_user_query)
                )
            except RuntimeError:
                pass

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