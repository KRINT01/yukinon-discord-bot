import io
import re
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord
from discord import app_commands

from config import (
    ALLOWED_GUILD_IDS,
    ACTIVE_FAST_MODEL,
    ACTIVE_CODE_MODEL,
    IMAGE_EXTENSIONS,
    THAI_MONTHS,
    URL_REGEX,
    PURE_DATETIME_PATTERN,
    api_log,
    get_cached_channel_attachments,
    load_temp_image_cache,
)
from persona import (
    is_owner_user,
    FAST_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    YUKIPEDIA_SYSTEM_PROMPT,
)
from llm import (
    query_local_ollama_chat,
    orchestrate_message_with_decision_1,
)
from media_context import (
    extract_all_context,
    format_code_or_log_attachment,
    extract_image_via_transient_ocr,
    extract_pdf_context_async,
    parse_docx_bytes,
    parse_xlsx_bytes_to_markdown,
    inspect_image_details_deep_async,
)
from image_engine import (
    send_chunked_response,
    process_manga_translation_request,
)
from user_memory import (
    user_profiles,
    ensure_user_profile_exists,
    update_user_memory_fields,
    format_user_memory_card,
    process_memory_request,
    process_ot_request,
    build_monthly_ot_summary_markdown,
    _parse_stored_ot_date,
    save_user_profiles_local,
    backup_user_profiles_to_discord,
)
from tools import ToolExecutionContext

_CONTEXT_ERROR_NOTICE_REGEX = re.compile(
    r"\[(?:Web extraction error|Error reading|Attachment Size Notice|PDF Notice|OCR could not extract|Error inspecting embedded image)[^\]]*\]",
    re.IGNORECASE,
)

_FAILURE_OR_REFUSAL_REPLY_REGEX = re.compile(
    r"(ไม่สามารถเข้าถึง|ไม่สามารถดึง|ไม่สามารถอ่าน|ไม่สามารถตรวจสอบ|ไม่สามารถเชื่อมต่อได้ชั่วคราว|"
    r"ไม่สามารถสร้างรูปภาพได้|ไม่สามารถตรวจจับข้อความ|ไม่พบรายการ|ไม่พบข้อมูล(?:ที่|ใน|เพียงพอ)|"
    r"ไม่มีข้อมูลเพียงพอ|ไฟล์มีขนาดใหญ่เกิน|ไม่มีข้อความที่สามารถอ่านได้|สแกนหรือเป็นรูปภาพ|"
    r"ส่งไฟล์.*มาใหม่|แนบไฟล์.*มาใหม่|ลืมคำสั่งก่อนหน้า|ตอนนี้ระบบขัดข้องชั่วคราว|ตอนนี้วาดรูปให้ไม่ได้ชั่วคราว|"
    r"\b(?:unable to access|cannot read|could not extract|no readable (?:article )?text|shallow, transparent attempt)\b)",
    re.IGNORECASE,
)

_SUBSTANTIVE_REQUEST_REGEX = re.compile(
    r"(เปรียบเทียบ|ข้อดี|ข้อเสีย|แนะนำ|วิเคราะห์|สรุป|อธิบาย|ต่างกันยังไง|ต่างกันอย่างไร|"
    r"เลือกอะไรดี|รุ่นไหนดี|แนวทาง|ไอเดีย|วางแผน|จัดสเปค|วิจารณ์|"
    r"\b(?:compare|comparison|pros|cons|recommend|analyze|analysis|explain|difference|best|review|strategy|idea)\b)",
    re.IGNORECASE,
)

_IMAGE_TRIAGE_TAG_REGEX = re.compile(
    r"\[ImageTriage:\s*type=([a-z_]+)\s*\|\s*conf=([0-9.]+)\s*\|\s*has_japanese_dialogue=(True|False)\]",
    re.IGNORECASE,
)


def _parse_image_triage_metadata(extracted_ctx: str) -> tuple[str, float, bool | None]:
    """Extracts `(image_type, confidence, has_japanese_dialogue)` from `[ImageTriage: ...]` if present."""
    if not extracted_ctx:
        return "", 0.0, None
    m = _IMAGE_TRIAGE_TAG_REGEX.search(extracted_ctx)
    if not m:
        return "", 0.0, None
    return m.group(1).strip().lower(), float(m.group(2) or 0.0), (m.group(3).lower() == "true")


def _has_japanese_in_ocr_body_only(extracted_ctx: str) -> bool:
    """Checks for Japanese dialogue strictly inside Vision/OCR blocks (never tweet captions)."""
    if not extracted_ctx:
        return False
    ocr_sections = re.findall(
        r"===\s*(?:Vision Analysis & OCR|Unified Multi-Image Visual & OCR Analysis)[^\n]*===\n([\s\S]*?)(?=\n===|\Z)",
        extracted_ctx,
        re.IGNORECASE,
    )
    if not ocr_sections:
        return False
    return bool(re.search(r"[ぁ-んァ-ヶ一-龯]", "\n".join(ocr_sections)))


def _strip_internal_prompt_hints(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\n*\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*?\)\s*$", "", text).strip()
    return cleaned


def _extract_clean_artist_or_topic_hint(cached_ctx: str, bot_reply: str) -> str:
    """Extracts clean artist handle/name from metadata or reply for follow-up searches."""
    for src in (cached_ctx or "", bot_reply or ""):
        m_auth = re.search(r"Author / Artist:\s*([^\n(|]+)(?:\(@([A-Za-z0-9_]+)\))?", src)
        if m_auth:
            name_part = m_auth.group(1).strip()
            handle_part = (m_auth.group(2) or "").strip()
            if name_part or handle_part:
                return f"{name_part} {handle_part} Pixiv X".strip()
        m_at = re.search(r"@([A-Za-z0-9_]{3,30})", src)
        if m_at:
            return f"{m_at.group(1)} Pixiv X".strip()
    return ""


def has_valid_extracted_context(extracted_ctx: str) -> bool:
    if not extracted_ctx or not extracted_ctx.strip():
        return False
    cleaned = _CONTEXT_ERROR_NOTICE_REGEX.sub("", extracted_ctx)
    cleaned = re.sub(r"===[^=\n]+===", "", cleaned).strip()
    return len(cleaned) >= 25


def is_failure_or_refusal_reply(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return bool(_FAILURE_OR_REFUSAL_REPLY_REGEX.search(answer[:1200]))


def should_attach_interactive_view(
    user_request: str,
    answer: str,
    use_coder: bool = False,
    did_search: bool = False,
    extracted_ctx: str = "",
    is_instant_dt: bool = False,
    decision_plan: dict | None = None,
) -> bool:
    if is_instant_dt or PURE_DATETIME_PATTERN.search(user_request or ""):
        return False
    if not answer or len(answer.strip()) < 60:
        return False
    if is_failure_or_refusal_reply(answer):
        return False

    valid_ctx = has_valid_extracted_context(extracted_ctx)
    if extracted_ctx and extracted_ctx.strip() and not valid_ctx and not did_search:
        return False

    if use_coder:
        return True
    if valid_ctx or did_search:
        return True

    # Gate on attach_buttons' OWN confidence (`buttons_confidence`) only when `attach_buttons` is actually present
    if decision_plan and "attach_buttons" in decision_plan:
        btn_conf = float(
            decision_plan.get("buttons_confidence")
            if decision_plan.get("buttons_confidence") is not None
            else (decision_plan.get("route_confidence") or 0.0)
        )
        if btn_conf >= 0.65:
            return bool(decision_plan.get("attach_buttons"))

    if _SUBSTANTIVE_REQUEST_REGEX.search(user_request or ""):
        return True
    return False


def get_previous_month_info(now_th: datetime) -> tuple[int, int, str]:
    first_of_this_month = now_th.replace(day=1)
    last_day_prev_month = first_of_this_month - timedelta(days=1)
    y_ce = last_day_prev_month.year
    m_num = last_day_prev_month.month
    return y_ce, m_num, f"{THAI_MONTHS[m_num]} พ.ศ. {y_ce + 543}"


async def _safe_send_ephemeral_error(interaction: discord.Interaction, err_msg: str) -> None:
    """Safely delivers an ephemeral Thai error notice regardless of whether the interaction was deferred."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(err_msg, ephemeral=True)
        else:
            await interaction.response.send_message(err_msg, ephemeral=True)
    except Exception:
        pass


class AutoDisappearingView(discord.ui.View):
    """
    Base View that automatically removes its buttons from the Discord message
    (or deletes the standalone button footer message) when `timeout` expires,
    handles unexpected button callback exceptions cleanly, and releases its
    message references to keep Render 512MB RAM clean.
    """

    def __init__(self, timeout: float = 300.0):
        super().__init__(timeout=timeout)
        self.message: discord.Message | discord.InteractionMessage | None = None
        self.delete_message_on_timeout: bool = False

    async def on_timeout(self) -> None:
        msg = self.message
        self.message = None
        if hasattr(self, "source_message"):
            self.source_message = None
        if msg is None:
            return
        try:
            if self.delete_message_on_timeout:
                await msg.delete()
            else:
                await msg.edit(view=None)
        except Exception:
            pass

    async def on_error(
        self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item
    ) -> None:
        api_log(
            f"UI View Error ({type(self).__name__} / {getattr(item, 'label', type(item).__name__)}) -> "
            f"{type(error).__name__}: {str(error)[:160]}"
        )
        await _safe_send_ephemeral_error(
            interaction,
            f"ติดขัดนิดหน่อยค่ะ (`{type(error).__name__}`) ลองใหม่อีกทีนะคะ",
        )


async def _deliver_button_followup_cleanly(interaction: discord.Interaction, answer: str) -> None:
    """
    Delivers a button follow-up answer cleanly WITHOUT any extra filler confirmation message:
    - Short answers (<=1900 chars): resolve the deferred interaction directly via `interaction.followup.send(answer)`.
    - Long/chunked answers (>1900 chars): delete the deferred thinking placeholder silently and post via `send_chunked_response`
      (or attach as `.md` file if `interaction.channel` is unavailable).
    """
    if not answer or not answer.strip():
        try:
            await interaction.delete_original_response()
        except Exception:
            pass
        return

    if len(answer) <= 1900:
        try:
            await interaction.followup.send(answer)
            return
        except Exception:
            pass

    if interaction.channel is not None:
        try:
            await interaction.delete_original_response()
        except Exception:
            pass
        await send_chunked_response(interaction.channel, answer)
        return

    try:
        await interaction.followup.send(
            content="**คำตอบยาวเกินไป ฉันแนบเป็นไฟล์ไว้ให้ด้านล่างนะคะ:**",
            file=discord.File(io.StringIO(answer), filename="output.md"),
        )
    except Exception:
        pass


async def _send_interaction_with_auto_view(
    interaction: discord.Interaction,
    answer: str,
    view: AutoDisappearingView | None,
    chunked_header: str = "❄️ เรียบร้อยแล้วค่ะ:",
    allow_view_on_error: bool = False,
):
    """
    Delivers a deferred slash/modal response with an auto-expiring View.
    Includes a `discord.NotFound` fallback to `interaction.channel.send(...)` in case
    the original thinking placeholder message was manually deleted before completion.
    """
    if not answer or not answer.strip():
        return
    effective_view = view if (allow_view_on_error or not is_failure_or_refusal_reply(answer)) else None

    if len(answer) <= 1900:
        try:
            if effective_view is not None:
                sent_msg = await interaction.followup.send(answer, view=effective_view, wait=True)
                effective_view.message = sent_msg
                effective_view.delete_message_on_timeout = False
            else:
                await interaction.followup.send(answer)
            return
        except discord.NotFound:
            if interaction.channel is not None:
                sent_msg = await interaction.channel.send(answer, view=effective_view)
                if effective_view is not None:
                    effective_view.message = sent_msg
                    effective_view.delete_message_on_timeout = False
                return
            raise

    if interaction.channel is not None:
        try:
            await interaction.followup.send(chunked_header)
        except discord.NotFound:
            pass
        await send_chunked_response(interaction.channel, answer)
        if effective_view is not None:
            footer_msg = await interaction.channel.send("-# ตัวเลือกเพิ่มเติม:", view=effective_view)
            effective_view.message = footer_msg
            effective_view.delete_message_on_timeout = True
    else:
        kwargs = {
            "content": chunked_header,
            "file": discord.File(io.StringIO(answer), filename="output.md"),
        }
        if effective_view is not None:
            sent_msg = await interaction.followup.send(**kwargs, view=effective_view, wait=True)
            effective_view.message = sent_msg
            effective_view.delete_message_on_timeout = False
        else:
            await interaction.followup.send(**kwargs)


async def _safe_edit_or_followup_after_defer(
    interaction: discord.Interaction,
    content: str,
    view: AutoDisappearingView | None,
) -> None:
    """
    Safely updates the original message after `interaction.response.defer()`.
    Handles >1950-char cards cleanly and falls back to `followup.send` / `channel.send`
    if the original message was deleted (`404 Unknown Message`).
    """
    if len(content) <= 1950:
        try:
            edited = await interaction.edit_original_response(content=content, view=view)
            if view is not None and hasattr(view, "message"):
                view.message = edited
                view.delete_message_on_timeout = False
            return
        except Exception:
            pass
    await _send_interaction_with_auto_view(interaction, content, view, allow_view_on_error=True)


# ==============================================================================
# 1. INTERACTIVE CODE REVIEW & STACK TRACE VIEW (Code Map / Explain / Optimize)
# ==============================================================================
class CodeReviewActionView(AutoDisappearingView):
    """Interactive follow-up buttons (Code Map, Explain, Optimize) attached to coding & stack trace responses."""

    def __init__(self, author_id: int, channel_id: int, original_prompt: str, original_answer: str):
        super().__init__(timeout=300.0)
        self.author_id = author_id
        self.channel_id = channel_id
        self.original_prompt = original_prompt[:2500]
        self.original_answer = original_answer[:3500]

    async def _run_followup(self, interaction: discord.Interaction, followup_instruction: str):
        await interaction.response.defer(thinking=True)
        cached_files = get_cached_channel_attachments(self.channel_id, max_total_chars=45000)
        composed = (
            f"{followup_instruction}\n\n"
            f"### [บริบทคำถามและการวิเคราะห์ก่อนหน้า]:\n"
            f"คำถามเดิม: {self.original_prompt}\n"
            f"คำตอบที่วิเคราะห์ไว้: {self.original_answer[:2500]}\n"
        )
        if cached_files:
            composed += f"\n### [ไฟล์โค้ด/Log ที่แนบไว้ในห้องแชท]:\n{cached_files[:35000]}"

        tool_ctx = ToolExecutionContext(
            channel_id=self.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            mode="code",
        )
        answer, _ = await query_local_ollama_chat(
            composed,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=self.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=True,
            speaker_user=interaction.user,
            decision_plan={
                "route": "code",
                "route_confidence": 0.99,
                "needs_web_search": False,
                "search_confidence": 0.99,
                "needs_tool_use": False,
                "tool_confidence": 0.90,
                "complexity_score": 2.0,
                "complexity_confidence": 0.90,
            },
            tool_ctx=tool_ctx,
        )
        await _deliver_button_followup_cleanly(interaction, answer)

    @discord.ui.button(label="Code Map", style=discord.ButtonStyle.primary, emoji="🗺️")
    async def btn_code_map(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "ช่วยสร้าง **Code Map (ไล่ลำดับการทำงานทีละขั้นตอน)** จากโค้ดหรือ Stack Trace ข้างต้น:\n"
            "- หากเป็นซอร์สโค้ด (หรือหลายไฟล์ที่เชื่อมโยงกัน): ให้ไล่ Flow การทำงานตั้งแต่จุดเริ่มต้นไปจนจบเป็นลำดับขั้น (`1. -> 2. -> 3.`) "
            "ข้ามเลเยอร์ต่างๆ (เช่น Component/Controller -> Service -> `@Transactional` -> Oracle SQL / Message Queue) "
            "โดยระบุชื่อคลาส/เมธอด/ฟังก์ชัน จุดที่ข้อมูลถูกแปลงค่า และจุดที่ State เปลี่ยนแปลงอย่างชัดเจน\n"
            "- หากเป็น Stack Trace / Error Log: ให้ตัดเฟรมของ Framework ที่ไม่จำเป็นออก แล้วไล่ลำดับ Call Path จากจุดเริ่มต้นลงไปจนถึงบรรทัดต้นตอใน `Caused by:` ชั้นในสุดค่ะ",
        )

    @discord.ui.button(label="Explain", style=discord.ButtonStyle.secondary, emoji="💡")
    async def btn_explain(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "ช่วย **อธิบายการทำงานและสาเหตุ (Explain)** ให้เข้าใจง่ายและตรงประเด็น:\n"
            "- หากเป็นบล็อกโค้ดหรือคิวรี SQL/PL/SQL: อธิบายตรรกะทีละส่วนแบบเข้าใจง่าย (รับ Input อะไร ประมวลผลเงื่อนไขอย่างไร คืนค่า Output อะไร และมี Edge Case ตรงไหนที่ต้องระวัง)\n"
            "- หากเป็น Stack Trace / Error Log: อธิบายกลไกเบื้องหลังของ Root Cause (เช่น ทำไมถึงเกิด Exception, รหัส `ORA-`, Spring Bean error, หรือ K8s `OOMKilled`/`CrashLoopBackOff`) ให้เห็นภาพชัดเจนโดยไม่ใช้ศัพท์อ้อมค้อมค่ะ",
        )

    @discord.ui.button(label="Optimize", style=discord.ButtonStyle.secondary, emoji="⚡")
    async def btn_optimize(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "ช่วย **ปรับปรุงประสิทธิภาพและโครงสร้าง (Optimize)** ของโค้ด คิวรี SQL หรือคอนฟิกข้างต้น:\n"
            "- วิเคราะห์จุดคอขวด (เช่น Oracle SQL Execution Plan, Missing Foreign Key/Composite Index, Implicit Type Conversion, JPA/Hibernate N+1, ลูปซ้ำซ้อน, Blocking Async I/O, React Re-render หรือ Resource Limits)\n"
            "- มอบโค้ด/SQL/คอนฟิกที่ Refactor และ Optimize แล้วให้พร้อมนำไปใช้งานจริง โดยระบุชื่อไฟล์ไว้ในบรรทัดแรกของบล็อกโค้ดด้วยค่ะ",
        )


# ==============================================================================
# 2. DYNAMIC DOMAIN-TAILORED SUGGESTION VIEW (PHASE 4: DECISION-1 TOP-K RANKING)
# ==============================================================================
# Each entry: key -> (label, emoji, instruction, use_search)
FOLLOWUP_ACTION_REGISTRY: dict[str, tuple[str, str, str, bool]] = {
    "pros_cons": (
        "เทียบข้อดี-ข้อเสีย",
        "⚖️",
        "ช่วยวิเคราะห์ข้อดี ข้อเสีย ความคุ้มค่า และข้อควรระวังของเรื่องนี้ในมุมมองที่ตรงไปตรงมาสไตล์ยูกิโนะให้หน่อยค่ะ",
        True,
    ),
    "summary_3": (
        "สรุปสั้น 3 ข้อ",
        "📌",
        "ช่วยสรุปใจความสำคัญและข้อสรุปหลักจากคำตอบ/เนื้อหาเมื่อครู่ให้กระชับที่สุดเพียง 3 ข้อ (Bullet points) ค่ะ",
        False,
    ),
    "action_checklist": (
        "สรุปสิ่งที่ต้องทำ",
        "📋",
        "ช่วยสรุปขั้นตอนที่ต้องทำ เงื่อนไขสำคัญ หรือรายการเช็กลิสต์ที่เอาไปใช้ต่อได้ทันทีจากเนื้อหาข้างต้นค่ะ",
        False,
    ),
    "deep_dive_details": (
        "เจาะลึกรายละเอียดสำคัญ",
        "🔍",
        "ช่วยขยายความรายละเอียดเชิงลึก ตัวเลข เงื่อนไข หรือจุดที่มักถูกมองข้ามจากเนื้อหาข้างต้นให้ชัดเจนขึ้นค่ะ",
        False,
    ),
    "thai_price_check": (
        "เช็ครุ่น & ราคาในไทย",
        "🏷️",
        "จากสินค้า โมเดล หรือฟิกเกอร์ที่ระบุไว้ในคำตอบก่อนหน้า ช่วยค้นหาข้อมูลรุ่นที่วางจำหน่ายจริง "
        "ไลน์สินค้า/สเกลที่เป็นไปได้ และราคาตลาดโดยประมาณในไทย (เช่น Shopee / ร้านโมเดล / ราคาเปิดตัว) ให้หน่อยค่ะ",
        True,
    ),
    "compare_alternatives": (
        "เทียบสเกล / ตัวเลือกอื่น",
        "💡",
        "ช่วยเปรียบเทียบความแตกต่างระหว่างสเกล รุ่นย่อย หรือตัวเลือกอื่นที่ใกล้เคียงกัน "
        "(เช่น รายละเอียด สเปค ความคุ้มค่า และข้อควรระวัง) แบบตรงไปตรงมาสไตล์ยูกิโนะค่ะ",
        True,
    ),
    "game_build_tips": (
        "แนะนำแนวทางเล่น / Build",
        "🎮",
        "ช่วยแนะนำแนวทางการเล่น การจัดสาย/อุปกรณ์ หรือเคล็ดลับสำคัญสำหรับผู้เล่นในเรื่องนี้พร้อมเช็คข้อมูลล่าสุดให้หน่อยค่ะ",
        True,
    ),
    "news_impact": (
        "เช็คผลกระทบ & แนวโน้ม",
        "📈",
        "ช่วยค้นหาอัปเดตล่าสุดและวิเคราะห์ผลกระทบ แนวโน้ม หรือข้อมูลเบื้องหลังเพิ่มเติมของประเด็นนี้ให้หน่อยค่ะ",
        True,
    ),
    "character_lore_jokes": (
        "เจาะลึกตัวละคร & มุก",
        "📖",
        "ช่วยอธิบายเจาะลึกเฉพาะความสัมพันธ์ของตัวละครในเรื่อง (Canon lore/นิสัยตัวละคร) และเกร็ดคำศัพท์หรือมุกภาษาญี่ปุ่นในฉากนี้เพิ่มเติม "
        "โดยไม่ต้องแปลบทสนทนาซ้ำทั้งหมดค่ะ",
        False,
    ),
    "find_artist_works": (
        "หาผลงานอื่นของศิลปิน",
        "🔎",
        "ช่วยค้นหาข้อมูลเพิ่มเติมเกี่ยวกับศิลปิน ผู้วาด (บัญชี X / Pixiv) หรือผลงาน/ซีรีส์ที่เกี่ยวข้องกับโพสต์นี้ให้หน่อยค่ะ",
        True,
    ),
}


def _select_dynamic_button_specs(
    domain: str,
    button_scores: dict[str, float] | None,
    has_foreign_dialogue: bool,
    has_artist_or_social: bool,
    triage_type: str,
) -> list[tuple[str, str, discord.ButtonStyle, str, bool]] | None:
    """
    Phase 4 Dynamic Button Ranker:
    Selects the top 2-3 follow-up buttons using Decision-1's calibrated `button_scores`
    (scored at 0 extra latency during the main orchestration pass), while preserving
    structural pins (`🎨 แปลลงภาพ (In-Bubble)` for visual Japanese comic dialogue) and
    context-aware eligibility guardrails. Returns `None` to fall back to static domain
    defaults if Decision-1 scores are absent or fewer than 2 strong actions qualify.
    """
    if not isinstance(button_scores, dict) or not button_scores:
        return None

    specs: list[tuple[str, str, discord.ButtonStyle, str, bool]] = []
    used_keys: set[str] = set()

    # Structural pin: Japanese comic/manga dialogue ALWAYS gets In-Bubble as primary Button #1
    if domain == "social_art" and has_foreign_dialogue:
        specs.append((
            "แปลลงภาพ (In-Bubble)",
            "🎨",
            discord.ButtonStyle.primary,
            "__IN_BUBBLE_MANGA_TH__",
            False,
        ))
        used_keys.add("__in_bubble__")

    scored_candidates: list[tuple[float, str]] = []
    for key, raw_p in button_scores.items():
        if key not in FOLLOWUP_ACTION_REGISTRY or key in used_keys:
            continue
        try:
            prob = float(raw_p)
        except Exception:
            continue

        # Context-aware eligibility guardrails to keep buttons sharp and relevant
        if key == "find_artist_works":
            if not has_artist_or_social or domain == "product" or triage_type == "real_world_product":
                continue
        if key == "character_lore_jokes":
            if domain not in ("social_art", "game") and prob < 0.75:
                continue
            if triage_type == "real_world_product" and prob < 0.78:
                continue
        if key == "game_build_tips" and domain != "game" and prob < 0.75:
            continue
        if key == "thai_price_check" and domain not in ("product", "general") and prob < 0.75:
            continue
        if key == "news_impact" and domain != "news" and prob < 0.72:
            continue

        if prob >= 0.55:
            scored_candidates.append((prob, key))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, key in scored_candidates:
        if len(specs) >= 3:
            break
        label, emoji, instruction, use_search = FOLLOWUP_ACTION_REGISTRY[key]
        style = discord.ButtonStyle.primary if len(specs) == 0 else discord.ButtonStyle.secondary
        specs.append((label, emoji, style, instruction, use_search))
        used_keys.add(key)

    if len(specs) < 2:
        return None
    return specs


class GeneralAssistantActionView(AutoDisappearingView):
    def __init__(
        self,
        channel_id: int,
        user_query: str,
        bot_reply: str,
        button_specs: list[tuple[str, str, discord.ButtonStyle, str, bool]],
        source_message: discord.Message | None = None,
    ):
        super().__init__(timeout=300.0)
        self.channel_id = channel_id
        self.user_query = _strip_internal_prompt_hints(user_query)[:1200]
        self.bot_reply = bot_reply[:2500]
        self.source_message = source_message

        for label, emoji, style, action_spec, use_search in button_specs[:3]:
            btn = discord.ui.Button(label=label[:34], emoji=emoji, style=style)
            if action_spec == "__IN_BUBBLE_MANGA_TH__":
                btn.callback = self._make_in_bubble_manga_callback()
            else:
                btn.callback = self._make_followup_callback(label, action_spec, use_search)
            self.add_item(btn)

    def _make_in_bubble_manga_callback(self):
        async def _callback(interaction: discord.Interaction):
            await interaction.response.send_message(
                "เดี๋ยวฉันแปลหน้านี้เป็นภาษาไทยให้ รอสักครู่นะคะ"
            )
            target_msg = self.source_message
            if target_msg is None:
                class _FallbackLinkMsg:
                    def __init__(self, q_txt: str, ch, usr):
                        self.id = 0
                        self.content = q_txt
                        self.attachments = []
                        self.embeds = []
                        self.stickers = []
                        self.reference = None
                        self.channel = ch
                        self.author = usr

                    async def reply(self, **kwargs):
                        return await self.channel.send(**kwargs)

                target_msg = _FallbackLinkMsg(self.user_query, interaction.channel, interaction.user)

            try:
                await process_manga_translation_request(target_msg, "แปลมังงะเป็นภาษาไทย")
            finally:
                try:
                    await interaction.delete_original_response()
                except Exception:
                    pass

        return _callback

    def _make_followup_callback(self, btn_label: str, instruction: str, use_search: bool):
        async def _callback(interaction: discord.Interaction):
            await self._run_general_followup(interaction, btn_label, instruction, use_search=use_search)
        return _callback

    async def _run_general_followup(
        self,
        interaction: discord.Interaction,
        btn_label: str,
        instruction: str,
        use_search: bool = True,
    ):
        await interaction.response.defer(thinking=True)
        cached_ctx = get_cached_channel_attachments(self.channel_id, max_total_chars=25000)

        is_deep_img = (instruction == "__DEEP_IMAGE_INSPECT__")
        deep_visual_block = ""
        if is_deep_img:
            instruction = (
                "ช่วยเจาะลึกองค์ประกอบภาพ มุมกล้อง แสงเงา ลายเส้น รายละเอียดเครื่องแต่งกาย/อุปกรณ์ "
                "และจุดน่าสนใจที่ปรากฏจริงในภาพนี้เพิ่มเติมจากคำตอบก่อนหน้า "
                "(กล่าวถึงเฉพาะสิ่งที่เห็นจริงในกรอบภาพเท่านั้น ห้ามแต่งเติมใบหน้า สีหน้า ดวงตา หรือส่วนที่ถูกครอปออกนอกเฟรมเด็ดขาด)"
            )
            candidate_msg_ids: list[int] = []
            if self.source_message is not None:
                sm_id = int(getattr(self.source_message, "id", 0) or 0)
                if sm_id:
                    candidate_msg_ids.append(sm_id)
                ref_obj = getattr(self.source_message, "reference", None)
                ref_id = int(getattr(ref_obj, "message_id", 0) or 0) if ref_obj is not None else 0
                if ref_id and ref_id not in candidate_msg_ids:
                    candidate_msg_ids.append(ref_id)

            for cid_msg in candidate_msg_ids:
                cached_imgs = await asyncio.to_thread(load_temp_image_cache, cid_msg)
                if cached_imgs and cached_imgs[0]:
                    imgs_list, meta_secs = cached_imgs
                    deep_vis = await inspect_image_details_deep_async(
                        imgs_list,
                        prev_reply=self.bot_reply,
                        post_meta_hint="\n".join(meta_secs),
                    )
                    if deep_vis:
                        deep_visual_block = f"\n\n=== Fresh Deep Visual Inspection (Ground Truth) ===\n{deep_vis}"
                    break

        search_hint_tag = ""
        if use_search:
            artist_hint = _extract_clean_artist_or_topic_hint(cached_ctx, self.bot_reply)
            if artist_hint and ("ศิลปิน" in btn_label or "ผลงาน" in btn_label):
                search_hint_tag = f"\n[SEARCH_HINT: {artist_hint}]"

        # Deterministic button plan — skips redundant ~350ms Decision-1 `ButtonFollowup` round-trip!
        dec_plan = {
            "route": "web_search" if use_search else "chat",
            "route_confidence": 0.95,
            "needs_web_search": use_search,
            "search_confidence": 0.95,
            "needs_tool_use": use_search,
            "tool_confidence": 0.95,
            "complexity_score": 1.6 if use_search else 1.3,
            "complexity_confidence": 0.90,
        }

        extra_grounding_rule = (
            " และห้ามแต่งเติมรายละเอียดของภาพ (เช่น ใบหน้า สีหน้า ดวงตา หรือส่วนที่ถูกครอปออกนอกเฟรม) "
            "ที่ไม่ได้ปรากฏจริงในข้อมูลภาพด้านล่างเด็ดขาด"
            if is_deep_img
            else ""
        )
        composed = (
            f"{instruction}{search_hint_tag}\n\n"
            f"[กฎสำคัญ]: ห้ามคัดลอกหรือเขียนคำตอบเดิมซ้ำ ห้ามสร้างตาราง และห้ามเขียนหัวข้อเมตาของตัวละครเด็ดขาด"
            f"{extra_grounding_rule}\n\n"
            f"[บริบทหัวข้อก่อนหน้า]: {self.user_query}\n"
            f"[สรุปคำตอบก่อนหน้า (ห้ามเขียนซ้ำ)]: {self.bot_reply[:1500]}"
        )
        if deep_visual_block:
            composed += deep_visual_block
        if cached_ctx:
            # Always include compact visual/document context so product/model-kit follow-ups know the exact identified entity
            ctx_limit = 18000 if not use_search else 5000
            composed += f"\n\n[เนื้อหาอ้างอิงจากไฟล์/ลิงก์ในห้องแชท]:\n{cached_ctx[:ctx_limit]}"

        tool_ctx = ToolExecutionContext(
            channel_id=self.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            mode="general",
        )
        answer, _ = await query_local_ollama_chat(
            composed,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT if use_search else FAST_SYSTEM_PROMPT,
            channel_id=self.channel_id,
            temperature=0.25 if is_deep_img else 0.45,
            allow_agentic_search=use_search,
            has_fresh_attachment=bool(cached_ctx or deep_visual_block),
            speaker_user=interaction.user,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
        await _deliver_button_followup_cleanly(interaction, answer)


def build_contextual_suggestion_view(
    channel_id: int,
    user_query: str,
    bot_reply: str,
    extracted_ctx: str = "",
    source_message: discord.Message | None = None,
    decision_plan: dict | None = None,
) -> GeneralAssistantActionView:
    ctx_lower = (extracted_ctx or "").lower()
    combined = f"{user_query} {bot_reply}".lower()
    triage_type, triage_conf, triage_has_jp = _parse_image_triage_metadata(extracted_ctx)

    # Gate on button_domain's OWN confidence (`domain_confidence`) rather than `route_confidence`
    dec_domain = ""
    if decision_plan:
        dom_conf = float(
            decision_plan.get("domain_confidence")
            if decision_plan.get("domain_confidence") is not None
            else (decision_plan.get("route_confidence") or 0.0)
        )
        if dom_conf >= 0.65:
            dec_domain = str(decision_plan.get("button_domain") or "").strip().lower()

    # Physical product / scale model kit / collectible photo -> dedicated product domain (never manga buttons!)
    if triage_type == "real_world_product" and triage_conf >= 0.65:
        domain = "product"
    elif dec_domain in ("document", "social_art", "game", "news"):
        domain = dec_domain
    elif any(
        tag in ctx_lower
        for tag in (
            "=== attached pdf",
            "=== attached scanned pdf",
            "=== attached word",
            "=== attached excel",
            "=== youtube video information",
            "<untrusted_web_search_results url=",
        )
    ):
        domain = "document"
    elif any(
        tag in ctx_lower
        for tag in ("=== embedded x", "=== embedded pixiv", "=== embedded bluesky", "=== image", "=== unified multi-image")
    ) or any(k in combined for k in ("pixiv", "twitter", "x.com", "ศิลปิน", "ผู้วาด", "อนิเมะ", "มังงะ", "anime", "manga")):
        domain = "social_art"
    elif any(k in combined for k in ("เกม", "game", "steam", "pso2", "ngs", "build", "คลาส", "class", "เมต้า", "meta")):
        domain = "game"
    elif any(k in combined for k in ("ข่าว", "news", "ราคา", "หุ้น", "ทอง", "บิตคอยน์", "อัปเดต")):
        domain = "news"
    else:
        domain = "general"

    # Trust ImageTriage's visual dialogue flag first; only fall back to OCR body (never tweet captions!)
    if triage_has_jp is not None:
        has_foreign_dialogue = triage_has_jp
    else:
        has_foreign_dialogue = _has_japanese_in_ocr_body_only(extracted_ctx)

    has_artist_or_social = bool(
        any(
            tag in ctx_lower
            for tag in ("=== embedded x", "=== embedded pixiv", "=== embedded bluesky", "author / artist:", "artist / illustrator:")
        )
        or domain == "social_art"
    )

    # Phase 4: Prefer Decision-1 dynamic top-K button ranking when calibrated `button_scores` are available
    dynamic_specs = _select_dynamic_button_specs(
        domain=domain,
        button_scores=(decision_plan or {}).get("button_scores"),
        has_foreign_dialogue=has_foreign_dialogue,
        has_artist_or_social=has_artist_or_social,
        triage_type=triage_type,
    )
    if dynamic_specs:
        return GeneralAssistantActionView(
            channel_id, user_query, bot_reply, dynamic_specs, source_message=source_message
        )

    # Fallback static domain-tailored specs (when Decision-1 was unavailable or on cooldown)
    if domain == "product":
        specs = [
            (
                "เช็ครุ่น & ราคาในไทย",
                "🏷️",
                discord.ButtonStyle.primary,
                "จากสินค้า โมเดล หรือฟิกเกอร์ที่ระบุไว้ในคำตอบก่อนหน้า ช่วยค้นหาข้อมูลรุ่นที่วางจำหน่ายจริง "
                "ไลน์สินค้า/สเกลที่เป็นไปได้ และราคาตลาดโดยประมาณในไทย (เช่น Shopee / ร้านโมเดล / ราคาเปิดตัว) ให้หน่อยค่ะ",
                True,
            ),
            (
                "เทียบสเกล / รุ่นใกล้เคียง",
                "⚖️",
                discord.ButtonStyle.secondary,
                "ช่วยเปรียบเทียบความแตกต่างระหว่างสเกล รุ่นย่อย หรือไลน์สินค้าที่ใกล้เคียงกันของสิ่งนี้ "
                "(เช่น รายละเอียดงาน ข้อต่อ ขนาด และความคุ้มค่า) แบบตรงไปตรงมาสไตล์ยูกิโนะค่ะ",
                True,
            ),
            (
                "สรุปจุดเด่น & รายละเอียด",
                "📌",
                discord.ButtonStyle.secondary,
                "ช่วยสรุปจุดเด่นทางกายภาพ สเปค หรือดีเทลที่น่าสนใจของสิ่งนี้จากภาพออกมาเป็น 3 ข้อสั้นๆ กระชับๆ ค่ะ",
                False,
            ),
        ]
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    if domain == "document":
        specs = [
            (
                "สรุปประเด็นสำคัญ 3 ข้อ",
                "📌",
                discord.ButtonStyle.primary,
                "ช่วยสรุปใจความสำคัญและข้อสรุปหลักจากเอกสาร/เนื้อหานี้ให้กระชับที่สุดเพียง 3 ข้อค่ะ",
                False,
            ),
            (
                "สรุปสิ่งที่ต้องทำ",
                "📋",
                discord.ButtonStyle.secondary,
                "ช่วยสรุปขั้นตอนที่ต้องทำ เงื่อนไขสำคัญ หรือรายการเช็กลิสต์ที่เอาไปใช้ต่อได้ทันทีจากเอกสาร/เนื้อหานี้ค่ะ",
                False,
            ),
            (
                "เจาะลึกรายละเอียดสำคัญ",
                "🔍",
                discord.ButtonStyle.secondary,
                "ช่วยขยายความรายละเอียดเชิงลึก ตัวเลข เงื่อนไข หรือจุดที่มักถูกมองข้ามในเอกสาร/เนื้อหานี้ให้ชัดเจนขึ้นค่ะ",
                False,
            ),
        ]
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    if domain == "social_art":
        specs = []
        if has_foreign_dialogue:
            specs.append(
                (
                    "แปลลงภาพ (In-Bubble)",
                    "🎨",
                    discord.ButtonStyle.primary,
                    "__IN_BUBBLE_MANGA_TH__",
                    False,
                )
            )
            specs.append(
                (
                    "เจาะลึกตัวละคร & มุก",
                    "📖",
                    discord.ButtonStyle.secondary,
                    "ช่วยอธิบายเจาะลึกเฉพาะความสัมพันธ์ของตัวละครในเรื่อง (Canon lore/นิสัยตัวละคร) และเกร็ดคำศัพท์หรือมุกภาษาญี่ปุ่นในฉากนี้เพิ่มเติม "
                    "โดยไม่ต้องแปลบทสนทนาซ้ำทั้งหมดค่ะ",
                    False,
                )
            )
        else:
            specs.append(
                (
                    "ดูรายละเอียดภาพเพิ่ม",
                    "📌",
                    discord.ButtonStyle.primary,
                    "__DEEP_IMAGE_INSPECT__",
                    False,
                )
            )

        specs.append(
            (
                "หาผลงานอื่นของศิลปิน",
                "🔍",
                discord.ButtonStyle.secondary,
                "ช่วยค้นหาข้อมูลเพิ่มเติมเกี่ยวกับศิลปิน ผู้วาด (บัญชี X / Pixiv) หรือผลงาน/ซีรีส์ที่เกี่ยวข้องกับโพสต์นี้ให้หน่อยค่ะ",
                True,
            )
        )
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    if domain == "game":
        specs = [
            (
                "แนะนำแนวทางเล่น / Build",
                "🎮",
                discord.ButtonStyle.primary,
                "ช่วยแนะนำแนวทางการเล่น การจัดสาย/อุปกรณ์ หรือเคล็ดลับสำคัญสำหรับผู้เล่นในเรื่องนี้พร้อมเช็คข้อมูลล่าสุดให้หน่อยค่ะ",
                True,
            ),
            (
                "เทียบข้อดี-ข้อเสีย",
                "⚖️",
                discord.ButtonStyle.secondary,
                "ช่วยวิเคราะห์จุดเด่น ข้อด้อย และความคุ้มค่าของเกม/สายการเล่นนี้ในมุมมองที่ตรงไปตรงมาสไตล์ยูกิโนะให้หน่อยค่ะ",
                True,
            ),
            (
                "สรุปสั้น 3 ข้อ",
                "📌",
                discord.ButtonStyle.secondary,
                "ช่วยสรุปใจความสำคัญจากคำตอบเมื่อครู่ให้กระชับที่สุดเพียง 3 ข้อค่ะ",
                False,
            ),
        ]
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    if domain == "news":
        specs = [
            (
                "เช็คผลกระทบ & แนวโน้ม",
                "📈",
                discord.ButtonStyle.primary,
                "ช่วยค้นหาอัปเดตล่าสุดและวิเคราะห์ผลกระทบหรือแนวโน้มที่อาจจะเกิดขึ้นต่อจากข่าวนี้ให้หน่อยค่ะ",
                True,
            ),
            (
                "สรุปสั้น 3 ข้อ",
                "📌",
                discord.ButtonStyle.secondary,
                "ช่วยสรุปข้อเท็จจริงสำคัญจากข่าวนี้ออกมาเป็น 3 ข้อสั้นๆ อ่านจบใน 10 วินาทีค่ะ",
                False,
            ),
            (
                "เช็คข้อมูลเพิ่มเติม",
                "🔍",
                discord.ButtonStyle.secondary,
                "ช่วยค้นหาข้อมูลเบื้องหลังและเช็คข้อเท็จจริงเพิ่มเติมของประเด็นนี้ให้หน่อยค่ะ",
                True,
            ),
        ]
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    specs = [
        (
            "เทียบข้อดี-ข้อเสีย",
            "⚖️",
            discord.ButtonStyle.primary,
            "ช่วยวิเคราะห์ข้อดี ข้อเสีย ความคุ้มค่า และข้อควรระวังของเรื่องนี้ในมุมมองที่ตรงไปตรงมาสไตล์ยูกิโนะให้หน่อยค่ะ",
            True,
        ),
        (
            "เสนอตัวเลือก / มุมมองเพิ่ม",
            "💡",
            discord.ButtonStyle.secondary,
            "จากเรื่องที่คุยกันเมื่อครู่ ช่วยเสนอทางเลือกอื่นที่น่าสนใจกว่า หรือมุมมองและข้อควรระวังที่ฉันควรรู้เพิ่มเติมให้หน่อยค่ะ",
            True,
        ),
        (
            "สรุปสั้น 3 ข้อ",
            "📌",
            discord.ButtonStyle.secondary,
            "ช่วยสรุปใจความสำคัญจากคำตอบเมื่อครู่ให้กระชับที่สุดเพียง 3 ข้อ (Bullet points) ค่ะ",
            False,
        ),
    ]
    return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)


# ==============================================================================
# 3. INTERACTIVE PERSISTENT USER MEMORY MODAL & ACTION BUTTONS (5-Field Structured)
# ==============================================================================
class MemoryEditModal(discord.ui.Modal, title="แก้ไขความจำถาวร (User Memory)"):
    # All modal TextInput labels strictly <= 45 chars (Discord Form Body limit!)
    display_name_input = discord.ui.TextInput(
        label="ชื่อที่แสดงในความจำ (Display Name)",
        placeholder="เช่น Partner (Owner) หรือชื่อเล่น",
        required=True,
        max_length=36,
    )
    pinned_facts_input = discord.ui.TextInput(
        label="ข้อมูลถาวรที่ปักหมุด (แยกบรรทัดหรือ ;)",
        style=discord.TextStyle.paragraph,
        placeholder="ข้อมูลถาวรที่ไม่ให้ถูกลบอัตโนมัติ (เช่น สเปคคอม, กาแฟที่ชอบ, งานที่ทำ)",
        required=False,
        max_length=900,
    )
    favorite_topics_input = discord.ui.TextInput(
        label="หัวข้อที่ชอบ (คั่นด้วยจุลภาค ,)",
        style=discord.TextStyle.paragraph,
        placeholder="เช่น PSO2 NGS, Anime & Manga, Software & Coding",
        required=False,
        max_length=400,
    )
    custom_notes_input = discord.ui.TextInput(
        label="บันทึกระยะสั้น (แยกบรรทัดหรือ ;)",
        style=discord.TextStyle.paragraph,
        placeholder="พิมพ์บันทึกระยะสั้นทีละบรรทัด หรือคั่นด้วย ;",
        required=False,
        max_length=800,
    )
    recent_topics_input = discord.ui.TextInput(
        label="เรื่องที่คุยล่าสุด (แยกบรรทัดหรือ ;)",
        style=discord.TextStyle.paragraph,
        placeholder="พิมพ์หัวข้อที่คุยล่าสุดทีละบรรทัด หรือคั่นด้วย ;",
        required=False,
        max_length=500,
    )

    def __init__(self, target_uid: str):
        super().__init__()
        self.target_uid = str(target_uid)
        prof = ensure_user_profile_exists(self.target_uid)
        # Clamp every default strictly to its TextInput max_length to prevent HTTP 400
        self.display_name_input.default = str(prof.get("display_name") or "")[:36]
        self.pinned_facts_input.default = "\n".join(prof.get("pinned_facts") or [])[:900]
        self.favorite_topics_input.default = ", ".join(prof.get("favorite_topics") or [])[:400]
        self.custom_notes_input.default = "\n".join(prof.get("custom_notes") or [])[:800]
        self.recent_topics_input.default = "\n".join(prof.get("recent_topics") or [])[:500]

    async def on_submit(self, interaction: discord.Interaction):
        # Defer immediately BEFORE Discord cloud backup so slow network never hits the 3-second limit!
        await interaction.response.defer(thinking=True)
        new_disp = self.display_name_input.value.strip()
        raw_pinned = self.pinned_facts_input.value.strip()
        raw_favs = self.favorite_topics_input.value.strip()
        raw_notes = self.custom_notes_input.value.strip()
        raw_recents = self.recent_topics_input.value.strip()

        new_pinned = [x.strip() for x in re.split(r"[;\n]+", raw_pinned) if x.strip()] if raw_pinned else []
        new_favs = [x.strip() for x in re.split(r"[,;،\n]+", raw_favs) if x.strip()] if raw_favs else []
        new_notes = [x.strip() for x in re.split(r"[;\n]+", raw_notes) if x.strip()] if raw_notes else []
        new_recents = [x.strip() for x in re.split(r"[;\n]+", raw_recents) if x.strip()] if raw_recents else []

        await update_user_memory_fields(
            self.target_uid,
            display_name=new_disp or None,
            pinned_facts=new_pinned,
            favorite_topics=new_favs,
            custom_notes=new_notes,
            recent_topics=new_recents,
        )

        card_text = format_user_memory_card(
            self.target_uid,
            header_title="บันทึกการแก้ไขความจำถาวร (#yukinon-memory) เรียบร้อยแล้วค่ะ",
        )
        view = MemoryActionView(requester_id=interaction.user.id, target_uid=self.target_uid)
        await _send_interaction_with_auto_view(
            interaction,
            card_text,
            view,
            chunked_header="❄️ บันทึกการแก้ไขความจำเรียบร้อยแล้วค่ะ:",
            allow_view_on_error=True,
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        api_log(f"MemoryEditModal Error -> {type(error).__name__}: {str(error)[:160]}")
        await _safe_send_ephemeral_error(
            interaction,
            f"บันทึกความจำติดขัดนิดหน่อยค่ะ (`{type(error).__name__}`) ลองใหม่อีกทีนะคะ",
        )


class MemoryActionView(AutoDisappearingView):
    def __init__(self, requester_id: int, target_uid: str):
        super().__init__(timeout=300.0)
        self.requester_id = int(requester_id)
        self.target_uid = str(target_uid)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id and not is_owner_user(interaction.user):
            await interaction.response.send_message(
                "ปุ่มนี้ให้เฉพาะเจ้าของข้อมูลกดเท่านั้นนะคะ", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="แก้ไขความจำ (Edit)", style=discord.ButtonStyle.primary, emoji="✏️")
    async def btn_edit_memory(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(MemoryEditModal(target_uid=self.target_uid))

    @discord.ui.button(label="ล้างเรื่องล่าสุด", style=discord.ButtonStyle.secondary, emoji="🧹")
    async def btn_clear_recent(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await update_user_memory_fields(self.target_uid, recent_topics=[])
        card_text = format_user_memory_card(
            self.target_uid, header_title="ล้างรายการเรื่องที่คุยล่าสุดเรียบร้อยแล้วค่ะ"
        )
        await _safe_edit_or_followup_after_defer(interaction, card_text, self)

    @discord.ui.button(label="รีเซ็ตความจำ", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def btn_reset_memory(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await update_user_memory_fields(
            self.target_uid,
            pinned_facts=[],
            favorite_topics=[],
            custom_notes=[],
            recent_topics=[],
        )
        card_text = format_user_memory_card(
            self.target_uid, header_title="รีเซ็ตความจำถาวรเรียบร้อยแล้วค่ะ (คงข้อมูล OT ไว้ให้ครบ)"
        )
        await _safe_edit_or_followup_after_defer(interaction, card_text, self)


def parse_memory_record_metadata(mem_reply_text: str) -> str:
    m = re.search(r"\[MEM_RECORD:\s*uid=(\d+)\s*\]", mem_reply_text or "")
    return m.group(1).strip() if m else ""


# ==============================================================================
# 4. INTERACTIVE OVERTIME (OT) MODAL & ACTION BUTTONS
# ==============================================================================
class OTEditModal(discord.ui.Modal, title="แก้ไขรายการบันทึก OT"):
    date_input = discord.ui.TextInput(
        label="วันที่ (DD/MM/YYYY พ.ศ. หรือ ค.ศ.)",
        placeholder="เช่น 17/11/2569",
        required=True,
        max_length=16,
    )
    time_input = discord.ui.TextInput(
        label="ช่วงเวลา OT (HH.MM - HH.MM)",
        placeholder="เช่น 18.00 - 21.00",
        required=True,
        max_length=20,
    )
    detail_input = discord.ui.TextInput(
        label="รายละเอียดงาน",
        style=discord.TextStyle.paragraph,
        placeholder="ระบุรายละเอียดงานที่ทำล่วงเวลา...",
        required=True,
        max_length=300,
    )

    def __init__(self, default_date: str = "", default_slot: str = "", default_detail: str = ""):
        super().__init__()
        if default_date:
            self.date_input.default = str(default_date)[:16]
        if default_slot:
            self.time_input.default = str(default_slot)[:20]
        if default_detail:
            self.detail_input.default = str(default_detail)[:300]

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        cmd_str = f"จด OT {self.date_input.value.strip()} {self.time_input.value.strip()} {self.detail_input.value.strip()}"

        class _ProxyMsg:
            def __init__(self, inter: discord.Interaction):
                self.author = inter.user
                self.channel = inter.channel
                self.guild = inter.guild

        reply = await process_ot_request(_ProxyMsg(interaction), cmd_str)
        rec_date, rec_slot, rec_detail = parse_ot_record_metadata(reply)
        view = (
            OTActionView(
                owner_id=interaction.user.id,
                record_date=rec_date,
                record_slot=rec_slot,
                record_detail=rec_detail,
            )
            if rec_date and not is_failure_or_refusal_reply(reply)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction,
            reply,
            view,
            chunked_header="❄️ อัปเดตเวลา OT เรียบร้อยแล้วค่ะ:",
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        api_log(f"OTEditModal Error -> {type(error).__name__}: {str(error)[:160]}")
        await _safe_send_ephemeral_error(
            interaction,
            f"บันทึกรายการ OT ติดขัดนิดหน่อยค่ะ (`{type(error).__name__}`) ลองใหม่อีกทีนะคะ",
        )


class OTActionView(AutoDisappearingView):
    def __init__(self, owner_id: int, record_date: str = "", record_slot: str = "", record_detail: str = ""):
        super().__init__(timeout=300.0)
        self.owner_id = owner_id
        self.record_date = record_date
        self.record_slot = record_slot
        self.record_detail = record_detail

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("ปุ่มนี้ให้เฉพาะเจ้าของรายการ OT กดเท่านั้นนะคะ", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="ดูตาราง OT เดือนนี้", style=discord.ButtonStyle.primary, emoji="📊")
    async def btn_monthly_table(self, interaction: discord.Interaction, button: discord.ui.Button):
        uid_str = str(interaction.user.id)
        ot_logs = (user_profiles.get(uid_str) or {}).get("ot_logs") or []
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        target_y, target_m = now_th.year, now_th.month
        if self.record_date:
            pk = _parse_stored_ot_date(self.record_date)
            if pk:
                target_y, target_m = pk[0], pk[1]
        summary_md = build_monthly_ot_summary_markdown(ot_logs, target_y, target_m)
        if len(summary_md) <= 1950:
            await interaction.response.send_message(summary_md, ephemeral=True)
        else:
            await interaction.response.send_message(
                content=f"📊 ตาราง OT เดือน `{target_m:02d}/{target_y + 543}` ยาวเกิน 2,000 ตัวอักษร ฉันแนบเป็นไฟล์ไว้ให้นะคะ:",
                file=discord.File(
                    io.StringIO(summary_md),
                    filename=f"ot_summary_{target_m:02d}_{target_y + 543}.md",
                ),
                ephemeral=True,
            )

    @discord.ui.button(label="แก้ไขรายการนี้", style=discord.ButtonStyle.secondary, emoji="✏️")
    async def btn_edit_ot(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(
            OTEditModal(
                default_date=self.record_date,
                default_slot=self.record_slot,
                default_detail=self.record_detail,
            )
        )

    @discord.ui.button(label="ลบรายการนี้", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def btn_delete_ot(self, interaction: discord.Interaction, button: discord.ui.Button):
        uid_str = str(interaction.user.id)
        prof = user_profiles.get(uid_str)
        if not prof or not self.record_date:
            await interaction.response.send_message("ไม่พบข้อมูลรายการที่จะลบค่ะ", ephemeral=True)
            return
        await interaction.response.defer()
        target_key = _parse_stored_ot_date(self.record_date)
        ot_logs = list(prof.get("ot_logs") or [])
        new_logs = [e for e in ot_logs if _parse_stored_ot_date(e.get("date", "")) != target_key]
        if len(new_logs) < len(ot_logs):
            prof["ot_logs"] = new_logs
            save_user_profiles_local()
            await backup_user_profiles_to_discord(force=True)
            self.stop()
            self.message = None
            await _safe_edit_or_followup_after_defer(
                interaction,
                f"🗑️ ลบรายการบันทึก OT ของวันที่ **`{self.record_date}`** เรียบร้อยแล้วค่ะ",
                None,
            )
        else:
            await interaction.followup.send(
                f"ไม่พบรายการวันที่ `{self.record_date}` ในระบบแล้วค่ะ", ephemeral=True
            )


def parse_ot_record_metadata(ot_reply_text: str) -> tuple[str, str, str]:
    m = re.search(
        r"\[OT_RECORD:\s*uid=\d+\s*\|\s*date=([0-9/\-]+)\s*\|\s*slot=([0-9:.]+-[0-9:.]+)\s*\|\s*hours=[0-9.]+\s*\|\s*detail=(.+?)\]",
        ot_reply_text or "",
    )
    if not m:
        return "", "", ""
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


# ==============================================================================
# 5. NATIVE SLASH COMMANDS & RIGHT-CLICK CONTEXT MENUS
# ==============================================================================
def register_app_commands(bot: discord.ext.commands.Bot):
    async def _verify_allowed_guild_interaction(interaction: discord.Interaction) -> bool:
        if not ALLOWED_GUILD_IDS or interaction.guild is None or interaction.guild.id in ALLOWED_GUILD_IDS:
            return True
        await _safe_send_ephemeral_error(
            interaction, "เซิร์ฟเวอร์นี้ยังไม่ได้เปิดสิทธิ์ให้ฉันใช้งานนะคะ"
        )
        return False

    bot.tree.interaction_check = _verify_allowed_guild_interaction

    @bot.tree.error
    async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        orig = getattr(error, "original", error)
        if isinstance(error, app_commands.CheckFailure):
            return
        cmd_name = getattr(interaction.command, "name", "unknown")
        api_log(f"AppCommand Error (/{cmd_name}) -> {type(orig).__name__}: {str(orig)[:180]}")
        await _safe_send_ephemeral_error(
            interaction,
            f"คำสั่ง `/{cmd_name}` ติดขัดนิดหน่อยค่ะ (`{type(orig).__name__}`) ลองใหม่อีกทีนะคะ",
        )

    @bot.tree.command(name="memory", description="ดู แก้ไข ปักหมุด หรือรีเซ็ตข้อมูลความจำถาวร (user_profiles.json)")
    @app_commands.describe(
        action="เลือกการทำงานกับความจำถาวร",
        value="ข้อความสำหรับปักหมุดข้อมูลถาวร เพิ่มบันทึก หรือเพิ่มหัวข้อที่ชอบ",
        member="เลือกสมาชิกที่ต้องการดู/แก้ไข (เฉพาะเจ้าของบอท เว้นว่างเพื่อดูของตัวเอง)",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="ดูข้อมูลความจำ (View & Buttons)", value="view"),
            app_commands.Choice(name="เปิดหน้าต่างแก้ไข (Edit Modal)", value="edit_modal"),
            app_commands.Choice(name="ปักหมุดข้อมูลถาวร (Pin Fact)", value="pin_fact"),
            app_commands.Choice(name="เพิ่มบันทึกระยะสั้น (Add Note)", value="add_note"),
            app_commands.Choice(name="เพิ่มหัวข้อที่ชอบ (Add Topic)", value="add_topic"),
            app_commands.Choice(name="ล้างเรื่องล่าสุด (Clear Recent)", value="clear_recent"),
        ]
    )
    async def slash_memory(
        interaction: discord.Interaction,
        action: str = "view",
        value: str = "",
        member: discord.User | None = None,
    ):
        target_user = member if (member is not None and is_owner_user(interaction.user)) else interaction.user
        target_uid = str(target_user.id)
        owner_flag = is_owner_user(target_user)
        raw_disp = getattr(target_user, "display_name", "") or getattr(target_user, "name", "")
        ensure_user_profile_exists(
            target_uid,
            display_name="Partner (Owner)" if owner_flag else raw_disp,
            username=getattr(target_user, "name", ""),
            is_owner=owner_flag,
        )

        # Modals MUST be sent as the immediate initial response (never deferred beforehand)
        if action == "edit_modal":
            await interaction.response.send_modal(MemoryEditModal(target_uid=target_uid))
            return

        await interaction.response.defer(thinking=True)

        class _ProxyMemMsg:
            def __init__(self):
                self.author = interaction.user
                self.channel = interaction.channel
                self.guild = interaction.guild

        mention_prefix = f"<@{target_uid}> " if target_uid != str(interaction.user.id) else ""
        if action == "pin_fact" and value.strip():
            cmd_str = f"{mention_prefix}ปักหมุดความจำ {value.strip()}"
        elif action == "add_note" and value.strip():
            cmd_str = f"{mention_prefix}เพิ่มบันทึก {value.strip()}"
        elif action == "add_topic" and value.strip():
            cmd_str = f"{mention_prefix}เพิ่มหัวข้อที่ชอบ {value.strip()}"
        elif action == "clear_recent":
            cmd_str = f"{mention_prefix}ล้างเรื่องล่าสุด"
        else:
            cmd_str = f"{mention_prefix}ดูความจำ"

        reply, resolved_uid = await process_memory_request(_ProxyMemMsg(), cmd_str)
        view = MemoryActionView(requester_id=interaction.user.id, target_uid=resolved_uid)
        await _send_interaction_with_auto_view(
            interaction,
            reply,
            view,
            chunked_header="❄️ ข้อมูลความจำพร้อมแล้วค่ะ:",
            allow_view_on_error=True,
        )

    @bot.tree.command(name="code", description="วิเคราะห์โค้ด, Stack Trace, SQL หรือ Manifest ตามมาตรฐาน Enterprise")
    @app_commands.describe(
        focus="เลือกความเชี่ยวชาญเฉพาะทางที่ต้องการเน้น",
        prompt="คำถามหรือวางโค้ด / Stack Trace ที่ต้องการให้ตรวจสอบ",
        attachment="แนบไฟล์ซอร์สโค้ด, Log, SQL, YAML หรือภาพหน้าจอ Error (ถ้ามี)",
    )
    @app_commands.choices(
        focus=[
            app_commands.Choice(name="Auto-Detect (Full-Stack & Stack Trace)", value="auto"),
            app_commands.Choice(name="Java / Maven / Spring Boot / JPA", value="java_spring"),
            app_commands.Choice(name="Oracle SQL / PL/SQL / Explain Plan", value="oracle_sql"),
            app_commands.Choice(name="Node.js / React / Bootstrap", value="node_react"),
            app_commands.Choice(name="Docker / Kubernetes / Microservices", value="docker_k8s"),
        ]
    )
    async def slash_code(
        interaction: discord.Interaction,
        prompt: str,
        focus: str = "auto",
        attachment: discord.Attachment | None = None,
    ):
        await interaction.response.defer(thinking=True)
        focus_notes = {
            "java_spring": "[Focus: Java 17/21, Maven pom.xml, Spring Boot, @Transactional, JPA/Hibernate N+1, and Exception Handling]",
            "oracle_sql": "[Focus: Oracle SQL, ORA- errors, Execution Plan, Foreign Key Indexes, Bind Variables, and PL/SQL Packages]",
            "node_react": "[Focus: Node.js Async/Event Loop, React Hooks/State, TypeScript, and Bootstrap 5 Grid/Utility best practices]",
            "docker_k8s": "[Focus: Multi-stage Dockerfile security, Kubernetes Probes/Resources, OOMKilled/CrashLoopBackOff, and Microservices/RabbitMQ]",
        }
        prefix = f"{focus_notes[focus]}\n" if focus in focus_notes else ""

        att_block = ""
        if attachment is not None:
            if attachment.size > 25 * 1024 * 1024:
                size_mb = round(attachment.size / (1024 * 1024), 2)
                await interaction.followup.send(
                    f"ไฟล์ `{attachment.filename}` ใหญ่เกิน 25 MB ค่ะ (`{size_mb} MB`) ลองตัดแบ่งไฟล์หรือส่งเฉพาะส่วนที่เกี่ยวข้องมานะคะ",
                    ephemeral=True,
                )
                return
            raw_b = await attachment.read()
            fname = attachment.filename.lower()
            if fname.endswith(IMAGE_EXTENSIONS):
                ocr_out = await extract_image_via_transient_ocr(raw_b, attachment.filename, run_source_lookup=False)
                att_block = f"\n\n=== Attached Screenshot OCR (`{attachment.filename}`) ===\n{ocr_out}"
            elif fname.endswith(".pdf"):
                pdf_block, _ = await extract_pdf_context_async(raw_b, attachment.filename)
                att_block = f"\n\n{pdf_block}"
            elif fname.endswith(".docx"):
                doc_txt = await asyncio.to_thread(parse_docx_bytes, raw_b)
                att_block = f"\n\n=== Attached Word Document: `{attachment.filename}` ===\n{doc_txt}"
            elif fname.endswith(".xlsx"):
                xls_txt = await asyncio.to_thread(parse_xlsx_bytes_to_markdown, raw_b)
                att_block = f"\n\n=== Attached Excel: `{attachment.filename}` ===\n{xls_txt}"
            else:
                decoded = raw_b.decode("utf-8", errors="replace")
                formatted = format_code_or_log_attachment(decoded, is_log=fname.endswith(".log"))
                att_block = f"\n\n=== Attached File: `{attachment.filename}` ===\n```\n{formatted}\n```"

        full_prompt = f"{prefix}{prompt}{att_block}"
        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            mode="code",
        )
        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(att_block),
            speaker_user=interaction.user,
            decision_plan={"route": "code", "route_confidence": 0.99, "needs_web_search": False, "search_confidence": 0.99},
            tool_ctx=tool_ctx,
        )
        view = (
            CodeReviewActionView(interaction.user.id, interaction.channel_id, prompt, answer)
            if should_attach_interactive_view(prompt, answer, use_coder=True, extracted_ctx=att_block)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ดูให้เรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.command(name="ot", description="บันทึกเวลาทำ OT, แก้ไข, ลบ หรือดูตารางสรุปรายเดือน")
    @app_commands.describe(
        action="เลือกการทำงาน (จด OT / สรุปตาราง / แก้ไข / ลบ)",
        detail="รายละเอียดงาน (เช่น แก้ระบบ e-Tax PDF)",
        date="วันที่ (เช่น วันนี้, เมื่อวาน, 15/11/2569 — เว้นว่างได้)",
        time_range="ช่วงเวลา (เช่น 18.00 - 21.00 — เว้นว่างได้สำหรับวันทำงานปกติ)",
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="จด / บันทึก OT", value="log"),
            app_commands.Choice(name="ดูตารางสรุป OT เดือนนี้", value="summary"),
            app_commands.Choice(name="ดูตารางสรุป OT เดือนที่แล้ว", value="prev_summary"),
            app_commands.Choice(name="แก้รายละเอียดงาน OT", value="edit"),
            app_commands.Choice(name="ลบรายการ OT", value="delete"),
        ]
    )
    async def slash_ot(
        interaction: discord.Interaction,
        action: str = "log",
        detail: str = "",
        date: str = "",
        time_range: str = "",
    ):
        await interaction.response.defer(thinking=True)
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        if action == "summary":
            cmd_text = f"สรุป OT {date}".strip()
        elif action == "prev_summary":
            py, pm, _ = get_previous_month_info(now_th)
            cmd_text = f"สรุป OT {pm:02d}/{py + 543}"
        elif action == "delete":
            cmd_text = f"ลบ OT {date}".strip()
        elif action == "edit":
            cmd_text = f"แก้ OT {date} {detail}".strip()
        else:
            parts = ["จด OT", date.strip(), time_range.strip(), detail.strip()]
            cmd_text = " ".join(p for p in parts if p)

        class _ProxyMsg:
            def __init__(self, inter: discord.Interaction):
                self.author = inter.user
                self.channel = inter.channel
                self.guild = inter.guild

        reply = await process_ot_request(_ProxyMsg(interaction), cmd_text)
        rec_date, rec_slot, rec_detail = parse_ot_record_metadata(reply)
        view = (
            OTActionView(interaction.user.id, rec_date, rec_slot, rec_detail)
            if rec_date and not is_failure_or_refusal_reply(reply)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, reply, view, chunked_header="❄️ สรุปเวลา OT พร้อมแล้วค่ะ:"
        )

    @bot.tree.command(name="yukipedia", description="ค้นหาข้อมูลเรียลไทม์ วิเคราะห์ข่าว เกม อนิเมะ หรือสรุปหัวข้อพร้อมปุ่มแนะนำ")
    @app_commands.describe(query="หัวข้อหรือคำถามที่ต้องการให้ Yukipedia ค้นหาและวิเคราะห์")
    async def slash_yukipedia(interaction: discord.Interaction, query: str):
        await interaction.response.defer(thinking=True)
        dec_plan = await orchestrate_message_with_decision_1(
            query,
            purpose="SlashYukipedia",
            channel_id=interaction.channel_id,
        )
        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            mode="general",
        )
        answer, did_search = await query_local_ollama_chat(
            query,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=False,
            speaker_user=interaction.user,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
        view = (
            build_contextual_suggestion_view(interaction.channel_id, query, answer, decision_plan=dec_plan)
            if should_attach_interactive_view(query, answer, did_search=did_search, decision_plan=dec_plan)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ข้อมูลจาก Yukipedia พร้อมแล้วค่ะ:"
        )

    @bot.tree.command(name="source", description="หาที่มาของรูปภาพ, งานวาด, อนิเมะ หรือโพสต์จากไฟล์แนบหรือลิงก์ (X/Pixiv/Embed)")
    @app_commands.describe(
        link="วางลิงก์ X (Twitter), Pixiv, phixiv, Bluesky หรือลิงก์รูปภาพ (ถ้าไม่ได้แนบไฟล์)",
        attachment="แนบไฟล์รูปภาพที่ต้องการหาที่มา (ถ้าไม่ได้วางลิงก์)",
        question="คำถามเพิ่มเติม (เช่น ใครวาด, ตอนที่เท่าไหร่, รุ่นอะไร)",
    )
    async def slash_source(
        interaction: discord.Interaction,
        link: str = "",
        attachment: discord.Attachment | None = None,
        question: str = "",
    ):
        if not link.strip() and attachment is None:
            await interaction.response.send_message(
                "แนบรูปหรือวางลิงก์ (X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วยสิคะ",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        class _FakeSourceMsg:
            def __init__(self):
                # Use interaction.id so /tmp image cache works for Deep Visual & In-Bubble follow-ups too
                self.id = int(interaction.id)
                self.content = link.strip()
                self.attachments = [attachment] if attachment is not None else []
                self.embeds = []
                self.stickers = []
                self.reference = None
                self.channel = interaction.channel
                self.author = interaction.user

            async def reply(self, **kwargs):
                return await interaction.channel.send(**kwargs)

        fake_msg = _FakeSourceMsg()
        extracted_ctx, _ = await extract_all_context(fake_msg)
        dec_plan = await orchestrate_message_with_decision_1(
            question.strip() or link.strip(),
            extracted_ctx=extracted_ctx,
            purpose="SlashSource",
            channel_id=interaction.channel_id,
        )
        user_q = question.strip() or (
            "ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด: "
            "หากเป็นภาพวาด/อนิเมะ/มังงะ ให้บอกชื่อเรื่อง ชื่อตัวละคร ตอนที่/Chapter ชื่อผู้วาด และแนบลิงก์ที่มา (เช่น Pixiv/X/MyAnimeList) ถ้ามี "
            "หากเป็นภาพถ่ายสิ่งของหรืออุปกรณ์ ให้ระบุยี่ห้อ รุ่น และรายละเอียดสำคัญของสิ่งนั้นค่ะ"
        )
        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            mode="general",
        )
        answer, did_search = await query_local_ollama_chat(
            f"{user_q}\n\n{extracted_ctx}",
            model=ACTIVE_FAST_MODEL,
            system_prompt=FAST_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=True,
            has_fresh_attachment=True,
            speaker_user=interaction.user,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id,
                link.strip() or user_q,
                answer,
                extracted_ctx=extracted_ctx,
                source_message=fake_msg,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                user_q, answer, did_search=did_search, extracted_ctx=extracted_ctx, decision_plan=dec_plan
            )
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ เช็คที่มาให้แล้วค่ะ:"
        )

    @bot.tree.command(name="manga", description="แปลหน้ามังงะในช่องคำพูดพร้อมใส่คำแปลลงในภาพ (รองรับไฟล์แนบและลิงก์ X/Pixiv)")
    @app_commands.describe(
        language="ภาษาปลายทางที่ต้องการแปล",
        link="วางลิงก์ X (Twitter), Pixiv/phixiv หรือลิงก์รูปมังงะ (รองรับหลายหน้าในโพสต์เดียว)",
        attachment="แนบไฟล์รูปหน้ามังงะ (ถ้าไม่ได้วางลิงก์)",
    )
    @app_commands.choices(
        language=[
            app_commands.Choice(name="ภาษาไทย (Thai)", value="Thai"),
            app_commands.Choice(name="ภาษาอังกฤษ (English)", value="English"),
        ]
    )
    async def slash_manga(
        interaction: discord.Interaction,
        language: str = "Thai",
        link: str = "",
        attachment: discord.Attachment | None = None,
    ):
        if not link.strip() and attachment is None:
            await interaction.response.send_message(
                "แนบรูปมังงะหรือวางลิงก์ X (Twitter) / Pixiv / phixiv มาด้วยสิคะ",
                ephemeral=True,
            )
            return

        lang_label = "ภาษาไทย" if language.lower() == "thai" else "ภาษาอังกฤษ"
        await interaction.response.send_message(
            f"เดี๋ยวฉันแปลหน้านี้เป็น{lang_label}ให้ รอสักครู่นะคะ"
        )

        class _FakeMangaMsg:
            def __init__(self):
                self.id = 0
                self.content = link.strip()
                self.attachments = [attachment] if attachment is not None else []
                self.embeds = []
                self.stickers = []
                self.reference = None
                self.channel = interaction.channel
                self.author = interaction.user

            async def reply(self, **kwargs):
                return await interaction.channel.send(**kwargs)

        class _FakeCtx:
            def __init__(self):
                self.message = _FakeMangaMsg()
                self.channel = interaction.channel

        try:
            await process_manga_translation_request(_FakeCtx(), f"translate manga to {language}")
        finally:
            try:
                await interaction.delete_original_response()
            except Exception:
                pass

    # --- Right-Click Message Context Menus (5 max per Discord app) ---
    @bot.tree.context_menu(name="Analyze Code / Stack Trace")
    async def ctx_analyze_code(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        base_text = message.content.strip() or "ช่วยตรวจสอบโค้ดหรือ Stack Trace ในข้อความนี้และระบุ Root Cause พร้อมวิธีแก้ไข"
        full_prompt = f"{base_text}\n\n{extracted_ctx}" if extracted_ctx else base_text

        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            message=message,
            mode="code",
        )
        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=interaction.user,
            decision_plan={"route": "code", "route_confidence": 0.99, "needs_web_search": False, "search_confidence": 0.99},
            tool_ctx=tool_ctx,
        )
        view = (
            CodeReviewActionView(interaction.user.id, interaction.channel_id, base_text, answer)
            if should_attach_interactive_view(base_text, answer, use_coder=True, extracted_ctx=extracted_ctx)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ดูโค้ดให้เรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Yukipedia Fact-Check / Summary")
    async def ctx_yukipedia_summary(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        base_text = message.content.strip() or "ช่วยตรวจสอบข้อเท็จจริงและสรุปเนื้อหาในข้อความนี้ให้หน่อย"
        dec_plan = await orchestrate_message_with_decision_1(
            base_text,
            extracted_ctx=extracted_ctx,
            purpose="CtxYukipedia",
            channel_id=interaction.channel_id,
        )
        full_prompt = f"{base_text}\n\n{extracted_ctx}" if extracted_ctx else base_text

        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            message=message,
            mode="general",
        )
        answer, did_search = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=interaction.user,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id,
                base_text,
                answer,
                extracted_ctx=extracted_ctx,
                source_message=message,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                base_text, answer, did_search=did_search, extracted_ctx=extracted_ctx, decision_plan=dec_plan
            )
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ สรุปข้อมูลให้เรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Find Image / Link Source")
    async def ctx_find_source(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        if not has_valid_extracted_context(extracted_ctx):
            await interaction.followup.send(
                "ข้อความนี้ไม่มีรูปหรือลิงก์ให้เช็คที่มาเลยค่ะ",
                ephemeral=True,
            )
            return
        dec_plan = await orchestrate_message_with_decision_1(
            message.content or "",
            extracted_ctx=extracted_ctx,
            purpose="CtxFindSource",
            channel_id=interaction.channel_id,
        )
        user_q = (
            "ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด: "
            "หากเป็นภาพวาด/อนิเมะ/มังงะ ให้บอกชื่อเรื่อง ชื่อตัวละคร ตอนที่/Chapter ชื่อผู้วาด และแนบลิงก์ที่มา (เช่น Pixiv/X/MyAnimeList) ถ้ามี "
            "หากเป็นภาพถ่ายสิ่งของหรืออุปกรณ์ ให้ระบุยี่ห้อ รุ่น และรายละเอียดสำคัญของสิ่งนั้นค่ะ"
        )
        tool_ctx = ToolExecutionContext(
            channel_id=interaction.channel_id,
            guild=interaction.guild,
            channel=interaction.channel,
            speaker_user=interaction.user,
            message=message,
            mode="general",
        )
        answer, did_search = await query_local_ollama_chat(
            f"{user_q}\n\n{extracted_ctx}",
            model=ACTIVE_FAST_MODEL,
            system_prompt=FAST_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=True,
            has_fresh_attachment=True,
            speaker_user=interaction.user,
            decision_plan=dec_plan,
            tool_ctx=tool_ctx,
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id,
                message.content or user_q,
                answer,
                extracted_ctx=extracted_ctx,
                source_message=message,
                decision_plan=dec_plan,
            )
            if should_attach_interactive_view(
                user_q, answer, did_search=did_search, extracted_ctx=extracted_ctx, decision_plan=dec_plan
            )
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ เช็คที่มาให้แล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Translate Manga Page (TH)")
    async def ctx_translate_manga_th(interaction: discord.Interaction, message: discord.Message):
        has_imgs = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in message.attachments)
        has_embeds_or_links = bool(message.embeds) or bool(URL_REGEX.search(message.content or ""))
        if not has_imgs and not has_embeds_or_links:
            await interaction.response.send_message(
                "ข้อความนี้ไม่มีรูปหรือลิงก์ภาพแนบอยู่ค่ะ คลิกขวาที่ข้อความที่มีรูปมังงะหรือลิงก์ X/Pixiv นะคะ",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            "เดี๋ยวฉันแปลหน้านี้เป็นภาษาไทยให้ รอสักครู่นะคะ"
        )
        try:
            await process_manga_translation_request(message, "แปลมังงะเป็นภาษาไทย")
        finally:
            try:
                await interaction.delete_original_response()
            except Exception:
                pass

    @bot.tree.context_menu(name="Translate Manga Page (EN)")
    async def ctx_translate_manga_en(interaction: discord.Interaction, message: discord.Message):
        has_imgs = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in message.attachments)
        has_embeds_or_links = bool(message.embeds) or bool(URL_REGEX.search(message.content or ""))
        if not has_imgs and not has_embeds_or_links:
            await interaction.response.send_message(
                "ข้อความนี้ไม่มีรูปหรือลิงก์ภาพแนบอยู่ค่ะ คลิกขวาที่ข้อความที่มีรูปมังงะหรือลิงก์ X/Pixiv นะคะ",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            "เดี๋ยวฉันแปลหน้านี้เป็นภาษาอังกฤษให้ รอสักครู่นะคะ"
        )
        try:
            await process_manga_translation_request(message, "แปลมังงะเป็นภาษาอังกฤษ")
        finally:
            try:
                await interaction.delete_original_response()
            except Exception:
                pass