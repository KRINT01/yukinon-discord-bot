import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord
from discord import app_commands

from config import (
    ACTIVE_FAST_MODEL,
    ACTIVE_CODE_MODEL,
    IMAGE_EXTENSIONS,
    THAI_MONTHS,
    URL_REGEX,
    PURE_DATETIME_PATTERN,
    get_cached_channel_attachments,
)
from persona import (
    FAST_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    YUKIPEDIA_SYSTEM_PROMPT,
)
from llm import query_local_ollama_chat
from media_context import (
    extract_all_context,
    format_code_or_log_attachment,
    extract_image_via_transient_ocr,
)
from image_engine import (
    send_chunked_response,
    process_manga_translation_request,
)
from user_memory import (
    user_profiles,
    process_ot_request,
    build_monthly_ot_summary_markdown,
    _parse_stored_ot_date,
    save_user_profiles_local,
    backup_user_profiles_to_discord,
)

_CONTEXT_ERROR_NOTICE_REGEX = re.compile(
    r"\[(?:Web extraction error|Error reading|Attachment Size Notice|PDF Notice|OCR could not extract|Error inspecting embedded image)[^\]]*\]",
    re.IGNORECASE,
)

_FAILURE_OR_REFUSAL_REPLY_REGEX = re.compile(
    r"(ไม่สามารถเข้าถึง|ไม่สามารถดึง|ไม่สามารถอ่าน|ไม่สามารถตรวจสอบ|ไม่สามารถเชื่อมต่อได้ชั่วคราว|"
    r"ไม่สามารถสร้างรูปภาพได้|ไม่สามารถตรวจจับข้อความ|ไม่พบรายการ|ไม่พบข้อมูล(?:ที่|ใน|เพียงพอ)|"
    r"ไม่มีข้อมูลเพียงพอ|ไฟล์มีขนาดใหญ่เกิน|ไม่มีข้อความที่สามารถอ่านได้|สแกนหรือเป็นรูปภาพ|"
    r"ส่งไฟล์.*มาใหม่|แนบไฟล์.*มาใหม่|ลืมคำสั่งก่อนหน้า|"
    r"\b(?:unable to access|cannot read|could not extract|no readable (?:article )?text|shallow, transparent attempt)\b)",
    re.IGNORECASE,
)

_SUBSTANTIVE_REQUEST_REGEX = re.compile(
    r"(เปรียบเทียบ|ข้อดี|ข้อเสีย|แนะนำ|วิเคราะห์|สรุป|อธิบาย|ต่างกันยังไง|ต่างกันอย่างไร|"
    r"เลือกอะไรดี|รุ่นไหนดี|แนวทาง|ไอเดีย|วางแผน|จัดสเปค|วิจารณ์|"
    r"\b(?:compare|comparison|pros|cons|recommend|analyze|analysis|explain|difference|best|review|strategy|idea)\b)",
    re.IGNORECASE,
)


def _strip_internal_prompt_hints(text: str) -> str:
    """Removes parenthetical system instructions from `user_query` so follow-up buttons never inherit meta-prompts."""
    if not text:
        return ""
    cleaned = re.sub(r"\n*\(ผู้ใช้ส่งลิงก์นี้มาให้[\s\S]*?\)\s*$", "", text).strip()
    return cleaned


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
    if _SUBSTANTIVE_REQUEST_REGEX.search(user_request or ""):
        return True
    return False


def get_previous_month_info(now_th: datetime) -> tuple[int, int, str]:
    first_of_this_month = now_th.replace(day=1)
    last_day_prev_month = first_of_this_month - timedelta(days=1)
    y_ce = last_day_prev_month.year
    m_num = last_day_prev_month.month
    return y_ce, m_num, f"{THAI_MONTHS[m_num]} พ.ศ. {y_ce + 543}"


class AutoDisappearingView(discord.ui.View):
    """
    Base View that automatically removes its buttons from the Discord message
    (or deletes the standalone button footer message) when `timeout` expires,
    and releases its message references to keep Render 512MB RAM clean.
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


async def _send_interaction_with_auto_view(
    interaction: discord.Interaction,
    answer: str,
    view: AutoDisappearingView | None,
    chunked_header: str = "❄️ ดำเนินการเรียบร้อยแล้วค่ะ:",
):
    if not answer or not answer.strip():
        return
    effective_view = None if is_failure_or_refusal_reply(answer) else view
    if len(answer) <= 1900:
        if effective_view is not None:
            sent_msg = await interaction.followup.send(answer, view=effective_view, wait=True)
            effective_view.message = sent_msg
            effective_view.delete_message_on_timeout = False
        else:
            await interaction.followup.send(answer)
    else:
        await interaction.followup.send(chunked_header)
        await send_chunked_response(interaction.channel, answer)
        if effective_view is not None:
            footer_msg = await interaction.channel.send("-# ตัวเลือกดำเนินการเพิ่มเติม:", view=effective_view)
            effective_view.message = footer_msg
            effective_view.delete_message_on_timeout = True


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

        answer, _ = await query_local_ollama_chat(
            composed,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=self.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=True,
            speaker_user=interaction.user,
        )
        await send_chunked_response(interaction.channel, answer)
        try:
            await interaction.followup.send("ดำเนินการตามที่เลือกเรียบร้อยแล้วค่ะ ❄️", ephemeral=True)
        except Exception:
            pass

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
# 2. DYNAMIC DOMAIN-TAILORED SUGGESTION VIEW
# ==============================================================================
class GeneralAssistantActionView(AutoDisappearingView):
    """
    Dynamically constructs 2-3 context-matched suggestion buttons tailored to the exact
    domain (Documents/PDFs, Art/Social Embeds, Gaming, News/Finance, or Product/Advice).
    """

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
                btn.callback = self._make_followup_callback(action_spec, use_search)
            self.add_item(btn)

    def _make_in_bubble_manga_callback(self):
        async def _callback(interaction: discord.Interaction):
            await interaction.response.send_message(
                "📖 กำลังแปลบทสนทนาและฝังข้อความภาษาไทยลงในช่องคำพูดของภาพให้สักครู่นะคะ...",
                ephemeral=True,
            )
            target_msg = self.source_message
            if target_msg is None:
                class _FallbackLinkMsg:
                    def __init__(self, q_txt: str, ch, usr):
                        self.id = 0
                        self.content = q_txt
                        self.attachments = []
                        self.embeds = []
                        self.reference = None
                        self.channel = ch
                        self.author = usr

                    async def reply(self, **kwargs):
                        return await self.channel.send(**kwargs)

                target_msg = _FallbackLinkMsg(self.user_query, interaction.channel, interaction.user)

            await process_manga_translation_request(target_msg, "แปลมังงะเป็นภาษาไทย")

        return _callback

    def _make_followup_callback(self, instruction: str, use_search: bool):
        async def _callback(interaction: discord.Interaction):
            await self._run_general_followup(interaction, instruction, use_search=use_search)
        return _callback

    async def _run_general_followup(self, interaction: discord.Interaction, instruction: str, use_search: bool = True):
        await interaction.response.defer(thinking=True)
        cached_ctx = get_cached_channel_attachments(self.channel_id, max_total_chars=25000)
        composed = (
            f"{instruction}\n\n"
            f"[กฎสำคัญ]: ห้ามคัดลอกหรือเขียนคำตอบเดิมซ้ำ ห้ามสร้างตาราง Markdown (`|---|---|`) "
            f"และห้ามเขียนหัวข้อว่า 'ความคิดเห็นในสไตล์ยูกิโนะ' เด็ดขาด\n\n"
            f"[บริบทหัวข้อก่อนหน้า]: {self.user_query}\n"
            f"[สรุปคำตอบก่อนหน้า (ห้ามเขียนซ้ำ)]: {self.bot_reply[:1500]}"
        )
        if cached_ctx and not use_search:
            composed += f"\n\n[เนื้อหาอ้างอิงจากไฟล์/ลิงก์ในห้องแชท]:\n{cached_ctx[:18000]}"

        answer, _ = await query_local_ollama_chat(
            composed,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT if use_search else FAST_SYSTEM_PROMPT,
            channel_id=self.channel_id,
            temperature=0.45,
            allow_agentic_search=use_search,
            has_fresh_attachment=bool(cached_ctx and not use_search),
            speaker_user=interaction.user,
        )
        await send_chunked_response(interaction.channel, answer)
        try:
            await interaction.followup.send("จัดเตรียมข้อมูลเพิ่มเติมให้เรียบร้อยแล้วค่ะ ❄️", ephemeral=True)
        except Exception:
            pass


def build_contextual_suggestion_view(
    channel_id: int,
    user_query: str,
    bot_reply: str,
    extracted_ctx: str = "",
    source_message: discord.Message | None = None,
) -> GeneralAssistantActionView:
    """
    Selects the exact set of interactive buttons based on what was actually shared or asked.
    """
    ctx_lower = (extracted_ctx or "").lower()
    combined = f"{user_query} {bot_reply}".lower()

    # 1. Documents, Spreadsheets, Web Articles & YouTube Videos
    if any(
        tag in ctx_lower
        for tag in (
            "=== attached pdf",
            "=== attached word",
            "=== attached excel",
            "=== youtube video information",
            "<untrusted_web_search_results url=",
        )
    ):
        specs = [
            (
                "สรุปประเด็นสำคัญ 3 ข้อ",
                "📌",
                discord.ButtonStyle.primary,
                "ช่วยสรุปใจความสำคัญและข้อสรุปหลักจากเอกสาร/เนื้อหานี้ให้กระชับที่สุดเพียง 3 ข้อค่ะ",
                False,
            ),
            (
                "สกัด Action Items / Checklist",
                "📋",
                discord.ButtonStyle.secondary,
                "ช่วยสกัดขั้นตอนที่ต้องทำ (Action Items), เงื่อนไขสำคัญ หรือ Checklist ที่นำไปใช้งานต่อได้ทันทีจากเอกสาร/เนื้อหานี้ค่ะ",
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

    # 2. Embedded Social Posts (X / Twitter, Pixiv, Bluesky), Artwork & Anime/Manga
    if any(
        tag in ctx_lower
        for tag in ("=== embedded x", "=== embedded pixiv", "=== embedded bluesky", "=== image inspection")
    ) or any(k in combined for k in ("pixiv", "twitter", "x.com", "ศิลปิน", "ผู้วาด", "อนิเมะ", "มังงะ", "anime", "manga")):
        has_foreign_dialogue = bool(re.search(r"[ぁ-んァ-ヶ一-龯]", extracted_ctx or ""))
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
                    "วิเคราะห์องค์ประกอบ & บรรยากาศ",
                    "📌",
                    discord.ButtonStyle.primary,
                    "ช่วยวิเคราะห์องค์ประกอบภาพ อารมณ์สีหน้า และรายละเอียดที่น่าสนใจในภาพนี้เพิ่มเติมจากที่เล่าไปเมื่อครู่ค่ะ",
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

    # 3. Video Games / Builds / Mechanics
    if any(k in combined for k in ("เกม", "game", "steam", "pso2", "ngs", "build", "คลาส", "class", "เมต้า", "meta")):
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

    # 4. News / Market / Financial Updates
    if any(k in combined for k in ("ข่าว", "news", "ราคา", "หุ้น", "ทอง", "บิตคอยน์", "อัปเดต")):
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
                "ตรวจสอบบริบทเพิ่มเติม",
                "🔍",
                discord.ButtonStyle.secondary,
                "ช่วยค้นหาข้อมูลเบื้องหลังและตรวจสอบข้อเท็จจริงเพิ่มเติมของประเด็นนี้ให้หน่อยค่ะ",
                True,
            ),
        ]
        return GeneralAssistantActionView(channel_id, user_query, bot_reply, specs, source_message=source_message)

    # 5. Hardware / Gear / Comparisons / General Recommendations
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
# 3. INTERACTIVE OVERTIME (OT) MODAL & ACTION BUTTONS
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
            self.date_input.default = default_date
        if default_slot:
            self.time_input.default = default_slot
        if default_detail:
            self.detail_input.default = default_detail

    async def on_submit(self, interaction: discord.Interaction):
        cmd_str = f"จด OT {self.date_input.value.strip()} {self.time_input.value.strip()} {self.detail_input.value.strip()}"

        class _ProxyMsg:
            def __init__(self, inter: discord.Interaction):
                self.author = inter.user
                self.channel = inter.channel
                self.guild = inter.guild

        reply = await process_ot_request(_ProxyMsg(interaction), cmd_str)
        view = OTActionView(
            owner_id=interaction.user.id,
            record_date=self.date_input.value.strip(),
            record_slot=self.time_input.value.strip(),
            record_detail=self.detail_input.value.strip(),
        )
        await interaction.response.send_message(reply, view=view)
        try:
            view.message = await interaction.original_response()
        except Exception:
            pass


class OTActionView(AutoDisappearingView):
    def __init__(self, owner_id: int, record_date: str = "", record_slot: str = "", record_detail: str = ""):
        super().__init__(timeout=300.0)
        self.owner_id = owner_id
        self.record_date = record_date
        self.record_slot = record_slot
        self.record_detail = record_detail

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("ปุ่มจัดการ OT นี้เป็นของเจ้าของรายการเท่านั้นนะคะ", ephemeral=True)
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
        await interaction.response.send_message(summary_md, ephemeral=True)

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
        target_key = _parse_stored_ot_date(self.record_date)
        ot_logs = list(prof.get("ot_logs") or [])
        new_logs = [e for e in ot_logs if _parse_stored_ot_date(e.get("date", "")) != target_key]
        if len(new_logs) < len(ot_logs):
            prof["ot_logs"] = new_logs
            save_user_profiles_local()
            await backup_user_profiles_to_discord(force=True)
            self.stop()
            self.message = None
            await interaction.response.edit_message(
                content=f"🗑️ ลบรายการบันทึก OT ของวันที่ **`{self.record_date}`** เรียบร้อยแล้วค่ะ",
                view=None,
            )
        else:
            await interaction.response.send_message(f"ไม่พบรายการวันที่ `{self.record_date}` ในระบบแล้วค่ะ", ephemeral=True)


def parse_ot_record_metadata(ot_reply_text: str) -> tuple[str, str, str]:
    m = re.search(
        r"\[OT_RECORD:\s*uid=\d+\s*\|\s*date=([0-9/\-]+)\s*\|\s*slot=([0-9:.]+-[0-9:.]+)\s*\|\s*hours=[0-9.]+\s*\|\s*detail=(.+?)\]",
        ot_reply_text or "",
    )
    if not m:
        return "", "", ""
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


# ==============================================================================
# 4. NATIVE SLASH COMMANDS & RIGHT-CLICK CONTEXT MENUS
# ==============================================================================
def register_app_commands(bot: discord.ext.commands.Bot):
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
        if attachment is not None and attachment.size <= 25 * 1024 * 1024:
            raw_b = await attachment.read()
            fname = attachment.filename.lower()
            if fname.endswith(IMAGE_EXTENSIONS):
                ocr_out = await extract_image_via_transient_ocr(raw_b, attachment.filename, run_source_lookup=False)
                att_block = f"\n\n=== Attached Screenshot OCR (`{attachment.filename}`) ===\n{ocr_out}"
            else:
                decoded = raw_b.decode("utf-8", errors="replace")
                formatted = format_code_or_log_attachment(decoded, is_log=fname.endswith(".log"))
                att_block = f"\n\n=== Attached File: `{attachment.filename}` ===\n```\n{formatted}\n```"

        full_prompt = f"{prefix}{prompt}{att_block}"
        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(att_block),
            speaker_user=interaction.user,
        )
        view = (
            CodeReviewActionView(interaction.user.id, interaction.channel_id, prompt, answer)
            if should_attach_interactive_view(prompt, answer, use_coder=True, extracted_ctx=att_block)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ผลการวิเคราะห์เสร็จสมบูรณ์แล้วค่ะ:"
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
        if view is not None:
            await interaction.response.send_message(reply, view=view)
            try:
                view.message = await interaction.original_response()
            except Exception:
                pass
        else:
            await interaction.response.send_message(reply)

    @bot.tree.command(name="yukipedia", description="ค้นหาข้อมูลเรียลไทม์ วิเคราะห์ข่าว เกม อนิเมะ หรือสรุปหัวข้อพร้อมปุ่มแนะนำ")
    @app_commands.describe(query="หัวข้อหรือคำถามที่ต้องการให้ Yukipedia ค้นหาและวิเคราะห์")
    async def slash_yukipedia(interaction: discord.Interaction, query: str):
        await interaction.response.defer(thinking=True)
        answer, did_search = await query_local_ollama_chat(
            query,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=False,
            speaker_user=interaction.user,
        )
        view = (
            build_contextual_suggestion_view(interaction.channel_id, query, answer)
            if should_attach_interactive_view(query, answer, did_search=did_search)
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
                "กรุณาแนบรูปภาพ หรือวางลิงก์ (เช่น X/Twitter, Pixiv/phixiv หรือลิงก์รูปภาพ) มาด้วยนะคะ",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        class _FakeSourceMsg:
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

        fake_msg = _FakeSourceMsg()
        extracted_ctx, _ = await extract_all_context(fake_msg)
        user_q = question.strip() or (
            "ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด: "
            "หากเป็นภาพวาด/อนิเมะ/มังงะ ให้บอกชื่อเรื่อง ชื่อตัวละคร ตอนที่/Chapter ชื่อผู้วาด และแนบลิงก์ที่มา (เช่น Pixiv/X/MyAnimeList) ถ้ามี "
            "หากเป็นภาพถ่ายสิ่งของหรืออุปกรณ์ ให้ระบุยี่ห้อ รุ่น และรายละเอียดสำคัญของสิ่งนั้นค่ะ"
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
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id, link.strip() or user_q, answer, extracted_ctx=extracted_ctx, source_message=fake_msg
            )
            if should_attach_interactive_view(user_q, answer, did_search=did_search, extracted_ctx=extracted_ctx)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ตรวจสอบที่มาเรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.command(name="manga", description="แปลหน้ามังงะในช่องคำพูดพร้อมฝังคำแปลลงในภาพ (รองรับไฟล์แนบและลิงก์ X/Pixiv)")
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
                "กรุณาแนบไฟล์รูปมังงะ หรือวางลิงก์ X (Twitter) / Pixiv / phixiv ที่มีรูปมังงะนะคะ",
                ephemeral=True,
            )
            return

        await interaction.response.send_message(
            f"📖 กำลังแปลหน้ามังงะเป็น **{language}** และจัดวางข้อความลงในช่องคำพูดให้สักครู่นะคะ...",
            ephemeral=True,
        )

        class _FakeMangaMsg:
            def __init__(self):
                self.id = 0
                self.content = link.strip()
                self.attachments = [attachment] if attachment is not None else []
                self.embeds = []
                self.reference = None
                self.channel = interaction.channel
                self.author = interaction.user

            async def reply(self, **kwargs):
                return await interaction.channel.send(**kwargs)

        class _FakeCtx:
            def __init__(self):
                self.message = _FakeMangaMsg()
                self.channel = interaction.channel

        await process_manga_translation_request(_FakeCtx(), f"translate manga to {language}")

    # --- Right-Click Message Context Menus ---
    @bot.tree.context_menu(name="Analyze Code / Stack Trace")
    async def ctx_analyze_code(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        base_text = message.content.strip() or "ช่วยตรวจสอบโค้ดหรือ Stack Trace ในข้อความนี้และระบุ Root Cause พร้อมวิธีแก้ไข"
        full_prompt = f"{base_text}\n\n{extracted_ctx}" if extracted_ctx else base_text

        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_CODE_MODEL,
            system_prompt=CODE_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.2,
            allow_agentic_search=False,
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=interaction.user,
        )
        view = (
            CodeReviewActionView(interaction.user.id, interaction.channel_id, base_text, answer)
            if should_attach_interactive_view(base_text, answer, use_coder=True, extracted_ctx=extracted_ctx)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ วิเคราะห์โค้ด/Stack Trace เรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Yukipedia Fact-Check / Summary")
    async def ctx_yukipedia_summary(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        base_text = message.content.strip() or "ช่วยตรวจสอบข้อเท็จจริงและสรุปเนื้อหาในข้อความนี้ให้หน่อย"
        full_prompt = f"{base_text}\n\n{extracted_ctx}" if extracted_ctx else base_text

        answer, did_search = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=interaction.user,
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id, base_text, answer, extracted_ctx=extracted_ctx, source_message=message
            )
            if should_attach_interactive_view(base_text, answer, did_search=did_search, extracted_ctx=extracted_ctx)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ตรวจสอบและสรุปข้อมูลเรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Find Image / Link Source")
    async def ctx_find_source(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        if not has_valid_extracted_context(extracted_ctx):
            await interaction.followup.send(
                "ไม่พบรูปภาพหรือลิงก์ที่สามารถตรวจสอบที่มาได้ในข้อความนี้ค่ะ",
                ephemeral=True,
            )
            return
        user_q = (
            "ช่วยระบุที่มาของภาพและลิงก์นี้อย่างละเอียด: "
            "หากเป็นภาพวาด/อนิเมะ/มังงะ ให้บอกชื่อเรื่อง ชื่อตัวละคร ตอนที่/Chapter ชื่อผู้วาด และแนบลิงก์ที่มา (เช่น Pixiv/X/MyAnimeList) ถ้ามี "
            "หากเป็นภาพถ่ายสิ่งของหรืออุปกรณ์ ให้ระบุยี่ห้อ รุ่น และรายละเอียดสำคัญของสิ่งนั้นค่ะ"
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
        )
        view = (
            build_contextual_suggestion_view(
                interaction.channel_id, message.content or user_q, answer, extracted_ctx=extracted_ctx, source_message=message
            )
            if should_attach_interactive_view(user_q, answer, did_search=did_search, extracted_ctx=extracted_ctx)
            else None
        )
        await _send_interaction_with_auto_view(
            interaction, answer, view, chunked_header="❄️ ตรวจสอบที่มาเรียบร้อยแล้วค่ะ:"
        )

    @bot.tree.context_menu(name="Translate Manga Page (TH)")
    async def ctx_translate_manga_th(interaction: discord.Interaction, message: discord.Message):
        has_imgs = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in message.attachments)
        has_embeds_or_links = bool(message.embeds) or bool(URL_REGEX.search(message.content or ""))
        if not has_imgs and not has_embeds_or_links:
            await interaction.response.send_message(
                "ข้อความที่เลือกไม่มีไฟล์รูปภาพหรือลิงก์ที่มีรูปภาพแนบอยู่ค่ะ กรุณาคลิกขวาที่ข้อความที่มีรูปมังงะหรือลิงก์ X/Pixiv นะคะ",
                ephemeral=True,
            )
            return
        await interaction.response.send_message("📖 กำลังแปลหน้ามังงะเป็นภาษาไทยให้สักครู่นะคะ...", ephemeral=True)
        await process_manga_translation_request(message, "แปลมังงะเป็นภาษาไทย")