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
    get_cached_channel_attachments,
)
from persona import (
    FAST_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    YUKIPEDIA_SYSTEM_PROMPT,
)
from llm import query_local_ollama_chat
from media_context import extract_all_context, format_code_or_log_attachment
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


def get_previous_month_info(now_th: datetime) -> tuple[int, int, str]:
    """Returns (prev_year_ce, prev_month_num, formatted_thai_month_year) for the 1st-of-month OT report."""
    first_of_this_month = now_th.replace(day=1)
    last_day_prev_month = first_of_this_month - timedelta(days=1)
    y_ce = last_day_prev_month.year
    m_num = last_day_prev_month.month
    return y_ce, m_num, f"{THAI_MONTHS[m_num]} พ.ศ. {y_ce + 543}"


# ==============================================================================
# 1. INTERACTIVE CODE REVIEW & STACK TRACE VIEW
# ==============================================================================
class CodeReviewActionView(discord.ui.View):
    """Interactive follow-up buttons attached to coding & stack trace responses."""

    def __init__(self, author_id: int, channel_id: int, original_prompt: str, original_answer: str):
        super().__init__(timeout=300)
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

    @discord.ui.button(label="เขียน Unit Tests", style=discord.ButtonStyle.primary, emoji="🧪")
    async def btn_unit_tests(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "ช่วยเขียน Unit Tests ที่ครอบคลุมทั้ง Happy Path และ Edge Cases/Exception จากโค้ดหรือบั๊กที่เพิ่งวิเคราะห์ไป "
            "(ใช้ JUnit 5 + Mockito สำหรับ Java/Spring Boot หรือ Jest/Vitest สำหรับ Node.js/React ตามภาษาของโค้ดเดิม) "
            "พร้อมระบุชื่อไฟล์ไว้บรรทัดแรกของบล็อกโค้ดด้วยค่ะ",
        )

    @discord.ui.button(label="เจาะลึก Root Cause", style=discord.ButtonStyle.secondary, emoji="🔍")
    async def btn_deep_dive(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "ช่วยอธิบายเจาะลึกกลไกเบื้องหลัง (Under-the-hood mechanism), ผลกระทบด้าน Concurrency / Memory / Database Execution Plan "
            "และแนวทางป้องกันในระดับ Production Architecture ของปัญหานี้อย่างละเอียดค่ะ",
        )

    @discord.ui.button(label="สร้าง Config / SQL / Infra", style=discord.ButtonStyle.secondary, emoji="🛠️")
    async def btn_infra_sql(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_followup(
            interaction,
            "จากโค้ดหรือปัญหาเมื่อครู่ ช่วยสร้างไฟล์ประกอบที่จำเป็นให้สมบูรณ์ เช่น Oracle SQL Index/DDL Script, "
            "`application.yml` / `pom.xml` snippet, หรือ `Dockerfile` / Kubernetes manifest ที่ปลอดภัยและพร้อมใช้งานจริงค่ะ",
        )


# ==============================================================================
# 2. INTERACTIVE GENERAL / YUKIPEDIA / RESEARCH SUGGESTION VIEW
# ==============================================================================
class GeneralAssistantActionView(discord.ui.View):
    """
    Context-aware suggestion buttons for non-code replies (News, Games, Anime, YouTube, Ideas, Products).
    Includes a dynamic 3rd 'Suggested Next Angle' button tailored to what Yukinon just answered.
    """

    def __init__(
        self,
        channel_id: int,
        user_query: str,
        bot_reply: str,
        suggested_label: str = "มุมมอง & คำแนะนำเพิ่ม",
        suggested_prompt: str = "",
    ):
        super().__init__(timeout=300)
        self.channel_id = channel_id
        self.user_query = user_query[:1200]
        self.bot_reply = bot_reply[:2500]
        self.suggested_prompt = suggested_prompt or (
            "จากหัวข้อที่เราเพิ่งคุยกัน ช่วยวิเคราะห์มุมมองที่น่าสนใจเพิ่มเติม ข้อควรระวัง หรือสิ่งที่คนมักมองข้ามให้หน่อยค่ะ"
        )
        self.btn_smart_suggest.label = suggested_label[:34]

    async def _run_general_followup(self, interaction: discord.Interaction, instruction: str, use_search: bool = True):
        await interaction.response.defer(thinking=True)
        composed = (
            f"{instruction}\n\n"
            f"[บริบทหัวข้อก่อนหน้า]: {self.user_query}\n"
            f"[ข้อมูลที่สรุปไว้]: {self.bot_reply[:1800]}"
        )
        answer, _ = await query_local_ollama_chat(
            composed,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT if use_search else FAST_SYSTEM_PROMPT,
            channel_id=self.channel_id,
            temperature=0.45,
            allow_agentic_search=use_search,
            has_fresh_attachment=False,
            speaker_user=interaction.user,
        )
        await send_chunked_response(interaction.channel, answer)
        try:
            await interaction.followup.send("จัดเตรียมข้อมูลเพิ่มเติมให้เรียบร้อยแล้วค่ะ ❄️", ephemeral=True)
        except Exception:
            pass

    @discord.ui.button(label="สรุปสั้น 3 ข้อ", style=discord.ButtonStyle.secondary, emoji="📌")
    async def btn_tldr(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_general_followup(
            interaction,
            "ช่วยสรุปใจความสำคัญจากคำตอบเมื่อครู่ให้กระชับที่สุดเพียง 3 ข้อ (Bullet points) อ่านจบใน 10 วินาทีค่ะ",
            use_search=False,
        )

    @discord.ui.button(label="เทียบข้อดี-ข้อเสีย", style=discord.ButtonStyle.secondary, emoji="⚖️")
    async def btn_pros_cons(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_general_followup(
            interaction,
            "ช่วยวิเคราะห์ข้อดี ข้อเสีย ความคุ้มค่า และข้อควรระวังของเรื่องนี้ในมุมมองที่ตรงไปตรงมาสไตล์ยูกิโนะให้หน่อยค่ะ",
            use_search=True,
        )

    @discord.ui.button(label="มุมมอง & คำแนะนำเพิ่ม", style=discord.ButtonStyle.primary, emoji="💡")
    async def btn_smart_suggest(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._run_general_followup(
            interaction,
            self.suggested_prompt,
            use_search=True,
        )


def build_contextual_suggestion_view(channel_id: int, user_query: str, bot_reply: str) -> GeneralAssistantActionView:
    """Chooses a smart, domain-specific 3rd suggestion button based on conversation topic."""
    combined = f"{user_query} {bot_reply}".lower()
    if any(k in combined for k in ("เกม", "game", "steam", "pso2", "ngs", "build", "คลาส", "class")):
        label = "แนะนำแนวทางเล่น / Build"
        prompt = "ช่วยแนะนำแนวทางการเล่น การจัดสาย/อุปกรณ์ หรือเคล็ดลับสำคัญสำหรับผู้เล่นในเรื่องนี้พร้อมเช็คข้อมูลล่าสุดให้หน่อยค่ะ"
    elif any(k in combined for k in ("อนิเมะ", "มังงะ", "นิยาย", "anime", "manga", "novel", "เรื่องย่อ")):
        label = "รีวิว & เรื่องคล้ายกัน"
        prompt = "ช่วยวิจารณ์จุดเด่นของเรื่องนี้ในสไตล์ยูกิโนะ พร้อมแนะนำเรื่องที่บรรยากาศหรือคุณภาพใกล้เคียงกัน 2-3 เรื่องค่ะ"
    elif any(k in combined for k in ("youtube", "youtu.be", "คลิป", "วิดีโอ")):
        label = "เจาะลึกประเด็นในคลิป"
        prompt = "จากเนื้อหาของวิดีโอนี้ ช่วยขยายความประเด็นสำคัญหรือข้อมูลเบื้องหลังที่เกี่ยวข้องเพิ่มเติมให้หน่อยค่ะ"
    elif any(k in combined for k in ("ข่าว", "news", "ราคา", "หุ้น", "ทอง", "บิตคอยน์", "อัปเดต")):
        label = "เช็คผลกระทบ & แนวโน้ม"
        prompt = "ช่วยค้นหาอัปเดตล่าสุดและวิเคราะห์ผลกระทบหรือแนวโน้มที่อาจจะเกิดขึ้นต่อจากข่าวนี้ให้หน่อยค่ะ"
    elif any(k in combined for k in ("กาแฟ", "espresso", "coffee", "อุปกรณ์", "สเปค", "spec", "แนะนำ")):
        label = "เทียบตัวเลือกอื่นที่คุ้มกว่า"
        prompt = "ช่วยเสนอทางเลือกอื่นที่น่าสนใจหรือคุ้มค่ากว่าในระดับใกล้เคียงกัน พร้อมเหตุผลเปรียบเทียบให้หน่อยค่ะ"
    else:
        label = "มุมมอง & คำแนะนำเพิ่ม"
        prompt = "จากเรื่องที่คุยกันเมื่อครู่ ยูกิโนะมีข้อสังเกต คำแนะนำ หรือประเด็นต่อยอดอะไรที่ฉันควรรู้เพิ่มเติมไหมคะ?"

    return GeneralAssistantActionView(
        channel_id=channel_id,
        user_query=user_query,
        bot_reply=bot_reply,
        suggested_label=label,
        suggested_prompt=prompt,
    )


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
        await interaction.response.send_message(
            reply,
            view=OTActionView(
                owner_id=interaction.user.id,
                record_date=self.date_input.value.strip(),
                record_slot=self.time_input.value.strip(),
                record_detail=self.detail_input.value.strip(),
            ),
        )


class OTActionView(discord.ui.View):
    """Interactive buttons under OT log confirmations for 1-click Monthly Table, Edit Modal, or Delete."""

    def __init__(self, owner_id: int, record_date: str = "", record_slot: str = "", record_detail: str = ""):
        super().__init__(timeout=300)
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
            for child in self.children:
                child.disabled = True
            await interaction.response.edit_message(
                content=f"🗑️ ลบรายการบันทึก OT ของวันที่ **`{self.record_date}`** เรียบร้อยแล้วค่ะ",
                view=self,
            )
        else:
            await interaction.response.send_message(f"ไม่พบรายการวันที่ `{self.record_date}` ในระบบแล้วค่ะ", ephemeral=True)


def parse_ot_record_metadata(ot_reply_text: str) -> tuple[str, str, str]:
    """Extracts (date, slot, detail) from the hidden -# [OT_RECORD: ...] footer if present."""
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
    """Registers all Slash Commands (/code, /ot, /yukipedia) and Right-Click Context Menus."""

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
        if attachment is not None and attachment.size <= 15 * 1024 * 1024:
            raw_b = await attachment.read()
            decoded = raw_b.decode("utf-8", errors="replace")
            formatted = format_code_or_log_attachment(decoded, is_log=attachment.filename.lower().endswith(".log"))
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
        view = CodeReviewActionView(interaction.user.id, interaction.channel_id, prompt, answer)
        if len(answer) <= 1900:
            await interaction.followup.send(answer, view=view)
        else:
            await interaction.followup.send("❄️ ผลการวิเคราะห์เสร็จสมบูรณ์แล้วค่ะ:", view=view)
            await send_chunked_response(interaction.channel, answer)

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
            if rec_date
            else OTActionView(interaction.user.id)
        )
        await interaction.response.send_message(reply, view=view)

    @bot.tree.command(name="yukipedia", description="ค้นหาข้อมูลเรียลไทม์ วิเคราะห์ข่าว เกม อนิเมะ หรือสรุปหัวข้อพร้อมปุ่มแนะนำ")
    @app_commands.describe(query="หัวข้อหรือคำถามที่ต้องการให้ Yukipedia ค้นหาและวิเคราะห์")
    async def slash_yukipedia(interaction: discord.Interaction, query: str):
        await interaction.response.defer(thinking=True)
        answer, _ = await query_local_ollama_chat(
            query,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=False,
            speaker_user=interaction.user,
        )
        view = build_contextual_suggestion_view(interaction.channel_id, query, answer)
        if len(answer) <= 1900:
            await interaction.followup.send(answer, view=view)
        else:
            await interaction.followup.send("❄️ ข้อมูลจาก Yukipedia พร้อมแล้วค่ะ:", view=view)
            await send_chunked_response(interaction.channel, answer)

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
        view = CodeReviewActionView(interaction.user.id, interaction.channel_id, base_text, answer)
        if len(answer) <= 1900:
            await interaction.followup.send(answer, view=view)
        else:
            await interaction.followup.send("❄️ วิเคราะห์โค้ด/Stack Trace เรียบร้อยแล้วค่ะ:", view=view)
            await send_chunked_response(interaction.channel, answer)

    @bot.tree.context_menu(name="Yukipedia Fact-Check / Summary")
    async def ctx_yukipedia_summary(interaction: discord.Interaction, message: discord.Message):
        await interaction.response.defer(thinking=True)
        extracted_ctx, _ = await extract_all_context(message)
        base_text = message.content.strip() or "ช่วยตรวจสอบข้อเท็จจริงและสรุปเนื้อหาในข้อความนี้ให้หน่อย"
        full_prompt = f"{base_text}\n\n{extracted_ctx}" if extracted_ctx else base_text

        answer, _ = await query_local_ollama_chat(
            full_prompt,
            model=ACTIVE_FAST_MODEL,
            system_prompt=YUKIPEDIA_SYSTEM_PROMPT,
            channel_id=interaction.channel_id,
            temperature=0.3,
            allow_agentic_search=True,
            has_fresh_attachment=bool(extracted_ctx),
            speaker_user=interaction.user,
        )
        view = build_contextual_suggestion_view(interaction.channel_id, base_text, answer)
        if len(answer) <= 1900:
            await interaction.followup.send(answer, view=view)
        else:
            await interaction.followup.send("❄️ ตรวจสอบและสรุปข้อมูลเรียบร้อยแล้วค่ะ:", view=view)
            await send_chunked_response(interaction.channel, answer)

    @bot.tree.context_menu(name="Translate Manga Page (TH)")
    async def ctx_translate_manga_th(interaction: discord.Interaction, message: discord.Message):
        has_imgs = any(a.filename.lower().endswith(IMAGE_EXTENSIONS) for a in message.attachments)
        if not has_imgs:
            await interaction.response.send_message(
                "ข้อความที่เลือกไม่มีไฟล์รูปภาพแนบอยู่ค่ะ กรุณาคลิกขวาที่ข้อความที่มีรูปมังงะนะคะ",
                ephemeral=True,
            )
            return
        await interaction.response.send_message("📖 กำลังแปลหน้ามังงะเป็นภาษาไทยให้สักครู่นะคะ...", ephemeral=True)
        await process_manga_translation_request(message, "แปลมังงะเป็นภาษาไทย")