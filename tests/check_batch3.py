"""
Assert-based self-check for Batch 3 prompt rule fixes (no test framework needed).
Run from the repo root:  python tests/check_batch3.py
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat_pipeline
import persona
import tools

TABLE_MARK = "Table Auto-Formatter Rule"
MIN_ARGS = dict(system_prompt=persona.FAST_SYSTEM_PROMPT, mode="general", channel_id=None)


def check_code_prompt_language():
    code = persona.CODE_SYSTEM_PROMPT
    assert "ตอบเป็นภาษาไทยที่สุภาพ" not in code
    assert "พิมพ์อังกฤษตอบอังกฤษ" in code and "ค่ะ" in code and "นะคะ" in code
    assert "แท็กเปิด" in code  # code-fence rules kept
    # English output must not be rewritten with Thai tokens, in owner or non-owner context
    for owner in (True, False):
        out = persona.sanitize_yukino_output(
            "Hi Krint, here is the fix: `x = 1`", "fix my code",
            speaker_user=None if owner else type("U", (), {"display_name": "Bob", "id": 1})(),
            speaker_name="Bob",
        )
        assert not persona.THAI_CHAR_REGEX.search(out), out
    th = persona.sanitize_yukino_output("แก้แล้วครับ Krint ดูนะ", "แก้โค้ดให้หน่อย", speaker_user=None)
    assert "นาย" in th and "ครับ" not in th, th


def check_fast_prompt_tools():
    fast = persona.FAST_SYSTEM_PROMPT
    spec_names = {s["name"] for s in tools.TOOL_SPECS}
    for name in ("web_search", "read_webpage", "get_weather", "calculate", "manage_overtime",
                 "manage_user_memory", "generate_image", "add_reaction", "youtube_inspect"):
        assert name in spec_names and f"`{name}`" in fast, name
    assert "ได้เฉพาะเมื่อระบบส่งเครื่องมือมาให้" in fast and "ห้ามอ้างว่าได้ค้นหา บันทึกความจำ" in fast


def check_table_rule_single_source():
    for p in (persona.FAST_SYSTEM_PROMPT, persona.YUKIPEDIA_SYSTEM_PROMPT, persona.CODE_SYSTEM_PROMPT):
        assert "สร้างตาราง Markdown" not in p and "| หัวข้อ |" not in p
    src = open(chat_pipeline.__file__, encoding="utf-8").read()
    assert "Overrides Blanket Table Ban" not in src and src.count(TABLE_MARK) == 2


class _Recorder:
    def __init__(self):
        self.systems = []

    async def __call__(self, messages, *args, **kwargs):
        self.systems.append(messages[0]["content"])
        return "ok"


async def check_static_prompt_stable_and_table_flags():
    rec = _Recorder()
    saved = {}
    for name in dir(chat_pipeline):
        if name.startswith("call_") and callable(getattr(chat_pipeline, name)):
            saved[name] = getattr(chat_pipeline, name)
            setattr(chat_pipeline, name, rec)
    try:
        async def run(prompt, **plan):
            before = len(rec.systems)
            await chat_pipeline.query_ai_chat(prompt, decision_plan=plan, allow_agentic_search=False, **MIN_ARGS)
            assert len(rec.systems) > before, "no provider call was captured"
            return rec.systems[before]

        a = await run("[ผู้ใช้] สวัสดี", complexity_score=0.1, complexity_confidence=0.9)
        b = await run("[ผู้ใช้] วันนี้อากาศเป็นยังไงบ้าง ขอยาวๆ หน่อย", complexity_score=0.3, complexity_confidence=0.9)
        assert a == b, "static_system_prompt must be identical across turns"
        assert a.count(TABLE_MARK) == 1 and "Strict Mode" not in a
        c = await run("[ผู้ใช้] สวัสดี", no_tables=True)
        assert c.count(TABLE_MARK) == 1 and "Strict Mode" in c
    finally:
        for name, fn in saved.items():
            setattr(chat_pipeline, name, fn)


def check_fact_check_block_conditional():
    src = open(chat_pipeline.__file__, encoding="utf-8").read()
    assert 'f"2. {fact_check_evidence_line}' in src
    assert src.count("fact_check_evidence_line =") == 3
    assert '"2. ตรวจสอบกับหลักฐานใน `<untrusted_web_search_results>` ด้านบน' not in src


def check_ot_channel_matcher_and_meta_key():
    import user_memory as um
    for ok in ("work", "ot", "ot-report", "my_work", "daily ot", "แจ้งงาน-ot"):
        assert um.is_ot_work_channel_name(ok), ok
    for bad in ("bot", "photos", "robot", "bot-commands", "general"):
        assert not um.is_ot_work_channel_name(bad), bad
    saved = dict(um.user_profiles)
    try:
        um.user_profiles.clear()
        um.user_profiles["123"] = {"ot_logs": []}
        assert um.user_profile_count() == 1 and um.get_last_ot_report_ym() == ""
        um.user_profiles[um.META_KEY] = {"last_ot_report_ym": "2026-10"}
        assert um.user_profile_count() == 1 and um.get_last_ot_report_ym() == "2026-10"
        assert not um.META_KEY.isdigit()  # cannot collide with Discord user ids
    finally:
        um.user_profiles.clear()
        um.user_profiles.update(saved)


async def _main():
    check_ot_channel_matcher_and_meta_key()
    check_code_prompt_language()
    check_fast_prompt_tools()
    check_table_rule_single_source()
    check_fact_check_block_conditional()
    await check_static_prompt_stable_and_table_flags()
    print("Batch 3 checks passed.")


if __name__ == "__main__":
    asyncio.run(_main())
