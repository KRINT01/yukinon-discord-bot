"""
Assert-based self-check for Batch 2 fixes (no test framework needed).
Run from the repo root:  python tests/check_batch2.py
Item 6 (fallback tool re-runs) has no check: the logic is nested inside query_ai_chat with no cheap seam.
"""
import os
import sys
import json
import time
import asyncio
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import llm
import llm_providers
import decision_orchestrator as dec
import user_memory


class _RaisingSession:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("network down")


async def check_groq_discovery_negative_cache():
    real_session = llm_providers.get_http_session
    real_key = llm_providers.GROQ_API_KEY
    sess = _RaisingSession()
    llm_providers.get_http_session = lambda: sess
    llm_providers.GROQ_API_KEY = "test-key"
    llm_providers._GROQ_AVAILABLE_MODELS_CACHE = []
    llm_providers._GROQ_MODELS_FAIL_UNTIL = 0.0
    try:
        assert await llm_providers.discover_available_groq_models() == []
        assert await llm_providers.discover_available_groq_models() == []
        assert sess.calls == 1, f"failed fetch must be negatively cached, got {sess.calls} requests"

        # Candidate list is capped at 3 even when many live models are cached
        llm_providers._GROQ_AVAILABLE_MODELS_CACHE = [f"vendor/model-{i}" for i in range(10)]
        llm_providers._GROQ_MODELS_LAST_FETCH_TS = time.time()
        cands = await llm_providers.get_groq_search_and_chat_candidates()
        assert 0 < len(cands) <= 3, cands
    finally:
        llm_providers.get_http_session = real_session
        llm_providers.GROQ_API_KEY = real_key
        llm_providers._GROQ_AVAILABLE_MODELS_CACHE = []
        llm_providers._GROQ_MODELS_LAST_FETCH_TS = 0.0
        llm_providers._GROQ_MODELS_FAIL_UNTIL = 0.0


async def check_malformed_decision_answers():
    real_call = dec.call_gemini_decision_orchestrator
    real_backend = dec._has_any_decision_backend
    shape = {"route": "x", "needs_web_search": {"noul": "abc"}}

    async def _fake(*args, **kwargs):
        return shape

    dec.call_gemini_decision_orchestrator = _fake
    dec._has_any_decision_backend = lambda allow_cf=True: True
    try:
        assert await dec.orchestrate_message_with_decision_1("ช่วยหาข่าวเกมใหม่ให้หน่อย") is None
        assert await dec.orchestrate_voice_utterance_with_decision_1("ยูกิโนะ ช่วยหาข่าวหน่อย") is None

        shape = {"is_stop_scribe_command": {"noul": "abc"}}
        assert await dec.check_voice_scribe_stop_with_decision_1("ยูกิโนะ สรุปให้หน่อย") is None

        shape = {"verdict": "x"}
        assert await dec.arbitrate_fact_check_with_decision_1("claim", "challenge", "evidence") is None

        shape = {"rel_1": {"score": "bad"}}
        items = [{"title": f"t{i}", "href": f"https://e.example/{i}", "body": "b"} for i in range(6)]
        kept = await dec.filter_search_snippets_with_decision_1("q", items, max_keep=4)
        assert kept == items[:4], kept
    finally:
        dec.call_gemini_decision_orchestrator = real_call
        dec._has_any_decision_backend = real_backend

    assert dec._parse_image_triage_answers({"image_type": {"choice": "a", "confidence": "bad"}}) is None


async def check_memory_consolidation():
    real_groq_key = config.GROQ_API_KEY
    real_gemma = llm.call_gemma_ai_studio_chat
    real_save = user_memory.save_user_profiles_local_async
    real_backup = user_memory.backup_user_profiles_to_discord
    uid = "batch2-uid"
    pinned = ["Uses an RTX 4090 GPU at home", "Has a cat named Mochi"]
    notes = [f"Recent context note number {i}" for i in range(11)]
    llm_out = json.dumps({
        "pinned_facts": ["Has a cat named Mochi", "Works as a Java backend developer"],
        "custom_notes": ["Merged recent context note"],
    })
    mutate = {"on": False}

    async def _fake_gemma(*args, **kwargs):
        if mutate["on"]:
            user_memory.user_profiles[uid]["custom_notes"].append("Added while consolidating")
        return llm_out

    async def _noop(*args, **kwargs):
        return None

    config.GROQ_API_KEY = ""
    llm.call_gemma_ai_studio_chat = _fake_gemma
    user_memory.save_user_profiles_local_async = _noop
    user_memory.backup_user_profiles_to_discord = _noop
    try:
        # Case 1: LLM drops a pinned fact -> it is kept, the new one is appended
        user_memory.user_profiles[uid] = {"custom_notes": list(notes), "pinned_facts": list(pinned)}
        user_memory._CONSOLIDATION_COOLDOWN.pop(uid, None)
        await user_memory.consolidate_user_memory_async(uid)
        prof = user_memory.user_profiles[uid]
        assert prof["pinned_facts"] == pinned + ["Works as a Java backend developer"], prof["pinned_facts"]
        assert prof["custom_notes"] == ["Merged recent context note"], prof["custom_notes"]

        # Case 2: profile changes during the LLM await -> nothing is overwritten, cooldown cleared
        user_memory.user_profiles[uid] = {"custom_notes": list(notes), "pinned_facts": list(pinned)}
        user_memory._CONSOLIDATION_COOLDOWN.pop(uid, None)
        mutate["on"] = True
        await user_memory.consolidate_user_memory_async(uid)
        prof = user_memory.user_profiles[uid]
        assert prof["pinned_facts"] == pinned, prof["pinned_facts"]
        assert prof["custom_notes"] == notes + ["Added while consolidating"], prof["custom_notes"]
        assert uid not in user_memory._CONSOLIDATION_COOLDOWN, "skip must allow a retry"
    finally:
        config.GROQ_API_KEY = real_groq_key
        llm.call_gemma_ai_studio_chat = real_gemma
        user_memory.save_user_profiles_local_async = real_save
        user_memory.backup_user_profiles_to_discord = real_backup
        user_memory.user_profiles.pop(uid, None)
        user_memory._CONSOLIDATION_COOLDOWN.pop(uid, None)


async def check_mentioned_names_not_saved():
    real = (
        user_memory.save_user_profiles_local,
        user_memory.save_user_profiles_local_async,
        user_memory.backup_user_profiles_to_discord,
        user_memory._schedule_bg_task,
    )

    async def _noop(*args, **kwargs):
        return None

    def _no_bg(coro):
        coro.close()
        return True

    user_memory.save_user_profiles_local = lambda: None
    user_memory.save_user_profiles_local_async = _noop
    user_memory.backup_user_profiles_to_discord = _noop
    user_memory._schedule_bg_task = _no_bg
    member = SimpleNamespace(display_name="Bobby", name="bobby_x", global_name="Bobby", roles=[])
    guild = SimpleNamespace(get_member=lambda i: member if i == 555 else None)
    speaker = SimpleNamespace(id=777, bot=False, display_name="Speaker", name="speaker_y")
    try:
        user_memory.user_profiles["555"] = {"display_name": "Bobby", "interaction_count": 3}
        resolved, _ = user_memory.resolve_user_mentions_and_build_dossier(
            "<@555> ชอบกินอะไร", guild, 1, speaker_user=speaker
        )
        assert "@Bobby" in resolved, resolved

        user_memory.record_user_interaction(speaker, resolved, learned_memo="Bobby เป็นเพื่อนสนิท")
        prof = user_memory.user_profiles["777"]
        for key in ("recent_topics", "custom_notes", "favorite_topics"):
            assert not any("bobby" in str(x).lower() for x in prof.get(key) or []), (key, prof.get(key))
        assert "ชอบกินอะไร" in (prof.get("recent_topics") or []), prof.get("recent_topics")

        _, target = await user_memory.apply_tool_memory_action(speaker, "pin_fact", "Bobby ชอบแมว")
        assert target == "" and not prof.get("pinned_facts"), prof.get("pinned_facts")
        _, target = await user_memory.apply_tool_memory_action(speaker, "pin_fact", "ชอบกินราเมงร้อนๆ")
        assert target == "777" and user_memory.user_profiles["777"]["pinned_facts"] == ["ชอบกินราเมงร้อนๆ"]
        _, target = await user_memory.apply_tool_memory_action(speaker, "add_note", "ไปเที่ยวกับ @bobby_x")
        assert target == "" and not user_memory.user_profiles["777"].get("custom_notes")
    finally:
        (
            user_memory.save_user_profiles_local,
            user_memory.save_user_profiles_local_async,
            user_memory.backup_user_profiles_to_discord,
            user_memory._schedule_bg_task,
        ) = real
        user_memory.user_profiles.pop("555", None)
        user_memory.user_profiles.pop("777", None)
        user_memory._MENTIONED_NAMES_BY_SPEAKER.pop("777", None)


async def _main():
    await check_groq_discovery_negative_cache()
    await check_malformed_decision_answers()
    await check_memory_consolidation()
    await check_mentioned_names_not_saved()
    print("Batch 2 checks passed.")


if __name__ == "__main__":
    asyncio.run(_main())
