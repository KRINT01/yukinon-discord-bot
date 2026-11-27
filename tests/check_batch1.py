"""
Assert-based self-check for Batch 1 fixes (no test framework needed).
Run from the repo root:  python tests/check_batch1.py
"""
import os
import sys
import asyncio
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import persona
import search_engine
import user_memory
import bot as bot_module


def check_owner_is_id_only():
    spoof = SimpleNamespace(id=111, name="krint", display_name="Krint", global_name="krint")
    real = SimpleNamespace(id=999, name="someone", display_name="x", global_name="x")

    persona.OWNER_USER_ID = ""
    assert not persona.is_owner_user(spoof), "owner features must be OFF when OWNER_USER_ID is unset"
    assert not persona.is_owner_user(real)

    persona.OWNER_USER_ID = "999"
    assert persona.is_owner_user(real), "owner must match by ID"
    assert not persona.is_owner_user(spoof), "name 'krint' must not grant owner"
    assert not persona.is_owner_user(None)


class _FakeResp:
    def __init__(self, status, location=None):
        self.status = status
        self.headers = {"Location": location} if location else {}
        self.released = False

    def release(self):
        self.released = True


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requested = []

    async def get(self, url, allow_redirects=True, **kwargs):
        assert allow_redirects is False, "redirects must be followed manually"
        self.requested.append(url)
        return self._responses.pop(0)


async def check_ssrf_redirects():
    real_validate = search_engine.validate_safe_public_url_async

    async def _validate(url):
        # Pretend the public test host resolves publicly (no DNS in the check); everything else is real
        if url.startswith("https://public.example/"):
            return url
        return await real_validate(url)

    search_engine.validate_safe_public_url_async = _validate
    try:
        # Redirect to loopback must be blocked
        sess = _FakeSession([_FakeResp(302, "http://127.0.0.1/admin")])
        try:
            await search_engine.open_safe_public_url_async(sess, "https://public.example/a")
            raise AssertionError("redirect to 127.0.0.1 was not blocked")
        except ValueError as e:
            assert "SSRF" in str(e)

        # Relative redirect on the same public host is followed; final response is returned
        final = _FakeResp(200)
        sess = _FakeSession([_FakeResp(301, "/b"), final])
        resp = await search_engine.open_safe_public_url_async(sess, "https://public.example/a")
        assert resp is final and sess.requested == ["https://public.example/a", "https://public.example/b"]

        # Redirect loop is capped
        sess = _FakeSession([_FakeResp(302, "/loop") for _ in range(10)])
        try:
            await search_engine.open_safe_public_url_async(sess, "https://public.example/a", max_redirects=3)
            raise AssertionError("redirect loop was not capped")
        except ValueError as e:
            assert "Too many redirects" in str(e)
    finally:
        search_engine.validate_safe_public_url_async = real_validate


def check_profiles_dict_identity(tmp_dir):
    path = os.path.join(tmp_dir, "user_profiles.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"42": {"user_id": "42", "ot_logs": []}}')
    user_memory.USER_PROFILES_FILE = path
    original_obj = user_memory.user_profiles
    user_memory.load_user_profiles_local()
    assert user_memory.user_profiles is original_obj, "dict must be mutated in place, not rebound"
    assert bot_module.user_profiles is user_memory.user_profiles, "bot.py must see the loaded profiles"
    assert "42" in bot_module.user_profiles


async def check_ot_invalid_minutes():
    user = SimpleNamespace(id=42, name="u", display_name="u", bot=False)
    msg = SimpleNamespace(author=user, channel=SimpleNamespace(id=7), guild=None)
    user_memory.ensure_user_profile_exists("42")

    reply = await user_memory.process_ot_request(msg, "จด OT 18.75 - 20.00")
    assert reply == user_memory._OT_INVALID_TIME_REPLY, reply

    reply = await user_memory.process_ot_request(msg, "จด OT ถึง 19.99")
    assert reply == user_memory._OT_INVALID_TIME_REPLY, reply

    reply = await user_memory.process_ot_request(msg, "จด OT 18.00 - 20.30 แก้บั๊ก")
    assert "OT_RECORD" in reply and "2.50" in reply, reply


async def check_bg_task_is_referenced():
    done = asyncio.Event()

    async def _job():
        done.set()

    assert user_memory._schedule_bg_task(_job())
    assert any(not t.done() for t in user_memory._BG_TASKS), "task must be held until done"
    await asyncio.wait_for(done.wait(), 1.0)
    await asyncio.sleep(0)
    assert not user_memory._BG_TASKS, "finished task must be released"


def check_mass_ping():
    assert bot_module.bot.allowed_mentions.everyone is False
    assert bot_module.bot.allowed_mentions.roles is False

    def _msg(guild_id, can_ping_everyone):
        perms = SimpleNamespace(mention_everyone=can_ping_everyone)
        guild = SimpleNamespace(id=guild_id, get_role=lambda rid: None)
        return SimpleNamespace(
            guild=guild,
            author=SimpleNamespace(id=1),
            channel=SimpleNamespace(permissions_for=lambda _m: perms),
        )

    bot_module._last_mass_ping_at.clear()
    ans = "@everyone ประชุม 10 โมงนะคะ"

    # Not requested by the user -> no override (safe default applies)
    am, out = bot_module._resolve_mass_ping(_msg(1, True), "สรุปประชุมให้หน่อย", ans)
    assert am is None and out == ans

    # Requested but author lacks permission -> no ping, note appended
    am, out = bot_module._resolve_mass_ping(_msg(1, False), "ช่วยแท็กทุกคนว่าประชุม 10 โมง", ans)
    assert am is None and out != ans

    # Requested with permission -> ping allowed once
    am, out = bot_module._resolve_mass_ping(_msg(1, True), "ช่วยแท็กทุกคนว่าประชุม 10 โมง", ans)
    assert am is not None and am.everyone is True and out == ans

    # Second request in the same guild within cooldown -> suppressed with a note
    am, out = bot_module._resolve_mass_ping(_msg(1, True), "ping everyone again", ans)
    assert am is None and "คูลดาวน์" in out

    # Cooldown is per guild
    am, _ = bot_module._resolve_mass_ping(_msg(2, True), "ping everyone", ans)
    assert am is not None


async def _main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        check_owner_is_id_only()
        await check_ssrf_redirects()
        check_profiles_dict_identity(tmp_dir)
        await check_ot_invalid_minutes()
        await check_bg_task_is_referenced()
        check_mass_ping()
    print("Batch 1 checks passed.")


if __name__ == "__main__":
    asyncio.run(_main())
