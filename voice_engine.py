import os
import io
import re
import time
import wave
import logging
import audioop
import tempfile
import threading
import asyncio
import subprocess
from collections import defaultdict, deque
from urllib.parse import quote_plus

import aiohttp
import discord
import discord.gateway as _dpy_gateway
import discord.voice_state as _dpy_voice_state
import edge_tts
import speech_recognition as sr

from config import (
    HTTPS,
    TTS_VOICE_TH,
    TTS_VOICE_EN,
    TTS_PITCH,
    TTS_RATE,
    STT_PRIMARY_LANG,
    DDGS_PROXY,
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    ACTIVE_FAST_MODEL,
    VOICE_WAKE_WORDS,
    THAI_CHAR_REGEX,
    URL_REGEX,
    CAT_TOPIC_REGEX,
    STUTTER_REGEX,
    tts_lock,
    voice_wake_word_enabled,
    state,
    davey,
    clean_url,
    get_http_session,
)
from persona import sanitize_yukino_output, VOICE_SYSTEM_PROMPT
from llm import query_local_ollama_chat

# Silence noisy RTCP SenderReport logs (type=200) and opus flush warnings from discord-ext-voice-recv
logging.getLogger("discord.ext.voice_recv").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.ERROR)

OPUS_SILENCE_FRAME = b"\xf8\xff\xfe"
PCM_SILENCE_20MS = b"\x00" * 3840

# Host constants split to prevent chat/markdown auto-linkers from mangling URLs
GTRANS_HOST = "translate." + "googleapis.com"
GROQ_API_HOST = "api." + "groq.com"

# Spoken test phrases (including common STT phonetic variations like ชัด/แชท, ตัดสอบ/ทดสอบ, หยิน/ยิน)
VOICE_TEST_PHRASE_REGEX = re.compile(
    r"(ได้ยินไหม|ได้ยินมั้ย|ได้ยินหรือเปล่า|ได้ยินฉันไหม|ไม่หยินไหม|ได้มิดิน|"
    r"ถ้าได้ยิน|ถ้าได้อีกแล้ว|พิมพ์ตอบในแชท|พิมพ์ตอบในชัด|ตอบในแชท|ตอบในชัด|"
    r"เทสเสียง|ทดสอบเสียง|ทดสอบไมค์|ตัดสอบ|ฮัลโหลเทส|"
    r"can\s+you\s+hear\s+me|if\s+you\s+hear\s+me|reply\s+in\s+chat|mic\s+test|testing\s+1\s*2\s*3)",
    re.IGNORECASE,
)

# Vocabulary hint for Whisper so Thai commands & Yukino's names are spelled accurately
WHISPER_THAI_PROMPT_HINT = (
    "ยูกิโนะ ยูกิน่อน สวัสดีค่ะ แนะนำตัวหน่อย ได้ยินไหม ถ้าได้ยินแล้วพิมพ์ตอบในแชท "
    "ทดสอบไมค์ เทสเสียง ทดสอบ วันนี้วันอะไร ตอนนี้เวลากี่โมง อากาศวันนี้เป็นยังไง"
)

STT_PHONETIC_FIXES = (
    (re.compile(r"ถ้าได้อีกแล้ว\s*พิมพ์ตอบในชัด"), "ถ้าได้ยินแล้วพิมพ์ตอบในแชท"),
    (re.compile(r"พิมพ์ตอบในชัด"), "พิมพ์ตอบในแชท"),
    (re.compile(r"ไม่หยินไหม"), "ได้ยินไหม"),
    (re.compile(r"ได้มิดินมัย"), "ได้ยินไหม"),
    (re.compile(r"แม่นำตัว"), "แนะนำตัว"),
    (re.compile(r"แนะนําตัว"), "แนะนำตัว"),
    (re.compile(r"ตัดสอบ"), "ทดสอบ"),
    (re.compile(r"^เธอสิ่ง(กัน)?$"), "เทสเสียง"),
    (re.compile(r"^สอบสิ่ง$"), "ทดสอบเสียง"),
)

_NATIVE_DPY_VOICE_WS_RECEIVED = _dpy_gateway.DiscordVoiceWebSocket.received_message
_NATIVE_DPY_VOICE_WS_BINARY = getattr(_dpy_gateway.DiscordVoiceWebSocket, "received_binary_message", None)
_ORIG_DPY_EXECUTE_TRANSITION = getattr(_dpy_voice_state.VoiceConnectionState, "_execute_transition", None)

try:
    from discord.ext import voice_recv
    import discord.ext.voice_recv.opus as _vr_opus
    import discord.ext.voice_recv.gateway as _vr_gateway
    import discord.ext.voice_recv.voice_client as _vr_vc

    _ORIG_VR_HOOK = getattr(_vr_gateway, "hook", None)
    _ORIG_DECODE_PACKET = _vr_opus.PacketDecoder._decode_packet
    _DAVE_LOG_TS = 0.0
    _KNOWN_VOICE_USER_IDS: set[int] = set()

    if _ORIG_DPY_EXECUTE_TRANSITION is not None:
        async def _patched_dpy_execute_transition(self, transition_id: int):
            await _ORIG_DPY_EXECUTE_TRANSITION(self, transition_id)
            dave_s = getattr(self, "dave_session", None)
            if dave_s is not None and hasattr(dave_s, "execute_transition"):
                try:
                    dave_s.execute_transition(int(transition_id))
                    print(
                        f"[DAVE Transition] Executed transition_id={transition_id} -> "
                        f"ready={getattr(dave_s, 'ready', '?')}, can_encrypt={getattr(dave_s, 'can_encrypt', '?')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[DAVE Transition Note] execute_transition({transition_id}): {e}", flush=True)

        _dpy_voice_state.VoiceConnectionState._execute_transition = _patched_dpy_execute_transition

    def _activate_dave_epoch_transition(conn, transition_id: int, b_op: int):
        dave_s = getattr(conn, "dave_session", None) if conn is not None else None
        if dave_s is None:
            return
        proto_ver = getattr(conn, "dave_protocol_version", None) or getattr(dave_s, "protocol_version", 1) or 1

        if hasattr(dave_s, "prepare_transition"):
            try:
                dave_s.prepare_transition(int(transition_id), int(proto_ver))
            except Exception as e:
                print(f"[DAVE Transition] prepare_transition({transition_id}, v={proto_ver}) note: {e}", flush=True)

        if hasattr(dave_s, "execute_transition"):
            try:
                dave_s.execute_transition(int(transition_id))
            except Exception as e:
                print(f"[DAVE Transition] execute_transition({transition_id}) note: {e}", flush=True)

        if not getattr(dave_s, "ready", False) and transition_id != 0:
            try:
                dave_s.prepare_transition(0, int(proto_ver))
                dave_s.execute_transition(0)
            except Exception:
                pass

    def _bridge_conn_to_vc(ws):
        conn = getattr(ws, "_connection", None)
        vc = getattr(conn, "voice_client", None) if conn is not None else None
        if vc is None and state.bot is not None and getattr(state.bot, "voice_clients", None):
            vc = state.bot.voice_clients[0]

        if conn is not None and vc is not None and conn is not vc:
            for attr_name in (
                "_add_ssrc", "_remove_ssrc", "_get_id_from_ssrc",
                "_packet_router", "_speaking_timer", "_reader", "_decoder", "guild"
            ):
                if not hasattr(conn, attr_name) and hasattr(vc, attr_name):
                    try:
                        setattr(conn, attr_name, getattr(vc, attr_name))
                    except Exception:
                        pass
        return conn, vc

    async def _inject_dave_clients_connect(ws, conn, vc):
        uid_ints = set(_KNOWN_VOICE_USER_IDS)
        if vc is not None and getattr(vc, "channel", None):
            for m in vc.channel.members:
                uid_ints.add(int(m.id))
        if state.bot is not None and getattr(state.bot, "user", None):
            uid_ints.add(int(state.bot.user.id))

        if uid_ints:
            try:
                await _NATIVE_DPY_VOICE_WS_RECEIVED(
                    ws, {"op": 11, "d": {"user_ids": [str(u) for u in uid_ints]}}
                )
            except Exception as e:
                print(f"[DAVE op=11 Sync Note] {type(e).__name__}: {e}", flush=True)

    async def _safe_vr_hook(ws, msg):
        if not isinstance(msg, dict):
            return
        _bridge_conn_to_vc(ws)
        op = msg.get("op")
        if _ORIG_VR_HOOK is not None and op in (2, 4, 5, 8, 12, 13, 18, 20):
            try:
                await _ORIG_VR_HOOK(ws, msg)
            except Exception as e:
                print(f"[Voice Hook Ignored] op={op} {type(e).__name__}: {e}", flush=True)

    if _ORIG_VR_HOOK is not None:
        _vr_gateway.hook = _safe_vr_hook
        setattr(_vr_vc, "hook", _safe_vr_hook)

    async def _dpy27_dave_compatible_received_message(self, msg):
        self._hook = _safe_vr_hook
        conn, vc = _bridge_conn_to_vc(self)

        if isinstance(msg, dict):
            op = msg.get("op")
            data = msg.get("d")
            if op in (21, 22, 24):
                print(f"[DAVE WS JSON] Transition event op={op} d={data}", flush=True)
            if isinstance(data, dict):
                uid_val = data.get("user_id")
                if uid_val and str(uid_val).isdigit():
                    _KNOWN_VOICE_USER_IDS.add(int(uid_val))
                if isinstance(data.get("user_ids"), list):
                    for u in data["user_ids"]:
                        if u and str(u).isdigit():
                            _KNOWN_VOICE_USER_IDS.add(int(u))

                if vc is not None and hasattr(vc, "_add_ssrc"):
                    if op == 5 and uid_val and data.get("ssrc") is not None:
                        try:
                            vc._add_ssrc(int(uid_val), int(data["ssrc"]))
                            print(f"[Voice WS] Mapped SSRC {data['ssrc']} -> user_id {uid_val}", flush=True)
                        except Exception:
                            pass
                    elif op == 12 and uid_val and data.get("audio_ssrc"):
                        try:
                            vc._add_ssrc(int(uid_val), int(data["audio_ssrc"]))
                            print(f"[Voice WS] Mapped audio_ssrc {data['audio_ssrc']} -> user_id {uid_val}", flush=True)
                        except Exception:
                            pass

            await _NATIVE_DPY_VOICE_WS_RECEIVED(self, msg)

            dave_s = getattr(conn, "dave_session", None) if conn is not None else None
            if dave_s is not None and isinstance(data, dict):
                t_id = int(data.get("transition_id", 0))
                if op == 21 and hasattr(dave_s, "prepare_transition"):
                    p_ver = int(data.get("protocol_version", getattr(conn, "dave_protocol_version", 1) or 1))
                    try:
                        dave_s.prepare_transition(t_id, p_ver)
                    except Exception:
                        pass
                elif op == 22 and hasattr(dave_s, "execute_transition"):
                    try:
                        dave_s.execute_transition(t_id)
                        print(f"[DAVE JSON op=22] Executed transition_id={t_id}, ready={getattr(dave_s, 'ready', '?')}", flush=True)
                    except Exception:
                        pass

            if op in (4, 5, 12):
                await _inject_dave_clients_connect(self, conn, vc)
            return

        await _NATIVE_DPY_VOICE_WS_RECEIVED(self, msg)

    _dpy_gateway.DiscordVoiceWebSocket.received_message = _dpy27_dave_compatible_received_message

    if _NATIVE_DPY_VOICE_WS_BINARY is not None:
        async def _dpy27_dave_compatible_binary_message(self, msg: bytes):
            self._hook = _safe_vr_hook
            conn, vc = _bridge_conn_to_vc(self)
            b_op = msg[2] if len(msg) > 2 else -1
            transition_id = int.from_bytes(msg[3:5], "big") if len(msg) >= 5 else 0

            if b_op in (25, 27):
                await _inject_dave_clients_connect(self, conn, vc)

            print(
                f"[DAVE WS Binary] Received binary MLS frame (op={b_op}, len={len(msg)}B, transition_id={transition_id})",
                flush=True,
            )
            try:
                await _NATIVE_DPY_VOICE_WS_BINARY(self, msg)
            except Exception as e:
                print(f"[DAVE WS Binary Error] op={b_op} {type(e).__name__}: {e}", flush=True)

            if b_op in (29, 30):
                _activate_dave_epoch_transition(conn, transition_id, b_op)

            dave_s = getattr(conn, "dave_session", None) if conn is not None else None
            if dave_s is not None and b_op in (25, 27, 29, 30):
                uids = dave_s.get_user_ids() if hasattr(dave_s, "get_user_ids") else "?"
                status = getattr(dave_s, "status", "?")
                ready = getattr(dave_s, "ready", "?")
                can_enc = getattr(dave_s, "can_encrypt", "?")
                print(
                    f"[DAVE State after op={b_op}] status={status}, ready={ready}, "
                    f"can_encrypt={can_enc}, group_user_ids={uids}",
                    flush=True,
                )

        _dpy_gateway.DiscordVoiceWebSocket.received_binary_message = _dpy27_dave_compatible_binary_message

    def _safe_dave_decode_packet(self, packet):
        global _DAVE_LOG_TS
        if not packet or not getattr(packet, "decrypted_data", None):
            try:
                return _ORIG_DECODE_PACKET(self, packet)
            except Exception:
                return packet, PCM_SILENCE_20MS

        raw_payload = packet.decrypted_data
        if len(raw_payload) > 3 and raw_payload != OPUS_SILENCE_FRAME:
            try:
                sink_obj = getattr(self, "sink", None)
                vc = (
                    getattr(sink_obj, "voice_client", None)
                    or getattr(sink_obj, "_voice_client", None)
                )
                if vc is None and state.bot is not None and getattr(state.bot, "voice_clients", None):
                    vc = state.bot.voice_clients[0]

                conn = getattr(vc, "_connection", None) if vc is not None else None
                dave_session = getattr(conn, "dave_session", None) if conn is not None else None
                dave_version = getattr(conn, "dave_protocol_version", 0) if conn is not None else 0

                if dave_session is not None and dave_version > 0 and davey is not None:
                    if not getattr(dave_session, "ready", False) and str(getattr(dave_session, "status", "")).endswith("active"):
                        _activate_dave_epoch_transition(conn, 0, 29)

                    ssrc = getattr(self, "ssrc", None) or getattr(packet, "ssrc", None)
                    user_id = getattr(self, "_cached_id", None) or (
                        vc._get_id_from_ssrc(ssrc) if (vc and hasattr(vc, "_get_id_from_ssrc") and ssrc) else None
                    )

                    if not user_id and vc and getattr(vc, "channel", None):
                        humans = [m for m in vc.channel.members if not m.bot]
                        if len(humans) == 1:
                            user_id = int(humans[0].id)
                            self._cached_id = user_id

                    if user_id:
                        uid_int = int(user_id)
                        self._cached_id = uid_int
                        # Only attempt DAVE decrypt if the packet has the 0xFAFA magic trailer!
                        # Unencrypted comfort-noise / silence packets sent by Discord do not end with 0xFAFA.
                        is_dave_framed = raw_payload.endswith(b"\xfa\xfa")
                        if is_dave_framed:
                            packet.decrypted_data = bytes(
                                dave_session.decrypt(uid_int, davey.MediaType.audio, raw_payload)
                            )
            except Exception as e:
                now = time.monotonic()
                if now - _DAVE_LOG_TS > 5.0:
                    _DAVE_LOG_TS = now
                    print(f"[DAVE Decrypt Note] ssrc={getattr(packet, 'ssrc', '?')}: {type(e).__name__}: {e}", flush=True)
                if raw_payload.endswith(b"\xfa\xfa"):
                    packet.decrypted_data = OPUS_SILENCE_FRAME

        try:
            return _ORIG_DECODE_PACKET(self, packet)
        except Exception:
            return packet, PCM_SILENCE_20MS

    _vr_opus.PacketDecoder._decode_packet = _safe_dave_decode_packet
    HAS_VOICE_RECV = True
except ImportError:
    voice_recv = None
    HAS_VOICE_RECV = False

state.has_voice_recv = HAS_VOICE_RECV


THAI_DAY_NUMBER_WORDS = {
    1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่", 5: "ห้า",
    6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า", 10: "สิบ",
    11: "สิบเอ็ด", 12: "สิบสอง", 13: "สิบสาม", 14: "สิบสี่", 15: "สิบห้า",
    16: "สิบหก", 17: "สิบเจ็ด", 18: "สิบแปด", 19: "สิบเก้า", 20: "ยี่สิบ",
    21: "ยี่สิบเอ็ด", 22: "ยี่สิบสอง", 23: "ยี่สิบสาม", 24: "ยี่สิบสี่", 25: "ยี่สิบห้า",
    26: "ยี่สิบหก", 27: "ยี่สิบเจ็ด", 28: "ยี่สิบแปด", 29: "ยี่สิบเก้า", 30: "สามสิบ",
    31: "สามสิบเอ็ด",
}


def _format_colloquial_thai_time(hh: int, mm: int) -> str:
    """Converts 24-hour (hh, mm) into natural everyday spoken Thai time."""
    if hh == 0:
        hour_phrase = "เที่ยงคืน"
    elif 1 <= hh <= 5:
        hour_phrase = f"ตี{THAI_DAY_NUMBER_WORDS[hh]}"
    elif 6 <= hh <= 11:
        hour_phrase = f"{THAI_DAY_NUMBER_WORDS[hh]}โมงเช้า"
    elif hh == 12:
        hour_phrase = "เที่ยง"
    elif hh == 13:
        hour_phrase = "บ่ายโมง"
    elif 14 <= hh <= 15:
        hour_phrase = f"บ่าย{THAI_DAY_NUMBER_WORDS[hh - 12]}โมง"
    elif 16 <= hh <= 18:
        hour_phrase = f"{THAI_DAY_NUMBER_WORDS[hh - 12]}โมงเย็น"
    else:
        # 19:00 (หนึ่งทุ่ม) to 23:00 (ห้าทุ่ม)
        hour_phrase = f"{THAI_DAY_NUMBER_WORDS[hh - 18]}ทุ่ม"

    if mm == 0:
        if hh in (0, 12):
            return f"{hour_phrase}ตรง"
        return hour_phrase
    if mm == 30:
        return f"{hour_phrase}ครึ่ง"
    return f"{hour_phrase} {mm} นาที"


def _normalize_thai_dates_and_times_for_tts(text: str) -> str:
    """
    Converts formal dates/times into natural conversational Thai speech:
    - Strips parenthetical '(ค.ศ. 2026)' and '(เวลาประเทศไทย UTC+7)'
    - Strips 'พ.ศ.' / 'พุทธศักราช' so '14 พฤศจิกายน พ.ศ. 2569' reads as 'สิบสี่ พฤศจิกายน 2569'
    - Converts '02:35 น.' -> 'ตีสอง 35 นาที', '00:00 น.' -> 'เที่ยงคืนตรง', '22:30 น.' -> 'สี่ทุ่มครึ่ง'
    """
    # 1. Remove parenthetical ISO/CE date and timezone notes
    s = re.sub(r"\(\s*ค\.ศ\.[^)]*\)", " ", text)
    s = re.sub(r"\(\s*B\.E\.[^)]*\)", " ", s)
    s = re.sub(r"\(\s*เวลาประเทศไทย[^)]*\)", " ", s)

    # 2. Convert 24-hour time 'HH:MM น.' (or 'เวลา HH:MM') into natural colloquial Thai time
    def _time_repl(m):
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f" {_format_colloquial_thai_time(hh, mm)} "
        return m.group(0)

    s = re.sub(r"\b(\d{1,2}):(\d{2})\s*(?:น\.)?", _time_repl, s)

    # 3. Convert 'ที่ <1-31>' into Thai words ('วันเสาร์ที่สิบสี่') so TTS never inserts an extra 'วันที่'
    def _day_num_repl(m):
        prefix = m.group(1)
        day_int = int(m.group(2))
        day_word = THAI_DAY_NUMBER_WORDS.get(day_int, str(day_int))
        return f"{prefix}{day_word} "

    s = re.sub(r"(ที่)\s*(\d{1,2})\b", _day_num_repl, s)

    # 4. Drop 'พ.ศ.' / 'พุทธศักราช' so the year is read naturally as '2569'
    s = re.sub(r"(?:พ\.ศ\.|พุทธศักราช)\s*(\d{4})", r"\1", s)
    return s


def clean_text_for_speech(text: str) -> str:
    cleaned = sanitize_yukino_output(text)
    cleaned = re.sub(r"```[\s\S]*?```", " รายละเอียดของโค้ดฉันส่งไว้ให้ในแชทแล้วค่ะ ", cleaned)
    cleaned = URL_REGEX.sub(" ลิงก์อ้างอิงอยู่ในแชทแล้วค่ะ ", cleaned)
    cleaned = _normalize_thai_dates_and_times_for_tts(cleaned)
    cleaned = re.sub(r"[\(\)\*\_\~\`\#\>|\[\]\"\']", " ", cleaned)
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()[:450]


def select_tts_voice(text: str) -> str:
    return TTS_VOICE_TH if THAI_CHAR_REGEX.search(text) else TTS_VOICE_EN


def _split_tts_chunks(text: str, max_len: int = 190) -> list[str]:
    words = text.split(" ")
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = w[:max_len]
    if cur:
        chunks.append(cur)
    return chunks


def _master_tts_audio_sync(mp3_bytes: bytes, extra_af: str = "") -> bytes:
    """
    Applies a warm, natural broadcast mastering filter (softens 3.2kHz robotic sibilance,
    normalizes loudness, and outputs clean 48kHz stereo CBR MP3).
    """
    base_af = "equalizer=f=3200:t=q:w=1.2:g=-2.5,acompressor=threshold=-18dB:ratio=2:attack=15:release=150,volume=1.35"
    full_af = f"{extra_af},{base_af}" if extra_af else base_af
    proc = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-af", full_af,
            "-ac", "2", "-ar", "48000", "-b:a", "128k",
            "-write_xing", "0",
            "-f", "mp3", "pipe:1", "-loglevel", "error"
        ],
        input=mp3_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return proc.stdout


async def _fallback_google_tts_with_yukino_pitch(spoken_text: str, lang_code: str) -> bytes:
    raw_mp3_parts = bytearray()
    headers = {"User-Agent": "Mozilla/5.0"}
    session = get_http_session()
    for chunk in _split_tts_chunks(spoken_text, max_len=190):
        tts_url = (
            f"{HTTPS}{GTRANS_HOST}/translate_tts"
            f"?ie=UTF-8&client=tw-ob&tl={lang_code}&q={quote_plus(chunk)}"
        )
        async with session.get(
            tts_url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)
        ) as resp:
            if resp.status == 200:
                raw_mp3_parts.extend(await resp.read())

    if not raw_mp3_parts:
        return b""

    # Sped up to atempo=1.15 (~1.20x total with asetrate=1.04) so Google TTS matches natural speaking speed
    steady_af = "asetrate=24000*1.04,aresample=48000,atempo=1.15"
    try:
        return await asyncio.to_thread(_master_tts_audio_sync, bytes(raw_mp3_parts), steady_af)
    except Exception:
        return bytes(raw_mp3_parts)


async def synthesize_tts_mp3_bytes(text: str) -> bytes:
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return b""
    is_thai = bool(THAI_CHAR_REGEX.search(spoken_text))
    voice_name = TTS_VOICE_TH if is_thai else TTS_VOICE_EN

    if time.time() >= state.edge_tts_cooldown_until:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = tmp_mp3.name
        try:
            # Avoid passing +0Hz/+0% which edge-tts rejects; only pass rate/pitch if non-zero
            comm_kwargs = {"text": spoken_text, "voice": voice_name, "proxy": DDGS_PROXY}
            if TTS_RATE and TTS_RATE not in ("+0%", "-0%", "0%"):
                comm_kwargs["rate"] = TTS_RATE
            if TTS_PITCH and TTS_PITCH not in ("+0Hz", "-0Hz", "0Hz"):
                comm_kwargs["pitch"] = TTS_PITCH

            communicate = edge_tts.Communicate(**comm_kwargs)
            await asyncio.wait_for(communicate.save(mp3_path), timeout=4.5)
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 256:
                with open(mp3_path, "rb") as f:
                    raw_edge_mp3 = f.read()
                try:
                    return await asyncio.to_thread(_master_tts_audio_sync, raw_edge_mp3)
                except Exception:
                    return raw_edge_mp3
        except Exception as e:
            # Cooldown for 30m when Render's IP is throttled by Microsoft so voice doesn't flip-flop mid-session
            state.edge_tts_cooldown_until = time.time() + 1800
            print(f"edge-tts unavailable ({e}); locking to steady Google TTS + Yukino filter for 30m...", flush=True)
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass

    lang_code = "th" if is_thai else "en"
    return await _fallback_google_tts_with_yukino_pitch(spoken_text, lang_code)


async def speak_in_voice_channel(guild: discord.Guild, text: str, fallback_channel=None):
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return

    mp3_bytes = await synthesize_tts_mp3_bytes(spoken_text)
    if not mp3_bytes:
        print("TTS synthesis returned empty audio bytes.", flush=True)
        return

    if not guild or not guild.voice_client or not guild.voice_client.is_connected():
        if fallback_channel:
            await fallback_channel.send(
                file=discord.File(io.BytesIO(mp3_bytes), filename="yukinon_voice.mp3")
            )
        return

    async with tts_lock:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3.write(mp3_bytes)
            mp3_path = tmp_mp3.name
        try:
            vc = guild.voice_client
            while vc and vc.is_connected() and vc.is_playing():
                await asyncio.sleep(0.2)
            if not vc or not vc.is_connected():
                return
            done_event = asyncio.Event()
            loop = state.bot.loop if state.bot else asyncio.get_running_loop()

            def _after_play(err):
                if err:
                    print(f"FFmpeg voice playback error: {err}", flush=True)
                try:
                    os.remove(mp3_path)
                except OSError:
                    pass
                loop.call_soon_threadsafe(done_event.set)

            print(f"[Voice TTS] Playing {len(mp3_bytes)} bytes in '{vc.channel.name}'...", flush=True)
            audio_source = discord.FFmpegPCMAudio(mp3_path, options="-vn -loglevel error")
            vc.play(audio_source, after=_after_play)
            await asyncio.wait_for(done_event.wait(), timeout=60)
        except Exception as e:
            print(f"TTS voice channel playback error: {e}", flush=True)
            try:
                os.remove(mp3_path)
            except OSError:
                pass


def _apply_phonetic_fixes(text: str) -> str:
    cleaned = text.strip()
    for pattern, replacement in STT_PHONETIC_FIXES:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def transcribe_wav_bytes_google_sync(wav_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    try:
        res = recognizer.recognize_google(audio_data, language=STT_PRIMARY_LANG)
        return _apply_phonetic_fixes(res) if res else ""
    except sr.UnknownValueError:
        try:
            res = recognizer.recognize_google(audio_data, language="en-US")
            return _apply_phonetic_fixes(res) if res else ""
        except Exception:
            return ""
    except Exception as e:
        print(f"Google STT error: {e}", flush=True)
        return ""


NOISE_HALLUCINATION_REGEX = re.compile(r"(<\|.*?\|>|[ðþæøåÐÞÆØÅ]|(.)\2\2)", re.IGNORECASE)
WHISPER_HALLUCINATIONS = {
    "ขอบคุณครับ", "ขอบคุณค่ะ", "โปรดติดตามตอนต่อไป", "พากย์ไทยโดย", "ซับไทยโดย",
    "thank you", "thanks for watching", "you", "bye", ".", "..", "..."
}


async def transcribe_wav_bytes_hybrid(wav_bytes: bytes) -> str:
    whisper_candidate = ""
    low_confidence = False

    if GROQ_API_KEY:
        try:
            form = aiohttp.FormData()
            form.add_field("file", wav_bytes, filename="speech.wav", content_type="audio/wav")
            form.add_field("model", GROQ_STT_MODEL)
            form.add_field("language", "th")
            form.add_field("prompt", WHISPER_THAI_PROMPT_HINT)
            form.add_field("temperature", "0")
            form.add_field("response_format", "verbose_json")
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            stt_url = f"{HTTPS}{GROQ_API_HOST}/openai/v1/audio/transcriptions"
            session = get_http_session()
            async with session.post(
                stt_url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    segments = data.get("segments") or []
                    valid_parts = []
                    for seg in segments:
                        no_speech = float(seg.get("no_speech_prob", 0.0))
                        avg_logprob = float(seg.get("avg_logprob", 0.0))
                        comp_ratio = float(seg.get("compression_ratio", 1.0))
                        seg_text = (seg.get("text") or "").strip()

                        # Only drop if no_speech is very high OR logprob is extremely low (< -1.35)
                        if no_speech > 0.60 or avg_logprob < -1.35 or comp_ratio > 2.4:
                            print(
                                f"[Voice VAD Filter] Dropped noise segment '{seg_text}' "
                                f"(no_speech={no_speech:.2f}, logprob={avg_logprob:.2f}, comp={comp_ratio:.2f})",
                                flush=True,
                            )
                            continue
                        if avg_logprob < -0.55:
                            low_confidence = True
                        if seg_text:
                            valid_parts.append(seg_text)

                    txt = _apply_phonetic_fixes(" ".join(valid_parts).strip())
                    if txt and not NOISE_HALLUCINATION_REGEX.search(txt):
                        whisper_candidate = txt
        except Exception as e:
            print(f"Groq Whisper STT error (falling back to Google STT): {e}", flush=True)

    # If Whisper had low confidence on a short Thai phrase (or returned empty), cross-check with Google Thai STT
    if not whisper_candidate or low_confidence:
        google_txt = await asyncio.to_thread(transcribe_wav_bytes_google_sync, wav_bytes)
        if google_txt:
            if whisper_candidate and google_txt != whisper_candidate:
                print(
                    f"[Voice STT Cross-Check] Whisper='{whisper_candidate}' (low conf) -> Using Google STT='{google_txt}'",
                    flush=True,
                )
            return google_txt

    return whisper_candidate


def convert_audio_bytes_to_wav_sync(raw_bytes: bytes) -> bytes:
    """
    Converts raw 48kHz stereo Discord PCM/audio to 16kHz mono WAV with speech band-pass filtering
    and dynamic audio normalization (dynaudnorm) so quiet microphones are amplified cleanly.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-af", "highpass=f=85,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.92:m=10",
            "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1", "-loglevel", "error"
        ],
        input=raw_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return proc.stdout


async def handle_spoken_voice_command(
    guild: discord.Guild, text_channel, speaker_user, speaker_name: str, spoken_text: str
):
    cleaned = _apply_phonetic_fixes(spoken_text)
    if (
        len(cleaned) < 2
        or cleaned.lower() in WHISPER_HALLUCINATIONS
        or NOISE_HALLUCINATION_REGEX.search(cleaned)
    ):
        return
    print(f"[Voice STT] {speaker_name} spoke: '{cleaned}' (wake_word_required={voice_wake_word_enabled[guild.id]})", flush=True)

    # Instant Voice-Test Trigger (Always replies in chat even if wakeword is on)
    if VOICE_TEST_PHRASE_REGEX.search(cleaned):
        test_reply = (
            f"ฉันได้ยินเสียงพูดในห้องเสียงชัดเจนแล้วค่ะ (ข้อความที่ได้ยินคือ: **\"{cleaned}\"**) "
            "ไม่ต้องพูดย้ำหลายรอบหรอกนะคะ"
        )
        test_reply = sanitize_yukino_output(test_reply, cleaned, speaker_user, speaker_name)
        if text_channel:
            await text_channel.send(
                f"✅ **[🎙️ ทดสอบไมค์ — ได้ยินเสียงของ `{speaker_name}` ชัดเจนค่ะ]:** `{cleaned}`\n"
                f"**[🔊 Yukinon Voice]:** {test_reply}"
            )
        await speak_in_voice_channel(guild, test_reply, fallback_channel=text_channel)
        return

    lower_txt = cleaned.lower()
    if voice_wake_word_enabled[guild.id]:
        if not any(lower_txt.startswith(w) or w in lower_txt[:22] for w in VOICE_WAKE_WORDS):
            print(f"[Voice STT] Ignored '{cleaned}' (no wake word detected; use !wakeword off to disable).", flush=True)
            return

    answer, _ = await query_local_ollama_chat(
        f"{speaker_name} พูดในห้องเสียงว่า: {cleaned}",
        model=ACTIVE_FAST_MODEL,
        system_prompt=VOICE_SYSTEM_PROMPT,
        channel_id=text_channel.id if text_channel else None,
        temperature=0.55,
        allow_agentic_search=True,
        speaker_user=speaker_user,
        speaker_name=speaker_name,
    )
    if text_channel:
        await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n**[🔊 Yukinon Voice]:** {answer}")
    await speak_in_voice_channel(guild, answer, fallback_channel=text_channel)


class SilenceGatedVoiceSink(voice_recv.AudioSink if HAS_VOICE_RECV else object):
    # Lowered RMS threshold from 150 -> 55 to match normal speaking volume (rms=55..225)
    SPEECH_FRAME_RMS_THRESHOLD = 55
    MIN_ACTIVE_SPEECH_FRAMES = 8

    def __init__(self, guild: discord.Guild, text_channel, vc=None):
        super().__init__()
        self.guild = guild
        self.text_channel = text_channel
        self._vc_ref = vc
        self._lock = threading.Lock()
        # 400ms pre-roll buffer (20 frames x 20ms) so initial consonants are never clipped
        self.pre_roll_buffers = defaultdict(lambda: deque(maxlen=20))
        self.user_buffers = defaultdict(bytearray)
        self.speech_frame_counts = defaultdict(int)
        self.last_spoken = {}
        self._last_rms_log = 0.0
        self.running = True
        loop = state.bot.loop if state.bot else asyncio.get_event_loop()
        self.monitor_task = loop.create_task(self._silence_monitor_loop())

    @property
    def voice_client(self):
        return self._vc_ref or getattr(self, "_voice_client", None) or self.guild.voice_client

    def wants_opus(self) -> bool:
        return False

    def _resolve_speaker(self, user, data):
        if user is not None:
            return user
        vc = self.voice_client
        if not vc:
            return None
        pkt = getattr(data, "packet", None)
        ssrc = getattr(pkt, "ssrc", None)
        if ssrc and hasattr(vc, "_get_id_from_ssrc"):
            uid = vc._get_id_from_ssrc(ssrc)
            if uid:
                member = self.guild.get_member(int(uid))
                if member:
                    return member
        if getattr(vc, "channel", None):
            humans = [m for m in vc.channel.members if not m.bot]
            if len(humans) == 1:
                return humans[0]
        return None

    def write(self, user, data):
        if not hasattr(data, "pcm") or not data.pcm:
            return
        speaker = self._resolve_speaker(user, data)
        if not speaker or speaker.bot:
            return
        if self.guild.voice_client and self.guild.voice_client.is_playing():
            return
        try:
            frame_rms = audioop.rms(data.pcm, 2)
        except Exception:
            return

        now = time.monotonic()
        if frame_rms > 25 and (now - self._last_rms_log > 5.0):
            self._last_rms_log = now
            print(f"[Voice Mic Level] speaker={speaker.display_name}, rms={frame_rms}", flush=True)

        with self._lock:
            if frame_rms >= self.SPEECH_FRAME_RMS_THRESHOLD:
                # If starting a new utterance, prepend the 400ms pre-roll frames so the first syllable is intact
                if speaker not in self.user_buffers or len(self.user_buffers[speaker]) == 0:
                    for pre_frame in self.pre_roll_buffers[speaker]:
                        self.user_buffers[speaker].extend(pre_frame)
                    self.pre_roll_buffers[speaker].clear()
                self.user_buffers[speaker].extend(data.pcm)
                self.speech_frame_counts[speaker] += 1
                self.last_spoken[speaker] = now
            elif speaker in self.user_buffers and (now - self.last_spoken.get(speaker, 0) < 1.0):
                self.user_buffers[speaker].extend(data.pcm)
            else:
                self.pre_roll_buffers[speaker].append(data.pcm)

    def cleanup(self):
        self.running = False
        if getattr(self, "monitor_task", None) and not self.monitor_task.done():
            self.monitor_task.cancel()
        with self._lock:
            self.pre_roll_buffers.clear()
            self.user_buffers.clear()
            self.speech_frame_counts.clear()
            self.last_spoken.clear()

    async def _silence_monitor_loop(self):
        loop = state.bot.loop if state.bot else asyncio.get_running_loop()
        while self.running:
            await asyncio.sleep(0.35)
            now = time.monotonic()
            ready_payloads = []
            with self._lock:
                for user, last_t in list(self.last_spoken.items()):
                    if now - last_t >= 1.1:
                        pcm_bytes = bytes(self.user_buffers.pop(user, b""))
                        active_frames = self.speech_frame_counts.pop(user, 0)
                        self.last_spoken.pop(user, None)
                        if active_frames >= self.MIN_ACTIVE_SPEECH_FRAMES and len(pcm_bytes) >= 30720:
                            ready_payloads.append((user, pcm_bytes, active_frames))
            for user, pcm_bytes, active_frames in ready_payloads:
                print(
                    f"[Voice VAD] Captured speech from {user.display_name} "
                    f"({active_frames} active frames, {len(pcm_bytes)} bytes)",
                    flush=True,
                )
                loop.create_task(self._process_user_pcm(user, pcm_bytes))

    async def _process_user_pcm(self, user, pcm_bytes: bytes):
        try:
            raw_wav_io = io.BytesIO()
            with wave.open(raw_wav_io, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm_bytes)
            compact_wav = await asyncio.to_thread(convert_audio_bytes_to_wav_sync, raw_wav_io.getvalue())
            spoken_text = await transcribe_wav_bytes_hybrid(compact_wav)
            if spoken_text:
                await handle_spoken_voice_command(
                    self.guild, self.text_channel, user, user.display_name, spoken_text
                )
            else:
                print(f"[Voice STT] Audio captured from {user.display_name}, but STT returned empty text.", flush=True)
        except Exception as e:
            print(f"Voice PCM processing error: {e}", flush=True)