import os
import io
import re
import math
import json
import time
import wave
import array
import base64
import logging
import threading
import asyncio
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from urllib.parse import quote_plus

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

import aiohttp
import discord
import discord.gateway as _dpy_gateway
import discord.voice_state as _dpy_voice_state
import edge_tts
import speech_recognition as sr

from config import (
    HTTPS,
    CONTAINER_RAM_LIMIT_MB,
    TTS_PRIMARY_ENGINE,
    TTS_PARADEE_SPEED,
    TTS_PARADEE_PITCH,
    TTS_PARADEE_MODEL_DIR,
    TTS_PARADEE_IDLE_UNLOAD_SEC,
    TTS_PARADEE_RAM_GUARD_MB,
    TTS_VOICE_TH,
    TTS_VOICE_EN,
    TTS_PITCH,
    TTS_RATE,
    TTS_GOOGLE_PITCH,
    TTS_GOOGLE_SPEED,
    TTS_SILENCE_TRIM,
    VOICE_SCRIBE_MAX_ENTRIES,
    VOICE_SCRIBE_MAX_DURATION_SEC,
    STT_PRIMARY_LANG,
    DDGS_PROXY,
    GEMINI_API_KEY,
    GEMINI_TRANSCRIBE_LIVE_MODEL,
    GROQ_API_KEY,
    GROQ_STT_MODEL,
    VOICE_WAKE_WORDS,
    THAI_CHAR_REGEX,
    URL_REGEX,
    CAT_TOPIC_REGEX,
    STUTTER_REGEX,
    tts_lock,
    manga_semaphore,
    voice_stt_semaphore,
    voice_wake_word_enabled,
    voice_scribe_sessions,
    state,
    davey,
    clean_url,
    api_log,
    get_http_session,
    release_unused_memory,
)
from persona import (
    is_owner_user,
    sanitize_yukino_output,
    FAST_SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT,
)
from llm import (
    query_ai_chat,
    orchestrate_voice_utterance_with_decision_1,
    check_voice_scribe_stop_with_decision_1,
)

# Silence noisy RTCP SenderReport logs (type=200) and opus flush warnings from discord-ext-voice-recv
logging.getLogger("discord.ext.voice_recv").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.ERROR)

OPUS_SILENCE_FRAME = b"\xf8\xff\xfe"
PCM_FRAME_BYTES = 3840  # 20ms @ 48kHz stereo 16-bit PCM
PCM_SILENCE_20MS = b"\x00" * PCM_FRAME_BYTES

# Hard 18-second cap per continuous utterance (~3.45 MB @ 48kHz stereo 16-bit) to protect 512MB Render RAM
MAX_UTTERANCE_BYTES = 192000 * 18

# Mode-aware silence flush windows: snappier in interactive voice chat, patient in Voice Scribe meetings
INTERACTIVE_SILENCE_FLUSH_SEC = 0.85
INTERACTIVE_HANGOVER_SEC = 0.75
SCRIBE_SILENCE_FLUSH_SEC = 1.20
SCRIBE_HANGOVER_SEC = 1.00

# Host constants split to prevent chat/markdown auto-linkers from mangling URLs
GTRANS_HOST = "translate." + "googleapis.com"
GROQ_API_HOST = "api." + "groq.com"
GEMINI_WS_HOST = "generativelanguage." + "googleapis.com"

# ==============================================================================
# MAKHAMTTS PARADEE 1.1 INT8 (OFFLINE THAI NEURAL TTS — LAZY LOAD + RAM GUARD)
# ==============================================================================
YUKINO_PARADEE_CUSTOM_DICT: dict[str, str] = {
    "ยูกิโนะชิตะ": "ยู-กิ-โนะ-ชิ-ตะ",
    "ยูกิโนะ": "ยู-กิ-โนะ",
    "ยูกิน่อน": "ยู-กิ-น่อน",
    "แพนซัง": "แพน-ซัง",
    "ยูกิพีเดีย": "ยู-กิ-พี-เดีย",
    "Yukipedia": "ยู-กิ-พี-เดีย",
    "Yukino": "ยู-กิ-โนะ",
    "Yukinon": "ยู-กิ-น่อน",
    "OT": "โอ-ที",
    "e-Tax": "อี-แท็กซ์",
    "PDF": "พี-ดี-เอฟ",
    "API": "เอ-พี-ไอ",
    "SQL": "เอส-คิว-แอล",
    "VPN": "วี-พี-เอ็น",
    "UniFi": "ยู-นิ-ไฟ",
    "Kiro": "คิ-โระ",
    "Docker": "ด็อก-เกอร์",
    "Discord": "ดิส-คอร์ด",
    "PSO2": "พี-เอส-โอ-ทู",
    "NGS": "เอ็น-จี-เอส",
}

_PARADEE_LOCK = threading.Lock()
_PARADEE_TTS_MODEL = None
_PARADEE_LAST_USED_TS: float = 0.0
_PARADEE_IDLE_TASK: asyncio.Task | None = None

# Active voice sink registry (keyed by guild_id) so Scribe Stop can flush mid-sentence buffers cleanly
_ACTIVE_VOICE_SINKS: dict[int, "SilenceGatedVoiceSink"] = {}

# Phase 3: Rolling per-guild multi-speaker voice room history (last 8 turns) & last bot reply timestamp
_VOICE_ROOM_HISTORY: dict[int, deque] = defaultdict(lambda: deque(maxlen=8))
_VOICE_LAST_BOT_REPLY_TS: dict[int, float] = {}


def _build_decim_lp_kernel(num_taps: int = 21, fc_hz: float = 6800.0, fs_hz: float = 48000.0):
    """Precomputes a 21-tap Blackman-windowed sinc anti-aliasing low-pass FIR kernel for 48kHz -> 16kHz decimation."""
    if not _HAS_NUMPY:
        return None
    n = np.arange(num_taps, dtype=np.float32) - (num_taps - 1) / 2.0
    fc = fc_hz / fs_hz
    h = np.sinc(2.0 * fc * n).astype(np.float32)
    win = np.blackman(num_taps).astype(np.float32)
    h *= win
    h /= np.sum(h)
    return h


_DECIM_LP_KERNEL = _build_decim_lp_kernel()


def _calc_pcm16_rms(pcm_bytes: bytes) -> float:
    """
    Python 3.13+ safe 16-bit signed PCM RMS calculator (replaces removed stdlib `audioop.rms`).
    Casts int16 samples to float64 before squaring so int16 overflow never occurs.
    """
    n_bytes = len(pcm_bytes)
    if n_bytes < 2:
        return 0.0
    if n_bytes & 1:
        pcm_bytes = pcm_bytes[:-1]
    if _HAS_NUMPY:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))
    arr = array.array("h")
    arr.frombytes(pcm_bytes)
    if not arr:
        return 0.0
    return math.sqrt(sum(x * x for x in arr) / float(len(arr)))


# Spoken test phrases (including common STT phonetic variations like ชัด/แชท, ตัดสอบ/ทดสอบ, หยิน/ยิน)
VOICE_TEST_PHRASE_REGEX = re.compile(
    r"(ได้ยินไหม|ได้ยินมั้ย|ได้ยินหรือเปล่า|ได้ยินฉันไหม|ไม่หยินไหม|ได้มิดิน|"
    r"ถ้าได้ยิน|ถ้าได้อีกแล้ว|พิมพ์ตอบในแชท|พิมพ์ตอบในชัด|ตอบในแชท|ตอบในชัด|"
    r"เทสเสียง|ทดสอบเสียง|ทดสอบไมค์|ตัดสอบ|ฮัลโหลเทส|"
    r"can\s+you\s+hear\s+me|if\s+you\s+hear\s+me|reply\s+in\s+chat|mic\s+test|testing\s+1\s*2\s*3)",
    re.IGNORECASE,
)

# Passive Voice Scribe Start & Stop triggers (avoids colliding with "จด OT" or casual speech)
VOICE_SCRIBE_START_REGEX = re.compile(
    r"(เริ่มจดบันทึก|เริ่มจดการประชุม|เริ่มจดประชุม|เริ่มจดสรุป|เริ่มบันทึกการประชุม|เริ่มบันทึกเสียง|"
    r"(?:ช่วย)?จดการประชุม|(?:ช่วย)?จดประชุม|(?:ช่วย)?จดสรุปการประชุม|(?:ช่วย)?จดบันทึกการประชุม|"
    r"(?:ช่วย)?ฟังแล้ว(?:จด|สรุป)|โหมดจดประชุม|โหมดจดบันทึก|"
    r"\b(?:start\s+scribe|start\s+recording|take\s+meeting\s+notes|record\s+meeting)\b)",
    re.IGNORECASE,
)
VOICE_SCRIBE_STOP_UNAMBIGUOUS_REGEX = re.compile(
    r"(หยุดจด|หยุดบันทึก|หยุดอัด|เลิกจด|จบการประชุม|จบประชุม|"
    r"สรุปการประชุม|สรุปประชุม|สรุปที่คุยกัน|สรุปบทสนทนา|สรุปทั้งหมดให้หน่อย|หยุดฟังแล้วสรุป|"
    r"\b(?:stop\s+scribe|stop\s+recording|end\s+meeting|summarize\s+meeting)\b)",
    re.IGNORECASE,
)
VOICE_SCRIBE_STOP_WAKE_SHORT_REGEX = re.compile(
    r"(?:สรุป(?:เลย|ให้หน่อย|ได้แล้ว|หน่อย)?|หยุด(?:เลย|ได้แล้ว)?|พอแล้ว|จบแล้ว|"
    r"\b(?:stop|summarize|summary|finish|done)\b)",
    re.IGNORECASE,
)
# Phase 3: Soft stop/bot cue gate during Scribe Mode — only invokes Decision-1 ScribeStopCheck when present (0 overhead on normal meeting speech)
_VOICE_SCRIBE_SOFT_STOP_CUE_REGEX = re.compile(
    r"(หยุด|สรุป|พอแล้ว|พอแค่นี้|จบแล้ว|เสร็จแล้ว|เลิก|ปิดโหมด|ยูกิ|ยูคิ|บอท|"
    r"\b(?:yukino|yukinon|yuki|bot|stop|summar\w*|wrap\s+up|finish\w*|done)\b)",
    re.IGNORECASE,
)

# Custom vocabulary terms for Gemini 3.5 Transcribe Live (up to 100 terms for optimal biasing)
GEMINI_LIVE_CUSTOM_VOCAB = [
    "ยูกิโนะ",
    "ยูกิน่อน",
    "Yukino",
    "Yukinon",
    "จด OT",
    "สรุป OT",
    "ลบ OT",
    "แก้ OT",
    "เริ่มจดบันทึก",
    "จดประชุม",
    "หยุดจด",
    "หยุดบันทึก",
    "สรุปการประชุม",
    "สรุปประชุม",
    "ดูความจำ",
    "จำไว้ว่า",
    "ยูกิพีเดีย",
    "Yukipedia",
    "แปลมังงะ",
    "PSO2",
    "NGS",
    "e-Tax",
    "Spring Boot",
    "Oracle",
    "Kubernetes",
    "Docker",
    "ทุ่มนึง",
    "สองทุ่ม",
    "สามทุ่ม",
    "หกโมง",
]

# Vocabulary hint for Groq Whisper fallback
WHISPER_THAI_PROMPT_HINT = (
    "ยูกิโนะ ยูกิน่อน Yukino Yukinon สวัสดีค่ะ แนะนำตัวหน่อย ได้ยินไหม ถ้าได้ยินแล้วพิมพ์ตอบในแชท "
    "ทดสอบไมค์ เทสเสียง ทดสอบ วันนี้วันอะไร ตอนนี้เวลากี่โมง อากาศวันนี้เป็นยังไง "
    "เริ่มจดบันทึก จดประชุม หยุดจด สรุปการประชุม จด OT 6 โมงถึงทุ่มนึง สรุป OT ลบ OT ดูความจำ ยูกิพีเดีย Yukipedia"
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
    (re.compile(r"(?:จด|กด|จอด)\s*(?:โอ\s*ที|ot)\b", re.IGNORECASE), "จด OT"),
    (re.compile(r"สรุป\s*(?:โอ\s*ที|ot)\b", re.IGNORECASE), "สรุป OT"),
    (re.compile(r"(?:ลบ|ยกเลิก)\s*(?:โอ\s*ที|ot)\b", re.IGNORECASE), "ลบ OT"),
    (re.compile(r"(?:แก้|แก้ไข)\s*(?:โอ\s*ที|ot)\b", re.IGNORECASE), "แก้ OT"),
)

_NATIVE_DPY_VOICE_WS_RECEIVED = _dpy_gateway.DiscordVoiceWebSocket.received_message
_NATIVE_DPY_VOICE_WS_BINARY = getattr(_dpy_gateway.DiscordVoiceWebSocket, "received_binary_message", None)
_ORIG_DPY_EXECUTE_TRANSITION = getattr(_dpy_voice_state.VoiceConnectionState, "_execute_transition", None)
_KNOWN_VOICE_USER_IDS: set[int] = set()

try:
    from discord.ext import voice_recv
    import discord.ext.voice_recv.opus as _vr_opus
    import discord.ext.voice_recv.gateway as _vr_gateway
    import discord.ext.voice_recv.voice_client as _vr_vc

    _ORIG_VR_HOOK = getattr(_vr_gateway, "hook", None)
    _ORIG_DECODE_PACKET = _vr_opus.PacketDecoder._decode_packet
    _DAVE_LOG_TS = 0.0

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


def has_baked_paradee_model() -> bool:
    """True when both pre-baked INT8 ONNX model and vocab config exist locally on disk."""
    if not TTS_PARADEE_MODEL_DIR:
        return False
    m_path = os.path.join(TTS_PARADEE_MODEL_DIR, "paradee_thai_int8.onnx")
    c_path = os.path.join(TTS_PARADEE_MODEL_DIR, "config.json")
    return os.path.isfile(m_path) and os.path.isfile(c_path)


def _get_current_rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
    except Exception:
        return 0.0


def _can_safely_load_paradee_now() -> tuple[bool, str]:
    """
    OOM guardrail for 512 MB Render containers:
    - Always allows synthesis if Paradee is already resident in RAM.
    - When unloaded on <=512 MB containers, skips loading during active manga rendering
      (`manga_semaphore.locked()`) or when RSS already exceeds `TTS_PARADEE_RAM_GUARD_MB`.
    """
    if _PARADEE_TTS_MODEL is not None:
        return True, ""
    if time.time() < state.paradee_tts_cooldown_until:
        return False, "cooldown"
    if CONTAINER_RAM_LIMIT_MB <= 512:
        if manga_semaphore.locked():
            return False, "manga_active_oom_guard"
        rss_mb = _get_current_rss_mb()
        if rss_mb >= float(TTS_PARADEE_RAM_GUARD_MB):
            return False, f"high_rss_{rss_mb:.0f}MB"
    return True, ""


def _load_paradee_model_sync():
    """
    Lazily instantiates PyThaiTTS `MakHamTTSParadee` (7.88M distilled INT8 ONNX, 8.07 MB):
    - Uses pre-baked `/app/models/paradee_thai` files when present (0 Hub network checks!).
    - Caps `intra_op_threads=1` on <=512 MB containers to keep ONNX arena RAM lean.
    """
    global _PARADEE_TTS_MODEL, _PARADEE_LAST_USED_TS
    with _PARADEE_LOCK:
        if _PARADEE_TTS_MODEL is not None:
            _PARADEE_LAST_USED_TS = time.time()
            return _PARADEE_TTS_MODEL

        t0 = time.monotonic()
        init_kwargs: dict = {
            "speed": max(0.75, min(1.50, float(TTS_PARADEE_SPEED))),
            "intra_op_threads": 1 if CONTAINER_RAM_LIMIT_MB <= 512 else 0,
            "custom_dict": dict(YUKINO_PARADEE_CUSTOM_DICT),
        }
        source_tag = "hf_hub"
        if has_baked_paradee_model():
            init_kwargs["model_path"] = os.path.join(TTS_PARADEE_MODEL_DIR, "paradee_thai_int8.onnx")
            init_kwargs["config_path"] = os.path.join(TTS_PARADEE_MODEL_DIR, "config.json")
            source_tag = "local_baked"

        try:
            from pythaitts.pretrained.makham import MakHamTTSParadee
            model = MakHamTTSParadee(**init_kwargs)
        except ImportError:
            from pythaitts import TTS
            model = TTS(pretrained="makham_paradee", **init_kwargs)

        _PARADEE_TTS_MODEL = model
        _PARADEE_LAST_USED_TS = time.time()
        state.paradee_tts_loaded = True
        rss_after = _get_current_rss_mb()
        api_log(
            f"Paradee TTS loaded ({source_tag}, threads={init_kwargs['intra_op_threads']}, "
            f"{time.monotonic() - t0:.2f}s, RSS={rss_after:.1f} MB)"
        )
        return _PARADEE_TTS_MODEL


def unload_paradee_tts(reason: str = "") -> bool:
    """
    Explicitly releases the in-RAM `MakHamTTSParadee` ONNX session & G2P dictionaries and
    triggers glibc `malloc_trim(0)` via `release_unused_memory()` so Render 512 MB RAM stays clean.
    """
    global _PARADEE_TTS_MODEL, _PARADEE_IDLE_TASK
    if _PARADEE_IDLE_TASK is not None and not _PARADEE_IDLE_TASK.done():
        try:
            if asyncio.current_task() is not _PARADEE_IDLE_TASK:
                _PARADEE_IDLE_TASK.cancel()
        except RuntimeError:
            _PARADEE_IDLE_TASK.cancel()
    _PARADEE_IDLE_TASK = None

    dropped = False
    with _PARADEE_LOCK:
        if _PARADEE_TTS_MODEL is not None:
            _PARADEE_TTS_MODEL = None
            dropped = True
    state.paradee_tts_loaded = False
    if dropped:
        before_mb, after_mb = release_unused_memory()
        api_log(
            f"Paradee TTS unloaded ({reason or 'cleanup'}) -> RAM {before_mb} MB -> {after_mb} MB"
        )
    return dropped


def _has_any_active_voice_client() -> bool:
    bot = state.bot
    if not bot or not getattr(bot, "voice_clients", None):
        return False
    return any(vc and vc.is_connected() for vc in bot.voice_clients)


def _schedule_paradee_idle_unload() -> None:
    """Auto-unloads Paradee after `TTS_PARADEE_IDLE_UNLOAD_SEC` when no voice channels are active."""
    global _PARADEE_IDLE_TASK
    if _PARADEE_TTS_MODEL is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _PARADEE_IDLE_TASK is not None and not _PARADEE_IDLE_TASK.done():
        return

    async def _idle_watcher():
        global _PARADEE_IDLE_TASK
        try:
            while _PARADEE_TTS_MODEL is not None:
                await asyncio.sleep(30.0)
                if _has_any_active_voice_client():
                    continue
                idle_for = time.time() - _PARADEE_LAST_USED_TS
                if idle_for >= float(TTS_PARADEE_IDLE_UNLOAD_SEC):
                    unload_paradee_tts(reason=f"idle {int(idle_for)}s")
                    return
        except asyncio.CancelledError:
            return
        finally:
            _PARADEE_IDLE_TASK = None

    _PARADEE_IDLE_TASK = loop.create_task(_idle_watcher())


def prewarm_paradee_tts_bg() -> None:
    """
    Non-blocking background pre-warm invoked on `!join` (during the 1.5s DAVE handshake)
    when `TTS_PRIMARY_ENGINE == "paradee"`, so the greeting plays immediately on Paradee.
    """
    if TTS_PRIMARY_ENGINE != "paradee" or _PARADEE_TTS_MODEL is not None:
        return
    ok, _ = _can_safely_load_paradee_now()
    if not ok:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _prewarm():
        try:
            await asyncio.to_thread(_load_paradee_model_sync)
        except ImportError:
            state.paradee_tts_cooldown_until = time.time() + 3600.0
            api_log("Paradee TTS prewarm skipped: pythaitts not installed (cooldown 1h)")
        except Exception as e:
            state.paradee_tts_cooldown_until = time.time() + 300.0
            api_log(f"Paradee TTS prewarm warning ({type(e).__name__}): {str(e)[:120]}")

    loop.create_task(_prewarm())


def _waveform_to_wav_bytes_sync(raw_out, default_sr: int = 24000) -> bytes:
    """
    Normalizes any PyThaiTTS output (`bytes`, `(waveform, sr)` tuple, or `np.ndarray`)
    into a valid 16-bit mono PCM WAV buffer in RAM (zero temp files).
    """
    if raw_out is None:
        return b""
    if isinstance(raw_out, (bytes, bytearray)):
        raw_b = bytes(raw_out)
        if raw_b[:4] == b"RIFF":
            return raw_b
        # Treat bare bytes as 16-bit mono PCM @ default_sr
        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(default_sr)
            wf.writeframes(raw_b)
        return out_io.getvalue()

    sr = default_sr
    wav_arr = raw_out
    if isinstance(raw_out, (tuple, list)) and len(raw_out) >= 2:
        # Handle either (waveform, sample_rate) or (sample_rate, waveform)
        if isinstance(raw_out[1], (int, float)) and int(raw_out[1]) >= 8000:
            wav_arr, sr = raw_out[0], int(raw_out[1])
        elif isinstance(raw_out[0], (int, float)) and int(raw_out[0]) >= 8000:
            sr, wav_arr = int(raw_out[0]), raw_out[1]

    if not _HAS_NUMPY:
        return b""

    arr = np.asarray(wav_arr)
    if arr.size == 0:
        return b""
    if arr.ndim > 1:
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr[0]

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm16 = np.clip(np.rint(arr * 32767.0), -32768.0, 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        pcm16 = np.clip(arr, -32768, 32767).astype(np.int16)
    else:
        pcm16 = arr

    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    return out_io.getvalue()


def _synthesize_paradee_wav_sync(spoken_text: str) -> bytes:
    """Thread-safe offline Thai synthesis via `MakHamTTSParadee` returning 24 kHz WAV bytes."""
    global _PARADEE_LAST_USED_TS
    model = _load_paradee_model_sync()
    speed = max(0.75, min(1.50, float(TTS_PARADEE_SPEED)))
    with _PARADEE_LOCK:
        _PARADEE_LAST_USED_TS = time.time()
        if hasattr(model, "tts") and callable(getattr(model, "tts")):
            try:
                raw_out = model.tts(spoken_text, return_type="waveform", speed=speed)
            except TypeError:
                raw_out = model.tts(spoken_text, return_type="waveform")
        else:
            try:
                raw_out = model(text=spoken_text, return_type="waveform", speed=speed)
            except TypeError:
                raw_out = model(spoken_text, return_type="waveform")
    default_sr = int(getattr(model, "sample_rate", None) or getattr(model, "SAMPLE_RATE", None) or 24000)
    return _waveform_to_wav_bytes_sync(raw_out, default_sr=default_sr)


async def _paradee_tts_with_mastering(
    spoken_text: str, output_format: str = "pcm"
) -> bytes:
    """
    Synthesizes Thai / Thai-English speech offline using `MakHamTTSParadee` (8.07 MB INT8 ONNX):
    - Native speed via `TTS_PARADEE_SPEED` (default `1.10`).
    - Subtle cool-persona pitch shift via `TTS_PARADEE_PITCH` (default `0.96`, tempo-compensated
      so final playback speed remains exact) + gentle `lowpass=f=9500` to smooth any 8M INT8 buzz.
    - Single FFmpeg mastering pass directly to 48kHz stereo PCM (or MP3 for chat fallback).
    """
    ok, skip_reason = _can_safely_load_paradee_now()
    if not ok:
        if skip_reason and skip_reason != "cooldown":
            api_log(f"Paradee TTS bypassed ({skip_reason}); falling back to Google TTS for this turn")
        return b""

    t0 = time.monotonic()
    try:
        raw_wav = await asyncio.to_thread(_synthesize_paradee_wav_sync, spoken_text)
    except ImportError:
        state.paradee_tts_cooldown_until = time.time() + 3600.0
        api_log("Paradee TTS unavailable (pythaitts not installed); cooling down 1h")
        return b""
    except Exception as e:
        state.paradee_tts_cooldown_until = time.time() + 300.0
        api_log(f"Paradee TTS synthesis error ({type(e).__name__}: {str(e)[:120]}); cooling down 5m")
        return b""

    if not raw_wav or len(raw_wav) <= 44:
        return b""

    pitch = max(0.85, min(1.15, float(TTS_PARADEE_PITCH)))
    af_parts: list[str] = []
    if abs(pitch - 1.0) >= 0.005:
        comp_atempo = max(0.50, min(2.00, 1.0 / pitch))
        af_parts.append(f"asetrate=24000*{pitch:.4f},aresample=48000,atempo={comp_atempo:.4f}")
    af_parts.append("lowpass=f=9500")
    paradee_af = ",".join(af_parts)

    _schedule_paradee_idle_unload()

    try:
        mastered = await asyncio.to_thread(
            _master_tts_audio_sync,
            raw_wav,
            paradee_af,
            True,
            output_format,
        )
        if mastered:
            state.active_tts_engine = "paradee"
            api_log(
                f"Paradee TTS -> OK (speed={TTS_PARADEE_SPEED:.2f}x, pitch={pitch:.2f}, "
                f"{len(mastered)}B {output_format}, {time.monotonic() - t0:.2f}s)"
            )
            return mastered
    except Exception as e:
        api_log(f"Paradee TTS mastering fallback ({type(e).__name__}): {str(e)[:80]}")
        try:
            if output_format == "pcm":
                return await asyncio.to_thread(_decode_audio_to_pcm48k_sync, raw_wav)
        except Exception:
            pass
    return b""


def register_voice_sink(guild_id: int, sink: "SilenceGatedVoiceSink") -> None:
    _ACTIVE_VOICE_SINKS[int(guild_id)] = sink


def unregister_voice_sink(guild_id: int) -> None:
    gid = int(guild_id)
    _ACTIVE_VOICE_SINKS.pop(gid, None)
    _VOICE_ROOM_HISTORY.pop(gid, None)
    _VOICE_LAST_BOT_REPLY_TS.pop(gid, None)
    if not _ACTIVE_VOICE_SINKS and not _has_any_active_voice_client():
        unload_paradee_tts(reason="voice disconnected")


def _record_voice_room_turn(guild_id: int | None, speaker_label: str, text: str, is_bot: bool = False) -> None:
    if not guild_id or not text:
        return
    gid = int(guild_id)
    now = time.time()
    clean_t = re.sub(r"\s+", " ", text).strip()[:240]
    if not clean_t:
        return
    _VOICE_ROOM_HISTORY[gid].append({
        "ts": now,
        "speaker": "Yukinon" if is_bot else (speaker_label or "Member")[:32],
        "is_bot": is_bot,
        "text": clean_t,
    })
    if is_bot:
        _VOICE_LAST_BOT_REPLY_TS[gid] = now


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
        hour_phrase = f"{THAI_DAY_NUMBER_WORDS[hh - 18]}ทุ่ม"

    if mm == 0:
        if hh in (0, 12):
            return f"{hour_phrase}ตรง"
        return hour_phrase
    if mm == 30:
        return f"{hour_phrase}ครึ่ง"
    return f"{hour_phrase} {mm} นาที"


def _normalize_thai_dates_and_times_for_tts(text: str) -> str:
    s = re.sub(r"\(\s*ค\.ศ\.[^)]*\)", " ", text)
    s = re.sub(r"\(\s*B\.E\.[^)]*\)", " ", s)
    s = re.sub(r"\(\s*เวลาประเทศไทย[^)]*\)", " ", s)

    def _time_repl(m):
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return _format_colloquial_thai_time(hh, mm)
        return m.group(0)

    # Convert both HH:MM and HH.MM (e.g., 18.00 - 19.00 น.) into colloquial Thai speech without extra pause spaces
    s = re.sub(r"\b(\d{1,2})[:.](\d{2})\s*(?:น\.)?", _time_repl, s)

    def _day_num_repl(m):
        prefix = m.group(1)
        day_int = int(m.group(2))
        day_word = THAI_DAY_NUMBER_WORDS.get(day_int, str(day_int))
        return f"{prefix}{day_word}"

    s = re.sub(r"(ที่)\s*(\d{1,2})\b", _day_num_repl, s)
    s = re.sub(r"(?:พ\.ศ\.|พุทธศักราช)\s*(\d{4})", r"\1", s)
    return s


def clean_text_for_speech(text: str) -> str:
    # Strip hidden OT/Memory record footers before speaking
    cleaned = re.sub(r"-#\s*\[(?:OT_RECORD|MEM_RECORD):[^\]]*\]", " ", text or "")
    cleaned = sanitize_yukino_output(cleaned)
    cleaned = re.sub(r"```[\s\S]*?```", " รายละเอียดของโค้ดฉันส่งไว้ให้ในแชทแล้วค่ะ ", cleaned)
    cleaned = URL_REGEX.sub(" ลิงก์อ้างอิงอยู่ในแชทแล้วค่ะ ", cleaned)
    cleaned = _normalize_thai_dates_and_times_for_tts(cleaned)
    # Remove Markdown formatting symbols without injecting stutter-causing commas between Thai words
    cleaned = re.sub(r"[\(\)\*\_\~\`\#\>\|\"'\[\]]", "", cleaned)
    cleaned = re.sub(r"(?<=[\u0E00-\u0E7F])\s*[,،]\s*(?=[\u0E00-\u0E7F])", " ", cleaned)
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()[:450]


def select_tts_voice(text: str) -> str:
    return TTS_VOICE_TH if THAI_CHAR_REGEX.search(text) else TTS_VOICE_EN


def _split_tts_chunks(text: str, max_len: int = 195) -> list[str]:
    """Splits text along natural sentence/clause endings first so Google TTS never stalls mid-phrase."""
    if len(text) <= max_len:
        return [text]

    raw_clauses = re.split(r"(?<=[.!?])\s+|(?<=ค่ะ)\s+|(?<=นะคะ)\s+|\n+", text)
    chunks: list[str] = []
    cur = ""
    for clause in raw_clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(cur) + len(clause) + (1 if cur else 0) <= max_len:
            cur = f"{cur} {clause}".strip() if cur else clause
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if len(clause) <= max_len:
            cur = clause
        else:
            for w in clause.split(" "):
                if len(cur) + len(w) + (1 if cur else 0) <= max_len:
                    cur = f"{cur} {w}".strip() if cur else w
                else:
                    if cur:
                        chunks.append(cur)
                    cur = w[:max_len]
    if cur:
        chunks.append(cur)
    return chunks


def _pad_pcm_to_opus_frame(pcm_bytes: bytes) -> bytes:
    """Zero-pads raw 48kHz stereo 16-bit PCM to a 3840-byte (20ms) frame boundary so PCMAudio never drops the tail."""
    if not pcm_bytes:
        return b""
    rem = len(pcm_bytes) % PCM_FRAME_BYTES
    if rem:
        return pcm_bytes + (b"\x00" * (PCM_FRAME_BYTES - rem))
    return pcm_bytes


def _decode_audio_to_pcm48k_sync(audio_bytes: bytes) -> bytes:
    """Plain fallback decode to raw 48kHz stereo 16-bit PCM if Yukino mastering filter fails."""
    proc = subprocess.run(
        [
            "ffmpeg", "-threads", "1", "-i", "pipe:0",
            "-ac", "2", "-ar", "48000",
            "-f", "s16le", "pipe:1", "-loglevel", "error"
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return _pad_pcm_to_opus_frame(proc.stdout)


def _master_tts_audio_sync(
    audio_bytes: bytes,
    extra_af: str = "",
    trim_internal_pauses: bool = True,
    output_format: str = "pcm",
) -> bytes:
    """
    Single-pass FFmpeg audio mastering chain tuned for Yukino's calm, cool kuudere voice:
    - `silenceremove`: tightens dead gaps >160ms (including chunk seams) down to a natural 70ms breath
    - `adelay=220:all=1`: prevents Discord Opus from clipping opening syllables
    - Subtle warmth (+1.5dB @ 240Hz) + de-harshing (-3.0dB @ 3.2kHz) + gentle compression
    - `output_format="pcm"` emits raw 48kHz stereo `s16le` directly for `discord.PCMAudio` (1 ffmpeg spawn total, zero disk I/O).
    - `output_format="mp3"` emits 128kbps MP3 for text-channel fallback file attachments.
    """
    stages = []
    if extra_af:
        stages.append(extra_af)
    if trim_internal_pauses and TTS_SILENCE_TRIM:
        stages.append(
            "silenceremove=stop_periods=-1:stop_duration=0.16:stop_threshold=-40dB:stop_silence=0.07"
        )
    stages.extend([
        "adelay=220:all=1",
        "equalizer=f=240:t=q:w=1.1:g=1.5",
        "equalizer=f=3200:t=q:w=1.2:g=-3.0",
        "acompressor=threshold=-18dB:ratio=2:attack=15:release=150",
        "volume=1.35",
    ])
    full_af = ",".join(stages)
    if output_format == "mp3":
        out_args = ["-ac", "2", "-ar", "48000", "-b:a", "128k", "-write_xing", "0", "-f", "mp3", "pipe:1"]
    else:
        out_args = ["-ac", "2", "-ar", "48000", "-f", "s16le", "pipe:1"]

    proc = subprocess.run(
        ["ffmpeg", "-threads", "1", "-i", "pipe:0", "-af", full_af, *out_args, "-loglevel", "error"],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return _pad_pcm_to_opus_frame(proc.stdout) if output_format != "mp3" else proc.stdout


async def _google_tts_with_yukino_filter(
    spoken_text: str, lang_code: str, output_format: str = "pcm"
) -> bytes:
    chunks = _split_tts_chunks(spoken_text, max_len=195)
    headers = {"User-Agent": "Mozilla/5.0"}
    session = get_http_session()

    async def _fetch_chunk(chunk_str: str) -> bytes:
        tts_url = (
            f"{HTTPS}{GTRANS_HOST}/translate_tts"
            f"?ie=UTF-8&client=tw-ob&tl={lang_code}&q={quote_plus(chunk_str)}"
        )
        try:
            async with session.get(
                tts_url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception as e:
            api_log(f"Google TTS chunk warning: {type(e).__name__}: {str(e)[:80]}")
        return b""

    if len(chunks) == 1:
        part_list = [await _fetch_chunk(chunks[0])]
    else:
        part_list = list(await asyncio.gather(*(_fetch_chunk(c) for c in chunks)))

    raw_mp3_parts = bytearray()
    for part in part_list:
        if part:
            raw_mp3_parts.extend(part)

    if not raw_mp3_parts:
        return b""

    raw_mp3 = bytes(raw_mp3_parts)
    # Cooler Yukino pitch (default 0.97 vs old 1.04) & slightly brisker smooth speed (default 1.24x)
    pitch = max(0.85, min(1.15, float(TTS_GOOGLE_PITCH)))
    speed = max(0.85, min(1.60, float(TTS_GOOGLE_SPEED)))
    atempo = max(0.50, min(2.00, speed / pitch))
    steady_af = f"asetrate=24000*{pitch:.4f},aresample=48000,atempo={atempo:.4f}"
    try:
        out = await asyncio.to_thread(
            _master_tts_audio_sync,
            raw_mp3,
            steady_af,
            True,
            output_format,
        )
        if out:
            state.active_tts_engine = "google"
        return out
    except Exception:
        if output_format == "mp3":
            state.active_tts_engine = "google"
            return raw_mp3
        try:
            out = await asyncio.to_thread(_decode_audio_to_pcm48k_sync, raw_mp3)
            if out:
                state.active_tts_engine = "google"
            return out
        except Exception:
            return b""


async def _edge_tts_with_mastering(
    spoken_text: str, is_thai: bool, output_format: str = "pcm"
) -> bytes:
    if time.time() < state.edge_tts_cooldown_until:
        return b""
    voice_name = TTS_VOICE_TH if is_thai else TTS_VOICE_EN
    try:
        comm_kwargs = {"text": spoken_text, "voice": voice_name, "proxy": DDGS_PROXY}
        if TTS_RATE and TTS_RATE not in ("+0%", "-0%", "0%"):
            comm_kwargs["rate"] = TTS_RATE
        if TTS_PITCH and TTS_PITCH not in ("+0Hz", "-0Hz", "0Hz"):
            comm_kwargs["pitch"] = TTS_PITCH

        communicate = edge_tts.Communicate(**comm_kwargs)

        async def _collect_stream() -> bytes:
            buf = bytearray()
            async for chunk in communicate.stream():
                if isinstance(chunk, dict) and chunk.get("type") == "audio" and chunk.get("data"):
                    buf.extend(chunk["data"])
            return bytes(buf)

        raw_edge_mp3 = await asyncio.wait_for(_collect_stream(), timeout=4.5)
        if len(raw_edge_mp3) > 256:
            try:
                out = await asyncio.to_thread(_master_tts_audio_sync, raw_edge_mp3, "", True, output_format)
                if out:
                    state.active_tts_engine = "edge"
                return out
            except Exception:
                if output_format == "mp3":
                    state.active_tts_engine = "edge"
                    return raw_edge_mp3
                try:
                    out = await asyncio.to_thread(_decode_audio_to_pcm48k_sync, raw_edge_mp3)
                    if out:
                        state.active_tts_engine = "edge"
                    return out
                except Exception:
                    return b""
    except Exception as e:
        state.edge_tts_cooldown_until = time.time() + 1800
        print(f"edge-tts unavailable ({e}); cooling down for 30m...", flush=True)
    return b""


async def synthesize_tts_audio_bytes(text: str, output_format: str = "pcm") -> bytes:
    """
    Consistent in-memory hybrid voice synthesis (zero temp files):
    - Default (`TTS_PRIMARY_ENGINE=paradee`):
      * Thai / Thai-English mixed: Offline `MakHamTTSParadee` 1.1 INT8 (speed=1.10, pitch=0.96)
        -> Google TTS -> Edge-TTS fallback.
      * Pure English: Google TTS (`tl=en`) -> Edge-TTS (`en-US-AriaNeural`).
    - Optional (`TTS_PRIMARY_ENGINE=edge`): Edge-TTS -> Google TTS -> Paradee (Thai).
    - Optional (`TTS_PRIMARY_ENGINE=google`): Google TTS -> Edge-TTS.
    - `output_format="pcm"` returns 48kHz stereo 16-bit PCM ready for `discord.PCMAudio`.
    - `output_format="mp3"` returns 128kbps MP3 for Discord text-channel attachments.
    """
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return b""
    is_thai = bool(THAI_CHAR_REGEX.search(spoken_text))
    lang_code = "th" if is_thai else "en"

    if TTS_PRIMARY_ENGINE == "paradee":
        if is_thai:
            paradee_bytes = await _paradee_tts_with_mastering(spoken_text, output_format=output_format)
            if paradee_bytes:
                return paradee_bytes
        google_bytes = await _google_tts_with_yukino_filter(spoken_text, lang_code, output_format=output_format)
        if google_bytes:
            return google_bytes
        return await _edge_tts_with_mastering(spoken_text, is_thai, output_format=output_format)

    if TTS_PRIMARY_ENGINE == "edge":
        edge_bytes = await _edge_tts_with_mastering(spoken_text, is_thai, output_format=output_format)
        if edge_bytes:
            return edge_bytes
        google_bytes = await _google_tts_with_yukino_filter(spoken_text, lang_code, output_format=output_format)
        if google_bytes:
            return google_bytes
        if is_thai:
            return await _paradee_tts_with_mastering(spoken_text, output_format=output_format)
        return b""

    google_bytes = await _google_tts_with_yukino_filter(spoken_text, lang_code, output_format=output_format)
    if google_bytes:
        return google_bytes
    return await _edge_tts_with_mastering(spoken_text, is_thai, output_format=output_format)


async def synthesize_tts_mp3_bytes(text: str) -> bytes:
    """Backward-compatible helper returning mastered MP3 bytes."""
    return await synthesize_tts_audio_bytes(text, output_format="mp3")


async def speak_in_voice_channel(guild: discord.Guild, text: str, fallback_channel=None):
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return

    vc_connected = bool(guild and guild.voice_client and guild.voice_client.is_connected())
    if not vc_connected:
        if fallback_channel:
            mp3_bytes = await synthesize_tts_mp3_bytes(spoken_text)
            if mp3_bytes:
                await fallback_channel.send(
                    file=discord.File(io.BytesIO(mp3_bytes), filename="yukinon_voice.mp3")
                )
            else:
                print("TTS synthesis returned empty audio bytes.", flush=True)
        return

    # Single-pass in-memory PCM synthesis -> zero temp files, zero 2nd FFmpeg decode spawn!
    pcm_bytes = await synthesize_tts_audio_bytes(spoken_text, output_format="pcm")
    if not pcm_bytes:
        print("TTS synthesis returned empty audio bytes.", flush=True)
        return

    async with tts_lock:
        pcm_stream = io.BytesIO(pcm_bytes)
        try:
            vc = guild.voice_client
            while vc and vc.is_connected() and vc.is_playing():
                await asyncio.sleep(0.2)
            if not vc or not vc.is_connected():
                pcm_stream.close()
                return
            done_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _after_play(err):
                if err:
                    print(f"Voice PCM playback error: {err}", flush=True)
                try:
                    pcm_stream.close()
                except Exception:
                    pass
                loop.call_soon_threadsafe(done_event.set)

            print(
                f"[Voice TTS] Playing {len(pcm_bytes)} PCM bytes ({state.active_tts_engine}) "
                f"in '{vc.channel.name}' (in-memory)...",
                flush=True,
            )
            audio_source = discord.PCMAudio(pcm_stream)
            vc.play(audio_source, after=_after_play)
            await asyncio.wait_for(done_event.wait(), timeout=60)
            _VOICE_LAST_BOT_REPLY_TS[int(guild.id)] = time.time()
        except Exception as e:
            print(f"TTS voice channel playback error: {e}", flush=True)
            try:
                pcm_stream.close()
            except Exception:
                pass


# ==============================================================================
# PASSIVE MULTI-SPEAKER VOICE SCRIBE & CONVERSATION SUMMARIZER (IN-RAM ONLY)
# ==============================================================================
def is_voice_scribe_active(guild_id: int | None) -> bool:
    if not guild_id:
        return False
    sess = voice_scribe_sessions.get(int(guild_id))
    return bool(sess and sess.get("active"))


def clear_voice_scribe_session(guild_id: int | None) -> None:
    """Explicitly purges any in-RAM Voice Scribe transcript & references for `guild_id`."""
    if not guild_id:
        return
    sess = voice_scribe_sessions.pop(int(guild_id), None)
    if isinstance(sess, dict):
        entries = sess.get("entries")
        if isinstance(entries, list):
            entries.clear()
        sess["text_channel"] = None
        sess.clear()


def _has_wake_word_prefix_or_near_start(text: str) -> bool:
    low = (text or "").strip().lower()
    return any(low.startswith(w) or w in low[:22] for w in VOICE_WAKE_WORDS)


def _is_voice_scribe_start_command(text: str) -> bool:
    if not text:
        return False
    # Never steal "จด OT" commands into Scribe Mode
    if re.search(r"(\bot\b|โอ\s*ที)", text, re.IGNORECASE):
        return False
    return bool(VOICE_SCRIBE_START_REGEX.search(text))


def _is_voice_scribe_stop_command_deterministic(text: str) -> bool:
    if not text:
        return False
    if re.search(r"(\bot\b|โอ\s*ที)", text, re.IGNORECASE):
        return False
    if VOICE_SCRIBE_STOP_UNAMBIGUOUS_REGEX.search(text):
        return True
    if _has_wake_word_prefix_or_near_start(text):
        body = _strip_wake_word_prefix(text)
        if VOICE_SCRIBE_STOP_WAKE_SHORT_REGEX.search(body):
            return True
    return False


def _format_scribe_duration(seconds: float) -> str:
    total = max(1, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h} ชม. {m} นาที"
    if m > 0:
        return f"{m} นาที {s} วินาที"
    return f"{s} วินาที"


async def start_voice_scribe_session(
    guild: discord.Guild,
    text_channel,
    started_by: str = "",
    topic: str = "",
    speak_ack: bool = True,
) -> bool:
    """Starts an in-RAM passive multi-speaker Voice Scribe session for `guild`."""
    if not guild:
        return False
    gid = int(guild.id)
    existing = voice_scribe_sessions.get(gid)
    if existing and existing.get("active") and not existing.get("stopping"):
        cnt = len(existing.get("entries") or [])
        if text_channel:
            await text_channel.send(
                f"📝 ตอนนี้ฉันกำลังจดให้ห้องนี้อยู่แล้วค่ะ (จดไปแล้ว `{cnt}` ประโยค) "
                "ถ้าจะให้สรุป พิมพ์ `!stopscribe` หรือพูดว่า `ยูกิโนะ หยุดจด` ได้เลยนะคะ"
            )
        return False

    vc = guild.voice_client
    vc_name = vc.channel.name if (vc and getattr(vc, "channel", None)) else "Voice Channel"
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    clean_topic = re.sub(r"\s+", " ", (topic or "")).strip()[:160]

    voice_scribe_sessions[gid] = {
        "active": True,
        "stopping": False,
        "guild_id": gid,
        "channel_id": text_channel.id if text_channel else 0,
        "text_channel": text_channel,
        "voice_channel_name": vc_name,
        "topic": clean_topic,
        "started_by": started_by or "Member",
        "started_at_ts": time.time(),
        "started_at_str": now_th.strftime("%H:%M"),
        "started_date_str": now_th.strftime("%Y%m%d_%H%M"),
        "entries": [],
    }

    topic_line = f"\n- **หัวข้อ:** `{clean_topic}`" if clean_topic else ""
    api_log(f"Voice Scribe STARTED -> guild={gid}, channel='{vc_name}', topic='{clean_topic}'")
    if text_channel:
        await text_channel.send(
            f"📝 **[เริ่มโหมดจดบันทึกการสนทนา (Voice Scribe)]**\n"
            f"- **ห้องเสียง:** `{vc_name}` | **เริ่มเวลา:** `{now_th.strftime('%H:%M')} น.`{topic_line}\n"
            f"- ระหว่างนี้ฉันจะตั้งใจฟังและจดให้เงียบๆ โดยไม่พิมพ์รบกวนในแชทและไม่พูดแทรกนะคะ\n"
            f'- คุยเสร็จแล้วพิมพ์ `!stopscribe` / `!หยุดจด` / `!leave` หรือพูดว่า **"ยูกิโนะ หยุดจด"** / **"สรุปการประชุม"** ได้เลยค่ะ'
        )
    if speak_ack and vc and vc.is_connected():
        await speak_in_voice_channel(
            guild,
            "เข้าใจแล้วค่ะ ฉันจะตั้งใจฟังและจดให้เงียบๆ โดยไม่พูดแทรกนะคะ คุยเสร็จแล้วบอกให้หยุดหรือสรุปได้เลยค่ะ",
            fallback_channel=None,
        )
    return True


def _append_scribe_entry(
    guild_id: int,
    speaker_user,
    speaker_name: str,
    text: str,
    utterance_ts: float | None = None,
) -> tuple[bool, bool]:
    """
    Appends one transcribed utterance into the in-RAM session.
    Returns `(recorded, hit_limit_auto_stop)`.
    """
    sess = voice_scribe_sessions.get(int(guild_id))
    if not sess or not sess.get("active"):
        return False, False

    entries: list[dict] = sess["entries"]
    ts_val = float(utterance_ts or time.time())
    dt_th = datetime.fromtimestamp(ts_val, ZoneInfo("Asia/Bangkok"))
    uid = int(getattr(speaker_user, "id", 0) or 0)
    disp = (
        getattr(speaker_user, "display_name", None)
        or speaker_name
        or f"Speaker_{uid % 10000}"
    )

    # Deduplicate identical back-to-back line from same speaker within 2.5s
    if entries:
        last = entries[-1]
        if last["speaker_id"] == uid and last["text"] == text and abs(ts_val - last["ts"]) < 2.5:
            return False, False

    entries.append({
        "ts": ts_val,
        "time_str": dt_th.strftime("%H:%M:%S"),
        "speaker_id": uid,
        "speaker": str(disp).strip()[:36],
        "text": text.strip(),
    })
    print(f"[Voice Scribe] [{dt_th.strftime('%H:%M:%S')}] {disp}: {text}", flush=True)

    duration = time.time() - float(sess.get("started_at_ts") or ts_val)
    hit_limit = (len(entries) >= VOICE_SCRIBE_MAX_ENTRIES) or (duration >= VOICE_SCRIBE_MAX_DURATION_SEC)
    return True, hit_limit


async def stop_voice_scribe_session(
    guild: discord.Guild,
    fallback_text_channel=None,
    triggered_by: str = "",
    speak_ack: bool = True,
    reason: str = "",
) -> bool:
    """
    Flushes any in-progress speaker buffers (excluding current task to avoid 6s self-wait),
    summarizes the full multi-speaker transcript, posts the summary + full transcript to Discord,
    and guarantees the in-RAM session is cleared.
    """
    if not guild:
        return False
    gid = int(guild.id)
    sess = voice_scribe_sessions.get(gid)
    if not sess or not sess.get("active"):
        return False
    if sess.get("stopping"):
        return False

    sess["stopping"] = True
    target_channel = fallback_text_channel or sess.get("text_channel")

    try:
        # 1. Flush any speakers who just finished talking right before !stop / !leave / voice stop
        sink = _ACTIVE_VOICE_SINKS.get(gid)
        if sink is not None:
            try:
                await sink.flush_pending_utterances(timeout_sec=6.0)
            except Exception as e:
                api_log(f"Voice Scribe flush note: {type(e).__name__}: {e}")

        # 2. Detach & clear in-RAM session immediately so memory is released even if LLM fails
        entries = sorted(list(sess.get("entries") or []), key=lambda x: x["ts"])
        vc_name = sess.get("voice_channel_name") or "Voice Channel"
        topic_hint = sess.get("topic") or ""
        started_ts = float(sess.get("started_at_ts") or time.time())
        started_str = sess.get("started_at_str") or ""
        started_date_str = sess.get("started_date_str") or datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y%m%d_%H%M")
        clear_voice_scribe_session(gid)

        ended_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        duration_str = _format_scribe_duration(time.time() - started_ts)

        if not entries:
            if target_channel:
                reason_note = f" ({reason})" if reason else ""
                await target_channel.send(
                    f"📝 หยุดจดให้ห้อง `{vc_name}` แล้วค่ะ{reason_note} — "
                    f"ช่วง `{duration_str}` ที่ผ่านมาไม่ได้ยินเสียงคุยกันเลย ไม่มีอะไรให้สรุปนะคะ"
                )
            if speak_ack and guild.voice_client and guild.voice_client.is_connected():
                await speak_in_voice_channel(guild, "หยุดจดแล้วค่ะ แต่เมื่อกี้ไม่ได้ยินเสียงคุยกันเลย ไม่มีอะไรให้สรุปนะคะ")
            return True

        speakers_ordered: list[str] = []
        for e in entries:
            if e["speaker"] not in speakers_ordered:
                speakers_ordered.append(e["speaker"])

        transcript_lines = [
            f"[{e['time_str']}] {e['speaker']}: {e['text']}" for e in entries
        ]
        full_transcript_text = "\n".join(transcript_lines)
        entries.clear()

        api_log(
            f"Voice Scribe STOPPED -> guild={gid}, utterances={len(transcript_lines)}, "
            f"speakers={len(speakers_ordered)}, chars={len(full_transcript_text)}"
        )

        if target_channel:
            await target_channel.send(
                f"📝 หยุดจดให้ห้อง `{vc_name}` แล้วค่ะ (`{len(transcript_lines)}` ประโยค | `{len(speakers_ordered)}` ผู้พูด | `{duration_str}`) "
                "กำลังสรุปเรื่องที่คุยกันให้ รอสักครู่นะคะ..."
            )

        topic_ctx = f"หัวข้อที่ระบุไว้ตอนเริ่ม: {topic_hint}\n" if topic_hint else ""
        summary_prompt = (
            f"ด้านล่างนี้คือบันทึกคำพูดจากการสนทนาในห้องเสียง `{vc_name}` "
            f"(เวลา {started_str} - {ended_th.strftime('%H:%M')} น. | ระยะเวลา {duration_str} | "
            f"ผู้เข้าร่วมพูด: {', '.join(speakers_ordered)})\n"
            f"{topic_ctx}"
            "โปรดวิเคราะห์และสรุปบทสนทนาทั้งหมดให้ชัดเจน กระชับ อ่านง่าย ในน้ำเสียงสุขุมเป็นธรรมชาติของยูกิโนะ "
            "(ห้ามสร้างตาราง Markdown และห้ามแต่งข้อมูลที่ไม่ได้พูดในบันทึก):\n"
            "1. **ภาพรวมหัวข้อที่สนทนา (Overall Summary)**: สรุปสั้นๆ 2-4 ประโยคว่าคุยเรื่องอะไรบ้าง\n"
            "2. **ประเด็นสำคัญแยกตามหัวข้อ (Key Discussion Points)**: จัดกลุ่มหัวข้อที่พูดถึงพร้อมระบุใจความสำคัญและผู้เสนอประเด็น\n"
            "3. **ข้อสรุปและสิ่งที่ต้องทำต่อ (Decisions & Action Items)**: ระบุสิ่งที่ตกลงกันแล้วหรือสิ่งที่ต้องทำต่อพร้อมชื่อผู้รับผิดชอบ (ถ้าไม่มีให้ระบุสั้นๆ ว่าเป็นการพูดคุยทั่วไป ไม่มี Action Items)\n\n"
            f"=== บันทึกบทสนทนา (Voice Transcript) ===\n{full_transcript_text[:95000]}"
        )

        summary_answer, _ = await query_ai_chat(
            summary_prompt,
            system_prompt=FAST_SYSTEM_PROMPT,
            mode="general",
            channel_id=target_channel.id if target_channel else None,
            temperature=0.25,
            allow_agentic_search=False,
            has_fresh_attachment=True,
            speaker_user=None,
            speaker_name="",
            decision_plan={
                "route": "chat",
                "route_confidence": 0.99,
                "needs_web_search": False,
                "search_confidence": 0.99,
                "needs_tool_use": False,
                "tool_confidence": 0.99,
                "complexity_score": 2.0,
                "complexity_confidence": 0.90,
            },
        )

        from image_engine import send_chunked_response

        header_meta = (
            f"📋 **[สรุปการสนทนาในห้องเสียง `{vc_name}`]**\n"
            f"- **ช่วงเวลา:** `{started_str} - {ended_th.strftime('%H:%M')} น.` (`{duration_str}`) | "
            f"**ผู้พูด ({len(speakers_ordered)}):** `{', '.join(speakers_ordered)}` | "
            f"**จำนวน:** `{len(transcript_lines)} ประโยค`"
        )
        if topic_hint:
            header_meta += f"\n- **หัวข้อ:** `{topic_hint}`"
        if reason:
            header_meta += f"\n- **สถานะ:** `{reason}`"

        if target_channel:
            await send_chunked_response(
                target_channel,
                f"{header_meta}\n\n{summary_answer}",
                filename_fallback=f"voice_summary_{started_date_str}.md",
            )

            # Deliver full timestamped transcript inline (short) or as downloadable .txt (long)
            if len(full_transcript_text) <= 1450:
                await target_channel.send(
                    f"**🎙️ บันทึกคำพูดทั้งหมด (Full Transcript):**\n```text\n{full_transcript_text}\n```"
                )
            else:
                file_header = (
                    f"Voice Channel: {vc_name}\n"
                    f"Time (TH): {started_str} - {ended_th.strftime('%H:%M')} ({duration_str})\n"
                    f"Speakers: {', '.join(speakers_ordered)}\n"
                    + (f"Topic: {topic_hint}\n" if topic_hint else "")
                    + ("=" * 60)
                    + "\n\n"
                )
                transcript_bytes = (file_header + full_transcript_text).encode("utf-8")
                await target_channel.send(
                    content="📎 **บันทึกคำพูดฉบับเต็ม (Full Voice Transcript):**",
                    file=discord.File(
                        io.BytesIO(transcript_bytes),
                        filename=f"voice_transcript_{started_date_str}.txt",
                    ),
                )

        if speak_ack and guild.voice_client and guild.voice_client.is_connected():
            await speak_in_voice_channel(
                guild,
                "สรุปเรื่องที่คุยกันส่งไว้ในแชทให้เรียบร้อยแล้วค่ะ",
                fallback_channel=None,
            )

        transcript_lines.clear()
        return True
    finally:
        clear_voice_scribe_session(gid)
        release_unused_memory()


# ==============================================================================
# 3-TIER HYBRID SPEECH-TO-TEXT PIPELINE
# ==============================================================================
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
    "ขอบคุณครับ", "ขอบคุณค่ะ", "ขอบคุณที่รับชม", "โปรดติดตามตอนต่อไป",
    "พากย์ไทยโดย", "ซับไทยโดย", "ซับไทย", "โอ้โห", "อ๋อ", "อืม", "เอ่อ",
    "thank you", "thanks for watching", "you", "bye", ".", "..", "..."
}


def _extract_pcm16k_from_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extracts raw 16-bit mono PCM bytes and sample rate from a WAV buffer."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate() or 16000
        pcm = wf.readframes(wf.getnframes())
    if len(pcm) % 2 == 1:
        pcm = pcm[:-1]
    return pcm, rate


def _parse_ws_json_frame(msg: aiohttp.WSMessage) -> dict | None:
    """Parses Google Live API WebSocket frames (sent as BINARY or TEXT UTF-8 JSON)."""
    if msg.type == aiohttp.WSMsgType.TEXT:
        try:
            data = json.loads(msg.data)
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    if msg.type == aiohttp.WSMsgType.BINARY:
        try:
            data = json.loads(msg.data.decode("utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


async def transcribe_wav_bytes_gemini_live(wav_bytes: bytes) -> str:
    """
    Tier 1 Primary STT: Streams 16kHz mono PCM audio to `gemini-3.5-transcribe-live`
    over Google AI Studio's Multimodal Live WebSocket (`BidiGenerateContent`):
    - Parses BOTH `BINARY` and `TEXT` JSON frames.
    - Uses official `inputAudioTranscription` (`languageCodes`, `customVocabulary`, `mode="SMART"`)
      with manual VAD (`activityStart` / `activityEnd`) for instant utterance finalization.
    - Reads `serverContent.inputTranscription` (and `interimInputTranscription` fallback).
    """
    if not GEMINI_API_KEY or not GEMINI_TRANSCRIBE_LIVE_MODEL:
        return ""
    if time.time() < state.gemini_transcribe_live_cooldown_until:
        return ""

    try:
        pcm_bytes, sample_rate = _extract_pcm16k_from_wav_bytes(wav_bytes)
    except Exception:
        return ""

    if len(pcm_bytes) < 3200:
        return ""

    ws_url = (
        f"wss://{GEMINI_WS_HOST}/ws/"
        f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={GEMINI_API_KEY}"
    )

    setup_configs = (
        (
            "manual_smart",
            {
                "setup": {
                    "model": f"models/{GEMINI_TRANSCRIBE_LIVE_MODEL}",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                    },
                    "inputAudioTranscription": {
                        "languageCodes": ["th-TH", "en-US"],
                        "customVocabulary": GEMINI_LIVE_CUSTOM_VOCAB,
                        "mode": "SMART",
                    },
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {
                            "disabled": True,
                        }
                    },
                }
            },
        ),
        (
            "hybrid_auto",
            {
                "setup": {
                    "model": f"models/{GEMINI_TRANSCRIBE_LIVE_MODEL}",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                    },
                    "inputAudioTranscription": {
                        "languageCodes": ["th-TH", "en-US"],
                        "customVocabulary": GEMINI_LIVE_CUSTOM_VOCAB,
                    },
                }
            },
        ),
    )

    session = get_http_session()
    last_err = ""

    for mode_label, setup_msg in setup_configs:
        t0 = time.monotonic()
        try:
            async with session.ws_connect(ws_url, timeout=10.0, heartbeat=20.0) as ws:
                await ws.send_json(setup_msg)

                setup_ok = False
                setup_deadline = time.monotonic() + 4.5
                while time.monotonic() < setup_deadline:
                    rem = max(0.2, setup_deadline - time.monotonic())
                    msg = await ws.receive(timeout=rem)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        close_reason = getattr(msg, "extra", "") or str(msg.data or "")
                        raise RuntimeError(f"WS closed during setup ({msg.data} {close_reason})")
                    data = _parse_ws_json_frame(msg)
                    if not data:
                        continue
                    if "setupComplete" in data:
                        setup_ok = True
                        break
                    if "error" in data:
                        raise RuntimeError(str(data["error"])[:180])

                if not setup_ok:
                    raise RuntimeError("Live STT setup handshake timed out")

                use_manual_vad = (mode_label == "manual_smart")
                if use_manual_vad:
                    await ws.send_json({"realtimeInput": {"activityStart": {}}})

                padded_pcm = pcm_bytes + (b"\x00" * (sample_rate * 2 * 3 // 10))
                chunk_size = 3200
                for offset in range(0, len(padded_pcm), chunk_size):
                    chunk = padded_pcm[offset : offset + chunk_size]
                    if len(chunk) % 2 == 1:
                        chunk = chunk[:-1]
                    if not chunk:
                        continue
                    await ws.send_json({
                        "realtimeInput": {
                            "audio": {
                                "mimeType": f"audio/pcm;rate={sample_rate}",
                                "data": base64.b64encode(chunk).decode("ascii"),
                            }
                        }
                    })

                if use_manual_vad:
                    await ws.send_json({"realtimeInput": {"activityEnd": {}}})
                else:
                    await ws.send_json({"realtimeInput": {"audioStreamEnd": True}})

                final_parts: list[str] = []
                latest_interim = ""
                fallback_model_parts: list[str] = []
                read_deadline = time.monotonic() + 6.5

                while time.monotonic() < read_deadline:
                    rem = max(0.15, read_deadline - time.monotonic())
                    recv_timeout = min(rem, 0.75) if final_parts else rem
                    try:
                        msg = await ws.receive(timeout=recv_timeout)
                    except asyncio.TimeoutError:
                        break

                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

                    data = _parse_ws_json_frame(msg)
                    if not data:
                        continue
                    if "error" in data:
                        raise RuntimeError(str(data["error"])[:180])

                    sc = data.get("serverContent") or {}
                    if not isinstance(sc, dict):
                        continue

                    input_tr = sc.get("inputTranscription")
                    if isinstance(input_tr, dict):
                        tr_txt = (input_tr.get("text") or "").strip()
                        if tr_txt:
                            if use_manual_vad:
                                final_parts = [tr_txt]
                            else:
                                final_parts.append(tr_txt)
                        if input_tr.get("finished"):
                            break

                    interim_tr = sc.get("interimInputTranscription")
                    if isinstance(interim_tr, dict) and interim_tr.get("text"):
                        latest_interim = str(interim_tr["text"]).strip()

                    model_turn = sc.get("modelTurn") or {}
                    for part in model_turn.get("parts") or []:
                        if isinstance(part, dict) and part.get("text") and not part.get("thought", False):
                            fallback_model_parts.append(part["text"])

                    if sc.get("turnComplete") or sc.get("generationComplete"):
                        break

                chosen_raw = (
                    " ".join(final_parts)
                    if final_parts
                    else (latest_interim or "".join(fallback_model_parts))
                )
                raw_transcript = re.sub(r"\s+", " ", chosen_raw).strip()
                fixed_transcript = _apply_phonetic_fixes(raw_transcript)
                elapsed = time.monotonic() - t0
                if (
                    fixed_transcript
                    and len(fixed_transcript) >= 2
                    and fixed_transcript.lower() not in WHISPER_HALLUCINATIONS
                    and not NOISE_HALLUCINATION_REGEX.search(fixed_transcript)
                ):
                    api_log(
                        f"Gemini Transcribe Live ({mode_label}) -> {GEMINI_TRANSCRIBE_LIVE_MODEL} 200 OK "
                        f"({elapsed:.2f}s): '{fixed_transcript[:80]}'"
                    )
                    return fixed_transcript
                api_log(
                    f"Gemini Transcribe Live ({mode_label}) -> empty/filtered transcript ({elapsed:.2f}s)"
                )
                return ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if mode_label == "manual_smart" and ("429" not in last_err and "quota" not in last_err.lower()):
                api_log(f"Gemini Transcribe Live (manual_smart) note ({last_err}); retrying hybrid_auto")
                continue
            break

    if last_err:
        if "429" in last_err or "quota" in last_err.lower():
            state.gemini_transcribe_live_cooldown_until = time.time() + 60.0
        else:
            state.gemini_transcribe_live_cooldown_until = time.time() + 45.0
        api_log(
            f"Gemini Transcribe Live -> {GEMINI_TRANSCRIBE_LIVE_MODEL} fallback ({last_err})"
        )
    return ""


async def transcribe_wav_bytes_hybrid(wav_bytes: bytes) -> str:
    """
    3-Tier Hybrid Speech-to-Text Pipeline:
    - Tier 1: `gemini-3.5-transcribe-live` (Real-time Live API WebSocket, custom Thai/EN/JP vocab)
    - Tier 2: Groq `whisper-large-v3` (with segment logprob / no_speech_prob VAD filter)
    - Tier 3: Google SpeechRecognition (`th-TH` / `en-US` cross-check)
    """
    live_txt = await transcribe_wav_bytes_gemini_live(wav_bytes)
    if live_txt:
        return live_txt

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

                        if no_speech > 0.55 or avg_logprob < -1.25 or comp_ratio > 2.4:
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
                    if txt and txt.lower() not in WHISPER_HALLUCINATIONS and not NOISE_HALLUCINATION_REGEX.search(txt):
                        whisper_candidate = txt
        except Exception as e:
            print(f"Groq Whisper STT error (falling back to Google STT): {e}", flush=True)

    if not whisper_candidate or low_confidence:
        google_txt = await asyncio.to_thread(transcribe_wav_bytes_google_sync, wav_bytes)
        if google_txt and google_txt.lower() not in WHISPER_HALLUCINATIONS:
            if whisper_candidate and google_txt != whisper_candidate:
                print(
                    f"[Voice STT Cross-Check] Whisper='{whisper_candidate}' (low conf) -> Using Google STT='{google_txt}'",
                    flush=True,
                )
            return google_txt

    return whisper_candidate


def convert_audio_bytes_to_wav_sync(raw_bytes: bytes) -> bytes:
    """
    FFmpeg audio converter for uploaded audio attachments (.ogg/.mp3/.m4a/.flac -> 16kHz mono WAV)
    with fan-hum rejection (180Hz high-pass) and capped gentle normalization (m=3).
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-threads", "1", "-i", "pipe:0",
            "-af", "highpass=f=180,lowpass=f=6800,dynaudnorm=f=150:g=15:p=0.88:m=3",
            "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1", "-loglevel", "error"
        ],
        input=raw_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return proc.stdout


def convert_pcm48k_stereo_to_wav16k_sync(pcm_bytes: bytes) -> bytes:
    """
    Zero-subprocess in-memory NumPy DSP converter for live 48kHz stereo Discord PCM -> 16kHz mono WAV:
    1. Stereo-to-mono float32 averaging.
    2. Anti-aliased 3:1 decimation via zero-copy strided 21-tap Blackman-sinc FIR (`fc=6800Hz`).
    3. Zero-phase FFT bandpass (`180Hz` high-pass cuts PC/UPS/switch fan rumble; `6800Hz` low-pass).
    4. Capped gentle RMS/peak normalization (`max_gain <= 3.0`, peak <= 0.88 FS).
    Falls back automatically to FFmpeg (`convert_audio_bytes_to_wav_sync`) if NumPy is unavailable.
    """
    if not _HAS_NUMPY or _DECIM_LP_KERNEL is None:
        raw_wav_io = io.BytesIO()
        with wave.open(raw_wav_io, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_bytes)
        return convert_audio_bytes_to_wav_sync(raw_wav_io.getvalue())

    n_bytes = len(pcm_bytes) - (len(pcm_bytes) % 4)
    if n_bytes < 12:
        return b""

    stereo = np.frombuffer(pcm_bytes[:n_bytes], dtype=np.int16).reshape(-1, 2)
    mono = (stereo[:, 0].astype(np.float32) + stereo[:, 1].astype(np.float32)) * 0.5

    taps = _DECIM_LP_KERNEL.size
    half = taps // 2
    n_out = mono.size // 3
    if n_out < 16:
        return b""

    padded = np.pad(mono, (half, half), mode="edge")
    decim = np.zeros(n_out, dtype=np.float32)
    end_base = n_out * 3
    for k in range(taps):
        w = float(_DECIM_LP_KERNEL[k])
        if w != 0.0:
            decim += w * padded[k : k + end_base : 3]

    # Zero-phase FFT bandpass (smooth raised-cosine roll-offs at 120-220Hz & 6500-7200Hz)
    n_fft = decim.size
    spec = np.fft.rfft(decim)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / 16000.0)

    hp = np.ones_like(freqs, dtype=np.float32)
    hp[freqs <= 120.0] = 0.0
    trans_hp = (freqs > 120.0) & (freqs < 220.0)
    hp[trans_hp] = 0.5 * (1.0 - np.cos(np.pi * (freqs[trans_hp] - 120.0) / 100.0))

    lp = np.ones_like(freqs, dtype=np.float32)
    lp[freqs >= 7200.0] = 0.0
    trans_lp = (freqs > 6500.0) & (freqs < 7200.0)
    lp[trans_lp] = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans_lp] - 6500.0) / 700.0))

    filtered = np.fft.irfft(spec * (hp * lp), n=n_fft).astype(np.float32)

    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
    rms = float(np.sqrt(np.mean(filtered * filtered))) if filtered.size else 0.0
    if rms > 1.0 and peak > 1.0:
        target_gain = min(3.0, 3200.0 / rms, (0.88 * 32767.0) / peak)
        if target_gain > 1.02 or peak > (0.88 * 32767.0):
            filtered *= target_gain

    pcm16 = np.clip(np.rint(filtered), -32768.0, 32767.0).astype(np.int16)
    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm16.tobytes())
    return out_io.getvalue()


def _strip_wake_word_prefix(text: str) -> str:
    """Strips leading wake word ('ยูกิโนะ ...' / 'Yukino, ...') so voice commands parse cleanly."""
    for w in sorted(VOICE_WAKE_WORDS, key=len, reverse=True):
        m = re.match(rf"^{re.escape(w)}[\s,、!！?？:：-]+(.+)$", text, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return text


def _count_humans_in_voice_channel(guild: discord.Guild | None) -> int:
    if not guild or not guild.voice_client or not getattr(guild.voice_client, "channel", None):
        return 1
    humans = [m for m in guild.voice_client.channel.members if not m.bot]
    return max(1, len(humans))


def _build_recent_group_voice_context_block(
    recent_turns: list[dict],
    current_speaker_label: str,
    window_sec: float = 60.0,
) -> str:
    """Formats recent multi-speaker room utterances (last 60s) so Yukino understands group references."""
    if not recent_turns:
        return ""
    now = time.time()
    lines: list[str] = []
    has_other_speaker = False
    for t in recent_turns[-5:]:
        if (now - float(t.get("ts") or 0.0)) > window_sec:
            continue
        sp = str(t.get("speaker") or "Speaker")
        txt = str(t.get("text") or "").strip()
        if not txt:
            continue
        if not t.get("is_bot") and sp != current_speaker_label:
            has_other_speaker = True
        lines.append(f"- {sp}: {txt}")
    if not lines or not has_other_speaker:
        return ""
    return "[บริบทบทสนทนาล่าสุดในห้องเสียง]:\n" + "\n".join(lines)


async def handle_spoken_voice_command(
    guild: discord.Guild,
    text_channel,
    speaker_user,
    speaker_name: str,
    spoken_text: str,
    utterance_ts: float | None = None,
):
    from user_memory import (
        is_ot_command_text,
        process_ot_request,
        is_memory_command_text,
        process_memory_request,
    )

    cleaned = _apply_phonetic_fixes(spoken_text)
    if (
        len(cleaned) < 2
        or cleaned.lower() in WHISPER_HALLUCINATIONS
        or NOISE_HALLUCINATION_REGEX.search(cleaned)
    ):
        return

    gid = int(guild.id) if guild else 0
    speaker_is_owner = is_owner_user(speaker_user)
    speaker_label_ai = "Partner (Owner)" if speaker_is_owner else (speaker_name or "Member")

    # 0. Passive Multi-Speaker Voice Scribe Mode:
    # Silently record everyone; stop on deterministic command OR Decision-1 natural stop confirmation!
    if guild and is_voice_scribe_active(gid):
        sess = voice_scribe_sessions.get(gid)
        if sess and not sess.get("stopping"):
            should_stop = _is_voice_scribe_stop_command_deterministic(cleaned)
            if (
                not should_stop
                and not re.search(r"(\bot\b|โอ\s*ที)", cleaned, re.IGNORECASE)
                and _VOICE_SCRIBE_SOFT_STOP_CUE_REGEX.search(cleaned)
            ):
                stop_check = await check_voice_scribe_stop_with_decision_1(
                    cleaned,
                    speaker_label=speaker_label_ai,
                    recent_scribe_entries=sess.get("entries") or [],
                )
                if stop_check and stop_check.get("is_stop_scribe_command"):
                    should_stop = True

            if should_stop:
                print(f"[Voice Scribe] Stop command detected from {speaker_name}: '{cleaned}'", flush=True)
                await stop_voice_scribe_session(
                    guild,
                    fallback_text_channel=text_channel,
                    triggered_by=speaker_name,
                    speak_ack=True,
                )
                return

            _, hit_limit = _append_scribe_entry(
                gid, speaker_user, speaker_name, cleaned, utterance_ts=utterance_ts
            )
            if hit_limit and sess and not sess.get("stopping"):
                await stop_voice_scribe_session(
                    guild,
                    fallback_text_channel=text_channel,
                    triggered_by="Auto-Limit",
                    speak_ack=True,
                    reason="ถึงขีดจำกัดระยะเวลาหรือจำนวนข้อความสูงสุดอัตโนมัติ",
                )
        return

    human_count_in_vc = _count_humans_in_voice_channel(guild)
    explicit_wake = _has_wake_word_prefix_or_near_start(cleaned)
    print(
        f"[Voice STT] {speaker_name} spoke: '{cleaned}' "
        f"(wake_word_required={voice_wake_word_enabled[gid]}, vc_humans={human_count_in_vc}, wake={explicit_wake})",
        flush=True,
    )

    # 1. Instant Voice-Test Trigger (Always replies in chat even if wakeword is on)
    if VOICE_TEST_PHRASE_REGEX.search(cleaned):
        test_reply = (
            f'ฉันได้ยินเสียงพูดในห้องเสียงชัดเจนแล้วค่ะ (ข้อความที่ได้ยินคือ: **"{cleaned}"**) '
            "ไม่ต้องพูดย้ำหลายรอบหรอกนะคะ"
        )
        test_reply = sanitize_yukino_output(test_reply, cleaned, speaker_user, speaker_name)
        _record_voice_room_turn(gid, speaker_label_ai, cleaned, is_bot=False)
        _record_voice_room_turn(gid, "Yukinon", test_reply, is_bot=True)
        if text_channel:
            await text_channel.send(
                f"✅ **[🎙️ ทดสอบไมค์ — ได้ยินเสียงของ `{speaker_name}` ชัดเจนค่ะ]:** `{cleaned}`\n"
                f"**[🔊 Yukinon Voice]:** {test_reply}"
            )
        await speak_in_voice_channel(guild, test_reply, fallback_channel=text_channel)
        return

    if voice_wake_word_enabled[gid] and not explicit_wake:
        print(f"[Voice STT] Ignored '{cleaned}' (no wake word detected; use !wakeword off to disable).", flush=True)
        _record_voice_room_turn(gid, speaker_label_ai, cleaned, is_bot=False)
        return

    command_body = _strip_wake_word_prefix(cleaned)
    chan_id = text_channel.id if text_channel else 0

    class _VoiceProxyMsg:
        def __init__(self):
            self.author = speaker_user
            self.channel = text_channel
            self.guild = guild
            self.content = command_body

    # Snapshot recent room history BEFORE appending current utterance so Decision-1 sees clean prior context
    recent_turns_snapshot = list(_VOICE_ROOM_HISTORY.get(gid) or ())
    last_bot_age = time.time() - _VOICE_LAST_BOT_REPLY_TS.get(gid, 0.0)
    _record_voice_room_turn(gid, speaker_label_ai, cleaned, is_bot=False)

    # 2. Fast deterministic triggers when explicitly addressed OR alone in VC (0 API latency)
    is_direct_single_or_wake = explicit_wake or (human_count_in_vc == 1)
    if is_direct_single_or_wake:
        if guild and _is_voice_scribe_start_command(command_body):
            topic_hint = VOICE_SCRIBE_START_REGEX.sub("", command_body).strip(" :-")
            await start_voice_scribe_session(
                guild,
                text_channel,
                started_by=speaker_name,
                topic=topic_hint,
                speak_ack=True,
            )
            return

        if speaker_user and is_ot_command_text(command_body, channel_id=chan_id, user_id=speaker_user.id):
            ot_reply = await process_ot_request(_VoiceProxyMsg(), command_body)
            _record_voice_room_turn(gid, "Yukinon", ot_reply, is_bot=True)
            if text_channel:
                await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n{ot_reply}")
            await speak_in_voice_channel(guild, ot_reply, fallback_channel=text_channel)
            return

        if speaker_user and is_memory_command_text(command_body):
            mem_reply, _ = await process_memory_request(_VoiceProxyMsg(), command_body)
            _record_voice_room_turn(gid, "Yukinon", mem_reply, is_bot=True)
            if text_channel:
                await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n{mem_reply}")
            await speak_in_voice_channel(guild, mem_reply, fallback_channel=text_channel)
            return

    # 3. Phase 3 Multi-Speaker Voice Orchestrator (Single Forward Pass via Decision-1):
    # Decides `addressed_to_bot`, `voice_action`, `route`, `complexity`, and guardrails together!
    dec_plan = await orchestrate_voice_utterance_with_decision_1(
        command_body,
        speaker_label=speaker_label_ai,
        recent_voice_turns=recent_turns_snapshot,
        human_count_in_vc=human_count_in_vc,
        last_bot_reply_age_sec=last_bot_age,
        explicit_wake_word=explicit_wake,
        channel_id=chan_id or None,
        purpose="VoiceRoute",
    )

    # 4. Smart Multi-Speaker "Should I Speak?" Gate
    if dec_plan is not None:
        addressed_prob = float(dec_plan.get("addressed_prob") or 0.0)
        v_action = str(dec_plan.get("voice_action") or "reply_chat")
        v_action_conf = float(dec_plan.get("voice_action_confidence") or 0.0)

        if not explicit_wake:
            if human_count_in_vc >= 2:
                if addressed_prob < 0.45 or (v_action == "ignore_side_talk" and addressed_prob < 0.58):
                    print(
                        f"[Voice Group Filter] Ignored side chatter from {speaker_name} "
                        f"(addressed={addressed_prob:.2f}, action={v_action}): '{cleaned}'",
                        flush=True,
                    )
                    return
            else:
                # Alone in VC: only stay silent on unmistakable self-muttering / game callouts
                if v_action == "ignore_side_talk" and v_action_conf >= 0.85 and addressed_prob < 0.25:
                    print(
                        f"[Voice Filter] Ignored unaddressed solo remark from {speaker_name} "
                        f"(addressed={addressed_prob:.2f}, conf={v_action_conf:.2f}): '{cleaned}'",
                        flush=True,
                    )
                    return

        # 4A. Natural-language Start Scribe ('เดี๋ยวคุยงานกันนะ ช่วยนั่งฟังจดให้หน่อย')
        if (
            guild
            and v_action == "start_scribe"
            and v_action_conf >= 0.72
            and (explicit_wake or human_count_in_vc == 1 or addressed_prob >= 0.55)
            and not re.search(r"(\bot\b|โอ\s*ที)", command_body, re.IGNORECASE)
        ):
            topic_hint = VOICE_SCRIBE_START_REGEX.sub("", command_body).strip(" :-")
            if topic_hint == command_body:
                topic_hint = ""
            await start_voice_scribe_session(
                guild,
                text_channel,
                started_by=speaker_name,
                topic=topic_hint,
                speak_ack=True,
            )
            return

        # 4B. Dynamic OT dispatch
        if (
            speaker_user
            and (
                (v_action == "ot_log" and v_action_conf >= 0.75)
                or (dec_plan.get("route") == "ot_log" and float(dec_plan.get("route_confidence") or 0.0) >= 0.78)
            )
        ):
            normalized_ot_cmd = command_body
            if not is_ot_command_text(normalized_ot_cmd, channel_id=chan_id, user_id=speaker_user.id):
                normalized_ot_cmd = f"จด OT {normalized_ot_cmd}"
            ot_reply = await process_ot_request(_VoiceProxyMsg(), normalized_ot_cmd)
            _record_voice_room_turn(gid, "Yukinon", ot_reply, is_bot=True)
            if text_channel:
                await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n{ot_reply}")
            await speak_in_voice_channel(guild, ot_reply, fallback_channel=text_channel)
            return

        # 4C. Dynamic Memory dispatch (deterministic command form; natural phrasing uses native tool)
        if speaker_user and v_action == "memory_op" and v_action_conf >= 0.78:
            if is_memory_command_text(command_body):
                mem_reply, _ = await process_memory_request(_VoiceProxyMsg(), command_body)
                _record_voice_room_turn(gid, "Yukinon", mem_reply, is_bot=True)
                if text_channel:
                    await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n{mem_reply}")
                await speak_in_voice_channel(guild, mem_reply, fallback_channel=text_channel)
                return
            dec_plan["needs_tool_use"] = True
            dec_plan["tool_confidence"] = max(float(dec_plan.get("tool_confidence") or 0.0), 0.90)
    else:
        # Decision-1 unavailable fallback: stay quiet on unaddressed 2+ human chatter outside active 15s reply window
        if human_count_in_vc >= 2 and not explicit_wake and last_bot_age > 15.0:
            print(
                f"[Voice Group Filter] Decision-1 unavailable; skipped unaddressed group speech from {speaker_name}: '{cleaned}'",
                flush=True,
            )
            return

    # 5. Voice Chat / Search / Weather / Tools (never leak Owner username into prompt text!)
    if speaker_is_owner:
        voice_prompt = command_body
    else:
        voice_prompt = f"{speaker_name} พูดในห้องเสียงว่า: {command_body}"

    if human_count_in_vc >= 2:
        group_ctx_block = _build_recent_group_voice_context_block(
            recent_turns_snapshot, speaker_label_ai, window_sec=60.0
        )
        if group_ctx_block:
            voice_prompt = f"{voice_prompt}\n\n{group_ctx_block}"

    answer, _ = await query_ai_chat(
        voice_prompt,
        system_prompt=VOICE_SYSTEM_PROMPT,
        mode="voice",
        channel_id=text_channel.id if text_channel else None,
        temperature=0.45,
        allow_agentic_search=True,
        speaker_user=speaker_user,
        speaker_name=speaker_name,
        decision_plan=dec_plan,
    )
    _record_voice_room_turn(gid, "Yukinon", answer, is_bot=True)
    if text_channel:
        await text_channel.send(f"**[🎙️ `{speaker_name}`]:** {cleaned}\n**[🔊 Yukinon Voice]:** {answer}")
    await speak_in_voice_channel(guild, answer, fallback_channel=text_channel)


class SilenceGatedVoiceSink(voice_recv.AudioSink if HAS_VOICE_RECV else object):
    """
    Adaptive Noise-Floor Multi-Speaker VAD Sink tuned for rooms with steady fan noise:
    - Python 3.13+ safe NumPy int16->float64 RMS calculation (zero `audioop` dependency).
    - Zero-subprocess NumPy 48kHz->16kHz anti-aliased decimation + 180Hz fan-hum bandpass for STT.
    - Mode-aware silence flush: snappier `0.85s` in interactive voice chat, patient `1.20s` in Voice Scribe Mode.
    - Tracks a slow per-speaker ambient noise floor EMA so steady fan hum (rms 30-65) never triggers STT.
    - Preserves per-speaker utterance start timestamps so overlapping multi-speaker conversations
      sort in exact chronological order during Voice Scribe Mode.
    - Caps concurrent STT tasks via `voice_stt_semaphore` (`Semaphore(2)`).
    """

    BASE_SPEECH_RMS_MIN = 110.0
    CONTINUE_SPEECH_RMS_MIN = 85.0
    MIN_PEAK_RMS = 145.0
    MIN_ACTIVE_SPEECH_FRAMES = 10

    def __init__(self, guild: discord.Guild, text_channel, vc=None):
        super().__init__()
        self.guild = guild
        self.text_channel = text_channel
        self._vc_ref = vc
        self._lock = threading.Lock()
        self.pre_roll_buffers = defaultdict(lambda: deque(maxlen=20))
        self.user_buffers = defaultdict(bytearray)
        self.speech_frame_counts = defaultdict(int)
        self.peak_rms = defaultdict(float)
        self.noise_floors = defaultdict(lambda: 45.0)
        self.utterance_started_at: dict = {}
        self.last_spoken = {}
        self._inflight_tasks: set[asyncio.Task] = set()
        self._last_rms_log = 0.0
        self.running = True
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = state.bot.loop if state.bot else asyncio.get_event_loop()
        self.monitor_task = self._loop.create_task(self._silence_monitor_loop())

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
            frame_rms = _calc_pcm16_rms(data.pcm)
        except Exception:
            return

        now = time.monotonic()
        gid = int(self.guild.id) if self.guild else 0
        hangover_sec = SCRIBE_HANGOVER_SEC if is_voice_scribe_active(gid) else INTERACTIVE_HANGOVER_SEC

        with self._lock:
            floor = self.noise_floors[speaker]
            start_thresh = max(self.BASE_SPEECH_RMS_MIN, floor * 2.25)
            cont_thresh = max(self.CONTINUE_SPEECH_RMS_MIN, floor * 1.75)

            buf = self.user_buffers[speaker]
            is_capturing = len(buf) > 0
            effective_thresh = cont_thresh if is_capturing else start_thresh

            if frame_rms >= effective_thresh:
                if not is_capturing:
                    self.utterance_started_at[speaker] = time.time()
                    for pre_frame in self.pre_roll_buffers[speaker]:
                        buf.extend(pre_frame)
                    self.pre_roll_buffers[speaker].clear()
                if len(buf) < MAX_UTTERANCE_BYTES:
                    buf.extend(data.pcm)
                self.speech_frame_counts[speaker] += 1
                if frame_rms > self.peak_rms[speaker]:
                    self.peak_rms[speaker] = frame_rms
                self.last_spoken[speaker] = now
            elif is_capturing and (now - self.last_spoken.get(speaker, 0) < hangover_sec):
                if len(buf) < MAX_UTTERANCE_BYTES:
                    buf.extend(data.pcm)
            else:
                if frame_rms > 5.0:
                    updated_floor = (floor * 0.94) + (min(frame_rms, 110.0) * 0.06)
                    self.noise_floors[speaker] = max(15.0, min(95.0, updated_floor))
                self.pre_roll_buffers[speaker].append(data.pcm)

            if frame_rms > 25.0 and (now - self._last_rms_log > 5.0):
                self._last_rms_log = now
                print(
                    f"[Voice Mic Level] speaker={speaker.display_name}, "
                    f"rms={int(frame_rms)}, floor={int(self.noise_floors[speaker])}, thresh={int(start_thresh)}",
                    flush=True,
                )

    def _spawn_pcm_task(self, user, pcm_bytes: bytes, utterance_ts: float) -> asyncio.Task:
        task = self._loop.create_task(self._process_user_pcm(user, pcm_bytes, utterance_ts))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return task

    async def flush_pending_utterances(self, timeout_sec: float = 6.0) -> None:
        """Flushes any buffered speaker utterances immediately and awaits other in-flight STT tasks."""
        ready_payloads = []
        with self._lock:
            for user in list(self.last_spoken.keys()):
                pcm_bytes = bytes(self.user_buffers.pop(user, b""))
                active_frames = self.speech_frame_counts.pop(user, 0)
                peak = self.peak_rms.pop(user, 0.0)
                floor = self.noise_floors.get(user, 45.0)
                utt_ts = self.utterance_started_at.pop(user, time.time())
                self.last_spoken.pop(user, None)
                min_peak = max(self.MIN_PEAK_RMS, floor * 2.6)
                if active_frames >= self.MIN_ACTIVE_SPEECH_FRAMES and peak >= min_peak and len(pcm_bytes) >= 38400:
                    ready_payloads.append((user, pcm_bytes, utt_ts))

        for user, pcm_bytes, utt_ts in ready_payloads:
            self._spawn_pcm_task(user, pcm_bytes, utt_ts)

        # Exclude current task so voice-triggered Scribe Stop never waits 6s on itself!
        cur_task = asyncio.current_task()
        pending = [t for t in list(self._inflight_tasks) if not t.done() and t is not cur_task]
        if pending:
            await asyncio.wait(pending, timeout=timeout_sec)

    def cleanup(self):
        self.running = False
        if self.guild:
            unregister_voice_sink(self.guild.id)
        if getattr(self, "monitor_task", None) and not self.monitor_task.done():
            self.monitor_task.cancel()
        with self._lock:
            self.pre_roll_buffers.clear()
            self.user_buffers.clear()
            self.speech_frame_counts.clear()
            self.peak_rms.clear()
            self.noise_floors.clear()
            self.utterance_started_at.clear()
            self.last_spoken.clear()

    async def _silence_monitor_loop(self):
        while self.running:
            await asyncio.sleep(0.30)
            now = time.monotonic()
            gid = int(self.guild.id) if self.guild else 0
            flush_sec = SCRIBE_SILENCE_FLUSH_SEC if is_voice_scribe_active(gid) else INTERACTIVE_SILENCE_FLUSH_SEC

            ready_payloads = []
            with self._lock:
                for user, last_t in list(self.last_spoken.items()):
                    buf_len = len(self.user_buffers.get(user, b""))
                    # Flush after mode-aware silence window (0.85s interactive / 1.20s Scribe) OR 18s RAM cap
                    if (now - last_t >= flush_sec) or (buf_len >= MAX_UTTERANCE_BYTES):
                        pcm_bytes = bytes(self.user_buffers.pop(user, b""))
                        active_frames = self.speech_frame_counts.pop(user, 0)
                        peak = self.peak_rms.pop(user, 0.0)
                        floor = self.noise_floors.get(user, 45.0)
                        utt_ts = self.utterance_started_at.pop(user, time.time())
                        self.last_spoken.pop(user, None)
                        min_peak = max(self.MIN_PEAK_RMS, floor * 2.6)
                        if (
                            active_frames >= self.MIN_ACTIVE_SPEECH_FRAMES
                            and peak >= min_peak
                            and len(pcm_bytes) >= 38400
                        ):
                            ready_payloads.append((user, pcm_bytes, active_frames, peak, floor, utt_ts))
                        elif active_frames > 0:
                            print(
                                f"[Voice VAD Dropped Noise] speaker={user.display_name}, "
                                f"frames={active_frames}, peak={int(peak)} < {int(min_peak)} (floor={int(floor)})",
                                flush=True,
                            )
            for user, pcm_bytes, active_frames, peak, floor, utt_ts in ready_payloads:
                print(
                    f"[Voice VAD] Captured speech from {user.display_name} "
                    f"({active_frames} active frames, peak={int(peak)}, floor={int(floor)}, {len(pcm_bytes)} bytes)",
                    flush=True,
                )
                self._spawn_pcm_task(user, pcm_bytes, utt_ts)

    async def _process_user_pcm(self, user, pcm_bytes: bytes, utterance_ts: float):
        try:
            spoken_text = ""
            # Semaphore covers only STT; holding it through the LLM/TTS reply would starve other speakers.
            # Ceiling: reply pipelines (Decision-1 + LLM) now run unbounded in parallel, one per utterance;
            # only playback is serialized by tts_lock. Upgrade path: a small separate reply semaphore if
            # provider rate limits get hit in busy voice channels.
            async with voice_stt_semaphore:
                compact_wav = await asyncio.to_thread(convert_pcm48k_stereo_to_wav16k_sync, pcm_bytes)
                if not compact_wav:
                    return
                spoken_text = await transcribe_wav_bytes_hybrid(compact_wav)
            if spoken_text:
                await handle_spoken_voice_command(
                    self.guild,
                    self.text_channel,
                    user,
                    user.display_name,
                    spoken_text,
                    utterance_ts=utterance_ts,
                )
            else:
                print(f"[Voice STT] Audio captured from {user.display_name}, but STT returned empty text.", flush=True)
        except Exception as e:
            print(f"Voice PCM processing error ({getattr(user, 'display_name', 'unknown')}): {e}", flush=True)