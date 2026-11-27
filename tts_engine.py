import os
import io
import sys
import ast
import re
import json
import time
import importlib.util
import wave
import threading
import asyncio
import subprocess
from urllib.parse import quote_plus

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

import aiohttp
import discord

from config import (
    HTTPS,
    CONTAINER_RAM_LIMIT_MB,
    TTS_PRIMARY_ENGINE,
    TTS_PARADEE_SPEED,
    TTS_PARADEE_PITCH,
    TTS_PARADEE_MODEL_DIR,
    TTS_PARADEE_IDLE_UNLOAD_SEC,
    TTS_PARADEE_RAM_GUARD_MB,
    TTS_PARADEE_TIMEOUT_SEC,
    TTS_GOOGLE_PITCH,
    TTS_GOOGLE_SPEED,
    TTS_SILENCE_TRIM,
    THAI_CHAR_REGEX,
    URL_REGEX,
    tts_lock,
    manga_semaphore,
    state,
    api_log,
    get_http_session,
    release_unused_memory,
)
from persona import sanitize_yukino_output

PCM_FRAME_BYTES = 3840  # 20ms @ 48kHz stereo 16-bit PCM

# Host constant split to prevent chat/markdown auto-linkers from mangling URLs
GTRANS_HOST = "translate." + "googleapis.com"

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
_PARADEE_ASYNC_LOCK: asyncio.Lock | None = None
_PARADEE_TTS_MODEL = None
_PARADEE_LAST_USED_TS: float = 0.0
_PARADEE_IDLE_TASK: asyncio.Task | None = None

# Shared per-guild last bot reply timestamp (updated on voice playback; imported by voice_engine.py)
_VOICE_LAST_BOT_REPLY_TS: dict[int, float] = {}


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


def _get_paradee_async_lock() -> asyncio.Lock:
    global _PARADEE_ASYNC_LOCK
    if _PARADEE_ASYNC_LOCK is None:
        _PARADEE_ASYNC_LOCK = asyncio.Lock()
    return _PARADEE_ASYNC_LOCK


def _can_safely_load_paradee_now() -> tuple[bool, str]:
    """
    OOM guardrail for 512 MB Render containers:
    - Always allows synthesis if Paradee is already resident in RAM (unless on cooldown).
    - When unloaded on <=512 MB containers, skips loading during active manga rendering
      (`manga_semaphore.locked()`) or when RSS already exceeds `TTS_PARADEE_RAM_GUARD_MB` (default 340 MB).
    """
    if time.time() < state.paradee_tts_cooldown_until:
        return False, "cooldown"
    if _PARADEE_TTS_MODEL is not None:
        return True, ""
    if CONTAINER_RAM_LIMIT_MB <= 512:
        if manga_semaphore.locked():
            return False, "manga_active_oom_guard"
        rss_mb = _get_current_rss_mb()
        if rss_mb >= float(TTS_PARADEE_RAM_GUARD_MB):
            return False, f"high_rss_{rss_mb:.0f}MB"
    return True, ""


_THAI_DIGITS_SIMPLE = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
_THAI_PLACES_SIMPLE = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน", "ล้าน")


def _int_to_thai_words(n: int) -> str:
    """Zero-dependency pure-Python integer-to-Thai reader (avoids importing heavy pythainlp)."""
    if n < 0:
        return "ลบ" + _int_to_thai_words(-n)
    if n == 0:
        return "ศูนย์"
    if n >= 1_000_000:
        millions, rem = divmod(n, 1_000_000)
        return _int_to_thai_words(millions) + "ล้าน" + (_int_to_thai_words(rem) if rem else "")
    digits = [int(ch) for ch in str(n)]
    length = len(digits)
    parts: list[str] = []
    for idx, d in enumerate(digits):
        if d == 0:
            continue
        pos = length - 1 - idx
        if pos == 0 and d == 1 and length > 1:
            parts.append("เอ็ด")
        elif pos == 1 and d == 2:
            parts.append("ยี่สิบ")
        elif pos == 1 and d == 1:
            parts.append("สิบ")
        else:
            parts.append(_THAI_DIGITS_SIMPLE[d] + _THAI_PLACES_SIMPLE[pos])
    return "".join(parts)


def _preprocess_thai_lightweight(text: str) -> str:
    """Expands Thai/Arabic digits & maiyamok (ๆ) cleanly without importing pythainlp."""
    thai_to_arabic = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    s = (text or "").translate(thai_to_arabic)
    # Expand comma-formatted numbers (e.g. 1,500 -> 1500)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    def _num_repl(m: re.Match) -> str:
        raw = m.group(0)
        if "." in raw:
            int_p, frac_p = raw.split(".", 1)
            frac_words = " ".join(_THAI_DIGITS_SIMPLE[int(c)] for c in frac_p if c.isdigit())
            return f"{_int_to_thai_words(int(int_p))}จุด{frac_words}"
        if len(raw) >= 7:
            return " ".join(_THAI_DIGITS_SIMPLE[int(c)] for c in raw)
        return _int_to_thai_words(int(raw))

    s = re.sub(r"\b\d+(?:\.\d+)?\b", _num_repl, s)
    # Expand maiyamok (ๆ) by repeating the preceding word/syllable chunk
    s = re.sub(r"([\u0E00-\u0E4E]{1,12})\s*ๆ", r"\1 \1", s)
    s = s.replace("ๆ", "")
    return s


def _dp_thai_tokenize_flat(text: str, vocab_dict: dict[str, str], max_len: int = 24) -> list[str]:
    """
    Zero-Trie minimum-token maximum-matching Thai word segmenter (newmm-equivalent DP).
    Reads directly from the flat `ipa_dict` keys (~2.5 MB RAM vs ~100 MB for `pythainlp.util.Trie`)
    and finishes in <0.5 ms per utterance without holding the GIL during a 62k-word Trie build!
    """
    n = len(text)
    if n == 0:
        return []
    INF = 10**9
    dp: list[tuple[int, int, int]] = [(INF, INF, n)] * (n + 1)
    dp[n] = (0, 0, n)

    for i in range(n - 1, -1, -1):
        ch = text[i]
        if ch.isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            u, t, _ = dp[j]
            dp[i] = (u, t + 1, j)
            continue
        if ch.isascii() and (ch.isalnum() or ch in "-_'"):
            j = i + 1
            while j < n and text[j].isascii() and (text[j].isalnum() or text[j] in "-_'"):
                j += 1
            u, t, _ = dp[j]
            dp[i] = (u, t + 1, j)
            continue

        best = (INF, INF, i + 1)
        limit = min(n, i + max_len)
        matched_any = False
        for j in range(limit, i, -1):
            sub = text[i:j]
            if sub in vocab_dict:
                matched_any = True
                u, t, _ = dp[j]
                cand = (u, t + 1, j)
                if cand < best:
                    best = cand

        u1, t1, _ = dp[i + 1]
        unk_cand = (u1 + 1, t1 + 1, i + 1)
        if not matched_any or unk_cand < best:
            best = unk_cand
        dp[i] = best

    tokens: list[str] = []
    i = 0
    while i < n:
        _, _, nxt = dp[i]
        if nxt <= i:
            nxt = i + 1
        tok = text[i:nxt]
        if (
            tokens
            and len(tok) == 1
            and ("\u0E00" <= tok <= "\u0E7F")
            and tok not in vocab_dict
            and tokens[-1] not in vocab_dict
            and all("\u0E00" <= c <= "\u0E7F" for c in tokens[-1])
        ):
            tokens[-1] += tok
        else:
            tokens.append(tok)
        i = nxt
    return tokens


def _locate_pythaitts_pkg_root() -> str:
    """Finds the installed `pythaitts` folder on disk purely via filesystem scan (ZERO Python import!)."""
    for base in sys.path:
        if not base or not isinstance(base, str):
            continue
        cand = os.path.join(base, "pythaitts")
        if os.path.isdir(cand):
            return cand
    try:
        spec = importlib.util.find_spec("pythaitts")
        if spec and spec.submodule_search_locations:
            return next(iter(spec.submodule_search_locations))
    except Exception:
        pass
    return ""


def _locate_pythaitts_ipa_json() -> str:
    """Finds PyThaiTTS's bundled `fastthaig2p/data/ipa.json` on disk WITHOUT importing `pythaitts` or `pythainlp`."""
    if TTS_PARADEE_MODEL_DIR:
        baked_ipa = os.path.join(TTS_PARADEE_MODEL_DIR, "ipa.json")
        if os.path.isfile(baked_ipa):
            return baked_ipa
    pkg_root = _locate_pythaitts_pkg_root()
    if pkg_root:
        for rel in (
            os.path.join("pretrained", "fastthaig2p", "data", "ipa.json"),
            os.path.join("pretrained", "makham", "data", "ipa.json"),
        ):
            cand = os.path.join(pkg_root, rel)
            if os.path.isfile(cand):
                return cand
    return ""


_FORBIDDEN_LEAF_IMPORTS = frozenset({
    "pythaitts", "pythainlp", "torch", "torchaudio", "transformers",
    "soundfile", "scipy", "tltk", "huggingface_hub", "onnxruntime",
})


def _load_stdlib_leaf_module_safely(file_path: str, alias_name: str):
    """
    Loads a single leaf `.py` file (e.g. `kokoro.py` or `en_g2p.py`) directly by file path
    ONLY after AST verification proves it has NO relative imports and NO heavy dependencies
    (`pythainlp`, `pythaitts`, `torch`, `tltk`, `soundfile`, etc.).
    Completely bypasses `pythaitts/__init__.py` so zero heavy NLP packages enter `sys.modules`!
    """
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root_mod = (a.name or "").split(".", 1)[0]
                    if root_mod in _FORBIDDEN_LEAF_IMPORTS:
                        return None
            elif isinstance(node, ast.ImportFrom):
                if getattr(node, "level", 0) > 0:
                    return None
                root_mod = (node.module or "").split(".", 1)[0]
                if root_mod in _FORBIDDEN_LEAF_IMPORTS:
                    return None
        spec = importlib.util.spec_from_file_location(alias_name, file_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _fallback_ipa_to_kokoro(ipa: str) -> str:
    """
    Zero-dependency Thai Wiktionary IPA -> Kokoro 5-tone phoneme mapper matching FastThaiG2P's
    `kokoro.py` (strips slashes/dots/unreleased markers, maps affricates & all 5 Chao tone marks).
    """
    s = (ipa or "").replace("/", "").replace(".", "").replace("̚", "").replace("̯", "")
    for src, dst in (
        ("t͡ɕʰ", "cʰ"), ("t͡ɕ", "c"), ("tɕʰ", "cʰ"), ("tɕ", "c"),
        ("˩˩˦", "↗"), ("˩˦", "↗"), ("˨˩", "↓"), ("˥˩", "↘"), ("˦˥", "↑"), ("˧", "→"),
    ):
        s = s.replace(src, dst)
    return re.sub(r"\s+", " ", s).strip()


class _LowRamParadeeEngine:
    """
    Ultra-lean direct ONNX Runtime runner for MakHamTTS Paradee 1.1 INT8 (8.07 MB):
    - Bypasses `pythainlp.util.Trie` (saves ~100 MB RAM and avoids multi-second GIL stalls on 0.1 vCPU!).
    - Configures `onnxruntime.SessionOptions` with `enable_cpu_mem_arena=False`, `enable_mem_pattern=False`,
      `intra_op_num_threads=1`, `inter_op_num_threads=1`, `ORT_SEQUENTIAL`, `ORT_ENABLE_BASIC`, and
      `allow_spinning=0` (prevents 0.1 vCPU CFS starvation & Discord heartbeat blocking on Render Free Tier!).
    """

    sample_rate = 24000

    def __init__(self, model_path: str, config_path: str, custom_dict: dict[str, str] | None = None):
        import onnxruntime as ort

        t0 = time.monotonic()
        rss_start = _get_current_rss_mb()
        api_log(f"Paradee TTS init start (low-RAM direct ORT, RSS={rss_start:.1f} MB)")

        with open(config_path, "r", encoding="utf-8") as cf:
            cfg = json.load(cf)
        self.vocab: dict[str, int] = dict(cfg.get("vocab") or {})
        if not self.vocab:
            raise RuntimeError("Paradee config.json missing 'vocab' mapping")

        # Load FastThaiG2P's 62k-word IPA table as a flat dict (~3-10 MB vs ~100 MB nested Trie)
        ipa_path = _locate_pythaitts_ipa_json()
        if not ipa_path:
            raise RuntimeError("Could not locate pythaitts fastthaig2p data/ipa.json")
        with open(ipa_path, "r", encoding="utf-8") as ipf:
            raw_ipa = json.load(ipf)

        self.ipa_dict: dict[str, str] = {}
        self.max_word_len = 12
        for k, v in raw_ipa.items():
            if not isinstance(k, str) or not k:
                continue
            val_s = v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else "")
            if isinstance(val_s, str) and val_s:
                self.ipa_dict[k] = val_s
                if len(k) > self.max_word_len:
                    self.max_word_len = min(28, len(k))
        del raw_ipa

        # Load ONLY stdlib-safe leaf files (`kokoro.py`, `en_g2p.py`) directly by path —
        # NEVER execute `from pythaitts...` (which imports `pythainlp` + ~270 MB into sys.modules!)
        self._ipa_to_kokoro = _fallback_ipa_to_kokoro
        self._en_g2p = None
        pkg_root = _locate_pythaitts_pkg_root()
        if pkg_root:
            f2p_dir = os.path.join(pkg_root, "pretrained", "fastthaig2p")
            leaf_kokoro = _load_stdlib_leaf_module_safely(
                os.path.join(f2p_dir, "kokoro.py"), "_paradee_leaf_kokoro"
            )
            if leaf_kokoro is not None and callable(getattr(leaf_kokoro, "ipa_to_kokoro", None)):
                self._ipa_to_kokoro = leaf_kokoro.ipa_to_kokoro
            leaf_eng2p = _load_stdlib_leaf_module_safely(
                os.path.join(f2p_dir, "en_g2p.py"), "_paradee_leaf_eng2p"
            )
            if leaf_eng2p is not None and callable(getattr(leaf_eng2p, "en_g2p", None)):
                self._en_g2p = leaf_eng2p.en_g2p

        # Compile custom_dict entries (supports raw `/ipa/` OR Thai hyphenated syllables like `ยู-กิ-โนะ`)
        self.custom_lower_map: dict[str, str] = {}
        for word, reading in (custom_dict or {}).items():
            w_clean = (word or "").strip()
            r_clean = (reading or "").strip()
            if not w_clean or not r_clean:
                continue
            if "/" in r_clean:
                ipa_val = r_clean
            else:
                sylls = [p.strip() for p in re.split(r"[\s\-]+", r_clean) if p.strip()]
                ipa_parts = [self._lookup_thai_piece(sy) for sy in sylls]
                ipa_val = " ".join(p for p in ipa_parts if p)
            if ipa_val:
                self.ipa_dict[w_clean] = ipa_val
                self.custom_lower_map[w_clean.lower()] = ipa_val
                if len(w_clean) > self.max_word_len:
                    self.max_word_len = min(28, len(w_clean))

        rss_g2p = _get_current_rss_mb()
        api_log(
            f"Paradee G2P ready ({len(self.ipa_dict):,} entries, en_g2p={self._en_g2p is not None}, "
            f"{time.monotonic() - t0:.2f}s, RSS={rss_g2p:.1f} MB)"
        )

        # Strict 512 MB / 0.1 vCPU ONNX Runtime SessionOptions (zero arena bloat, zero thread spinning)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        try:
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception:
            pass

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        in_names = [inp.name for inp in self.session.get_inputs()]
        self._ids_input_name = in_names[0] if in_names else "input_ids"
        self._speed_input_name = in_names[1] if len(in_names) > 1 else "speed"
        rss_end = _get_current_rss_mb()
        api_log(
            f"Paradee ORT session ready (inputs={in_names}, {time.monotonic() - t0:.2f}s, RSS={rss_end:.1f} MB)"
        )

    def _lookup_thai_piece(self, word: str) -> str:
        if word in self.ipa_dict:
            return self.ipa_dict[word]
        # Greedy sub-word split for rare OOV Thai compounds
        out: list[str] = []
        i = 0
        n = len(word)
        while i < n:
            matched = False
            for j in range(min(n, i + 12), i, -1):
                sub = word[i:j]
                if sub in self.ipa_dict:
                    out.append(self.ipa_dict[sub])
                    i = j
                    matched = True
                    break
            if not matched:
                i += 1
        return " ".join(out)

    def _text_to_phonemes(self, text: str) -> str:
        pre = _preprocess_thai_lightweight(text)
        tokens = _dp_thai_tokenize_flat(pre, self.ipa_dict, max_len=self.max_word_len)
        ipa_chunks: list[str] = []
        for tok in tokens:
            if not tok:
                continue
            if tok.isspace():
                ipa_chunks.append(" ")
                continue
            if tok in self.ipa_dict:
                ipa_chunks.append(self.ipa_dict[tok])
                continue
            low = tok.lower()
            if low in self.custom_lower_map:
                ipa_chunks.append(self.custom_lower_map[low])
                continue
            if tok.isascii():
                if re.search(r"[A-Za-z]", tok) and self._en_g2p is not None:
                    try:
                        en_ipa = self._en_g2p(tok)
                        if en_ipa:
                            ipa_chunks.append(str(en_ipa))
                            continue
                    except Exception:
                        pass
                if tok in (",", ".", "!", "?", ";", ":"):
                    ipa_chunks.append(tok)
                continue
            sub_ipa = self._lookup_thai_piece(tok)
            if sub_ipa:
                ipa_chunks.append(sub_ipa)

        raw_ipa = " ".join(p for p in ipa_chunks if p)
        raw_ipa = re.sub(r"\s+", " ", raw_ipa).strip()
        return self._ipa_to_kokoro(raw_ipa)

    @staticmethod
    def _split_natural_clauses(text: str) -> list[str]:
        """
        Splits multi-sentence Thai/English text at natural sentence boundaries (`นะคะ `, `ค่ะ `, `. `, `! `, `? `)
        so each sentence gets a calm end-of-utterance cadence and a 160ms breath pause instead of rushing
        directly into closing phrases like `ลาก่อนค่ะ`.
        """
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return []
        raw_parts = re.split(r"(?<=[.!?])\s+|(?<=นะคะ)\s+|(?<=ค่ะ)\s+|\n+", cleaned)
        clauses: list[str] = []
        for part in raw_parts:
            p = part.strip()
            if not p:
                continue
            # Merge ultra-tiny 1-2 char fragments into previous clause, but keep real phrases like `ลาก่อนค่ะ` separate
            if clauses and len(p) <= 3:
                clauses[-1] = f"{clauses[-1]} {p}"
            else:
                clauses.append(p)
        return clauses or [cleaned]

    def _run_onnx_ids(self, ch_ids: list[int], speed: float):
        seq = [0] + ch_ids + [0]
        feed = {
            self._ids_input_name: np.array([seq], dtype=np.int64),
            self._speed_input_name: np.array([float(speed)], dtype=np.float32),
        }
        out = self.session.run(None, feed)[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def synthesize_waveform(self, text: str, speed: float = 1.05):
        clauses = self._split_natural_clauses(text)
        if not clauses:
            return np.zeros(0, dtype=np.float32)

        chunk_limit = 420
        space_id = self.vocab.get(" ")
        comma_id = self.vocab.get(",")
        clause_gap = np.zeros(int(self.sample_rate * 0.16), dtype=np.float32)
        chunk_gap = np.zeros(int(self.sample_rate * 0.06), dtype=np.float32)
        tail_pad = np.zeros(int(self.sample_rate * 0.20), dtype=np.float32)

        clause_wavs: list = []
        for c_idx, clause_text in enumerate(clauses):
            # Insert a mild comma pause after `แล้ว ` mid-clause if `,` is supported in vocab
            prepped_clause = clause_text
            if comma_id is not None:
                prepped_clause = re.sub(r"(แล้ว)\s+", r"\1 , ", prepped_clause)

            phonemes = self._text_to_phonemes(prepped_clause)
            if not phonemes:
                continue
            all_ids = [self.vocab[c] for c in phonemes if c in self.vocab]
            if not all_ids:
                continue

            # Gentle deceleration (0.96x) on very short standalone phrases (e.g. `ลาก่อนค่ะ`) so they stay crisp, not rushed or sleepy
            eff_speed = float(speed) * (0.96 if len(all_ids) <= 22 else 1.0)
            eff_speed = max(0.75, min(1.50, eff_speed))

            id_chunks: list[list[int]] = []
            start = 0
            total = len(all_ids)
            while start < total:
                end = min(total, start + chunk_limit)
                if end < total and space_id is not None:
                    cut = end
                    while cut > start + 120 and all_ids[cut - 1] != space_id:
                        cut -= 1
                    if cut > start + 120:
                        end = cut
                id_chunks.append(all_ids[start:end])
                start = end

            sub_parts: list = []
            for ch_ids in id_chunks:
                arr = self._run_onnx_ids(ch_ids, eff_speed)
                if arr.size > 0:
                    sub_parts.append(arr)

            if not sub_parts:
                continue
            if len(sub_parts) == 1:
                clause_wavs.append(sub_parts[0])
            else:
                stitched_sub: list = []
                for si, sp in enumerate(sub_parts):
                    if si > 0:
                        stitched_sub.append(chunk_gap)
                    stitched_sub.append(sp)
                clause_wavs.append(np.concatenate(stitched_sub))

        if not clause_wavs:
            return np.zeros(0, dtype=np.float32)

        final_parts: list = []
        for idx, cwav in enumerate(clause_wavs):
            if idx > 0:
                final_parts.append(clause_gap)
            final_parts.append(cwav)
        final_parts.append(tail_pad)
        return np.concatenate(final_parts)


def _resolve_paradee_model_files_sync() -> tuple[str, str]:
    """Returns `(model_path, config_path)` from pre-baked local dir, or downloads via huggingface_hub."""
    if has_baked_paradee_model():
        return (
            os.path.join(TTS_PARADEE_MODEL_DIR, "paradee_thai_int8.onnx"),
            os.path.join(TTS_PARADEE_MODEL_DIR, "config.json"),
        )
    from huggingface_hub import hf_hub_download
    m_path = hf_hub_download("pythainlp/MakHamTTS", "paradee_thai/paradee_thai_int8.onnx")
    c_path = hf_hub_download("pythainlp/MakHamTTS", "paradee_thai/config.json")
    return m_path, c_path


def _load_paradee_model_sync():
    """
    Lazily instantiates `_LowRamParadeeEngine` (Zero-Trie, Zero-Arena, Zero-Spin ONNX Runtime).
    Falls back to `pythaitts` wrapper ONLY if local `ipa.json` cannot be located.
    """
    global _PARADEE_TTS_MODEL, _PARADEE_LAST_USED_TS
    with _PARADEE_LOCK:
        if _PARADEE_TTS_MODEL is not None:
            _PARADEE_LAST_USED_TS = time.time()
            return _PARADEE_TTS_MODEL

        model_path, config_path = _resolve_paradee_model_files_sync()
        engine = _LowRamParadeeEngine(
            model_path=model_path,
            config_path=config_path,
            custom_dict=dict(YUKINO_PARADEE_CUSTOM_DICT),
        )
        if time.time() < state.paradee_tts_cooldown_until:
            del engine
            release_unused_memory()
            raise RuntimeError("Paradee init finished after timeout cooldown; released to save RAM")
        _PARADEE_TTS_MODEL = engine
        _PARADEE_LAST_USED_TS = time.time()
        state.paradee_tts_loaded = True
        return _PARADEE_TTS_MODEL


def unload_paradee_tts(reason: str = "") -> bool:
    """
    Explicitly releases the in-RAM `_LowRamParadeeEngine` ONNX session & flat IPA dict and
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
    No-op kept for API compatibility: intentionally does NOT pre-load during `!join` DAVE
    handshake so voice connection & greeting stay instant and never spawn competing loader threads.
    """
    return


def _waveform_to_wav_bytes_sync(raw_out, default_sr: int = 24000) -> bytes:
    """
    Normalizes any waveform output (`bytes`, `(waveform, sr)` tuple, or `np.ndarray`)
    into a valid 16-bit mono PCM WAV buffer in RAM (zero temp files).
    """
    if raw_out is None:
        return b""
    if isinstance(raw_out, (bytes, bytearray)):
        raw_b = bytes(raw_out)
        if raw_b[:4] == b"RIFF":
            return raw_b
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
    """Thread-safe offline Thai synthesis via `_LowRamParadeeEngine` returning 24 kHz WAV bytes."""
    global _PARADEE_LAST_USED_TS
    model = _load_paradee_model_sync()
    speed = max(0.75, min(1.50, float(TTS_PARADEE_SPEED)))
    with _PARADEE_LOCK:
        _PARADEE_LAST_USED_TS = time.time()
        raw_out = model.synthesize_waveform(spoken_text, speed=speed)
    default_sr = int(getattr(model, "sample_rate", None) or 24000)
    return _waveform_to_wav_bytes_sync(raw_out, default_sr=default_sr)


async def _paradee_tts_with_mastering(
    spoken_text: str, output_format: str = "pcm"
) -> bytes:
    """
    Synthesizes Thai / Thai-English speech offline using `_LowRamParadeeEngine`:
    - Serialized by `_PARADEE_ASYNC_LOCK` so concurrent turns never spawn duplicate loader threads.
    - Hard `asyncio.wait_for(..., timeout=TTS_PARADEE_TIMEOUT_SEC)` guarantees instant fallback to
      Google TTS if Render 0.1 vCPU ever stalls (never silences Yukino or blocks Discord heartbeats!).
    - Native speed via `TTS_PARADEE_SPEED` (default `1.05`) + tempo-compensated cool pitch
      via `TTS_PARADEE_PITCH` (default `0.94`) + crisp cool EQ (no muffled treble cut).
    """
    ok, skip_reason = _can_safely_load_paradee_now()
    if not ok:
        if skip_reason and skip_reason != "cooldown":
            api_log(f"Paradee TTS bypassed ({skip_reason}); falling back to Google TTS for this turn")
        return b""

    t0 = time.monotonic()
    timeout_sec = float(TTS_PARADEE_TIMEOUT_SEC) if _PARADEE_TTS_MODEL is None else min(8.0, float(TTS_PARADEE_TIMEOUT_SEC))
    try:
        async with _get_paradee_async_lock():
            ok_inner, skip_inner = _can_safely_load_paradee_now()
            if not ok_inner:
                if skip_inner and skip_inner != "cooldown":
                    api_log(f"Paradee TTS bypassed ({skip_inner}); falling back to Google TTS")
                return b""
            raw_wav = await asyncio.wait_for(
                asyncio.to_thread(_synthesize_paradee_wav_sync, spoken_text),
                timeout=timeout_sec,
            )
    except asyncio.TimeoutError:
        state.paradee_tts_cooldown_until = time.time() + 300.0
        api_log(
            f"Paradee TTS timed out after {timeout_sec:.1f}s (RSS={_get_current_rss_mb():.1f} MB); "
            "cooling down 5m and falling back to Google TTS"
        )
        return b""
    except ImportError as e:
        state.paradee_tts_cooldown_until = time.time() + 3600.0
        api_log(f"Paradee TTS unavailable ({e}); cooling down 1h")
        return b""
    except Exception as e:
        state.paradee_tts_cooldown_until = time.time() + 300.0
        api_log(f"Paradee TTS synthesis error ({type(e).__name__}: {str(e)[:140]}); cooling down 5m")
        return b""

    if not raw_wav or len(raw_wav) <= 44:
        return b""

    pitch = max(0.82, min(1.15, float(TTS_PARADEE_PITCH)))
    af_parts: list[str] = []
    if abs(pitch - 1.0) >= 0.005:
        comp_atempo = max(0.50, min(2.00, 1.0 / pitch))
        af_parts.append(f"asetrate=24000*{pitch:.4f},aresample=48000,atempo={comp_atempo:.4f}")
    # Crisp cool Yukino timbre: subtle warmth + light upper-mid taming (no treble shelf so consonants stay sharp)
    af_parts.extend([
        "equalizer=f=220:t=q:w=1.0:g=1.0",
        "equalizer=f=2600:t=q:w=1.3:g=-2.0",
        "lowpass=f=9500",
    ])
    paradee_af = ",".join(af_parts)

    _schedule_paradee_idle_unload()

    try:
        mastered = await asyncio.to_thread(
            _master_tts_audio_sync,
            raw_wav,
            paradee_af,
            False,
            output_format,
            False,
        )
        if mastered:
            state.active_tts_engine = "paradee"
            api_log(
                f"Paradee TTS -> OK (speed={TTS_PARADEE_SPEED:.2f}x, pitch={pitch:.2f}, "
                f"{len(mastered)}B {output_format}, {time.monotonic() - t0:.2f}s, RSS={_get_current_rss_mb():.1f} MB)"
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
    apply_default_eq: bool = True,
) -> bytes:
    """
    Single-pass FFmpeg audio mastering chain tuned for Yukino's calm, cool kuudere voice:
    - `silenceremove` (Google TTS only): tightens dead gaps >160ms down to 70ms; disabled on Paradee to preserve natural clause breaths
    - `adelay=220:all=1`: prevents Discord Opus from clipping opening syllables
    - Default EQ (+1.5dB @ 240Hz, -3.0dB @ 3.2kHz) when `apply_default_eq=True`, or custom Paradee cool EQ via `extra_af`
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
    stages.append("adelay=220:all=1")
    if apply_default_eq:
        stages.extend([
            "equalizer=f=240:t=q:w=1.1:g=1.5",
            "equalizer=f=3200:t=q:w=1.2:g=-3.0",
        ])
    stages.extend([
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


async def synthesize_tts_audio_bytes(text: str, output_format: str = "pcm") -> bytes:
    """
    Consistent in-memory hybrid voice synthesis (zero temp files):
    - Default (`TTS_PRIMARY_ENGINE=paradee`):
      * Thai / Thai-English mixed: Offline `MakHamTTSParadee` 1.1 INT8 (speed=1.05, pitch=0.94)
        -> Google TTS zero-RAM fallback.
      * Pure English: Google TTS (`tl=en`) for natural English pronunciation.
    - Optional (`TTS_PRIMARY_ENGINE=google`): Google TTS (`tl=th` / `tl=en`).
    - `output_format="pcm"` returns 48kHz stereo 16-bit PCM ready for `discord.PCMAudio`.
    - `output_format="mp3"` returns 128kbps MP3 for Discord text-channel attachments.
    """
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return b""
    is_thai = bool(THAI_CHAR_REGEX.search(spoken_text))
    lang_code = "th" if is_thai else "en"

    if TTS_PRIMARY_ENGINE == "paradee" and is_thai:
        paradee_bytes = await _paradee_tts_with_mastering(spoken_text, output_format=output_format)
        if paradee_bytes:
            return paradee_bytes

    return await _google_tts_with_yukino_filter(spoken_text, lang_code, output_format=output_format)


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