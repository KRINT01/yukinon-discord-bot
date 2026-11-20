import re
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any

import discord
from simpleeval import SimpleEval

from config import (
    DEFAULT_WEATHER_CITY,
    api_log,
)
from search_engine import (
    run_safe_web_search_async,
    fetch_webpage_async,
    fetch_live_weather_context,
)
from media_context import (
    extract_youtube_id,
    fetch_youtube_context_async,
)

MAX_TOOL_OUTPUT_CHARS = 7500
MAX_TOOLS_PER_TURN = 6


# ==============================================================================
# CANONICAL NATIVE FUNCTION-CALLING TOOL SPECIFICATIONS
# (Auto-compiled into both Google AI Studio & OpenAI/Groq/Cloudflare schemas)
# ==============================================================================
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "modes": ("general", "code", "voice"),
        "description": (
            "Search the live web (Google Grounding, News RSS, Wikipedia, Jikan MAL, Steam TH, Tavily, DuckDuckGo) "
            "for up-to-date facts, news, prices, game/anime details, software docs, or fact-checking."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "Concise, specific search query (prefer English keywords for technical/game/anime entities, Thai for local Thai news).",
                "required": True,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results (1 to 6, default 5).",
                "required": False,
            },
        },
    },
    {
        "name": "read_webpage",
        "modes": ("general", "code"),
        "description": "Download and extract the readable article text from a specific public HTTP/HTTPS URL (SSRF-protected).",
        "parameters": {
            "url": {
                "type": "string",
                "description": "Full public http:// or https:// URL to read.",
                "required": True,
            },
        },
    },
    {
        "name": "get_weather",
        "modes": ("general", "voice"),
        "description": (
            "Fetch live weather conditions, temperature, humidity, and 3-hour rain probability forecast "
            "for a city or district in Thailand (defaults to Thung Song Hong, Lak Si, Bangkok if omitted)."
        ),
        "parameters": {
            "location": {
                "type": "string",
                "description": "Thai province, district, or city name (leave empty for default location).",
                "required": False,
            },
        },
    },
    {
        "name": "youtube_inspect",
        "modes": ("general",),
        "description": "Extract video title, uploader, description, and subtitles/transcript from a YouTube URL or video ID.",
        "parameters": {
            "url": {
                "type": "string",
                "description": "Full YouTube video URL or 11-character video ID.",
                "required": True,
            },
        },
    },
    {
        "name": "calculate",
        "modes": ("general", "code", "voice"),
        "description": (
            "Evaluate mathematical expressions accurately (arithmetic, percentages, currency conversions, "
            "time/data sizing, powers). Always use this instead of doing mental arithmetic."
        ),
        "parameters": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression using digits and operators + - * / ** % ( ), e.g. '(4500 * 0.22) * 1.07'.",
                "required": True,
            },
        },
    },
    {
        "name": "manage_overtime",
        "modes": ("general", "voice"),
        "description": (
            "Log, update, delete, or display the monthly summary table of the user's Overtime (OT) work hours. "
            "Use whenever the user wants to record OT, edit OT, delete an OT day, or check their monthly OT table."
        ),
        "parameters": {
            "command": {
                "type": "string",
                "description": (
                    "Normalized Thai OT command string, e.g. 'จด OT แก้ระบบ e-Tax', "
                    "'จด OT 15/11/2569 18.00 - 21.00 Deploy K8s', 'สรุป OT', 'สรุป OT เดือนตุลาคม', or 'ลบ OT เมื่อวาน'."
                ),
                "required": True,
            },
        },
    },
    {
        "name": "manage_user_memory",
        "modes": ("general", "code", "voice"),
        "description": (
            "Read or update the current speaker's persistent long-term profile stored in #yukinon-memory. "
            "Use 'pin_fact' for permanent personal preferences/facts the user explicitly asks to remember or clearly states; "
            "use 'add_note' for short rolling observations; use 'view' to inspect saved memory."
        ),
        "parameters": {
            "action": {
                "type": "string",
                "enum": ["view", "pin_fact", "add_note", "add_topic", "delete_fact", "delete_note"],
                "description": "Memory operation to perform.",
                "required": True,
            },
            "value": {
                "type": "string",
                "description": "Short concise fact, note, topic name, or keyword/index to delete (not needed for 'view').",
                "required": False,
            },
        },
    },
    {
        "name": "generate_image",
        "modes": ("general",),
        "description": (
            "Queue an AI image/illustration generation via Cloudflare FLUX or Leonardo Phoenix "
            "when the user explicitly asks to draw, create, or generate a new picture."
        ),
        "parameters": {
            "prompt": {
                "type": "string",
                "description": "Detailed, literal English visual description of the image to generate.",
                "required": True,
            },
            "style": {
                "type": "string",
                "enum": ["default", "anime", "hq", "fl2", "fl2hq"],
                "description": "Optional rendering model/style preset (default = FLUX.1-schnell, anime = 2D anime style, hq = Leonardo Phoenix, fl2/fl2hq = FLUX.2 Klein).",
                "required": False,
            },
        },
    },
    {
        "name": "add_reaction",
        "modes": ("general",),
        "description": "Add a single subtle Unicode emoji reaction to the user's Discord message when emotionally fitting (e.g., snowflake, paw prints, panda, tea).",
        "parameters": {
            "emoji": {
                "type": "string",
                "description": "A single standard Unicode emoji character.",
                "required": True,
            },
        },
    },
]

_GEMINI_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "object": "OBJECT",
    "array": "ARRAY",
}


def build_gemini_tool_declarations(mode: str = "general") -> list[dict[str, Any]]:
    """Builds Google AI Studio `functionDeclarations` list filtered for the requested mode."""
    decls: list[dict[str, Any]] = []
    for spec in TOOL_SPECS:
        if mode not in spec["modes"]:
            continue
        props: dict[str, Any] = {}
        req: list[str] = []
        for p_name, p_cfg in spec["parameters"].items():
            entry: dict[str, Any] = {
                "type": _GEMINI_TYPE_MAP.get(p_cfg["type"], "STRING"),
                "description": p_cfg["description"],
            }
            if "enum" in p_cfg:
                entry["enum"] = list(p_cfg["enum"])
            props[p_name] = entry
            if p_cfg.get("required"):
                req.append(p_name)

        fn_decl: dict[str, Any] = {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": {
                "type": "OBJECT",
                "properties": props,
            },
        }
        if req:
            fn_decl["parameters"]["required"] = req
        decls.append(fn_decl)
    return decls


def build_openai_tool_declarations(mode: str = "general") -> list[dict[str, Any]]:
    """Builds OpenAI-compatible `tools` list for Groq & Cloudflare Workers AI."""
    tools_out: list[dict[str, Any]] = []
    for spec in TOOL_SPECS:
        if mode not in spec["modes"]:
            continue
        props: dict[str, Any] = {}
        req: list[str] = []
        for p_name, p_cfg in spec["parameters"].items():
            entry: dict[str, Any] = {
                "type": p_cfg["type"],
                "description": p_cfg["description"],
            }
            if "enum" in p_cfg:
                entry["enum"] = list(p_cfg["enum"])
            props[p_name] = entry
            if p_cfg.get("required"):
                req.append(p_name)

        params_obj: dict[str, Any] = {
            "type": "object",
            "properties": props,
        }
        if req:
            params_obj["required"] = req

        tools_out.append({
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": params_obj,
            },
        })
    return tools_out


GEMINI_TOOLS_BY_MODE = {
    m: build_gemini_tool_declarations(m) for m in ("general", "code", "voice")
}
OPENAI_TOOLS_BY_MODE = {
    m: build_openai_tool_declarations(m) for m in ("general", "code", "voice")
}
VALID_TOOL_NAMES = {s["name"] for s in TOOL_SPECS}


@dataclass
class ToolExecutionContext:
    """Mutable per-turn execution context shared between the LLM tool loop and bot/voice callers."""

    channel_id: int | None = None
    guild: discord.Guild | None = None
    channel: Any = None
    speaker_user: Any = None
    message: discord.Message | None = None
    mode: str = "general"  # "general" | "code" | "voice"

    # Accumulated side effects & telemetry
    did_search: bool = False
    did_weather: bool = False
    generated_images: list[str] = field(default_factory=list)
    reactions: list[str] = field(default_factory=list)
    ot_reply_text: str = ""
    ot_reply_meta: tuple[str, str, str] | None = None
    memory_updated_uid: str = ""
    tools_used: list[str] = field(default_factory=list)
    _seen_call_keys: set[str] = field(default_factory=set)
    _total_calls: int = 0


def _safe_parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return {}


class _ToolProxyMessage:
    """Lightweight message proxy for OT/Memory handlers invoked from voice or slash interactions."""

    def __init__(self, author, channel, guild, content: str = ""):
        self.author = author
        self.channel = channel
        self.guild = guild
        self.content = content


async def execute_tool_call(
    tool_name: str,
    raw_args: Any,
    ctx: ToolExecutionContext,
) -> str:
    """Executes a single native function call safely, recording side effects on `ctx`."""
    t_name = (tool_name or "").strip().lower()
    args = _safe_parse_tool_args(raw_args)

    if t_name not in VALID_TOOL_NAMES:
        return f"[Tool Error: Unknown tool '{t_name}'.]"

    if ctx._total_calls >= MAX_TOOLS_PER_TURN:
        return "[Tool Limit Reached: Summarize and answer the user now with the data already gathered.]"

    dedup_key = f"{t_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    if dedup_key in ctx._seen_call_keys:
        return f"[Tool Notice: '{t_name}' was already called with identical arguments in this turn. Reuse its earlier result.]"
    ctx._seen_call_keys.add(dedup_key)
    ctx._total_calls += 1

    t0 = time.monotonic()
    try:
        if t_name == "web_search":
            query = str(args.get("query") or "").strip()[:180]
            if not query:
                return "[Tool Error: 'query' is required.]"
            max_res = max(1, min(6, int(args.get("max_results") or 5)))
            res = await run_safe_web_search_async(query, max_results=max_res)
            if res:
                ctx.did_search = True
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> web_search('{query[:45]}') ({time.monotonic() - t0:.2f}s)")
            return (res or f"[No web search results found for '{query}'.]")[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "read_webpage":
            url = str(args.get("url") or "").strip()
            if not url:
                return "[Tool Error: 'url' is required.]"
            text = await fetch_webpage_async(url, max_chars=6500, timeout_sec=6.0)
            ctx.did_search = True
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> read_webpage('{url[:55]}') ({time.monotonic() - t0:.2f}s)")
            return f'<webpage_content url="{url}">\n{text}\n</webpage_content>'[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "get_weather":
            loc = str(args.get("location") or DEFAULT_WEATHER_CITY).strip()
            res = await fetch_live_weather_context(loc, channel_id=ctx.channel_id)
            if res:
                ctx.did_weather = True
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> get_weather('{loc[:35]}') ({time.monotonic() - t0:.2f}s)")
            return (res or f"[Could not retrieve weather for '{loc}'.]")[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "youtube_inspect":
            url = str(args.get("url") or "").strip()
            if not url:
                return "[Tool Error: 'url' is required.]"
            vid = extract_youtube_id(url) or url[:11]
            res = await fetch_youtube_context_async(vid, max_chars=7000)
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> youtube_inspect('{vid}') ({time.monotonic() - t0:.2f}s)")
            return (res or f"[Could not inspect YouTube video '{url}'.]")[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "calculate":
            expr = str(args.get("expression") or "").strip()
            expr_clean = re.sub(r"[^0-9+\-*/().%\s^]", "", expr).replace("^", "**").strip()
            if not expr_clean:
                return "[Tool Error: Invalid or empty math expression.]"
            val = SimpleEval().eval(expr_clean)
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> calculate('{expr_clean}') = {val}")
            return f"Calculation Result: {expr_clean} = {val}"

        if t_name == "manage_overtime":
            cmd = str(args.get("command") or "").strip()
            if not cmd:
                return "[Tool Error: 'command' is required.]"
            if ctx.speaker_user is None:
                return "[Tool Error: Speaker user context unavailable for OT logging.]"
            from user_memory import process_ot_request

            proxy = ctx.message or _ToolProxyMessage(
                ctx.speaker_user,
                ctx.channel or type("_Ch", (), {"id": ctx.channel_id or 0})(),
                ctx.guild,
                content=cmd,
            )
            ot_reply = await process_ot_request(proxy, cmd)
            ctx.ot_reply_text = ot_reply
            m_meta = re.search(
                r"\[OT_RECORD:\s*uid=\d+\s*\|\s*date=([0-9/\-]+)\s*\|\s*slot=([0-9:.]+-[0-9:.]+)\s*\|\s*hours=[0-9.]+\s*\|\s*detail=(.+?)\]",
                ot_reply or "",
            )
            if m_meta:
                ctx.ot_reply_meta = (
                    m_meta.group(1).strip(),
                    m_meta.group(2).strip(),
                    m_meta.group(3).strip(),
                )
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> manage_overtime('{cmd[:40]}') ({time.monotonic() - t0:.2f}s)")
            return (
                "Overtime action completed. Include the exact recorded date, time slot, hours, and summary below in your response:\n"
                + ot_reply
            )[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "manage_user_memory":
            action = str(args.get("action") or "view").strip().lower()
            value = str(args.get("value") or "").strip()
            if ctx.speaker_user is None:
                return "[Tool Error: Speaker user context unavailable for memory operations.]"
            from user_memory import apply_tool_memory_action

            res_text, target_uid = await apply_tool_memory_action(ctx.speaker_user, action, value)
            if action != "view" and target_uid:
                ctx.memory_updated_uid = target_uid
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> manage_user_memory({action}, '{value[:35]}') ({time.monotonic() - t0:.2f}s)")
            return res_text[:MAX_TOOL_OUTPUT_CHARS]

        if t_name == "generate_image":
            prompt = str(args.get("prompt") or "").strip()
            style = str(args.get("style") or "default").strip().lower()
            if not prompt:
                return "[Tool Error: 'prompt' is required.]"
            flag_map = {
                "anime": "--anime ",
                "hq": "--hq ",
                "fl2": "--fl2 ",
                "fl2hq": "--fl2hq ",
            }
            full_p = f"{flag_map.get(style, '')}{prompt}"
            if len(ctx.generated_images) < 2:
                ctx.generated_images.append(full_p)
            ctx.tools_used.append(t_name)
            api_log(f"ToolExec -> generate_image(style={style}, '{prompt[:45]}')")
            return (
                f"[Image generation queued for prompt: '{full_p}'. "
                "The rendered image will be automatically attached right after your text reply — "
                "tell the user briefly in Yukino's voice that you drew it for them.]"
            )

        if t_name == "add_reaction":
            emoji = str(args.get("emoji") or "").strip().split()[0] if args.get("emoji") else ""
            if emoji and len(emoji) <= 8 and emoji not in ctx.reactions and len(ctx.reactions) < 2:
                ctx.reactions.append(emoji)
            ctx.tools_used.append(t_name)
            return f"[Reaction '{emoji}' queued for the user's message.]"

        return f"[Tool Error: Unhandled tool '{t_name}'.]"
    except Exception as e:
        api_log(f"ToolExec -> {t_name} FAILED ({type(e).__name__}: {str(e)[:100]})")
        return f"[Tool Execution Error in '{t_name}': {type(e).__name__}: {e}]"


async def execute_tool_calls_parallel(
    calls: list[tuple[str, Any]],
    ctx: ToolExecutionContext,
) -> list[str]:
    """Executes a batch of requested tool calls concurrently while preserving order."""
    if not calls:
        return []
    if len(calls) == 1:
        return [await execute_tool_call(calls[0][0], calls[0][1], ctx)]

    tasks = [execute_tool_call(name, raw_args, ctx) for name, raw_args in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            out.append(f"[Tool Execution Exception: {type(r).__name__}: {r}]")
        else:
            out.append(str(r))
    return out