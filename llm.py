"""
Backward-compatible LLM facade re-exporting the split multi-cloud modules:
- `llm_providers.py`      : Google AI Studio (Hydrogen, Flash-Lite, Gemma 4), Cloudflare Workers AI, Groq LPU
- `decision_orchestrator.py`: 4-Phase `gemini-decision-1` + `@cf/cloudflare/clef-flash` fallback orchestrator
- `chat_pipeline.py`      : Unified `query_ai_chat()` pipeline with prefix caching, native tools, and passive memory capture
"""

from llm_providers import (  # noqa: F401
    get_google_model_cooldown,
    set_google_model_cooldown,
    is_google_ai_studio_congested,
    _fetch_google_ai_studio_model_names,
    discover_available_groq_models,
    get_groq_search_and_chat_candidates,
    discover_active_groq_vision_models,
    discover_active_gemma_models,
    _discover_active_gemini_flash_model,
    call_gemini_hydrogen_generate,
    call_gemini_hydrogen_chat,
    call_gemma_ai_studio_chat,
    call_gemma_ai_studio_vision,
    call_gemini_generate_content_self_healing,
    call_gemini_chat_messages,
    call_cloudflare_gemma_chat,
    _call_cf_gemma_hedged_with_groq,
    call_cloudflare_gpt_oss,
    call_cloudflare_vision_chat,
    call_cloudflare_vision_fallback_qwen,
    call_cloudflare_multi_image_vision_qwen,
    call_groq_vision_chat,
    call_groq_multi_image_vision,
    call_groq_chat,
)
from decision_orchestrator import (  # noqa: F401
    MAX_NATIVE_INLINE_PDF_BYTES,
    call_gemini_decision_orchestrator,
    orchestrate_message_with_decision_1,
    orchestrate_voice_utterance_with_decision_1,
    check_voice_scribe_stop_with_decision_1,
    filter_search_snippets_with_decision_1,
    arbitrate_fact_check_with_decision_1,
    triage_image_with_decision_1,
    triage_pdf_with_decision_1,
    extract_scanned_pdf_with_gemini_native,
)
from chat_pipeline import (  # noqa: F401
    query_ai_chat,
)