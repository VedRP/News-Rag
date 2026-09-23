"""
Provider-agnostic LLM entry point with automatic fallback.

Every other module in the codebase should call backend.llm.client.chat() instead of
reaching into a specific provider (backend.llm.groq_client / backend.llm.providers.*
directly) -- this is what makes the LLM vendor actually swappable at runtime, not just
at config time, per context.md's "keep the LLM vendor swappable" cost philosophy.

chat() tries each configured provider in order (default: Gemini, then Groq) and moves
to the next only on a RetryableProviderError (expired/invalid key, rate limit, server
overload) -- confirmed necessary firsthand during Phase 6 development, when the
project's Groq key expired mid-session and Gemini's free tier hit transient 503
"high demand" overload errors within the same few minutes.

Configure provider priority via LLM_PROVIDER_ORDER in .env, e.g. "groq,gemini".
"""
import os
from typing import List, Optional

from .providers import gemini_provider, groq_provider
from .providers.errors import RetryableProviderError

_PROVIDERS = {"groq": groq_provider, "gemini": gemini_provider}
DEFAULT_PROVIDER_ORDER = ["gemini", "groq"]


def _provider_order() -> List[str]:
    raw = os.getenv("LLM_PROVIDER_ORDER")
    if not raw:
        return DEFAULT_PROVIDER_ORDER
    names = [p.strip().lower() for p in raw.split(",") if p.strip()]
    known = [n for n in names if n in _PROVIDERS]
    return known or DEFAULT_PROVIDER_ORDER


def chat(
    system_prompt: str,
    user_prompt: str,
    role: str = "gen",
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: Optional[int] = 1024,
) -> str:
    """
    role: "fast" for cheap/low-latency calls (intent parsing, translation), "gen" for
    the higher-quality grounded-generation model. Each provider maps this to its own
    actual model ID (Groq and Gemini don't share model names).
    """
    last_error: Optional[Exception] = None
    for name in _provider_order():
        provider = _PROVIDERS[name]
        try:
            return provider.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                role=role,
                temperature=temperature,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )
        except RetryableProviderError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"All configured LLM providers failed ({', '.join(_provider_order())}). "
        f"Last error: {last_error}"
    )
