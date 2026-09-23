"""Groq provider wrapper for backend.llm.client's fallback router."""
from typing import Optional

import groq as groq_sdk

from backend.llm import groq_client
from .errors import RetryableProviderError

MODELS = {"fast": groq_client.FAST_MODEL, "gen": groq_client.GEN_MODEL}


def chat(
    system_prompt: str,
    user_prompt: str,
    role: str = "gen",
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: Optional[int] = 1024,
) -> str:
    model = MODELS.get(role, MODELS["gen"])
    try:
        return groq_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
    except ValueError as e:
        # groq_client.get_client() raises ValueError when GROQ_API_KEY isn't set at all.
        raise RetryableProviderError(f"Groq not configured: {e}") from e
    except groq_sdk.AuthenticationError as e:
        raise RetryableProviderError(f"Groq authentication failed (expired/invalid key): {e}") from e
    except groq_sdk.RateLimitError as e:
        raise RetryableProviderError(f"Groq rate limited: {e}") from e
    except groq_sdk.APIStatusError as e:
        status = getattr(e, "status_code", None)
        if status is not None and status >= 500:
            raise RetryableProviderError(f"Groq server error ({status}): {e}") from e
        raise
