"""Gemini provider wrapper for backend.llm.client's fallback router."""
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .errors import RetryableProviderError

load_dotenv()

# "*-latest" aliases rather than pinned version numbers: Gemini model IDs churn fast
# (verified directly against the API -- "gemini-2.5-flash" is already retired for new
# users as of this writing) and the aliases are what Google keeps pointed at a current,
# non-deprecated model.
MODELS = {
    "fast": os.getenv("GEMINI_FAST_MODEL", "gemini-flash-latest"),
    "gen": os.getenv("GEMINI_GEN_MODEL", "gemini-pro-latest"),
}

_client: Optional["genai.Client"] = None


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RetryableProviderError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


def chat(
    system_prompt: str,
    user_prompt: str,
    role: str = "gen",
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: Optional[int] = 1024,
) -> str:
    client = _get_client()
    model = MODELS.get(role, MODELS["gen"])

    config_kwargs = {
        "system_instruction": system_prompt,
        "temperature": temperature,
    }
    if max_tokens:
        config_kwargs["max_output_tokens"] = max_tokens
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except genai_errors.ServerError as e:
        raise RetryableProviderError(f"Gemini server error/overloaded: {e}") from e
    except genai_errors.ClientError as e:
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        message = str(e).lower()
        if status in (401, 403, 429) or "expired" in message or "quota" in message or "permission" in message:
            raise RetryableProviderError(f"Gemini client error ({status}): {e}") from e
        raise

    return (response.text or "").strip()
