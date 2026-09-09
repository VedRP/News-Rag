import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GEN_MODEL = os.getenv("GROQ_GEN_MODEL", "openai/gpt-oss-120b")

_client: Optional[Groq] = None

def get_client() -> Groq:
    """Returns a singleton Groq client instance, raising an error if no API key is found."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Please set GROQ_API_KEY in your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client

def chat(
    system_prompt: str,
    user_prompt: str,
    model: str = GEN_MODEL,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: Optional[int] = 1024,
) -> str:
    """
    Sends a chat completion request to the Groq API.
    
    Args:
        system_prompt: Guiding instructions for the model.
        user_prompt: User input or formatted context query.
        model: Target model ID on Groq.
        temperature: Sampling temperature (low for factual/grounded generation).
        json_mode: When True, enforces strict JSON object output.
        max_tokens: Maximum tokens in response.
        
    Returns:
        Generated text string.
    """
    client = get_client()
    kwargs: Dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""
