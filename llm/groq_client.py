import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

FAST_MODEL = "llama-3.1-8b-instant"       # intent parsing, cheap/fast tasks
GEN_MODEL  = "llama-3.3-70b-versatile"    # answer generation

_client = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client

def chat(system_prompt: str, user_prompt: str, model: str = GEN_MODEL,
          temperature: float = 0.3, json_mode: bool = False) -> str:
    client = get_client()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content
