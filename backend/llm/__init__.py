"""
LLM integration package.

backend.llm.client.chat() is the canonical entry point -- provider-agnostic, with
automatic fallback across configured providers (see backend/llm/client.py). Prefer
`from backend.llm.client import chat` over reaching into a specific provider module.
"""
from .client import chat

__all__ = ["chat"]
