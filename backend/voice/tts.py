"""
Thin client for the local IndicF5 TTS service (see tts_service/server.py).

The main project runs on Python 3.14; IndicF5 needs Python 3.10 with its own
dependency stack, so it runs as a separate local process (its own venv) exposing
one HTTP endpoint. This module just calls that endpoint -- it does not import
torch/transformers/IndicF5 itself, so the main backend stays free of that
dependency weight and version conflicts.

Before this works: `tts_service/server.py` must be running
(tts_service\\.venv\\Scripts\\python.exe -m uvicorn server:app --port 8100),
and tts_service/reference_audio/manifest.json must have real reference clips
configured for the languages you want to synthesize (see that folder's README).
"""
import os

import requests

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8100")


class TTSServiceUnavailable(Exception):
    """Raised when the local IndicF5 service isn't reachable or isn't ready."""


def is_available() -> bool:
    try:
        resp = requests.get(f"{TTS_SERVICE_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def synthesize(text: str, language: str = "hindi") -> bytes:
    """
    Calls the local IndicF5 service and returns synthesized speech as WAV bytes.
    Raises TTSServiceUnavailable if the service isn't running/reachable, or
    requests.HTTPError for a request the service itself rejected (e.g. no
    reference audio configured for that language).
    """
    try:
        resp = requests.post(
            f"{TTS_SERVICE_URL}/synthesize",
            json={"text": text, "language": language},
            timeout=60,
        )
    except requests.RequestException as e:
        raise TTSServiceUnavailable(
            f"Could not reach the TTS service at {TTS_SERVICE_URL}. Is it running? "
            f"(tts_service\\.venv\\Scripts\\python.exe -m uvicorn server:app --port 8100) "
            f"Underlying error: {e}"
        ) from e

    resp.raise_for_status()
    return resp.content
