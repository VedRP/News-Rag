"""
Local text-to-speech via Piper (github.com/OHF-Voice/piper1-gpl), replacing the
earlier IndicF5 HTTP-service approach per user direction. Piper is a fast,
single-pass ONNX model -- measured ~0.15s synthesis for a short sentence on this
machine's CPU, vs. IndicF5's ~4.5 minutes -- and runs directly in the main
environment, so no separate Python version/venv/HTTP service is needed.

Voices are downloaded once via:
    python -m piper.download_voices hi_IN-pratham-medium mr_IN-google-medium en_US-lessac-medium
into backend/voice/piper_voices/ (see VOICE_FILES below for the exact mapping).
"""
import io
import wave
from pathlib import Path
from typing import Dict, Optional

from piper import PiperVoice

VOICES_DIR = Path(__file__).resolve().parent / "piper_voices"

VOICE_FILES: Dict[str, str] = {
    "english": "en_US-lessac-medium.onnx",
    "hindi": "hi_IN-pratham-medium.onnx",
    "marathi": "mr_IN-google-medium.onnx",
}


class VoiceNotAvailable(Exception):
    """Raised when no Piper voice file is present for the requested language."""


_loaded_voices: Dict[str, "PiperVoice"] = {}


def _get_voice(language: str) -> "PiperVoice":
    language = (language or "english").lower()
    if language not in _loaded_voices:
        filename = VOICE_FILES.get(language)
        if filename is None:
            raise VoiceNotAvailable(
                f"No Piper voice configured for language {language!r}. "
                f"Configured languages: {sorted(VOICE_FILES.keys())}"
            )
        voice_path = VOICES_DIR / filename
        if not voice_path.exists():
            raise VoiceNotAvailable(
                f"Voice file not found: {voice_path}. Download it with: "
                f"python -m piper.download_voices {filename.removesuffix('.onnx')}"
            )
        _loaded_voices[language] = PiperVoice.load(str(voice_path))
    return _loaded_voices[language]


def is_available(language: Optional[str] = None) -> bool:
    """Checks whether a voice file exists for `language` (or any configured language if None)."""
    languages = [language] if language else list(VOICE_FILES.keys())
    return all((VOICES_DIR / VOICE_FILES[lang]).exists() for lang in languages if lang in VOICE_FILES)


def synthesize(text: str, language: str = "hindi") -> bytes:
    """Synthesizes `text` in `language` and returns WAV bytes."""
    voice = _get_voice(language)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()
