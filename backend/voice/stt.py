"""
Local speech-to-text via faster-whisper (github.com/SYSTRAN/faster-whisper).

Model weights (medium, ~1.5GB) auto-download from Hugging Face on first use and
are cached locally -- no API key, fully local. Records from the default
microphone via sounddevice; recording stops when you press Enter (simplest
reliable approach for a CLI loop -- no voice-activity-detection tuning needed).
"""
import queue
import sys
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
SAMPLE_RATE = 16000  # Whisper's expected input sample rate

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    """Loads and caches the faster-whisper model (int8 on CPU for speed)."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def record_until_enter() -> np.ndarray:
    """
    Records from the default microphone until the user presses Enter.
    Returns mono float32 audio at SAMPLE_RATE, shape (n_samples,).
    """
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    print("Recording... press Enter to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    chunks: List[np.ndarray] = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).flatten()


def transcribe_audio(audio: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Transcribes float32 mono audio at SAMPLE_RATE. Returns (text, detected_language_code).
    `language` can force a language (ISO 639-1, e.g. "hi"); omit to auto-detect.
    """
    model = get_model()
    segments, info = model.transcribe(audio, language=language, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip(), info.language


def listen_and_transcribe(language: Optional[str] = None) -> Tuple[str, str]:
    """Records from the mic until Enter, then transcribes. Convenience wrapper."""
    audio = record_until_enter()
    if audio.size == 0:
        return "", language or "unknown"
    return transcribe_audio(audio, language=language)
