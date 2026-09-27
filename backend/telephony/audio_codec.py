"""
Audio format conversion between Twilio's telephony audio (8kHz mono mu-law, per
Twilio Media Streams: https://www.twilio.com/docs/voice/media-streams) and the
formats our existing STT (faster-whisper -- 16kHz mono float32) and TTS (Piper --
WAV at each voice's own native sample rate) already use.

Python's stdlib `audioop` module (mu-law codec + sample-rate conversion) was
removed in Python 3.13. `audioop-lts` (in requirements.txt) is the official
drop-in replacement package and provides the same API under the same name.
"""
import audioop
import io
import wave

import numpy as np

TWILIO_SAMPLE_RATE = 8000  # Twilio Media Streams' fixed audio format
_PCM16_WIDTH = 2  # bytes per sample; audioop works in linear PCM internally


def twilio_mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decodes 8kHz mu-law bytes (as received from Twilio) to 8kHz 16-bit PCM bytes."""
    return audioop.ulaw2lin(mulaw_bytes, _PCM16_WIDTH)


def pcm16_to_twilio_mulaw(pcm16_bytes: bytes) -> bytes:
    """Encodes 8kHz 16-bit PCM bytes to 8kHz mu-law bytes (for sending back to Twilio)."""
    return audioop.lin2ulaw(pcm16_bytes, _PCM16_WIDTH)


def frame_rms(pcm16_bytes: bytes) -> int:
    """RMS loudness of a 16-bit PCM frame -- used for simple energy-based silence detection."""
    if not pcm16_bytes:
        return 0
    return audioop.rms(pcm16_bytes, _PCM16_WIDTH)


def twilio_audio_to_whisper_input(mulaw_bytes: bytes) -> np.ndarray:
    """
    Converts buffered 8kHz mu-law bytes (accumulated from Twilio 'media' events) into
    16kHz mono float32, exactly what backend.voice.stt.transcribe_audio() expects.
    """
    pcm16_8k = twilio_mulaw_to_pcm16(mulaw_bytes)
    pcm16_16k, _ = audioop.ratecv(pcm16_8k, _PCM16_WIDTH, 1, TWILIO_SAMPLE_RATE, 16000, None)
    return np.frombuffer(pcm16_16k, dtype=np.int16).astype(np.float32) / 32768.0


def wav_bytes_to_twilio_mulaw(wav_bytes: bytes) -> bytes:
    """
    Downconverts a WAV file (as produced by backend.voice.tts.synthesize(), at
    that voice's own native sample rate/channel count) to 8kHz mono mu-law bytes
    ready to stream back to Twilio.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    if channels == 2:
        pcm_bytes = audioop.tomono(pcm_bytes, sample_width, 0.5, 0.5)
    if sample_width != _PCM16_WIDTH:
        pcm_bytes = audioop.lin2lin(pcm_bytes, sample_width, _PCM16_WIDTH)
    if frame_rate != TWILIO_SAMPLE_RATE:
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, _PCM16_WIDTH, 1, frame_rate, TWILIO_SAMPLE_RATE, None)

    return pcm16_to_twilio_mulaw(pcm_bytes)
