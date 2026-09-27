"""
Handles one live phone call's bidirectional audio over Twilio Media Streams
(https://www.twilio.com/docs/voice/media-streams/websocket-messages) -- the
WebSocket protocol Twilio speaks once a call is connected via <Connect><Stream>.

Message flow from Twilio: "connected" -> "start" -> many "media" (base64 8kHz
mu-law, ~20ms/frame) -> "stop". We buffer inbound media and use simple
energy-based silence detection (no "press Enter"/manual cue is possible on a
real phone call) to decide when the caller has finished a turn, then run the
same STT -> handle_turn() -> TTS pipeline server.py's /voice endpoint uses, and
stream the answer back as outbound "media" events on the same connection.
"""
import base64
import json
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

from backend.backend_validation import normalize_language
from backend.conversation.pipeline import handle_turn
from backend.conversation.state import SessionState, new_session
from backend.telephony.audio_codec import (
    frame_rms,
    pcm16_to_twilio_mulaw,
    twilio_audio_to_whisper_input,
    twilio_mulaw_to_pcm16,
    wav_bytes_to_twilio_mulaw,
)
from backend.voice import stt, tts

logger = logging.getLogger("twilio_stream")

# Tuned for typical phone-line background noise vs. speech on 16-bit PCM; not
# scientifically derived -- adjust if real calls trigger too early/late.
SILENCE_RMS_THRESHOLD = 400
SILENCE_DURATION_MS = 900  # how long a caller must stop talking before we act
FRAME_DURATION_MS = 20  # Twilio sends one "media" event per 20ms of audio
MAX_UTTERANCE_MS = 20000  # safety cap so a stuck-open mic can't grow the buffer forever
OUTBOUND_CHUNK_BYTES = 160  # 20ms of 8kHz 8-bit mu-law audio per outbound frame


class TwilioCallSession:
    """Per-call state: the caller's audio buffer, silence timer, and conversation session."""

    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.stream_sid: Optional[str] = None
        self.state: SessionState = new_session(call_sid)
        self._mulaw_buffer = bytearray()
        self._silence_run_ms = 0
        self._speech_detected = False
        self._buffered_ms = 0

    def accept_frame(self, mulaw_frame: bytes) -> bool:
        """
        Feeds one decoded inbound audio frame in. Returns True when the caller has
        just finished an utterance (buffer is ready to transcribe).
        """
        pcm16 = twilio_mulaw_to_pcm16(mulaw_frame)
        loud = frame_rms(pcm16) > SILENCE_RMS_THRESHOLD

        self._mulaw_buffer.extend(mulaw_frame)
        self._buffered_ms += FRAME_DURATION_MS

        if loud:
            self._speech_detected = True
            self._silence_run_ms = 0
        elif self._speech_detected:
            self._silence_run_ms += FRAME_DURATION_MS

        utterance_done = self._speech_detected and (
            self._silence_run_ms >= SILENCE_DURATION_MS or self._buffered_ms >= MAX_UTTERANCE_MS
        )
        return utterance_done

    def take_utterance(self) -> bytes:
        """Pops the buffered audio for processing and resets for the next turn."""
        audio = bytes(self._mulaw_buffer)
        self._mulaw_buffer.clear()
        self._silence_run_ms = 0
        self._speech_detected = False
        self._buffered_ms = 0
        return audio


async def _send_audio(websocket: WebSocket, stream_sid: str, mulaw_bytes: bytes) -> None:
    """Streams mu-law audio back to Twilio as a sequence of outbound 'media' events."""
    for offset in range(0, len(mulaw_bytes), OUTBOUND_CHUNK_BYTES):
        chunk = mulaw_bytes[offset : offset + OUTBOUND_CHUNK_BYTES]
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                }
            )
        )
    # Lets the caller know playback of this turn's answer has finished, if we ever
    # want to react to it (e.g. re-enable listening); harmless to send either way.
    await websocket.send_text(
        json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": "answer_done"}})
    )


async def _handle_utterance(websocket: WebSocket, session: TwilioCallSession, mulaw_audio: bytes) -> None:
    audio_array = twilio_audio_to_whisper_input(mulaw_audio)
    text, detected_language = stt.transcribe_audio(audio_array)
    if not text.strip():
        return

    spoken_language = normalize_language(detected_language)
    if spoken_language:
        session.state.language = spoken_language

    try:
        result = handle_turn(session.state, text)
    except Exception:
        logger.exception("handle_turn() raised for call %s, text: %r", session.call_sid, text)
        return

    try:
        wav_bytes = tts.synthesize(result.spoken_answer, language=session.state.language)
    except tts.VoiceNotAvailable:
        logger.exception("No TTS voice available for call %s (language=%s)", session.call_sid, session.state.language)
        return

    mulaw_reply = wav_bytes_to_twilio_mulaw(wav_bytes)
    if session.stream_sid:
        await _send_audio(websocket, session.stream_sid, mulaw_reply)


async def handle_call(websocket: WebSocket) -> None:
    """FastAPI WebSocket route handler -- one call for the whole lifetime of a call."""
    await websocket.accept()
    session: Optional[TwilioCallSession] = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            message: Dict[str, Any] = json.loads(raw_message)
            event = message.get("event")

            if event == "start":
                start_info = message.get("start", {})
                call_sid = start_info.get("callSid", "unknown-call")
                session = TwilioCallSession(call_sid)
                session.stream_sid = start_info.get("streamSid")
                logger.info("Call started: %s", call_sid)

            elif event == "media" and session is not None:
                payload_b64 = message.get("media", {}).get("payload", "")
                mulaw_frame = base64.b64decode(payload_b64)
                if session.accept_frame(mulaw_frame):
                    utterance_audio = session.take_utterance()
                    await _handle_utterance(websocket, session, utterance_audio)

            elif event == "stop":
                logger.info("Call ended: %s", session.call_sid if session else "unknown")
                break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected (call %s)", session.call_sid if session else "unknown")
