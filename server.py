"""
Minimal FastAPI wrapper around the existing conversation pipeline
(backend.conversation.pipeline.handle_turn) -- exposes it over local HTTP so it
can later be tunneled with ngrok. Does not touch the RAG/STT/TTS/conversation
code itself; this is purely an HTTP layer in front of what already exists.

Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import logging
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.conversation.pipeline import handle_turn
from backend.conversation.state import new_session
from backend.voice import stt, tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI(title="News-Rag Backend")


@app.on_event("startup")
def _print_start_message() -> None:
    print("Server started at http://localhost:8000")

# Single shared session for this minimal local-dev API -- no multi-user/session
# concept requested yet. Created lazily on first request, not at import time.
_session = None


def _get_session():
    global _session
    if _session is None:
        _session = new_session("api-session")
    return _session


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        result = handle_turn(_get_session(), message)
    except Exception:
        logger.exception("handle_turn() raised an exception for message: %r", message)
        raise HTTPException(status_code=500, detail="Internal error while processing the message.")

    return {
        "response": result.spoken_answer,
        "kind": result.kind,
        "stories": result.stories,
        "citation": result.citation,
    }


@app.post("/voice")
async def voice(audio: UploadFile = File(...), language: Optional[str] = Form(None)) -> Response:
    """
    Full spoken round trip: uploaded audio -> existing faster-whisper STT ->
    existing handle_turn() -> existing Piper TTS -> WAV audio back. One request in,
    one WAV response out -- the shape a telephony/voice client will call per utterance.
    """
    audio_bytes = await audio.read()

    try:
        text, _detected_language = stt.transcribe_bytes(audio_bytes, language=language)
    except Exception:
        logger.exception("STT failed while transcribing uploaded audio.")
        raise HTTPException(status_code=500, detail="Internal error while transcribing audio.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the uploaded audio.")

    try:
        result = handle_turn(_get_session(), text)
    except Exception:
        logger.exception("handle_turn() raised an exception for transcribed text: %r", text)
        raise HTTPException(status_code=500, detail="Internal error while processing the message.")

    try:
        answer_audio = tts.synthesize(result.spoken_answer, language=_get_session().language)
    except tts.VoiceNotAvailable as e:
        logger.exception("TTS voice unavailable.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        logger.exception("TTS failed while synthesizing the answer.")
        raise HTTPException(status_code=500, detail="Internal error while synthesizing speech.")

    return Response(
        content=answer_audio,
        media_type="audio/wav",
        headers={
            # Header values must be Latin-1; transcripts in Hindi/Marathi aren't, so
            # percent-encode (RFC 5987 style) rather than let non-ASCII text crash the response.
            "X-Transcript": quote(text),
            "X-Response-Kind": result.kind,
        },
    )
