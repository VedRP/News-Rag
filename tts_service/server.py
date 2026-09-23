"""
Local HTTP service wrapping AI4Bharat's IndicF5 TTS model.

Runs in its own Python 3.10 venv (tts_service/.venv) since IndicF5 needs deps
incompatible with the main project's Python 3.14 environment. The main backend
(backend/voice/tts.py, Phase 7) talks to this over localhost HTTP -- same pattern
as talking to Qdrant -- rather than shelling out per call (which would reload the
model, ~0.4B params, on every request).

IndicF5 is reference-audio-based (voice cloning), not a fixed-voice TTS: every
call needs a short reference clip + its exact transcript, per language. See
reference_audio/manifest.json for the configured clips -- populate that before
this service can actually synthesize anything.

Run: .venv\\Scripts\\python.exe -m uvicorn server:app --port 8100
"""
import io
import json
import os

# f5_tts (IndicF5's base architecture) wraps its vocoder in torch.compile() internally.
# With torch 2.14 (much newer than what f5_tts was developed/tested against) this hits
# a dynamo tracer bug -- "Tensor on device cpu is not on the expected device meta!" --
# inside vocos's mel-spectrogram window creation. Disabling dynamo globally makes
# torch.compile(fn) a no-op (falls back to eager execution), sidestepping the bug
# without patching vendored library code. Must be set before torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from pathlib import Path
from typing import Optional

import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# .env lives at the project root, one level up from this service's own directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SERVICE_DIR = Path(__file__).resolve().parent
REFERENCE_MANIFEST_PATH = SERVICE_DIR / "reference_audio" / "manifest.json"
MODEL_REPO_ID = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

app = FastAPI(title="IndicF5 TTS Service")

_model = None
_reference_manifest: Optional[dict] = None


def _load_reference_manifest() -> dict:
    global _reference_manifest
    if _reference_manifest is None:
        if not REFERENCE_MANIFEST_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"{REFERENCE_MANIFEST_PATH} not found. IndicF5 needs a reference audio "
                    "clip + its transcript per language before it can synthesize anything -- "
                    "see reference_audio/README.md."
                ),
            )
        _reference_manifest = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return _reference_manifest


def _get_model():
    global _model
    if _model is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=500,
                detail="HF_TOKEN not set in .env -- required to download the gated ai4bharat/IndicF5 weights.",
            )
        # Import here, not at module load time: this is the slow (~model download +
        # torch init) step, and we want the server to start and answer /health even
        # if the model isn't loaded yet.
        from transformers import AutoModel

        # Setting the env var (rather than calling huggingface_hub.login(), which
        # does an extra `whoami` network round-trip to validate the token) is enough
        # for from_pretrained()'s own authenticated requests -- and avoids that
        # round-trip failing outright when HF_HUB_OFFLINE=1 is set (all files
        # already cached locally, no network needed at all).
        os.environ.setdefault("HF_TOKEN", hf_token)
        _model = AutoModel.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    return _model


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "hindi"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    manifest = _load_reference_manifest()
    ref = manifest.get(req.language.lower())
    if ref is None:
        raise HTTPException(
            status_code=400,
            detail=f"No reference audio configured for language '{req.language}'. "
                   f"Configured languages: {sorted(manifest.keys())}",
        )

    ref_audio_path = SERVICE_DIR / "reference_audio" / ref["audio_file"]
    if not ref_audio_path.exists():
        raise HTTPException(status_code=500, detail=f"Reference audio file missing: {ref_audio_path}")

    model = _get_model()
    audio = model(req.text, ref_audio_path=str(ref_audio_path), ref_text=ref["transcript"])

    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
