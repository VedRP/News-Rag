"""
Phase 7: full local voice loop.

    Laptop Microphone -> faster-whisper -> conversation pipeline -> Piper TTS -> Speaker

Run: python scripts/voice_loop.py

Each turn: press Enter to start recording, speak, press Enter again to stop.
Your speech is transcribed, run through the same Phase 4-6 conversation
pipeline the text-based CLI (chat_cli.py) uses, and the answer is spoken back
through Piper. Say "quit" (typed or spoken) to exit.
"""
import io
import os
import sys
import tempfile
import winsound
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.backend_validation import normalize_language
from backend.conversation.pipeline import handle_turn
from backend.conversation.state import new_session
from backend.voice.stt import get_model as get_whisper_model, listen_and_transcribe
from backend.voice.tts import VoiceNotAvailable, is_available as tts_is_available, synthesize


def play_wav(wav_bytes: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        temp_path = f.name
    try:
        winsound.PlaySound(temp_path, winsound.SND_FILENAME)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def main() -> int:
    print("=" * 75)
    print("Phase 7 -- Full Local Voice Loop")
    print("=" * 75)

    if not tts_is_available():
        print("[WARNING] Not all Piper voices are present -- see backend/voice/tts.py's VOICE_FILES.")

    print("Loading faster-whisper model (fast if already cached, ~6 min on first-ever run)...")
    get_whisper_model()
    print("Ready.\n")

    state = new_session("voice-loop")
    while True:
        print("Press Enter to start recording (or type 'quit' + Enter to exit):")
        pre = input().strip().lower()
        if pre in ("quit", "exit", "q"):
            break

        # Always auto-detect (don't hint Whisper toward the session's current language) --
        # so whichever language the user actually speaks in this turn drives the reply,
        # rather than the system staying locked to whatever language it started in.
        text, detected_lang = listen_and_transcribe(language=None)
        if not text.strip():
            print("(heard nothing, try again)\n")
            continue

        print(f"You said ({detected_lang}): {text}")
        if text.strip().lower() in ("quit", "exit"):
            break

        spoken_language = normalize_language(detected_lang)
        if spoken_language:
            state.language = spoken_language

        result = handle_turn(state, text)
        print(f"Assistant [{result.kind}]: {result.spoken_answer}\n")

        try:
            wav_bytes = synthesize(result.spoken_answer, language=state.language)
            play_wav(wav_bytes)
        except VoiceNotAvailable as e:
            print(f"[Could not speak the answer: {e}]")

        if result.kind == "end":
            break

    print("Ending session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
