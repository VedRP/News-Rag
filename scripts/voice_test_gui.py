"""
Phase 7 test window: type OR speak your question, hear the answer spoken back.

Run: python scripts/voice_test_gui.py

- Type in the box and click Submit, OR click Record, speak, click Stop --
  either way, your input goes through the same Phase 4-6 conversation pipeline
  (intent parse -> retrieval -> grounded generation -> localization) and the
  answer is spoken aloud via Piper.
- Language dropdown picks which Piper voice speaks the answer, and hints
  faster-whisper toward that language when you record.
"""
import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import ttk

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.voice.stt import SAMPLE_RATE, transcribe_audio
from backend.voice.tts import VoiceNotAvailable, is_available as tts_is_available, synthesize

LANGUAGES = ["english", "hindi", "marathi"]
WHISPER_LANGUAGE_CODES = {"english": "en", "hindi": "hi", "marathi": "mr"}


class VoiceTestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Phase 7 - Type or Speak Test Window")
        root.geometry("680x520")

        self.session_state = None
        self.recording = False
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: sd.InputStream | None = None

        top = ttk.Frame(root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Language:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="english")
        ttk.Combobox(
            top, textvariable=self.language_var, values=LANGUAGES, width=10, state="readonly"
        ).pack(side=tk.LEFT, padx=(4, 20))

        self.record_button = ttk.Button(top, text="Record", command=self.on_record_toggle)
        self.record_button.pack(side=tk.LEFT)

        self.service_status_label = ttk.Label(root, text="Checking...", padding=(10, 0))
        self.service_status_label.pack(fill=tk.X)

        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill=tk.X)
        self.text_entry = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.text_entry.pack(fill=tk.X)
        self.text_entry.insert("1.0", "Give me the latest news")

        self.submit_button = ttk.Button(root, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=(0, 10))

        ttk.Label(root, text="Answer text:", padding=(10, 0)).pack(fill=tk.X)
        self.answer_box = tk.Text(root, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.answer_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="Starting...")
        ttk.Label(root, textvariable=self.status_var, padding=(10, 4), relief=tk.SUNKEN).pack(fill=tk.X)

        self._warm_up()

    def _warm_up(self) -> None:
        """
        Warms up Qdrant/embedding/LLM/whisper clients on the MAIN thread before any
        background-thread work happens -- see git history for why: the exact same
        pipeline call hung indefinitely the first time its first-ever network/model
        init happened inside a background threading.Thread (a known Windows issue
        with first-use SSL/model init off the main thread).
        """
        self.status_var.set("Warming up (embedding model, Qdrant, Whisper, LLM)...")
        self.root.update()
        try:
            from backend.embeddings.embedder import get_model as get_embedding_model
            from backend.retrieval.qdrant_client import ensure_collection, get_client
            from backend.voice.stt import get_model as get_whisper_model

            get_embedding_model()
            ensure_collection(get_client())
            get_whisper_model()
        except Exception as e:
            self.status_var.set(f"Warm-up failed: {type(e).__name__}: {e}")
            return

        try:
            from backend.llm.client import chat

            chat("You are a test.", "Say OK.", role="fast", temperature=0.0, max_tokens=5)
        except Exception:
            pass  # Non-fatal: typed "Speak this text"-style use doesn't strictly need it either.

        tts_ok = tts_is_available()
        self.service_status_label.config(
            text="Piper voices: all present" if tts_ok else "Piper voices: MISSING some -- see backend/voice/tts.py"
        )
        self.status_var.set("Ready.")

    def _set_answer_text(self, text: str) -> None:
        self.answer_box.config(state=tk.NORMAL)
        self.answer_box.delete("1.0", tk.END)
        self.answer_box.insert("1.0", text)
        self.answer_box.config(state=tk.DISABLED)

    # --- Recording ---

    def on_record_toggle(self) -> None:
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording_and_transcribe()

    def _start_recording(self) -> None:
        self.recording = True
        self.record_button.config(text="Stop")
        self.status_var.set("Recording... click Stop when done.")
        self._audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            self._audio_queue.put(indata.copy())

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)
        self._stream.start()

    def _stop_recording_and_transcribe(self) -> None:
        self.recording = False
        self.record_button.config(text="Record", state=tk.DISABLED)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        chunks = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())
        audio = np.concatenate(chunks, axis=0).flatten() if chunks else np.zeros(0, dtype=np.float32)

        if audio.size == 0:
            self.status_var.set("Heard nothing -- try again.")
            self.record_button.config(state=tk.NORMAL)
            return

        self.status_var.set("Transcribing...")
        threading.Thread(target=self._transcribe_then_ask, args=(audio,), daemon=True).start()

    def _transcribe_then_ask(self, audio: np.ndarray) -> None:
        try:
            whisper_lang = WHISPER_LANGUAGE_CODES.get(self.language_var.get())
            text, detected_lang = transcribe_audio(audio, language=whisper_lang)
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Transcription error: {type(e).__name__}: {e}")
            self.root.after(0, lambda: self.record_button.config(state=tk.NORMAL))
            return

        self.root.after(0, self._on_transcribed, text, detected_lang)

    def _on_transcribed(self, text: str, detected_lang: str) -> None:
        self.record_button.config(state=tk.NORMAL)
        if not text.strip():
            self.status_var.set("Heard nothing understandable -- try again.")
            return
        self.text_entry.delete("1.0", tk.END)
        self.text_entry.insert("1.0", text)
        self.status_var.set(f"Heard ({detected_lang}): {text}")
        self.on_submit()

    # --- Submit / ask / speak ---

    def on_submit(self) -> None:
        utterance = self.text_entry.get("1.0", tk.END).strip()
        if not utterance:
            self.status_var.set("Type or speak something first.")
            return
        self.submit_button.config(state=tk.DISABLED)
        self.status_var.set("Thinking...")
        threading.Thread(target=self._ask_and_speak, args=(utterance,), daemon=True).start()

    def _ask_and_speak(self, utterance: str) -> None:
        try:
            from backend.conversation.state import new_session
            from backend.conversation.pipeline import handle_turn

            if self.session_state is None:
                self.session_state = new_session("voice-test-gui")
            result = handle_turn(self.session_state, utterance)
            self.root.after(0, self._set_answer_text, result.spoken_answer)

            self.root.after(0, self.status_var.set, "Synthesizing speech...")
            wav_bytes = synthesize(result.spoken_answer, language=self.session_state.language)
            self._play_audio(wav_bytes)
            self.root.after(0, self.status_var.set, "Done.")
        except VoiceNotAvailable as e:
            self.root.after(0, self.status_var.set, f"No voice available: {e}")
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {type(e).__name__}: {e}")
        finally:
            self.root.after(0, lambda: self.submit_button.config(state=tk.NORMAL))

    def _play_audio(self, wav_bytes: bytes) -> None:
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


def main() -> None:
    root = tk.Tk()
    VoiceTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
