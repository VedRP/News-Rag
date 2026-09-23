"""
Phase 7 manual test window: type text, hear it spoken.

Run: python scripts/voice_test_gui.py

Two modes:
- "Speak this text" -- sends your typed text straight to the local IndicF5 TTS
  service and plays it back. No LLM involved, so this works even while the
  Groq/Gemini keys are exhausted.
- "Ask the assistant" -- runs your text through the full Phase 4-6 conversation
  pipeline (intent parse -> retrieval -> grounded generation -> localization)
  and speaks the resulting answer. Needs a working LLM provider.

Requires tts_service/server.py running (see that file's docstring) and its
reference_audio/manifest.json populated -- if the service isn't reachable, or a
step needs something not installed, this shows that plainly rather than
pretending it worked.
"""
import os
import sys
import tempfile
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.voice.tts import TTSServiceUnavailable, is_available, synthesize

LANGUAGES = ["english", "hindi", "marathi"]


class VoiceTestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Phase 7 - Voice Test Window")
        root.geometry("640x480")

        self.session_state = None  # lazily created only if "Ask the assistant" is used

        top = ttk.Frame(root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Language:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="hindi")
        ttk.Combobox(
            top, textvariable=self.language_var, values=LANGUAGES, width=10, state="readonly"
        ).pack(side=tk.LEFT, padx=(4, 20))

        self.mode_var = tk.StringVar(value="speak")
        ttk.Radiobutton(top, text="Speak this text", variable=self.mode_var, value="speak").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Ask the assistant", variable=self.mode_var, value="ask").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        self.service_status_label = ttk.Label(root, text="Checking TTS service...", padding=(10, 0))
        self.service_status_label.pack(fill=tk.X)

        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill=tk.X)
        self.text_entry = tk.Text(input_frame, height=4, wrap=tk.WORD)
        self.text_entry.pack(fill=tk.X)
        self.text_entry.insert("1.0", "नमस्ते, आज की मुख्य खबर क्या है?")

        self.submit_button = ttk.Button(root, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=(0, 10))

        ttk.Label(root, text="Answer text:", padding=(10, 0)).pack(fill=tk.X)
        self.answer_box = tk.Text(root, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.answer_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var, padding=(10, 4), relief=tk.SUNKEN).pack(fill=tk.X)

        self._refresh_service_status()
        self._warm_up()

    def _warm_up(self) -> None:
        """
        Pre-initializes the Qdrant client, the BGE-M3 embedding model, and the LLM
        router's HTTP/SSL setup on the MAIN thread, before any background-thread work
        happens. Found the hard way: the exact same handle_turn() call that completes
        in seconds when run directly hung indefinitely when its first invocation (and
        therefore each client's first-ever network/SSL init) happened inside a
        background threading.Thread here -- a known class of Windows issue with
        first-use SSL context setup off the main thread. Warming up here avoids ever
        hitting that path from a background thread.
        """
        self.status_var.set("Warming up (loading embedding model, connecting to Qdrant)...")
        self.root.update()
        try:
            from backend.embeddings.embedder import get_model as get_embedding_model
            from backend.retrieval.qdrant_client import ensure_collection, get_client

            get_embedding_model()
            ensure_collection(get_client())
        except Exception as e:
            self.status_var.set(f"Warm-up failed: {type(e).__name__}: {e}")
            return

        try:
            from backend.llm.client import chat

            chat("You are a test.", "Say OK.", role="fast", temperature=0.0, max_tokens=5)
        except Exception:
            pass  # Non-fatal here: "Speak this text" mode doesn't need an LLM at all.

        self.status_var.set("Ready.")

    def _refresh_service_status(self) -> None:
        available = is_available()
        self.service_status_label.config(
            text=(
                "TTS service: reachable at http://localhost:8100"
                if available
                else "TTS service: NOT reachable -- start tts_service (see its server.py docstring) first."
            )
        )

    def _set_answer_text(self, text: str) -> None:
        self.answer_box.config(state=tk.NORMAL)
        self.answer_box.delete("1.0", tk.END)
        self.answer_box.insert("1.0", text)
        self.answer_box.config(state=tk.DISABLED)

    def on_submit(self) -> None:
        utterance = self.text_entry.get("1.0", tk.END).strip()
        if not utterance:
            self.status_var.set("Type something first.")
            return
        self.submit_button.config(state=tk.DISABLED)
        self.status_var.set("Working...")
        threading.Thread(target=self._run, args=(utterance, self.mode_var.get(), self.language_var.get()), daemon=True).start()

    def _run(self, utterance: str, mode: str, language: str) -> None:
        try:
            if mode == "speak":
                answer_text = utterance
            else:
                answer_text = self._ask_assistant(utterance)
                self.root.after(0, self._set_answer_text, answer_text)

            self.root.after(
                0,
                self.status_var.set,
                "Synthesizing speech... this runs on CPU and can take several minutes "
                "(measured ~4-5 min per short sentence). Not frozen -- just slow.",
            )
            audio_bytes = synthesize(answer_text, language=language)
            self._play_audio(audio_bytes)
            self.root.after(0, self.status_var.set, "Done.")
        except TTSServiceUnavailable as e:
            self.root.after(0, self.status_var.set, f"TTS service unavailable: {e}")
        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {type(e).__name__}: {e}")
        finally:
            self.root.after(0, lambda: self.submit_button.config(state=tk.NORMAL))
            self.root.after(0, self._refresh_service_status)

    def _ask_assistant(self, utterance: str) -> str:
        from backend.conversation.state import new_session
        from backend.conversation.pipeline import handle_turn

        if self.session_state is None:
            self.session_state = new_session("voice-test-gui")
        result = handle_turn(self.session_state, utterance)
        return result.spoken_answer

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
