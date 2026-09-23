# Reference audio clips (required, not yet provided)

IndicF5 is a reference-audio-based (voice cloning) TTS model, not a fixed-voice one.
Every synthesis call needs a short (a few seconds), clean, single-speaker reference
clip **plus the exact transcript of what's spoken in it**. The model then speaks new
text in a voice similar to the reference.

This directory is currently empty. `manifest.json` needs one entry per supported
language, shaped like:

```json
{
  "english": { "audio_file": "en_ref.wav", "transcript": "exact text spoken in en_ref.wav" },
  "hindi":   { "audio_file": "hi_ref.wav", "transcript": "hi_ref.wav में जो बोला गया है वही टेक्स्ट" },
  "marathi": { "audio_file": "mr_ref.wav", "transcript": "mr_ref.wav मध्ये जे बोलले आहे तोच मजकूर" }
}
```

Do not fabricate these -- real audio requires an actual recording. Options:
- Record ~5-10 seconds of yourself (or someone who's consented) speaking clearly in
  each language, save as 16-24kHz mono WAV, and transcribe it exactly.
- Use a public-domain/permissively-licensed sample clip (with accurate transcript).

Once real clips + transcripts are here and `manifest.json` is filled in, the service
in `../server.py` can synthesize speech for that language.
