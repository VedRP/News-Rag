# IndicF5 TTS Service

Local HTTP wrapper around AI4Bharat's IndicF5 TTS model, running in its own Python
3.10 venv (the main project is Python 3.14; IndicF5's dependency stack needs 3.10).

## One-time setup

```powershell
# 1. Python 3.10 (installed via winget on this machine: winget install --id Python.Python.3.10)
py -3.10 -m venv .venv

# 2. IndicF5 itself (pulls torch, transformers, f5_tts, etc. -- large download)
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/ai4bharat/IndicF5.git"

# 3. This service's own deps, INCLUDING two fixes for issues hit during setup:
#    - transformers<4.50: IndicF5's setup.py leaves transformers unbounded, so step 2
#      installs the latest (5.x) by default, which crashes with a device-mismatch
#      error inside the model's vocoder init (transformers 5.x's meta-device loading
#      isn't compatible with this model's older custom code). Their own
#      requirements.txt (not used by a git install) pins <4.50 -- we do the same here.
#    - torchcodec: torchaudio.load() needs it in recent torchaudio versions; not
#      pulled in automatically.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. FFmpeg SHARED build (not the static build!) -- torchcodec dynamically loads
#    FFmpeg's shared libraries (avcodec-*.dll etc.) at runtime. The static build
#    (winget install --id Gyan.FFmpeg) does NOT include these as separate DLLs.
winget install --id Gyan.FFmpeg.Shared
# Then make sure its bin/ directory (containing avcodec-*.dll etc.) is on PATH
# before starting the server -- see start command below. Restart your shell first
# so winget's PATH change takes effect, or prepend it manually like the example does.
```

## Accepting the gated model + getting a token (must be done by a human, once)

1. Log into huggingface.co, visit https://huggingface.co/ai4bharat/IndicF5, and
   click through to accept the model's terms of use.
2. Generate an access token (Settings -> Access Tokens) and put it in the project
   root `.env` as `HF_TOKEN=...` (gitignored, never commit it).

## Running

```powershell
powershell -File tts_service\start_server.ps1
```

This handles the FFmpeg-shared-on-PATH and offline-mode setup below automatically.
(Equivalent manual command, if you need to run it differently: `cd tts_service`
then `.\.venv\Scripts\python.exe -m uvicorn server:app --port 8100`, after putting
the FFmpeg shared `bin/` dir on PATH and setting the offline env vars yourself.)

`GET /health` should return `{"status": "ok"}` almost immediately. The model itself
(and the gated weight download, first run only) only loads on the first
`POST /synthesize` call, not at startup -- so the server can be "up" before the
model is ready to actually generate anything.

**After the first successful run** (model + vocoder weights fully cached locally),
set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before starting the server.
This isn't just a speed optimization -- during development, an unrelated
"check-for-updates" network call (made even when using cached files) stalled on a
flaky connection and hung the whole request indefinitely with zero CPU usage. Going
offline once everything's cached avoids that failure mode entirely, not just the
latency of the check.

## Two more fixes, already applied in server.py

- `f5_tts` (IndicF5's base architecture) wraps its vocoder in `torch.compile()`
  internally. With this project's torch version (much newer than what `f5_tts` was
  tested against), that hits a dynamo tracer bug. `server.py` sets
  `TORCHDYNAMO_DISABLE=1` before importing torch, which makes `torch.compile(fn)` a
  no-op (falls back to eager execution) rather than patching vendored library code.
- Don't call `huggingface_hub.login(token=...)` -- it does an extra `whoami` network
  round-trip just to validate the token, which (a) isn't needed since
  `from_pretrained()` already authenticates its own requests via the `HF_TOKEN` env
  var, and (b) hard-fails under `HF_HUB_OFFLINE=1`. `server.py` just sets the env var.

## Known limitation: CPU inference is slow (measured, not estimated)

This machine has no GPU. IndicF5 is a flow-matching TTS model (~0.4B params, ~32
sampling steps) -- measured **~4.5 minutes for one short Hindi sentence** with a
warm (already-loaded) model; the very first call after server startup is slower
still (model load + weight materialization on top of that). This is expected, not
a bug -- but it means:
- Any HTTP client calling `/synthesize` needs a long timeout (`backend/voice/tts.py`
  uses 600s).
- Real-time or near-real-time voice interaction isn't practical on this hardware as-is.
  If that matters more than avoiding a GPU dependency, revisit later (GPU, a smaller/
  faster model, or fewer sampling steps if IndicF5 exposes that as a parameter --
  it currently doesn't, per its `model.py`'s `forward()` signature).

## Reference audio

See `reference_audio/README.md` -- IndicF5 is reference-audio-based (voice cloning),
not a fixed-voice TTS.
