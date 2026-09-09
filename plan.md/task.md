# task.md — Claude Code Build Instructions

**Read `context.md` in this repo before doing anything else.** It is the source of truth for architecture, phase order, tech stack, schemas, and non-goals. If anything in this file seems to conflict with `context.md`, stop and ask rather than guessing which one wins.

This file breaks the project into phase-sized tasks for Claude Code to execute one at a time, inside a real terminal/IDE session with file system and bash access.

---

## 0. Non-negotiable operating rules

These apply to every phase below, at all times:

1. **Never invent or assume an API key, credential, account, or paid service exists.** If a task needs one (Groq API key, a specific model access, a paid TTS/STT service, etc.), stop and explicitly ask me for it, or ask me to go get it and tell you when it's set. Do not write code that silently expects a key to "just be there" without first confirming with me that I have it and where it's stored.
2. **Never assume a library, model weight, PDF, dataset, or tool is already installed/downloaded.** Check first (e.g. `pip show`, `which`, checking for a file/folder). If it's missing, tell me exactly what needs to be installed or downloaded and how, and wait for my go-ahead — don't just install things unprompted if it involves downloading large models, creating accounts, or anything with cost/storage implications. Small, obviously-safe dev dependencies (e.g. a pip package already listed in `requirements.txt`) are fine to install directly.
3. **Never hallucinate file contents, PDF contents, or data.** If you need to see what's actually in the `news/` folder, a PDF, or any file, read it first — don't guess at its structure or content and proceed as if you know it.
4. **If something fails, stop and report — do not loop.** Do not repeatedly retry the same failing approach, silently swallow errors, or keep "fixing" in circles without surfacing what's actually going wrong. If you try something and it fails, tell me what failed, what you think the cause is, and what you'd like to try next — then wait for confirmation if the next step is non-trivial (changes scope, needs a new dependency, needs a credential, etc.).
5. **Do not skip ahead to a later phase.** If implementing the current phase naturally surfaces something from a later phase (e.g. you're in Phase 1 and start wanting to add embeddings), stop, note it, and ask whether I want to pull that forward or stay scoped to the current phase.
6. **Explain before you build.** For each phase: briefly explain what the component does and why it's needed, propose the smallest working version, then implement it. Don't dump a large amount of code with no explanation.
7. **Test before moving on.** Each phase below has a "Definition of done" — actually run it and show me real output before considering the phase complete or starting the next one.
8. **When you encounter a newspaper PDF, inspect its actual structure before deciding how to parse it.** Don't assume all PDFs have the same layout — check columns, headers/footers, how articles are visually separated, whether it's text-based or scanned/image-based (which would need OCR — flag this to me explicitly if so, since it changes the pipeline).
9. **Ask me one focused question at a time when something is genuinely ambiguous**, rather than making a silent judgment call on something that materially affects architecture, cost, or data handling.

If you are ever unsure whether something requires my input under these rules, err on the side of asking.

---

## Phase 1 — PDF Ingestion (current phase — start here)

**Goal:** Turn newspaper PDFs in `news/` into clean, well-chunked text with useful metadata. No embeddings, no Qdrant, no LLM yet.

**Before writing code:**
- Check whether a `news/` folder with PDFs already exists in this repo. If it's empty or missing, ask me to add at least one sample PDF before proceeding — do not fabricate sample text to test against.
- Once a PDF is available, extract a page or two and show me the raw output so we can both see what the real structure looks like (columns, article boundaries, noise like page numbers/ads) before deciding on a cleaning/chunking strategy.
- Confirm the PDF extraction library to use (default suggestion: PyMuPDF) — check if it's already in `requirements.txt`/installed; if not, tell me and ask before installing.

**Build:**
- `backend/ingestion/extract.py` — extracts raw text (and page numbers) from a PDF.
- `backend/ingestion/clean.py` — removes noise (headers, footers, page numbers, boilerplate) based on what we actually observed in the sample PDF.
- `backend/ingestion/chunk.py` — splits cleaned text into article/section-level chunks (not one giant blob, not one chunk per PDF).
- `backend/ingestion/metadata.py` — attaches metadata per chunk per the schema in `context.md` Section 7 (source filename, page, date, language, location, topics — some fields may need to start as `null`/best-effort until later phases add smarter extraction).
- A small test script that runs the full pipeline against one PDF and prints the resulting chunks + metadata so I can review quality.

**Definition of done:** Running the test script against a real PDF I provided produces chunks that look like coherent article-sized pieces of text (not garbled, not cut mid-sentence arbitrarily, not one massive blob), each with at least source filename and page number attached. I have reviewed this output and confirmed it looks right before we move to Phase 2.

**Do not in this phase:** touch BGE-M3, Qdrant, any LLM, or any voice component.

---

## Phase 2 — Embeddings + Qdrant

**Goal:** Turn Phase 1's chunks into vectors and store them for retrieval. No LLM yet.

**Before writing code:**
- Confirm Qdrant is running locally (`docker run -p 6333:6333 qdrant/qdrant`) — check the port responds; if it's not running, tell me the exact command to run and wait rather than trying to start Docker yourself unless you've confirmed you have permission/access to do so in this environment.
- Confirm `BAAI/bge-m3` is not yet downloaded, tell me the approximate download size and that it will download on first run via the embedding library, and confirm I'm fine with that before triggering the download.

**Build:**
- `backend/embeddings/embedder.py` — wraps bge-m3, embeds text into 1024-dim vectors.
- `backend/retrieval/qdrant_client.py` — connects to local Qdrant, creates the collection (schema per `context.md` Section 7/8), with payload indexes on the metadata fields we'll filter on.
- `backend/ingestion/index.py` — takes Phase 1's chunks, embeds them, upserts into Qdrant.
- A test script: embed a hand-typed query, run a vector search against the indexed chunks from a real PDF, print back the top results with scores.

**Definition of done:** A real query against real indexed newspaper content returns plausible top-k results with reasonable similarity scores. I have reviewed this and confirmed retrieval looks sane before moving on.

**Do not in this phase:** add an LLM, intent parsing, or voice.

---

## Phase 3 — Basic RAG (text only)

**Goal:** A working text-based, grounded news Q&A loop. No voice, no intent extraction, no conversation state yet — a single question in, a single grounded answer out.

**Before writing code:**
- Ask me for a Groq API key (or confirm I already have one set up) before writing any code that calls Groq. Do not proceed with LLM calls assuming a key is present — check for it (e.g. via an expected env var) and if missing, stop and ask me to provide it or tell you where it's stored.
- Confirm which Groq-hosted model to target (propose a sensible default, but confirm with me rather than assuming).

**Build:**
- `backend/llm/groq_client.py` — thin wrapper for chat completions.
- `backend/rag/answer.py` — implements: question → Qdrant retrieval (Phase 2) → build context from top chunks → grounded generation prompt (hallucination rules per `context.md` Section 12, verbatim, don't water them down) → answer.
- A CLI test script: type a question, get a grounded answer printed, along with which source/page chunks were used.

**Definition of done:** Asking a real question about content actually present in the indexed PDFs produces an answer that is clearly grounded in that content (and cites source/page), and asking something not covered produces an honest "insufficient information" response rather than a fabricated answer. Both cases tested and shown to me.

**Do not in this phase:** add structured intent parsing (the question can be typed as plain text — no JSON extraction of topic/location/time yet), conversation state, multilingual handling, or voice.

---

## Phase 4 — Intent / Entity Extraction

**Goal:** Convert natural-language requests into structured `{intent, topic, location, time, language}` per `context.md` Section 9, and validate them before retrieval.

**Build:**
- `backend/llm/prompts/intent_parser.py` — system prompt + call that outputs strict JSON per the schema in `context.md`.
- `backend/backend_validation.py` — validates the parsed structure (known topic taxonomy, sane location format, etc.) before it's used to build a Qdrant filter.
- Wire this in front of Phase 3's retrieval: structured request → metadata filter + vector search → same grounded generation as Phase 3.
- Test script covering a range of real utterances, including at least one deliberately ambiguous/nonsense one to confirm it doesn't force a confident-but-wrong parse.

**Definition of done:** A handful of test utterances (including a genuinely ambiguous one) produce sensible structured output, and the ambiguous one is flagged as low-confidence/unclear rather than guessed at.

---

## Phase 5 — Conversation State

**Goal:** Support follow-ups without repeating context, per `context.md` Section 10.

**Build:**
- `backend/conversation/state.py` — in-memory session state (language, location, currentTopic, currentStory, previousStories, conversationHistory).
- Follow-up reference resolution (e.g. "tell me more about number 2", "who announced it") against the current session state.
- CLI test loop demonstrating: ask for news → get a numbered set of results → ask a follow-up referring to one of them → correct story is used.

**Definition of done:** A multi-turn CLI conversation with at least one follow-up reference resolves correctly, shown to me directly.

**Do not in this phase:** add multilingual switching logic (that's Phase 6) or voice.

---

## Phase 6 — Multilingual

**Goal:** Support requests and responses in multiple languages, with mid-session switching, per `context.md` Section 11.

**Before writing code:** confirm which languages we're actually testing against first (don't try to validate all ten from `context.md` at once) — ask me which 2–3 to start with if not already decided.

**Build:**
- Extend intent parsing to detect/handle language and "change language" requests.
- Extend generation prompt to respond in the session's target language.
- Test: same underlying English-indexed content, queried and answered in a non-English language, plus a mid-conversation language switch that actually changes subsequent answers.

**Definition of done:** A language switch mid-CLI-session visibly changes the language of subsequent answers, not just an acknowledgment.

---

## Phase 7 — Local Voice

**Goal:** Full local voice loop: microphone → faster-whisper → AI system (Phases 3–6) → TTS → speaker. Still no phone.

**Before writing code:**
- Confirm faster-whisper is not yet installed/downloaded; tell me model size options (tiny/base/small/etc.) and approximate download sizes, and let me choose before downloading.
- Once we reach the TTS half of this phase, propose 2–3 local multilingual TTS candidates (e.g. Piper, Coqui XTTS, an Indic-focused model) with a short tradeoff summary (language coverage, quality, install complexity, hardware needs) and ask me to pick rather than unilaterally choosing and installing one.
- Confirm microphone/speaker access works in this environment before building around it — if Claude Code is running somewhere without local audio device access, flag that immediately rather than trying to build around a missing capability.

**Build:**
- `backend/voice/stt.py` — wraps faster-whisper, mic input → text.
- `backend/voice/tts.py` — wraps the chosen local TTS model, text → audio → speaker playback.
- A small runnable loop tying mic → STT → existing AI pipeline (Phases 3–6) → TTS → speaker.

**Definition of done:** A spoken question into the mic produces a spoken, grounded answer out of the speaker, demonstrated live.

---

## Phase 8 — News Intelligence Upgrades

**Goal:** Story clustering/deduplication, cross-source confirmation, priority ranking, source attribution — per `context.md` Sections 13 and 17.

Only start this phase once Phase 7 is genuinely working end to end. This phase is naturally more open-ended — propose a concrete smallest-first slice (e.g. simple duplicate detection across two similarly-worded chunks before full clustering) and confirm the slice with me before building it, since "improve news intelligence" is broad enough to sprawl if left unscoped.

---

## Phase 9 — Phone Integration (last, not now)

**Goal:** Bolt the existing, already-working local system onto a telephony provider (Exotel, per `context.md`).

**Before writing any code in this phase:**
- Confirm I actually want to start this phase — it introduces real external accounts/costs (Exotel account, a virtual number, possibly ngrok) that don't exist in any earlier phase.
- Ask me explicitly for: an Exotel account/API credentials, confirmation of a virtual number, and how I want to expose the local WebSocket server during development (ngrok or otherwise) — do not assume any of these are already set up.

Do not begin scaffolding this phase's code until Phases 1–8 are complete and I've explicitly said to start it.

---

## How to work through this file

- Work top to bottom, one phase per session (or per few sessions if a phase is large) — don't parallelize phases.
- At the start of a session, tell me which phase you're starting and restate that phase's goal in one or two sentences before touching code.
- At the end of a phase, restate the Definition of done and confirm it's actually met with real output, not a description of what output would look like.
- If you ever find yourself about to write a fake/mocked API key, a fake PDF, fake retrieval results, or any other stand-in for something real just to keep moving — stop instead and ask me for the real thing.
