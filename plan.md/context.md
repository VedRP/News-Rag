# context.md — AI News Assistant: Master Project Context

> This is the canonical reference for the project. It supersedes earlier drafts (an early telephony-first spec, and a mid-point version that introduced Exotel/voice architecture too early). If anything here conflicts with an older file, this one wins. Update this file, not the older ones, as the project evolves.

---

## 1. What This Project Is

A personal, multilingual, conversational AI news assistant, built as a **learning project first, portfolio project second**. Not a commercial product, not launching publicly.

Long-term vision: a person calls a phone number, picks a language, the assistant speaks with them, gives important news, and understands natural follow-ups, topic/location filtering, and weather requests — grounded in a real retrieval system rather than the LLM's own pretrained knowledge.

**But that phone version is the last phase, not the starting point.** The current and near-term work is 100% local — laptop only, no telephony, no phone number, no Exotel/Twilio/DTMF.

### What I actually want to learn (not just ship)
- How embeddings work, and specifically how BGE-M3 works in practice
- How vector search and metadata filtering work together
- How RAG pipelines are actually structured (retrieval ≠ generation ≠ the LLM itself)
- How chunking strategy affects retrieval quality
- How LLMs convert natural language into structured intent/entities
- How structured outputs / JSON-mode generation works
- How conversation state and follow-up resolution work
- How hallucination is controlled through prompting and grounding
- How Whisper integrates into a backend pipeline
- How TTS integrates into a backend pipeline
- How the full voice loop (mic → STT → AI → TTS → speaker) fits together
- How WebSockets work for streaming audio (later)
- How the eventual telephony architecture works (later)

I do not want a generic ML roadmap — I want to understand this specific system deeply enough to explain how every layer works.

---

## 2. Ground Rules for How This Should Be Built

1. Explain what a component actually does before building it.
2. Explain *why* it's needed and how it connects to the rest of the system.
3. Don't hide important concepts behind libraries — explain what the library is doing internally when it matters.
4. Build incrementally, one phase at a time (see Section 9). Do not jump ahead.
5. Give the smallest working version of each phase, test it, then move on.
6. Do not hand over an entire codebase at once.
7. Prefer free/local solutions. Don't assume paid APIs are acceptable by default.
8. When a technology choice is made, explain the alternatives considered and why this one fits *this* project.
9. Don't introduce a technology/model before the phase that actually needs it (e.g. no TTS talk during PDF ingestion, no telephony talk during RAG).
10. When newspaper PDFs are provided, inspect their actual structure first — don't assume all newspaper layouts are the same.

---

## 3. Current Phase (Where We Actually Are)

**Phase 1: PDF ingestion.** Nothing beyond this — no embeddings, no Qdrant, no LLM, no voice — should be implemented yet. The immediate next concrete step is: I provide a sample newspaper PDF, we inspect its structure together, and decide on an extraction/chunking strategy for it specifically.

---

## 4. Cost Philosophy

This is a ₹0-as-possible learning project.

- Local models, local vector DB, local database, local dev server — preferred everywhere it's practical.
- PDFs as the news source (no paid news APIs, no scraping) for now.
- Groq for LLM inference during development because it's fast for experimentation — but treat its free tier as not permanent/unlimited, and keep the actual RAG/embedding/database pipeline fully under local control so the LLM vendor is swappable.
- STT: local (faster-whisper), not a paid hosted API.
- TTS: local open-source multilingual model preferred initially (Piper / Coqui XTTS / Indic-focused models — to be evaluated when we reach that phase), not assumed to require ElevenLabs.
- Only bring in a paid service when it provides something genuinely hard to reproduce locally. The one component that's unavoidably external, eventually, is the phone network itself (Section 10).

---

## 5. Current (Near-Term) Architecture — Local Only

```text
                         NEWSPAPER PDFs
                              │
                              ▼
                       PDF INGESTION
                              │
                              ▼
                       TEXT EXTRACTION
                              │
                              ▼
                           CLEANING
                              │
                              ▼
                           CHUNKING
                              │
                              ▼
                           METADATA
                              │
                              ▼
                           BGE-M3
                              │
                              ▼
                           QDRANT
                              │
                              │
                              ▼
🎤 USER ──→ WHISPER ──→ CONVERSATION MANAGER
                              │
                              ▼
                    INTENT + ENTITY EXTRACTION
                              │
                              ▼
                       BACKEND VALIDATION
                              │
                              ▼
                         QUERY CREATION
                              │
                              ▼
                    METADATA + VECTOR SEARCH
                              │
                              ▼
                     RELEVANT NEWS CHUNKS
                              │
                              ▼
                             LLM
                              │
                              ▼
                        ANSWER TEXT
                              │
                              ▼
                             TTS
                              │
                              ▼
                       🔊 LAPTOP SPEAKER
```

Key principle baked into this diagram: **the raw user sentence never goes straight to an LLM that just answers freely.** It always goes STT → LLM/NLU → structured request → backend validation → retrieval → LLM generation. This keeps the system controllable and gives us somewhere concrete to enforce grounding.

---

## 6. Future Architecture — Phone Version (Later, Not Now)

The design principle: **the core AI system must not depend on the phone.** The phone is eventually just another input/output layer bolted onto the same local system — it does not get built into the system's foundations.

```text
Phone
 ↓
Telephony provider (candidate: Exotel, for India — bidirectional WebSocket
                     streaming, so caller audio streams to our server and
                     our audio streams back)
 ↓
WebSocket
 ↓
Voice server
 ↓
Whisper (same STT component as the local version)
 ↓
AI system (same conversation manager / intent / RAG / LLM as the local version)
 ↓
TTS (same TTS component as the local version)
 ↓
WebSocket
 ↓
Telephony provider
 ↓
Phone
```

During development, a local WebSocket endpoint would be exposed via a tunnel (e.g. ngrok) so the AI/data components can keep running on the laptop while the telephony provider reaches them over the internet. The only genuinely unavoidable external dependency in the whole project is the connection to the telephone network itself — everything else (PDFs, Qdrant, BGE-M3, RAG, conversation state, backend, STT, LLM inference target, TTS) can stay local or self-hosted.

DTMF (keypad input) is explicitly out of scope until this phase, and even then it's only for simple navigation (customize / weather / repeat / end) — news topic/location selection stays natural-language, never a keypad menu.

---

## 7. News Data Source

**Newspaper PDFs**, provided by me, placed in a local folder:

```text
project/
│
├── news/
│   ├── newspaper_01.pdf
│   ├── newspaper_02.pdf
│   ├── newspaper_03.pdf
│   └── ...
│
├── backend/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── rag/
│   ├── conversation/
│   ├── voice/
│   └── api/
│
└── ...
```

Do not use live news APIs or web scraping in the initial version. Live ingestion (RSS/APIs) is a possible later replacement/extension, not a current requirement.

### Ingestion pipeline (Phase 1)

```text
PDF
 ↓
Text extraction
 ↓
Cleaning
 ↓
Article/section detection
 ↓
Chunking
 ↓
Metadata extraction
 ↓
BGE-M3 embedding      (Phase 2, not Phase 1)
 ↓
Qdrant                (Phase 2, not Phase 1)
```

An entire PDF is never embedded as a single document — always chunked into article/section-level pieces first, since a newspaper PDF mixes many unrelated stories.

### Chunk metadata (schema will evolve; this is the starting shape)

```json
{
  "text": "Heavy rainfall was reported across Mumbai...",
  "source": "newspaper_01.pdf",
  "page": 4,
  "date": "2026-09-08",
  "language": "english",
  "location": ["Mumbai", "Maharashtra"],
  "topics": ["weather", "mumbai"]
}
```

Important metadata fields to preserve per chunk: source newspaper/filename, page number, publication date, language, topic(s), location, and article/section identifier if it can be determined. Source + page + date are also what later enables answer attribution ("According to the newspaper report on page 4...").

---

## 8. Core Components (Reference)

| Component | Role | Notes |
|---|---|---|
| PDF ingestion | Extract, clean, chunk, tag newspaper PDFs | Phase 1. Layout is inspected per-PDF, not assumed uniform. |
| BAAI/bge-m3 | Embedding model | Multilingual (100+ languages, good for Indian languages), 1024-dim dense vectors, supports long inputs, can run locally, also supports sparse/ColBERT modes if we want hybrid search later. Same model embeds both news chunks and user queries. |
| Qdrant | Vector database | Run locally via Docker. Stores embeddings + metadata; handles semantic/vector retrieval. `docker run -p 6333:6333 qdrant/qdrant` |
| MongoDB/PostgreSQL | Traditional DB (later, not urgent) | For users, sessions, preferences, conversation history, story records — not needed for the earliest phases; don't over-engineer this early. |
| RAG pipeline | Retrieval-Augmented Generation | Not a model — a pipeline: query understanding → metadata filtering → vector search → (optional reranking) → context → LLM. |
| LLM | Two roles | (1) NLU: natural language → structured intent/entities JSON. (2) Generation: retrieved evidence → grounded natural-language answer. Groq initially (inference platform, not the model itself — serves Llama/Qwen-class models); local LLM is a possible later swap. |
| faster-whisper | Speech-to-text | Local, not a paid hosted API. Modular so it can be swapped later. |
| TTS | Text-to-speech | Local open-source multilingual model preferred (Piper / Coqui XTTS / Indic-focused models — evaluate at that phase based on language support, quality, hardware needs, ease of integration). |
| Conversation Manager | Session/context state | Tracks language, location, current topic, current story, previous stories, conversation history; resolves follow-up references. |
| Exotel (later) | Telephony gateway | Only handles phone/audio transport in the future phone phase — not the AI itself. |

Keep these responsibilities separate in code, not blended — this separation is a deliberate architectural principle, not incidental.

---

## 9. Natural Language Understanding

No keypad category menus for news topics/locations — everything is natural language, converted internally into structured data.

Example utterances the system should handle:
- "Give me cricket news."
- "Give me Mumbai news."
- "What's happening in Maharashtra?"
- "Give me today's cricket news from Maharashtra."
- "What's happening in Bollywood?"
- "Tell me about financial markets in Mumbai."
- "Marathi mein Mumbai ki cricket news batao."

Structured output example:

```json
{
  "intent": "news",
  "topic": "cricket",
  "location": { "state": "Maharashtra" },
  "time": "today",
  "language": "marathi"
}
```

### Intent taxonomy (internal only, never shown as a menu)
`news`, `weather`, `repeat`, `more_details`, `follow_up`, `change_language`, `customize`, `end`

### Topic taxonomy (semantic categories, not a keypad list)
`cricket`, `sports`, `bollywood`, `politics`, `business`, `finance`, `technology`, `education`, `science`, `health`, `automobile`, `lifestyle`, `weather`, `general`

Location is understood at multiple levels — city, state, or country — inferred from the utterance (e.g. "Mumbai" implies city; if state/country aren't explicitly said but are unambiguous, they can be inferred, e.g. Mumbai → Maharashtra → India).

---

## 10. Conversation State

```json
{
  "language": "marathi",
  "location": { "city": "Mumbai", "state": "Maharashtra" },
  "currentTopic": "cricket",
  "currentStory": "...",
  "previousStories": [],
  "conversationHistory": []
}
```

This is what makes follow-ups possible without the user repeating context:
- "Tell me more about number 2."
- "Tell me more about that."
- "Who announced it?"
- "When did this happen?"
- "What about Maharashtra?"
- "Change to Hindi."

The conversation manager resolves these against session state rather than treating each utterance as independent.

---

## 11. Multilingual Support

Target languages (not all need day-one support, but the architecture should be language-agnostic so more can be added later): English, Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi.

Flow once voice is added:

```text
User speaks Marathi
       ↓
Whisper
       ↓
Marathi text
       ↓
Intent extraction
       ↓
Retrieval (BGE-M3 handles cross-lingual retrieval)
       ↓
LLM (generates in Marathi)
       ↓
Marathi answer
       ↓
TTS
       ↓
Speaker
```

Language can change mid-session ("Change to Hindi") and the session's `language` field updates accordingly for all future responses.

---

## 12. Hallucination Control

Because this is a factual news system:

1. Answer using only the retrieved newspaper evidence.
2. Do not invent facts, names, numbers, dates, or outcomes absent from retrieved context.
3. If evidence is insufficient, say so plainly rather than guessing.
4. Distinguish confirmed facts from claims/reports; preserve source uncertainty where the source itself is uncertain.
5. Preserve source metadata (source, page, date) per chunk so answers can eventually cite where information came from.
6. Don't let the LLM's own pretrained knowledge override or supplement retrieved current information.

---

## 13. Advanced Features (Explicitly Deferred, Not MVP)

These are real parts of the long-term vision but must not block early phases:

- **News priority/ranking** beyond simple recency — factoring in source reliability, cross-source confirmation, public importance, geographic relevance, user relevance, breaking-news signal.
- **Story deduplication/clustering** — recognizing that multiple newspapers reporting the same event ("Heavy rain hits Mumbai" / "Mumbai sees heavy rainfall" / "Heavy rainfall disrupts Mumbai") are one underlying story with multiple corroborating sources, rather than three separate stories.
- **Hybrid retrieval** — combining BM25/sparse retrieval with dense vector search, plus reranking.
- **Cross-source verification** using the clustering above.

Build correct basic retrieval first; layer these in during Phase 8.

---

## 14. Recommended Tech Stack

```text
Language:        Python (AI/RAG side). Node/Express/Mongo already known and
                  usable elsewhere if there's a specific architectural reason —
                  but Python is preferred for this project because of the ML ecosystem.
PDF processing:   PyMuPDF or similarly suitable extraction library
STT:              faster-whisper (local)
Embeddings:       BAAI/bge-m3 (local)
Vector DB:        Qdrant (local, via Docker)
LLM:              Groq initially (Llama/Qwen-class hosted models) OR local LLM later
TTS:              Local open-source multilingual TTS (evaluate at that phase)
Backend:          Python API/server
News source:      Local newspaper PDFs
Weather:          Weather API (later phase)
Telephony:        Exotel (later phase, India-focused), bidirectional WebSocket
Dev tunnel:       ngrok or equivalent (later phase, for exposing local WebSocket)
```

Avoid introducing unnecessary microservices early.

---

## 15. Development Strategy — Phases

**Phase 1 — PDF ingestion** *(current phase)*
PDF → text extraction → cleaning → chunking → metadata. Test that extracted chunks are actually correct before moving on.

**Phase 2 — Embeddings + Qdrant**
Chunks → BGE-M3 → Qdrant. Test: query → embedding → Qdrant → top relevant chunks. No LLM involved yet.

**Phase 3 — Basic RAG**
Question → Qdrant → relevant chunks → LLM → answer. This is the first working text-based news assistant — no voice, no intent extraction yet, just prove retrieval + grounded generation works.

**Phase 4 — Intent/entity extraction**
Natural language → intent, topic, location, time, language → backend validation → retrieval (now driven by structured requests instead of raw queries).

**Phase 5 — Conversation state**
Current topic, current story, previous stories, conversation history, language, location. Implement follow-up resolution.

**Phase 6 — Multilingual**
Language detection, multilingual retrieval, multilingual generation, language switching mid-session.

**Phase 7 — Local voice**
Microphone → faster-whisper → AI system → TTS → speaker. This is the first complete voice assistant — still 100% local, no phone involved.

**Phase 8 — News intelligence upgrades**
Story clustering, deduplication, cross-source confirmation, priority ranking, source attribution.

**Phase 9 — Phone integration**
Only after the local system fully works: Phone → Exotel → WebSocket → the existing voice/AI system (unchanged). This is explicitly last, and "the call" — telephony — is exactly what's being deferred per this document's title question ("what the call we implement later").

---

## 16. Non-Goals For Now

Do not build, discuss implementation details of, or introduce dependencies for, until their designated phase arrives:
- Exotel / Twilio / any telephony provider
- Real phone numbers, inbound call handling
- DTMF keypad logic
- WebSocket audio streaming
- ngrok/tunneling
- Live news APIs or web scraping
- A production-grade database architecture (Mongo/Postgres can wait)
- Reranking, hybrid search, story clustering, priority scoring

---

## 17. Immediate Next Step

Provide a sample newspaper PDF so we can inspect its actual layout (columns, article boundaries, headers/footers, how stories are visually separated) and design an extraction + chunking strategy suited to that structure, rather than assuming a generic one-size-fits-all approach. This is the concrete start of Phase 1.
