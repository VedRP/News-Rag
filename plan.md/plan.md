# Voice News Assistant — RAG + LLM Subsystem Plan

Scope: this plan covers **only** the RAG + LLM layer (sections 8–11, 16–17, 20 of the master spec) — embedding, vector storage/retrieval, intent parsing, and answer generation. Telephony/STT/TTS are out of scope here.

## 0. Stack for this subsystem

| Layer | Choice |
|---|---|
| LLM | Groq API (e.g. `llama-3.3-70b-versatile` for generation, `llama-3.1-8b-instant` for cheap/fast intent parsing) |
| Embeddings | `BAAI/bge-m3` (1024-dim, multilingual, supports dense + sparse + ColBERT — we'll use dense only for MVP) |
| Vector DB | Qdrant, local via Docker |
| Reranking (later) | bge-reranker-v2-m3 (optional, phase 2) |

Qdrant local:
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
Dashboard: `http://localhost:6333/dashboard`

---

## 1. Project structure

```
rag-llm/
├── .env                     # GROQ_API_KEY, QDRANT_URL
├── requirements.txt
├── embeddings/
│   └── embedder.py          # bge-m3 wrapper
├── vectorstore/
│   ├── qdrant_client.py     # connection + collection setup
│   └── schema.py            # payload schema / models
├── ingestion/
│   └── index_story.py       # embed + upsert a news story
├── retrieval/
│   ├── metadata_filter.py   # build Qdrant filter from structured intent
│   └── retriever.py         # vector search + filter + (later) rerank
├── llm/
│   ├── groq_client.py       # thin wrapper around Groq SDK
│   ├── prompts/
│   │   ├── intent_parser.py
│   │   ├── rag_answer.py
│   │   ├── followup_resolver.py
│   │   ├── headline_writer.py
│   │   └── language_router.py
│   └── pipeline.py          # ties everything together end-to-end
├── session/
│   └── state.py             # session dict + update logic
└── main.py                  # CLI test harness (types instead of speaks, per MVP)
```

```txt
# requirements.txt
groq
qdrant-client
FlagEmbedding      # for bge-m3
python-dotenv
pydantic
```

---

## 2. Qdrant collection schema

Collection name: `news_stories`

```python
# vectorstore/schema.py
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "news_stories"
VECTOR_SIZE = 1024  # bge-m3 dense output dim
DISTANCE = Distance.COSINE

# Payload fields per point:
# {
#   "story_id": str,
#   "title": str,
#   "summary": str,          # short brief used for headline reading
#   "content": str,           # full text used for "tell me more"
#   "language": str,          # "en" | "hi" | "mr" | ...
#   "country": str,
#   "state": str | None,
#   "city": str | None,
#   "topics": list[str],
#   "published_at": str,      # ISO 8601
#   "priority": float,        # 0-1, from ranking engine
#   "sources": list[str],
#   "source_count": int,
#   "cluster_id": str,        # groups duplicate articles of same event
# }
```

```python
# vectorstore/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .schema import COLLECTION_NAME, VECTOR_SIZE, DISTANCE

def get_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")

def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )
        # payload indexes for fast metadata filtering
        for field, schema in [
            ("language", "keyword"),
            ("country", "keyword"),
            ("state", "keyword"),
            ("city", "keyword"),
            ("topics", "keyword"),
            ("published_at", "datetime"),
            ("priority", "float"),
            ("cluster_id", "keyword"),
        ]:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
```

---

## 3. Embedding wrapper (bge-m3)

```python
# embeddings/embedder.py
from FlagEmbedding import BGEM3FlagModel

_model = None

def get_model():
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model

def embed_text(text: str) -> list[float]:
    model = get_model()
    out = model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"][0].tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    out = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"].tolist()
```

Notes:
- bge-m3 is multilingual and cross-lingual, so a Marathi query can retrieve an English-language article and vice versa — important since your ingested news may be mostly English/Hindi while users speak Marathi/Tamil/etc.
- For MVP, embed `title + " " + summary + " " + content[:1000]` per story — long enough for context, short enough to stay fast.

---

## 4. Ingestion (embed + upsert)

```python
# ingestion/index_story.py
import uuid
from qdrant_client.models import PointStruct
from embeddings.embedder import embed_text
from vectorstore.qdrant_client import get_client
from vectorstore.schema import COLLECTION_NAME

def index_story(story: dict):
    text_to_embed = f"{story['title']} {story.get('summary','')} {story.get('content','')[:1000]}"
    vector = embed_text(text_to_embed)
    point = PointStruct(
        id=story.get("story_id") or str(uuid.uuid4()),
        vector=vector,
        payload=story,
    )
    get_client().upsert(collection_name=COLLECTION_NAME, points=[point])
```

---

## 5. Groq client wrapper

```python
# llm/groq_client.py
import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

FAST_MODEL = "llama-3.1-8b-instant"       # intent parsing, cheap/fast tasks
GEN_MODEL  = "llama-3.3-70b-versatile"    # answer generation

def chat(system_prompt: str, user_prompt: str, model: str = GEN_MODEL,
          temperature: float = 0.3, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content
```

---

## 6. THE PROMPTS

This is the core deliverable — five prompts, each with a clear job. Keep every prompt **single-purpose**. Don't merge intent parsing and answer generation into one call; it makes hallucination control and debugging much harder.

### 6.1 Intent / NLU Parser (Section 3 & 15 of spec)

Model: `FAST_MODEL`, `json_mode=True`, `temperature=0`

```python
# llm/prompts/intent_parser.py

INTENT_SYSTEM_PROMPT = """You are the intent-understanding module of a voice-based news assistant. \
You convert a single user utterance (which may be in English, Hindi, Marathi, or a mix/code-switched \
combination of Indian languages, and may contain speech-to-text errors) into a strict JSON object. \
You do not answer the question. You do not add commentary. Output ONLY valid JSON, nothing else.

## Session context you will receive
You will be given the current session state (language, last shown stories, current topic/location) \
as JSON before the user's utterance. Use it to resolve references like "that story", "number 2", \
"what about Maharashtra", "change to Hindi" etc.

## Output schema
{
  "intent": "news" | "followup" | "weather" | "change_language" | "repeat" | "end_call" | "unclear",
  "topic": string | null,           // e.g. "cricket", "bollywood", "finance", "politics", "technology", null if not specified
  "location": {
    "city": string | null,
    "state": string | null,
    "country": string | null
  } | null,
  "time": "today" | "latest" | "this_week" | null,
  "language": string | null,        // ISO 639-1 code if the user is requesting/switching language, else null
  "reference": {
    "type": "story_number" | "current_story" | "none",
    "value": integer | null         // e.g. 2 for "tell me more about number 2"
  },
  "raw_query_for_search": string,   // a clean, normalized version of what to semantically search for
  "confidence": number              // 0-1, your confidence in this parse
}

## Rules
1. If the user's utterance doesn't specify a field, use null — do NOT invent a value.
2. If the utterance references "location" implicitly via a place name (e.g. "Mumbai ki cricket news"), \
   put it under location.city and infer state/country only if unambiguous (Mumbai -> state: Maharashtra, country: India).
3. If intent is "change_language", set the "language" field to the target language and leave other fields null \
   unless the user combined it with a new request (e.g. "Hindi mein Mumbai news do" -> intent: news, language: hi).
4. If the utterance is a follow-up ("tell me more", "who announced it", "number 2", "why is this important") \
   set intent to "followup" and fill "reference" accordingly. Do not guess a topic/location for pure follow-ups.
5. If you cannot confidently parse the utterance, set intent to "unclear" and confidence below 0.5.
6. Never fabricate a location or topic the user did not say or clearly imply.
7. Output must be a single JSON object. No markdown fences, no explanation.
"""

def build_user_prompt(session_state: dict, utterance: str) -> str:
    import json
    return f"""SESSION STATE:
{json.dumps(session_state, ensure_ascii=False)}

USER UTTERANCE:
{utterance}"""
```

### 6.2 RAG Answer Generation (Sections 8, 16, 17)

Model: `GEN_MODEL`, `temperature=0.3` (low, this is factual)

```python
# llm/prompts/rag_answer.py

RAG_SYSTEM_PROMPT = """You are the voice of a phone-based news assistant speaking to someone who is \
listening, not reading. Your job is to answer the user's news question using ONLY the retrieved \
articles provided to you. You must not use outside knowledge about current events.

## Hard rules (violating these is a critical failure)
1. Base every factual claim strictly on the RETRIEVED_ARTICLES provided below. Do not invent names, \
   numbers, dates, quotes, or outcomes that are not present in the retrieved text.
2. If the retrieved articles do not contain enough information to answer, say so plainly \
   (in the target language) instead of guessing — e.g. "Available sources don't have that detail yet."
3. Distinguish confirmed facts from claims/allegations/reports. Use hedged language ("according to X", \
   "reportedly") for anything not independently confirmed by multiple sources.
4. When multiple independent sources reported the same event, you may mention that for confidence \
   (e.g. "This was reported by N sources"), using the source_count field given to you.
5. If sources conflict, state that they conflict rather than picking one version silently.

## Style rules (this is spoken aloud over a phone call)
6. Write in {target_language}. Do not mix languages unless the source content requires a proper noun \
   that has no natural translation.
7. Keep sentences short. No bullet points, no markdown, no headers — this will be converted to speech.
8. Do NOT use the word "headline" or list numbers like "1. 2. 3." in a way that sounds like a list on a call — \
   use natural spoken transitions instead ("The next story is...", "Also,...").
9. Match the requested detail level:
   - "brief" mode: 1 sentence per story, just the core fact.
   - "detailed" mode: 3-6 sentences, including who/what/when/where, and why it matters if that's in the source.
10. End detailed answers with a natural spoken prompt for what the user can do next \
    (e.g. ask about another number, change topic, or ask for weather) ONLY if instructed to do so \
    via the include_next_step_prompt flag.

You will receive: the user's question, a detail_level flag, a target_language, and a JSON array of \
retrieved articles (each with title, summary, content, sources, source_count, published_at). \
Respond with plain spoken text only — no JSON, no markdown."""

def build_user_prompt(question: str, detail_level: str, target_language: str,
                        include_next_step_prompt: bool, retrieved_articles: list[dict]) -> str:
    import json
    return f"""USER QUESTION: {question}
DETAIL LEVEL: {detail_level}
TARGET LANGUAGE: {target_language}
INCLUDE NEXT STEP PROMPT: {include_next_step_prompt}

RETRIEVED_ARTICLES:
{json.dumps(retrieved_articles, ensure_ascii=False, indent=2)}"""
```

### 6.3 Follow-up Resolver (Section 10)

Used when `intent == "followup"`. Resolves the reference against session state, then re-uses the RAG answer prompt with the resolved story as sole context.

```python
# llm/prompts/followup_resolver.py

FOLLOWUP_SYSTEM_PROMPT = """You resolve follow-up questions in a news phone call into a clear, \
self-contained question plus the story it refers to. You do not answer the question yourself.

You will be given:
- the last list of stories shown to the user (with index numbers and story_ids)
- the currently "focused" story_id if one exists
- the user's follow-up utterance

Determine:
1. Which story_id the follow-up refers to (use "reference.value" as an index into the last shown list \
   if given, otherwise use the currently focused story_id).
2. Rewrite the follow-up into a clear standalone question a search/answer system could use, \
   e.g. "who announced it" + focused story about a policy -> \
   "Who announced the new Maharashtra government scheme?"

Output ONLY this JSON:
{
  "resolved_story_id": string | null,
  "standalone_question": string,
  "resolution_confidence": number
}

If you cannot determine which story is being referenced, set resolved_story_id to null and \
resolution_confidence below 0.5 — the caller will then ask the user to clarify."""
```

### 6.4 Headline / Brief Writer (used at ingestion time, Section 6 & 20)

Generates the short spoken headline stored per story cluster, so the RAG answer prompt at call-time doesn't have to compress on the fly.

```python
# llm/prompts/headline_writer.py

HEADLINE_SYSTEM_PROMPT = """You write a single one-sentence spoken-news headline (max ~25 words) \
summarizing a cluster of articles about the same event, in English, for storage as the "summary" field. \
Do not add opinion or speculation. Do not use clickbait phrasing. State only what is confirmed across \
the given articles. If the articles disagree on a detail, omit that detail rather than guessing.

Input: an array of article title+content pairs describing the same event.
Output: plain text, one sentence, no quotation marks, no trailing period issues, English only."""
```
Translation into the caller's language happens at answer-generation time (6.2), not here — keeps one canonical English summary per story in the DB, translated on demand. This avoids re-running clustering/summarization per language.

### 6.5 Language Router (Section 12)

Lightweight, `FAST_MODEL`, only invoked when intent_parser returns `intent == "change_language"` or language is ambiguous.

```python
# llm/prompts/language_router.py

LANGUAGE_ROUTER_SYSTEM_PROMPT = """Given a user utterance possibly mixing languages/scripts \
(e.g. Hinglish, Marathi in Latin script, code-switched Hindi-English), identify the language the user \
wants FUTURE responses in. Return ONLY an ISO 639-1 code from this supported set: \
en, hi, mr, gu, bn, ta, te, kn, ml, pa. If unclear or unsupported, return "unclear"."""
```

---

## 7. Retrieval logic (metadata filter + vector search)

```python
# retrieval/metadata_filter.py
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

def build_filter(intent: dict) -> Filter | None:
    must = []
    loc = intent.get("location") or {}
    if loc.get("city"):
        must.append(FieldCondition(key="city", match=MatchValue(value=loc["city"])))
    if loc.get("state") and not loc.get("city"):
        must.append(FieldCondition(key="state", match=MatchValue(value=loc["state"])))
    if loc.get("country") and not loc.get("city") and not loc.get("state"):
        must.append(FieldCondition(key="country", match=MatchValue(value=loc["country"])))
    if intent.get("topic"):
        must.append(FieldCondition(key="topics", match=MatchAny(any=[intent["topic"]])))
    return Filter(must=must) if must else None
```

```python
# retrieval/retriever.py
from embeddings.embedder import embed_text
from vectorstore.qdrant_client import get_client
from vectorstore.schema import COLLECTION_NAME
from .metadata_filter import build_filter

def retrieve(intent: dict, top_k: int = 5) -> list[dict]:
    query_vector = embed_text(intent["raw_query_for_search"])
    qfilter = build_filter(intent)
    results = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
    ).points
    # sort by a blend of vector score and stored priority (simple MVP heuristic)
    scored = sorted(results, key=lambda r: (0.7 * r.score + 0.3 * (r.payload.get("priority") or 0)), reverse=True)
    return [r.payload for r in scored]
```

Phase 2 improvement: add a cross-encoder reranker (`bge-reranker-v2-m3`) between retrieval and this sort step for better precision once you have real query volume to test against.

---

## 8. End-to-end pipeline (ties 6.1 → 6.2/6.3 together)

```python
# llm/pipeline.py
import json
from llm.groq_client import chat, FAST_MODEL, GEN_MODEL
from llm.prompts.intent_parser import INTENT_SYSTEM_PROMPT, build_user_prompt as intent_user_prompt
from llm.prompts.rag_answer import RAG_SYSTEM_PROMPT, build_user_prompt as rag_user_prompt
from retrieval.retriever import retrieve

def handle_turn(session_state: dict, utterance: str) -> str:
    # 1. Parse intent
    raw = chat(INTENT_SYSTEM_PROMPT, intent_user_prompt(session_state, utterance),
                model=FAST_MODEL, temperature=0, json_mode=True)
    intent = json.loads(raw)

    if intent["intent"] == "unclear" or intent["confidence"] < 0.5:
        return "clarify"  # trigger a clarifying question in target language, no LLM call needed

    if intent["intent"] == "change_language":
        session_state["language"] = intent["language"]
        return "language_switched"

    # 2. Retrieve
    articles = retrieve(intent, top_k=5 if intent["intent"] == "news" else 1)

    # 3. Generate answer
    detail_level = "detailed" if intent["intent"] == "followup" else "brief"
    answer = chat(
        RAG_SYSTEM_PROMPT.format(target_language=session_state["language"]),
        rag_user_prompt(
            question=utterance,
            detail_level=detail_level,
            target_language=session_state["language"],
            include_next_step_prompt=True,
            retrieved_articles=articles,
        ),
        model=GEN_MODEL,
        temperature=0.3,
    )

    # 4. Update session state
    session_state["previous_stories"] = articles
    session_state["current_topic"] = intent.get("topic") or session_state.get("current_topic")
    return answer
```

---

## 9. MVP build order (maps to spec Section 20)

1. Stand up Qdrant locally, create `news_stories` collection.
2. Write `embedder.py`, confirm bge-m3 embeds English + Hindi + Marathi text and cosine-similar queries return sane matches.
3. Manually seed 10–20 sample stories (hand-written JSON, skip real ingestion pipeline for now), index them.
4. Build `retriever.py`, test plain vector search with no filter.
5. Wire up `rag_answer` prompt against Groq, test "give me today's top news" and "tell me more about the second story" via a CLI loop (`main.py`), English only.
6. Add `intent_parser` prompt, replace hand-typed test calls with real NLU parsing.
7. Add `metadata_filter` filtering (city/state/topic).
8. Add multilingual: test the same flow with Hindi/Marathi utterances and target_language switching.
9. Add session state persistence (in-memory dict is fine for MVP, keyed by a fake session_id).
10. Add weather intent branch (separate simple API call, not RAG).
11. Only after all the above work in a text CLI: move to real news ingestion pipeline, then telephony/STT/TTS.

## 10. Things to test explicitly before calling this "done"

- Ask for news with no articles matching the filter → should say sources don't have it, not hallucinate.
- Ask a follow-up with no prior story shown → should ask for clarification, not guess.
- Conflicting sources on the same cluster → answer should note the conflict.
- Code-switched query ("Marathi mein Mumbai ki cricket news batao") → intent parser should still extract topic/location/language correctly.
- Rapid language switch mid-call → subsequent RAG answers must actually change language, not just acknowledge the switch.
