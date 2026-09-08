# IDE Coding Prompts — RAG + LLM Subsystem

How to use this file: paste one prompt at a time into your IDE's AI assistant (Claude Code, Cursor, Copilot Chat, etc.), in order. Each prompt assumes the previous ones are already done and working. Don't skip ahead — the pipeline is intentionally built bottom-up so you can test each layer in isolation before wiring it together. Where a prompt references `plan.md`, attach/paste that file alongside so the assistant has the exact schemas and prompt text to use instead of inventing its own.

---

## Prompt 0 — Project scaffold

```
Set up a new Python project called "rag-llm" for the RAG + LLM subsystem described in
plan.md (attached). Create the exact folder/file structure from section 1 of plan.md,
with empty placeholder files (just a module docstring in each). Create a requirements.txt
with: groq, qdrant-client, FlagEmbedding, python-dotenv, pydantic, torch. Create a .env.example
with GROQ_API_KEY= and QDRANT_URL=http://localhost:6333. Create a .gitignore for Python
(venv, __pycache__, .env). Do not implement any logic yet — just the skeleton.
```

---

## Prompt 1 — Qdrant connection + collection

```
Implement vectorstore/qdrant_client.py and vectorstore/schema.py exactly as specified in
section 2 of plan.md: collection name "news_stories", 1024-dim cosine vectors, payload
indexes on language, country, state, city, topics, published_at, priority, cluster_id.
Then write a small script scripts/init_qdrant.py that connects, calls ensure_collection(),
and prints whether the collection was created or already existed. I'll run
`docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`
myself before running it — don't try to start Docker from code.
```

---

## Prompt 2 — Embedding wrapper

```
Implement embeddings/embedder.py using BAAI/bge-m3 via the FlagEmbedding library, exactly
as specified in section 3 of plan.md (embed_text and embed_batch, dense vectors only,
lazy-loaded singleton model, fp16). Then write a quick manual test script
scripts/test_embedder.py that embeds three short sentences — one English, one Hindi, one
Marathi, all about the same topic (e.g. cricket) — and prints the cosine similarity between
each pair, so I can eyeball that cross-lingual similarity looks reasonable. Use numpy for
the cosine similarity calc.
```

---

## Prompt 3 — Seed data + ingestion

```
Implement ingestion/index_story.py per section 4 of plan.md. Then create data/seed_stories.json
with 15 hand-written sample news stories matching the payload schema in plan.md section 2
(story_id, title, summary, content, language, country, state, city, topics, published_at,
priority, sources, source_count, cluster_id). Mix topics (cricket, bollywood, finance,
politics), mix cities (Mumbai, Pune, Delhi), mostly English content but 2-3 in Hindi. Then
write scripts/seed.py that loads the JSON and calls index_story() for each entry, with a
progress print per story.
```

---

## Prompt 4 — Retrieval (no LLM yet)

```
Implement retrieval/metadata_filter.py and retrieval/retriever.py exactly per section 7 of
plan.md. Write scripts/test_retriever.py that takes a hardcoded fake "intent" dict (topic:
"cricket", location: {city: "Mumbai"}, raw_query_for_search: "cricket news mumbai") and
prints the titles + scores of what comes back, so I can confirm filtering and vector search
both work before adding any LLM calls.
```

---

## Prompt 5 — Groq client + RAG answer prompt

```
Implement llm/groq_client.py and llm/prompts/rag_answer.py exactly as written in sections
5 and 6.2 of plan.md — do not paraphrase or shorten the system prompt, use it verbatim.
Then write scripts/test_rag_answer.py: take the retrieval results from Prompt 4's test
script, call the RAG answer prompt with detail_level="brief", target_language="English",
include_next_step_prompt=False, and print the spoken-style answer. Confirm it only uses
facts present in the seed data.
```

---

## Prompt 6 — Intent parser

```
Implement llm/prompts/intent_parser.py exactly as written in section 6.1 of plan.md, verbatim
system prompt. Write scripts/test_intent_parser.py that runs these five test utterances
through it with an empty session_state ({}): "Give me today's cricket news from Mumbai",
"Tell me more about number 2", "Change to Hindi", "Marathi mein Mumbai ki cricket news
batao", "asdkj random gibberish". Print the parsed JSON for each and flag whether
intent/confidence look right, especially that the gibberish one comes back "unclear".
```

---

## Prompt 7 — Full pipeline wiring

```
Implement llm/pipeline.py exactly per section 8 of plan.md, wiring together intent_parser
-> retriever -> rag_answer. Then implement session/state.py with a simple in-memory dict
keyed by session_id (see the session state shape in section 11 of the master spec — I'll
paste that section too). Write main.py as an interactive CLI loop: prompt "You: ", call
handle_turn(session_state, utterance), print "Assistant: <answer>", loop until user types
"quit". Seed one session_state with language: "en" at the start. Handle the "clarify" and
"language_switched" sentinel returns from handle_turn with simple hardcoded English
fallback lines for now.
```

---

## Prompt 8 — Follow-up resolver

```
Implement llm/prompts/followup_resolver.py exactly per section 6.3 of plan.md. Wire it into
pipeline.py: when intent_parser returns intent == "followup", call the follow-up resolver
with session_state["previous_stories"] and session_state.get("focused_story_id"), get back
resolved_story_id + standalone_question, fetch that single story's full payload from Qdrant
by ID (add a get_story_by_id helper to retrieval/retriever.py), and pass it as the sole
retrieved_articles entry into rag_answer with detail_level="detailed". If
resolution_confidence < 0.5, return a clarification message instead of calling rag_answer.
Update main.py's CLI loop to test: ask for cricket news, then ask "tell me more about
number 1", then ask a nonsense follow-up with no prior context to confirm it asks for
clarification.
```

---

## Prompt 9 — Language routing + multilingual test

```
Implement llm/prompts/language_router.py per section 6.5 of plan.md. In pipeline.py, when
intent_parser's intent is "change_language" but the "language" field came back null or
"unclear", call the language router as a fallback before failing. Update main.py so that
after a language switch, all subsequent rag_answer calls actually pass the new
target_language through. Test manually in the CLI: ask for news in English, say "change to
Marathi", ask for news again, confirm the response is actually in Marathi and not just
acknowledging the switch in English.
```

---

## Prompt 10 — Weather branch (non-RAG)

```
Add a simple weather intent branch to pipeline.py per section 13 of the master spec: when
intent_parser returns intent == "weather", check session_state for an existing city; if
present call a stub function get_weather(city) that returns a hardcoded fake forecast dict
for now (I'll wire a real weather API later), and generate a one-sentence spoken response
via a small new prompt (add llm/prompts/weather_answer.py, same "spoken over a phone,
single language, no markdown" style rules as rag_answer). If no city is in session_state,
return a clarifying question asking which location, and store the next answer as the
session's location once given.
```

---

## Prompt 11 — Error handling & guardrail tests

```
Go through pipeline.py and every prompts/*.py file and add proper error handling: what
happens if Groq's response isn't valid JSON when json_mode is expected (retry once, then
fail gracefully with a spoken "having trouble understanding" fallback), what happens if
Qdrant returns zero results (rag_answer should already handle "insufficient info" per its
system prompt — write a test to confirm it actually does), and what happens if the Groq API
call itself throws (timeout/rate limit) — catch and return a generic spoken apology. Write
scripts/test_guardrails.py covering: empty retrieval results, malformed intent JSON
(simulate by mocking), and a Groq exception (mock the client to raise).
```

---

## Notes for whoever's driving the IDE assistant

- After each prompt, actually run the corresponding test script before moving to the next prompt. These prompts build directly on each other's files.
- If the IDE assistant starts inventing its own prompt wording instead of using plan.md's verbatim system prompts, stop it and point it back at the file — the hallucination-control and style rules in those prompts are load-bearing, not filler.
- Keep `GROQ_API_KEY` out of anything you commit; `.env` is gitignored from Prompt 0.
- Docker/Qdrant must be running before Prompts 1, 3, 4, 7+ — the assistant won't start it for you.
