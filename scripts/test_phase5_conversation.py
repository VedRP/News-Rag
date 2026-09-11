"""
Phase 5 automated verification: multi-turn conversation with follow-up resolution.

Run: python scripts/test_phase5_conversation.py

Indexes real newspaper pages, then drives a scripted multi-turn conversation through
backend.conversation.pipeline.handle_turn(), asserting that:
- a fresh news request returns a numbered story listing
- "tell me more about number 2" resolves to exactly that story (not a guess)
- an un-numbered follow-up ("who announced it") resolves via current_story
- "repeat that" replays the prior answer verbatim
- "change to Hindi" updates session state
- a nonsense utterance is routed to clarification, not a guess
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.pipeline import process_newspaper_pdf
from backend.ingestion.index import index_chunks
from backend.conversation.state import new_session
from backend.conversation.pipeline import handle_turn

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' -- ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("=" * 75)
    print("PHASE 5: Conversation State + Follow-up Resolution")
    print("=" * 75)

    pdf_path = os.path.join("News", "1788917688_6aa0b7b8ebd38_0.pdf")
    if os.path.exists(pdf_path):
        print("\n[1/2] Ingesting & indexing sample newspaper pages...")
        chunks = process_newspaper_pdf(pdf_path, max_pages=10)
        index_chunks(chunks)
        print(f"      [OK] Indexed {len(chunks)} newspaper chunks into Qdrant.")
    else:
        print(f"\n[Warning] {pdf_path} not found. Running against existing collection.")

    print("\n[2/2] Running scripted multi-turn conversation...\n")
    state = new_session("phase5-test")

    print("--- Turn 1: fresh news request ---")
    t1 = handle_turn(state, "Give me politics news.")
    print(t1.spoken_answer)
    check("Turn 1 is a news_listing", t1.kind == "news_listing")
    check("Turn 1 produced at least 2 stories", len(t1.stories) >= 2, f"got {len(t1.stories)}")

    if len(t1.stories) < 2:
        print("\n[ABORT] Not enough stories retrieved to test follow-up resolution.")
        return 1

    target_story = t1.stories[1]  # story "number 2"
    print("\n--- Turn 2: \"tell me more about number 2\" ---")
    t2 = handle_turn(state, "Tell me more about number 2")
    print(t2.spoken_answer)
    check("Turn 2 is a story_detail", t2.kind == "story_detail")
    check(
        "Turn 2 resolved to story #2's actual source/page",
        bool(t2.citation)
        and t2.citation.get("source") == target_story.get("source")
        and t2.citation.get("page") == target_story.get("page"),
        f"citation={t2.citation}, expected source/page from story #2={target_story.get('source')}/{target_story.get('page')}",
    )
    check(
        "current_story updated to story #2",
        state.current_story is not None and state.current_story.get("number") == 2,
    )

    print("\n--- Turn 3: un-numbered follow-up \"who announced it\" ---")
    t3 = handle_turn(state, "who announced it")
    print(t3.spoken_answer)
    check("Turn 3 is a story_detail", t3.kind == "story_detail")
    check(
        "Turn 3 stayed on the same story via current_story (no explicit number given)",
        bool(t3.citation) and t3.citation.get("source") == target_story.get("source")
        and t3.citation.get("page") == target_story.get("page"),
    )

    print("\n--- Turn 4: \"repeat that\" ---")
    t4 = handle_turn(state, "repeat that")
    print(t4.spoken_answer)
    check("Turn 4 is a repeat", t4.kind == "repeat")
    check("Turn 4 replayed turn 3's answer verbatim", t4.spoken_answer == t3.spoken_answer)

    print("\n--- Turn 5: \"Change to Hindi\" ---")
    t5 = handle_turn(state, "Change to Hindi")
    print(t5.spoken_answer)
    check("Turn 5 is a language_ack", t5.kind == "language_ack")
    check("Session language updated to hindi", state.language.lower() == "hindi", f"got {state.language!r}")

    print("\n--- Turn 6: nonsense utterance ---")
    t6 = handle_turn(state, "asdkj random gibberish blah blah")
    print(t6.spoken_answer)
    check("Turn 6 is a clarify (not a guessed answer)", t6.kind == "clarify")

    print("\n--- Conversation history length ---")
    check("6 turns recorded in conversation_history", len(state.conversation_history) == 6, f"got {len(state.conversation_history)}")

    print("\n" + "=" * 75)
    if FAILURES:
        print(f"PHASE 5 VERIFICATION: {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        print("=" * 75)
        return 1

    print("PHASE 5 VERIFICATION COMPLETE -- all checks passed")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
