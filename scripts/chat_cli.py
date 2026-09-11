"""
Interactive Phase 5 CLI -- talk to the conversational news assistant yourself.

Run: python scripts/chat_cli.py

On start, offers to (re-)index newspaper PDFs from News/ into Qdrant, then drops
you into a "You: " loop. Try a flow like:

    You: Give me politics news.
    (numbered list of stories comes back)
    You: Tell me more about number 2
    (detailed, cited answer about that specific story)
    You: who announced it
    (follow-up resolved against the same story, no number needed)
    You: repeat that
    You: Change to Hindi
    You: quit

Commands: "quit" / "exit" ends the session. "state" prints the raw session
state (language, current topic/story, history length) so you can see what
the assistant is tracking.
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.extract import GarbledPDFTextError
from backend.ingestion.pipeline import process_newspaper_pdf
from backend.ingestion.index import index_chunks
from backend.retrieval.qdrant_client import get_client, reset_collection
from backend.conversation.state import new_session
from backend.conversation.pipeline import handle_turn


def offer_indexing() -> None:
    pdfs = sorted(glob.glob(os.path.join("News", "*.pdf")))
    if not pdfs:
        print("[No PDFs found in News/ -- running against whatever is already indexed in Qdrant.]")
        return

    print("Newspaper PDFs found in News/:")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. {os.path.basename(p)}")
    print("  0. Skip -- use whatever's already indexed in Qdrant")

    choice = input("Index which one? [1]: ").strip() or "1"
    if choice == "0":
        return
    try:
        idx = int(choice) - 1
        pdf_path = pdfs[idx]
    except (ValueError, IndexError):
        print("Invalid choice, skipping indexing.")
        return

    max_pages_raw = input("How many pages to ingest? [10]: ").strip() or "10"
    try:
        max_pages = int(max_pages_raw)
    except ValueError:
        max_pages = 10

    clear_first = input(
        "Clear previously indexed data first, for a clean test run? [Y/n]: "
    ).strip().lower()
    if clear_first != "n":
        print("Clearing existing Qdrant collection...")
        reset_collection(get_client())
        print("[OK] Collection cleared.")

    print(f"Ingesting {os.path.basename(pdf_path)} (up to {max_pages} pages)...")
    try:
        chunks = process_newspaper_pdf(pdf_path, max_pages=max_pages)
    except GarbledPDFTextError as e:
        print(f"[SKIPPED] {e}")
        print("Pick a different PDF, or choose 0 to use whatever's already indexed.\n")
        return
    index_chunks(chunks)
    print(f"[OK] Indexed {len(chunks)} chunks.\n")


def print_state(state) -> None:
    print(
        f"  language={state.language!r} current_topic={state.current_topic!r} "
        f"current_story={(state.current_story or {}).get('title')!r} "
        f"previous_stories={len(state.previous_stories)} "
        f"history_turns={len(state.conversation_history)}"
    )


def main() -> int:
    print("=" * 75)
    print("Phase 5 -- Conversational News Assistant (manual test CLI)")
    print("=" * 75)
    offer_indexing()

    print("Type a news request, a follow-up, or 'quit' to exit. Type 'state' to inspect session state.\n")

    state = new_session("cli-session")
    while True:
        try:
            utterance = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if not utterance:
            continue
        if utterance.lower() in ("quit", "exit", "q"):
            print("Ending session.")
            break
        if utterance.lower() == "state":
            print_state(state)
            continue

        result = handle_turn(state, utterance)
        print(f"\nAssistant [{result.kind}]:")
        print(result.spoken_answer)
        print()

        if result.kind == "end":
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
