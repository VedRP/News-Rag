"""
Interactive Phase 5 CLI -- talk to the conversational news assistant yourself.

Run: python scripts/chat_cli.py

On start, offers to (re-)fetch live news from newsdata.io and/or gnews.io into
Qdrant (replacing the earlier PDF-based ingestion per later project direction --
see backend/ingestion/newsdata_pipeline.py and gnews_pipeline.py), then drops you
into a "You: " loop. Try a flow like:

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.newsdata_client import NewsDataAPIError
from backend.ingestion.newsdata_pipeline import fetch_and_convert as newsdata_fetch
from backend.ingestion.gnews_client import GNewsAPIError
from backend.ingestion.gnews_pipeline import fetch_and_convert as gnews_fetch
from backend.ingestion.index import index_chunks
from backend.retrieval.qdrant_client import get_client, reset_collection
from backend.conversation.state import new_session
from backend.conversation.pipeline import handle_turn


def offer_indexing() -> None:
    choice = input("Fetch fresh news? [Y/n]: ").strip().lower()
    if choice == "n":
        print("[Skipping fetch -- using whatever's already indexed in Qdrant.]\n")
        return

    print("Sources: 1) newsdata.io  2) gnews.io  3) both")
    source_choice = input("Which source(s)? [3]: ").strip() or "3"
    use_newsdata = source_choice in ("1", "3")
    use_gnews = source_choice in ("2", "3")

    country = input("Country code (e.g. in, us) [in]: ").strip() or "in"
    language = input("Language code (e.g. en, hi) [en]: ").strip() or "en"
    category = input("Category, optional (e.g. sports, technology, business) []: ").strip() or None

    clear_first = input(
        "Clear previously indexed data first, for a clean test run? [Y/n]: "
    ).strip().lower()
    if clear_first != "n":
        print("Clearing existing Qdrant collection...")
        reset_collection(get_client())
        print("[OK] Collection cleared.")

    total = 0

    if use_newsdata:
        print("Fetching latest news from newsdata.io...")
        try:
            chunks = newsdata_fetch(
                country=[country], language=[language], category=[category] if category else None, size=10
            )
            total += index_chunks(chunks)
            print(f"[OK] Indexed {len(chunks)} articles from newsdata.io.")
        except (NewsDataAPIError, ValueError) as e:
            print(f"[FAILED] newsdata.io: {e}")

    if use_gnews:
        print("Fetching latest news from gnews.io...")
        try:
            chunks = gnews_fetch(country=country, language=language, category=category, max_articles=10)
            total += index_chunks(chunks)
            print(f"[OK] Indexed {len(chunks)} articles from gnews.io.")
        except (GNewsAPIError, ValueError) as e:
            print(f"[FAILED] gnews.io: {e}")

    if total == 0:
        print("[No articles indexed -- try different country/language/category, or check API keys.]\n")
    else:
        print(f"[OK] Indexed {total} articles total.\n")


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
