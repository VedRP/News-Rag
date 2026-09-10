"""
Phase 4 end-to-end verification: raw utterance -> intent parse -> validation ->
metadata-filtered + vector retrieval -> grounded generation.

Run: python scripts/test_phase4_structured_rag.py

Reuses Phase 1/2 ingestion+indexing against a real newspaper PDF, then drives
backend.rag.answer.answer_structured_question() with a mix of a topic-filtered
query, a location-filtered query, and a deliberately ambiguous utterance that
should be routed to a clarifying response instead of a guessed answer.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.pipeline import process_newspaper_pdf
from backend.ingestion.index import index_chunks
from backend.rag.answer import answer_structured_question


def run_case(label: str, utterance: str, top_k: int = 4) -> None:
    print("\n" + "-" * 75)
    print(f"[{label}]\nUtterance: \"{utterance}\"")
    print("-" * 75)
    result = answer_structured_question(utterance, top_k=top_k)
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nChunks used: {result.chunks_used} | Top similarity: {result.top_similarity:.4f}")
    if result.citations:
        print("Citations:")
        for cit in result.citations:
            print(f"  - {cit['source']} (Page {cit['page']}) | {cit['title']} (Score: {cit['score']})")


def main() -> int:
    print("=" * 75)
    print("PHASE 4: Structured Intent -> Filtered Retrieval -> Grounded Answer")
    print("=" * 75)

    pdf_path = os.path.join("News", "1788917688_6aa0b7b8ebd38_0.pdf")
    if os.path.exists(pdf_path):
        print("\n[1/2] Ingesting & indexing sample newspaper pages...")
        chunks = process_newspaper_pdf(pdf_path, max_pages=10)
        index_chunks(chunks)
        print(f"      [OK] Indexed {len(chunks)} newspaper chunks into Qdrant.")
    else:
        print(f"\n[Warning] {pdf_path} not found. Running against existing collection.")

    print("\n[2/2] Running structured queries...")

    run_case(
        "TEST 1: TOPIC-FILTERED QUERY",
        "Give me politics news.",
    )

    run_case(
        "TEST 2: LOCATION-FILTERED QUERY",
        "What's happening in Delhi?",
    )

    run_case(
        "TEST 3: AMBIGUOUS UTTERANCE (should clarify, not guess)",
        "asdkj random gibberish blah blah",
    )

    print("\n" + "=" * 75)
    print("PHASE 4 END-TO-END VERIFICATION COMPLETE")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
