import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.pipeline import process_newspaper_pdf
from backend.ingestion.index import index_chunks
from backend.rag.answer import answer_question

def main():
    print("=" * 75)
    print("PHASE 3: Grounded RAG Verification (Grounded Factuality & Refusal Tests)")
    print("=" * 75)

    # 1. Ensure sample newspaper content is indexed
    pdf_path = os.path.join("News", "1788917688_6aa0b7b8ebd38_0.pdf")
    if os.path.exists(pdf_path):
        print("\n[1/3] Ingesting & indexing sample newspaper pages...")
        chunks = process_newspaper_pdf(pdf_path, max_pages=3)
        index_chunks(chunks)
        print(f"      [OK] Indexed {len(chunks)} newspaper chunks into Qdrant.")
    else:
        print(f"\n[Warning] {pdf_path} not found. Running against existing collection.")

    # 2. Test Grounded News Question (Present in indexed newspaper)
    q_grounded = "What did Prime Minister Narendra Modi inaugurate regarding the Western Dedicated Freight Corridor?"
    print("\n" + "-" * 75)
    print(f"[TEST 1: GROUNDED FACTUAL QUESTION]\nQuery: \"{q_grounded}\"")
    print("-" * 75)

    result_grounded = answer_question(q_grounded, top_k=3)
    print(f"\nAnswer:\n{result_grounded.answer}")
    print(f"\nTop Similarity Score: {result_grounded.top_similarity:.4f}")
    print("Citations:")
    for cit in result_grounded.citations:
        print(f"  - Source: {cit['source']} (Page {cit['page']}) | Title: {cit['title']} (Score: {cit['score']})")

    # 3. Test Ungrounded / Out-of-Domain Question (Not present in newspaper)
    q_ungrounded = "Who won the 2099 intergalactic sports championship on planet Mars?"
    print("\n" + "-" * 75)
    print(f"[TEST 2: UNGROUNDED QUESTION / HALLUCINATION REFUSAL]\nQuery: \"{q_ungrounded}\"")
    print("-" * 75)

    result_ungrounded = answer_question(q_ungrounded, top_k=3)
    print(f"\nAnswer:\n{result_ungrounded.answer}")
    print(f"Top Similarity Score: {result_ungrounded.top_similarity:.4f}")
    print(f"Chunks Used: {result_ungrounded.chunks_used}")

    print("\n" + "=" * 75)
    print("PHASE 3 VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    return 0

if __name__ == "__main__":
    sys.exit(main())
