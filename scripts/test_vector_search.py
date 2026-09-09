import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.pipeline import process_newspaper_pdf
from backend.ingestion.index import index_chunks
from backend.embeddings.embedder import embed_text
from backend.retrieval.qdrant_client import get_client, ensure_collection, COLLECTION_NAME

def main():
    pdf_path = os.path.join("News", "1788917688_6aa0b7b8ebd38_0.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return 1

    print("=" * 70)
    print("PHASE 2: Embeddings + Qdrant Vector Retrieval Test")
    print("=" * 70)

    # 1. Process sample pages from the real newspaper PDF
    print("\n1. Ingesting sample pages from newspaper PDF...")
    chunks = process_newspaper_pdf(pdf_path, max_pages=3)
    print(f"   [OK] Extracted {len(chunks)} chunks.")

    # 2. Embed and Index into Qdrant
    print("\n2. Embedding chunks with BAAI/bge-m3 and indexing into Qdrant...")
    count = index_chunks(chunks)
    print(f"   [OK] Successfully indexed {count} points into Qdrant collection '{COLLECTION_NAME}'.")

    # 3. Test Vector Search with hand-typed queries
    test_queries = [
        "Western Dedicated Freight Corridor inaugurated by Prime Minister",
        "Opposition politics and election campaigns in Delhi",
    ]

    client = get_client()

    for q in test_queries:
        print("\n" + "-" * 70)
        print(f"QUERY: \"{q}\"")
        print("-" * 70)
        
        query_vector = embed_text(q)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=2,
            with_payload=True,
        ).points

        if not results:
            print("   No results returned.")
            continue

        for rank, res in enumerate(results, 1):
            payload = res.payload or {}
            score = res.score
            title = payload.get("title", "No Title")
            page = payload.get("page", "?")
            source = payload.get("source", "?")
            snippet = payload.get("text", "")[:200].replace("\n", " ")
            print(f"\n   Match #{rank} (Cosine Similarity: {score:.4f}):")
            print(f"   Title:   {title}")
            print(f"   Source:  {source} (Page {page})")
            print(f"   Snippet: {snippet}...")

    print("\n" + "=" * 70)
    print("PHASE 2 VECTOR RETRIEVAL TEST COMPLETE")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
