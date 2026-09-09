import sys
import os
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.pipeline import process_newspaper_pdf

def main():
    pdf_path = os.path.join("News", "1788917688_6aa0b7b8ebd38_0.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return 1

    print("=" * 70)
    print(f"Running Phase 1 PDF Ingestion Pipeline on: {pdf_path}")
    print("=" * 70)

    # Process first 5 pages for comprehensive verification
    chunks = process_newspaper_pdf(pdf_path, max_pages=5)
    
    print(f"\n[OK] Processed 5 pages.")
    print(f"[OK] Total article chunks extracted: {len(chunks)}")
    
    if not chunks:
        print("ERROR: No chunks extracted!")
        return 1
        
    avg_words = sum(c["word_count"] for c in chunks) / len(chunks)
    print(f"[OK] Average words per chunk: {avg_words:.1f}")

    print("\n" + "=" * 70)
    print("SAMPLE EXTRACTED CHUNKS & METADATA")
    print("=" * 70)

    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk #{i} ---")
        print(f"Title:    {chunk['title']}")
        print(f"Source:   {chunk['source']}")
        print(f"Page:     {chunk['page']}")
        print(f"Date:     {chunk['date']}")
        print(f"Language: {chunk['language']}")
        print(f"Location: {chunk['location']}")
        print(f"Topics:   {chunk['topics']}")
        print(f"Word Ct:  {chunk['word_count']}")
        print("-" * 40)
        # Print first 250 chars of text
        snippet = chunk['text'][:250].replace('\n', ' ')
        print(f"Text:     {snippet}...")

    print("\n" + "=" * 70)
    print("PHASE 1 INGESTION VERIFICATION COMPLETE")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
