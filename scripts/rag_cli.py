import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.rag.answer import answer_question

def main():
    print("=" * 70)
    print("AI News Assistant — Phase 3 Grounded RAG CLI")
    print("Type your question to search the indexed newspaper.")
    print("Type 'exit' or 'quit' to close.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("\nRetrieving evidence and generating grounded answer...")
        result = answer_question(user_input, top_k=3)

        print("\n" + "=" * 50)
        print("GROUNDED ANSWER:")
        print(result.answer)
        print("=" * 50)

        if result.citations:
            print("\nCITATIONS & SOURCES USED:")
            for cit in result.citations:
                print(f"  • Source: {cit['source']} | Page: {cit['page']} | Date: {cit['date']}")
                print(f"    Title:  {cit['title']} (Cosine Similarity: {cit['score']:.4f})")
        else:
            print(f"(No relevant newspaper sources matched above similarity threshold. Score: {result.top_similarity:.4f})")

if __name__ == "__main__":
    main()
