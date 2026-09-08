import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from retrieval.retriever import retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    fake_intent = {
        "topic": "cricket",
        "location": {
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India"
        },
        "raw_query_for_search": "cricket news mumbai"
    }

    print("--- Running Test Retriever ---")
    print(f"Intent query: {fake_intent['raw_query_for_search']}")
    print(f"Filter: topic={fake_intent['topic']}, city={fake_intent['location']['city']}\n")

    try:
        articles = retrieve(fake_intent, top_k=5)
        print(f"Retrieved {len(articles)} articles:\n")
        for i, art in enumerate(articles, 1):
            print(f"{i}. Title: {art.get('title')}")
            print(f"   City: {art.get('city')} | Topic: {art.get('topics')} | Priority: {art.get('priority')}")
            print(f"   Summary: {art.get('summary')}\n")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
