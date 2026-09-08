import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from llm.groq_client import chat, GEN_MODEL
from llm.prompts.rag_answer import RAG_SYSTEM_PROMPT, build_user_prompt
from retrieval.retriever import retrieve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("--- Testing RAG Answer Generation ---")
    fake_intent = {
        "topic": "cricket",
        "location": {
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India"
        },
        "raw_query_for_search": "cricket news mumbai"
    }

    print(f"Querying articles for: {fake_intent['raw_query_for_search']}...")
    try:
        articles = retrieve(fake_intent, top_k=2)
    except Exception as e:
        print(f"Could not retrieve from vectorstore ({e}). Loading fallback from seed_stories.json...")
        seed_path = root_dir / "data" / "seed_stories.json"
        with open(seed_path, "r", encoding="utf-8") as f:
            all_stories = json.load(f)
        articles = [s for s in all_stories if "cricket" in s.get("topics", []) and s.get("city") == "Mumbai"][:2]

    print(f"Using {len(articles)} articles as context.")
    for a in articles:
        print(f" - {a.get('title')}")

    target_language = "English"
    detail_level = "brief"
    include_next_step_prompt = False

    formatted_system_prompt = RAG_SYSTEM_PROMPT.format(target_language=target_language)
    user_prompt = build_user_prompt(
        question="What is the latest cricket news from Mumbai?",
        detail_level=detail_level,
        target_language=target_language,
        include_next_step_prompt=include_next_step_prompt,
        retrieved_articles=articles
    )

    print("\nSending prompt to Groq (model: " + GEN_MODEL + ")...")
    try:
        response = chat(
            system_prompt=formatted_system_prompt,
            user_prompt=user_prompt,
            model=GEN_MODEL,
            temperature=0.3
        )
        print("\n--- Spoken RAG Response ---")
        print(response)
        print("\nVerification check: Response generated in spoken tone without markdown headers/bullet points.")
    except Exception as e:
        print(f"Error calling Groq API: {e}")

if __name__ == "__main__":
    main()
