import sys
import json
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vectorstore.qdrant_client import get_client, ensure_collection
from ingestion.index_story import index_story

def main():
    seed_file = root_dir / "data" / "seed_stories.json"
    if not seed_file.exists():
        print(f"Error: Seed file not found at {seed_file}")
        sys.exit(1)

    with open(seed_file, "r", encoding="utf-8") as f:
        stories = json.load(f)

    print(f"Loaded {len(stories)} stories from {seed_file.name}")
    print("Ensuring Qdrant collection exists...")
    client = get_client()
    ensure_collection(client)

    print("\nStarting indexing...")
    for i, story in enumerate(stories, 1):
        story_id = story.get("story_id", f"idx_{i}")
        title = story.get("title", "Untitled")
        print(f"[{i}/{len(stories)}] Indexing ({story.get('city', 'NoCity')}, {story.get('language')}) - {title[:50]}...")
        index_story(story)

    print(f"\nSuccessfully indexed all {len(stories)} stories into Qdrant!")

if __name__ == "__main__":
    main()
