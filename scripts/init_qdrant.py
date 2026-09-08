import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vectorstore.qdrant_client import get_client, ensure_collection
from vectorstore.schema import COLLECTION_NAME

def main():
    print("Connecting to Qdrant...")
    try:
        client = get_client()
        created = ensure_collection(client)
        if created:
            print(f"Collection '{COLLECTION_NAME}' was successfully created with payload indexes.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already existed.")
    except Exception as e:
        print(f"Error connecting to Qdrant or ensuring collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
