import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

COLLECTION_NAME = "news_stories"
VECTOR_SIZE = 1024
DISTANCE = Distance.COSINE

_qdrant_client = None

def get_client() -> QdrantClient:
    """Returns a singleton QdrantClient instance."""
    global _qdrant_client
    if _qdrant_client is None:
        target = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # If QDRANT_URL is an HTTP endpoint, try it; if connection fails or not running, fall back to local disk storage
        if target.startswith("http://") or target.startswith("https://"):
            try:
                test_client = QdrantClient(url=target, timeout=2.0)
                test_client.get_collections()
                _qdrant_client = test_client
            except Exception:
                # Fallback to embedded disk storage
                storage_path = os.path.abspath("./qdrant_storage")
                os.makedirs(storage_path, exist_ok=True)
                _qdrant_client = QdrantClient(path=storage_path)
        elif target == ":memory:":
            _qdrant_client = QdrantClient(location=":memory:")
        else:
            storage_path = os.path.abspath(target)
            os.makedirs(storage_path, exist_ok=True)
            _qdrant_client = QdrantClient(path=storage_path)
            
    return _qdrant_client

def ensure_collection(client: QdrantClient) -> bool:
    """Ensures the news_stories collection and its payload indexes exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )
        
        # Payload indexes for filtering
        index_fields = [
            ("language", "keyword"),
            ("country", "keyword"),
            ("state", "keyword"),
            ("city", "keyword"),
            ("location", "keyword"),
            ("topics", "keyword"),
            ("date", "keyword"),
            ("source", "keyword"),
            ("page", "integer"),
        ]
        for field, schema in index_fields:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass
        return True
    return False
