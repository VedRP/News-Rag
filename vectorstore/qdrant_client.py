import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from .schema import COLLECTION_NAME, VECTOR_SIZE, DISTANCE

load_dotenv()

_qdrant_client = None

def get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        target = os.getenv("QDRANT_URL", "http://localhost:6333")
        if target.startswith("http://") or target.startswith("https://"):
            _qdrant_client = QdrantClient(url=target)
        elif target == ":memory:":
            _qdrant_client = QdrantClient(location=":memory:")
        else:
            _qdrant_client = QdrantClient(path=target)
    return _qdrant_client


def ensure_collection(client: QdrantClient) -> bool:
    """Ensures the collection exists. Returns True if created, False if already existed."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )
        # payload indexes for fast metadata filtering
        for field, schema in [
            ("language", "keyword"),
            ("country", "keyword"),
            ("state", "keyword"),
            ("city", "keyword"),
            ("topics", "keyword"),
            ("published_at", "datetime"),
            ("priority", "float"),
            ("cluster_id", "keyword"),
        ]:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        return True
    return False
