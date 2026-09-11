import os
import shutil
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

COLLECTION_NAME = "news_stories"
VECTOR_SIZE = 1024
DISTANCE = Distance.COSINE

_qdrant_client = None
_local_storage_path = None  # set when _qdrant_client is embedded/disk-backed, else None

def get_client() -> QdrantClient:
    """Returns a singleton QdrantClient instance."""
    global _qdrant_client, _local_storage_path
    if _qdrant_client is None:
        target = os.getenv("QDRANT_URL", "http://localhost:6333")

        # If QDRANT_URL is an HTTP endpoint, try it; if connection fails or not running, fall back to local disk storage
        if target.startswith("http://") or target.startswith("https://"):
            try:
                test_client = QdrantClient(url=target, timeout=2.0)
                test_client.get_collections()
                _qdrant_client = test_client
                _local_storage_path = None
            except Exception:
                # Fallback to embedded disk storage
                storage_path = os.path.abspath("./qdrant_storage")
                os.makedirs(storage_path, exist_ok=True)
                _qdrant_client = QdrantClient(path=storage_path)
                _local_storage_path = storage_path
        elif target == ":memory:":
            _qdrant_client = QdrantClient(location=":memory:")
            _local_storage_path = None
        else:
            storage_path = os.path.abspath(target)
            os.makedirs(storage_path, exist_ok=True)
            _qdrant_client = QdrantClient(path=storage_path)
            _local_storage_path = storage_path

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


def reset_collection(client: QdrantClient) -> QdrantClient:
    """Drops and recreates the collection, discarding all previously indexed points.

    Useful for local dev/testing so old chunks from earlier ad-hoc indexing runs
    (e.g. mismatched schemas, half-finished test data -- this bit us during Phase 5
    manual testing) don't silently mix into retrieval results for a fresh session.

    Returns the client to use afterwards (== the module singleton). For the embedded/
    disk-backed local client, the qdrant-client library's delete_collection() can leave
    the collection's storage.sqlite file locked on Windows, so shutil.rmtree() inside it
    silently no-ops (ignore_errors=True) and the next create_collection() call just
    reopens the old, undeleted data. Work around that by fully closing the client first
    (releasing the sqlite handles) and deleting the on-disk collection directory
    ourselves, then rebuilding the singleton from scratch.
    """
    global _qdrant_client

    if _local_storage_path is not None:
        try:
            client.close()
        except Exception:
            pass
        collection_dir = os.path.join(_local_storage_path, "collection", COLLECTION_NAME)
        shutil.rmtree(collection_dir, ignore_errors=True)
        _qdrant_client = None
        fresh_client = get_client()
        ensure_collection(fresh_client)
        return fresh_client

    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION_NAME)
    ensure_collection(client)
    return client
