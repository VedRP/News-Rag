from embeddings.embedder import embed_text
from vectorstore.qdrant_client import get_client
from vectorstore.schema import COLLECTION_NAME
from .metadata_filter import build_filter

def retrieve(intent: dict, top_k: int = 5) -> list[dict]:
    query_vector = embed_text(intent["raw_query_for_search"])
    qfilter = build_filter(intent)
    results = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
    ).points
    # sort by a blend of vector score and stored priority (simple MVP heuristic)
    scored = sorted(results, key=lambda r: (0.7 * r.score + 0.3 * (r.payload.get("priority") or 0)), reverse=True)
    return [r.payload for r in scored]
