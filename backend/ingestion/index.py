import uuid
from typing import List, Dict, Any
from qdrant_client.models import PointStruct
from backend.embeddings.embedder import embed_batch, embed_text
from backend.retrieval.qdrant_client import get_client, ensure_collection, COLLECTION_NAME

def index_chunks(chunks: List[Dict[str, Any]], batch_size: int = 16) -> int:
    """
    Takes Phase 1 article chunks, generates BGE-M3 embeddings, and upserts into Qdrant.
    
    Args:
        chunks: List of chunk metadata dictionaries.
        batch_size: Number of texts to embed in each batch.
        
    Returns:
        Number of chunks indexed.
    """
    if not chunks:
        return 0
        
    client = get_client()
    ensure_collection(client)
    
    # Prepare texts to embed: title + first 1200 chars of text for dense contextual relevance
    texts_to_embed = [
        f"{c.get('title', '')} {c.get('text', '')[:1200]}".strip()
        for c in chunks
    ]
    
    vectors = embed_batch(texts_to_embed, batch_size=batch_size)
    points: List[PointStruct] = []
    
    for i, chunk in enumerate(chunks):
        # Create deterministic UUID from source, page, and title
        unique_key = f"{chunk.get('source', '')}:{chunk.get('page', 0)}:{chunk.get('title', '')}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))
        
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors[i],
                payload=chunk,
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
