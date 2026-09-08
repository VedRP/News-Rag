import uuid
from qdrant_client.models import PointStruct
from embeddings.embedder import embed_text
from vectorstore.qdrant_client import get_client
from vectorstore.schema import COLLECTION_NAME

def index_story(story: dict):
    text_to_embed = f"{story['title']} {story.get('summary','')} {story.get('content','')[:1000]}"
    vector = embed_text(text_to_embed)
    
    # Use existing story_id if valid UUID or generate deterministic/random UUID
    raw_id = story.get("story_id")
    point_id = None
    if raw_id:
        try:
            point_id = str(uuid.UUID(str(raw_id)))
        except (ValueError, AttributeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))
    else:
        point_id = str(uuid.uuid4())
        
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload=story,
    )
    get_client().upsert(collection_name=COLLECTION_NAME, points=[point])
