from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "news_stories"
VECTOR_SIZE = 1024  # bge-m3 dense output dim
DISTANCE = Distance.COSINE

# Payload fields per point:
# {
#   "story_id": str,
#   "title": str,
#   "summary": str,          # short brief used for headline reading
#   "content": str,           # full text used for "tell me more"
#   "language": str,          # "en" | "hi" | "mr" | ...
#   "country": str,
#   "state": str | None,
#   "city": str | None,
#   "topics": list[str],
#   "published_at": str,      # ISO 8601
#   "priority": float,        # 0-1, from ranking engine
#   "sources": list[str],
#   "source_count": int,
#   "cluster_id": str,        # groups duplicate articles of same event
# }
