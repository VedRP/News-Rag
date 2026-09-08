from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

def build_filter(intent: dict) -> Filter | None:
    must = []
    loc = intent.get("location") or {}
    if loc.get("city"):
        must.append(FieldCondition(key="city", match=MatchValue(value=loc["city"])))
    if loc.get("state") and not loc.get("city"):
        must.append(FieldCondition(key="state", match=MatchValue(value=loc["state"])))
    if loc.get("country") and not loc.get("city") and not loc.get("state"):
        must.append(FieldCondition(key="country", match=MatchValue(value=loc["country"])))
    if intent.get("topic"):
        must.append(FieldCondition(key="topics", match=MatchAny(any=[intent["topic"]])))
    return Filter(must=must) if must else None
