from FlagEmbedding import BGEM3FlagModel

_model = None

def get_model():
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model

def embed_text(text: str) -> list[float]:
    model = get_model()
    out = model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"][0].tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    out = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"].tolist()
