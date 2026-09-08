import sys
from pathlib import Path
import numpy as np

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from embeddings.embedder import embed_text

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    print("Testing BAAI/bge-m3 embedding wrapper...")

    sentences = {
        "English": "India won the cricket match against Australia in Mumbai.",
        "Hindi": "भारत ने मुंबई में ऑस्ट्रेलिया के खिलाफ क्रिकेट मैच जीता।",
        "Marathi": "भारताने मुंबईत ऑस्ट्रेलियाविरुद्धचा क्रिकेट सामना जिंकला.",
    }

    embeddings = {}
    for lang, text in sentences.items():
        print(f"Embedding ({lang}): {text}")
        vec = embed_text(text)
        embeddings[lang] = vec
        print(f"  -> Vector dimension: {len(vec)}")

    print("\n--- Cosine Similarities Between Pairs ---")
    pairs = [
        ("English", "Hindi"),
        ("English", "Marathi"),
        ("Hindi", "Marathi"),
    ]

    for l1, l2 in pairs:
        sim = cosine_similarity(embeddings[l1], embeddings[l2])
        print(f"Similarity ({l1} <-> {l2}): {sim:.4f}")

if __name__ == "__main__":
    main()
