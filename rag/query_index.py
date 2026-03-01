import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "vector_store"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = faiss.read_index(f"{INDEX_DIR}/rules.index")
chunks = np.load(f"{INDEX_DIR}/chunks.npy", allow_pickle=True)

def search(query, k=3):

    embedding = model.encode([query])

    distances, indices = index.search(embedding, k)

    return [chunks[i] for i in indices[0]]

if __name__ == "__main__":

    results = search("large dog airplane")

    for r in results:
        print("-", r)