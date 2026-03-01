import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "../rag_data/transport_rules.txt"
INDEX_DIR = "vector_store"

os.makedirs(INDEX_DIR, exist_ok=True)

# load model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# read rules
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# split into chunks
chunks = [
    chunk.strip()
    for chunk in text.split("\n")
    if chunk.strip() and not chunk.strip().endswith("RULES")
]

print(f"Loaded {len(chunks)} chunks")

# embed
embeddings = model.encode(chunks, convert_to_numpy=True)

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

index.add(embeddings)

# save
faiss.write_index(index, f"{INDEX_DIR}/rules.index")

np.save(f"{INDEX_DIR}/chunks.npy", chunks)

print("Index saved.")