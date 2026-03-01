import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

from agents.routing_agent import estimate_times


# ---------- PATH SETUP ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "rag", "vector_store", "rules.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "rag", "vector_store", "chunks.npy")

print("Loading FAISS index from:", INDEX_PATH)
print("Loading chunks from:", CHUNKS_PATH)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")


# ---------- vLLM CLIENT ----------

client = OpenAI(
    base_url="http://176.118.70.14:44000/v1",
    api_key="EMPTY"
)

MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"


# ---------- EMBEDDING MODEL ----------

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ---------- LOAD INDEX ----------

index = faiss.read_index(INDEX_PATH)
chunks = np.load(CHUNKS_PATH, allow_pickle=True)


# ---------- RAG RETRIEVAL ----------

def retrieve_rules(query, k=4):

    emb = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(emb, k)

    return "\n".join(chunks[i] for i in indices[0])


# ---------- PROMPT ----------

def build_prompt(animal, distance_km, rules, eta):

    return f"""
Animal: {animal}
Distance: {distance_km} km

Estimated travel times:

Car: {eta["car"]} hours
Train: {eta["train"]} hours
Plane: {eta["plane"]} hours

Transport rules:
{rules}

Respond ONLY with valid JSON:

{{
 "car": {{"allowed": true, "eta_hours": {eta["car"]}}},
 "train": {{"allowed": true, "eta_hours": {eta["train"]}}},
 "plane": {{"allowed": true, "eta_hours": {eta["plane"]}}}
}}
"""


# ---------- LLM QUERY ----------

def query_llm(prompt):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content


# ---------- POLICY AGENT ----------

def policy_decision(animal, distance_km):

    eta = estimate_times(distance_km)

    rules = retrieve_rules(animal)

    prompt = build_prompt(animal, distance_km, rules, eta)

    return query_llm(prompt)


# ---------- TEST ----------

if __name__ == "__main__":

    print(policy_decision("large dog", 800))