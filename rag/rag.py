# Write a chatbot with RAG!

import ollama
import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

def query(client, model, text):
    hits = client.query_points(
        collection_name="claims",
        query=model.encode(text).tolist(),
        limit=10,
    ).points

    for hit in hits:
        print(hit.payload, "score:", hit.score)
        

    return [hit.payload for hit in hits]
    

def chat(client, model):
    messages = [
        {"role": "system", "content": "You are an expert fact checker who has read all of feverous_train_challenges.jsonl"},
    ]

    while True:
        user_input = input("What do you want to say: ")
        
        # Retrieval
        docs = query(client, model, user_input)
        # Augment
        # prompt = "\n".join(f"Claim: {doc['claim']}\nLabel: {doc['label']}" for doc in docs)
        prompt = "\n".join(doc['label'] for doc in docs)

        prompt += "\nBased on the above documents, label this query with a percentage: " + user_input        
        
        messages.append({"role": "user", "content": prompt})

        # Generation
        response = ollama.chat("llama3.2:1b", messages=messages)
        
        print(response["message"]["content"])  # {"role": "assistant", "content": "..."}
        
        messages.append(response["message"])


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(path="db")

    chat(client, model)


if __name__ == "__main__":
    main()