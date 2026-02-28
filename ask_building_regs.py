"""
ask_building_regs.py

Loads chunks.json and lets you ask questions about the Building Regulations,
answered by Claude. Uses numpy for vector search instead of ChromaDB,
so it works with Python 3.14.

First run:  builds embeddings and saves them to disk (takes a few minutes)
Later runs: loads saved embeddings and goes straight to questions

Usage:
    python3 ask_building_regs.py
"""

import os
import json
import pickle
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.pkl"
TOP_K = 10  # number of chunks to retrieve per question
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Embedding and search
# ---------------------------------------------------------------------------

def build_embeddings(chunks, model):
    """Embed all chunks and save to disk."""
    print(f"Embedding {len(chunks)} chunks (this takes a few minutes)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    data = {"chunks": chunks, "embeddings": embeddings}
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}\n")
    return chunks, embeddings


def load_embeddings():
    """Load previously saved embeddings from disk."""
    print(f"Loading saved embeddings from {EMBEDDINGS_FILE}...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['chunks'])} chunks.\n")
    return data["chunks"], data["embeddings"]


def search(question, chunks, embeddings, model, top_k=TOP_K):
    """Find the most relevant chunks for a question."""
    question_embedding = model.encode([question])

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalised = embeddings / np.clip(norms, 1e-10, None)
    q_norm = question_embedding / np.linalg.norm(question_embedding)
    scores = normalised @ q_norm.T
    scores = scores.flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Ask Claude
# ---------------------------------------------------------------------------

def ask_claude(question, relevant_chunks, claude_client):
    context_parts = []
    for chunk, score in relevant_chunks:
        meta = chunk["metadata"]
        ref = f"[{meta['title']} - {meta['topic']}]"
        context_parts.append(f"{ref}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="""You are a helpful assistant that answers questions about
England's national Building Regulations, based on the official Approved Documents.

When answering:
- Always cite the specific Approved Document you are drawing from
- Be precise and practical
- If the answer isn't clearly covered by the provided excerpts, say so honestly
- Always end with a reminder that users should verify requirements with a
  qualified building control officer for actual building work""",
        messages=[
            {
                "role": "user",
                "content": f"Here are relevant excerpts from the Approved Documents:\n\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nNo API key found!")
        print("Please set your Anthropic API key by typing this in Terminal")
        print("(replace sk-ant-... with your actual key):\n")
        print("  export ANTHROPIC_API_KEY=sk-ant-...\n")
        print("Then run this script again.")
        return

    claude = anthropic.Anthropic(api_key=api_key)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Load or build embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        chunks, embeddings = load_embeddings()
    else:
        if not os.path.exists(CHUNKS_FILE):
            print(f"Cannot find {CHUNKS_FILE}. Please run parse_approved_docs.py first.")
            return
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunks, embeddings = build_embeddings(chunks, model)

    print("=" * 55)
    print("  Building Regulations Q&A")
    print("  Ask anything about the Approved Documents.")
    print("  Type 'quit' to exit.")
    print("=" * 55)
    print()

    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSearching regulations and asking Claude...\n")
        relevant = search(question, chunks, embeddings, model)
        answer = ask_claude(question, relevant, claude)
        print(f"Answer:\n{answer}\n")
        print("-" * 55)
        print()


if __name__ == "__main__":
    main()
