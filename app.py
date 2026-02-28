"""
app.py

Streamlit web interface for the Building Regulations Q&A app.

Usage:
    streamlit run app.py
"""

import os
import json
import pickle
import anthropic
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.pkl"
TOP_K = 10
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Building Regulations Q&A",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Building Regulations Q&A")
st.caption("Ask questions about England's national Building Regulations (Approved Documents)")

# ---------------------------------------------------------------------------
# Load model and embeddings (cached so they only load once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading document index...")
def load_data():
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error(f"Cannot find {EMBEDDINGS_FILE}. Please run ask_building_regs.py first to build the embeddings.")
        st.stop()
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]


@st.cache_resource(show_spinner="Connecting to Claude...")
def load_claude():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("No ANTHROPIC_API_KEY found. Please set it in your terminal before running this app.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(question, chunks, embeddings, model, top_k=TOP_K):
    question_embedding = model.encode([question])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalised = embeddings / np.clip(norms, 1e-10, None)
    q_norm = question_embedding / np.linalg.norm(question_embedding)
    scores = (normalised @ q_norm.T).flatten()
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
- Always cite the specific Approved Document and paragraph you are drawing from
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
# Main UI
# ---------------------------------------------------------------------------

model = load_model()
chunks, embeddings = load_data()
claude = load_claude()

# Example questions
st.markdown("**Example questions:**")
examples = [
    "What are the lighting requirements for external staircases?",
    "What is the minimum headroom for a staircase?",
    "What are the fire escape requirements for a three storey house?",
    "What is the maximum rise and going for a domestic staircase?",
]
cols = st.columns(2)
for i, example in enumerate(examples):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

st.divider()

# Question input
question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the guarding requirements for a balcony?",
    height=100
)

if st.button("Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching regulations and asking Claude..."):
            relevant = search(question, chunks, embeddings, model)
            answer = ask_claude(question, relevant, claude)

        st.divider()
        st.markdown("### Answer")
        st.markdown(answer)

        # Show sources in an expander
        with st.expander("📄 Source excerpts used"):
            for i, (chunk, score) in enumerate(relevant, 1):
                meta = chunk["metadata"]
                st.markdown(f"**{i}. {meta['title']}** — {meta['topic']} (relevance: {score:.2f})")
                st.caption(chunk["text"][:300] + "...")
                st.divider()

st.divider()
st.caption("⚠️ This tool is for information only. Always verify requirements with a qualified building control officer before carrying out any building work.")
