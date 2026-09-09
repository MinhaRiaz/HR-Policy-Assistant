import hashlib
import os
from typing import List, Dict, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="centered",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SIMILARITY = 0.30


# ---------------------------------------------------------
# Cached models
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client(api_key: str):
    return Groq(api_key=api_key)


# ---------------------------------------------------------
# PDF processing
# ---------------------------------------------------------
def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    """Extract text from each PDF page while preserving page numbers."""
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


def normalize_text(text: str) -> str:
    """Clean whitespace without changing the meaning of the policy."""
    lines = [line.strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def chunk_text(text: str, page_number: int) -> List[Dict]:
    """Create overlapping word-based chunks from a page."""
    words = normalize_text(text).split()

    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk,
                }
            )

        if end >= len(words):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)
        chunk_id += 1

    return chunks


def build_chunks(pdf_bytes: bytes) -> List[Dict]:
    """Extract pages and split them into searchable chunks."""
    pages = extract_pdf_pages(pdf_bytes)

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["text"], page["page"]))

    return all_chunks


def build_faiss_index(chunks: List[Dict], model) -> faiss.Index:
    """Create a normalized inner-product FAISS index."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------
def retrieve_chunks(
    question: str,
    index: faiss.Index,
    chunks: List[Dict],
    model,
    top_k: int = TOP_K,
) -> List[Tuple[Dict, float]]:
    """Retrieve the most semantically similar chunks."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append((chunks[idx], float(score)))

    return results


# ---------------------------------------------------------
# Groq answer generation
# ---------------------------------------------------------
def answer_from_policy(
    question: str,
    retrieved: List[Tuple[Dict, float]],
    client: Groq,
) -> str:
    """Generate an answer strictly from retrieved policy context."""
    if not retrieved:
        return "Sorry, I couldn't find this information in the uploaded HR policy."

    context_parts = []

    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i} | Page {chunk['page']} | Similarity {score:.3f}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY from the supplied HR policy context.

Rules:
1. Do not use outside knowledge.
2. Do not invent, assume, or infer policy rules that are not supported by the context.
3. If the context does not contain the answer, reply exactly:
   "Sorry, this information is not found in the uploaded HR policy."
4. Give a concise, clear answer.
5. When the context supports the answer, mention the relevant page number(s).
6. If the policy contains conditions, exceptions, limits, or approval requirements, preserve them accurately.
7. Do not provide legal advice. If the user asks for a legal interpretation rather than the policy wording, say that the uploaded policy does not provide legal advice.
"""

    user_prompt = f"""
HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the HR POLICY CONTEXT.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=700,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# Source display
# ---------------------------------------------------------
def show_sources(retrieved: List[Tuple[Dict, float]]):
    """Display the chunks used as evidence for the answer."""
    if not retrieved:
        return

    st.markdown("### 📌 RAG Sources")

    for i, (chunk, score) in enumerate(retrieved, start=1):
        with st.expander(
            f"Source {i} — Page {chunk['page']} — similarity {score:.3f}"
        ):
            st.write(chunk["text"])


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------
def main():
    st.title("📘 HR Policy Assistant")
    st.caption(
        "Upload an HR Policy PDF and ask questions. "
        "Answers are grounded in the uploaded document."
    )

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. Add it in "
            "Streamlit Cloud → App settings → Secrets."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"],
        help="Upload a text-based HR policy PDF.",
    )

    if uploaded_file is None:
        st.info("Upload a PDF to start asking questions.")
        return

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Rebuild the index only when the uploaded PDF changes.
    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Reading PDF and building the RAG index..."):
            chunks = build_chunks(pdf_bytes)

            if not chunks:
                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a text-based PDF rather than a scanned image PDF."
                )
                st.stop()

            embedding_model = load_embedding_model()
            index = build_faiss_index(chunks, embedding_model)

            st.session_state.file_hash = file_hash
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.document_name = uploaded_file.name

        st.success(
            f"Indexed **{len(chunks)} chunks** from **{uploaded_file.name}**."
        )

    st.markdown("---")

    question = st.text_input(
        "Ask a question about the HR policy",
        placeholder="Example: How many annual leave days are employees entitled to?",
    )

    ask = st.button("Ask HR Policy", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        embedding_model = load_embedding_model()

        with st.spinner("Searching the HR policy..."):
            retrieved = retrieve_chunks(
                question=question.strip(),
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                model=embedding_model,
            )

        # If even the best match is weak, don't send unrelated context to the LLM.
        if not retrieved or retrieved[0][1] < MIN_SIMILARITY:
            st.warning(
                "Sorry, this information is not found in the uploaded HR policy."
            )
            return

        client = load_groq_client(api_key)

        with st.spinner("Generating an answer from the policy..."):
            answer = answer_from_policy(question.strip(), retrieved, client)

        st.markdown("### 💬 Answer")
        st.write(answer)

        show_sources(retrieved)


if __name__ == "__main__":
    main()
