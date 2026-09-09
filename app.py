````python
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
# Sidebar - How RAG Works
# ---------------------------------------------------------
with st.sidebar:
    st.title("📖 How RAG Works")

    st.markdown(
        """
        **1. 📄 Upload PDF**

        Your HR Policy PDF is uploaded to the application.

        **2. ✂️ Chunking**

        The PDF text is divided into smaller sections.

        **3. 🧠 Embeddings**

        Sentence Transformers converts each section into a numerical vector.

        **4. 🔎 FAISS Retrieval**

        FAISS finds the sections most relevant to your question.

        **5. 🤖 Groq AI**

        The retrieved policy content is sent to the
        `openai/gpt-oss-120b` model.

        **6. 📌 Sources**

        The application shows the PDF pages used to generate the answer.

        ---

        ### 🔒 Grounded Answers

        The assistant is instructed to answer **only from the uploaded HR policy**.

        If the information cannot be found, it will say:

        **"Sorry, this information is not found in the uploaded HR policy."**
        """
    )


# ---------------------------------------------------------
# Cached Models
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client(api_key: str):
    return Groq(api_key=api_key)


# ---------------------------------------------------------
# PDF Processing
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
    """Clean whitespace without changing policy meaning."""

    lines = [line.strip() for line in text.splitlines()]

    return " ".join(
        line for line in lines if line
    )


def chunk_text(text: str, page_number: int) -> List[Dict]:
    """Create overlapping word-based chunks from a page."""

    words = normalize_text(text).split()

    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):

        end = min(
            start + CHUNK_SIZE,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

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

        start = max(
            end - CHUNK_OVERLAP,
            start + 1
        )

        chunk_id += 1

    return chunks


def build_chunks(pdf_bytes: bytes) -> List[Dict]:
    """Extract pages and split them into searchable chunks."""

    pages = extract_pdf_pages(pdf_bytes)

    all_chunks = []

    for page in pages:
        all_chunks.extend(
            chunk_text(
                page["text"],
                page["page"]
            )
        )

    return all_chunks


def build_faiss_index(
    chunks: List[Dict],
    model
) -> faiss.Index:
    """Create a normalized FAISS inner-product index."""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Generate Suggested Questions
# ---------------------------------------------------------
def generate_suggested_questions(
    chunks: List[Dict],
    client: Groq
) -> List[str]:
    """
    Generate 5-6 questions based on the uploaded HR policy.
    """

    # Use a representative amount of text so the prompt
    # does not become unnecessarily large.
    max_chunks = min(len(chunks), 12)

    selected_chunks = chunks[:max_chunks]

    policy_text = "\n\n".join(
        [
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in selected_chunks
        ]
    )

    prompt = f"""
You are an HR Policy Assistant.

Generate exactly 6 useful questions that an employee
could ask about the uploaded HR policy.

IMPORTANT RULES:

1. Questions MUST be based only on the supplied policy text.
2. Do not ask questions about information that is not present.
3. Make questions practical and useful for employees.
4. Cover different policy topics when possible.
5. Keep each question short and natural.
6. Do not include answers.
7. Return ONLY a valid JSON array of 6 strings.
8. Do not use markdown.
9. Do not number the questions.

Example format:

[
  "What is the annual leave entitlement?",
  "How can employees request leave?",
  "What are the working hours?",
  "What benefits are available to employees?",
  "What is the attendance policy?",
  "What happens when an employee resigns?"
]

UPLOADED HR POLICY:

{policy_text}
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise employee questions "
                        "from HR policy documents."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=500,
        )

        result = response.choices[0].message.content.strip()

        # Remove markdown code fences if the model adds them.
        result = result.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()

        import json

        questions = json.loads(result)

        if isinstance(questions, list):

            questions = [
                str(q).strip()
                for q in questions
                if str(q).strip()
            ]

            return questions[:6]

    except Exception:
        pass

    # Fallback questions if generation fails.
    return [
        "What is the purpose of this HR policy?",
        "What are the main employee responsibilities?",
        "What benefits are mentioned in the policy?",
        "What does the policy say about leave?",
        "What does the policy say about working hours?",
        "What are the important workplace rules?",
    ]


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

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            (
                chunks[idx],
                float(score)
            )
        )

    return results


# ---------------------------------------------------------
# Groq Answer Generation
# ---------------------------------------------------------
def answer_from_policy(
    question: str,
    retrieved: List[Tuple[Dict, float]],
    client: Groq,
) -> str:
    """Generate an answer strictly from retrieved policy context."""

    if not retrieved:
        return (
            "Sorry, this information is not found "
            "in the uploaded HR policy."
        )

    context_parts = []

    for i, (chunk, score) in enumerate(
        retrieved,
        start=1
    ):

        context_parts.append(
            f"""
[Source {i} | Page {chunk['page']} | Similarity {score:.3f}]

{chunk['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY from the supplied HR policy context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not assume policy rules.
4. If the answer is not supported by the context, reply exactly:

"Sorry, this information is not found in the uploaded HR policy."

5. Give a concise and clear answer.
6. Mention the relevant PDF page number(s).
7. Preserve conditions, exceptions, limits and approval requirements.
8. Do not provide legal advice.
9. Do not mention similarity scores in the answer.
10. Do not mention the internal retrieval process unless asked.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer using ONLY the HR POLICY CONTEXT.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        temperature=0.1,
        max_completion_tokens=700,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# Source Display
# ---------------------------------------------------------
def show_sources(
    retrieved: List[Tuple[Dict, float]]
):
    """Display the chunks used as evidence."""

    if not retrieved:
        return

    st.markdown("### 📌 RAG Sources")

    for i, (chunk, score) in enumerate(
        retrieved,
        start=1
    ):

        with st.expander(
            f"Source {i} — Page {chunk['page']} — similarity {score:.3f}"
        ):

            st.write(
                chunk["text"]
            )


# ---------------------------------------------------------
# Ask Question Function
# ---------------------------------------------------------
def process_question(
    question: str,
    api_key: str
):

    if not question.strip():
        st.warning(
            "Please enter a question."
        )
        return

    embedding_model = load_embedding_model()

    with st.spinner(
        "Searching the HR policy..."
    ):

        retrieved = retrieve_chunks(
            question=question.strip(),
            index=st.session_state.index,
            chunks=st.session_state.chunks,
            model=embedding_model,
        )

    # Reject weak matches.
    if (
        not retrieved
        or retrieved[0][1] < MIN_SIMILARITY
    ):

        st.warning(
            "Sorry, this information is not found "
            "in the uploaded HR policy."
        )

        return

    client = load_groq_client(
        api_key
    )

    with st.spinner(
        "Generating answer from the policy..."
    ):

        answer = answer_from_policy(
            question.strip(),
            retrieved,
            client
        )

    st.markdown("### 💬 Answer")

    st.write(answer)

    show_sources(
        retrieved
    )


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    st.title("📘 HR Policy Assistant")

    st.caption(
        "Ask questions about your HR Policy PDF "
        "and get answers grounded in the document."
    )

    # -----------------------------------------------------
    # API Key
    # -----------------------------------------------------

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY")
    )

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Cloud → App settings → Secrets."
        )

        st.stop()

    # -----------------------------------------------------
    # PDF Upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "📄 Upload HR Policy PDF",
        type=["pdf"],
        help="Upload a text-based HR policy PDF."
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload an HR Policy PDF to get started."
        )

        return

    # -----------------------------------------------------
    # PDF Hash
    # -----------------------------------------------------

    pdf_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # -----------------------------------------------------
    # Build RAG Index
    # -----------------------------------------------------

    if (
        st.session_state.get("file_hash")
        != file_hash
    ):

        with st.spinner(
            "📚 Reading PDF and building RAG index..."
        ):

            chunks = build_chunks(
                pdf_bytes
            )

            if not chunks:

                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a text-based PDF rather than "
                    "a scanned image PDF."
                )

                st.stop()

            embedding_model = load_embedding_model()

            index = build_faiss_index(
                chunks,
                embedding_model
            )

            # Save RAG data in session.
            st.session_state.file_hash = file_hash

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.document_name = (
                uploaded_file.name
            )

            # -------------------------------------------------
            # Generate PDF-specific questions
            # -------------------------------------------------

            client = load_groq_client(
                api_key
            )

            with st.spinner(
                "💡 Creating questions from your HR policy..."
            ):

                suggested_questions = (
                    generate_suggested_questions(
                        chunks,
                        client
                    )
                )

            st.session_state.suggested_questions = (
                suggested_questions
            )

            # Reset previous answer.
            st.session_state.last_question = None

            st.session_state.last_answer = None

    # -----------------------------------------------------
    # PDF Successfully Indexed
    # -----------------------------------------------------

    st.success(
        f"✅ **{uploaded_file.name}** is ready!"
    )

    # -----------------------------------------------------
    # Suggested Questions
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### 💡 Try these questions"
    )

    st.caption(
        "These questions were generated from your uploaded HR policy."
    )

    suggested_questions = st.session_state.get(
        "suggested_questions",
        []
    )

    # Create two columns for a clean layout.
    for i in range(
        0,
        len(suggested_questions),
        2
    ):

        col1, col2 = st.columns(2)

        if i < len(suggested_questions):

            with col1:

                if st.button(
                    suggested_questions[i],
                    key=f"suggested_question_{i}",
                    use_container_width=True,
                ):

                    st.session_state.selected_question = (
                        suggested_questions[i]
                    )

        if i + 1 < len(suggested_questions):

            with col2:

                if st.button(
                    suggested_questions[i + 1],
                    key=f"suggested_question_{i + 1}",
                    use_container_width=True,
                ):

                    st.session_state.selected_question = (
                        suggested_questions[i + 1]
                    )

    # -----------------------------------------------------
    # User's Own Question
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### 💬 Or ask your own question"
    )

    question = st.text_input(
        "Your question",
        value=st.session_state.get(
            "selected_question",
            ""
        ),
        placeholder=(
            "Example: How many annual leave days "
            "are employees entitled to?"
        ),
        label_visibility="collapsed",
    )

    ask = st.button(
        "🔎 Ask HR Policy",
        type="primary",
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Process Question
    # -----------------------------------------------------

    if ask:

        st.session_state.selected_question = (
            question
        )

        process_question(
            question,
            api_key
        )


# ---------------------------------------------------------
# Run Application
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
````
