import os
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(
    page_title="RAG Document Assistant",
    layout="wide",
    page_icon="📘"
)

# --- API Key ---
GROQ_API_KEY = "paste your llm api"
os.makedirs("sessions", exist_ok=True)

# --- Groq client ---
client = Groq(api_key=GROQ_API_KEY)

# --- Embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# UTILITY FUNCTIONS
# ==============================
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Splits text into chunks."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks):
    """Builds a FAISS index from text chunks."""
    if not chunks or len(chunks) == 0:
        raise ValueError("No valid text chunks found. Your PDF might not contain readable text.")

    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(f"Invalid embeddings shape: {embeddings.shape}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.success(f" FAISS index built successfully with {len(chunks)} chunks.")
    return index

def query_rag(query, index, chunks, top_k=3):
    """Queries FAISS index and gets relevant context using Groq API."""
    if not query.strip():
        return "Please enter a valid question."

    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    context = "\n\n".join(retrieved)

    if not context.strip():
        return "No relevant content found in the document."

    prompt = f"""
You are an AI assistant that answers questions using only the provided context.
Be concise, clear, and factual.

Context:
{context}

Question: {query}
Answer:
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Groq API error: {e}"

# ==============================
# STREAMLIT UI
# ==============================
st.markdown("""
<h1 style='text-align:center; color:#0077b6;'> Retrieval-Augmented Generation (RAG) Assistant</h1>
<p style='text-align:center; font-size:16px; color:grey;'>
Upload your PDF and ask questions about its content. Powered by Groq + Sentence Transformers.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from the PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    char_count = len(text)
    st.info(f"📄 Extracted **{char_count:,}** characters from the document.")

    if len(text.strip()) == 0:
        st.error("No readable text found. This PDF might be image-based. Please upload an OCR version.")
    else:
        with st.spinner("Splitting text and building FAISS index..."):
            chunks = chunk_text(text)
            index = build_faiss_index(chunks)

        st.success("PDF processed successfully!")

        st.markdown("---")
        st.markdown("### 💬 Ask a question about your document")
        user_query = st.text_input("Type your question below:")

        if user_query:
            with st.spinner("Thinking..."):
                answer = query_rag(user_query, index, chunks)

            st.markdown("### Answer")
            st.success(answer)

else:
    st.info("Please upload a PDF to begin.")

st.markdown("""
---
<div style='text-align: center; font-size: 15px; color: grey;'>
Developed by <b>Slahitha S</b> | <i>Powered by Groq + FAISS + Streamlit</i>
</div>
""", unsafe_allow_html=True)

