"""
Streamlit UI for the DL-RAG-Toolkit chatbot.
Upload panel + chat panel + source attribution display.

Run from repo root: streamlit run advanced/app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import tempfile
import os

from advanced.ingestion.loaders import (
    load_document, UnsupportedFileTypeError, DocumentLoadError
)
from advanced.ingestion.chunking import chunk_documents
from advanced.retrieval.retriever import index_documents, clear_collection
from advanced.chain.rag_chain import RAGChain
from advanced.config import SUPPORTED_EXTENSIONS

st.set_page_config(
    page_title="DL-RAG-Toolkit — Document Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---- Session state initialization ----
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "session_id" not in st.session_state:
    # Unique collection per browser session so users don't share document context
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]


def get_chain():
    if st.session_state.chain is None:
        collection_name = f"session_{st.session_state.session_id}"
        st.session_state.chain = RAGChain(collection_name=collection_name)
    return st.session_state.chain


def process_uploaded_file(uploaded_file):
    """
    Saves an uploaded file to a temp path, loads it, chunks it, and indexes it.
    Returns (success: bool, message: str).
    """
    ext = Path(uploaded_file.name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"'{ext}' is not supported. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

    # Write to a temp file since loaders expect a filesystem path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        docs = load_document(tmp_path)
        # Restore the original filename in metadata (temp file has a random name)
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name

        chunks = chunk_documents(docs)

        collection_name = f"session_{st.session_state.session_id}"
        num_indexed = index_documents(chunks, collection_name=collection_name)

        return True, f"Indexed {num_indexed} chunks from '{uploaded_file.name}'"

    except UnsupportedFileTypeError as e:
        return False, str(e)
    except DocumentLoadError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error processing '{uploaded_file.name}': {e}"
    finally:
        os.unlink(tmp_path)  # clean up temp file regardless of outcome


# ---- Sidebar: Upload panel ----
with st.sidebar:
    st.header("📄 Documents")
    st.caption("Supported: PDF, DOCX, TXT, XLSX")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

        if new_files:
            for uploaded_file in new_files:
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    success, message = process_uploaded_file(uploaded_file)

                if success:
                    st.session_state.indexed_files.append(uploaded_file.name)
                    st.success(message)
                else:
                    st.error(message)

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Indexed Documents")
        for fname in st.session_state.indexed_files:
            st.text(f"✓ {fname}")

        st.divider()
        if st.button("🗑️ Clear all documents", use_container_width=True):
            collection_name = f"session_{st.session_state.session_id}"
            clear_collection(collection_name)
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.chain = None
            st.rerun()
    else:
        st.info("Upload a document to start chatting.")


# ---- Main panel: Chat ----
st.title("DL-RAG-Toolkit — Document Chatbot")
st.caption("Ask questions about your uploaded documents. Answers are grounded only in what you upload — no hallucinated information from outside sources.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.indexed_files):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_chain()
            try:
                result = chain.ask(prompt)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                if sources:
                    st.caption(f"📎 Sources: {', '.join(sources)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"Something went wrong generating a response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

if not st.session_state.indexed_files:
    st.info("👈 Upload a document in the sidebar to begin.")