"""
ChromaDB vector store setup and retrieval logic.
Handles both indexing new documents and querying the store.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from advanced.config import CHROMA_PERSIST_DIR, RETRIEVAL_K
from advanced.embeddings.embedder import get_embedding_model


def get_vector_store(collection_name: str = "rag_chatbot"):
    """
    Returns a ChromaDB vector store instance, persisted to disk.
    Reused across the session so repeated calls don't re-embed existing data.
    """
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def index_documents(chunks: list[Document], collection_name: str = "rag_chatbot") -> int:
    """
    Embeds and stores document chunks in ChromaDB.

    Args:
        chunks: chunked Documents (from ingestion/chunking.py)
        collection_name: which Chroma collection to write to

    Returns:
        int: number of chunks indexed
    """
    if not chunks:
        raise ValueError("No chunks provided to index — check ingestion/chunking output.")

    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks)

    return len(chunks)


def retrieve_relevant_chunks(query: str, collection_name: str = "rag_chatbot",
                              k: int = None) -> list[tuple[Document, float]]:
    """
    Retrieves the top-k most relevant chunks for a query, with similarity scores.

    Args:
        query: the user's question
        collection_name: which Chroma collection to search
        k: number of results (defaults to config.RETRIEVAL_K)

    Returns:
        list of (Document, similarity_score) tuples, ordered by relevance.
        Empty list if the collection has no documents yet.
    """
    k = k or RETRIEVAL_K
    vector_store = get_vector_store(collection_name)

    # similarity_search_with_relevance_scores returns (doc, score) where
    # score is normalized 0-1 (higher = more relevant) — used for the
    # "no relevant results found" threshold check downstream
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        return results
    except Exception:
        # Empty/uninitialized collection — treat as no results, not a crash
        return []


def clear_collection(collection_name: str = "rag_chatbot"):
    """Deletes all documents in a collection — used when starting a fresh session."""
    vector_store = get_vector_store(collection_name)
    vector_store.delete_collection()