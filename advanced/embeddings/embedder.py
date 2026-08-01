"""
Embedding model wrapper — sentence-transformers, local and free.
Kept as a thin wrapper so the rest of the app depends on this interface,
not directly on langchain's embedding class — makes swapping models later trivial.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from advanced.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Returns a LangChain-compatible embedding model instance.
    Loaded once and reused — sentence-transformers models are not cheap to reload.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # cosine similarity works cleanly on normalized vectors
    )