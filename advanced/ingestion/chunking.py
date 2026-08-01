"""
Chunking logic for ingested documents.
Uses LangChain's RecursiveCharacterTextSplitter — splits on paragraph/sentence
boundaries where possible before falling back to hard character limits,
which preserves semantic coherence better than a naive fixed-size split.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into overlapping chunks for embedding/retrieval.

    Args:
        documents: list of LangChain Documents (from loaders.py)

    Returns:
        list[Document]: chunked documents, metadata preserved + chunk index added
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph -> line -> sentence -> word -> char
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index within its source — useful for citation display
    source_counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        source_counters[source] = source_counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[source]

    return chunks