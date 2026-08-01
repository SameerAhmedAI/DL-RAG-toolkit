"""
Central configuration for the RAG chatbot.
All tunable values live here — no magic numbers scattered across modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from repo root

# ---- API ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Create a .env file at the repo root with:\n"
        "GROQ_API_KEY=your_key_here"
    )

# ---- Paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PERSIST_DIR = REPO_ROOT / "advanced" / "chroma_db"
TEST_DOCUMENTS_DIR = REPO_ROOT / "data" / "test_documents"

# ---- Chunking ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---- Embeddings ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for demo-scale docs

# ---- Retrieval ----
RETRIEVAL_K = 4  # number of chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.3  # below this, treat as "no relevant results found"

# ---- Chat ----
MAX_CHAT_HISTORY_TURNS = 5  # how many past exchanges to keep in memory

# ---- Supported file types ----
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}