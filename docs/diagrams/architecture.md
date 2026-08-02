# Architecture — Neural Forge Labs

## App Flow & Architecture

### Intermediate Level (per model)
Linear pipeline, no app layer:

```
Raw Dataset → Preprocessing → Model Definition (OOP class)
→ Training Loop (with validation) → Evaluation on Test Set
→ Metrics + Plots saved to /results → results.md written
```

Each of the 5 models is fully independent. No shared runtime state between them — shared code (e.g. a plotting helper) lives in a `common/` module imported by each, not duplicated five times.

### Advanced Level (RAG Chatbot)

```
[User uploads file via Streamlit]
        ↓
[Ingestion Layer] — format-specific loader (PDF/DOCX/TXT/XLSX)
        ↓
[Chunking] — recursive character splitter, overlap for context continuity
        ↓
[Embedding] — sentence-transformers (local, free, matches prior project choice)
        ↓
[ChromaDB] — persisted vector store, one collection per session or per doc-set
        ↓
[User asks question via Streamlit chat]
        ↓
[Retriever] — top-k similarity search against ChromaDB
        ↓
[LangChain Conversational Chain] — injects retrieved chunks + chat history into prompt
        ↓
[Groq LLM] — generates answer grounded in retrieved context
        ↓
[Response + Source Chunks] — rendered in Streamlit, memory updated
```

**State transitions** (for the required State Diagram):
`Idle → Uploading → Ingesting → Ready → Querying → Retrieving → Generating → Answered → (loop to Querying, or Idle on new upload)`
Error states branch off `Ingesting` (unsupported format / corrupt file) and `Retrieving` (no relevant chunks found → explicit "not found" response, not a hallucination).

## Folder & File Structure

```
neural-forge-labs/
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Memory.md
├── README.md                       # top-level, written last
├── requirements.txt
├── .github/workflows/ci.yml
│
├── intermediate/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── plotting.py             # shared loss/accuracy curve helpers
│   │   └── metrics.py
│   ├── ann/
│   │   ├── ann.ipynb
│   │   ├── model.py                # OOP model class
│   │   ├── train.py
│   │   └── results.md
│   ├── cnn/                        # same pattern
│   ├── rnn/                        # same pattern
│   ├── lstm/                       # same pattern
│   └── transfer_learning/          # same pattern
│
├── advanced/
│   ├── app/
│   │   └── streamlit_app.py        # entry point
│   ├── ingestion/
│   │   ├── loaders.py              # PDF/DOCX/TXT/XLSX loaders
│   │   └── chunking.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── memory/
│   │   └── conversation_memory.py
│   ├── chain/
│   │   └── rag_chain.py            # LangChain orchestration + Groq call
│   ├── config.py                   # model names, chunk size, k, etc.
│   └── tests/
│       └── test_ingestion.py       # minimum viable test for CI
│
├── data/
│   ├── sample_datasets/            # for intermediate models
│   └── test_documents/             # for RAG demo (PDF/DOCX/TXT/XLSX samples)
│
└── docs/
    └── diagrams/
        ├── architecture.png
        ├── flow_diagram.png
        └── state_diagram.png
```

## Tech Stack

**Intermediate Level**
- Python, PyTorch (or TensorFlow — pick one, stay consistent across all 5, do not mix)
- scikit-learn (metrics, preprocessing)
- matplotlib (plots)
- Jupyter for exploration, `.py` scripts for the actual reusable/OOP code

**Advanced Level**
- LangChain (orchestration)
- ChromaDB (vector store, local persistence)
- sentence-transformers (embeddings, local/free)
- Groq API (LLM inference — llama-3.3-70b-versatile, consistent with prior project decisions)
- Streamlit (UI)
- Loaders: `pypdf` (PDF), `python-docx` (DOCX), `openpyxl` (XLSX), built-in (TXT)

**Shared**
- GitHub Actions (CI: lint + basic test run on push)
- pytest (minimum test coverage on ingestion layer — required by CI/CD deliverable, don't over-invest here)