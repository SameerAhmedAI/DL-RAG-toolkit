"""
Core RAG orchestration: retrieves relevant chunks, builds a grounded prompt,
calls Groq, and returns an answer with source attribution.

Explicit no-hallucination guarantee: if retrieval finds nothing above the
similarity threshold, the LLM is never called with an empty/irrelevant
context — a fixed fallback message is returned instead.

Follow-up questions are rewritten into standalone questions using conversation
history BEFORE retrieval — otherwise pronoun-dependent follow-ups ("how tall is it?")
retrieve poorly since the raw question alone lacks the referent.
"""

from groq import Groq

from advanced.config import GROQ_API_KEY, GROQ_MODEL, SIMILARITY_THRESHOLD
from advanced.retrieval.retriever import retrieve_relevant_chunks
from advanced.memory.conversation_memory import ConversationMemory


NO_RESULTS_MESSAGE = (
    "I couldn't find relevant information in the uploaded documents to answer that. "
    "Try rephrasing your question, or check that the relevant document has been uploaded."
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided document context.

Rules:
- Only use information from the "Context" section below to answer.
- If the context doesn't contain enough information to answer, say so explicitly — do not guess or use outside knowledge.
- Cite which source document your answer comes from when possible.
- Keep answers concise and directly relevant to the question."""

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up
into a standalone question that includes all necessary context. If the question is already
standalone (doesn't depend on prior context), return it unchanged. Return ONLY the rewritten
question, nothing else.

Conversation History:
{history}

Follow-up Question: {question}

Standalone Question:"""


class RAGChain:
    """Orchestrates retrieval + generation for a single chatbot session."""

    def __init__(self, collection_name: str = "rag_chatbot"):
        self.collection_name = collection_name
        self.client = Groq(api_key=GROQ_API_KEY)
        self.memory = ConversationMemory()

    def _rewrite_query(self, question: str) -> str:
        """
        Rewrites a potentially context-dependent question into a standalone one,
        using conversation history. Skipped if there's no history yet (first turn).
        """
        if len(self.memory) == 0:
            return question

        history = self.memory.get_formatted_history()
        prompt = QUERY_REWRITE_PROMPT.format(history=history, question=question)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic rewriting
            max_tokens=150,
        )

        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question

    def _build_prompt(self, question: str, retrieved_chunks: list) -> str:
        context_blocks = []
        for doc, score in retrieved_chunks:
            source = doc.metadata.get("source_file", "unknown")
            context_blocks.append(f"[Source: {source}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_blocks)

        return f"Context:\n{context}\n\nQuestion: {question}"

    def ask(self, question: str) -> dict:
        """
        Answers a question using RAG.

        Returns:
            dict with keys: answer (str), sources (list[str]), grounded (bool)
            grounded=False means the fallback was used (no relevant context found)
        """
        # Rewrite follow-ups into standalone questions BEFORE retrieval
        search_query = self._rewrite_query(question)

        retrieved = retrieve_relevant_chunks(search_query, collection_name=self.collection_name)

        relevant = [(doc, score) for doc, score in retrieved if score >= SIMILARITY_THRESHOLD]

        if not relevant:
            answer = NO_RESULTS_MESSAGE
            self.memory.add_turn(question, answer)
            return {"answer": answer, "sources": [], "grounded": False}

        prompt = self._build_prompt(question, relevant)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content
        sources = sorted(set(doc.metadata.get("source_file", "unknown") for doc, _ in relevant))

        self.memory.add_turn(question, answer)

        return {"answer": answer, "sources": sources, "grounded": True}

    def reset(self):
        """Clears conversation memory — used when starting a new chat session."""
        self.memory.clear()