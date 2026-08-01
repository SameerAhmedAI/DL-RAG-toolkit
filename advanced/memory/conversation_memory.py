"""
Conversation memory for multi-turn chat.
Keeps the last N exchanges so follow-up questions ("what about the second one?")
resolve correctly against prior context.
"""

from advanced.config import MAX_CHAT_HISTORY_TURNS


class ConversationMemory:
    """
    Simple turn-based memory — stores (question, answer) pairs and formats
    them for injection into the LLM prompt. Not using LangChain's built-in
    memory classes (many are deprecated/in flux as of this LangChain version)
    — a plain list is more predictable and easier to debug.
    """

    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or MAX_CHAT_HISTORY_TURNS
        self.history: list[dict] = []  # [{"question": ..., "answer": ...}]

    def add_turn(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # drop oldest turn

    def get_formatted_history(self) -> str:
        """Returns history as a plain-text block for prompt injection."""
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)