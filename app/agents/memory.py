from collections import deque

from app.config import settings


class ConversationMemory:
    """Per-user in-memory conversation history, capped at max_messages entries."""

    def __init__(self, max_messages: int = 20) -> None:
        self._store: dict[str, deque[dict[str, str]]] = {}
        self._max = max_messages

    def get(self, thread_id: str) -> list[dict[str, str]]:
        return list(self._store.get(thread_id, []))

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self._store:
            self._store[thread_id] = deque(maxlen=self._max)
        self._store[thread_id].append({"role": role, "content": content})

    def clear(self, thread_id: str) -> None:
        self._store.pop(thread_id, None)


memory_store = ConversationMemory(max_messages=settings.conversation_max_messages)
