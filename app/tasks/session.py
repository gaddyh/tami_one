from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional


class DueDateResolution(StrEnum):
    UNKNOWN = "unknown"
    PROVIDED = "provided"
    INTENTIONALLY_ABSENT = "intentionally_absent"


class ReviewSession:
    def __init__(
        self,
        tenant_id: str,
        chat_id: str,
        item_ids: list[str],
    ) -> None:
        self.tenant_id = tenant_id
        self.chat_id = chat_id
        self.item_ids = item_ids
        self.current_index: int = 0

        self.candidate_subject: str = ""
        self.candidate_due_at: str = ""
        self.due_date_resolution: DueDateResolution = DueDateResolution.UNKNOWN
        self.clarification_count: int = 0

        self.created_count: int = 0
        self.started_at: datetime = datetime.now(timezone.utc)
        self.last_activity_at: datetime = self.started_at

    @property
    def current_item_id(self) -> Optional[str]:
        if self.current_index < len(self.item_ids):
            return self.item_ids[self.current_index]
        return None

    @property
    def remaining_count(self) -> int:
        return len(self.item_ids) - self.current_index

    def touch(self) -> None:
        self.last_activity_at = datetime.now(timezone.utc)

    def reset_candidate_state(self) -> None:
        self.candidate_subject = ""
        self.candidate_due_at = ""
        self.due_date_resolution = DueDateResolution.UNKNOWN
        self.clarification_count = 0

    def advance(self) -> Optional[str]:
        self.current_index += 1
        self.reset_candidate_state()
        return self.current_item_id


class ReviewSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], ReviewSession] = {}

    def start(
        self, tenant_id: str, chat_id: str, item_ids: list[str]
    ) -> ReviewSession:
        session = ReviewSession(tenant_id, chat_id, item_ids)
        self._sessions[(tenant_id, chat_id)] = session
        return session

    def get(self, tenant_id: str, chat_id: str) -> ReviewSession | None:
        return self._sessions.get((tenant_id, chat_id))

    def end(self, tenant_id: str, chat_id: str) -> ReviewSession | None:
        return self._sessions.pop((tenant_id, chat_id), None)
