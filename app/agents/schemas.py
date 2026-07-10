from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field


class ReminderStatus(StrEnum):
    CREATED = "created"
    NEEDS_CLARIFICATION = "needs_clarification"
    IGNORED = "ignored"


class ConversationTurn(BaseModel):
    role: str
    text: str


class Reminder(BaseModel):
    reminder_id: str
    what: str
    when: datetime
    created_at: datetime


class ReminderInput(BaseModel):
    input_id: str
    text: str
    current_time: datetime
    prior_context: list[ConversationTurn] | None = None


class ReminderOutput(BaseModel):
    input_id: str
    status: ReminderStatus
    reminder: Reminder | None = None
    clarification_question: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    language: str = "en"
    rendered_message: str | None = None
