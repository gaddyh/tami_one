"""Derived conversation state — pure function of last_message_at, no stored column."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import settings
from app.db.models import Conversation


def conversation_state(conv: Conversation, now: datetime | None = None) -> str:
    """Return 'active', 'dormant', or 'closed' based on last_message_at age."""
    if now is None:
        from app.db.models import utc_now
        now = utc_now()

    last_at = conv.last_message_at
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age = now - last_at
    if age > timedelta(days=settings.conversation_closed_days):
        return "closed"
    if age > timedelta(hours=settings.conversation_dormant_hours):
        return "dormant"
    return "active"
