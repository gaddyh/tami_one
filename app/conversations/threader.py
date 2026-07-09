"""Deterministic conversation threading — zero LLM calls.

Assigns each incoming message to a conversation using:
1. Quoted-reply inheritance (with staleness rule)
2. Time-gap sessionization
3. Conservative merge bias (over-merge degrades gracefully, over-split doesn't)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import settings
from app.conversations.state import conversation_state
from app.db.cache import conversations_by_chat
from app.db.models import ChatMessage, Conversation, new_id, utc_now
from app.routers.green_api import MessageEvent

logger = logging.getLogger(__name__)


def assign_conversation(
    event: MessageEvent,
    tenant_id: str,
    session: Session,
) -> str:
    """Assign a message to a conversation and return the conversation_id.

    Must be called within a DB session. Creates/updates Conversation rows
    and sets ChatMessage.conversation_id.
    """
    chat_id = event.chat_id
    sent_at = event.message_time

    # --- 1. Quoted reply with staleness rule ---
    if event.quoted_message_id:
        conv_id = _try_quoted_reply(
            event.quoted_message_id,
            chat_id,
            tenant_id,
            sent_at,
            session,
        )
        if conv_id:
            _bump_conversation(conv_id, sent_at, session)
            return conv_id

    # --- 2. Time gap sessionization ---
    conv_id = _try_gap_session(
        chat_id, tenant_id, sent_at, session,
        exclude_message_id=event.provider_message_id,
    )
    if conv_id:
        _bump_conversation(conv_id, sent_at, session)
        return conv_id

    # --- 3. New conversation ---
    conv_id = _create_conversation(tenant_id, chat_id, sent_at, session)
    return conv_id


def _try_quoted_reply(
    quoted_message_id: str,
    chat_id: str,
    tenant_id: str,
    now: datetime,
    session: Session,
) -> str | None:
    """Look up the quoted message and inherit its conversation if active/dormant."""
    quoted = session.exec(
        select(ChatMessage).where(
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.chat_id == chat_id,
            ChatMessage.provider_message_id == quoted_message_id,
        )
    ).first()

    if not quoted or not quoted.conversation_id:
        logger.debug(
            "Quoted reply lookup miss: quoted_message_id=%s not found or no conversation",
            quoted_message_id,
        )
        return None

    conv = session.get(Conversation, quoted.conversation_id)
    if not conv:
        return None

    state = conversation_state(conv, now)
    if state == "closed":
        logger.debug(
            "Quoted reply to closed conversation %s — starting new", conv.id
        )
        return None

    return conv.id


def _try_gap_session(
    chat_id: str,
    tenant_id: str,
    now: datetime,
    session: Session,
    exclude_message_id: str | None = None,
) -> str | None:
    """Check if the most recent message in chat is within the session gap."""
    query = (
        select(ChatMessage)
        .where(
            ChatMessage.tenant_id == tenant_id,
            ChatMessage.chat_id == chat_id,
            ChatMessage.conversation_id.is_not(None),
        )
        .order_by(ChatMessage.sent_at.desc())
        .limit(1)
    )
    if exclude_message_id:
        query = query.where(ChatMessage.provider_message_id != exclude_message_id)

    recent = session.exec(query).first()

    if not recent:
        return None

    # Normalize both to aware UTC — SQLite may return naive datetimes
    recent_at = recent.sent_at
    if recent_at.tzinfo is None:
        recent_at = recent_at.replace(tzinfo=timezone.utc)
    now_aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

    gap = now_aware - recent_at
    if gap > timedelta(minutes=settings.session_gap_minutes):
        return None

    # Verify the conversation isn't closed (derived state)
    conv = session.get(Conversation, recent.conversation_id)
    if not conv:
        return None

    state = conversation_state(conv, now)
    if state == "closed":
        return None

    return conv.id


def _create_conversation(
    tenant_id: str,
    chat_id: str,
    sent_at: datetime,
    session: Session,
) -> str:
    """Create a new Conversation row and return its id."""
    conv = Conversation(
        id=new_id(),
        tenant_id=tenant_id,
        chat_id=chat_id,
        summary="",
        last_message_at=sent_at,
        started_at=sent_at,
    )
    session.add(conv)
    session.flush()

    # Update in-memory cache
    from app.db.cache import conversations_by_chat
    convs = conversations_by_chat.setdefault(chat_id, [])
    convs.append(conv)

    logger.info("Created conversation %s for chat %s", conv.id, chat_id)
    return conv.id


def _bump_conversation(
    conv_id: str,
    sent_at: datetime,
    session: Session,
) -> None:
    """Update last_message_at on an existing conversation."""
    conv = session.get(Conversation, conv_id)
    if conv:
        conv.last_message_at = sent_at
        conv.updated_at = utc_now()
        session.merge(conv)
        session.flush()
