"""Digest endpoint — renders open commitments and unanswered threads.

Commitments drive the digest; conversations decorate. Open commitments
appear regardless of their conversation's state. Unanswered threads are
conversations where the last message is inbound with no outbound reply.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from sqlmodel import Session, select

from app.config import settings
from app.conversations.state import conversation_state
from app.db.cache import chat_senders
from app.db.engine import engine
from app.db.models import ChatMessage, CommitmentItem, CommitmentStatus, Conversation
from app.routers.green_api import MessageDirection

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_group_chat(chat_id: str) -> bool:
    """Detect group chats: @g.us suffix or more than 2 distinct senders."""
    if chat_id.endswith("@g.us"):
        return True
    senders = chat_senders.get(chat_id, set())
    return len(senders) > 2


def _verify_auth(authorization: str | None) -> None:
    expected = settings.expected_authorization_header
    if expected and authorization != expected:
        raise HTTPException(status_code=403, detail="Invalid authorization")


@router.get("/digest/{tenant_id}")
async def get_digest(
    tenant_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return the digest: open commitments + conversations awaiting reply."""
    _verify_auth(authorization)

    now = datetime.now(timezone.utc)

    # --- Primary: open commitments (chat-scoped, regardless of conversation state) ---
    open_commitments: list[dict[str, Any]] = []
    with Session(engine) as session:
        rows = session.exec(
            select(CommitmentItem).where(
                CommitmentItem.tenant_id == tenant_id,
                CommitmentItem.status.in_([
                    CommitmentStatus.WAITING,
                    CommitmentStatus.UNCLEAR,
                ]),
            )
        ).all()

        # Preload conversation summaries
        conv_cache: dict[str, Conversation] = {}
        for row in rows:
            if _is_group_chat(row.chat_id):
                continue

            conv_summary = None
            if row.conversation_id and row.conversation_id not in conv_cache:
                conv = session.get(Conversation, row.conversation_id)
                conv_cache[row.conversation_id] = conv
            if row.conversation_id:
                conv = conv_cache.get(row.conversation_id)
                if conv:
                    conv_summary = conv.summary

            age_days = (now - row.created_at).total_seconds() / 86400 if row.created_at else None

            open_commitments.append({
                "chat_id": row.chat_id,
                "conversation_summary": conv_summary,
                "required_action": row.required_action,
                "committed_party": row.committed_party,
                "deadline": row.deadline,
                "status": row.status.value,
                "age_days": round(age_days, 1) if age_days is not None else None,
            })

    # --- Secondary: conversations awaiting reply ---
    awaiting_reply: list[dict[str, Any]] = []
    with Session(engine) as session:
        all_convs = session.exec(
            select(Conversation).where(
                Conversation.tenant_id == tenant_id,
            )
        ).all()

        for conv in all_convs:
            if _is_group_chat(conv.chat_id):
                continue

            state = conversation_state(conv, now)
            if state == "closed":
                continue

            # Find the last message in this conversation
            last_msg = session.exec(
                select(ChatMessage)
                .where(
                    ChatMessage.chat_id == conv.chat_id,
                    ChatMessage.conversation_id == conv.id,
                )
                .order_by(ChatMessage.sent_at.desc())
                .limit(1)
            ).first()

            if not last_msg or last_msg.direction != MessageDirection.INBOUND:
                continue

            # Check for any outbound reply after the last inbound
            reply = session.exec(
                select(ChatMessage)
                .where(
                    ChatMessage.chat_id == conv.chat_id,
                    ChatMessage.conversation_id == conv.id,
                    ChatMessage.direction == MessageDirection.OUTBOUND,
                    ChatMessage.sent_at > last_msg.sent_at,
                )
                .limit(1)
            ).first()

            if reply:
                continue

            hours_waiting = (now - last_msg.sent_at).total_seconds() / 3600

            # Urgency boost: ? in the last inbound message
            has_question = "?" in (last_msg.text or "")

            awaiting_reply.append({
                "chat_id": conv.chat_id,
                "conversation_summary": conv.summary or None,
                "last_inbound_text": last_msg.text,
                "hours_waiting": round(hours_waiting, 1),
                "urgent": has_question,
            })

    return {
        "tenant_id": tenant_id,
        "generated_at": now.isoformat(),
        "open_commitments": open_commitments,
        "awaiting_reply": awaiting_reply,
    }
