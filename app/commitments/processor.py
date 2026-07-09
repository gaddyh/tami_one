"""Drain the message buffer, extract commitments via LLM, and persist rows.

Drain cycle per conversation batch:
1. Extract with prior conversation summary + new raw messages + chat-scoped existing commitments
2. On success: persist commitments → update summary → mark processed_at (atomic)
3. On failure: increment extraction_attempts, re-enqueue if under cap
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlmodel import Session, select

from app.commitments.extractor import extract_commitments
from app.commitments.models import Commitment, CommitmentStatus, NotificationType
from app.conversations.summarizer import summarize_conversation
from app.db.cache import ChatBufferKey, message_buffer
from app.db.engine import engine
from app.db.models import ChatMessage, CommitmentItem, Conversation, utc_now
from app.routers.green_api import MessageDirection, MessageEvent
from app.config import settings

logger = logging.getLogger(__name__)


def format_messages_for_llm(events: list[MessageEvent]) -> list[dict]:
    """Convert MessageEvent objects into the dict shape the extractor expects."""
    return [
        {
            "senderName": "me" if e.direction == MessageDirection.OUTBOUND else (e.sender_name or e.chat_name or e.sender or "unknown"),
            "senderId": e.sender or e.chat_id,
            "textMessage": e.text,
            "messageId": e.provider_message_id,
        }
        for e in events
        if e.text  # skip empty / non-text messages
    ]


def format_messages_for_summary(events: list[MessageEvent]) -> str:
    """Format messages for the summarizer (same shape as extraction text)."""
    return "\n".join(
        f"{'me' if e.direction == MessageDirection.OUTBOUND else (e.sender_name or e.sender or 'unknown')}: "
        f"{e.text or ''}"
        for e in events
        if e.text
    )


async def drain_and_process() -> dict[ChatBufferKey, list[CommitmentItem]]:
    """Drain the message buffer and run commitment extraction per conversation.

    Groups by (tenant_id, chat_id, conversation_id). Extracts with prior
    conversation summary + new messages. On success: persists commitments,
    updates summary, marks processed_at. On failure: increments attempts,
    re-enqueues if under cap.

    Returns a mapping of (tenant_id, chat_id) -> committed CommitmentItem rows.
    """
    drained = await message_buffer.drain()
    results: dict[ChatBufferKey, list[CommitmentItem]] = {}

    # Group events by (tenant_id, chat_id, conversation_id)
    grouped: dict[tuple[str, str, str | None], list[MessageEvent]] = defaultdict(list)
    for (tenant_id, chat_id), events in drained.items():
        for event in events:
            conv_id = event.conversation_id
            grouped[(tenant_id, chat_id, conv_id)].append(event)

    for (tenant_id, chat_id, conversation_id), events in grouped.items():
        if not events:
            continue

        messages = format_messages_for_llm(events)
        if not messages:
            continue

        chat_name = events[0].chat_name
        source_message_ids = [
            e.provider_message_id for e in events if e.provider_message_id
        ]

        # Load prior conversation summary (empty if no conversation or new)
        prior_summary = ""
        if conversation_id:
            with Session(engine) as session:
                conv = session.get(Conversation, conversation_id)
                if conv:
                    prior_summary = conv.summary

        # Load chat-scoped existing commitments (NOT conversation-scoped)
        with Session(engine) as session:
            existing_rows = session.exec(
                select(CommitmentItem).where(
                    CommitmentItem.tenant_id == tenant_id,
                    CommitmentItem.chat_id == chat_id,
                    CommitmentItem.status != CommitmentStatus.DISMISSED,
                )
            ).all()

            existing = [
                Commitment(
                    id=r.id,
                    chat_id=r.chat_id,
                    committed_party=r.committed_party,
                    required_action=r.required_action,
                    deadline=r.deadline,
                    context=r.context,
                    status=r.status,
                    notification=r.notification,
                )
                for r in existing_rows
            ]

        # --- Extract with prior summary + new messages ---
        try:
            commitments = await extract_commitments(
                chat_id=chat_id,
                chat_name=chat_name,
                messages=messages,
                existing=existing,
                conversation_summary=prior_summary,
            )
        except Exception:
            logger.exception(
                "Failed to extract commitments for chat %s conversation %s",
                chat_id,
                conversation_id,
            )
            await _handle_extraction_failure(
                tenant_id, chat_id, source_message_ids, events
            )
            continue

        # --- On success: persist commitments → update summary → mark processed ---
        items: list[CommitmentItem] = []
        with Session(engine) as session:
            for c in commitments:
                if c.id:
                    existing_item = session.get(CommitmentItem, c.id)
                    if existing_item and existing_item.tenant_id == tenant_id:
                        existing_item.committed_party = c.committed_party
                        existing_item.required_action = c.required_action
                        existing_item.deadline = c.deadline
                        existing_item.context = c.context
                        existing_item.status = CommitmentStatus(c.status)
                        existing_item.notification = NotificationType(c.notification)
                        existing_item.source_message_ids = source_message_ids
                        # Don't update conversation_id — it's the origin, not the current conversation
                        session.add(existing_item)
                        items.append(existing_item)
                        continue

                item = CommitmentItem(
                    tenant_id=tenant_id,
                    chat_id=chat_id,
                    conversation_id=conversation_id,
                    committed_party=c.committed_party,
                    required_action=c.required_action,
                    deadline=c.deadline,
                    context=c.context,
                    status=c.status,
                    notification=c.notification,
                    source_message_ids=source_message_ids,
                )
                session.add(item)
                items.append(item)

            # Update conversation summary (extract first, summarize after)
            if conversation_id:
                try:
                    new_summary = await summarize_conversation(
                        prior_summary=prior_summary,
                        new_messages=format_messages_for_summary(events),
                    )
                    conv = session.get(Conversation, conversation_id)
                    if conv:
                        conv.summary = new_summary
                        conv.updated_at = utc_now()
                        session.add(conv)
                except Exception:
                    logger.exception(
                        "Failed to update summary for conversation %s",
                        conversation_id,
                    )

            # Mark ChatMessage rows as processed
            if source_message_ids:
                for msg_row in session.exec(
                    select(ChatMessage).where(
                        ChatMessage.tenant_id == tenant_id,
                        ChatMessage.chat_id == chat_id,
                        ChatMessage.provider_message_id.in_(
                            [mid for mid in source_message_ids if mid]
                        ),
                    )
                ).all():
                    msg_row.processed_at = utc_now()
                    session.add(msg_row)

            session.commit()
            logger.info(
                "Persisted %d commitment(s) for chat %s conversation %s",
                len(items),
                chat_id,
                conversation_id,
            )

        results[(tenant_id, chat_id)] = items

    return results


async def _handle_extraction_failure(
    tenant_id: str,
    chat_id: str,
    source_message_ids: list[str | None],
    events: list[MessageEvent],
) -> None:
    """Increment extraction_attempts and re-enqueue if under cap."""
    with Session(engine) as session:
        for msg_row in session.exec(
            select(ChatMessage).where(
                ChatMessage.tenant_id == tenant_id,
                ChatMessage.chat_id == chat_id,
                ChatMessage.provider_message_id.in_(
                    [mid for mid in source_message_ids if mid]
                ),
            )
        ).all():
            msg_row.extraction_attempts += 1
            session.add(msg_row)
        session.commit()

        # Check if any row has exceeded the cap
        maxed = any(
            row.extraction_attempts >= settings.max_extraction_attempts
            for row in session.exec(
                select(ChatMessage).where(
                    ChatMessage.tenant_id == tenant_id,
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.provider_message_id.in_(
                        [mid for mid in source_message_ids if mid]
                    ),
                )
            ).all()
        )

    if maxed:
        logger.error(
            "Poison batch for chat %s — extraction_attempts at cap, not re-enqueueing",
            chat_id,
        )
        return

    # Re-enqueue for next drain cycle
    await message_buffer.requeue_front({(tenant_id, chat_id): events})

