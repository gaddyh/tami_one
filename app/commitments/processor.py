"""Drain the message buffer, extract commitments via LLM, and persist rows."""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from app.commitments.extractor import extract_commitments
from app.commitments.models import Commitment, CommitmentStatus, NotificationType
from app.db.cache import ChatBufferKey, message_buffer
from app.db.engine import engine
from app.db.models import CommitmentItem
from app.routers.green_api import MessageDirection, MessageEvent

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


async def drain_and_process() -> dict[ChatBufferKey, list[CommitmentItem]]:
    """Drain the message buffer and run commitment extraction per chat.

    Returns a mapping of (tenant_id, chat_id) -> committed CommitmentItem rows.
    """
    drained = await message_buffer.drain()
    results: dict[ChatBufferKey, list[CommitmentItem]] = {}

    for (tenant_id, chat_id), events in drained.items():
        if not events:
            continue

        messages = format_messages_for_llm(events)
        if not messages:
            continue

        chat_name = events[0].chat_name
        source_message_ids = [
            e.provider_message_id for e in events if e.provider_message_id
        ]

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

        try:
            commitments = await extract_commitments(
                chat_id=chat_id,
                chat_name=chat_name,
                messages=messages,
                existing=existing,
            )
        except Exception:
            logger.exception(
                "Failed to extract commitments for chat %s", chat_id
            )
            await message_buffer.requeue_front({(tenant_id, chat_id): events})
            continue

        if not commitments:
            logger.info("No commitments found for chat %s", chat_id)
            results[(tenant_id, chat_id)] = []
            continue

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
                        session.add(existing_item)
                        items.append(existing_item)
                        continue

                item = CommitmentItem(
                    tenant_id=tenant_id,
                    chat_id=chat_id,
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

            session.commit()
            logger.info(
                "Persisted %d commitment(s) for chat %s",
                len(items),
                chat_id,
            )

        results[(tenant_id, chat_id)] = items

    return results
