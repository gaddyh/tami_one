"""Insert Contact rows and persist messages from a normalized Green API message event.

Uses an in-memory cache to skip DB queries for known contacts. Only inserts
new contacts — existing rows are never updated (metadata is write-once).
Messages are persisted to ChatMessage (dedup gate) and appended to an in-memory
queue per (tenant_id, chat_id) for later processing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.db.cache import (
    accounts_by_instance,
    chat_senders,
    contacts_by_tenant_chat_id,
    large_chats,
    message_buffer,
)
from app.db.engine import engine
from app.db.models import ChatMessage, Contact, MessageType
from app.conversations.threader import assign_conversation
from app.routers.green_api import MessageDirection, MessageEvent
from app.config import settings

logger = logging.getLogger(__name__)


async def upsert_contact_and_chat(event: MessageEvent, session: Session | None = None) -> dict:
    """Create Contact if needed, persist ChatMessage (dedup gate), assign conversation, buffer.

    Uses the in-memory cache for existence checks. Only inserts — never
    updates existing rows.
    """
    if not event.idInstance:
        logger.warning("Message event has no idInstance, skipping upsert")
        return {"ok": False, "reason": "no idInstance"}

    id_instance = str(event.idInstance)

    account = accounts_by_instance.get(id_instance)
    if not account:
        logger.warning(
            "No WhatsAppAccount for idInstance=%s, skipping upsert",
            id_instance,
        )
        return {"ok": False, "reason": "no account"}

    tenant_id = account.tenant_id
    chat_id = event.chat_id

    own_session = session is None
    if own_session:
        session = Session(engine)

    try:
        created_contact = False

        # --- Contact (insert-only) ---
        contact_key = (tenant_id, chat_id)
        contact = contacts_by_tenant_chat_id.get(contact_key)

        if not contact:
            contact = Contact(
                tenant_id=tenant_id,
                chat_id=chat_id,
                display_name=event.chat_name,
            )
            session.add(contact)
            session.flush()
            contacts_by_tenant_chat_id[contact_key] = contact
            created_contact = True
            logger.info("Created Contact: %s (%s)", contact.id, chat_id)
            session.commit()

        # --- Large chat filter ---
        if chat_id in large_chats:
            logger.info("Skipping large chat %s", chat_id)
            return {
                "ok": True,
                "contact_id": contact.id,
                "created_contact": created_contact,
                "skipped_large_chat": True,
            }

        if event.direction == MessageDirection.INBOUND:
            sender_id = event.sender or event.sender_name or chat_id
            senders = chat_senders.setdefault(chat_id, set())
            senders.add(sender_id)
            if len(senders) > settings.max_group_participants:
                large_chats.add(chat_id)
                logger.info(
                    "Marked chat %s as large: %d distinct senders (max %d)",
                    chat_id,
                    len(senders),
                    settings.max_group_participants,
                )
                return {
                    "ok": True,
                    "contact_id": contact.id,
                    "created_contact": created_contact,
                    "skipped_large_chat": True,
                    "sender_count": len(senders),
                }

        # --- Persist ChatMessage (dedup gate) ---
        msg_type = _map_message_type(event.message_type)
        chat_msg = ChatMessage(
            tenant_id=tenant_id,
            chat_id=chat_id,
            provider_message_id=event.provider_message_id or "",
            direction=MessageDirection(event.direction),
            message_type=msg_type,
            sender_name=event.sender_name,
            sender_chat_id=event.sender,
            text=event.text,
            quoted_message_id=event.quoted_message_id,
            sent_at=event.message_time,
        )

        try:
            session.add(chat_msg)
            session.flush()
        except IntegrityError:
            # Duplicate webhook delivery — skip buffering entirely
            session.rollback()
            logger.info(
                "Duplicate message %s for chat %s — skipping (dedup gate)",
                event.provider_message_id,
                chat_id,
            )
            return {
                "ok": True,
                "contact_id": contact.id,
                "created_contact": created_contact,
                "duplicate": True,
            }

        # --- Assign conversation (threader) ---
        conversation_id = assign_conversation(event, tenant_id, session)
        chat_msg.conversation_id = conversation_id
        session.add(chat_msg)
        session.commit()

        # --- Buffer message for later processing ---
        # Carry conversation_id on the event so drain can group by it
        buffered_event = replace(event, conversation_id=conversation_id)
        await message_buffer.append(tenant_id=tenant_id, event=buffered_event)
        logger.info(
            "Buffered message for (%s, %s) conversation=%s",
            tenant_id,
            chat_id,
            conversation_id,
            extra={"event": asdict(buffered_event)},
        )

        return {
            "ok": True,
            "contact_id": contact.id,
            "created_contact": created_contact,
            "conversation_id": conversation_id,
        }
    finally:
        if own_session:
            session.close()


def _map_message_type(raw: str | None) -> MessageType:
    """Map Green API typeMessage string to internal MessageType enum."""
    if not raw:
        return MessageType.TEXT
    mapping = {
        "textMessage": MessageType.TEXT,
        "extendedTextMessage": MessageType.TEXT,
        "audioMessage": MessageType.AUDIO,
        "imageMessage": MessageType.IMAGE,
        "documentMessage": MessageType.DOCUMENT,
        "videoMessage": MessageType.VIDEO,
    }
    return mapping.get(raw, MessageType.UNKNOWN)
