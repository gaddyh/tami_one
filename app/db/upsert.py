"""Insert Contact rows and queue messages from a normalized Green API message event.

Uses an in-memory cache to skip DB queries for known contacts. Only inserts
new contacts — existing rows are never updated (metadata is write-once).
Messages are appended to an in-memory queue per (tenant_id, chat_id) for
later processing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from sqlmodel import Session

from app.db.cache import (
    accounts_by_instance,
    chat_senders,
    contacts_by_tenant_chat_id,
    large_chats,
    message_buffer,
)
from app.db.engine import engine
from app.db.models import Contact
from app.routers.green_api import MessageDirection, MessageEvent
from app.config import settings

logger = logging.getLogger(__name__)


async def upsert_contact_and_chat(event: MessageEvent, session: Session | None = None) -> dict:
    """Create Contact if needed and append message to the message buffer.

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

        # --- Buffer message for later processing ---
        await message_buffer.append(tenant_id=tenant_id, event=event)
        logger.info(
            "Buffered message for (%s, %s)",
            tenant_id,
            chat_id,
            extra={"event": asdict(event)},
        )

        return {
            "ok": True,
            "contact_id": contact.id,
            "created_contact": created_contact,
        }
    finally:
        if own_session:
            session.close()
