"""Insert Contact and Chat rows from a normalized Green API message event.

Uses an in-memory cache to skip DB queries for known rows. Only inserts
new rows — existing rows are never updated (metadata is write-once).
"""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from app.db.cache import (
    accounts_by_instance,
    chats_by_tenant_chat_id,
    contacts_by_tenant_chat_id,
)
from app.db.engine import engine
from app.db.models import Chat, Contact, WhatsAppAccount
from app.routers.green_api import MessageEvent

logger = logging.getLogger(__name__)


def _extract_phone(chat_id: str) -> str | None:
    """Extract phone number from a WhatsApp chat ID like '972546610653@c.us'."""
    if "@" in chat_id:
        return chat_id.split("@")[0]
    return chat_id


def _is_group(chat_id: str) -> bool:
    return chat_id.endswith("@g.us")


def upsert_contact_and_chat(event: MessageEvent, session: Session | None = None) -> dict:
    """Create Contact and Chat rows if they don't exist yet.

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
    phone = _extract_phone(event.chat_id)
    is_group = _is_group(event.chat_id)

    own_session = session is None
    if own_session:
        session = Session(engine)

    try:
        created_contact = False
        created_chat = False

        # --- Contact (insert-only) ---
        contact_key = (tenant_id, phone)
        contact = contacts_by_tenant_chat_id.get(contact_key)

        if not contact:
            contact = Contact(
                tenant_id=tenant_id,
                chat_id=phone,
                display_name=event.chat_name,
            )
            session.add(contact)
            session.flush()
            contacts_by_tenant_chat_id[contact_key] = contact
            created_contact = True
            logger.info("Created Contact: %s (%s)", contact.id, phone)

        # --- Chat (insert-only) ---
        chat_key = (tenant_id, event.chat_id)
        chat = chats_by_tenant_chat_id.get(chat_key)

        if not chat:
            chat = Chat(
                tenant_id=tenant_id,
                whatsapp_account_id=account.id,
                provider_chat_id=event.chat_id,
                title=event.chat_name,
                is_group=is_group,
                primary_contact_id=contact.id if not is_group else None,
                last_message_at=event.message_time,
            )
            session.add(chat)
            session.flush()
            chats_by_tenant_chat_id[chat_key] = chat
            created_chat = True
            logger.info("Created Chat: %s (%s)", chat.id, event.chat_id)

        if created_contact or created_chat:
            session.commit()

        return {
            "ok": True,
            "contact_id": contact.id,
            "chat_id": chat.id,
            "created_contact": created_contact,
            "created_chat": created_chat,
        }
    finally:
        if own_session:
            session.close()
