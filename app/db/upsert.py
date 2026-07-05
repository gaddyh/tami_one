"""Upsert Contact and Chat rows from a normalized Green API message event."""

from __future__ import annotations

import logging

from sqlmodel import Session, select

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
    """Create or update Contact and Chat rows for a Green API message event.

    Looks up the WhatsAppAccount by provider_instance_id (idInstance) to
    determine the tenant. If no account is found, does nothing.
    """
    if not event.idInstance:
        logger.warning("Message event has no idInstance, skipping upsert")
        return {"ok": False, "reason": "no idInstance"}

    if session is None:
        with Session(engine) as session:
            return _upsert(session, event)

    return _upsert(session, event)


def _upsert(session: Session, event: MessageEvent) -> dict:
    account = session.exec(
        select(WhatsAppAccount).where(
            WhatsAppAccount.provider_instance_id == event.idInstance
        )
    ).first()

    if not account:
        logger.warning(
            "No WhatsAppAccount for idInstance=%s, skipping upsert",
            event.idInstance,
        )
        return {"ok": False, "reason": "no account"}

    tenant_id = account.tenant_id
    created_contact = False
    created_chat = False

    # --- Contact ---
    phone = _extract_phone(event.chat_id)
    contact = session.exec(
        select(Contact).where(
            Contact.tenant_id == tenant_id,
            Contact.phone_number == phone,
        )
    ).first()

    if not contact:
        contact = Contact(
            tenant_id=tenant_id,
            phone_number=phone,
            display_name=event.chat_name,
        )
        session.add(contact)
        session.flush()
        created_contact = True
        logger.info("Created Contact: %s (%s)", contact.id, phone)
    else:
        if event.chat_name and not contact.display_name:
            contact.display_name = event.chat_name
            session.add(contact)

    # --- Chat ---
    chat = session.exec(
        select(Chat).where(
            Chat.tenant_id == tenant_id,
            Chat.provider_chat_id == event.chat_id,
        )
    ).first()

    if not chat:
        chat = Chat(
            tenant_id=tenant_id,
            whatsapp_account_id=account.id,
            provider_chat_id=event.chat_id,
            title=event.chat_name,
            is_group=_is_group(event.chat_id),
            primary_contact_id=contact.id if not _is_group(event.chat_id) else None,
            last_message_at=event.message_time,
        )
        session.add(chat)
        session.flush()
        created_chat = True
        logger.info("Created Chat: %s (%s)", chat.id, event.chat_id)
    else:
        chat.last_message_at = event.message_time
        if event.chat_name and not chat.title:
            chat.title = event.chat_name
        session.add(chat)

    session.commit()

    return {
        "ok": True,
        "contact_id": contact.id,
        "chat_id": chat.id,
        "created_contact": created_contact,
        "created_chat": created_chat,
    }
