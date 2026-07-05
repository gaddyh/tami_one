"""In-memory cache of DB rows to avoid queries on every message."""

from __future__ import annotations

import logging

from sqlmodel import Session, select

from app.db.engine import engine
from app.db.models import Chat, Contact, WhatsAppAccount

logger = logging.getLogger(__name__)

# provider_instance_id -> WhatsAppAccount
accounts_by_instance: dict[str, WhatsAppAccount] = {}

# (tenant_id, chat_id) -> Contact
contacts_by_tenant_chat_id: dict[tuple[str, str | None], Contact] = {}

# (tenant_id, provider_chat_id) -> Chat
chats_by_tenant_chat_id: dict[tuple[str, str], Chat] = {}


def load_cache() -> None:
    """Load all rows from DB into in-memory dicts."""
    with Session(engine) as session:
        accounts = session.exec(select(WhatsAppAccount)).all()
        accounts_by_instance.clear()
        accounts_by_instance.update(
            {a.provider_instance_id: a for a in accounts}
        )

        contacts = session.exec(select(Contact)).all()
        contacts_by_tenant_chat_id.clear()
        contacts_by_tenant_chat_id.update(
            {(c.tenant_id, c.chat_id): c for c in contacts}
        )

        chats = session.exec(select(Chat)).all()
        chats_by_tenant_chat_id.clear()
        chats_by_tenant_chat_id.update(
            {(c.tenant_id, c.provider_chat_id): c for c in chats}
        )

    logger.info(
        "Cache loaded: %d accounts, %d contacts, %d chats",
        len(accounts_by_instance),
        len(contacts_by_tenant_phone),
        len(chats_by_tenant_chat_id),
    )
