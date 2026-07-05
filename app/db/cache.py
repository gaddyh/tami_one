"""In-memory cache of DB rows and message queue."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlmodel import Session, select

from app.db.engine import engine
from app.db.models import Contact, WhatsAppAccount
from app.routers.green_api import MessageEvent

logger = logging.getLogger(__name__)

# provider_instance_id -> WhatsAppAccount
accounts_by_instance: dict[str, WhatsAppAccount] = {}

# (tenant_id, chat_id) -> Contact
contacts_by_tenant_chat_id: dict[tuple[str, str | None], Contact] = {}

# (tenant_id, chat_id) -> list of MessageEvent waiting to be processed
messages_by_chat: dict[tuple[str, str], list[MessageEvent]] = defaultdict(list)


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

    logger.info(
        "Cache loaded: %d accounts, %d contacts",
        len(accounts_by_instance),
        len(contacts_by_tenant_chat_id),
    )
