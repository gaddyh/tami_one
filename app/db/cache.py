"""In-memory cache of DB rows and message buffer."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import DefaultDict

from sqlmodel import Session, select

from app.db.engine import engine
from app.db.models import Contact, WhatsAppAccount
from app.routers.green_api import MessageEvent

logger = logging.getLogger(__name__)

# provider_instance_id -> WhatsAppAccount
accounts_by_instance: dict[str, WhatsAppAccount] = {}

# (tenant_id, chat_id) -> Contact
contacts_by_tenant_chat_id: dict[tuple[str, str | None], Contact] = {}


ChatBufferKey = tuple[str, str]  # tenant_id, chat_id


class MessageBuffer:
    def __init__(self) -> None:
        self._messages_by_chat: DefaultDict[
            ChatBufferKey,
            list[MessageEvent],
        ] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def append(
        self,
        *,
        tenant_id: str,
        event: MessageEvent,
    ) -> None:
        key = (tenant_id, event.chat_id)

        async with self._lock:
            self._messages_by_chat[key].append(event)

    async def drain(self) -> dict[ChatBufferKey, list[MessageEvent]]:
        """Atomically pop all currently buffered messages.

        We copy/swap under lock, then process outside the lock.
        """
        async with self._lock:
            drained = dict(self._messages_by_chat)
            self._messages_by_chat.clear()

        return drained

    async def requeue_front(
        self,
        batches: dict[ChatBufferKey, list[MessageEvent]],
    ) -> None:
        """Put failed batches back at the front.

        Newer messages may already exist in the buffer, so failed messages
        should stay before them.
        """
        async with self._lock:
            for key, failed_messages in batches.items():
                existing = self._messages_by_chat.get(key, [])
                self._messages_by_chat[key] = failed_messages + existing


message_buffer = MessageBuffer()


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
