"""In-memory cache of DB rows and message buffer."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import DefaultDict

from sqlmodel import Session, select

from app.db.engine import engine
from app.db.models import Contact, Conversation, WhatsAppAccount
from app.routers.green_api import MessageEvent

logger = logging.getLogger(__name__)

# provider_instance_id -> WhatsAppAccount
accounts_by_instance: dict[str, WhatsAppAccount] = {}

# (tenant_id, chat_id) -> Contact
contacts_by_tenant_chat_id: dict[tuple[str, str | None], Contact] = {}

# chat_id -> set of distinct inbound sender identifiers
chat_senders: dict[str, set[str]] = {}

# chat_ids that have been marked as too large (exceed max_group_participants)
large_chats: set[str] = set()

# chat_id -> list of Conversation (active/dormant only, for fast ingest lookup)
conversations_by_chat: dict[str, list[Conversation]] = {}


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

    def requeue_front_sync(
        self,
        batches: dict[ChatBufferKey, list[MessageEvent]],
    ) -> None:
        """Synchronous version of requeue_front for non-async contexts."""
        for key, failed_messages in batches.items():
            existing = self._messages_by_chat.get(key, [])
            self._messages_by_chat[key] = failed_messages + existing


message_buffer = MessageBuffer()


def load_cache() -> None:
    """Load all rows from DB into in-memory dicts and recover unprocessed messages."""
    from app.conversations.state import conversation_state
    from app.db.models import ChatMessage
    from app.config import settings
    from datetime import datetime, timezone

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

        # Load conversations — only non-closed (derived) for ingest cache
        all_convs = session.exec(select(Conversation)).all()
        conversations_by_chat.clear()
        now = datetime.now(timezone.utc)
        for conv in all_convs:
            state = conversation_state(conv, now)
            if state != "closed":
                conversations_by_chat.setdefault(conv.chat_id, []).append(conv)

    logger.info(
        "Cache loaded: %d accounts, %d contacts, %d chats with conversations",
        len(accounts_by_instance),
        len(contacts_by_tenant_chat_id),
        len(conversations_by_chat),
    )

    # --- Startup recovery: re-enqueue unprocessed messages ---
    _recover_unprocessed_messages()


def _recover_unprocessed_messages() -> None:
    """Re-enqueue ChatMessage rows that were never successfully processed."""
    import asyncio
    from app.db.models import ChatMessage
    from app.routers.green_api import MessageDirection, MessageEvent
    from app.config import settings

    with Session(engine) as session:
        unprocessed = session.exec(
            select(ChatMessage).where(
                ChatMessage.processed_at.is_(None),
                ChatMessage.extraction_attempts < settings.max_extraction_attempts,
            )
        ).all()

        if not unprocessed:
            return

        logger.info("Recovering %d unprocessed messages from DB", len(unprocessed))

        for row in unprocessed:
            # Recover chat_name from Contact cache
            contact = contacts_by_tenant_chat_id.get((row.tenant_id, row.chat_id))
            chat_name = contact.display_name if contact else None

            event = MessageEvent(
                provider_message_id=row.provider_message_id,
                idInstance=None,
                wId=None,
                chat_id=row.chat_id,
                chat_name=chat_name,
                sender=row.sender_chat_id,
                sender_name=row.sender_name,
                direction=MessageDirection(row.direction),
                message_type=row.message_type.value if row.message_type else None,
                message_time=row.sent_at,
                text=row.text,
                raw_type_webhook=None,
                quoted_message_id=row.quoted_message_id,
                conversation_id=row.conversation_id,
            )

            # Schedule the buffer append on the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    message_buffer.append(tenant_id=row.tenant_id, event=event)
                )
            except RuntimeError:
                # No running loop — append synchronously (tests, CLI)
                message_buffer._messages_by_chat[
                    (row.tenant_id, row.chat_id)
                ].append(event)

        logger.info("Recovery complete")
