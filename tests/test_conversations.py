"""Tests for conversation threading, derived state, and digest logic."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.config import settings
from app.conversations.state import conversation_state
from app.db.models import (
    ChatMessage,
    CommitmentItem,
    CommitmentStatus,
    Conversation,
    Contact,
    MessageDirection,
    MessageType,
    Tenant,
    WhatsAppAccount,
    WhatsAppProvider,
    new_id,
    utc_now,
)


# --- Fixtures ---

@pytest.fixture
def db_session():
    """In-memory SQLite session for each test."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    tenant = Tenant(id="tenant-1", name="Test")
    session.add(tenant)
    session.commit()

    yield session, tenant, engine
    session.close()


def _make_msg(
    tenant_id: str,
    chat_id: str = "972500000001@c.us",
    provider_message_id: str = "msg-1",
    sent_at: datetime | None = None,
    direction: MessageDirection = MessageDirection.INBOUND,
    conversation_id: str | None = None,
    quoted_message_id: str | None = None,
    text: str = "hello",
) -> ChatMessage:
    return ChatMessage(
        tenant_id=tenant_id,
        chat_id=chat_id,
        provider_message_id=provider_message_id,
        direction=direction,
        message_type=MessageType.TEXT,
        text=text,
        sent_at=sent_at or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        conversation_id=conversation_id,
        quoted_message_id=quoted_message_id,
    )


# --- Derived state tests ---

class TestConversationState:
    def test_active(self):
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=datetime.now(timezone.utc),
        )
        assert conversation_state(conv) == "active"

    def test_dormant(self):
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert conversation_state(conv) == "dormant"

    def test_closed(self):
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=datetime.now(timezone.utc) - timedelta(days=8),
        )
        assert conversation_state(conv) == "closed"

    def test_boundary_dormant(self):
        now = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=now - timedelta(hours=24, minutes=1),
        )
        assert conversation_state(conv, now) == "dormant"

    def test_boundary_closed(self):
        now = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=now - timedelta(days=7, minutes=1),
        )
        assert conversation_state(conv, now) == "closed"

    def test_tz_naive_safe(self):
        """SQLite may return tz-naive datetimes — helper must handle it."""
        conv = Conversation(
            tenant_id="t1",
            chat_id="c1",
            last_message_at=datetime(2025, 1, 1, 12, 0, 0),  # naive
        )
        now = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)  # aware
        # Should not raise
        state = conversation_state(conv, now)
        assert state == "active"


# --- Threading tests ---

class TestThreading:
    def test_gap_sessionization_same_conversation(self, db_session):
        """Messages within the gap threshold join the same conversation."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # First message — creates a new conversation
        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="m1")
        session.add(msg1)
        session.flush()
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent
        event1 = MessageEvent(
            provider_message_id="m1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="hello", raw_type_webhook="incomingMessageReceived",
        )
        conv_id1 = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id1
        session.add(msg1)
        session.commit()

        # Second message 10 min later — should join same conversation
        t1 = t0 + timedelta(minutes=10)
        msg2 = _make_msg(tenant.id, sent_at=t1, provider_message_id="m2")
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="m2", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="world", raw_type_webhook="incomingMessageReceived",
        )
        conv_id2 = assign_conversation(event2, tenant.id, session)
        assert conv_id1 == conv_id2

    def test_gap_sessionization_new_conversation(self, db_session):
        """Messages beyond the gap threshold start a new conversation."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="m1")
        session.add(msg1)
        session.flush()
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent
        event1 = MessageEvent(
            provider_message_id="m1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="hello", raw_type_webhook="incomingMessageReceived",
        )
        conv_id1 = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id1
        session.add(msg1)
        session.commit()

        # Second message 2 hours later — new conversation
        t1 = t0 + timedelta(hours=2)
        msg2 = _make_msg(tenant.id, sent_at=t1, provider_message_id="m2")
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="m2", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="new topic", raw_type_webhook="incomingMessageReceived",
        )
        conv_id2 = assign_conversation(event2, tenant.id, session)
        assert conv_id1 != conv_id2

    def test_quoted_reply_inherits_active_conversation(self, db_session):
        """Quoted reply to a message in an active conversation inherits it."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="orig-1")
        session.add(msg1)
        session.flush()
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent
        event1 = MessageEvent(
            provider_message_id="orig-1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="original", raw_type_webhook="incomingMessageReceived",
        )
        conv_id1 = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id1
        session.add(msg1)
        session.commit()

        # Reply quoting orig-1, 10 min later
        t1 = t0 + timedelta(minutes=10)
        msg2 = _make_msg(
            tenant.id, sent_at=t1, provider_message_id="reply-1",
            quoted_message_id="orig-1",
        )
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="reply-1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="reply", raw_type_webhook="incomingMessageReceived",
            quoted_message_id="orig-1",
        )
        conv_id2 = assign_conversation(event2, tenant.id, session)
        assert conv_id1 == conv_id2

    def test_quoted_reply_to_closed_starts_new(self, db_session):
        """Quoted reply to a message in a closed conversation starts a new one."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="orig-1")
        session.add(msg1)
        session.flush()
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent
        event1 = MessageEvent(
            provider_message_id="orig-1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="original", raw_type_webhook="incomingMessageReceived",
        )
        conv_id1 = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id1
        session.add(msg1)
        session.commit()

        # Reply 10 days later (closed) quoting orig-1
        t1 = t0 + timedelta(days=10)
        msg2 = _make_msg(
            tenant.id, sent_at=t1, provider_message_id="reply-1",
            quoted_message_id="orig-1",
        )
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="reply-1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="btw", raw_type_webhook="incomingMessageReceived",
            quoted_message_id="orig-1",
        )
        conv_id2 = assign_conversation(event2, tenant.id, session)
        assert conv_id1 != conv_id2

    def test_quoted_reply_lookup_miss_falls_back_to_gap(self, db_session):
        """Quoted reply to a message that predates persistence falls back to gap rule."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # First message creates a conversation
        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="m1")
        session.add(msg1)
        session.flush()
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent
        event1 = MessageEvent(
            provider_message_id="m1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="hello", raw_type_webhook="incomingMessageReceived",
        )
        conv_id1 = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id1
        session.add(msg1)
        session.commit()

        # Reply 5 min later quoting a non-existent message
        t1 = t0 + timedelta(minutes=5)
        msg2 = _make_msg(
            tenant.id, sent_at=t1, provider_message_id="m2",
            quoted_message_id="nonexistent",
        )
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="m2", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="reply", raw_type_webhook="incomingMessageReceived",
            quoted_message_id="nonexistent",
        )
        conv_id2 = assign_conversation(event2, tenant.id, session)
        # Should fall back to gap rule and join the same conversation
        assert conv_id1 == conv_id2

    def test_last_message_at_bumped_at_ingest(self, db_session):
        """assign_conversation bumps Conversation.last_message_at."""
        from app.conversations.threader import assign_conversation

        session, tenant, engine = db_session
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(minutes=10)

        from app.routers.green_api import MessageDirection as GDirection, MessageEvent

        # First message — insert ChatMessage (like upsert does), then thread
        msg1 = _make_msg(tenant.id, sent_at=t0, provider_message_id="m1")
        session.add(msg1)
        session.flush()
        event1 = MessageEvent(
            provider_message_id="m1", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="hello", raw_type_webhook="incomingMessageReceived",
        )
        conv_id = assign_conversation(event1, tenant.id, session)
        msg1.conversation_id = conv_id
        session.add(msg1)
        session.commit()

        conv = session.get(Conversation, conv_id)
        # SQLite returns tz-naive — normalize for comparison
        assert conv.last_message_at.replace(tzinfo=None) == t0.replace(tzinfo=None)

        # Second message — insert ChatMessage, then thread
        msg2 = _make_msg(tenant.id, sent_at=t1, provider_message_id="m2")
        session.add(msg2)
        session.flush()
        event2 = MessageEvent(
            provider_message_id="m2", idInstance=None, wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t1, text="world", raw_type_webhook="incomingMessageReceived",
        )
        assign_conversation(event2, tenant.id, session)
        session.commit()
        session.expire_all()

        conv = session.get(Conversation, conv_id)
        assert conv.last_message_at.replace(tzinfo=None) == t1.replace(tzinfo=None)


# --- Ingestion idempotency test ---

class TestIngestionIdempotency:
    @pytest.mark.asyncio
    async def test_duplicate_message_id_skipped(self, db_session):
        """Duplicate provider_message_id should be rejected by dedup gate."""
        from app.db.upsert import upsert_contact_and_chat
        from app.db.cache import message_buffer, accounts_by_instance, contacts_by_tenant_chat_id
        from app.routers.green_api import MessageDirection as GDirection, MessageEvent

        session, tenant, engine = db_session

        # Set up account in cache
        account = WhatsAppAccount(
            id="acc-1",
            tenant_id=tenant.id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id="7700673764",
        )
        session.add(account)
        session.commit()
        accounts_by_instance["7700673764"] = account

        # Clear buffer
        message_buffer._messages_by_chat.clear()
        contacts_by_tenant_chat_id.clear()

        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = MessageEvent(
            provider_message_id="dup-1", idInstance="7700673764", wId=None,
            chat_id="972500000001@c.us", chat_name="Test", sender="s1",
            sender_name="A", direction=GDirection.INBOUND, message_type="textMessage",
            message_time=t0, text="hello", raw_type_webhook="incomingMessageReceived",
        )

        # First insert succeeds
        result1 = await upsert_contact_and_chat(event, session=session)
        assert result1["ok"] is True
        assert result1.get("duplicate") is not True

        # Second insert with same provider_message_id — should be deduped
        result2 = await upsert_contact_and_chat(event, session=session)
        assert result2["ok"] is True
        assert result2.get("duplicate") is True

        # Buffer should have only 1 message
        drained = message_buffer._messages_by_chat
        key = (tenant.id, "972500000001@c.us")
        assert len(drained.get(key, [])) == 1
