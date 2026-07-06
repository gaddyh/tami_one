"""Tests for upsert_contact_and_chat using a temp SQLite DB."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.cache import (
    accounts_by_instance,
    chat_senders,
    contacts_by_tenant_chat_id,
    large_chats,
    message_buffer,
)
from app.db.models import (
    Contact,
    Tenant,
    TenantKind,
    WhatsAppAccount,
    WhatsAppProvider,
)
from app.db.upsert import upsert_contact_and_chat
from app.routers.green_api import MessageDirection, MessageEvent


@pytest.fixture()
async def db_session():
    """Create a fresh temp SQLite DB with all tables for each test."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        tenant = Tenant(name="Test Tenant", kind=TenantKind.SOLO)
        session.add(tenant)
        session.flush()

        account = WhatsAppAccount(
            tenant_id=tenant.id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id="7700673764",
            chat_id="972546610653@c.us",
            display_name="Test WhatsApp",
        )
        session.add(account)
        session.commit()
        session.refresh(account)
        session.refresh(tenant)

        accounts_by_instance.clear()
        accounts_by_instance[account.provider_instance_id] = account
        contacts_by_tenant_chat_id.clear()
        await message_buffer.drain()

        yield session, tenant, account

    engine.dispose()
    os.unlink(db_path)


def _make_event(
    chat_id: str = "972546610653@c.us",
    chat_name: str | None = "Gaddy",
    id_instance: str | None = "7700673764",
    message_time: datetime | None = None,
    direction: MessageDirection = MessageDirection.INBOUND,
    sender: str | None = "972546610653@c.us",
    sender_name: str | None = "Gaddy",
    provider_message_id: str = "msg-123",
) -> MessageEvent:
    return MessageEvent(
        provider_message_id=provider_message_id,
        idInstance=id_instance,
        wId=None,
        chat_id=chat_id,
        chat_name=chat_name,
        sender=sender,
        sender_name=sender_name,
        direction=direction,
        message_type="textMessage",
        message_time=message_time or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        text="hello",
        raw_type_webhook="incomingMessageReceived",
    )


async def test_creates_new_contact_and_queues_message(db_session):
    session, tenant, _ = db_session
    event = _make_event()

    result = await upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is True

    contact = session.exec(select(Contact)).first()
    assert contact is not None
    assert contact.chat_id == "972546610653@c.us"
    assert contact.display_name == "Gaddy"
    assert contact.tenant_id == tenant.id

    drained = await message_buffer.drain()
    key = (tenant.id, "972546610653@c.us")
    assert len(drained[key]) == 1
    assert drained[key][0].text == "hello"


async def test_second_message_skips_existing_contact(db_session):
    session, tenant, _ = db_session
    event1 = _make_event(message_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    await upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(message_time=datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc))
    result = await upsert_contact_and_chat(event2, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is False

    contacts = list(session.exec(select(Contact)))
    assert len(contacts) == 1

    drained = await message_buffer.drain()
    key = (tenant.id, "972546610653@c.us")
    assert len(drained[key]) == 2


async def test_no_id_instance_returns_not_ok(db_session):
    session, _, _ = db_session
    event = _make_event(id_instance=None)

    result = await upsert_contact_and_chat(event, session=session)

    assert result["ok"] is False
    assert result["reason"] == "no idInstance"


async def test_unknown_account_returns_not_ok(db_session):
    session, _, _ = db_session
    event = _make_event(id_instance="9999999999")

    result = await upsert_contact_and_chat(event, session=session)

    assert result["ok"] is False
    assert result["reason"] == "no account"


async def test_existing_contact_not_updated(db_session):
    session, _, _ = db_session
    event1 = _make_event(chat_name=None)
    await upsert_contact_and_chat(event1, session=session)

    contact = session.exec(select(Contact)).first()
    assert contact.display_name is None

    event2 = _make_event(chat_name="Gaddy Updated")
    await upsert_contact_and_chat(event2, session=session)

    session.refresh(contact)
    assert contact.display_name is None


async def test_different_chats_create_separate_contacts_and_queues(db_session):
    session, tenant, _ = db_session
    event1 = _make_event(chat_id="972500000001@c.us", chat_name="Alice")
    await upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(chat_id="972500000002@c.us", chat_name="Bob")
    await upsert_contact_and_chat(event2, session=session)

    contacts = list(session.exec(select(Contact)))
    assert len(contacts) == 2

    drained = await message_buffer.drain()
    assert len(drained[(tenant.id, "972500000001@c.us")]) == 1
    assert len(drained[(tenant.id, "972500000002@c.us")]) == 1


async def test_fourth_distinct_sender_marks_chat_large(db_session):
    session, tenant, _ = db_session
    chat_id = "group@c.us"
    chat_senders.clear()
    large_chats.clear()

    senders = [
        ("972500000001@c.us", "Alice"),
        ("972500000002@c.us", "Bob"),
        ("972500000003@c.us", "Carol"),
    ]
    for sender, name in senders:
        event = _make_event(
            chat_id=chat_id,
            chat_name="Test Group",
            sender=sender,
            sender_name=name,
            provider_message_id=f"msg-{sender}",
        )
        await upsert_contact_and_chat(event, session=session)

    drained = await message_buffer.drain()
    assert len(drained[(tenant.id, chat_id)]) == 3

    event = _make_event(
        chat_id=chat_id,
        chat_name="Test Group",
        sender="972500000004@c.us",
        sender_name="Dave",
        provider_message_id="msg-dave",
    )
    result = await upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    assert result["skipped_large_chat"] is True
    assert result["sender_count"] == 4

    drained = await message_buffer.drain()
    assert (tenant.id, chat_id) not in drained


async def test_already_marked_chat_skips_immediately(db_session):
    session, tenant, _ = db_session
    chat_id = "marked-group@c.us"
    chat_senders.clear()
    large_chats.clear()
    large_chats.add(chat_id)

    event = _make_event(
        chat_id=chat_id,
        chat_name="Marked Group",
        sender="972500000001@c.us",
        sender_name="Alice",
    )
    result = await upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    assert result["skipped_large_chat"] is True

    drained = await message_buffer.drain()
    assert (tenant.id, chat_id) not in drained


async def test_outbound_messages_dont_count_as_senders(db_session):
    session, tenant, _ = db_session
    chat_id = "small-group@c.us"
    chat_senders.clear()
    large_chats.clear()

    for i in range(5):
        event = _make_event(
            chat_id=chat_id,
            chat_name="Small Group",
            sender="972546610653@c.us",
            sender_name="me",
            direction=MessageDirection.OUTBOUND,
            provider_message_id=f"out-{i}",
        )
        await upsert_contact_and_chat(event, session=session)

    drained = await message_buffer.drain()
    assert len(drained[(tenant.id, chat_id)]) == 5
    assert chat_id not in large_chats
