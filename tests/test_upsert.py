"""Tests for upsert_contact_and_chat using a temp SQLite DB."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.cache import (
    accounts_by_instance,
    contacts_by_tenant_chat_id,
    messages_by_chat,
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
def db_session():
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
        messages_by_chat.clear()

        yield session, tenant, account

    engine.dispose()
    os.unlink(db_path)


def _make_event(
    chat_id: str = "972546610653@c.us",
    chat_name: str | None = "Gaddy",
    id_instance: str | None = "7700673764",
    message_time: datetime | None = None,
) -> MessageEvent:
    return MessageEvent(
        provider_message_id="msg-123",
        idInstance=id_instance,
        wId=None,
        chat_id=chat_id,
        chat_name=chat_name,
        direction=MessageDirection.INBOUND,
        message_type="textMessage",
        message_time=message_time or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        text="hello",
        raw_type_webhook="incomingMessageReceived",
    )


def test_creates_new_contact_and_queues_message(db_session):
    session, tenant, _ = db_session
    event = _make_event()

    result = upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is True
    assert result["queued_messages"] == 1

    contact = session.exec(select(Contact)).first()
    assert contact is not None
    assert contact.chat_id == "972546610653@c.us"
    assert contact.display_name == "Gaddy"
    assert contact.tenant_id == tenant.id

    key = (tenant.id, "972546610653@c.us")
    assert len(messages_by_chat[key]) == 1
    assert messages_by_chat[key][0].text == "hello"


def test_second_message_skips_existing_contact(db_session):
    session, tenant, _ = db_session
    event1 = _make_event(message_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(message_time=datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc))
    result = upsert_contact_and_chat(event2, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is False
    assert result["queued_messages"] == 2

    contacts = list(session.exec(select(Contact)))
    assert len(contacts) == 1

    key = (tenant.id, "972546610653@c.us")
    assert len(messages_by_chat[key]) == 2


def test_no_id_instance_returns_not_ok(db_session):
    session, _, _ = db_session
    event = _make_event(id_instance=None)

    result = upsert_contact_and_chat(event, session=session)

    assert result["ok"] is False
    assert result["reason"] == "no idInstance"


def test_unknown_account_returns_not_ok(db_session):
    session, _, _ = db_session
    event = _make_event(id_instance="9999999999")

    result = upsert_contact_and_chat(event, session=session)

    assert result["ok"] is False
    assert result["reason"] == "no account"


def test_existing_contact_not_updated(db_session):
    session, _, _ = db_session
    event1 = _make_event(chat_name=None)
    upsert_contact_and_chat(event1, session=session)

    contact = session.exec(select(Contact)).first()
    assert contact.display_name is None

    event2 = _make_event(chat_name="Gaddy Updated")
    upsert_contact_and_chat(event2, session=session)

    session.refresh(contact)
    assert contact.display_name is None


def test_different_chats_create_separate_contacts_and_queues(db_session):
    session, tenant, _ = db_session
    event1 = _make_event(chat_id="972500000001@c.us", chat_name="Alice")
    upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(chat_id="972500000002@c.us", chat_name="Bob")
    upsert_contact_and_chat(event2, session=session)

    contacts = list(session.exec(select(Contact)))
    assert len(contacts) == 2

    assert len(messages_by_chat[(tenant.id, "972500000001@c.us")]) == 1
    assert len(messages_by_chat[(tenant.id, "972500000002@c.us")]) == 1
