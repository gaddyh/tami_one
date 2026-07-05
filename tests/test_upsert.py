"""Tests for upsert_contact_and_chat using a temp SQLite DB."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.models import (
    Chat,
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
        # Seed a tenant + WhatsApp account
        tenant = Tenant(name="Test Tenant", kind=TenantKind.SOLO)
        session.add(tenant)
        session.flush()

        account = WhatsAppAccount(
            tenant_id=tenant.id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id="7700673764",
            phone_number="972546610653",
            display_name="Test WhatsApp",
        )
        session.add(account)
        session.commit()
        session.refresh(account)
        session.refresh(tenant)

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


def test_creates_new_contact_and_chat(db_session):
    session, tenant, account = db_session
    event = _make_event()

    result = upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is True
    assert result["created_chat"] is True

    contact = session.exec(select(Contact)).first()
    assert contact is not None
    assert contact.phone_number == "972546610653"
    assert contact.display_name == "Gaddy"
    assert contact.tenant_id == tenant.id

    chat = session.exec(select(Chat)).first()
    assert chat is not None
    assert chat.provider_chat_id == "972546610653@c.us"
    assert chat.is_group is False
    assert chat.primary_contact_id == contact.id
    assert chat.whatsapp_account_id == account.id


def test_second_message_updates_existing(db_session):
    session, _, _ = db_session
    event1 = _make_event(message_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(message_time=datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc))
    result = upsert_contact_and_chat(event2, session=session)

    assert result["ok"] is True
    assert result["created_contact"] is False
    assert result["created_chat"] is False

    contacts = list(session.exec(select(Contact)))
    chats = list(session.exec(select(Chat)))
    assert len(contacts) == 1
    assert len(chats) == 1

    chat = chats[0]
    expected = datetime(2025, 1, 2, 14, 0, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    assert chat.last_message_at == expected


def test_group_chat_sets_is_group_and_no_primary_contact(db_session):
    session, _, _ = db_session
    event = _make_event(
        chat_id="120363000000000000@g.us",
        chat_name="Family Group",
    )

    result = upsert_contact_and_chat(event, session=session)

    assert result["ok"] is True
    chat = session.exec(select(Chat)).first()
    assert chat.is_group is True
    assert chat.primary_contact_id is None


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


def test_fills_missing_display_name_on_existing_contact(db_session):
    session, _, _ = db_session
    event1 = _make_event(chat_name=None)
    upsert_contact_and_chat(event1, session=session)

    contact = session.exec(select(Contact)).first()
    assert contact.display_name is None

    event2 = _make_event(chat_name="Gaddy Updated")
    upsert_contact_and_chat(event2, session=session)

    session.refresh(contact)
    assert contact.display_name == "Gaddy Updated"


def test_different_chats_create_separate_rows(db_session):
    session, _, _ = db_session
    event1 = _make_event(chat_id="972500000001@c.us", chat_name="Alice")
    upsert_contact_and_chat(event1, session=session)

    event2 = _make_event(chat_id="972500000002@c.us", chat_name="Bob")
    upsert_contact_and_chat(event2, session=session)

    contacts = list(session.exec(select(Contact)))
    chats = list(session.exec(select(Chat)))
    assert len(contacts) == 2
    assert len(chats) == 2
