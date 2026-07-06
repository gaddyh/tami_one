"""Tests for the commitment extraction and persistence mechanism."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.commitments.extractor import _format_existing
from app.commitments.models import Commitment
from app.commitments.processor import drain_and_process, format_messages_for_llm
from app.db.cache import message_buffer
from app.db.models import (
    CommitmentItem,
    CommitmentItemStatus,
    CommitmentNotification,
    Tenant,
    TenantKind,
)
from app.routers.green_api import MessageDirection, MessageEvent


@pytest.fixture()
async def test_engine():
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
        session.commit()
        session.refresh(tenant)

        await message_buffer.drain()

        yield engine, tenant

    engine.dispose()
    os.unlink(db_path)


def _make_event(
    chat_id: str = "972546610653@c.us",
    chat_name: str | None = "Gaddy",
    text: str | None = "hello",
    provider_message_id: str = "msg-1",
) -> MessageEvent:
    return MessageEvent(
        provider_message_id=provider_message_id,
        idInstance="7700673764",
        wId=None,
        chat_id=chat_id,
        chat_name=chat_name,
        direction=MessageDirection.INBOUND,
        message_type="textMessage",
        message_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        text=text,
        raw_type_webhook="incomingMessageReceived",
    )


# ─── format_messages_for_llm ──────────────────────────────────────────


def test_format_messages_skips_non_text():
    events = [
        _make_event(text="hello", provider_message_id="m1"),
        _make_event(text=None, provider_message_id="m2"),
        _make_event(text="", provider_message_id="m3"),
    ]
    result = format_messages_for_llm(events)
    assert len(result) == 1
    assert result[0]["textMessage"] == "hello"
    assert result[0]["messageId"] == "m1"


def test_format_messages_maps_fields():
    event = _make_event(
        chat_id="group@c.us",
        chat_name="Family Group",
        text="send the report by Friday",
        provider_message_id="msg-99",
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "Family Group"
    assert result[0]["senderId"] == "group@c.us"
    assert result[0]["textMessage"] == "send the report by Friday"
    assert result[0]["messageId"] == "msg-99"


# ─── _format_existing ─────────────────────────────────────────────────


def test_format_existing_none():
    assert _format_existing(None) == "(none)"


def test_format_existing_empty():
    assert _format_existing([]) == "(none)"


def test_format_existing_lists_items():
    commitments = [
        Commitment(
            id="abc-123",
            chat_id="group@c.us",
            committed_party="Alice",
            required_action="Send report",
            deadline="Friday",
            context="Team meeting",
            status="open",
        ),
    ]
    result = _format_existing(commitments)
    assert "id=abc-123" in result
    assert "party=Alice" in result
    assert "action=Send report" in result
    assert "status=open" in result


# ─── drain_and_process ────────────────────────────────────────────────


async def test_drain_inserts_new_commitments(test_engine):
    engine, tenant = test_engine

    event = _make_event(text="I will send the documents by tomorrow")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    mock_commitments = [
        Commitment(
            id=None,
            chat_id="972546610653@c.us",
            committed_party="Gaddy",
            required_action="Send the documents",
            deadline="tomorrow",
            context="I will send the documents by tomorrow",
            status="open",
        ),
    ]

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=mock_commitments,
        ),
    ):
        results = await drain_and_process()

    key = (tenant.id, "972546610653@c.us")
    assert key in results
    assert len(results[key]) == 1

    with Session(engine) as session:
        items = session.exec(
            select(CommitmentItem).where(CommitmentItem.tenant_id == tenant.id)
        ).all()
        assert len(items) == 1
        assert items[0].committed_party == "Gaddy"
        assert items[0].required_action == "Send the documents"
        assert items[0].status == CommitmentItemStatus.OPEN
        assert items[0].source_message_ids == ["msg-1"]


async def test_drain_updates_existing_commitment(test_engine):
    engine, tenant = test_engine

    with Session(engine) as session:
        existing_item = CommitmentItem(
            tenant_id=tenant.id,
            chat_id="972546610653@c.us",
            committed_party="Gaddy",
            required_action="Send the documents",
            deadline="tomorrow",
            context="I will send the documents by tomorrow",
            status=CommitmentItemStatus.OPEN,
            notification=CommitmentNotification.DAILY_DIGEST,
            source_message_ids=["msg-1"],
        )
        session.add(existing_item)
        session.commit()
        session.refresh(existing_item)
        existing_id = existing_item.id

    event = _make_event(text="I sent the documents", provider_message_id="msg-2")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    mock_commitments = [
        Commitment(
            id=existing_id,
            chat_id="972546610653@c.us",
            committed_party="Gaddy",
            required_action="Send the documents",
            deadline="tomorrow",
            context="I will send the documents by tomorrow",
            status="done",
        ),
    ]

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=mock_commitments,
        ),
    ):
        results = await drain_and_process()

    key = (tenant.id, "972546610653@c.us")
    assert len(results[key]) == 1

    with Session(engine) as session:
        items = session.exec(
            select(CommitmentItem).where(CommitmentItem.tenant_id == tenant.id)
        ).all()
        assert len(items) == 1
        assert items[0].id == existing_id
        assert items[0].status == CommitmentItemStatus.DONE
        assert "msg-2" in items[0].source_message_ids


async def test_drain_no_commitments(test_engine):
    engine, tenant = test_engine

    event = _make_event(text="how are you?")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        results = await drain_and_process()

    key = (tenant.id, "972546610653@c.us")
    assert key in results
    assert results[key] == []

    with Session(engine) as session:
        items = session.exec(
            select(CommitmentItem).where(CommitmentItem.tenant_id == tenant.id)
        ).all()
        assert len(items) == 0


async def test_drain_requeues_on_extractor_failure(test_engine):
    engine, tenant = test_engine

    event = _make_event(text="I will do it tomorrow")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenAI is down"),
        ),
    ):
        results = await drain_and_process()

    key = (tenant.id, "972546610653@c.us")
    assert key not in results

    drained = await message_buffer.drain()
    assert key in drained
    assert len(drained[key]) == 1
    assert drained[key][0].text == "I will do it tomorrow"


async def test_drain_skips_non_text_messages(test_engine):
    engine, tenant = test_engine

    event = _make_event(text=None, provider_message_id="audio-1")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    mock_fn = AsyncMock(return_value=[])

    with (
        patch("app.commitments.processor.engine", engine),
        patch("app.commitments.processor.extract_commitments", mock_fn),
    ):
        results = await drain_and_process()

    mock_fn.assert_not_called()
    assert len(results) == 0


async def test_drain_passes_existing_to_extractor(test_engine):
    engine, tenant = test_engine

    with Session(engine) as session:
        existing_item = CommitmentItem(
            tenant_id=tenant.id,
            chat_id="972546610653@c.us",
            committed_party="Bob",
            required_action="Pay invoice",
            deadline="Monday",
            context="Pay the invoice by Monday",
            status=CommitmentItemStatus.OPEN,
        )
        session.add(existing_item)
        session.commit()
        session.refresh(existing_item)

    event = _make_event(text="I paid the invoice", provider_message_id="msg-2")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    mock_fn = AsyncMock(return_value=[])

    with (
        patch("app.commitments.processor.engine", engine),
        patch("app.commitments.processor.extract_commitments", mock_fn),
    ):
        await drain_and_process()

    mock_fn.assert_awaited_once()
    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["existing"] is not None
    assert len(call_kwargs["existing"]) == 1
    assert call_kwargs["existing"][0].id == existing_item.id
    assert call_kwargs["existing"][0].required_action == "Pay invoice"


async def test_drain_does_not_fetch_dismissed_commitments(test_engine):
    engine, tenant = test_engine

    with Session(engine) as session:
        dismissed = CommitmentItem(
            tenant_id=tenant.id,
            chat_id="972546610653@c.us",
            committed_party="Bob",
            required_action="Old task",
            deadline=None,
            context="outdated",
            status=CommitmentItemStatus.DISMISSED,
        )
        session.add(dismissed)
        session.commit()

    event = _make_event(text="new message")
    await message_buffer.append(tenant_id=tenant.id, event=event)

    mock_fn = AsyncMock(return_value=[])

    with (
        patch("app.commitments.processor.engine", engine),
        patch("app.commitments.processor.extract_commitments", mock_fn),
    ):
        await drain_and_process()

    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["existing"] == []


async def test_drain_processes_multiple_chats(test_engine):
    engine, tenant = test_engine

    event1 = _make_event(
        chat_id="chat-a@c.us",
        chat_name="Chat A",
        text="I will call you",
        provider_message_id="m-a",
    )
    event2 = _make_event(
        chat_id="chat-b@c.us",
        chat_name="Chat B",
        text="Send the contract",
        provider_message_id="m-b",
    )
    await message_buffer.append(tenant_id=tenant.id, event=event1)
    await message_buffer.append(tenant_id=tenant.id, event=event2)

    async def _per_chat_extract(*, chat_id, **kwargs):
        if chat_id == "chat-a@c.us":
            return [Commitment(
                id=None,
                chat_id="chat-a@c.us",
                committed_party="Alice",
                required_action="Call back",
                context="I will call you",
            )]
        return [Commitment(
            id=None,
                chat_id="chat-b@c.us",
                committed_party="Bob",
                required_action="Send contract",
                context="Send the contract",
        )]

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new=AsyncMock(side_effect=_per_chat_extract),
        ),
    ):
        results = await drain_and_process()

    assert len(results[(tenant.id, "chat-a@c.us")]) == 1
    assert len(results[(tenant.id, "chat-b@c.us")]) == 1

    with Session(engine) as session:
        items = session.exec(
            select(CommitmentItem).where(CommitmentItem.tenant_id == tenant.id)
        ).all()
        assert len(items) == 2
