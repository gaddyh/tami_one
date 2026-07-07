"""Tests for the commitment extraction and persistence mechanism."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.commitments.commitments_agent import format_existing_commitments, normalize_commitments
from app.commitments.models import Commitment, CommitmentStatus, NotificationType
from app.commitments.processor import drain_and_process, format_messages_for_llm
from app.db.cache import message_buffer
from app.db.models import (
    CommitmentItem,
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
    direction: MessageDirection = MessageDirection.INBOUND,
    sender: str | None = "972546610653@c.us",
    sender_name: str | None = "Gaddy",
) -> MessageEvent:
    return MessageEvent(
        provider_message_id=provider_message_id,
        idInstance="7700673764",
        wId=None,
        chat_id=chat_id,
        chat_name=chat_name,
        sender=sender,
        sender_name=sender_name,
        direction=direction,
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
        sender="972555555555@c.us",
        sender_name="Alice",
        text="send the report by Friday",
        provider_message_id="msg-99",
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "Alice"
    assert result[0]["senderId"] == "972555555555@c.us"
    assert result[0]["textMessage"] == "send the report by Friday"
    assert result[0]["messageId"] == "msg-99"


def test_format_messages_outbound_shows_me():
    event = _make_event(
        text="I will send it tomorrow",
        direction=MessageDirection.OUTBOUND,
        sender_name=None,
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "me"


def test_format_messages_falls_back_to_chat_name():
    event = _make_event(
        chat_name="אקו",
        sender_name=None,
        sender=None,
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "אקו"


# ─── production log data tests ───────────────────────────────────────


def test_format_messages_preserves_hebrew_sender_name():
    event = _make_event(
        chat_id="120363402798412519@g.us",
        chat_name="דולב הורים תשפו",
        sender="972584874367@c.us",
        sender_name="מיקה🤍",
        text="קודם כל, אני רוצה להגיד תודה רבה",
        provider_message_id="ACC67E247DB0C5F3",
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "מיקה🤍"
    assert result[0]["textMessage"] == "קודם כל, אני רוצה להגיד תודה רבה"


def test_format_messages_skips_reaction_message():
    event = _make_event(
        chat_id="120363402798412519@g.us",
        chat_name="דולב הורים תשפו",
        sender="972523722476@c.us",
        sender_name="נטע אגסי",
        text=None,
        provider_message_id="ACE0261269492354",
    )
    result = format_messages_for_llm([event])
    assert len(result) == 0


def test_format_messages_skips_quoted_message():
    event = _make_event(
        chat_id="120363167449170920@g.us",
        chat_name="המלצות לבעלי מקצוע עמק החולה 🏡",
        sender="972528735288@c.us",
        sender_name="מוריה ציפורה",
        text=None,
        provider_message_id="ACC8C4841FFE8493",
    )
    result = format_messages_for_llm([event])
    assert len(result) == 0


def test_format_messages_preserves_multiline_hebrew_text():
    event = _make_event(
        chat_id="972502024942-1412833028@g.us",
        chat_name="תושבי שדה נחמיה",
        sender="972502525306@c.us",
        sender_name="קרן מלול",
        text='נשות קהילה יקרות\nשהשתתפו ב "אישה אישה"- נשכח אבוב עם סימון x בצבע זהב עליו.\nאשמח לקבלו בחזרה.\nתודה🙏',
        provider_message_id="ACC4905FE7ADF55A",
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "קרן מלול"
    assert "נשות קהילה יקרות" in result[0]["textMessage"]
    assert "תודה🙏" in result[0]["textMessage"]


def test_format_messages_outbound_real_data_shows_me():
    event = _make_event(
        chat_id="972507674593@c.us",
        chat_name="מעוז כחלון",
        sender="972546610653@c.us",
        sender_name="גדי היינקין",
        text="הי אחי, מה המצב? \nיש לך במקרה אבוב להשאיל לי ליום שישי?",
        direction=MessageDirection.OUTBOUND,
        provider_message_id="3B5DCD6899DB4F1D",
    )
    result = format_messages_for_llm([event])
    assert result[0]["senderName"] == "me"
    assert "אבוב להשאיל" in result[0]["textMessage"]


def test_format_messages_mixed_chat_real_data():
    events = [
        _make_event(
            chat_id="972502024942-1412833028@g.us",
            chat_name="תושבי שדה נחמיה",
            sender="972504434301@c.us",
            sender_name="מעיין צור",
            text=None,
            provider_message_id="AC8C7B02B9186CA",
        ),
        _make_event(
            chat_id="972502024942-1412833028@g.us",
            chat_name="תושבי שדה נחמיה",
            sender="972526587091@c.us",
            sender_name="Moshe GIL Sibony",
            text="של הבנות שלו",
            provider_message_id="AC84C9FAB2BC11B4",
        ),
        _make_event(
            chat_id="972502024942-1412833028@g.us",
            chat_name="תושבי שדה נחמיה",
            sender="972542205582@c.us",
            sender_name="רועי לשם",
            text="יש לו שותף סמוי (אוליגרך....)",
            provider_message_id="AC76D5A7F368071C",
        ),
    ]
    result = format_messages_for_llm(events)
    assert len(result) == 2
    assert result[0]["senderName"] == "Moshe GIL Sibony"
    assert result[0]["textMessage"] == "של הבנות שלו"
    assert result[1]["senderName"] == "רועי לשם"
    assert result[1]["textMessage"] == "יש לו שותף סמוי (אוליגרך....)"


async def test_extract_real_data_hebrew_multiline_to_llm():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    messages = [
        {"senderName": "קרן מלול", "textMessage": 'נשות קהילה יקרות\nשהשתתפו ב "אישה אישה"- נשכח אבוב.\nאשמח לקבלו בחזרה.\nתודה🙏'},
    ]

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="972502024942-1412833028@g.us",
            chat_name="תושבי שדה נחמיה",
            messages=messages,
        )

    assert "קרן מלול: נשות קהילה יקרות" in captured["messages"]
    assert "תודה🙏" in captured["messages"]


async def test_extract_real_data_outbound_shows_me():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    messages = [
        {"senderName": "me", "textMessage": "הי אחי, מה המצב? \nיש לך במקרה אבוב להשאיל לי ליום שישי?"},
    ]

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="972507674593@c.us",
            chat_name="מעוז כחלון",
            messages=messages,
        )

    assert captured["messages"].startswith("me: הי אחי")
    assert "אבוב להשאיל" in captured["messages"]


# ─── extract_commitments message formatting ──────────────────────────


async def test_extract_uses_sender_name_in_messages_text():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="chat@c.us",
            chat_name="Test",
            messages=[
                {"senderName": "Alice", "textMessage": "hello"},
                {"senderName": "Bob", "textMessage": "hi there"},
            ],
        )

    lines = captured["messages"].split("\n")
    assert lines[0] == "Alice: hello"
    assert lines[1] == "Bob: hi there"


async def test_extract_falls_back_to_sender_id():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="chat@c.us",
            chat_name="Test",
            messages=[
                {"senderId": "972555555555@c.us", "textMessage": "no senderName"},
            ],
        )

    assert captured["messages"] == "972555555555@c.us: no senderName"


async def test_extract_falls_back_to_text_key():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="chat@c.us",
            chat_name="Test",
            messages=[
                {"senderName": "Alice", "text": "uses text key instead"},
            ],
        )

    assert captured["messages"] == "Alice: uses text key instead"


async def test_extract_preserves_me_sender_from_processor():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="chat@c.us",
            chat_name="Test",
            messages=[
                {"senderName": "me", "textMessage": "I will send it tomorrow"},
                {"senderName": "Alice", "textMessage": "thanks!"},
            ],
        )

    lines = captured["messages"].split("\n")
    assert lines[0] == "me: I will send it tomorrow"
    assert lines[1] == "Alice: thanks!"


async def test_extract_empty_text_falls_back_to_empty_string():
    from app.commitments import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["messages"] = kwargs["messages"]
        return type("Pred", (), {"commitments": []})()

    with patch.object(extractor_mod.commitment_agent, "acall", side_effect=_fake_acall):
        await extractor_mod.extract_commitments(
            chat_id="chat@c.us",
            chat_name="Test",
            messages=[
                {"senderName": "Alice"},
            ],
        )

    assert captured["messages"] == "Alice: "


# ─── format_existing_commitments ─────────────────────────────────────


def test_format_existing_commitments_none():
    assert format_existing_commitments(None) == "[]"


def test_format_existing_commitments_empty():
    assert format_existing_commitments([]) == "[]"


def test_format_existing_commitments_lists_items():
    commitments = [
        Commitment(
            id="abc-123",
            chat_id="group@c.us",
            committed_party="Alice",
            required_action="Send report",
            deadline="2025-01-10",
            context="Team meeting",
            status="waiting",
        ),
    ]
    result = format_existing_commitments(commitments)
    import json
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "abc-123"
    assert parsed[0]["committed_party"] == "Alice"
    assert parsed[0]["required_action"] == "Send report"
    assert parsed[0]["status"] == "waiting"


# ─── normalize_commitments ────────────────────────────────────────────


def test_normalize_commitments_overwrites_chat_id_and_name():
    commitments = [
        Commitment(
            id=None,
            chat_id="wrong@c.us",
            chat_name="Wrong Name",
            committed_party="Alice",
            required_action="Send report",
            context="Team meeting",
            status="waiting",
        ),
    ]
    result = normalize_commitments(
        commitments=commitments,
        chat_id="correct@c.us",
        chat_name="Correct Name",
    )
    assert result[0].chat_id == "correct@c.us"
    assert result[0].chat_name == "Correct Name"
    assert result[0].committed_party == "Alice"
    assert result[0].required_action == "Send report"


def test_normalize_commitments_preserves_other_fields():
    commitments = [
        Commitment(
            id="abc-123",
            chat_id="old@c.us",
            chat_name="Old",
            committed_party="Bob",
            required_action="Pay invoice",
            deadline="2025-01-06",
            context="Pay the invoice by Monday",
            status="done",
            notification="urgent",
        ),
    ]
    result = normalize_commitments(
        commitments=commitments,
        chat_id="new@c.us",
        chat_name="New",
    )
    assert result[0].id == "abc-123"
    assert result[0].committed_party == "Bob"
    assert result[0].required_action == "Pay invoice"
    assert result[0].deadline == "2025-01-06"
    assert result[0].status == "done"
    assert result[0].notification == "urgent"


def test_normalize_commitments_empty_list():
    result = normalize_commitments(
        commitments=[],
        chat_id="chat@c.us",
        chat_name="Chat",
    )
    assert result == []


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
            deadline="2025-01-07",
            context="I will send the documents by tomorrow",
            status="waiting",
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
        assert items[0].status == CommitmentStatus.WAITING
        assert items[0].source_message_ids == ["msg-1"]


async def test_drain_updates_existing_commitment(test_engine):
    engine, tenant = test_engine

    with Session(engine) as session:
        existing_item = CommitmentItem(
            tenant_id=tenant.id,
            chat_id="972546610653@c.us",
            committed_party="Gaddy",
            required_action="Send the documents",
            deadline="2025-01-07",
            context="I will send the documents by tomorrow",
            status=CommitmentStatus.WAITING,
            notification=NotificationType.DAILY_DIGEST,
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
            deadline="2025-01-07",
            context="I will send the documents by tomorrow",
            status="done",
            notification="none",
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
        assert items[0].status == CommitmentStatus.DONE
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
            deadline="2025-01-06",
            context="Pay the invoice by Monday",
            status=CommitmentStatus.WAITING,
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
            status=CommitmentStatus.DISMISSED,
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


# ─── integration: real DSPy agent call ────────────────────────────────


@pytest.fixture(scope="session")
def _dspy_configured():
    """Configure DSPy once for all integration tests in this session."""
    import os

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")

    from app.commitments.commitments_agent import configure_dspy

    class _FakeSettings:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_api_key = api_key

    configure_dspy(_FakeSettings())


@pytest.mark.integration
async def test_dspy_agent_real_call_outbound_commitment(_dspy_configured):
    """Call the actual DSPy agent with a production outbound message.

    Run with: pytest -m integration -v
    """
    from app.commitments.commitments_agent import CommitmentAgent
    from app.commitments.models import Commitment

    agent = CommitmentAgent()

    pred = await agent.acall(
        chat_id="972507674593@c.us",
        chat_name="מעוז כחלון",
        existing_commitments_json="[]",
        messages="me: הי אחי, מה המצב? \nיש לך במקרה אבוב להשאיל לי ליום שישי?",
    )

    commitments = pred.commitments
    assert isinstance(commitments, list)
    if commitments:
        c = commitments[0]
        assert isinstance(c, Commitment)
        assert c.chat_id == "972507674593@c.us"
        assert c.required_action
        assert c.status in list(CommitmentStatus)


@pytest.mark.integration
async def test_dspy_agent_real_call_volunteer_message(_dspy_configured):
    """Call the actual DSPy agent with a real volunteer-recruitment message."""
    from app.commitments.commitments_agent import CommitmentAgent
    from app.commitments.models import Commitment

    agent = CommitmentAgent()

    text = (
        "מיקה🤍: קודם כל, אני רוצה להגיד תודה רבה לשני ואביבית המדהימות על ההתנדבות "
        "לפעילות הורים מדריכים!!\n"
        "אנחנו נשמח להפוך את זה לקונספט קבוע אבל זה תלוי בכם וברוח וברצון הטוב של כולנו🤩"
    )

    pred = await agent.acall(
        chat_id="120363402798412519@g.us",
        chat_name="דולב הורים תשפו",
        existing_commitments_json="[]",
        messages=text,
    )

    commitments = pred.commitments
    assert isinstance(commitments, list)
    for c in commitments:
        assert isinstance(c, Commitment)
        assert c.chat_id == "120363402798412519@g.us"
        assert c.required_action
        assert c.status in list(CommitmentStatus)
