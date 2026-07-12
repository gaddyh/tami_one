"""Tests for the item extraction agent and DB persistence."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.models import ItemRecord, ReminderItem, ReminderItemStatus


@pytest.fixture()
def test_engine():
    """Create a fresh temp SQLite DB with all tables for each test."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    yield engine

    engine.dispose()
    os.unlink(db_path)


# ─── extract_item ─────────────────────────────────────────────────────


async def test_extract_item_pure_entity():
    from app.items import extractor as extractor_mod

    async def _fake_acall(**kwargs):
        return type("Pred", (), {"subject": "אופניים", "due_at": ""})()

    with patch.object(extractor_mod.item_agent, "acall", side_effect=_fake_acall):
        result = await extractor_mod.extract_item(
            text="אופניים",
            current_time="2025-07-13T10:00:00+03:00",
        )

    assert result["subject"] == "אופניים"
    assert result["due_at"] == ""


async def test_extract_item_with_exact_time():
    from app.items import extractor as extractor_mod

    async def _fake_acall(**kwargs):
        return type("Pred", (), {"subject": "רשיון בינלאומי", "due_at": "2025-07-13T17:00:00"})()

    with patch.object(extractor_mod.item_agent, "acall", side_effect=_fake_acall):
        result = await extractor_mod.extract_item(
            text="רשיון בינלאומי בחמש",
            current_time="2025-07-13T10:00:00+03:00",
        )

    assert result["subject"] == "רשיון בינלאומי"
    assert result["due_at"] == "2025-07-13T17:00:00"


async def test_extract_item_with_relative_time():
    from app.items import extractor as extractor_mod

    async def _fake_acall(**kwargs):
        return type("Pred", (), {"subject": "נעלי ים חגור ק״ש", "due_at": "2025-07-13T11:00:00"})()

    with patch.object(extractor_mod.item_agent, "acall", side_effect=_fake_acall):
        result = await extractor_mod.extract_item(
            text="נעלי ים חגור ק״ש בעוד שעה",
            current_time="2025-07-13T10:00:00+03:00",
        )

    assert result["subject"] == "נעלי ים חגור ק״ש"
    assert result["due_at"] == "2025-07-13T11:00:00"


async def test_extract_item_strips_time_from_subject():
    from app.items import extractor as extractor_mod

    captured: dict = {}

    async def _fake_acall(**kwargs):
        captured["text"] = kwargs["text"]
        return type("Pred", (), {"subject": "עירית", "due_at": "2025-07-14T08:00:00"})()

    with patch.object(extractor_mod.item_agent, "acall", side_effect=_fake_acall):
        result = await extractor_mod.extract_item(
            text="עירית מחר ב8",
            current_time="2025-07-13T10:00:00+03:00",
        )

    assert result["subject"] == "עירית"
    assert "מחר" not in result["subject"]
    assert result["due_at"] == "2025-07-14T08:00:00"


# ─── DB persistence ───────────────────────────────────────────────────


def test_item_record_persists_without_due_at(test_engine):
    with Session(test_engine) as session:
        record = ItemRecord(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="אופניים",
            due_at=None,
            reminder_id=None,
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        assert record.id is not None
        assert record.subject == "אופניים"
        assert record.due_at is None
        assert record.reminder_id is None


def test_item_record_persists_with_due_at_and_reminder(test_engine):
    due_at = datetime(2025, 7, 13, 17, 0, 0)

    with Session(test_engine) as session:
        reminder_id = "r-abc12345"
        reminder = ReminderItem(
            tenant_id="default",
            chat_id="sender@c.us",
            reminder_id=reminder_id,
            what="רשיון בינלאומי",
            due_at=due_at,
            status=ReminderItemStatus.PENDING,
        )
        session.add(reminder)

        record = ItemRecord(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="רשיון בינלאומי",
            due_at=due_at,
            reminder_id=reminder_id,
        )
        session.add(record)
        session.commit()

    with Session(test_engine) as session:
        records = session.exec(select(ItemRecord)).all()
        assert len(records) == 1
        assert records[0].subject == "רשיון בינלאומי"
        assert records[0].due_at == due_at
        assert records[0].reminder_id == "r-abc12345"

        reminders = session.exec(select(ReminderItem)).all()
        assert len(reminders) == 1
        assert reminders[0].what == "רשיון בינלאומי"
        assert reminders[0].status == ReminderItemStatus.PENDING


def test_item_record_reminder_id_nullable(test_engine):
    with Session(test_engine) as session:
        record = ItemRecord(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="נעלי ים",
            due_at=None,
            reminder_id=None,
        )
        session.add(record)
        session.commit()

    with Session(test_engine) as session:
        records = session.exec(select(ItemRecord)).all()
        assert len(records) == 1
        assert records[0].reminder_id is None


# ─── E2E integration: real DSPy agent + real Postgres DB ─────────────


def _make_text_message(msg_id: str, sender: str, text: str) -> dict:
    return {"from": sender, "id": msg_id, "type": "text", "text": text}


@pytest.fixture(autouse=True)
def _clear_seen_ids():
    from app.routers.business_webhook import _seen_message_ids
    _seen_message_ids.clear()
    yield
    _seen_message_ids.clear()


@pytest.fixture(scope="session")
def _dspy_configured():
    """Configure DSPy once for all integration tests in this session."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")

    from app.commitments.commitments_agent import configure_dspy
    from app.config import settings
    configure_dspy(settings)


@pytest.fixture(scope="session")
def _pg_engine():
    """Use the real Postgres engine from DATABASE_URL, create tables once."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        pytest.skip("DATABASE_URL not set — skipping integration tests")

    from app.db.engine import init_db
    init_db()

    from app.db.engine import engine as real_engine
    yield real_engine


@pytest.mark.integration
async def test_e2e_pure_entity_no_time(_dspy_configured, _pg_engine):
    """Input: 'אופניים' → ItemRecord saved, no reminder, reply 'נשמר: אופניים'."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = "972500000001@c.us"
    msg_id = "e2e-int-1"
    msg = _make_text_message(msg_id, sender, "אופניים")

    mock_wa = AsyncMock()
    mock_wa.send_typing_indicator = AsyncMock()

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None)

    # Assert reply
    mock_wa.send_text.assert_awaited_once()
    assert mock_wa.send_text.call_args.kwargs["to"] == sender
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "אופניים" in body

    # Assert DB
    with Session(real_engine) as session:
        records = session.exec(
            select(ItemRecord).where(ItemRecord.chat_id == sender)
        ).all()
        assert len(records) >= 1
        latest = records[-1]
        assert latest.subject == "אופניים"
        assert latest.due_at is None
        assert latest.reminder_id is None

        reminders = session.exec(
            select(ReminderItem).where(ReminderItem.chat_id == sender)
        ).all()
        assert len(reminders) == 0


@pytest.mark.integration
async def test_e2e_entity_with_exact_time(_dspy_configured, _pg_engine):
    """Input: 'רשיון בינלאומי בחמש' → ItemRecord + ReminderItem, reply includes time."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = "972500000002@c.us"
    msg = _make_text_message("e2e-int-2", sender, "רשיון בינלאומי בחמש")

    mock_wa = AsyncMock()
    mock_wa.send_typing_indicator = AsyncMock()

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None)

    # Assert reply
    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "רשיון בינלאומי" in body

    # Assert DB
    with Session(real_engine) as session:
        records = session.exec(
            select(ItemRecord).where(ItemRecord.chat_id == sender)
        ).all()
        assert len(records) >= 1
        latest = records[-1]
        assert latest.subject == "רשיון בינלאומי"
        assert latest.due_at is not None
        assert latest.reminder_id is not None

        reminders = session.exec(
            select(ReminderItem).where(ReminderItem.chat_id == sender)
        ).all()
        assert len(reminders) >= 1
        latest_rem = reminders[-1]
        assert latest_rem.what == "רשיון בינלאומי"
        assert latest_rem.status == ReminderItemStatus.PENDING
        assert latest.reminder_id == latest_rem.reminder_id


@pytest.mark.integration
async def test_e2e_entity_with_relative_time(_dspy_configured, _pg_engine):
    """Input: 'נעלי ים חגור ק״ש בעוד שעה' → ItemRecord + ReminderItem with computed time."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = "972500000003@c.us"
    msg = _make_text_message("e2e-int-3", sender, "נעלי ים חגור ק״ש בעוד שעה")

    mock_wa = AsyncMock()
    mock_wa.send_typing_indicator = AsyncMock()

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None)

    # Assert reply
    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "נעלי ים חגור ק״ש" in body

    # Assert DB
    with Session(real_engine) as session:
        records = session.exec(
            select(ItemRecord).where(ItemRecord.chat_id == sender)
        ).all()
        assert len(records) >= 1
        latest = records[-1]
        assert latest.subject == "נעלי ים חגור ק״ש"
        assert latest.due_at is not None
        assert latest.reminder_id is not None

        reminders = session.exec(
            select(ReminderItem).where(ReminderItem.chat_id == sender)
        ).all()
        assert len(reminders) >= 1
        latest_rem = reminders[-1]
        assert latest_rem.what == "נעלי ים חגור ק״ש"
        assert latest_rem.status == ReminderItemStatus.PENDING
        assert latest.reminder_id == latest_rem.reminder_id


@pytest.mark.integration
async def test_e2e_entity_with_tomorrow_time(_dspy_configured, _pg_engine):
    """Input: 'עירית מחר ב8' → ItemRecord + ReminderItem, subject stripped of time words."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = "972500000004@c.us"
    msg = _make_text_message("e2e-int-4", sender, "עירית מחר ב8")

    mock_wa = AsyncMock()
    mock_wa.send_typing_indicator = AsyncMock()

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None)

    # Assert reply
    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "עירית" in body

    # Assert DB
    with Session(real_engine) as session:
        records = session.exec(
            select(ItemRecord).where(ItemRecord.chat_id == sender)
        ).all()
        assert len(records) >= 1
        latest = records[-1]
        assert latest.subject == "עירית"
        assert latest.due_at is not None
        assert latest.reminder_id is not None

        reminders = session.exec(
            select(ReminderItem).where(ReminderItem.chat_id == sender)
        ).all()
        assert len(reminders) >= 1
        latest_rem = reminders[-1]
        assert latest_rem.what == "עירית"
        assert latest_rem.status == ReminderItemStatus.PENDING
        assert latest.reminder_id == latest_rem.reminder_id


@pytest.mark.integration
async def test_e2e_relative_minutes_no_entity(_dspy_configured, _pg_engine):
    """Input: 'עשר דקות' → subject is the time expression itself, ReminderItem created."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = "972500000005@c.us"
    msg = _make_text_message("e2e-int-5", sender, "עשר דקות")

    mock_wa = AsyncMock()
    mock_wa.send_typing_indicator = AsyncMock()

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None)

    # Assert reply
    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "עשר דקות" in body

    # Assert DB
    with Session(real_engine) as session:
        records = session.exec(
            select(ItemRecord).where(ItemRecord.chat_id == sender)
        ).all()
        assert len(records) >= 1
        latest = records[-1]
        assert latest.subject == "עשר דקות"
        assert latest.due_at is not None
        assert latest.reminder_id is not None

        reminders = session.exec(
            select(ReminderItem).where(ReminderItem.chat_id == sender)
        ).all()
        assert len(reminders) >= 1
        latest_rem = reminders[-1]
        assert latest_rem.what == "עשר דקות"
        assert latest_rem.status == ReminderItemStatus.PENDING
        assert latest.reminder_id == latest_rem.reminder_id
