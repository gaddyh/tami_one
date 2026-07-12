"""Tests for the task review agent, session management, and DB models."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db.models import (
    ItemRecord,
    ItemReviewStatus,
    Task,
    TaskStatus,
)
from app.tasks.session import DueDateResolution, ReviewSession, ReviewSessionStore


# ─── Fixtures ─────────────────────────────────────────────────────────


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


@pytest.fixture()
def mock_wa():
    return AsyncMock()


def _make_item(
    engine,
    tenant_id: str = "default",
    chat_id: str = "sender@c.us",
    subject: str = "אופניים",
    due_at: datetime | None = None,
) -> ItemRecord:
    with Session(engine) as session:
        record = ItemRecord(
            tenant_id=tenant_id,
            chat_id=chat_id,
            subject=subject,
            due_at=due_at,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record


def _make_settings():
    from app.config import Settings
    import dataclasses

    fields = {f.name: getattr(__import__("app.config", fromlist=["settings"]).settings, f.name)
              for f in dataclasses.fields(Settings)}
    return Settings(**fields)


# ─── Session tests ────────────────────────────────────────────────────


def test_session_initial_state():
    session = ReviewSession("default", "chat1", ["item-a", "item-b"])

    assert session.tenant_id == "default"
    assert session.chat_id == "chat1"
    assert session.current_index == 0
    assert session.current_item_id == "item-a"
    assert session.candidate_subject == ""
    assert session.candidate_due_at == ""
    assert session.due_date_resolution == DueDateResolution.UNKNOWN
    assert session.clarification_count == 0
    assert session.created_count == 0
    assert session.remaining_count == 2


def test_session_advance_resets_candidate_state():
    session = ReviewSession("default", "chat1", ["item-a", "item-b"])
    session.candidate_subject = "some task"
    session.candidate_due_at = "2025-07-13T17:00:00"
    session.due_date_resolution = DueDateResolution.PROVIDED
    session.clarification_count = 2

    next_id = session.advance()

    assert next_id == "item-b"
    assert session.current_index == 1
    assert session.candidate_subject == ""
    assert session.candidate_due_at == ""
    assert session.due_date_resolution == DueDateResolution.UNKNOWN
    assert session.clarification_count == 0


def test_session_advance_past_end_returns_none():
    session = ReviewSession("default", "chat1", ["item-a"])
    assert session.current_item_id == "item-a"

    result = session.advance()

    assert result is None
    assert session.current_item_id is None
    assert session.remaining_count == 0


def test_session_store_tuple_keys():
    store = ReviewSessionStore()

    session = store.start("tenant1", "chat1", ["item-a"])
    assert store.get("tenant1", "chat1") is session
    assert store.get("tenant1", "chat2") is None
    assert store.get("tenant2", "chat1") is None

    ended = store.end("tenant1", "chat1")
    assert ended is session
    assert store.get("tenant1", "chat1") is None


def test_session_store_end_nonexistent_returns_none():
    store = ReviewSessionStore()
    assert store.end("nope", "nope") is None


# ─── Model tests ──────────────────────────────────────────────────────


def test_task_persists_with_source_item(test_engine):
    item = _make_item(test_engine, subject="אופניים")

    with Session(test_engine) as session:
        task = Task(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="לקחת אופניים לתיקון",
            due_at=datetime(2025, 7, 14, 10, 0, 0),
            source_item_id=item.id,
            status=TaskStatus.ACTIVE,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.id is not None
        assert task.source_item_id == item.id
        assert task.status == TaskStatus.ACTIVE


def test_task_persists_without_source_item(test_engine):
    with Session(test_engine) as session:
        task = Task(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="manual task",
            due_at=None,
            source_item_id=None,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        assert task.id is not None
        assert task.source_item_id is None
        assert task.due_at is None


def test_task_unique_constraint_on_source_item(test_engine):
    item = _make_item(test_engine, subject="אופניים")

    with Session(test_engine) as session:
        task1 = Task(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="first task",
            source_item_id=item.id,
        )
        session.add(task1)
        session.commit()

    with Session(test_engine) as session:
        task2 = Task(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="duplicate task",
            source_item_id=item.id,
        )
        session.add(task2)
        with pytest.raises(Exception):
            session.commit()


def test_item_record_default_review_status_is_pending(test_engine):
    item = _make_item(test_engine, subject="test subject")

    with Session(test_engine) as session:
        record = session.get(ItemRecord, item.id)
        assert record.review_status == ItemReviewStatus.PENDING
        assert record.converted_at is None


def test_item_record_can_be_marked_converted(test_engine):
    item = _make_item(test_engine, subject="test subject")

    with Session(test_engine) as session:
        record = session.get(ItemRecord, item.id)
        record.review_status = ItemReviewStatus.CONVERTED
        record.converted_at = datetime.now(timezone.utc)
        session.add(record)
        session.commit()

    with Session(test_engine) as session:
        record = session.get(ItemRecord, item.id)
        assert record.review_status == ItemReviewStatus.CONVERTED
        assert record.converted_at is not None


# ─── Processor: start_review ──────────────────────────────────────────


async def test_start_review_no_items(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)

    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "אין פריטים" in body
    assert store.get("default", "sender@c.us") is None


async def test_start_review_presents_first_item(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")
    _make_item(test_engine, subject="רשיון", due_at=datetime(2025, 7, 14, 10, 0, 0))

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)

    session = store.get("default", "sender@c.us")
    assert session is not None
    assert len(session.item_ids) == 2

    mock_wa.send_text.assert_awaited_once()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "אופניים" in body


async def test_start_review_undated_items_first(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="dated1", due_at=datetime(2025, 7, 14, 10, 0, 0))
    _make_item(test_engine, subject="undated1")
    _make_item(test_engine, subject="dated2", due_at=datetime(2025, 7, 13, 10, 0, 0))
    _make_item(test_engine, subject="undated2")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)

    session = store.get("default", "sender@c.us")
    items = session.item_ids

    with Session(test_engine) as db:
        subjects = [db.get(ItemRecord, iid).subject for iid in items]

    assert subjects[0] == "undated1"
    assert subjects[1] == "undated2"
    assert subjects[2] == "dated2"  # nearer due_at first
    assert subjects[3] == "dated1"


async def test_start_review_already_active(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        # Second call while session is active
        await service.start_review("default", "sender@c.us", mock_wa)

    assert mock_wa.send_text.await_count == 2
    second_body = mock_wa.send_text.call_args.kwargs["body"]
    assert "כבר בסשן" in second_body


# ─── Processor: handle_reply — successful conversion ──────────────────


async def test_handle_reply_creates_task_and_converts_item(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    item = _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "לקחת אופניים לתיקון",
            "updated_due_at": "2025-07-14T10:00:00",
            "due_date_resolution": "provided",
            "needs_clarification": "false",
            "clarification_question": "",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "לקחת לתיקון מחר ב10",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    with Session(test_engine) as session:
        tasks = session.exec(select(Task)).all()
        assert len(tasks) == 1
        assert tasks[0].subject == "לקחת אופניים לתיקון"
        assert tasks[0].source_item_id == item.id
        assert tasks[0].status == TaskStatus.ACTIVE

        record = session.get(ItemRecord, item.id)
        assert record.review_status == ItemReviewStatus.CONVERTED
        assert record.converted_at is not None

    sess = store.get("default", "sender@c.us")
    assert sess is None  # session ended, no more items


async def test_handle_reply_completes_session_summary(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "לקחת אופניים לתיקון",
            "updated_due_at": "",
            "due_date_resolution": "intentionally_absent",
            "needs_clarification": "false",
            "clarification_question": "",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "לקחת לתיקון, אין תאריך",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "סיימנו" in body
    assert "1" in body  # created 1 task


# ─── Processor: handle_reply — clarification ──────────────────────────


async def test_handle_reply_needs_clarification_stays_on_item(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "",
            "updated_due_at": "",
            "due_date_resolution": "unknown",
            "needs_clarification": "true",
            "clarification_question": "מה תרצה לעשות עם האופניים?",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "לא בטוח",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "מה תרצה" in body

    session = store.get("default", "sender@c.us")
    assert session is not None
    assert session.current_index == 0  # still on first item
    assert session.clarification_count == 1


async def test_handle_reply_clarification_limit_leaves_item_pending(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        clarification_pred = type("Pred", (), {
            "updated_subject": "",
            "updated_due_at": "",
            "due_date_resolution": "unknown",
            "needs_clarification": "true",
            "clarification_question": "מה תרצה?",
        })()
        agent.aforward.return_value = clarification_pred

        for _ in range(settings.review_clarification_limit):
            await service.handle_reply(
                "default", "sender@c.us", "לא בטוח",
                mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
            )

    # After limit, should have sent the "left for later" message and ended session
    bodies = [call.kwargs["body"] for call in mock_wa.send_text.call_args_list]
    assert any("לא הצלחתי" in b for b in bodies)

    session = store.get("default", "sender@c.us")
    assert session is None  # session ended

    # Item should still be pending
    with Session(test_engine) as db:
        record = db.exec(select(ItemRecord).where(ItemRecord.subject == "אופניים")).first()
        assert record.review_status == ItemReviewStatus.PENDING


# ─── Processor: handle_reply — invalid LLM output ─────────────────────


async def test_handle_reply_invalid_due_at_treated_as_clarification(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "לקחת אופניים לתיקון",
            "updated_due_at": "not-a-date",
            "due_date_resolution": "provided",
            "needs_clarification": "false",
            "clarification_question": "",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "לתקן מחר",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    session = store.get("default", "sender@c.us")
    assert session is not None
    assert session.clarification_count == 1
    body = mock_wa.send_text.call_args.kwargs["body"]
    # Should send a clarification or fallback
    assert len(body) > 0


async def test_handle_reply_empty_subject_treated_as_clarification(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "",
            "updated_due_at": "",
            "due_date_resolution": "intentionally_absent",
            "needs_clarification": "false",
            "clarification_question": "",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "משהו",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    session = store.get("default", "sender@c.us")
    assert session is not None
    assert session.clarification_count == 1


async def test_handle_reply_llm_exception_increments_clarification(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.side_effect = RuntimeError("LLM failed")

        await service.handle_reply(
            "default", "sender@c.us", "משהו",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    session = store.get("default", "sender@c.us")
    assert session is not None
    assert session.clarification_count == 1


# ─── Processor: end_review ────────────────────────────────────────────


async def test_end_review_sends_summary_with_remaining(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="item1")
    _make_item(test_engine, subject="item2")
    _make_item(test_engine, subject="item3")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        await service.end_review("default", "sender@c.us", mock_wa)

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "עצרתי" in body
    assert "3" in body  # remaining items in session
    assert store.get("default", "sender@c.us") is None


async def test_end_review_nonexistent_session_is_noop(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    await service.end_review("nope", "nope", mock_wa)
    mock_wa.send_text.assert_not_awaited()


# ─── Processor: idempotency ───────────────────────────────────────────


async def test_idempotent_replay_does_not_increment_count(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    item = _make_item(test_engine, subject="אופניים")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    # Pre-create a Task for this item (simulating a previous run)
    with Session(test_engine) as session:
        existing_task = Task(
            tenant_id="default",
            chat_id="sender@c.us",
            subject="already created",
            source_item_id=item.id,
            status=TaskStatus.ACTIVE,
        )
        session.add(existing_task)
        session.commit()

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        agent.aforward.return_value = type("Pred", (), {
            "updated_subject": "לקחת אופניים לתיקון",
            "updated_due_at": "",
            "due_date_resolution": "intentionally_absent",
            "needs_clarification": "false",
            "clarification_question": "",
        })()

        await service.handle_reply(
            "default", "sender@c.us", "לקחת לתיקון",
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    # Should not have incremented created_count (Task already existed)
    # Session should have ended (no more items)
    assert store.get("default", "sender@c.us") is None

    # Only one Task in DB
    with Session(test_engine) as session:
        tasks = session.exec(select(Task).where(Task.source_item_id == item.id)).all()
        assert len(tasks) == 1

    # Item should be marked converted
    with Session(test_engine) as session:
        record = session.get(ItemRecord, item.id)
        assert record.review_status == ItemReviewStatus.CONVERTED


# ─── Processor: stale item handling ───────────────────────────────────


async def test_stale_item_already_converted_skipped(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    item1 = _make_item(test_engine, subject="item1")
    item2 = _make_item(test_engine, subject="item2")

    # Mark item1 as converted before session starts processing it
    with Session(test_engine) as session:
        record = session.get(ItemRecord, item1.id)
        record.review_status = ItemReviewStatus.CONVERTED
        record.converted_at = datetime.now(timezone.utc)
        session.add(record)
        session.commit()

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)

    # Should have skipped item1 and presented item2
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "item2" in body

    session = store.get("default", "sender@c.us")
    assert session.current_item_id == item2.id


# ─── Processor: end keyword ───────────────────────────────────────────


async def test_end_keyword_ends_session(test_engine, mock_wa):
    from app.tasks import processor as proc_mod

    _make_item(test_engine, subject="item1")
    _make_item(test_engine, subject="item2")

    settings = _make_settings()
    agent = AsyncMock()
    store = ReviewSessionStore()
    service = proc_mod.TaskReviewService(agent=agent, session_store=store, settings=settings)

    with patch.object(proc_mod, "engine", test_engine):
        await service.start_review("default", "sender@c.us", mock_wa)
        mock_wa.reset_mock()

        await service.handle_reply(
            "default", "sender@c.us", settings.review_end_keyword,
            mock_wa, datetime(2025, 7, 13, 12, 0, 0, tzinfo=timezone.utc),
        )

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "עצרתי" in body
    assert store.get("default", "sender@c.us") is None


# ─── E2E integration: real DSPy + real Postgres ───────────────────────


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


def _build_real_service():
    """Build a TaskReviewService with a real TaskReviewAgent (real DSPy)."""
    from app.tasks.processor import TaskReviewService
    from app.tasks.session import ReviewSessionStore
    from app.tasks.task_agent import TaskReviewAgent

    settings = _make_settings()
    store = ReviewSessionStore()
    agent = TaskReviewAgent()
    return TaskReviewService(agent=agent, session_store=store, settings=settings), store


def _seed_item(engine, sender: str, subject: str, due_at: datetime | None = None) -> str:
    """Insert a pending ItemRecord into the real DB and return its id."""
    with Session(engine) as session:
        record = ItemRecord(
            tenant_id="default",
            chat_id=sender,
            subject=subject,
            due_at=due_at,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id


def _cleanup_items(engine, sender: str) -> None:
    """Remove ItemRecords and Tasks for a sender to keep tests isolated."""
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM task WHERE chat_id = :chat_id"), {"chat_id": sender})
        conn.execute(text("DELETE FROM itemrecord WHERE chat_id = :chat_id"), {"chat_id": sender})
        conn.commit()


@pytest.fixture()
def _cleanup_after(request):
    """Yield the sender phone, then clean up test data after the test (even on failure)."""
    sender = request.param
    yield sender
    from app.db.engine import engine as real_engine
    _cleanup_items(real_engine, sender)


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000100@c.us"], indirect=True)
async def test_e2e_trigger_starts_session(_dspy_configured, _pg_engine, _cleanup_after):
    """User sends 'בוא' → session starts, first item presented with deterministic template."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after
    _seed_item(real_engine, sender, "אופניים")
    _seed_item(real_engine, sender, "רשיון")

    service, store = _build_real_service()
    mock_wa = AsyncMock()
    msg = _make_text_message("e2e-review-1", sender, "בוא")

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None, task_review_service=service)

    mock_wa.send_text.assert_awaited()
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "אופניים" in body
    assert store.get("default", sender) is not None

    store.end("default", sender)


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000101@c.us"], indirect=True)
async def test_e2e_trigger_no_items(_dspy_configured, _pg_engine, _cleanup_after):
    """User sends 'בוא' with no pending items → 'no items' message."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after

    service, store = _build_real_service()
    mock_wa = AsyncMock()
    msg = _make_text_message("e2e-review-2", sender, "בוא")

    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg, transcriber=None, task_review_service=service)

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "אין פריטים" in body
    assert store.get("default", sender) is None


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000102@c.us"], indirect=True)
async def test_e2e_full_session_create_task(_dspy_configured, _pg_engine, _cleanup_after):
    """Full flow: trigger → present item → user reply → Task created, item converted, summary."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after
    item_id = _seed_item(real_engine, sender, "אופניים")

    service, store = _build_real_service()
    mock_wa = AsyncMock()

    # Step 1: trigger
    msg1 = _make_text_message("e2e-review-3a", sender, "בוא")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg1, transcriber=None, task_review_service=service)

    assert store.get("default", sender) is not None
    mock_wa.reset_mock()

    # Step 2: user reply with action + due time
    msg2 = _make_text_message("e2e-review-3b", sender, "לקחת לתיקון מחר ב10")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg2, transcriber=None, task_review_service=service)

    # Task created in DB
    with Session(real_engine) as session:
        tasks = session.exec(select(Task).where(Task.source_item_id == item_id)).all()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.ACTIVE

        record = session.get(ItemRecord, item_id)
        assert record.review_status == ItemReviewStatus.CONVERTED

    # Session ended (no more items)
    assert store.get("default", sender) is None

    # Completion summary sent
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "סיימנו" in body


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000103@c.us"], indirect=True)
async def test_e2e_end_keyword_via_webhook(_dspy_configured, _pg_engine, _cleanup_after):
    """User sends 'בטל' during session → session ends with summary."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after
    _seed_item(real_engine, sender, "item1")
    _seed_item(real_engine, sender, "item2")

    service, store = _build_real_service()
    mock_wa = AsyncMock()

    # Start session
    msg1 = _make_text_message("e2e-review-4a", sender, "בוא")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg1, transcriber=None, task_review_service=service)
    mock_wa.reset_mock()

    # End session
    msg2 = _make_text_message("e2e-review-4b", sender, "בטל")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg2, transcriber=None, task_review_service=service)

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "עצרתי" in body
    assert store.get("default", sender) is None


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000104@c.us"], indirect=True)
async def test_e2e_multiple_items_sequential(_dspy_configured, _pg_engine, _cleanup_after):
    """Two items: first converted, second converted, session ends with summary."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after
    item1_id = _seed_item(real_engine, sender, "אופניים")
    item2_id = _seed_item(real_engine, sender, "רשיון")

    service, store = _build_real_service()
    mock_wa = AsyncMock()

    # Start
    msg1 = _make_text_message("e2e-review-5a", sender, "בוא")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg1, transcriber=None, task_review_service=service)
    mock_wa.reset_mock()

    # Reply to first item
    msg2 = _make_text_message("e2e-review-5b", sender, "לקחת לתיקון, אין תאריך")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg2, transcriber=None, task_review_service=service)

    # First item converted, second presented
    with Session(real_engine) as session:
        record1 = session.get(ItemRecord, item1_id)
        assert record1.review_status == ItemReviewStatus.CONVERTED

    sess = store.get("default", sender)
    assert sess is not None
    assert sess.current_item_id == item2_id
    assert sess.created_count == 1
    mock_wa.reset_mock()

    # Reply to second item
    msg3 = _make_text_message("e2e-review-5c", sender, "לחדש רשיון ב20 ביולי ב9 בבוקר")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg3, transcriber=None, task_review_service=service)

    # Both items converted, two tasks, session ended
    with Session(real_engine) as session:
        tasks = session.exec(select(Task).where(Task.chat_id == sender)).all()
        assert len(tasks) == 2
        assert all(t.status == TaskStatus.ACTIVE for t in tasks)

        record2 = session.get(ItemRecord, item2_id)
        assert record2.review_status == ItemReviewStatus.CONVERTED

    assert store.get("default", sender) is None
    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "סיימנו" in body


@pytest.mark.integration
@pytest.mark.parametrize("_cleanup_after", ["972500000105@c.us"], indirect=True)
async def test_e2e_trigger_while_active_reminds_user(_dspy_configured, _pg_engine, _cleanup_after):
    """User sends 'בוא' while session already active → reminded, no new session."""
    from app.routers import business_webhook as webhook
    from app.db.engine import engine as real_engine

    sender = _cleanup_after
    _seed_item(real_engine, sender, "item1")
    _seed_item(real_engine, sender, "item2")

    service, store = _build_real_service()
    mock_wa = AsyncMock()

    # Start session
    msg1 = _make_text_message("e2e-review-6a", sender, "בוא")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg1, transcriber=None, task_review_service=service)

    session = store.get("default", sender)
    assert session is not None
    original_index = session.current_index
    mock_wa.reset_mock()

    # Send 'בוא' again
    msg2 = _make_text_message("e2e-review-6b", sender, "בוא")
    with (
        patch.object(webhook, "wa", mock_wa),
        patch.object(webhook, "engine", real_engine),
        patch("app.tasks.processor.engine", real_engine),
    ):
        await webhook.process_single_message(msg2, transcriber=None, task_review_service=service)

    body = mock_wa.send_text.call_args.kwargs["body"]
    assert "כבר בסשן" in body

    # Same session, no change
    session = store.get("default", sender)
    assert session is not None
    assert session.current_index == original_index

    store.end("default", sender)
