"""End-to-end tests: webhook payload → persist → thread → drain → extract → DB.

Simulates the full pipeline with a mocked LLM (no API calls) to verify
that a message flows from ingestion through to a persisted CommitmentItem
with correct conversation linkage.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.commitments.models import Commitment, CommitmentStatus, NotificationType
from app.db.cache import (
    accounts_by_instance,
    contacts_by_tenant_chat_id,
    message_buffer,
)
from app.db.models import (
    ChatMessage,
    CommitmentItem,
    Conversation,
    Tenant,
    TenantKind,
    WhatsAppAccount,
    WhatsAppProvider,
)
from app.db.upsert import upsert_contact_and_chat
from app.commitments.processor import drain_and_process
from app.routers.green_api import MessageDirection, MessageEvent


@pytest.fixture
async def e2e_env():
    """Fresh temp DB with tenant + account, caches primed, buffer empty."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)

    tenant_id = None
    with Session(engine) as session:
        tenant = Tenant(name="E2E Tenant", kind=TenantKind.SOLO)
        session.add(tenant)
        session.commit()
        session.refresh(tenant)
        tenant_id = tenant.id

        account = WhatsAppAccount(
            tenant_id=tenant_id,
            provider=WhatsAppProvider.GREEN_API,
            provider_instance_id="7700673764",
            chat_id=None,
        )
        session.add(account)
        session.commit()
        session.refresh(account)

    # Prime caches
    accounts_by_instance.clear()
    accounts_by_instance["7700673764"] = account
    contacts_by_tenant_chat_id.clear()
    await message_buffer.drain()

    yield engine, tenant_id

    accounts_by_instance.clear()
    contacts_by_tenant_chat_id.clear()
    await message_buffer.drain()
    engine.dispose()
    os.unlink(db_path)


def _make_event(
    tenant_id: str,
    chat_id: str = "972500000001@c.us",
    chat_name: str = "Boris",
    text: str = "hello",
    provider_message_id: str = "msg-1",
    direction: MessageDirection = MessageDirection.INBOUND,
    sender: str | None = "972500000001@c.us",
    sender_name: str | None = "Boris",
    sent_at: datetime | None = None,
    quoted_message_id: str | None = None,
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
        message_time=sent_at or datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        text=text,
        raw_type_webhook="incomingMessageReceived",
        quoted_message_id=quoted_message_id,
    )


@pytest.mark.asyncio
async def test_e2e_single_commitment_persisted(e2e_env):
    """Full flow: inbound message → persist → thread → drain → extract → CommitmentItem in DB."""
    engine, tenant_id = e2e_env

    # 1. Simulate webhook: normalize + upsert (persist + thread + buffer)
    event = _make_event(
        tenant_id,
        text="I'll send the documents by Friday",
        provider_message_id="e2e-msg-1",
    )
    with patch("app.db.upsert.engine", engine):
        result = await upsert_contact_and_chat(event)

    assert result["ok"] is True
    assert result.get("duplicate") is not True
    assert result.get("conversation_id") is not None

    conv_id = result["conversation_id"]

    # 2. Verify ChatMessage was persisted
    with Session(engine) as session:
        msgs = list(session.exec(select(ChatMessage)))
        assert len(msgs) == 1
        assert msgs[0].provider_message_id == "e2e-msg-1"
        assert msgs[0].conversation_id == conv_id
        assert msgs[0].processed_at is None  # not yet drained

    # 3. Verify Conversation was created
    with Session(engine) as session:
        conv = session.get(Conversation, conv_id)
        assert conv is not None
        assert conv.chat_id == "972500000001@c.us"
        assert conv.summary == ""  # empty until drain updates it

    # 4. Mock the extractor to return a commitment, then drain
    mock_commitment = Commitment(
        id=None,
        chat_id="972500000001@c.us",
        chat_name="Boris",
        committed_party="Boris",
        required_action="Send the documents",
        deadline="2025-01-03",
        context="I'll send the documents by Friday",
        status=CommitmentStatus.WAITING,
        notification=NotificationType.NONE,
    )

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=[mock_commitment],
        ),
        patch(
            "app.commitments.processor.summarize_conversation",
            new_callable=AsyncMock,
            return_value="Boris will send documents by Friday",
        ),
    ):
        results = await drain_and_process()

    # 5. Verify commitment was persisted to DB
    key = (tenant_id, "972500000001@c.us")
    assert key in results
    assert len(results[key]) == 1

    with Session(engine) as session:
        items = list(session.exec(select(CommitmentItem)))
        assert len(items) == 1
        item = items[0]
        assert item.required_action == "Send the documents"
        assert item.committed_party == "Boris"
        assert item.status == CommitmentStatus.WAITING
        assert item.conversation_id == conv_id
        assert item.tenant_id == tenant_id

    # 6. Verify ChatMessage.processed_at was set
    with Session(engine) as session:
        msg = session.exec(select(ChatMessage)).first()
        assert msg.processed_at is not None

    # 7. Verify Conversation.summary was updated
    with Session(engine) as session:
        conv = session.get(Conversation, conv_id)
        assert conv.summary == "Boris will send documents by Friday"


@pytest.mark.asyncio
async def test_e2e_duplicate_webhook_no_double_extract(e2e_env):
    """Duplicate webhook delivery (same provider_message_id) is deduped — no double buffer."""
    engine, tenant_id = e2e_env

    event = _make_event(
        tenant_id,
        text="I'll call the supplier",
        provider_message_id="e2e-dup-1",
    )

    # First delivery
    with patch("app.db.upsert.engine", engine):
        result1 = await upsert_contact_and_chat(event)
    assert result1["ok"] is True
    assert result1.get("duplicate") is not True

    # Second delivery (same provider_message_id)
    with patch("app.db.upsert.engine", engine):
        result2 = await upsert_contact_and_chat(event)
    assert result2["ok"] is True
    assert result2.get("duplicate") is True

    # Buffer should have exactly 1 message
    drained = await message_buffer.drain()
    key = (tenant_id, "972500000001@c.us")
    assert key in drained
    assert len(drained[key]) == 1


@pytest.mark.asyncio
async def test_e2e_cross_conversation_commitment_update(e2e_env):
    """Commitment made in conv A, completed in conv B — chat-scoped lifecycle works."""
    engine, tenant_id = e2e_env

    t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 3, 14, 0, 0, tzinfo=timezone.utc)  # 2 days later → new conversation

    # Message 1: Boris promises to send documents
    event1 = _make_event(
        tenant_id,
        text="I'll send the documents by Friday",
        provider_message_id="e2e-cross-1",
        sent_at=t0,
    )
    with patch("app.db.upsert.engine", engine):
        result1 = await upsert_contact_and_chat(event1)
    conv_a = result1["conversation_id"]

    # Drain with mock extractor — creates the commitment
    mock_commitment = Commitment(
        id=None,
        chat_id="972500000001@c.us",
        chat_name="Boris",
        committed_party="Boris",
        required_action="Send the documents",
        deadline="2025-01-03",
        context="I'll send the documents by Friday",
        status=CommitmentStatus.WAITING,
    )

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=[mock_commitment],
        ),
        patch(
            "app.commitments.processor.summarize_conversation",
            new_callable=AsyncMock,
            return_value="Boris will send documents",
        ),
    ):
        await drain_and_process()

    # Get the commitment ID
    with Session(engine) as session:
        items = list(session.exec(select(CommitmentItem)))
        assert len(items) == 1
        commitment_id = items[0].id
        assert items[0].conversation_id == conv_a

    # Message 2: 2 days later, Boris says "I sent the documents" → new conversation
    event2 = _make_event(
        tenant_id,
        text="I sent the documents yesterday",
        provider_message_id="e2e-cross-2",
        sent_at=t1,
    )
    with patch("app.db.upsert.engine", engine):
        result2 = await upsert_contact_and_chat(event2)
    conv_b = result2["conversation_id"]

    # Must be a different conversation (gap > 45 min)
    assert conv_a != conv_b

    # Drain — extractor should see the existing commitment (chat-scoped) and mark it done
    mock_update = Commitment(
        id=commitment_id,
        chat_id="972500000001@c.us",
        chat_name="Boris",
        committed_party="Boris",
        required_action="Send the documents",
        deadline="2025-01-03",
        context="I sent the documents yesterday",
        status=CommitmentStatus.DONE,
    )

    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            return_value=[mock_update],
        ),
        patch(
            "app.commitments.processor.summarize_conversation",
            new_callable=AsyncMock,
            return_value="Boris confirmed documents sent",
        ),
    ):
        await drain_and_process()

    # Verify: single commitment, now DONE, conversation_id still = conv_a (origin)
    with Session(engine) as session:
        items = list(session.exec(select(CommitmentItem)))
        assert len(items) == 1  # no duplicate
        assert items[0].status == CommitmentStatus.DONE
        assert items[0].conversation_id == conv_a  # origin conversation


@pytest.mark.asyncio
async def test_e2e_quoted_reply_same_conversation(e2e_env):
    """Quoted reply to an active conversation inherits its conversation_id."""
    engine, tenant_id = e2e_env

    t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)

    # First message
    event1 = _make_event(
        tenant_id,
        text="Can you send the report?",
        provider_message_id="e2e-quote-1",
        sent_at=t0,
    )
    with patch("app.db.upsert.engine", engine):
        result1 = await upsert_contact_and_chat(event1)
    conv_id = result1["conversation_id"]

    # Reply quoting the first message
    event2 = _make_event(
        tenant_id,
        text="Sure, I'll send it",
        provider_message_id="e2e-quote-2",
        sent_at=t1,
        quoted_message_id="e2e-quote-1",
    )
    with patch("app.db.upsert.engine", engine):
        result2 = await upsert_contact_and_chat(event2)

    # Should inherit the same conversation
    assert result2["conversation_id"] == conv_id

    # Both messages in same conversation
    with Session(engine) as session:
        msgs = list(session.exec(select(ChatMessage).order_by(ChatMessage.sent_at)))
        assert len(msgs) == 2
        assert msgs[0].conversation_id == conv_id
        assert msgs[1].conversation_id == conv_id
        assert msgs[1].quoted_message_id == "e2e-quote-1"


@pytest.mark.asyncio
async def test_e2e_failure_requeues_and_no_summary_update(e2e_env):
    """Failed extraction re-enqueues messages and does NOT update the summary."""
    engine, tenant_id = e2e_env

    event = _make_event(
        tenant_id,
        text="I'll do something",
        provider_message_id="e2e-fail-1",
    )
    with patch("app.db.upsert.engine", engine):
        result = await upsert_contact_and_chat(event)
    conv_id = result["conversation_id"]

    # Drain with extractor that raises
    with (
        patch("app.commitments.processor.engine", engine),
        patch(
            "app.commitments.processor.extract_commitments",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM is down"),
        ),
    ):
        results = await drain_and_process()

    # No commitments persisted
    key = (tenant_id, "972500000001@c.us")
    assert key not in results

    # Summary NOT updated (still empty)
    with Session(engine) as session:
        conv = session.get(Conversation, conv_id)
        assert conv.summary == ""

    # ChatMessage.processed_at still NULL
    with Session(engine) as session:
        msg = session.exec(select(ChatMessage)).first()
        assert msg.processed_at is None
        assert msg.extraction_attempts == 1

    # Message was re-enqueued — drain again should find it
    drained = await message_buffer.drain()
    assert key in drained
    assert len(drained[key]) == 1
