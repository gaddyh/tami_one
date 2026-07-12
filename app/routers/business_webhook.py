import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from zoneinfo import ZoneInfo
from sqlmodel import Session

from app.config import settings
from app.db.engine import engine
from app.db.models import ItemRecord, ReminderItem, ReminderItemStatus
from app.agents.memory import memory_store
from app.items.extractor import extract_item
from app.services.transcription import handle_360dialog_audio_message, Transcriber
from app.services.whatsapp import (
    Dialog360Client,
    expected_basic_auth_header,
    iter_incoming_messages,
)

logger = logging.getLogger(__name__)

router = APIRouter()
wa = Dialog360Client(settings)
_seen_message_ids: set[str] = set()


def verify_webhook_auth(authorization: str | None) -> None:
    if settings.webhook_auth_mode == "none":
        return

    if settings.webhook_auth_mode == "bearer":
        expected = f"Bearer {settings.webhook_bearer_token}"
        if not settings.webhook_bearer_token or authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return

    if settings.webhook_auth_mode == "basic":
        expected = expected_basic_auth_header(
            settings.webhook_basic_user,
            settings.webhook_basic_pass,
        )
        if (
            not settings.webhook_basic_user
            or not settings.webhook_basic_pass
            or authorization != expected
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return


@router.post("/webhook/360dialog")
async def webhook_360dialog(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    360dialog webhook endpoint.

    Important:
    Return 200 immediately, then process the WhatsApp message in the background.
    """
    verify_webhook_auth(authorization)

    payload = await request.json()
    logger.info("Accepted 360dialog webhook")

    background_tasks.add_task(process_webhook_payload, payload, request.app.state.transcriber)

    return {
        "ok": True,
        "accepted": True,
    }


async def process_webhook_payload(payload: dict[str, Any], transcriber: Transcriber) -> None:
    """
    Background processing.

    This runs after the HTTP 200 response has already been returned to 360dialog.
    """
    try:
        messages = list(iter_incoming_messages(payload))

        if not messages:
            logger.info("Webhook had no incoming messages to handle")
            return

        for message in messages:
            await process_single_message(message, transcriber)

    except Exception:
        logger.exception("Failed processing webhook payload")


async def process_single_message(message: dict[str, Any], transcriber: Transcriber) -> None:
    sender = message["from"]
    message_id = message.get("id", "")
    msg_type = message.get("type", "")

    if message_id and message_id in _seen_message_ids:
        logger.info("Skipping duplicate message_id=%s", message_id)
        return
    if message_id:
        _seen_message_ids.add(message_id)

    try:
        if msg_type == "text":
            user_msg = message.get("text", "")

            await wa.send_typing_indicator(message_id)

        elif msg_type == "audio":
            media_id = message.get("media_id", "")
            mime_type = message.get("mime_type", "")

            if not media_id:
                raise ValueError("Missing audio media id")

            await wa.send_typing_indicator(message_id)

            user_msg = await handle_360dialog_audio_message(
                wa=wa,
                transcriber=transcriber,
                media_id=media_id,
                mime_type=mime_type,
            )

        else:
            logger.info("Ignoring unsupported message type=%s", msg_type)
            return

        logger.info(
            "Processing message_id=%s from=%s type=%s",
            message_id,
            sender,
            msg_type,
        )

        # Current time in user-local timezone (for LLM to resolve relative dates correctly)
        tz = ZoneInfo(settings.tenant_timezone)
        current_time_local = datetime.now(tz)

        # Extract subject and optional due_at via DSPy agent
        result = await extract_item(
            text=user_msg,
            current_time=current_time_local.isoformat(),
        )

        subject = result["subject"]
        due_at_str = result["due_at"]

        # Parse due_at if present
        due_at_utc: datetime | None = None
        if due_at_str:
            try:
                due_at_dt = datetime.fromisoformat(due_at_str)
                if due_at_dt.tzinfo is not None:
                    due_at_utc = due_at_dt.astimezone(timezone.utc).replace(tzinfo=None)
                else:
                    due_at_utc = due_at_dt
            except (ValueError, TypeError):
                logger.warning("Failed to parse due_at=%r", due_at_str)

        reply_text = ""

        with Session(engine) as session:
            # Persist ItemRecord
            reminder_id: str | None = None

            if due_at_utc is not None:
                reminder_id = f"r-{uuid.uuid4().hex[:8]}"
                reminder_item = ReminderItem(
                    tenant_id="default",
                    chat_id=sender,
                    reminder_id=reminder_id,
                    what=subject,
                    due_at=due_at_utc,
                    status=ReminderItemStatus.PENDING,
                )
                session.add(reminder_item)

            record = ItemRecord(
                tenant_id="default",
                chat_id=sender,
                subject=subject,
                due_at=due_at_utc,
                reminder_id=reminder_id,
            )
            session.add(record)
            session.commit()

        if due_at_utc is not None:
            reply_text = f"נשמר: {subject} — תזכיר לך ב-{due_at_str}"
        else:
            reply_text = f"נשמר: {subject}"

        # Send reply
        if reply_text:
            await wa.send_text(to=sender, body=reply_text)

        # Update memory_store with both user message and agent reply
        if sender:
            memory_store.append(sender, "user", user_msg)
            if reply_text:
                memory_store.append(sender, "assistant", reply_text)

    except Exception:
        logger.exception(
            "Failed handling message_id=%s type=%s",
            message_id,
            msg_type,
        )

        try:
            await wa.send_text(
                to=sender,
                body="Sorry, I couldn't process that message.",
            )
        except Exception:
            logger.exception("Failed sending error message to user")
