import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from zoneinfo import ZoneInfo
from sqlmodel import Session

from app.agents.core import run_agent
from app.agents.schemas import ConversationTurn, ReminderInput, ReminderStatus
from app.config import settings
from app.db.engine import engine
from app.db.models import ReminderItem, ReminderItemStatus
from app.agents.memory import memory_store
from app.services.transcription import handle_360dialog_audio_message
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

    background_tasks.add_task(process_webhook_payload, payload)

    return {
        "ok": True,
        "accepted": True,
    }


async def process_webhook_payload(payload: dict[str, Any]) -> None:
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
            await process_single_message(message)

    except Exception:
        logger.exception("Failed processing webhook payload")


async def process_single_message(message: dict[str, Any]) -> None:
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
                settings=settings,
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

        # Build prior context from memory_store
        history = memory_store.get(sender) if sender else []
        prior_context = [
            ConversationTurn(role=h["role"], text=h["content"])
            for h in history
        ] if history else None

        # Current time in user-local timezone (for LLM to resolve relative dates correctly)
        tz = ZoneInfo(settings.tenant_timezone)
        current_time_local = datetime.now(tz)

        input_data = ReminderInput(
            input_id=message_id or sender,
            text=user_msg,
            current_time=current_time_local,
            prior_context=prior_context,
        )

        # DSPy calls are blocking — run in thread to avoid stalling the event loop
        result = await asyncio.to_thread(run_agent, input_data)

        # Determine reply text and persist if created
        reply_text = ""

        if result.status == ReminderStatus.CREATED and result.reminder:
            # Persist ReminderItem to DB
            reminder = result.reminder
            # Strip tzinfo for SQLite (naive-UTC at DB boundary)
            due_at_utc = reminder.when
            if due_at_utc.tzinfo is not None:
                due_at_utc = due_at_utc.astimezone(timezone.utc).replace(tzinfo=None)

            with Session(engine) as session:
                item = ReminderItem(
                    tenant_id="default",  # TODO: use actual tenant_id when multi-tenant
                    chat_id=sender,
                    reminder_id=reminder.reminder_id,
                    what=reminder.what,
                    due_at=due_at_utc,
                    status=ReminderItemStatus.PENDING,
                    rendered_message=result.rendered_message,
                )
                session.add(item)
                session.commit()

            reply_text = result.rendered_message or "Reminder set."

        elif result.status == ReminderStatus.NEEDS_CLARIFICATION:
            reply_text = result.clarification_question or "Could you clarify?"

        elif result.status == ReminderStatus.IGNORED:
            reply_text = "I can set reminders for you — try 'remind me to...'"

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
