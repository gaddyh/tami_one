import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from app.agents.core import run_agent
from app.config import settings
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
            "Replying to message_id=%s from=%s type=%s",
            message_id,
            sender,
            msg_type,
        )

        result = await run_agent(user_msg, thread_id=sender)

        reply = result.reply
        await wa.send_text(to=sender, body=reply)

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
