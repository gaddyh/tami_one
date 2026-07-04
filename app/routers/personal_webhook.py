# src/personal_attention_manager/main.py

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.config import settings
from app.routers.green_api import normalize_green_api_message_event


_log_level = getattr(
    logging,
    os.getenv("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)

logging.basicConfig(level=_log_level)

logger = logging.getLogger("waiting-for-you")

router = APIRouter()

def verify_green_api_authorization(request: Request) -> None:
    authorization = request.headers.get("authorization")

    if authorization != settings.expected_authorization_header:
        logger.warning("Rejected webhook with invalid Authorization header")
        raise HTTPException(status_code=403, detail="Invalid webhook authorization")


async def handle_green_api_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Green API webhook entry point.

    This handler only ingests WhatsApp message events.

    It does not:
    - call an AI agent
    - reply to the customer through Green API
    - transcribe audio
    - send business notifications directly

    The scheduler is responsible for notification timing.
    """

    event = normalize_green_api_message_event(payload)

    if event is None:
        logger.info(
            "Ignoring unsupported webhook type=%s",
            payload.get("typeWebhook"),
        )

        return {
            "ok": True,
            "ignored": "unsupported_webhook_type",
            "typeWebhook": payload.get("typeWebhook"),
        }

    logger.info(
        "Green API message event chat_id=%s chat_name=%s direction=%s type=%s",
        event.chat_id,
        event.chat_name,
        event.direction,
        event.message_type,
    )

    if settings.allowed_chat_ids and event.chat_id not in settings.allowed_chat_ids:
        logger.info(
            "Ignoring message from non-allowed chat_id=%s chat_name=%s",
            event.chat_id,
            event.chat_name,
        )

        return {
            "ok": True,
            "ignored": "chat_not_allowed",
            "chatId": event.chat_id,
            "chatName": event.chat_name,
        }

    return {
        "ok": True,
        "handled": "message_event",
        "event": {
            "providerMessageId": event.provider_message_id,
            "chatId": event.chat_id,
            "chatName": event.chat_name,
            "direction": event.direction.value,
            "messageType": event.message_type,
            "messageTime": event.message_time.isoformat(),
            "hasText": event.text is not None,
        },
        "chatState": None
        if chat is None
        else {
            "chatId": chat.chat_id,
            "contactName": chat.contact_name,
            "status": chat.status.value,
            "waitingSince": (
                chat.waiting_since.isoformat()
                if chat.waiting_since
                else None
            ),
            "lastInboundMessageAt": (
                chat.last_inbound_message_at.isoformat()
                if chat.last_inbound_message_at
                else None
            ),
            "lastOutboundMessageAt": (
                chat.last_outbound_message_at.isoformat()
                if chat.last_outbound_message_at
                else None
            ),
            "lastMessageDirection": (
                chat.last_message_direction.value
                if chat.last_message_direction
                else None
            ),
        },
    }
    
@router.get("/debug/settings")
async def debug_settings() -> dict[str, Any]:
    """
    Temporary dev endpoint.

    Do not return secrets.
    Remove or protect this before production.
    """

    return {
        "green_api_base_url": settings.green_api_base_url,
        "green_api_id_instance_configured": bool(settings.green_api_id_instance),
        "green_api_token_instance_configured": bool(
            settings.green_api_token_instance
        ),
        "webhook_secret_configured": bool(settings.webhook_secret),
        "allowed_chat_ids_count": len(settings.allowed_chat_ids),
        "timezone": settings.timezone,
    }


@router.post("/webhook/green-api")
async def green_api_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Green API webhook endpoint.

    This endpoint only ingests message events.
    It does not reply to the customer's WhatsApp chat.
    """

    verify_green_api_authorization(request)
    
    payload = await request.json()

    logger.info(
        "Green API webhook received type=%s idMessage=%s",
        payload.get("typeWebhook"),
        payload.get("idMessage"),
    )

    background_tasks.add_task(handle_green_api_webhook, payload)

    return {"status": "ok"}