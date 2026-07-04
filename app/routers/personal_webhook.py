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

    logger.info("Green API message event: %s", event)

    return {"ok": True, "event": event}

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

    #verify_green_api_authorization(request)
    
    payload = await request.json()

    logger.info(
        "Green API webhook received type=%s idMessage=%s",
        payload.get("typeWebhook"),
        payload.get("idMessage"),
    )

    background_tasks.add_task(handle_green_api_webhook, payload)

    return {"status": "ok"}