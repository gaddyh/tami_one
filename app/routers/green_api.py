# src/personal_attention_manager/green_api.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from enum import StrEnum

class MessageDirection(StrEnum):
    INBOUND = "inbound"    # customer -> owner
    OUTBOUND = "outbound"  # owner -> customer


INBOUND_WEBHOOK_TYPES = {
    "incomingMessageReceived",
}

OUTBOUND_WEBHOOK_TYPES = {
    "outgoingMessageReceived",
    "outgoingAPIMessageReceived",
}


@dataclass(frozen=True)
class MessageEvent:
    """
    Normalized message event from Green API.

    This is the object the rest of the app should use.
    The rest of our system should not care about Green API's raw JSON shape.
    """

    provider_message_id: str | None
    idInstance: str | None
    wId: str | None
    chat_id: str
    chat_name: str | None
    sender: str | None
    sender_name: str | None
    direction: MessageDirection
    message_type: str | None
    message_time: datetime
    text: str | None
    raw_type_webhook: str | None
    quoted_message_id: str | None = None
    quoted_text: str | None = None
    conversation_id: str | None = None


def normalize_green_api_message_event(
    payload: dict[str, Any],
) -> MessageEvent | None:
    """
    Convert a raw Green API webhook payload into a normalized internal event.

    Returns None for webhooks that are not message events we care about.
    """

    type_webhook = get_type_webhook(payload)
    direction = get_message_direction(type_webhook)

    if direction is None:
        return None

    chat_id = get_chat_id(payload)
    if not chat_id:
        return None

    message_data = get_message_data(payload)

    quoted_message_id, quoted_text = extract_quoted_message(message_data)

    return MessageEvent(
        provider_message_id=get_message_id(payload),
        idInstance=payload.get("instanceData", {}).get("idInstance"),
        wId=payload.get("instanceData", {}).get("wid"),
        chat_id=chat_id,
        chat_name=get_chat_name(payload),
        sender=get_sender(payload),
        sender_name=get_sender_name(payload),
        direction=direction,
        message_type=get_message_type(payload),
        message_time=get_message_time(payload),
        text=extract_message_text(message_data),
        raw_type_webhook=type_webhook,
        quoted_message_id=quoted_message_id,
        quoted_text=quoted_text,
    )


def get_message_direction(
    type_webhook: str | None,
) -> MessageDirection | None:
    if type_webhook in INBOUND_WEBHOOK_TYPES:
        return MessageDirection.INBOUND

    if type_webhook in OUTBOUND_WEBHOOK_TYPES:
        return MessageDirection.OUTBOUND

    return None


def get_type_webhook(payload: dict[str, Any]) -> str | None:
    return payload.get("typeWebhook")


def get_message_id(payload: dict[str, Any]) -> str | None:
    return payload.get("idMessage")


def get_message_time(payload: dict[str, Any]) -> datetime:
    """
    Green API usually sends timestamp as epoch seconds.

    If missing or invalid, fall back to current UTC time.
    """

    timestamp = payload.get("timestamp")

    if isinstance(timestamp, int | float):
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    return datetime.now(timezone.utc)


def get_chat_id(payload: dict[str, Any]) -> str | None:
    sender_data = payload.get("senderData") or {}
    return sender_data.get("chatId")


def get_chat_name(payload: dict[str, Any]) -> str | None:
    sender_data = payload.get("senderData") or {}
    return sender_data.get("chatName")


def get_sender(payload: dict[str, Any]) -> str | None:
    sender_data = payload.get("senderData") or {}
    return sender_data.get("sender")


def get_sender_name(payload: dict[str, Any]) -> str | None:
    sender_data = payload.get("senderData") or {}
    return sender_data.get("senderName")


def get_message_data(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("messageData") or {}


def get_message_type(payload: dict[str, Any]) -> str | None:
    return get_message_data(payload).get("typeMessage")


def extract_message_text(message_data: dict[str, Any]) -> str | None:
    """
    Extract text when the message has text.

    For audio/image/document messages, this returns None.
    They can still count as inbound messages for waiting-chat detection.
    """

    type_message = message_data.get("typeMessage")

    if type_message == "textMessage":
        return extract_text_message(message_data)

    if type_message == "extendedTextMessage":
        return extract_extended_text_message(message_data)

    return None


def extract_text_message(message_data: dict[str, Any]) -> str | None:
    text = (
        (message_data.get("textMessageData") or {})
        .get("textMessage", "")
        .strip()
    )

    return text or None


def extract_extended_text_message(message_data: dict[str, Any]) -> str | None:
    text = (
        (message_data.get("extendedTextMessageData") or {})
        .get("text", "")
        .strip()
    )

    return text or None


def extract_quoted_message(message_data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract quoted-message id and text from extendedTextMessageData.

    Green API puts quoted reply info inside extendedTextMessageData.quotedMessage.
    The stanzaId field is the provider_message_id of the quoted message.
    Returns (quoted_message_id, quoted_text) or (None, None) if not a reply.
    """
    ext_data = message_data.get("extendedTextMessageData") or {}
    quoted = ext_data.get("quotedMessage") or {}
    if not quoted:
        return None, None

    stanza_id = quoted.get("stanzaId")
    quoted_text = None
    if quoted.get("typeMessage") == "textMessage":
        quoted_text = (quoted.get("textMessage") or "").strip() or None

    return stanza_id, quoted_text
