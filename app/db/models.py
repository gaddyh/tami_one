"""SQLModel table definitions for the application."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional
from uuid import uuid4

from sqlmodel import SQLModel, Field
from sqlalchemy import JSON, Column, UniqueConstraint

from app.commitments.models import CommitmentStatus, NotificationType


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TenantKind(StrEnum):
    SOLO = "solo"
    TEAM = "team"


class UserRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class WhatsAppProvider(StrEnum):
    GREEN_API = "green_api"
    WHATSAPP_CLOUD = "whatsapp_cloud"


class ChatKind(StrEnum):
    CLIENT = "client"
    BANK = "bank"
    LAWYER = "lawyer"
    INTERNAL = "internal"
    FAMILY = "family"
    UNKNOWN = "unknown"


class MessageDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    SYSTEM = "system"


class MessageType(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    UNKNOWN = "unknown"


class WaitingStatus(StrEnum):
    NOT_WAITING = "not_waiting"
    MAYBE_WAITING = "maybe_waiting"
    WAITING_ON_AVI = "waiting_on_avi"
    WAITING_ON_OTHER = "waiting_on_other"
    SNOOZED = "snoozed"
    DONE = "done"
    UNCLEAR = "unclear"


class WaitingParty(StrEnum):
    AVI = "avi"
    CLIENT = "client"
    BANK = "bank"
    LAWYER = "lawyer"
    OTHER = "other"
    UNKNOWN = "unknown"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WaitingItemStatus(StrEnum):
    OPEN = "open"
    DONE = "done"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"


class FeedbackAction(StrEnum):
    DONE = "done"
    SNOOZE = "snooze"
    NOT_WAITING = "not_waiting"
    WRONG = "wrong"
    WRONG_PARTY = "wrong_party"
    MARK_WAITING = "mark_waiting"


class Tenant(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)

    name: str
    kind: TenantKind = TenantKind.SOLO

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

class WhatsAppAccount(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider", "provider_instance_id"),
    )

    id: str = Field(default_factory=new_id, primary_key=True)

    tenant_id: str = Field(index=True, foreign_key="tenant.id")

    provider: WhatsAppProvider
    provider_instance_id: str

    chat_id: Optional[str] = Field(default=None, index=True)
    display_name: Optional[str] = None

    is_active: bool = True

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

class Contact(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("tenant_id", "chat_id"),
    )

    id: str = Field(default_factory=new_id, primary_key=True)

    tenant_id: str = Field(index=True, foreign_key="tenant.id")

    display_name: Optional[str] = None
    chat_id: Optional[str] = Field(default=None, index=True)

    kind: ChatKind = ChatKind.UNKNOWN

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Chat(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider_chat_id"),
    )

    id: str = Field(default_factory=new_id, primary_key=True)

    tenant_id: str = Field(index=True, foreign_key="tenant.id")
    whatsapp_account_id: str = Field(index=True, foreign_key="whatsappaccount.id")

    provider_chat_id: str = Field(index=True)

    title: Optional[str] = None
    kind: ChatKind = ChatKind.UNKNOWN

    is_group: bool = False
    is_active: bool = True

    primary_contact_id: Optional[str] = Field(default=None, foreign_key="contact.id")

    last_message_at: Optional[datetime] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ChatMessage(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("tenant_id", "chat_id", "provider_message_id"),
    )

    id: str = Field(default_factory=new_id, primary_key=True)

    tenant_id: str = Field(index=True, foreign_key="tenant.id")

    # Internal FK to Chat.id
    chat_id: str = Field(index=True, foreign_key="chat.id")

    # Original WhatsApp / Green API message id
    provider_message_id: str = Field(index=True)

    direction: MessageDirection
    message_type: MessageType = MessageType.TEXT

    sender_name: Optional[str] = None
    sender_chat_id: Optional[str] = Field(default=None, index=True)

    text: Optional[str] = None

    sent_at: datetime = Field(index=True)
    received_at: datetime = Field(default_factory=utc_now)

    created_at: datetime = Field(default_factory=utc_now)


class CommitmentItem(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)

    tenant_id: str = Field(index=True, foreign_key="tenant.id")
    chat_id: str = Field(index=True)

    committed_party: Optional[str] = None
    required_action: str
    deadline: Optional[str] = None
    context: str

    status: CommitmentStatus = CommitmentStatus.WAITING
    notification: NotificationType = NotificationType.NONE

    source_message_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)