from enum import StrEnum

from pydantic import BaseModel, Field


class CommitmentStatus(StrEnum):
    WAITING = "waiting"
    DONE = "done"
    UNCLEAR = "unclear"
    DISMISSED = "dismissed"


class NotificationType(StrEnum):
    NONE = "none"
    DAILY_DIGEST = "daily_digest"
    URGENT = "urgent"


class Commitment(BaseModel):
    id: str | None = Field(
        default=None,
        description="Existing commitment id when updating; null for brand-new commitments.",
    )
    chat_id: str = Field(
        description="The WhatsApp chat id. Must match the input chat_id.",
    )
    chat_name: str | None = Field(
        default=None,
        description="The WhatsApp chat/group name. Must match the input chat_name.",
    )

    committed_party: str | None = Field(
        default=None,
        description="Who is expected to do the action. Use null if unclear.",
    )
    required_action: str = Field(
        description="The action someone is expected to do. If vague, write the vague action and set status='unclear'.",
    )
    deadline: str | None = Field(
        default=None,
        description="Explicit deadline only in ISO 8601 format (e.g. 2025-07-10T14:30:00). Do not infer or invent deadlines.",
    )
    context: str = Field(
        description="Short evidence/context from the WhatsApp messages.",
    )

    status: CommitmentStatus = Field(
        default=CommitmentStatus.WAITING,
        description="waiting for active commitments; done if completed; dismissed if cancelled; unclear if vague.",
    )

    notification: NotificationType = Field(
        default=NotificationType.NONE,
        description="none unless the commitment should appear in a daily digest or urgent notification.",
    )


class CommitmentList(BaseModel):
    commitments: list[Commitment] = Field(default_factory=list)
