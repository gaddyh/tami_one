from pydantic import BaseModel
from typing import Literal


class Commitment(BaseModel):
    id: str | None = None
    chat_id: str
    chat_name: str | None = None

    committed_party: str | None
    required_action: str
    deadline: str | None = None
    context: str

    status: Literal[
        "open",
        "done",
        "waiting",
        "unclear",
        "dismissed",
    ] = "open"

    notification: Literal[
        "none",
        "daily_digest",
        "urgent",
    ] = "daily_digest"
