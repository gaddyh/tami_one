from app.commitments.commitments_agent import (
    CommitmentAgent,
    format_existing_commitments,
    normalize_commitments,
)
from app.commitments.models import Commitment

commitment_agent = CommitmentAgent()


async def extract_commitments(
    *,
    chat_id: str,
    chat_name: str | None,
    messages: list[dict],
    existing: list[Commitment] | None = None,
    current_datetime: str | None = None,
) -> list[Commitment]:
    text = "\n".join(
        f"{m.get('senderName') or m.get('senderId')}: "
        f"{m.get('textMessage') or m.get('text') or ''}"
        for m in messages
    )

    pred = await commitment_agent.acall(
        chat_id=chat_id,
        chat_name=chat_name,
        existing_commitments_json=format_existing_commitments(existing),
        messages=text,
        current_datetime=current_datetime,
    )

    commitments = normalize_commitments(
        commitments=pred.commitments,
        chat_id=chat_id,
        chat_name=chat_name,
    )

    return commitments
