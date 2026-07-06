from openai import AsyncOpenAI
from app.config import settings
from app.commitments.models import Commitment, CommitmentList

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def extract_commitments(
    *,
    chat_id: str,
    chat_name: str | None,
    messages: list[dict],
    existing: list[Commitment] | None = None,
) -> list[Commitment]:
    text = "\n".join(
        f"{m.get('senderName') or m.get('senderId')}: "
        f"{m.get('textMessage') or m.get('text') or ''}"
        for m in messages
    )

    response = await client.responses.parse(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": """
Extract and update commitments from WhatsApp group history.

A commitment means someone is expected to do something.
Do not invent deadlines.
If the party is unclear, use null.
If action is vague, mark status=unclear.
Return empty list if no commitment exists.

If a new message updates, completes, or dismisses an existing
commitment, return that commitment with the same id and updated
fields (e.g. status=done, new deadline).
For brand-new commitments, set id to null.
""",
            },
            {
                "role": "user",
                "content": f"""
chat_id: {chat_id}
chat_name: {chat_name}

Existing commitments:
{_format_existing(existing)}

Messages:
{text}
""",
            },
        ],
        text_format=CommitmentList,
    )

    return response.output_parsed.commitments


def _format_existing(existing: list[Commitment] | None) -> str:
    if not existing:
        return "(none)"
    lines = []
    for c in existing:
        lines.append(
            f"id={c.id} | party={c.committed_party} | action={c.required_action} "
            f"| deadline={c.deadline} | status={c.status} | context={c.context}"
        )
    return "\n".join(lines)
