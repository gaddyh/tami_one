from __future__ import annotations

from langsmith import traceable

from app.items.item_agent import ItemAgent

item_agent = ItemAgent()


@traceable(name="extract_item", run_type="llm")
async def extract_item(*, text: str, current_time: str) -> dict:
    """Extract subject and due_at from a user message.

    Returns {"subject": str, "due_at": str} where due_at is ISO datetime or empty string.
    """
    pred = await item_agent.acall(text=text, current_time=current_time)
    return {
        "subject": _safe_str(pred.subject),
        "due_at": _safe_str(pred.due_at),
    }


def _safe_str(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()
