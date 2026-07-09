"""Rolling conversation summary via DSPy.

One LLM call per conversation per drain cycle, only for conversations
whose extraction succeeded. Generates a one-line rolling summary that
gives the extractor topic context without unbounded message history.
"""

from __future__ import annotations

import logging

import dspy

logger = logging.getLogger(__name__)


class SummarizeConversation(dspy.Signature):
    """Update a one-line rolling summary of a WhatsApp conversation.

    Given the existing summary (may be empty) and new messages, produce
    a concise one-line summary that captures the current topic and any
    open obligations or pending questions. Keep it under 100 characters.
    Write in the same language as the messages (Hebrew or English).
    """

    conversation_summary: str = dspy.InputField(
        desc="Current rolling summary, or empty string if this is a new conversation."
    )
    new_messages: str = dspy.InputField(
        desc="New WhatsApp messages in this conversation since the last summary update."
    )

    updated_summary: str = dspy.OutputField(
        desc="One-line rolling summary (max ~100 chars). Captures the current topic and key obligations."
    )


_summarizer: dspy.Predict | None = None


def _get_summarizer() -> dspy.Predict:
    global _summarizer
    if _summarizer is None:
        _summarizer = dspy.Predict(SummarizeConversation)
    return _summarizer


async def summarize_conversation(
    *,
    prior_summary: str,
    new_messages: str,
) -> str:
    """Generate an updated rolling summary.

    Args:
        prior_summary: Current summary, or "" for new conversations.
        new_messages: Formatted new messages (senderName: text per line).

    Returns:
        Updated one-line summary.
    """
    predictor = _get_summarizer()
    pred = await predictor.acall(
        conversation_summary=prior_summary,
        new_messages=new_messages,
    )
    summary = pred.updated_summary.strip()
    if not summary:
        return prior_summary
    return summary
