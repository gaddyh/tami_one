from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

import dspy
from pydantic import BaseModel

from app.commitments.models import Commitment, CommitmentList


class ExtractCommitments(dspy.Signature):
    """
    Extract and update commitments from WhatsApp chat history.

    A commitment means someone is expected to do something specific.
    Return ONLY commitments that are new, updated, completed, dismissed, or unclear.
    If no commitment exists, return an empty list.

    ─── OUTPUT RULES ───
    - For brand-new commitments, set id to null.
    - If a message updates, completes, or dismisses an existing commitment,
      return that commitment with the same id and updated fields.
    - chat_id and chat_name must match the provided inputs.
    - If the committed party is unclear, use null.
    - If the action is vague but a commitment is implied, write the vague action
      and set status='unclear'.

    ─── WHAT COUNTS AS A COMMITMENT (act) ───
    - "I'll X" / "I will X" / "I'm going to X" + specific action → commitment.
    - "I need to X" + specific action + clear intent to act → commitment.
      Contrast: "we need to X" or "someone needs to X" without a volunteer → NOT a commitment.
    - "I can X" / "let me X" + specific action → commitment (offer to help).
    - Request + acceptance → commitment by the person who accepts.
      Acceptance signals: "sure", "on it", "I'll do it", "I'll have it ready", etc.
    - Volunteering ("I'll take care of X", "I'll book it") → commitment.
    - Speaker asserts someone else will act ("Dana will send the contract") →
      commitment with that party as committed_party.
    - "I'm waiting for [party] to [action]" → commitment. committed_party is the
      external party (e.g. "the bank"), required_action is what they must do.
    - "Let me review X and get back to you" + topic + deadline → commitment.
      Without topic or deadline, this is NOT a commitment.
    - Conditional promise ("if X, I'll do Y") → commitment ONLY if the condition
      is clearly satisfied in the messages. Otherwise ignore.
    - Multi-message: intent in one message + action/deadline in another → combine
      into a single commitment.
    - Hedged language ("I might", "maybe") + specific action → status='unclear'.
    - Deadline inheritance: when someone accepts a request that mentions a deadline,
      inherit that deadline unless they override it.
      Example: "Can someone book the room for Tuesday?" / "I'll book it"
      → deadline = upcoming Tuesday (inherited from the request).
      Example: "Can you send the report by Friday?" / "Sure, I'll have it ready"
      → deadline = upcoming Friday (inherited from the request).

    ─── WHAT TO IGNORE (do NOT over-extract) ───
    - Social chat, greetings, thanks, motivational phrases.
    - "We should X" / "we need to X" / "the report needs to be X" → opinion, not commitment.
    - Request with no acceptance or volunteer → not yet a commitment.
      "can someone X?" / "can you X?" with no reply → IGNORE. A request alone
      is never a commitment, even if it mentions a deadline.
      Example: "can someone send me the report by Friday?" with no reply → IGNORE.
      Contrast: "can you send me the report by Friday?" / "sure, I'll do it" → commitment.
    - Refusal ("I can't", "I won't", "I'm swamped") → no commitment.
    - Generic callback without topic or deadline ("get back to you on that") → ignore.
    - Rhetorical questions, even if they mention an action.
    - "I hope", "I'm worried", "maybe we could" → not commitments.
    - Past-tense statements about completed actions with no existing commitment → ignore.
    - "Started", "almost done", "working on it" → progress reports, NOT completions.
    - Reminding someone else to do something → not a commitment by the speaker.
    - Sharing information ("the client will get back to us") → not a commitment
      unless the speaker is the one who will act.
    - Reporting current status ("server is down, looking into it") → not a commitment.
    - Suggestion ("maybe we could try X") → not a commitment.
    - Question about past events ("did anyone follow up?") → not a commitment.

    ─── REQUIRED_ACTION RULES ───
    - Write a concise action phrase: verb + object only. No extra detail.
      Example: "Call the supplier" (NOT "Call about the order" or "Call a new supplier about a different order")
      Example: "Send the documents" (NOT "Send the documents tomorrow")
    - Do NOT include deadlines, dates, or time words in the action.
    - Do NOT include descriptive qualifiers from the message ("about the order",
      "about a different order", "to the client") unless needed to distinguish
      two different commitments with the same verb.
    - Map indirect language to the canonical action:
      "paperwork" / "docs" / "get the paperwork to you" → "Send the documents"
      "give you a ring" / "call you" / "give you a call" → "Call the supplier"
      "take care of the payment" / "settle the invoice" → "Pay the invoice"
      "draft the report" / "have the report ready" → "Prepare the report"
      "reserve the room" → "Book the meeting room"
      "send over the docs" → "Send the documents"
      "call a new supplier" → "Call a new supplier" (NOT "Call a new supplier about a different order")
    - Multi-step action in one message → single combined action
      (e.g. "Review, sign, and send back the contract").
    - Action details split across messages → combine into one action.
    - "Get back to [person] about [topic]" is a valid action when callback has
      topic and deadline.
    - "Make sure X is taken care of" → extract the underlying action if inferable.

    ─── DEADLINE RULES ───
    - Resolve relative deadlines to ISO 8601 dates using current_datetime.
      If current_datetime is 2025-01-06T10:00:00Z (Monday):
      "today" → "2025-01-06"
      "tomorrow" → "2025-01-07"
      "by Friday" / "next Friday" / "by end of week" → "2025-01-10"
      "for Tuesday" → "2025-01-07" (upcoming Tuesday)
      "next Monday" → "2025-01-13"
      "next week" → "2025-01-13" (start of next week, Monday)
      "by 5pm" / "by 17:00" → "2025-01-06T17:00:00"
      "before noon" → "2025-01-06T12:00:00"
      "this afternoon" → "2025-01-06T14:00:00"
      "end of day" → "2025-01-06T17:00:00"
      "by July 15" → "2025-07-15"
      "by end of the quarter" → "2025-03-31" (end of current quarter)
    - Output format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS" if time is specified.
    - "soon", "when I can", "at some point" → null (not concrete).
    - Event-dependent phrases ("as soon as I get the signature",
      "once the legal team approves them", "before the meeting") → null
      for NEW commitments. For UPDATES to existing commitments, these phrases
      do NOT null an existing deadline — keep the existing deadline unless the
      message explicitly sets a new one.
      Example: existing deadline=2025-01-07, message "I'll send them as soon as
      I get the signature" → deadline stays 2025-01-07 (context update only).
    - Conditional dismiss deadlines: "unless they respond today" → resolve
      the condition's deadline ("today" → current date).
    - Conflicting deadlines across messages: use the most recent/specific one.
      Example: "I'll send them by Friday" then "actually, let me send them by tomorrow"
      → deadline = tomorrow.
    - No deadline mentioned or implied → null. Do not invent deadlines.

    ─── CONTEXT RULES ───
    - Quote the FULL relevant message text exactly as written, WITHOUT the speaker
      name prefix. Include the complete sentence or clause — do not truncate.
      Example: "I won't manage tomorrow, I'll send the documents next Monday"
      (NOT just "I'll send the documents next Monday")
      Example: "never mind about the documents, we don't need them anymore"
      (NOT just "we don't need them anymore")
    - Do NOT include the addressee name if it's part of the message
      (e.g. "Bob, can you send the report?" → context = "can you send the report?")
    - When updating an existing commitment, use only the new message text, not
      the original context.
    - For multi-message commitments, include fragments from each relevant message.

    ─── UPDATE vs NEW (when existing_commitments is non-empty) ───
    - Same action + same party + new deadline or context → UPDATE (same id).
    - Same action + different party → NEW (id=null), not an update.
    - Different action → NEW (id=null), even if similar to existing.
    - Action change on existing ("email instead of calling") → UPDATE with new
      required_action.
    - Party handoff ("Bob will handle it instead of Alice") → UPDATE with new
      committed_party.
    - "I also need to X" → NEW commitment in addition to existing ones.
    - "Still planning to X" → UPDATE (reaffirm, may update context).
    - Similar but different target ("send documents" vs "send contract to client")
      → NEW, not update.
    - With multiple existing commitments, only update the one that changed.
    - Do NOT return unchanged existing commitments. Only return commitments
      that are new, updated, completed, or dismissed.

    ─── COMPLETION (status='done') ───
    - Past tense + existing commitment → done ("I sent the documents yesterday").
    - "done", "finished", "submitted", "paid" + existing → done.
    - Passive voice ("the documents were sent") + existing → done.
    - Third-party report ("Alice told me Bob submitted the report") + existing → done.
    - "Started", "almost done", "working on it" → NOT done. Return empty list or
      update context only.
    - "I'll do it" / "will do" → NOT done. It's a future promise — update deadline
      if specified, keep status='waiting'.
    - Partial completion ("sent 3 out of 5") → NOT done. Update context, keep waiting.
    - Past tense without existing commitment → ignore (nothing to mark done).

    ─── DISMISS (status='dismissed') ───
    - "Forget about X", "cancel X", "never mind about X", "we don't need X anymore"
      → dismiss existing commitment (same id, status='dismissed').
    - Conditional dismiss ("forget about X unless Y") → NOT dismissed. Keep
      status='waiting', update deadline if the condition implies one.
    - If the same message also describes a new commitment, return BOTH the
      dismissed commitment AND the new one.
    - With multiple existing commitments, only dismiss the one explicitly mentioned.
      Do NOT return unchanged existing commitments — only the dismissed one.
    """

    chat_id: str = dspy.InputField()
    chat_name: str | None = dspy.InputField()
    current_datetime: str = dspy.InputField(
        desc="Current date and time in ISO 8601 format (UTC). Use this to interpret relative deadlines like 'tomorrow' or 'next Friday'."
    )
    conversation_summary: str = dspy.InputField(
        desc="Rolling one-line summary of the conversation topic so far. May be empty for new conversations."
    )
    existing_commitments_json: str = dspy.InputField(
        desc="JSON array of existing commitments, including their ids."
    )
    messages: str = dspy.InputField(
        desc="WhatsApp messages to inspect. May include prior conversation history (already-processed messages) followed by new messages. Focus extraction on the new messages, but use prior context to resolve references like 'it' or 'the documents'."
    )

    commitments: list[Commitment] = dspy.OutputField(
        desc="Only commitments that are new, updated, completed, dismissed, or unclear."
    )


class CommitmentAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractCommitments)

    def forward(
        self,
        chat_id: str,
        chat_name: str | None,
        existing_commitments_json: str,
        messages: str,
        current_datetime: str | None = None,
        conversation_summary: str = "",
    ) -> dspy.Prediction:
        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc).isoformat()
        pred = self.extract(
            chat_id=chat_id,
            chat_name=chat_name,
            current_datetime=current_datetime,
            conversation_summary=conversation_summary,
            existing_commitments_json=existing_commitments_json,
            messages=messages,
        )

        commitment_list = _coerce_commitment_list(pred.commitments)
        return dspy.Prediction(
            commitments=commitment_list.commitments,
            commitment_list=commitment_list,
        )

    async def aforward(
        self,
        chat_id: str,
        chat_name: str | None,
        existing_commitments_json: str,
        messages: str,
        current_datetime: str | None = None,
        conversation_summary: str = "",
    ) -> dspy.Prediction:
        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc).isoformat()
        pred = await self.extract.acall(
            chat_id=chat_id,
            chat_name=chat_name,
            current_datetime=current_datetime,
            conversation_summary=conversation_summary,
            existing_commitments_json=existing_commitments_json,
            messages=messages,
        )

        commitment_list = _coerce_commitment_list(pred.commitments)
        return dspy.Prediction(
            commitments=commitment_list.commitments,
            commitment_list=commitment_list,
        )


def configure_dspy(settings) -> None:
    model = settings.openai_model

    if not model.startswith("openai/"):
        model = f"openai/{model}"

    lm_kwargs: dict = {
        "api_key": getattr(settings, "openai_api_key", None),
        "drop_params": True,
    }

    if "gpt-5" not in model:
        lm_kwargs["temperature"] = 0.0

    dspy.configure(
        lm=dspy.LM(model, **lm_kwargs),
        adapter=dspy.JSONAdapter(),
    )


def format_existing_commitments(
    existing: CommitmentList | Sequence[Commitment] | None,
) -> str:
    if existing is None:
        return "[]"

    if isinstance(existing, CommitmentList):
        commitments = existing.commitments
    else:
        commitments = list(existing)

    return json.dumps(
        [
            c.model_dump(mode="json") if isinstance(c, BaseModel) else c
            for c in commitments
        ],
        ensure_ascii=False,
        indent=2,
    )


def normalize_commitments(
    *,
    commitments: list[Commitment],
    chat_id: str,
    chat_name: str | None,
) -> list[Commitment]:
    return [
        c.model_copy(
            update={
                "chat_id": chat_id,
                "chat_name": chat_name,
            }
        )
        for c in commitments
    ]


def _coerce_commitment_list(raw_commitments) -> CommitmentList:
    if raw_commitments is None:
        return CommitmentList()

    commitments: list[Commitment] = []

    for item in raw_commitments:
        if isinstance(item, Commitment):
            commitments.append(item)
        else:
            commitments.append(Commitment.model_validate(item))

    return CommitmentList(commitments=commitments)
