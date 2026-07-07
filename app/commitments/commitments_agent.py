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

    A commitment means someone is expected to do something.

    Rules:
    - Return an empty list if no commitment exists.
    - If a message updates, completes, or dismisses an existing commitment,
      return that commitment with the same id and updated fields.
    - For brand-new commitments, set id to null.
    - chat_id and chat_name must match the provided inputs.
    - If the committed party is unclear, use null.
    - If the action is vague, write the vague action and set status='unclear'.

    Deadline rules:
    - Resolve relative deadlines to ISO 8601 dates using current_datetime.
      "by end of week" → the upcoming Friday's date.
      "today" → current_datetime's date.
      "tomorrow" → current_datetime's date + 1 day.
      "by Friday" / "next Friday" → the upcoming Friday's date.
      "for Tuesday" → the upcoming Tuesday's date.
      "next Monday" → the upcoming Monday's date.
    - Output the resolved date in YYYY-MM-DD format.
    - If no deadline is mentioned or implied, use null. Do not invent deadlines.

    required_action rules:
    - Write a concise action phrase: verb + object only (e.g. "Send the documents",
      "Book the meeting room", "Call the supplier").
    - Do NOT include deadlines, dates, or time words in the action — those go in
      the deadline field (e.g. write "Pay the invoice", not "Pay the invoice today").

    context rules:
    - Quote the relevant message text exactly as written, WITHOUT the speaker
      name prefix (e.g. write "I'll call the supplier today", NOT "Gaddy: I'll call
      the supplier today").
    - When updating an existing commitment, include only the new message that
      triggers the update, not the original context.

    Act vs Ignore rules (critical — do NOT over-extract):
    - A request without acceptance is NOT a commitment. Someone must explicitly
      agree, volunteer, or state intent to act.
    - "We should do X" is an opinion, NOT a commitment. Look for "I will" or
      "I'll" or explicit acceptance.
    - Generic social intent ("let's catch up", "we should hang out") is NOT a
      commitment. There must be a specific action, topic, or deadline.
    - A conditional promise ("if X, I'll do Y") is NOT a commitment unless the
      condition is clearly satisfied in the messages.
    - A rhetorical question is NOT a commitment, even if it mentions an action.
    - "I hope", "I'm worried", "maybe we could" are NOT commitments.
    - Past-tense statements about completed actions are NOT new commitments
      unless they match an existing commitment (use status='done' in that case).
    - "Started", "almost done", "working on it" are progress reports, NOT
      completions. Do not mark existing commitments as done.
    - Reminding someone else to do something is NOT a commitment by the speaker.
    - Sharing information ("the client will get back to us") is NOT a commitment
      unless the speaker is the one who will act.

    Waiting commitments:
    - "I'm waiting for [party] to [action]" IS a commitment. The committed_party
      is the external party who must act (e.g. "the bank"), the required_action
      is what they must do (e.g. "Process the loan"), and status is 'waiting'.
    - "Nothing I can do until they respond" reinforces a waiting commitment,
      it does NOT cancel it.

    Dismiss + new:
    - "Forget about [X]", "cancel [X]", "never mind about [X]" dismisses an
      existing commitment (return it with status='dismissed').
    - If the same message also describes a new commitment, return BOTH the
      dismissed commitment AND the new one.
    """

    chat_id: str = dspy.InputField()
    chat_name: str | None = dspy.InputField()
    current_datetime: str = dspy.InputField(
        desc="Current date and time in ISO 8601 format (UTC). Use this to interpret relative deadlines like 'tomorrow' or 'next Friday'."
    )
    existing_commitments_json: str = dspy.InputField(
        desc="JSON array of existing commitments, including their ids."
    )
    messages: str = dspy.InputField(
        desc="Recent WhatsApp messages/history to inspect."
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
    ) -> dspy.Prediction:
        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc).isoformat()
        pred = self.extract(
            chat_id=chat_id,
            chat_name=chat_name,
            current_datetime=current_datetime,
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
    ) -> dspy.Prediction:
        if current_datetime is None:
            current_datetime = datetime.now(timezone.utc).isoformat()
        pred = await self.extract.acall(
            chat_id=chat_id,
            chat_name=chat_name,
            current_datetime=current_datetime,
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
