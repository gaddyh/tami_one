from __future__ import annotations

import json
from typing import Sequence

import dspy
from pydantic import BaseModel

from app.commitments.models import Commitment, CommitmentList


class ExtractCommitments(dspy.Signature):
    """
    Extract and update commitments from WhatsApp group history.

    A commitment means someone is expected to do something.

    Rules:
    - Do not invent deadlines.
    - If the committed party is unclear, use null.
    - If the action is vague, mark status='unclear'.
    - Return an empty list if no commitment exists.
    - If a message updates, completes, or dismisses an existing commitment,
      return that commitment with the same id and updated fields.
    - For brand-new commitments, set id to null.
    - chat_id and chat_name must match the provided inputs.

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
    """

    chat_id: str = dspy.InputField()
    chat_name: str | None = dspy.InputField()
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
    ) -> dspy.Prediction:
        pred = self.extract(
            chat_id=chat_id,
            chat_name=chat_name,
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
    ) -> dspy.Prediction:
        pred = await self.extract.acall(
            chat_id=chat_id,
            chat_name=chat_name,
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
