from __future__ import annotations

from datetime import datetime, timezone

import dspy


class ProcessReviewReply(dspy.Signature):
    """Process a user's reply during a task review session and return the complete updated candidate state.

    You are given:
    - raw_subject: the original item subject from the user's saved items
    - current_candidate_subject: the actionable task description extracted so far (may be empty)
    - current_candidate_due_at: ISO datetime extracted so far (may be empty)
    - current_due_date_resolution: one of "unknown", "provided", "intentionally_absent"
    - user_reply: the user's latest message
    - current_time: ISO datetime for resolving relative time expressions

    You must return the COMPLETE updated state (not partial deltas):
    - updated_subject: the actionable task description, or empty if not yet determined
    - updated_due_at: ISO 8601 datetime if a due date/time was provided, or empty
    - due_date_resolution: "provided" if a date was given, "intentionally_absent" if the user
      explicitly said no date / doesn't know, "unknown" if not yet discussed
    - needs_clarification: true if you need more info to form an actionable task
    - clarification_question: natural language question in Hebrew if needs_clarification is true

    RULES:
    - The subject must be an actionable task (e.g. "לקחת אופניים לתיקון"), not just the raw entity.
    - If the user says "לא יודע", "אין תאריך", "sometime", "no date" → set due_date_resolution to
      "intentionally_absent" and updated_due_at to empty.
    - If the user provides a time but no action, keep the current candidate_subject and ask for the action.
    - If the user provides an action but no time, ask "מתי תרצה לבצע את זה?" and set
      due_date_resolution to "unknown" (not "intentionally_absent").
    - Merge information from the current candidate state with the new reply. Do not lose
      previously extracted information unless the user explicitly changes it.
    - The clarification question must be in Hebrew and reference the specific missing information.
    """

    raw_subject: str = dspy.InputField(desc="The original item subject from saved items")
    current_candidate_subject: str = dspy.InputField(desc="Actionable task description extracted so far, or empty")
    current_candidate_due_at: str = dspy.InputField(desc="ISO datetime extracted so far, or empty")
    current_due_date_resolution: str = dspy.InputField(desc="unknown, provided, or intentionally_absent")
    user_reply: str = dspy.InputField(desc="The user's latest message")
    current_time: str = dspy.InputField(desc="ISO 8601 datetime for resolving relative times")

    updated_subject: str = dspy.OutputField(desc="Complete actionable task description, or empty if not yet determined")
    updated_due_at: str = dspy.OutputField(desc="ISO 8601 datetime if provided, or empty")
    due_date_resolution: str = dspy.OutputField(desc="provided, intentionally_absent, or unknown")
    needs_clarification: str = dspy.OutputField(desc="true or false")
    clarification_question: str = dspy.OutputField(desc="Hebrew clarification question if needed, or empty")


class TaskReviewAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process = dspy.Predict(ProcessReviewReply)

    def forward(
        self,
        *,
        raw_subject: str,
        current_candidate_subject: str,
        current_candidate_due_at: str,
        current_due_date_resolution: str,
        user_reply: str,
        current_time: str | None = None,
    ) -> dspy.Prediction:
        if current_time is None:
            current_time = datetime.now(timezone.utc).isoformat()
        pred = self.process(
            raw_subject=raw_subject,
            current_candidate_subject=current_candidate_subject,
            current_candidate_due_at=current_candidate_due_at,
            current_due_date_resolution=current_due_date_resolution,
            user_reply=user_reply,
            current_time=current_time,
        )
        return dspy.Prediction(
            updated_subject=pred.updated_subject,
            updated_due_at=pred.updated_due_at,
            due_date_resolution=pred.due_date_resolution,
            needs_clarification=pred.needs_clarification,
            clarification_question=pred.clarification_question,
        )

    async def aforward(
        self,
        *,
        raw_subject: str,
        current_candidate_subject: str,
        current_candidate_due_at: str,
        current_due_date_resolution: str,
        user_reply: str,
        current_time: str | None = None,
    ) -> dspy.Prediction:
        if current_time is None:
            current_time = datetime.now(timezone.utc).isoformat()
        pred = await self.process.acall(
            raw_subject=raw_subject,
            current_candidate_subject=current_candidate_subject,
            current_candidate_due_at=current_candidate_due_at,
            current_due_date_resolution=current_due_date_resolution,
            user_reply=user_reply,
            current_time=current_time,
        )
        return dspy.Prediction(
            updated_subject=pred.updated_subject,
            updated_due_at=pred.updated_due_at,
            due_date_resolution=pred.due_date_resolution,
            needs_clarification=pred.needs_clarification,
            clarification_question=pred.clarification_question,
        )
