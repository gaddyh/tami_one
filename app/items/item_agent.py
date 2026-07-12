from __future__ import annotations

from datetime import datetime, timezone

import dspy


class ExtractItem(dspy.Signature):
    """Extract a subject and optional due_at from a short user message.

    The input may be:
    - A pure entity / topic: "עירית", "אופניים", "רשיון בינלאומי", "נעלי ים חגור ק״ש"
    - An entity with an exact time reference: "בחמש" (at 5), "מחר ב8" (tomorrow at 8)
    - An entity with a relative time reference: "עשר דקות" (10 minutes),
      "בעוד שעה" (in an hour)

    SUBJECT RULES:
    - Extract the core subject / entity from the message.
    - Strip time-related words from the subject. The subject should be the WHAT,
      not the WHEN.
    - If the message is only a time expression with no entity, set subject to
      the time expression itself (e.g. "עשר דקות" → subject="עשר דקות").
    - Keep the subject in the original language of the message.

    DUE_AT RULES:
    - Resolve any time reference to an ISO 8601 datetime using current_time.
    - Exact times: "בחמש" → today at 17:00 (if current_time is before 17:00,
      otherwise tomorrow at 05:00 or 17:00 next day — use context).
      "מחר ב8" → tomorrow at 08:00.
    - Relative times: "עשר דקות" → current_time + 10 minutes.
      "בעוד שעה" → current_time + 1 hour.
    - If no time reference is present, set due_at to empty string.
    - Output due_at in ISO 8601 format (e.g. "2025-07-13T17:00:00").
    """

    text: str = dspy.InputField(desc="The user message, possibly containing an entity and/or time reference")
    current_time: str = dspy.InputField(desc="ISO 8601 datetime for resolving relative times")
    subject: str = dspy.OutputField(desc="The extracted subject/entity, or the time expression if no entity is present")
    due_at: str = dspy.OutputField(desc="ISO 8601 datetime if a time reference is present, or empty string")


class ItemAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ExtractItem)

    def forward(self, *, text: str, current_time: str | None = None) -> dspy.Prediction:
        if current_time is None:
            current_time = datetime.now(timezone.utc).isoformat()
        pred = self.extract(text=text, current_time=current_time)
        return dspy.Prediction(subject=pred.subject, due_at=pred.due_at)

    async def aforward(self, *, text: str, current_time: str | None = None) -> dspy.Prediction:
        if current_time is None:
            current_time = datetime.now(timezone.utc).isoformat()
        pred = await self.extract.acall(text=text, current_time=current_time)
        return dspy.Prediction(subject=pred.subject, due_at=pred.due_at)
