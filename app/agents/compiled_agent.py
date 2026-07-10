from __future__ import annotations

import dspy

from app.agents.signatures import (
    ClassifyIntent,
    ExtractReminder,
    GenerateClarification,
    RenderReminder,
)
from app.config import settings


class ReminderAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(ClassifyIntent)
        self.extractor = dspy.Predict(ExtractReminder)
        self.clarifier = dspy.Predict(GenerateClarification)
        self.renderer = dspy.Predict(RenderReminder)


def load_compiled(path: str) -> ReminderAgent:
    agent = ReminderAgent()
    agent.load(path)
    return agent


_compiled_agent: ReminderAgent | None = None


def get_compiled_agent() -> ReminderAgent:
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = load_compiled(settings.compiled_agent_path)
    return _compiled_agent
