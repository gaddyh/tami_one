from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.agents.compiled_agent import get_compiled_agent
from app.agents.schemas import (
    ConversationTurn,
    Reminder,
    ReminderInput,
    ReminderOutput,
    ReminderStatus,
)


def _format_context(prior_context: list[ConversationTurn] | None) -> str:
    if not prior_context:
        return ""
    lines = []
    for turn in prior_context:
        lines.append(f"{turn.role}: {turn.text}")
    return "\n".join(lines)


def _is_empty(value: str | None) -> bool:
    return value is None or value.strip() == ""


def run_agent(input_data: ReminderInput) -> ReminderOutput:
    """Process a reminder request and return either a reminder or a clarification question."""
    agent = get_compiled_agent()
    classifier = agent.classifier
    extractor = agent.extractor
    clarifier = agent.clarifier
    renderer = agent.renderer

    context_str = _format_context(input_data.prior_context)

    # Step 1: Classify intent + detect language
    cls_result = classifier(
        text=input_data.text,
        prior_context=context_str,
    )

    intent = cls_result.intent.strip().lower()
    confidence = float(cls_result.confidence)
    language = getattr(cls_result, "language", "en").strip().lower() or "en"

    if intent == "irrelevant":
        return ReminderOutput(
            input_id=input_data.input_id,
            status=ReminderStatus.IGNORED,
            confidence=min(max(confidence, 0.0), 1.0),
            language=language,
        )

    # Step 2: Extract fields
    ext_result = extractor(
        text=input_data.text,
        prior_context=context_str,
        current_time=input_data.current_time.isoformat(),
    )

    task = ext_result.task.strip() if not _is_empty(ext_result.task) else ""
    # Guard: LLM may return placeholder values when no task was specified
    _INVALID_TASKS = {"remind me", "don't forget", "don't forget to", "make sure i",
                      "reminder", "empty", "none", "n/a", "{task}", "{empty}", "",
                      "not specified", "not stated", "no task", "unknown"}
    if (task.lower() in _INVALID_TASKS
            or task.startswith("{") or task.endswith("}")
            or task.startswith("(") or task.startswith("[")
            or "not explicitly" in task.lower()
            or "leave this" in task.lower()
            or "leave it" in task.lower()):
        task = ""
    date_raw = ext_result.date_raw.strip() if not _is_empty(ext_result.date_raw) else ""
    time_raw = ext_result.time_raw.strip() if not _is_empty(ext_result.time_raw) else ""
    resolved = (
        ext_result.resolved_datetime.strip()
        if not _is_empty(ext_result.resolved_datetime)
        else ""
    )

    # Step 3: Check completeness
    missing: list[str] = []
    if _is_empty(task):
        missing.append("task")
    if _is_empty(date_raw):
        missing.append("date")
    if _is_empty(time_raw):
        missing.append("time")

    # Step 4: If missing fields, ask clarification
    if missing:
        missing_for_llm = [f for f in missing if f != "task"]
        if not task:
            clar_result = clarifier(
                task="",
                missing_fields="task",
                language=language,
            )
            return ReminderOutput(
                input_id=input_data.input_id,
                status=ReminderStatus.NEEDS_CLARIFICATION,
                clarification_question=clar_result.question.strip(),
                missing_fields=missing,
                confidence=0.5,
                language=language,
            )

        clar_result = clarifier(
            task=task,
            missing_fields=", ".join(missing_for_llm) if missing_for_llm else "time, date",
            language=language,
        )

        return ReminderOutput(
            input_id=input_data.input_id,
            status=ReminderStatus.NEEDS_CLARIFICATION,
            clarification_question=clar_result.question.strip(),
            missing_fields=missing,
            confidence=0.7,
            language=language,
        )

    # Step 5: Validate resolved datetime
    if _is_empty(resolved):
        clar_result = clarifier(
            task=task,
            missing_fields="time",
            language=language,
        )
        return ReminderOutput(
            input_id=input_data.input_id,
            status=ReminderStatus.NEEDS_CLARIFICATION,
            clarification_question=clar_result.question.strip(),
            missing_fields=["time"],
            confidence=0.6,
            language=language,
        )

    try:
        resolved_dt = datetime.fromisoformat(resolved)
    except (ValueError, TypeError):
        clar_result = clarifier(
            task=task,
            missing_fields="date, time",
            language=language,
        )
        return ReminderOutput(
            input_id=input_data.input_id,
            status=ReminderStatus.NEEDS_CLARIFICATION,
            clarification_question=clar_result.question.strip(),
            missing_fields=["date", "time"],
            confidence=0.5,
            language=language,
        )

    # Normalize: if naive, assume it's in the same tz as current_time
    if resolved_dt.tzinfo is None and input_data.current_time.tzinfo is not None:
        resolved_dt = resolved_dt.replace(tzinfo=input_data.current_time.tzinfo)

    # Convert to UTC
    if resolved_dt.tzinfo is not None:
        resolved_dt_utc = resolved_dt.astimezone(timezone.utc)
    else:
        resolved_dt_utc = resolved_dt

    # Past-due validation: if the resolved datetime is in the past, ask for a valid future time
    current_time_utc = input_data.current_time
    if current_time_utc.tzinfo is None:
        current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
    current_time_utc = current_time_utc.astimezone(timezone.utc)

    if resolved_dt_utc <= current_time_utc:
        clar_result = clarifier(
            task=task,
            missing_fields="time",
            language=language,
        )
        return ReminderOutput(
            input_id=input_data.input_id,
            status=ReminderStatus.NEEDS_CLARIFICATION,
            clarification_question=clar_result.question.strip(),
            missing_fields=["time"],
            confidence=0.6,
            language=language,
        )

    # Step 6: Create reminder
    reminder = Reminder(
        reminder_id=f"r-{uuid.uuid4().hex[:8]}",
        what=task,
        when=resolved_dt_utc,
        created_at=input_data.current_time,
    )

    # Step 7: Render confirmation message in user's language
    render_result = renderer(
        task=task,
        when=resolved_dt_utc.isoformat(),
        language=language,
    )
    rendered_message = render_result.message.strip()

    return ReminderOutput(
        input_id=input_data.input_id,
        status=ReminderStatus.CREATED,
        reminder=reminder,
        confidence=0.9,
        language=language,
        rendered_message=rendered_message,
    )
