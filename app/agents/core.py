from __future__ import annotations

import time
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
    return value is None or not isinstance(value, str) or value.strip() == ""


def _safe_str(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _safe_confidence(value, default: float = 0.5) -> float:
    try:
        c = float(value)
    except (ValueError, TypeError):
        return default
    return min(max(c, 0.0), 1.0)


_INVALID_TASKS = {"remind me", "don't forget", "don't forget to", "make sure i",
                  "reminder", "empty", "none", "n/a", "{task}", "{empty}", "",
                  "not specified", "not stated", "no task", "unknown"}

_FALLBACK_CLARIFICATION = {
    "en": "Sorry, I didn't catch that. What would you like me to remind you about, and when?",
    "he": "סליחה, לא הבנתי. על מה ומתי תרצה שאזכיר לך?",
}


def _clean_task(raw) -> str:
    task = _safe_str(raw)
    lowered = task.lower()
    if (lowered in _INVALID_TASKS
            or task.startswith("{") or task.endswith("}")
            or task.startswith("(") or task.startswith("[")
            or "not explicitly" in lowered
            or "leave this" in lowered
            or "leave it" in lowered):
        return ""
    return task


def run_agent(input_data: ReminderInput) -> ReminderOutput:
    """Process a reminder request and return either a reminder or a clarification question.

    Guarantees: never raises, always sets ``language``, and records per-stage plus
    total latency in ``timings`` (milliseconds).
    """
    agent = get_compiled_agent()

    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    language = "en"

    def _timed(stage: str, module, **kwargs):
        t0 = time.perf_counter()
        try:
            return module(**kwargs)
        finally:
            timings[f"{stage}_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    def _finish(**kwargs) -> ReminderOutput:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
        kwargs.setdefault("language", language)
        return ReminderOutput(input_id=input_data.input_id, timings=dict(timings), **kwargs)

    try:
        classifier = agent.classifier
        extractor = agent.extractor
        clarifier = agent.clarifier
        renderer = agent.renderer

        context_str = _format_context(input_data.prior_context)

        # Step 1: Classify intent + detect language
        cls_result = _timed(
            "classify", classifier,
            text=input_data.text,
            prior_context=context_str,
        )

        intent = _safe_str(cls_result.intent).lower()
        confidence = _safe_confidence(getattr(cls_result, "confidence", None), default=0.5)
        language = _safe_str(getattr(cls_result, "language", "en")).lower() or "en"

        if intent == "irrelevant":
            return _finish(
                status=ReminderStatus.IGNORED,
                confidence=confidence,
            )

        # Step 2: Extract fields
        ext_result = _timed(
            "extract", extractor,
            text=input_data.text,
            prior_context=context_str,
            current_time=input_data.current_time.isoformat(),
        )

        task = _clean_task(getattr(ext_result, "task", ""))
        date_raw = _safe_str(getattr(ext_result, "date_raw", ""))
        time_raw = _safe_str(getattr(ext_result, "time_raw", ""))
        resolved = _safe_str(getattr(ext_result, "resolved_datetime", ""))

        # Step 3: Check completeness
        _RELATIVE_HINTS = ("in ", "עוד ", "בעוד ")
        is_relative_time = (
            time_raw
            and any(time_raw.lower().startswith(h) for h in _RELATIVE_HINTS)
        )
        missing: list[str] = []
        if is_relative_time and not _is_empty(resolved):
            pass
        else:
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
                clar_result = _timed(
                    "clarify", clarifier,
                    task="", missing_fields="task", language=language,
                )
                return _finish(
                    status=ReminderStatus.NEEDS_CLARIFICATION,
                    clarification_question=_safe_str(clar_result.question),
                    missing_fields=missing,
                    confidence=0.5,
                )

            clar_result = _timed(
                "clarify", clarifier,
                task=task,
                missing_fields=", ".join(missing_for_llm) if missing_for_llm else "time, date",
                language=language,
            )
            return _finish(
                status=ReminderStatus.NEEDS_CLARIFICATION,
                clarification_question=_safe_str(clar_result.question),
                missing_fields=missing,
                confidence=0.7,
            )

        # Step 5: Validate resolved datetime
        if _is_empty(resolved):
            clar_result = _timed(
                "clarify", clarifier,
                task=task, missing_fields="time", language=language,
            )
            return _finish(
                status=ReminderStatus.NEEDS_CLARIFICATION,
                clarification_question=_safe_str(clar_result.question),
                missing_fields=["time"],
                confidence=0.6,
            )

        try:
            resolved_dt = datetime.fromisoformat(resolved)
        except (ValueError, TypeError):
            clar_result = _timed(
                "clarify", clarifier,
                task=task, missing_fields="date, time", language=language,
            )
            return _finish(
                status=ReminderStatus.NEEDS_CLARIFICATION,
                clarification_question=_safe_str(clar_result.question),
                missing_fields=["date", "time"],
                confidence=0.5,
            )

        # Step 6: Create reminder
        reminder = Reminder(
            reminder_id=f"r-{uuid.uuid4().hex[:8]}",
            what=task,
            when=resolved_dt,
            created_at=input_data.current_time,
        )

        # Step 7: Render confirmation message in user's language
        render_result = _timed(
            "render", renderer,
            task=task, when=resolved_dt.isoformat(), language=language,
        )

        return _finish(
            status=ReminderStatus.CREATED,
            reminder=reminder,
            confidence=0.9,
            rendered_message=_safe_str(render_result.message),
        )
    except Exception:
        fallback = _FALLBACK_CLARIFICATION.get(language, _FALLBACK_CLARIFICATION["en"])
        return _finish(
            status=ReminderStatus.NEEDS_CLARIFICATION,
            clarification_question=fallback,
            missing_fields=["task", "date", "time"],
            confidence=0.0,
        )