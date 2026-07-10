from __future__ import annotations

import dspy


class ClassifyIntent(dspy.Signature):
    """Classify the intent of a user message in the context of a reminder conversation.

    A reminder request asks the agent to remind the user about something later.
    This includes: "remind me to...", "don't forget to...", "set an alarm for 7am",
    "set a timer for 10 minutes", and similar time-based trigger requests.
    A clarification answer is a short response to a previous agent question (e.g. just "3pm").
    Anything else is irrelevant, including: weather questions, jokes, general questions,
    or commands that are not about reminders or time-based triggers.

    Detect the language of the user's message (ISO 639-1 code, e.g. 'en', 'he', 'es', 'fr').
    """

    text: str = dspy.InputField()
    prior_context: str = dspy.InputField(desc="Prior conversation turns, or empty")
    intent: str = dspy.OutputField(desc="reminder_request, clarification_answer, or irrelevant")
    language: str = dspy.OutputField(desc="ISO 639-1 language code of the user's message, e.g. 'en', 'he', 'es'")
    confidence: float = dspy.OutputField()


class ExtractReminder(dspy.Signature):
    """Extract task, date, and time from a reminder request or clarification answer.

    If the message is a clarification answer (e.g. just "3pm"), use the prior_context
    to determine the task and any previously specified date.
    Resolve relative dates (today, tomorrow) using current_time.

    DATE RESOLUTION RULES:
    - "this <weekday>" means the nearest occurrence of that weekday, including today.
      If today IS that weekday, "this friday" = today.
    - "next <weekday>" means the nearest upcoming occurrence of that weekday.
      E.g. if today is Friday July 10, "next monday" = July 13 (the nearest Monday).
      If today is Friday and user says "next friday", that means 7 days from now (July 17),
      NOT today.
    - "<weekday>" alone (e.g. "on friday") means the nearest upcoming occurrence,
      excluding today. If today is Friday, "on friday" = next Friday (7 days from now).

    RELATIVE TIME: Handle expressions like "in an hour", "in 30 minutes", "in 2 hours",
    "עוד שעה" (in an hour), "בעוד חצי שעה" (in half an hour), etc.
    These specify a time relative to current_time. Set time_raw to the expression as written
    and compute the resolved_datetime accordingly.

    TIME EXPRESSIONS: Recognize all common time formats including:
    - 12-hour: "3pm", "3 pm", "9am", "9:00 am", "3:30 pm"
    - 24-hour: "15:00", "09:00", "11:15"
    - Named: "noon" (12:00), "midnight" (00:00), "morning", "afternoon", "evening"
    - Relative: "in an hour", "in 30 minutes", "in 2 hours"
    Any of these counts as a time expression. Do NOT leave time_raw empty if the user
    said "at noon", "at midnight", or any similar expression.

    CRITICAL: Only extract what the user EXPLICITLY stated. Do NOT infer or default
    any field. If the user did not mention a date, leave date_raw empty — do NOT
    assume "today". If the user did not mention a time, leave time_raw empty.
    Only return resolved_datetime if BOTH date and time are explicitly present.
    A relative time expression like "in an hour" counts as specifying BOTH date and time
    (since it resolves to a specific datetime).

    TASK EXTRACTION: The task is the action only, without any leading words like
    "remind me to", "to", "don't forget to", "make sure I", or "that I need to".
    For example: "remind me to call dad" -> task="call dad" (NOT "remind me to call dad"
    or "to call dad"). If no task/action is specified, leave task EMPTY.
    A message like "remind me tomorrow at 3pm" has NO task — the user said when but not what.
    Do NOT use "remind me" or "don't forget" as the task.
    """

    text: str = dspy.InputField()
    prior_context: str = dspy.InputField(desc="Prior conversation turns, or empty")
    current_time: str = dspy.InputField(desc="ISO datetime for resolving relative dates and times")
    task: str = dspy.OutputField(desc="The action to be reminded about, or empty if not a reminder")
    date_raw: str = dspy.OutputField(
        desc="Date expression as explicitly written by the user (e.g. 'tomorrow', 'today', 'Monday'). "
        "EMPTY if the user did not mention any date. Do NOT default to today."
    )
    time_raw: str = dspy.OutputField(
        desc="Time expression as explicitly written by the user (e.g. '3pm', '15:00', 'noon', 'midnight', '6pm', 'in an hour', 'עוד שעה'). "
        "EMPTY only if the user did not mention ANY time. 'at noon' IS a time. 'at 6pm' IS a time."
    )
    resolved_datetime: str = dspy.OutputField(
        desc="ISO 8601 datetime if both date and time are explicitly present (including relative times like 'in an hour'), or empty"
    )


class GenerateClarification(dspy.Signature):
    """Generate a natural language clarification question for missing reminder fields.

    The question should reference the task and ask specifically for the missing information.
    The question MUST be in the same language as the user's message.
    """

    task: str = dspy.InputField()
    missing_fields: str = dspy.InputField(desc="Comma-separated list: time, date, or both")
    language: str = dspy.InputField(desc="ISO 639-1 language code for the response, e.g. 'en', 'he', 'es'")
    question: str = dspy.OutputField(
        desc="Natural language clarification question in the specified language, referencing the task"
    )


class RenderReminder(dspy.Signature):
    """Generate a human-readable confirmation message for a created reminder.

    The message MUST be in the specified language.
    """

    task: str = dspy.InputField()
    when: str = dspy.InputField(desc="ISO 8601 datetime of the reminder")
    language: str = dspy.InputField(desc="ISO 639-1 language code, e.g. 'en', 'he', 'es'")
    message: str = dspy.OutputField(
        desc="Natural language confirmation message in the specified language"
    )
