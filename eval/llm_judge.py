"""LLM-as-judge for semantic field comparison in commitment extraction evals.

When deterministic metrics (exact match, token-F1, word-overlap) fail on
semantic fields (required_action, deadline, context), this module provides
an LLM fallback that judges whether the expected and actual values are
semantically equivalent.

Usage is optional — the eval runner defaults to deterministic-only mode.
Enable with `--llm-judge` on the CLI.
"""

from __future__ import annotations

from typing import Any

import dspy

# Fields that warrant LLM judging when deterministic checks fail
JUDGE_FIELDS = frozenset({"required_action", "deadline", "context"})

# Module-level cache: (field, expected, actual) -> bool
# Reset per eval run via reset_cache()
_cache: dict[tuple[str, str, str], bool] = {}


def reset_cache() -> None:
    """Clear the judge cache. Call once per eval run."""
    _cache.clear()


def get_cache_size() -> int:
    return len(_cache)


class JudgeField(dspy.Signature):
    """Judge whether two values for a commitment field are semantically equivalent.

    Two values are equivalent if they convey the same meaning, even if worded
    differently. For example:
    - "settle the invoice" and "pay the invoice" are equivalent (required_action)
    - "by Friday" and "Friday" are equivalent (deadline)
    - "Bob agreed to send the report" and "sure, I'll have it ready" are NOT
      equivalent unless the context makes clear they refer to the same thing

    Return equivalent=True only if a human annotator would accept both as
    matching. When in doubt, return equivalent=False.
    """

    field_name: str = dspy.InputField(desc="The commitment field being compared")
    expected_value: str = dspy.InputField(desc="The reference/expected value")
    actual_value: str = dspy.InputField(desc="The predicted/actual value")

    equivalent: bool = dspy.OutputField(desc="True if semantically equivalent, False otherwise")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the judgment")


_judge_predictor: dspy.Predict | None = None


def _get_predictor() -> dspy.Predict:
    global _judge_predictor
    if _judge_predictor is None:
        _judge_predictor = dspy.Predict(JudgeField)
    return _judge_predictor


def judge_field(field: str, expected: str, actual: str) -> bool:
    """Judge whether expected and actual are semantically equivalent.

    Uses an LLM via DSPy. Results are cached per (field, expected, actual) key.

    Args:
        field: The commitment field name (must be in JUDGE_FIELDS)
        expected: The reference value
        actual: The predicted value

    Returns:
        True if the LLM judges them semantically equivalent, False otherwise.
    """
    key = (field, expected.lower().strip(), actual.lower().strip())
    if key in _cache:
        return _cache[key]

    predictor = _get_predictor()
    pred = predictor(
        field_name=field,
        expected_value=expected,
        actual_value=actual,
    )

    result = bool(pred.equivalent)
    _cache[key] = result
    return result


def judge_field_safe(field: str, expected: str, actual: str) -> bool:
    """Like judge_field but returns False on any error (fail-safe)."""
    try:
        return judge_field(field, expected, actual)
    except Exception:
        return False
