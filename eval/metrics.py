"""Metrics for commitment extraction evaluation."""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime

import dspy

from app.commitments.models import Commitment
from eval.llm_judge import JUDGE_FIELDS, judge_field_safe

_FIELDS = [
    "id",
    "chat_id",
    "chat_name",
    "committed_party",
    "required_action",
    "deadline",
    "context",
    "status",
    "notification",
]

# Time assigned to date-only deadlines (e.g. "2025-01-10" → "2025-01-10T14:00:00").
# Configurable via environment variable for flexibility.
_DATE_ONLY_DEFAULT_HOUR = int(os.environ.get("EVAL_DATE_ONLY_HOUR", "14"))

_REQUIRED_ACTION_F1_THRESHOLD = 0.75


def commitment_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    *,
    use_llm_judge: bool = False,
) -> float:
    """Compare predicted commitments against expected commitments.

    Returns the fraction of expected commitments that matched:
    matched_commitments / len(expected_commitments).

    - Empty expected + empty actual → 1.0 (correctly ignored)
    - Empty expected + non-empty actual → 0.0 (false positive)
    - Non-empty expected → matched / len(expected)
    """
    expected = example.expected_commitments
    actual = prediction.commitments

    if not expected:
        return 1.0 if not actual else 0.0

    expected_n = _normalize_for_comparison(expected)
    actual_n = _normalize_for_comparison(actual)

    matched = 0
    for idx, exp in enumerate(expected_n):
        act = actual_n[idx] if idx < len(actual_n) else {}
        field_match = True
        for field in _FIELDS:
            ev = exp.get(field, "—")
            av = act.get(field, "—")
            if field == "context":
                if not _word_overlap(str(ev), str(av)):
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    field_match = False
                    break
            elif field == "required_action":
                if _token_f1(str(ev), str(av)) < _REQUIRED_ACTION_F1_THRESHOLD:
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    field_match = False
                    break
            elif field == "deadline":
                if not _deadline_equal(ev, av):
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    field_match = False
                    break
            elif ev != av:
                field_match = False
                break
        if field_match:
            matched += 1

    return matched / len(expected_n)


def compare_commitments(
    expected: list[Commitment], actual: list[Commitment], *, use_llm_judge: bool = False,
) -> list[dict]:
    """Compare two commitment lists field-by-field.

    Returns a list of mismatch dicts, each with keys:
        index, field, expected, actual
    Empty list means perfect match.
    """
    expected_n = _normalize_for_comparison(expected)
    actual_n = _normalize_for_comparison(actual)

    mismatches: list[dict] = []

    if len(expected_n) != len(actual_n):
        max_len = max(len(expected_n), len(actual_n))
        for idx in range(max_len):
            exp = expected_n[idx] if idx < len(expected_n) else {}
            act = actual_n[idx] if idx < len(actual_n) else {}
            for field in _FIELDS:
                ev = exp.get(field, "—")
                av = act.get(field, "—")
                if ev != av:
                    mismatches.append(
                        {"index": idx, "field": field, "expected": ev, "actual": av}
                    )
        return mismatches

    for exp, act in zip(expected_n, actual_n):
        idx = expected_n.index(exp)
        for field in _FIELDS:
            ev = exp.get(field, "—")
            av = act.get(field, "—")
            if field == "context":
                if not _word_overlap(str(ev), str(av)):
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    mismatches.append(
                        {"index": idx, "field": field, "expected": ev, "actual": av}
                    )
            elif field == "required_action":
                if _token_f1(str(ev), str(av)) < _REQUIRED_ACTION_F1_THRESHOLD:
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    mismatches.append(
                        {"index": idx, "field": field, "expected": ev, "actual": av}
                    )
            elif field == "deadline":
                if not _deadline_equal(ev, av):
                    if use_llm_judge and judge_field_safe(field, str(ev), str(av)):
                        continue
                    mismatches.append(
                        {"index": idx, "field": field, "expected": ev, "actual": av}
                    )
            elif ev != av:
                mismatches.append(
                    {"index": idx, "field": field, "expected": ev, "actual": av}
                )

    return mismatches


def act_vs_ignore_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Check whether the agent correctly decided to act or ignore.

    - Expected empty + actual empty → 1.0 (correctly ignored)
    - Expected empty + actual non-empty → 0.0 (false positive)
    - Expected non-empty + actual empty → 0.0 (false negative)
    - Expected non-empty + actual non-empty → 1.0 (correctly acted)
    """
    expected_empty = len(example.expected_commitments) == 0
    actual_empty = len(prediction.commitments) == 0

    if expected_empty == actual_empty:
        return 1.0

    return 0.0


def _normalize_for_comparison(commitments: list[Commitment]) -> list[dict]:
    """Sort and serialize commitments for deterministic comparison.

    String fields are lowercased so case differences don't count as mismatches.
    """
    dumped = [c.model_dump(mode="json") for c in commitments]
    for d in dumped:
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = v.lower()
    dumped.sort(key=lambda c: (c.get("required_action", ""), c.get("committed_party") or ""))
    return dumped


def _normalize_deadline(value: str | None) -> str | None:
    """Normalize an ISO 8601 deadline for comparison.

    Date-only strings (e.g. '2025-01-10') are treated as 14:00 local time.
    Timezone offsets are stripped (compared as naive local times).
    Non-ISO strings are returned lowercased for backward compatibility.
    """
    if value is None or value == "—" or not str(value).strip():
        return None
    s = str(value).strip()
    # _normalize_for_comparison lowercases all strings, so try both
    # original and uppercased to handle lowercase 't'/'z' in ISO 8601.
    for candidate in (s, s.upper()):
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(candidate, fmt)
                if fmt == "%Y-%m-%d":
                    dt = dt.replace(hour=_DATE_ONLY_DEFAULT_HOUR)
                return dt.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
    return s.lower()


def _deadline_equal(expected: str | None, actual: str | None) -> bool:
    """Compare two deadline values with ISO 8601 normalization."""
    return _normalize_deadline(expected) == _normalize_deadline(actual)


def _word_overlap(expected: str, actual: str, threshold: float = 0.8) -> bool:
    """Check that at least `threshold` fraction of expected words appear in actual.

    Punctuation is stripped so quotes and other marks don't affect matching.
    """
    if not expected.strip():
        return not actual.strip()
    exp_words = set(re.findall(r"\w+", expected.lower()))
    act_words = set(re.findall(r"\w+", actual.lower()))
    if not exp_words:
        return True
    overlap = exp_words & act_words
    return len(overlap) / len(exp_words) >= threshold


def _token_f1(expected: str, actual: str) -> float:
    """Token-level F1 score between two strings.

    Tokens are lowercased word matches. Returns 1.0 for identical strings,
    0.0 for no overlap.
    """
    exp_tokens = re.findall(r"\w+", expected.lower())
    act_tokens = re.findall(r"\w+", actual.lower())
    if not exp_tokens and not act_tokens:
        return 1.0
    if not exp_tokens or not act_tokens:
        return 0.0

    exp_counts = Counter(exp_tokens)
    act_counts = Counter(act_tokens)

    common = exp_counts & act_counts
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(act_tokens)
    recall = num_common / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)
