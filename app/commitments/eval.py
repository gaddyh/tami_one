"""DSPy evaluation scaffolding for commitment extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

import dspy

from app.commitments.commitments_agent import CommitmentAgent
from app.commitments.models import Commitment

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


def commitment_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Compare predicted commitments against expected commitments.

    Uses normalized field-by-field comparison: sorts both lists by
    (required_action, committed_party) and compares all fields.
    The 'context' field uses a word-overlap threshold (80% of expected
    words must appear in actual) instead of exact match.
    """
    mismatches = compare_commitments(
        example.expected_commitments, prediction.commitments
    )
    return 1.0 if not mismatches else 0.0


def compare_commitments(
    expected: list[Commitment], actual: list[Commitment]
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


def make_example(
    *,
    chat_id: str,
    chat_name: str | None,
    existing_commitments_json: str,
    messages: str,
    expected_commitments: list[Commitment],
) -> dspy.Example:
    return dspy.Example(
        chat_id=chat_id,
        chat_name=chat_name,
        existing_commitments_json=existing_commitments_json,
        messages=messages,
        expected_commitments=expected_commitments,
    ).with_inputs(
        "chat_id",
        "chat_name",
        "existing_commitments_json",
        "messages",
    )


def build_devset(devset_path: Path | None = None) -> list[dspy.Example]:
    """Load devset examples from a JSON file.

    Each JSON entry has: chat_id, chat_name, existing_commitments_json,
    messages, expected_commitments (list of commitment dicts).
    """
    if devset_path is None:
        devset_path = Path(__file__).parent.parent.parent / "tests" / "evals" / "devset.json"

    with open(devset_path) as f:
        raw = json.load(f)

    examples: list[dspy.Example] = []
    for entry in raw:
        expected = [Commitment.model_validate(c) for c in entry["expected_commitments"]]
        examples.append(
            make_example(
                chat_id=entry["chat_id"],
                chat_name=entry.get("chat_name"),
                existing_commitments_json=entry.get("existing_commitments_json", "[]"),
                messages=entry["messages"],
                expected_commitments=expected,
            )
        )

    return examples


def run_evaluation(
    devset: list[dspy.Example] | None = None,
    agent: dspy.Module | None = None,
    *,
    display_table: bool = True,
) -> float:
    """Run dspy.Evaluate on the devset and return the score."""
    if devset is None:
        devset = build_devset()
    if agent is None:
        agent = CommitmentAgent()

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=commitment_metric,
        num_threads=8,
        display_progress=True,
        display_table=display_table,
    )

    return evaluate(agent)


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
