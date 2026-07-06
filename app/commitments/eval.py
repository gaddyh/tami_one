"""DSPy evaluation scaffolding for commitment extraction."""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from app.commitments.commitments_agent import CommitmentAgent
from app.commitments.models import Commitment


def commitment_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Compare predicted commitments against expected commitments.

    Uses normalized field-by-field equality: sorts both lists by
    (required_action, committed_party) and compares all fields.
    """
    expected = _normalize_for_comparison(example.expected_commitments)
    actual = _normalize_for_comparison(prediction.commitments)

    if actual == expected:
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
        display_table=True,
    )

    return evaluate(agent)


def _normalize_for_comparison(commitments: list[Commitment]) -> list[dict]:
    """Sort and serialize commitments for deterministic comparison."""
    dumped = [c.model_dump(mode="json") for c in commitments]
    dumped.sort(key=lambda c: (c.get("required_action", ""), c.get("committed_party") or ""))
    return dumped
