"""Dataset loading and example construction for commitment evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from app.commitments.models import Commitment


def make_example(
    *,
    chat_id: str,
    chat_name: str | None,
    existing_commitments_json: str,
    messages: str,
    expected_commitments: list[Commitment],
    category: str = "",
    scenario: str = "",
    difficulty: str = "",
) -> dspy.Example:
    return dspy.Example(
        chat_id=chat_id,
        chat_name=chat_name,
        existing_commitments_json=existing_commitments_json,
        messages=messages,
        expected_commitments=expected_commitments,
        category=category,
        scenario=scenario,
        difficulty=difficulty,
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
        devset_path = Path(__file__).parent.parent / "tests" / "evals" / "devset.json"

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
                category=entry.get("category", ""),
                scenario=entry.get("scenario", ""),
                difficulty=entry.get("difficulty", ""),
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
    from app.commitments.commitments_agent import CommitmentAgent

    if devset is None:
        devset = build_devset()
    if agent is None:
        agent = CommitmentAgent()

    from eval.metrics import commitment_metric

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=commitment_metric,
        num_threads=8,
        display_progress=True,
        display_table=display_table,
    )

    return evaluate(agent)
