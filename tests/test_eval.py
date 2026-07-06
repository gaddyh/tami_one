"""Tests for the DSPy evaluation scaffolding."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest

from app.commitments.eval import (
    act_vs_ignore_metric,
    build_devset,
    commitment_metric,
    make_example,
)
from app.commitments.models import Commitment


def test_act_vs_ignore_correctly_ignored():
    example = dspy.Example(expected_commitments=[]).with_inputs()
    prediction = dspy.Prediction(commitments=[])
    assert act_vs_ignore_metric(example, prediction) == 1.0


def test_act_vs_ignore_false_positive():
    example = dspy.Example(expected_commitments=[]).with_inputs()
    prediction = dspy.Prediction(commitments=[
        Commitment(
            id=None, chat_id="c@c.us", required_action="test", context="test",
        ),
    ])
    assert act_vs_ignore_metric(example, prediction) == 0.0


def test_act_vs_ignore_false_negative():
    expected = [
        Commitment(
            id=None, chat_id="c@c.us", required_action="test", context="test",
        ),
    ]
    example = dspy.Example(expected_commitments=expected).with_inputs()
    prediction = dspy.Prediction(commitments=[])
    assert act_vs_ignore_metric(example, prediction) == 0.0


def test_act_vs_ignore_correctly_acted():
    expected = [
        Commitment(
            id=None, chat_id="c@c.us", required_action="test", context="test",
        ),
    ]
    example = dspy.Example(expected_commitments=expected).with_inputs()
    prediction = dspy.Prediction(commitments=expected)
    assert act_vs_ignore_metric(example, prediction) == 1.0


def test_commitment_metric_match():
    expected = [
        Commitment(
            id=None,
            chat_id="chat@c.us",
            committed_party="Alice",
            required_action="Send report",
            deadline="Friday",
            context="Alice will send the report by Friday",
            status="waiting",
        ),
    ]
    example = dspy.Example(expected_commitments=expected).with_inputs()
    prediction = dspy.Prediction(commitments=expected)

    score = commitment_metric(example, prediction)
    assert score == 1.0


def test_commitment_metric_no_match():
    expected = [
        Commitment(
            id=None,
            chat_id="chat@c.us",
            committed_party="Alice",
            required_action="Send report",
            deadline="Friday",
            context="Alice will send the report by Friday",
            status="waiting",
        ),
    ]
    actual = [
        Commitment(
            id=None,
            chat_id="chat@c.us",
            committed_party="Bob",
            required_action="Send invoice",
            deadline=None,
            context="Bob will send the invoice",
            status="waiting",
        ),
    ]
    example = dspy.Example(expected_commitments=expected).with_inputs()
    prediction = dspy.Prediction(commitments=actual)

    score = commitment_metric(example, prediction)
    assert score == 0.0


def test_commitment_metric_empty_match():
    example = dspy.Example(expected_commitments=[]).with_inputs()
    prediction = dspy.Prediction(commitments=[])

    score = commitment_metric(example, prediction)
    assert score == 1.0


def test_commitment_metric_order_independent():
    c1 = Commitment(
        id=None,
        chat_id="chat@c.us",
        committed_party="Alice",
        required_action="Send report",
        context="context a",
    )
    c2 = Commitment(
        id=None,
        chat_id="chat@c.us",
        committed_party="Bob",
        required_action="Send invoice",
        context="context b",
    )
    example = dspy.Example(expected_commitments=[c1, c2]).with_inputs()
    prediction = dspy.Prediction(commitments=[c2, c1])

    score = commitment_metric(example, prediction)
    assert score == 1.0


def test_make_example_has_correct_inputs():
    example = make_example(
        chat_id="chat@c.us",
        chat_name="Test Chat",
        existing_commitments_json="[]",
        messages="Alice: hello",
        expected_commitments=[],
    )

    assert example.chat_id == "chat@c.us"
    assert example.chat_name == "Test Chat"
    assert example.existing_commitments_json == "[]"
    assert example.messages == "Alice: hello"
    assert example.expected_commitments == []


def test_make_example_with_inputs_marked():
    example = make_example(
        chat_id="chat@c.us",
        chat_name="Test Chat",
        existing_commitments_json="[]",
        messages="Alice: hello",
        expected_commitments=[],
    )

    input_keys = set(example.inputs().keys())
    assert input_keys == {"chat_id", "chat_name", "existing_commitments_json", "messages"}


def test_build_devset_loads_all_examples():
    devset_path = Path(__file__).parent / "evals" / "devset.json"
    examples = build_devset(devset_path)

    assert len(examples) == 10
    for ex in examples:
        assert isinstance(ex, dspy.Example)
        assert hasattr(ex, "chat_id")
        assert hasattr(ex, "messages")
        assert hasattr(ex, "expected_commitments")


def test_build_devset_examples_have_inputs():
    devset_path = Path(__file__).parent / "evals" / "devset.json"
    examples = build_devset(devset_path)

    for ex in examples:
        input_keys = set(ex.inputs().keys())
        assert "chat_id" in input_keys
        assert "messages" in input_keys
