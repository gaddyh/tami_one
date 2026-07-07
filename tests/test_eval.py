"""Tests for the DSPy evaluation scaffolding."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest
import yaml

from eval.dataset import (
    build_devset,
    make_example,
)
from eval.metrics import (
    act_vs_ignore_metric,
    commitment_metric,
)
from app.commitments.models import Commitment

_DATA_DIR = Path(__file__).parent / "evals" / "data"


def _count_yaml_examples() -> int:
    """Count total examples across all category YAML files (excluding challenge)."""
    category_files = [
        "act_vs_ignore.yaml",
        "args_party.yaml",
        "args_deadline.yaml",
        "args_required_action.yaml",
        "lifecycle_update_vs_new.yaml",
        "lifecycle_completion.yaml",
        "cardinality.yaml",
    ]
    total = 0
    for filename in category_files:
        with (_DATA_DIR / filename).open() as f:
            total += len(yaml.safe_load(f))
    return total


def _count_yaml_dev_examples() -> int:
    """Count examples that hash to dev split."""
    import hashlib
    layer_map = {
        "act_vs_ignore": 1, "args_party": 2, "args_deadline": 2,
        "args_required_action": 2, "lifecycle_update_vs_new": 3,
        "lifecycle_completion": 3, "cardinality": 4,
    }
    category_files = [
        "act_vs_ignore.yaml", "args_party.yaml", "args_deadline.yaml",
        "args_required_action.yaml", "lifecycle_update_vs_new.yaml",
        "lifecycle_completion.yaml", "cardinality.yaml",
    ]
    dev_count = 0
    for filename in category_files:
        with (_DATA_DIR / filename).open() as f:
            for raw in yaml.safe_load(f):
                eid = f"{raw['category']}/{raw['scenario']}"
                if raw.get("split"):
                    split = raw["split"]
                else:
                    layer = layer_map.get(raw.get("category", ""), 3)
                    h = hashlib.sha256(eid.encode()).hexdigest()
                    bucket = int(h[:8], 16) % 100
                    if layer == 1: split = "train" if bucket < 60 else ("dev" if bucket < 80 else "test")
                    elif layer == 2: split = "train" if bucket < 50 else ("dev" if bucket < 75 else "test")
                    elif layer == 3: split = "train" if bucket < 35 else ("dev" if bucket < 65 else "test")
                    elif layer == 4: split = "train" if bucket < 25 else ("dev" if bucket < 50 else "test")
                    else: split = "test"
                if split == "dev":
                    dev_count += 1
    return dev_count


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
            deadline="2025-01-10",
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
            deadline="2025-01-10",
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
        current_datetime="2025-01-06T10:00:00Z",
        existing_commitments_json="[]",
        messages="Alice: hello",
        expected_commitments=[],
    )

    assert example.chat_id == "chat@c.us"
    assert example.chat_name == "Test Chat"
    assert example.current_datetime == "2025-01-06T10:00:00Z"
    assert example.existing_commitments_json == "[]"
    assert example.messages == "Alice: hello"
    assert example.expected_commitments == []


def test_make_example_with_inputs_marked():
    example = make_example(
        chat_id="chat@c.us",
        chat_name="Test Chat",
        current_datetime="2025-01-06T10:00:00Z",
        existing_commitments_json="[]",
        messages="Alice: hello",
        expected_commitments=[],
    )

    input_keys = set(example.inputs().keys())
    assert input_keys == {"chat_id", "chat_name", "current_datetime", "existing_commitments_json", "messages"}


def test_build_devset_loads_all_examples():
    devset_path = Path(__file__).parent / "evals" / "devset.json"
    examples = build_devset(devset_path)

    expected_count = _count_yaml_dev_examples()
    assert len(examples) == expected_count
    for ex in examples:
        assert isinstance(ex, dspy.Example)
        assert hasattr(ex, "chat_id")
        assert hasattr(ex, "messages")
        assert hasattr(ex, "expected_commitments")
        assert hasattr(ex, "difficulty")
        assert hasattr(ex, "current_datetime")


def test_build_devset_examples_have_inputs():
    devset_path = Path(__file__).parent / "evals" / "devset.json"
    examples = build_devset(devset_path)

    for ex in examples:
        input_keys = set(ex.inputs().keys())
        assert "chat_id" in input_keys
        assert "messages" in input_keys
        assert "current_datetime" in input_keys


def test_validators_pass_on_current_yaml():
    """All validators should pass against the current YAML files."""
    import importlib.util
    gen_path = Path(__file__).parent / "evals" / "generate_devset.py"
    spec = importlib.util.spec_from_file_location("generate_devset", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = mod.load_raw_examples()
    challenge = mod.load_raw_challenge_examples()
    split_map = mod._load_raw_split_map(raw)
    errors, _ = mod.run_validators(raw, challenge, split_map)
    assert errors == [], f"Validation errors: {errors}"


def test_deterministic_splits():
    """Same YAML input should produce same split assignment across runs."""
    import importlib.util
    gen_path = Path(__file__).parent / "evals" / "generate_devset.py"
    spec = importlib.util.spec_from_file_location("generate_devset", gen_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = mod.load_raw_examples()
    split_map_1 = mod._load_raw_split_map(raw)
    split_map_2 = mod._load_raw_split_map(raw)
    assert split_map_1 == split_map_2


def test_cardinality_category_in_pool():
    """Cardinality examples should exist in the YAML pool."""
    cardinality_path = _DATA_DIR / "cardinality.yaml"
    assert cardinality_path.exists()
    with cardinality_path.open() as f:
        examples = yaml.safe_load(f)
    assert len(examples) >= 12, f"Expected >=12 cardinality examples, got {len(examples)}"
    for ex in examples:
        assert ex["category"] == "cardinality"
        assert len(ex.get("expected_commitments", [])) >= 2, (
            f"Cardinality example {ex['scenario']} should have 2+ expected commitments"
        )
