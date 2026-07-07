"""Tests for eval.llm_judge — LLM-as-judge for semantic field comparison."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.commitments.models import Commitment, CommitmentStatus
from eval.llm_judge import (
    JUDGE_FIELDS,
    judge_field,
    judge_field_safe,
    reset_cache,
    get_cache_size,
)
from eval.metrics import compare_commitments, commitment_metric


# ── Unit tests for llm_judge module ──


def setup_function():
    """Reset cache before each test."""
    reset_cache()


def test_judge_fields_set():
    assert JUDGE_FIELDS == frozenset({"required_action", "deadline", "context"})
    assert "committed_party" not in JUDGE_FIELDS
    assert "status" not in JUDGE_FIELDS


def test_judge_field_caches_results():
    """Same (field, expected, actual) should only call the LLM once."""
    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        result1 = judge_field("required_action", "pay the invoice", "settle the invoice")
        result2 = judge_field("required_action", "pay the invoice", "settle the invoice")

    assert result1 is True
    assert result2 is True
    assert mock_predictor.return_value.call_count == 1  # cached on second call


def test_judge_field_different_inputs_not_cached():
    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        judge_field("required_action", "pay the invoice", "settle the invoice")
        judge_field("required_action", "send the docs", "send the documents")

    assert mock_predictor.return_value.call_count == 2


def test_judge_field_safe_returns_false_on_error():
    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(side_effect=RuntimeError("LLM error"))
        result = judge_field_safe("required_action", "pay", "settle")

    assert result is False


def test_reset_cache_clears_entries():
    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        judge_field("required_action", "pay", "settle")
        assert get_cache_size() == 1
        reset_cache()
        assert get_cache_size() == 0


# ── Integration tests for metrics with use_llm_judge ──


def _make_commitment(**kwargs) -> Commitment:
    defaults = dict(
        chat_id="test",
        required_action="Pay the invoice",
        deadline="Friday",
        context="I will pay by Friday",
        status=CommitmentStatus.WAITING,
    )
    defaults.update(kwargs)
    return Commitment(**defaults)


def test_compare_commitments_without_llm_judge():
    """Deterministic mode: verb synonym should fail (token-F1 < 0.75)."""
    expected = [_make_commitment(required_action="Pay the invoice")]
    actual = [_make_commitment(required_action="Settle the invoice")]

    mismatches = compare_commitments(expected, actual, use_llm_judge=False)
    assert any(m["field"] == "required_action" for m in mismatches)


def test_compare_commitments_with_llm_judge_match():
    """LLM judge mode: verb synonym should pass when LLM says equivalent."""
    expected = [_make_commitment(required_action="Pay the invoice")]
    actual = [_make_commitment(required_action="Settle the invoice")]

    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        mismatches = compare_commitments(expected, actual, use_llm_judge=True)

    assert not any(m["field"] == "required_action" for m in mismatches)


def test_compare_commitments_with_llm_judge_no_match():
    """LLM judge mode: genuinely different actions should still fail."""
    expected = [_make_commitment(required_action="Pay the invoice")]
    actual = [_make_commitment(required_action="Send the documents")]

    mock_pred = MagicMock()
    mock_pred.equivalent = False

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        mismatches = compare_commitments(expected, actual, use_llm_judge=True)

    assert any(m["field"] == "required_action" for m in mismatches)


def test_compare_commitments_llm_judge_only_for_semantic_fields():
    """LLM judge should NOT be called for non-semantic fields like committed_party."""
    expected = [_make_commitment(committed_party="Alice", required_action="Pay the invoice")]
    actual = [_make_commitment(committed_party="Bob", required_action="Pay the invoice")]

    with patch("eval.llm_judge.judge_field_safe") as mock_judge:
        mock_judge.return_value = True
        compare_commitments(expected, actual, use_llm_judge=True)

    # Should not be called for committed_party
    called_fields = [call.args[0] for call in mock_judge.call_args_list]
    assert "committed_party" not in called_fields


def test_compare_commitments_deadline_llm_judge():
    """Deadline mismatch should be rescued by LLM judge when equivalent."""
    expected = [_make_commitment(deadline="by Friday")]
    actual = [_make_commitment(deadline="Friday")]

    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        mismatches = compare_commitments(expected, actual, use_llm_judge=True)

    assert not any(m["field"] == "deadline" for m in mismatches)


def test_commitment_metric_with_llm_judge():
    """commitment_metric should improve when LLM judge rescues semantic matches."""
    expected = [_make_commitment(required_action="Pay the invoice", deadline="by Friday")]
    actual = [_make_commitment(required_action="Settle the invoice", deadline="Friday")]

    # Without LLM judge: should be 0.0 (both fields mismatch)
    import dspy
    ex = dspy.Example(expected_commitments=expected, messages="")
    pred = dspy.Prediction(commitments=actual)

    score_without = commitment_metric(ex, pred, use_llm_judge=False)
    assert score_without == 0.0

    # With LLM judge: should be 1.0 (both fields rescued)
    reset_cache()
    mock_pred = MagicMock()
    mock_pred.equivalent = True

    with patch("eval.llm_judge._get_predictor") as mock_predictor:
        mock_predictor.return_value = MagicMock(return_value=mock_pred)
        score_with = commitment_metric(ex, pred, use_llm_judge=True)

    assert score_with == 1.0
