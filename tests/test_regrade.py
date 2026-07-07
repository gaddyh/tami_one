"""Tests for frozen predictions and regrade workflow."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from eval.llm_judge import (
    JudgeCacheMiss,
    _cache,
    _make_cache_key,
    reset_cache,
    save_verdicts,
    load_verdicts,
    set_offline_mode,
    judge_field,
    get_verdict_log,
)


@pytest.fixture(autouse=True)
def _reset_judge_state():
    """Ensure clean judge state before and after each test."""
    reset_cache()
    yield
    reset_cache()


def _make_predictions_jsonl(tmpdir: Path, rows: list[dict]) -> Path:
    """Write a predictions.jsonl file and return its path."""
    path = tmpdir / "predictions.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


_MATCH_ROW = {
    "split": "dev",
    "example_id": "test/match",
    "category": "test",
    "scenario": "match",
    "difficulty": "easy",
    "inputs": {
        "chat_id": "c@c.us",
        "chat_name": "Test",
        "current_datetime": "2025-01-06T10:00:00Z",
        "existing_commitments_json": "[]",
        "messages": "Alice: I will send the report by Friday",
    },
    "expected_commitments": [
        {
            "id": None,
            "chat_id": "c@c.us",
            "chat_name": "Test",
            "committed_party": "Alice",
            "required_action": "Send the report",
            "deadline": "2025-01-10",
            "context": "Alice will send the report by Friday",
            "status": "waiting",
            "notification": "none",
        }
    ],
    "actual_commitments": [
        {
            "id": None,
            "chat_id": "c@c.us",
            "chat_name": "Test",
            "committed_party": "Alice",
            "required_action": "Send the report",
            "deadline": "2025-01-10",
            "context": "Alice will send the report by Friday",
            "status": "waiting",
            "notification": "none",
        }
    ],
}


def test_regrade_no_judge_matches_original_no_judge_score():
    """Regrading frozen predictions without judge should produce same failure count."""
    from scripts.regrade_predictions import _regrade_split, _load_predictions

    with TemporaryDirectory() as tmpdir:
        pred_path = _make_predictions_jsonl(Path(tmpdir), [_MATCH_ROW])
        rows = _load_predictions(pred_path)
        result = _regrade_split(rows, "dev", use_llm_judge=False)

        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tn"] == 0


def test_regrade_detects_mismatch():
    """Regrading should detect field mismatches in frozen predictions."""
    from scripts.regrade_predictions import _regrade_split, _load_predictions

    row = {
        **_MATCH_ROW,
        "example_id": "test/mismatch",
        "scenario": "mismatch",
        "actual_commitments": [
            {
                "id": None,
                "chat_id": "c@c.us",
                "chat_name": "Test",
                "committed_party": "Bob",
                "required_action": "Send invoice",
                "deadline": None,
                "context": "Bob will send the invoice",
                "status": "waiting",
                "notification": "none",
            }
        ],
    }

    with TemporaryDirectory() as tmpdir:
        pred_path = _make_predictions_jsonl(Path(tmpdir), [row])
        rows = _load_predictions(pred_path)
        result = _regrade_split(rows, "dev", use_llm_judge=False)

        assert result["tp"] == 1
        assert result["per_example"][0]["metric_score"] == 0.0


def test_regrade_false_negative():
    """Regrading should detect false negatives (expected non-empty, actual empty)."""
    from scripts.regrade_predictions import _regrade_split, _load_predictions

    row = {**_MATCH_ROW, "example_id": "test/fn", "scenario": "fn", "actual_commitments": []}

    with TemporaryDirectory() as tmpdir:
        pred_path = _make_predictions_jsonl(Path(tmpdir), [row])
        rows = _load_predictions(pred_path)
        result = _regrade_split(rows, "dev", use_llm_judge=False)

        assert result["fn"] == 1
        assert result["tp"] == 0


def test_judge_verdict_roundtrip():
    """Save and load judge verdicts — cache should match."""
    with TemporaryDirectory() as tmpdir:
        verdicts_path = Path(tmpdir) / "judge_verdicts.jsonl"

        # Manually populate cache and verdict log
        key = _make_cache_key("deadline", "2025-01-10", "2025-01-10T00:00:00")
        _cache[key] = True
        import eval.llm_judge as lj
        lj._verdict_log.append({
            "cache_key": lj._make_cache_key_hash("deadline", "2025-01-10", "2025-01-10T00:00:00"),
            "judge_prompt_version": lj.JUDGE_PROMPT_VERSION,
            "normalization_version": lj.NORMALIZATION_VERSION,
            "field": "deadline",
            "expected": "2025-01-10",
            "actual": "2025-01-10T00:00:00",
            "expected_normalized": "2025-01-10T14:00:00",
            "actual_normalized": "2025-01-10T00:00:00",
            "verdict": True,
            "reasoning": "Same date.",
            "judge_model": "test-model",
        })

        save_verdicts(verdicts_path)
        assert verdicts_path.exists()

        # Clear and reload
        reset_cache()
        count = load_verdicts(verdicts_path)
        assert count == 1

        # Cache should have the verdict
        assert key in _cache
        assert _cache[key] is True


def test_offline_judge_cache_miss_raises():
    """In offline mode, cache misses should raise JudgeCacheMiss, not return False."""
    set_offline_mode(True)

    # Populate one entry
    key = _make_cache_key("deadline", "2025-01-10", "2025-01-10T00:00:00")
    _cache[key] = True

    # Hit → True
    assert judge_field("deadline", "2025-01-10", "2025-01-10T00:00:00") is True

    # Miss → raises JudgeCacheMiss
    with pytest.raises(JudgeCacheMiss, match="Offline mode"):
        judge_field("deadline", "2025-01-07", "2025-01-08")


def test_regrade_does_not_call_agent():
    """Regrading should work without any agent instantiation or LLM calls."""
    from scripts.regrade_predictions import _regrade_split, _load_predictions

    with TemporaryDirectory() as tmpdir:
        pred_path = _make_predictions_jsonl(Path(tmpdir), [_MATCH_ROW])
        rows = _load_predictions(pred_path)
        # This should succeed without any DSPy LM configured
        result = _regrade_split(rows, "dev", use_llm_judge=False)
        assert result["n"] == 1
