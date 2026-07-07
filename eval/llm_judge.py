"""LLM-as-judge for semantic field comparison in commitment extraction evals.

When deterministic metrics (exact match, token-F1, word-overlap) fail on
semantic fields (required_action, deadline, context), this module provides
an LLM fallback that judges whether the expected and actual values are
semantically equivalent.

Usage is optional — the eval runner defaults to deterministic-only mode.
Enable with `--llm-judge` on the CLI.

Verdict persistence:
    save_verdicts(path)   — write all judge calls to JSONL
    load_verdicts(path)   — pre-populate cache from JSONL
    set_offline(True)     — cache-only mode, no LLM calls (miss raises JudgeCacheMiss)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import dspy

# Fields that warrant LLM judging when deterministic checks fail
JUDGE_FIELDS = frozenset({"required_action", "deadline", "context"})

# Versioning — bump when judge prompt or normalization logic changes
JUDGE_PROMPT_VERSION = "v1"
NORMALIZATION_VERSION = "deadline_iso_v1"


class JudgeCacheMiss(RuntimeError):
    """Raised when offline mode is active and a judge verdict is not in cache."""


# Module-level cache: (field, expected, actual) -> bool
# Reset per eval run via reset_cache()
_cache: dict[tuple[str, str, str], bool] = {}

# Log of every verdict made this run (for saving to disk)
_verdict_log: list[dict[str, Any]] = []

# When True, judge_field() only checks cache; misses raise JudgeCacheMiss
_offline_mode: bool = False

# Model name recorded in saved verdicts (set by eval_runner)
_judge_model: str = ""


def reset_cache() -> None:
    """Clear the judge cache and verdict log. Call once per eval run."""
    _cache.clear()
    _verdict_log.clear()
    _offline_mode = False


def get_cache_size() -> int:
    return len(_cache)


def set_offline_mode(enabled: bool) -> None:
    """When enabled, judge_field() only checks cache; misses raise JudgeCacheMiss."""
    global _offline_mode
    _offline_mode = enabled


def set_judge_model(model: str) -> None:
    """Record the model name for verdict persistence."""
    global _judge_model
    _judge_model = model


def get_verdict_log() -> list[dict[str, Any]]:
    """Return the log of all verdicts made this run."""
    return list(_verdict_log)


def save_verdicts(path: Path) -> None:
    """Write all verdicts from this run to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for v in _verdict_log:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")


def _make_cache_key(field: str, expected: str, actual: str) -> tuple[str, str, str]:
    """Build the deduplication key for judge verdicts."""
    return (field, expected.lower().strip(), actual.lower().strip())


def _make_cache_key_hash(field: str, expected: str, actual: str) -> str:
    """Build a stable hash for audit/logging purposes."""
    payload = json.dumps(
        {
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "field": field,
            "expected": expected.lower().strip(),
            "actual": actual.lower().strip(),
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def load_verdicts(path: Path) -> int:
    """Pre-populate cache from a JSONL file. Returns number loaded."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            key = _make_cache_key(v["field"], v["expected"], v["actual"])
            _cache[key] = v["verdict"]
            count += 1
    return count


class JudgeField(dspy.Signature):
    """Judge whether two values for a commitment field are semantically equivalent.

    Two values are equivalent if they convey the same meaning, even if worded
    differently. For example:
    - "settle the invoice" and "pay the invoice" are equivalent (required_action)
    - "2025-01-10" and "2025-01-10T00:00:00" are equivalent (deadline)
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

    Raises:
        JudgeCacheMiss: If offline mode is active and the verdict is not cached.
    """
    key = _make_cache_key(field, expected, actual)
    if key in _cache:
        return _cache[key]

    if _offline_mode:
        raise JudgeCacheMiss(
            f"Offline mode: no cached verdict for field={field!r} "
            f"expected={expected!r} actual={actual!r}"
        )

    predictor = _get_predictor()
    pred = predictor(
        field_name=field,
        expected_value=expected,
        actual_value=actual,
    )

    result = bool(pred.equivalent)
    _cache[key] = result

    # Compute normalized values for audit
    from eval.metrics import _normalize_deadline
    exp_normalized = _normalize_deadline(expected) if field == "deadline" else expected.lower().strip()
    act_normalized = _normalize_deadline(actual) if field == "deadline" else actual.lower().strip()

    _verdict_log.append({
        "cache_key": _make_cache_key_hash(field, expected, actual),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "field": field,
        "expected": expected,
        "actual": actual,
        "expected_normalized": exp_normalized,
        "actual_normalized": act_normalized,
        "verdict": result,
        "reasoning": getattr(pred, "reasoning", ""),
        "judge_model": _judge_model,
    })
    return result


def judge_field_safe(field: str, expected: str, actual: str) -> bool:
    """Like judge_field but returns False on any error except JudgeCacheMiss.

    JudgeCacheMiss is re-raised because it indicates a missing verdict in
    offline mode — a hard error that should fail the regrade, not be silenced.
    """
    try:
        return judge_field(field, expected, actual)
    except JudgeCacheMiss:
        raise
    except Exception:
        return False
