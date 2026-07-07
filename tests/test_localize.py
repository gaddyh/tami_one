"""Tests for eval.localize — verifies deterministic classification of fake failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.localize import (
    LocalizedFailure,
    build_report,
    localize,
)

CASES_DIR = Path(__file__).parent / "evals" / "localize_cases"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Per-category tests: verify every fake failure gets the expected root cause ──


@pytest.mark.parametrize(
    "filename,expected_root_cause",
    [
        ("required_action_normalization.jsonl", "required_action_normalization"),
        ("deadline_normalization.jsonl", "deadline_normalization"),
        ("context_metric_noise.jsonl", "context_metric_noise"),
        ("update_vs_new_matching.jsonl", "update_vs_new_matching"),
    ],
)
def test_field_mismatch_root_causes(filename: str, expected_root_cause: str) -> None:
    failures = _load_jsonl(CASES_DIR / filename)
    localized = localize(failures)
    for lf in localized:
        assert lf.root_cause == expected_root_cause, (
            f"Expected {expected_root_cause} for scenario {lf.scenario}, got {lf.root_cause}"
        )


def test_false_positive_root_causes() -> None:
    failures = _load_jsonl(CASES_DIR / "policy_false_positive.jsonl")
    localized = localize(failures)
    # conditional_promise_untriggered → over_extraction_policy
    # alice_asks_bob_bob_refuses → over_extraction_policy
    for lf in localized:
        assert lf.root_cause == "over_extraction_policy", (
            f"Expected over_extraction_policy for {lf.scenario}, got {lf.root_cause}"
        )


def test_false_negative_root_causes() -> None:
    failures = _load_jsonl(CASES_DIR / "policy_false_negative.jsonl")
    localized = localize(failures)
    # hedged_commitment_unclear → under_extraction_policy
    # party_implied_by_role → party_resolution (third party)
    root_causes = {lf.scenario: lf.root_cause for lf in localized}
    assert root_causes["hedged_commitment_unclear"] == "under_extraction_policy"
    assert root_causes["party_implied_by_role"] == "party_resolution"


# ── Subcause tests ──


def test_required_action_subcauses() -> None:
    failures = _load_jsonl(CASES_DIR / "required_action_normalization.jsonl")
    localized = localize(failures)
    subcauses = {lf.scenario: lf.subcause for lf in localized}
    # settle vs pay: different verbs, same word count → verb_synonym
    assert subcauses["settle_invoice_vs_pay_invoice"] == "verb_synonym"
    # draft vs prepare: different verbs, same word count → verb_synonym
    assert subcauses["draft_report_vs_prepare_report"] == "verb_synonym"
    # send over the docs vs send the documents: more words in actual → too_specific
    assert subcauses["send_over_docs_vs_send_documents"] == "too_specific"
    # give the supplier a call vs call the supplier: more words in actual → too_specific
    assert subcauses["give_call_vs_call_supplier"] == "too_specific"


def test_deadline_subcauses() -> None:
    failures = _load_jsonl(CASES_DIR / "deadline_normalization.jsonl")
    localized = localize(failures)
    subcauses = {lf.scenario: lf.subcause for lf in localized}
    assert subcauses["deadline_5pm_vs_by_5pm"] == "prefix_by"
    assert subcauses["deadline_1700_vs_by_1700"] == "prefix_by"
    assert subcauses["deadline_end_of_week_vs_by_end_of_week"] == "prefix_by"
    assert subcauses["deadline_relative_context"] == "event_based_deadline"


def test_update_vs_new_subcauses() -> None:
    failures = _load_jsonl(CASES_DIR / "update_vs_new_matching.jsonl")
    localized = localize(failures)
    subcauses = {lf.scenario: lf.subcause for lf in localized}
    assert subcauses["expected_new_actual_update"] == "expected_new_actual_update"
    assert subcauses["expected_update_actual_new"] == "expected_update_actual_new"


def test_false_positive_subcauses() -> None:
    failures = _load_jsonl(CASES_DIR / "policy_false_positive.jsonl")
    localized = localize(failures)
    subcauses = {lf.scenario: lf.subcause for lf in localized}
    assert subcauses["conditional_promise_untriggered"] == "conditional_commitment"
    assert subcauses["alice_asks_bob_bob_refuses"] == "refusal_not_commitment"


def test_false_negative_subcauses() -> None:
    failures = _load_jsonl(CASES_DIR / "policy_false_negative.jsonl")
    localized = localize(failures)
    subcauses = {lf.scenario: lf.subcause for lf in localized}
    assert subcauses["hedged_commitment_unclear"] == "hedged_commitment"
    assert subcauses["party_implied_by_role"] == "third_party_obligation"


# ── Mixed failures: golden summary ──


def test_mixed_failures_root_cause_counts() -> None:
    failures = _load_jsonl(CASES_DIR / "mixed_failures.jsonl")
    localized = localize(failures)
    report = build_report(localized)

    expected_counts = {
        "required_action_normalization": 4,
        "deadline_normalization": 4,
        "context_metric_noise": 2,
        "update_vs_new_matching": 2,
        "under_extraction_policy": 1,
        "party_resolution": 1,
        "over_extraction_policy": 2,
        "lifecycle_policy": 2,
    }
    assert dict(report.root_cause_counts) == expected_counts, (
        f"Root cause counts mismatch: {dict(report.root_cause_counts)} != {expected_counts}"
    )


def test_mixed_failures_top_actions() -> None:
    failures = _load_jsonl(CASES_DIR / "mixed_failures.jsonl")
    localized = localize(failures)
    report = build_report(localized)

    # Top 2 should be the highest priority (impact * confidence / cost)
    top_2_causes = [a[0] for a in report.top_actions[:2]]
    # required_action: 4 * 0.9 / 2 = 1.8
    # context_metric_noise: 2 * 0.9 / 1 = 1.8
    # deadline: 4 * 0.9 / 2 = 1.8
    # All three are tied at 1.8 — required_action must be in top 2 (highest impact)
    assert "required_action_normalization" in top_2_causes
    # One of the other two tied causes should also be in top 2
    assert top_2_causes[1] in ("context_metric_noise", "deadline_normalization")


def test_mixed_failures_total_count() -> None:
    failures = _load_jsonl(CASES_DIR / "mixed_failures.jsonl")
    localized = localize(failures)
    assert len(localized) == 18


def test_all_failures_have_subcause() -> None:
    failures = _load_jsonl(CASES_DIR / "mixed_failures.jsonl")
    localized = localize(failures)
    for lf in localized:
        assert lf.subcause != "", f"Empty subcause for {lf.scenario}"
        assert lf.repair_type != "", f"Empty repair_type for {lf.scenario}"


def test_repair_type_consistency() -> None:
    """Each root cause should always map to the same repair type."""
    failures = _load_jsonl(CASES_DIR / "mixed_failures.jsonl")
    localized = localize(failures)
    repair_map: dict[str, str] = {}
    for lf in localized:
        if lf.root_cause in repair_map:
            assert lf.repair_type == repair_map[lf.root_cause], (
                f"Inconsistent repair_type for {lf.root_cause}: "
                f"{lf.repair_type} vs {repair_map[lf.root_cause]}"
            )
        else:
            repair_map[lf.root_cause] = lf.repair_type
