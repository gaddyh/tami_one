"""Failure localization layer for commitment extraction evals.

Reads a failures.jsonl file from an eval run and assigns each failure a root
cause, subcause, repair type, and confidence. Outputs a ranked summary table
with suggested next actions based on priority = impact * confidence / cost.

Usage:
    python -m eval.localize runs/20260707_023551/failures.jsonl
    python -m eval.localize runs/20260707_023551/failures.jsonl --json
    python -m eval.localize runs/20260707_023551/failures.jsonl --md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

# Repair type costs (lower = cheaper to fix)
_REPAIR_COST: dict[str, int] = {
    "metric": 1,
    "postprocess": 2,
    "metric_or_postprocess": 2,
    "signature_rule": 3,
    "product_decision": 4,
}

# Root cause → (repair_type, top_repair_description)
_REPAIR_MAP: dict[str, tuple[str, str]] = {
    "required_action_normalization": ("metric_or_postprocess", "Normalize/match action semantically"),
    "deadline_normalization": ("metric_or_postprocess", "Canonicalize deadline phrases"),
    "context_metric_noise": ("metric", "Soften context match (word overlap, not exact)"),
    "update_vs_new_matching": ("postprocess", "Match by id/action; filter unchanged existing"),
    "party_resolution": ("signature_rule", "Add implied-party / third-party rules"),
    "over_extraction_policy": ("signature_rule", "Tighten act-vs-ignore boundary"),
    "under_extraction_policy": ("signature_rule", "Loosen extraction for hedged/implicit cases"),
    "lifecycle_policy": ("signature_rule", "Distinguish started/almost-done from done"),
    "multi_field_mismatch": ("postprocess", "Holistic commitment matching + normalization"),
}

# Subcause labels for display
_SUBCAUSE_LABELS: dict[str, dict[str, str]] = {
    "required_action_normalization": {
        "verb_synonym": "Action verb synonyms (settle→pay, draft→prepare)",
        "object_synonym": "Action object synonyms (docs→documents, ring→call)",
        "too_specific": "Action too specific vs expected",
        "too_generic": "Action too generic vs expected",
    },
    "deadline_normalization": {
        "prefix_by": "Missing/extra 'by' prefix",
        "time_format": "Time format mismatch (17:00 vs 5pm)",
        "vague_relative_phrase": "Relative deadline phrasing (end of week)",
        "event_based_deadline": "Event-based deadline (before the meeting)",
    },
    "context_metric_noise": {
        "paraphrase": "Context paraphrased but same meaning",
        "truncated": "Context truncated or abbreviated",
        "extra_context": "Context includes extra surrounding text",
    },
    "update_vs_new_matching": {
        "expected_new_actual_update": "Expected new commitment, agent updated existing",
        "expected_update_actual_new": "Expected update, agent created new",
        "unchanged_existing_leak": "Agent returned unchanged existing alongside new",
        "multiple_commitment_alignment": "Multiple commitments misaligned",
    },
    "under_extraction_policy": {
        "hedged_commitment": "Hedged/vague commitment not extracted",
        "third_party_obligation": "Third-party obligation not extracted",
        "group_obligation": "Group 'we need' not extracted",
        "external_waiting": "External party waiting not extracted",
    },
    "over_extraction_policy": {
        "conditional_commitment": "Conditional/untriggered promise extracted",
        "progress_not_completion": "Progress statement extracted as commitment",
        "refusal_not_commitment": "Refusal extracted as commitment",
    },
    "lifecycle_policy": {
        "started_not_done": "'Started' treated as done",
        "almost_done_not_done": "'Almost done' treated as done",
        "partial_completion": "Partial completion treated as full",
        "conditional_dismissal": "Conditional dismissal not handled",
    },
    "party_resolution": {
        "implied_party": "Implied party from context not resolved",
        "third_party_named": "Third-party name not extracted correctly",
        "party_handoff": "Party handoff not detected",
    },
}


@dataclass
class LocalizedFailure:
    root_cause: str
    subcause: str
    repair_type: str
    confidence: float
    impact: int = 1
    split: str = ""
    category: str = ""
    scenario: str = ""
    difficulty: str = ""
    error_type: str = ""
    mismatched_fields: list[str] = field(default_factory=list)
    messages: str = ""
    expected_commitments: list[dict] = field(default_factory=list)
    actual_commitments: list[dict] = field(default_factory=list)
    detail: str = ""


@dataclass
class LocalizationReport:
    root_cause_counts: Counter
    subcause_counts: Counter
    top_actions: list[tuple[str, float, str]]
    localized: list[LocalizedFailure]


def _has_unchanged_existing(failure: dict[str, Any]) -> bool:
    """Detect when actual includes an unchanged existing commitment alongside a new one."""
    existing = failure.get("existing_commitments", [])
    actual = failure.get("actual_commitments", [])
    if not existing or not actual:
        return False
    for ex in existing:
        for act in actual:
            if ex.get("id") and act.get("id") == ex.get("id"):
                if ex.get("required_action") == act.get("required_action"):
                    return True
    return False


def _detect_required_action_subcause(failure: dict[str, Any]) -> str:
    expected = failure.get("expected_commitments", [])
    actual = failure.get("actual_commitments", [])
    if not expected or not actual:
        return "verb_synonym"
    exp_action = (expected[0].get("required_action") or "").lower()
    act_action = (actual[0].get("required_action") or "").lower()
    exp_words = set(exp_action.split())
    act_words = set(act_action.split())
    if exp_words == act_words:
        return "verb_synonym"
    if len(exp_words) <= 1 and len(act_words) <= 1:
        return "verb_synonym"
    if len(exp_words) < len(act_words):
        return "too_specific"
    if len(exp_words) > len(act_words):
        return "too_generic"
    common = exp_words & act_words
    if len(common) == 0:
        return "verb_synonym"
    # If the first word (verb) differs, it's a verb synonym
    exp_first = exp_action.split()[0] if exp_action.split() else ""
    act_first = act_action.split()[0] if act_action.split() else ""
    if exp_first != act_first:
        return "verb_synonym"
    return "object_synonym"


def _detect_deadline_subcause(failure: dict[str, Any]) -> str:
    expected = failure.get("expected_commitments", [])
    actual = failure.get("actual_commitments", [])
    if not expected or not actual:
        return "prefix_by"
    exp_dl = (expected[0].get("deadline") or "").lower()
    act_dl = (actual[0].get("deadline") or "").lower()
    if exp_dl.replace("by ", "").strip() == act_dl.replace("by ", "").strip():
        return "prefix_by"
    if any(c in exp_dl + act_dl for c in ":"):
        return "time_format"
    if any(w in exp_dl + act_dl for w in ("end of", "quarter", "week", "soon", "later")):
        return "vague_relative_phrase"
    if any(w in exp_dl + act_dl for w in ("before", "after", "meeting", "event")):
        return "event_based_deadline"
    return "prefix_by"


def _detect_context_subcause(failure: dict[str, Any]) -> str:
    expected = failure.get("expected_commitments", [])
    actual = failure.get("actual_commitments", [])
    if not expected or not actual:
        return "paraphrase"
    exp_ctx = expected[0].get("context") or ""
    act_ctx = actual[0].get("context") or ""
    if len(act_ctx) < len(exp_ctx) * 0.5:
        return "truncated"
    if len(act_ctx) > len(exp_ctx) * 1.5:
        return "extra_context"
    return "paraphrase"


def _detect_update_vs_new_subcause(failure: dict[str, Any]) -> str:
    expected = failure.get("expected_commitments", [])
    actual = failure.get("actual_commitments", [])
    fields = set(failure.get("mismatched_fields", []))

    if _has_unchanged_existing(failure) and len(actual) > len(expected):
        return "unchanged_existing_leak"
    if expected and actual:
        exp_id = expected[0].get("id")
        act_id = actual[0].get("id")
        if exp_id is None and act_id is not None:
            return "expected_new_actual_update"
        if exp_id is not None and act_id is None:
            return "expected_update_actual_new"
    if len(expected) > 1 or len(actual) > 1:
        return "multiple_commitment_alignment"
    if "id" in fields and len(fields) > 5:
        return "expected_new_actual_update"
    return "expected_new_actual_update"


def _detect_false_positive_subcause(failure: dict[str, Any]) -> str:
    category = failure.get("category", "")
    scenario = failure.get("scenario", "")
    messages = failure.get("messages", "").lower()

    if category == "lifecycle_completion" or "started" in scenario or "almost" in scenario:
        return "progress_not_completion"
    if "conditional" in scenario or "if " in messages:
        return "conditional_commitment"
    if "refuse" in scenario or "won't" in messages or "sorry" in messages:
        return "refusal_not_commitment"
    return "conditional_commitment"


def _detect_false_negative_subcause(failure: dict[str, Any]) -> str:
    scenario = failure.get("scenario", "")
    messages = failure.get("messages", "").lower()

    if "hedged" in scenario or "might" in messages or "not sure" in messages or "probably" in messages:
        return "hedged_commitment"
    if "third" in scenario or "contractor" in messages or "lawyer" in messages:
        return "third_party_obligation"
    if "group" in scenario or "we need" in messages or "we should" in messages:
        return "group_obligation"
    if "external" in scenario or "waiting" in scenario or "bank" in messages:
        return "external_waiting"
    return "hedged_commitment"


def _detect_lifecycle_subcause(failure: dict[str, Any]) -> str:
    scenario = failure.get("scenario", "")
    messages = failure.get("messages", "").lower()

    if "started" in scenario or "started" in messages:
        return "started_not_done"
    if "almost" in scenario or "almost" in messages:
        return "almost_done_not_done"
    if "partial" in scenario or "3 out of" in messages or "some of" in messages:
        return "partial_completion"
    if "conditional" in scenario or "unless" in messages:
        return "conditional_dismissal"
    return "started_not_done"


def _detect_party_subcause(failure: dict[str, Any]) -> str:
    scenario = failure.get("scenario", "")
    if "implied" in scenario or "implicit" in scenario:
        return "implied_party"
    if "handoff" in scenario:
        return "party_handoff"
    return "third_party_named"


def _classify_failure(failure: dict[str, Any]) -> LocalizedFailure:
    error_type = failure.get("error_type", "")
    fields = failure.get("mismatched_fields", [])
    field_set = set(fields)
    category = failure.get("category", "")
    scenario = failure.get("scenario", "")

    # FALSE_POSITIVE
    if error_type == "FALSE_POSITIVE":
        if category == "lifecycle_completion" and ("started" in scenario or "almost" in scenario):
            root_cause = "lifecycle_policy"
            subcause = _detect_lifecycle_subcause(failure)
        elif category == "lifecycle_completion":
            root_cause = "lifecycle_policy"
            subcause = _detect_lifecycle_subcause(failure)
        else:
            root_cause = "over_extraction_policy"
            subcause = _detect_false_positive_subcause(failure)
        return LocalizedFailure(
            root_cause=root_cause,
            subcause=subcause,
            repair_type=_REPAIR_MAP[root_cause][0],
            confidence=0.9,
            split=failure.get("split", ""),
            category=category,
            scenario=scenario,
            difficulty=failure.get("difficulty", ""),
            error_type=error_type,
            mismatched_fields=fields,
            messages=failure.get("messages", ""),
            expected_commitments=failure.get("expected_commitments", []),
            actual_commitments=failure.get("actual_commitments", []),
            detail=f"Expected 0, got {len(failure.get('actual_commitments', []))}",
        )

    # FALSE_NEGATIVE
    if error_type == "FALSE_NEGATIVE":
        if "external" in scenario or "waiting" in scenario or "third" in scenario or "contractor" in scenario or "implied" in scenario or "party" in scenario:
            if "external" in scenario or "waiting" in scenario:
                root_cause = "under_extraction_policy"
                subcause = "external_waiting"
            else:
                root_cause = "party_resolution"
                subcause = "third_party_obligation"
            confidence = 0.8
        elif "group" in scenario or "we_need" in scenario:
            root_cause = "under_extraction_policy"
            subcause = "group_obligation"
            confidence = 0.8
        else:
            root_cause = "under_extraction_policy"
            subcause = _detect_false_negative_subcause(failure)
            confidence = 0.85
        return LocalizedFailure(
            root_cause=root_cause,
            subcause=subcause,
            repair_type=_REPAIR_MAP[root_cause][0],
            confidence=confidence,
            split=failure.get("split", ""),
            category=category,
            scenario=scenario,
            difficulty=failure.get("difficulty", ""),
            error_type=error_type,
            mismatched_fields=fields,
            messages=failure.get("messages", ""),
            expected_commitments=failure.get("expected_commitments", []),
            actual_commitments=failure.get("actual_commitments", []),
            detail=f"Expected {len(failure.get('expected_commitments', []))}, got 0",
        )

    # FIELD_MISMATCH
    if field_set == {"context"}:
        root_cause = "context_metric_noise"
        subcause = _detect_context_subcause(failure)
    elif field_set == {"required_action"}:
        root_cause = "required_action_normalization"
        subcause = _detect_required_action_subcause(failure)
    elif field_set == {"deadline"}:
        root_cause = "deadline_normalization"
        subcause = _detect_deadline_subcause(failure)
    elif field_set == {"committed_party"}:
        root_cause = "party_resolution"
        subcause = _detect_party_subcause(failure)
    elif field_set == {"required_action", "deadline"}:
        root_cause = "required_action_normalization"
        subcause = _detect_required_action_subcause(failure)
    elif field_set == {"deadline", "status"}:
        root_cause = "deadline_normalization"
        subcause = _detect_deadline_subcause(failure)
    elif "id" in field_set and len(field_set) > 5:
        root_cause = "update_vs_new_matching"
        subcause = _detect_update_vs_new_subcause(failure)
    elif "id" in field_set:
        root_cause = "update_vs_new_matching"
        subcause = _detect_update_vs_new_subcause(failure)
    elif _has_unchanged_existing(failure) and len(field_set) > 3:
        root_cause = "update_vs_new_matching"
        subcause = "unchanged_existing_leak"
    elif "required_action" in field_set:
        root_cause = "required_action_normalization"
        subcause = _detect_required_action_subcause(failure)
    elif "deadline" in field_set:
        root_cause = "deadline_normalization"
        subcause = _detect_deadline_subcause(failure)
    elif "committed_party" in field_set:
        root_cause = "party_resolution"
        subcause = _detect_party_subcause(failure)
    else:
        root_cause = "multi_field_mismatch"
        subcause = "multiple_commitment_alignment"

    return LocalizedFailure(
        root_cause=root_cause,
        subcause=subcause,
        repair_type=_REPAIR_MAP[root_cause][0],
        confidence=0.9 if len(field_set) <= 2 else 0.75,
        split=failure.get("split", ""),
        category=category,
        scenario=scenario,
        difficulty=failure.get("difficulty", ""),
        error_type=error_type,
        mismatched_fields=fields,
        messages=failure.get("messages", ""),
        expected_commitments=failure.get("expected_commitments", []),
        actual_commitments=failure.get("actual_commitments", []),
        detail=f"Fields: {', '.join(fields)}" if fields else "",
    )


def localize(failures: list[dict[str, Any]]) -> list[LocalizedFailure]:
    return [_classify_failure(f) for f in failures]


def build_report(localized: list[LocalizedFailure]) -> LocalizationReport:
    root_cause_counts: Counter[str] = Counter()
    subcause_counts: Counter[str] = Counter()
    cause_confidence: dict[str, float] = {}

    for lf in localized:
        root_cause_counts[lf.root_cause] += 1
        subcause_counts[f"{lf.root_cause}.{lf.subcause}"] += 1
        cause_confidence[lf.root_cause] = max(
            cause_confidence.get(lf.root_cause, 0), lf.confidence
        )

    # Priority = impact * confidence / cost
    priorities: list[tuple[str, float, str]] = []
    for cause, count in root_cause_counts.items():
        repair_type, repair_desc = _REPAIR_MAP.get(cause, ("unknown", "—"))
        cost = _REPAIR_COST.get(repair_type, 3)
        conf = cause_confidence.get(cause, 0.5)
        priority = count * conf / cost
        priorities.append((cause, priority, repair_desc))

    priorities.sort(key=lambda x: x[1], reverse=True)

    return LocalizationReport(
        root_cause_counts=root_cause_counts,
        subcause_counts=subcause_counts,
        top_actions=priorities,
        localized=localized,
    )


def print_summary(localized: list[LocalizedFailure]) -> None:
    report = build_report(localized)

    # Root cause table with subcause breakdown
    table = Table(title="Failure Localization — Root Causes", show_lines=False)
    table.add_column("Root Cause", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Top Subcause", style="dim")
    table.add_column("Repair Type", style="yellow")
    table.add_column("Priority", justify="right", style="green")

    for cause, count in report.root_cause_counts.most_common():
        repair_type, _ = _REPAIR_MAP.get(cause, ("unknown", "—"))
        cost = _REPAIR_COST.get(repair_type, 3)
        conf = max(lf.confidence for lf in localized if lf.root_cause == cause)
        priority = count * conf / cost
        sub_for_cause = {k: v for k, v in report.subcause_counts.items() if k.startswith(cause + ".")}
        top_sub = max(sub_for_cause, key=sub_for_cause.get).split(".", 1)[1] if sub_for_cause else "—"
        table.add_row(cause, str(count), top_sub, repair_type, f"{priority:.1f}")

    console.print()
    console.print(table)

    # Top 2 suggested actions
    console.print()
    console.print("[bold]Top 2 Suggested Repairs[/bold]")
    for i, (cause, priority, repair_desc) in enumerate(report.top_actions[:2], 1):
        count = report.root_cause_counts[cause]
        console.print(f"  {i}. [bold cyan]{cause}[/bold cyan] ({count} failures, priority {priority:.1f})")
        console.print(f"     → {repair_desc}")

    # Per-split breakdown
    split_causes: dict[str, Counter[str]] = {}
    for lf in localized:
        split_causes.setdefault(lf.split, Counter())[lf.root_cause] += 1

    split_table = Table(title="Root Causes by Split", show_lines=False)
    split_table.add_column("Root Cause", style="bold")
    for split in sorted(split_causes):
        split_table.add_column(split.upper(), justify="right", style="cyan")
    split_table.add_column("Total", justify="right", style="bold")

    for cause, _ in report.root_cause_counts.most_common():
        row = [cause]
        total = 0
        for split in sorted(split_causes):
            count = split_causes[split].get(cause, 0)
            row.append(str(count) if count else "—")
            total += count
        row.append(str(total))
        split_table.add_row(*row)

    console.print()
    console.print(split_table)

    # Top examples per root cause
    console.print()
    console.print("[bold]Top examples per root cause:[/bold]")
    for cause, _ in report.root_cause_counts.most_common():
        examples = [lf for lf in localized if lf.root_cause == cause][:3]
        console.print(f"\n  [bold cyan]{cause}[/bold cyan] ({report.root_cause_counts[cause]} total)")
        for ex in examples:
            detail = f"  [{ex.split}] {ex.category}/{ex.scenario}"
            if ex.detail:
                detail += f" — {ex.detail}"
            detail += f"  [dim]({ex.subcause})[/dim]"
            console.print(detail)
            msg_preview = ex.messages.replace("\n", " | ")[:80]
            console.print(f"    [dim]{msg_preview}[/dim]")


def print_markdown_table(localized: list[LocalizedFailure]) -> None:
    report = build_report(localized)

    table = Table(title="Failure Localization — Root Causes", show_lines=False)
    table.add_column("Root Cause", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Repair Type", style="yellow")
    table.add_column("Priority", justify="right", style="green")

    for cause, count in report.root_cause_counts.most_common():
        repair_type, _ = _REPAIR_MAP.get(cause, ("unknown", ""))
        cost = _REPAIR_COST.get(repair_type, 3)
        conf = max(lf.confidence for lf in localized if lf.root_cause == cause)
        priority = count * conf / cost
        table.add_row(cause, str(count), repair_type, f"{priority:.1f}")

    console.print()
    console.print(table)

    console.print()
    console.print("[bold]Top 2 Suggested Repairs[/bold]")
    for i, (cause, priority, repair_desc) in enumerate(report.top_actions[:2], 1):
        count = report.root_cause_counts[cause]
        console.print(f"  {i}. [bold cyan]{cause}[/bold cyan] ({count} failures, priority {priority:.1f})")
        console.print(f"     → {repair_desc}")


def to_json(localized: list[LocalizedFailure]) -> str:
    report = build_report(localized)
    output = []
    for lf in localized:
        output.append(
            {
                "root_cause": lf.root_cause,
                "subcause": lf.subcause,
                "repair_type": lf.repair_type,
                "confidence": lf.confidence,
                "split": lf.split,
                "category": lf.category,
                "scenario": lf.scenario,
                "difficulty": lf.difficulty,
                "error_type": lf.error_type,
                "mismatched_fields": lf.mismatched_fields,
                "detail": lf.detail,
                "messages": lf.messages,
            }
        )
    output.append(
        {
            "summary": {
                "root_cause_counts": dict(report.root_cause_counts.most_common()),
                "subcause_counts": dict(report.subcause_counts.most_common()),
                "top_actions": [
                    {"root_cause": c, "priority": p, "repair": r}
                    for c, p, r in report.top_actions[:5]
                ],
            }
        }
    )
    return json.dumps(output, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize eval failures to root causes")
    parser.add_argument("failures_path", help="Path to failures.jsonl")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--md", action="store_true", help="Output markdown table")
    args = parser.parse_args()

    path = Path(args.failures_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise SystemExit(1)

    with path.open() as f:
        failures = [json.loads(line) for line in f if line.strip()]

    localized = localize(failures)

    if args.json:
        print(to_json(localized))
    elif args.md:
        print_markdown_table(localized)
    else:
        console.print(f"\n[bold]Localized {len(localized)} failures from {path.name}[/bold]")
        print_summary(localized)


if __name__ == "__main__":
    main()
