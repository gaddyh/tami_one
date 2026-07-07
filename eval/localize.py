"""Failure localization layer for commitment extraction evals.

Reads a failures.jsonl file from an eval run and assigns each failure a root
cause, repair type, and confidence. Outputs a ranked summary table so the
next improvement decision is obvious.

Usage:
    python -m eval.localize runs/20260707_023551/failures.jsonl
    python -m eval.localize runs/20260707_023551/failures.jsonl --json
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


@dataclass
class LocalizedFailure:
    root_cause: str
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


def _classify_failure(failure: dict[str, Any]) -> LocalizedFailure:
    error_type = failure.get("error_type", "")
    fields = failure.get("mismatched_fields", [])
    field_set = set(fields)
    category = failure.get("category", "")
    scenario = failure.get("scenario", "")

    # FALSE_POSITIVE: agent extracted when it shouldn't have
    if error_type == "FALSE_POSITIVE":
        if category == "lifecycle_completion":
            if "started" in scenario or "almost" in scenario:
                root_cause = "lifecycle_policy"
            else:
                root_cause = "over_extraction_policy"
        elif category == "act_vs_ignore":
            root_cause = "over_extraction_policy"
        else:
            root_cause = "over_extraction_policy"
        return LocalizedFailure(
            root_cause=root_cause,
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

    # FALSE_NEGATIVE: agent missed a commitment
    if error_type == "FALSE_NEGATIVE":
        if "unclear" in scenario or "vague" in scenario or "hedged" in scenario:
            root_cause = "under_extraction_policy"
            confidence = 0.85
        elif "external" in scenario or "waiting" in scenario or "third" in scenario:
            root_cause = "party_resolution"
            confidence = 0.8
        elif "group" in scenario or "we_need" in scenario:
            root_cause = "party_resolution"
            confidence = 0.8
        else:
            root_cause = "under_extraction_policy"
            confidence = 0.7
        return LocalizedFailure(
            root_cause=root_cause,
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

    # FIELD_MISMATCH: agent found commitment but fields don't match
    # Check single-field cases first (most specific)
    if field_set == {"context"}:
        root_cause = "context_metric_noise"
    elif field_set == {"required_action"}:
        root_cause = "required_action_normalization"
    elif field_set == {"deadline"}:
        root_cause = "deadline_normalization"
    elif field_set == {"committed_party"}:
        root_cause = "party_resolution"
    # Two-field combos
    elif field_set == {"required_action", "deadline"}:
        root_cause = "required_action_normalization"
    elif field_set == {"deadline", "status"}:
        root_cause = "deadline_normalization"
    # Update-vs-new: id mismatch with many fields (agent returned existing instead of new)
    elif "id" in field_set and len(field_set) > 5:
        root_cause = "update_vs_new_matching"
    elif "required_action" in field_set and "id" in field_set:
        root_cause = "update_vs_new_matching"
    # Unchanged existing leak: actual includes unchanged existing commitment
    elif _has_unchanged_existing(failure) and len(field_set) > 3:
        root_cause = "update_vs_new_matching"
    # Remaining multi-field: pick the dominant field
    elif "required_action" in field_set:
        root_cause = "required_action_normalization"
    elif "deadline" in field_set:
        root_cause = "deadline_normalization"
    elif "committed_party" in field_set:
        root_cause = "party_resolution"
    else:
        root_cause = "multi_field_mismatch"

    return LocalizedFailure(
        root_cause=root_cause,
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


def print_summary(localized: list[LocalizedFailure]) -> None:
    # Root cause summary table
    cause_counts: Counter[str] = Counter()
    cause_confidence: dict[str, float] = {}
    for lf in localized:
        cause_counts[lf.root_cause] += 1
        cause_confidence[lf.root_cause] = max(
            cause_confidence.get(lf.root_cause, 0), lf.confidence
        )

    table = Table(title="Failure Localization — Root Causes", show_lines=False)
    table.add_column("Root Cause", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Repair Type", style="yellow")
    table.add_column("Top Repair", style="green")
    table.add_column("Confidence", justify="right")

    for cause, count in cause_counts.most_common():
        repair_type, top_repair = _REPAIR_MAP.get(cause, ("unknown", "—"))
        conf = cause_confidence[cause]
        table.add_row(cause, str(count), repair_type, top_repair, f"{conf:.1f}")

    console.print()
    console.print(table)

    # Per-split breakdown
    split_causes: dict[str, Counter[str]] = {}
    for lf in localized:
        split_causes.setdefault(lf.split, Counter())[lf.root_cause] += 1

    split_table = Table(title="Root Causes by Split", show_lines=False)
    split_table.add_column("Root Cause", style="bold")
    for split in sorted(split_causes):
        split_table.add_column(split.upper(), justify="right", style="cyan")
    split_table.add_column("Total", justify="right", style="bold")

    for cause, _ in cause_counts.most_common():
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
    for cause, _ in cause_counts.most_common():
        examples = [lf for lf in localized if lf.root_cause == cause][:3]
        console.print(f"\n  [bold cyan]{cause}[/bold cyan] ({cause_counts[cause]} total)")
        for ex in examples:
            detail = f"  [{ex.split}] {ex.category}/{ex.scenario}"
            if ex.detail:
                detail += f" — {ex.detail}"
            console.print(detail)
            msg_preview = ex.messages.replace("\n", " | ")[:80]
            console.print(f"    [dim]{msg_preview}[/dim]")


def print_markdown_table(localized: list[LocalizedFailure]) -> None:
    """Output a Rich-rendered markdown-style summary table."""
    cause_counts: Counter[str] = Counter()
    for lf in localized:
        cause_counts[lf.root_cause] += 1

    table = Table(title="Failure Localization — Root Causes", show_lines=False)
    table.add_column("Root Cause", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Repair Type", style="yellow")

    for cause, count in cause_counts.most_common():
        repair_type, _ = _REPAIR_MAP.get(cause, ("unknown", ""))
        table.add_row(cause, str(count), repair_type)

    console.print()
    console.print(table)


def to_json(localized: list[LocalizedFailure]) -> str:
    output = []
    for lf in localized:
        output.append(
            {
                "root_cause": lf.root_cause,
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
