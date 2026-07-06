"""CLI entry point for running a DSPy evaluation smoke test.

Usage:
    # Full devset (12 examples)
    .venv/bin/python scripts/smoke_eval.py

    # Run a specific split
    .venv/bin/python scripts/smoke_eval.py --split dev
    .venv/bin/python scripts/smoke_eval.py --split test
    .venv/bin/python scripts/smoke_eval.py --split train

    # Run all splits with aggregated results table
    .venv/bin/python scripts/smoke_eval.py --all

    # First 3 examples only
    .venv/bin/python scripts/smoke_eval.py --limit 3

    # Use a specific model
    .venv/bin/python scripts/smoke_eval.py --model gpt-4o-mini

    # Verbose: show actual vs expected per example
    .venv/bin/python scripts/smoke_eval.py --verbose
"""

import argparse
import json as _json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import settings
from app.commitments.commitments_agent import CommitmentAgent, configure_dspy
from app.commitments.eval import (
    _token_f1,
    _word_overlap,
    act_vs_ignore_metric,
    build_devset,
    compare_commitments,
    commitment_metric,
)

console = Console()

_DETAIL_FIELDS = ["committed_party", "required_action", "deadline", "context", "status"]

_SPLITS = {
    "train": "trainset.json",
    "dev": "devset.json",
    "test": "testset.json",
}


def _split_path(split: str) -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "evals" / _SPLITS[split]


def _mismatch_table(mismatches: list[dict]) -> Table:
    """Build a rich table from compare_commitments mismatch dicts."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Field", style="bold")
    table.add_column("Expected", style="green")
    table.add_column("Actual", style="red")
    table.add_column("Match", justify="center", width=5)

    for m in mismatches:
        ev_str = str(m["expected"]) if m["expected"] is not None else "None"
        av_str = str(m["actual"]) if m["actual"] is not None else "None"
        table.add_row(
            str(m["index"]),
            m["field"],
            ev_str,
            av_str,
            "[red]✗[/]",
            style="red",
        )

    return table


def _commitment_details(commitments: list, label: str, style: str) -> Table:
    """Print commitment fields: committed_party, required_action, deadline, context."""
    table = Table(show_header=True, header_style=f"bold {style}", title=label)
    table.add_column("#", style="dim", width=3)
    table.add_column("committed_party", style=style)
    table.add_column("required_action", style=style)
    table.add_column("deadline", style=style)
    table.add_column("context", style=style)
    table.add_column("status", style=style)

    for idx, c in enumerate(commitments):
        vals = c.model_dump(mode="json") if hasattr(c, "model_dump") else c
        table.add_row(
            str(idx),
            str(vals.get("committed_party", "—")),
            str(vals.get("required_action", "—")),
            str(vals.get("deadline", "—")),
            str(vals.get("context", "—")),
            str(vals.get("status", "—")),
        )

    return table


def _run_split(
    split: str,
    *,
    limit: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run eval on a single split and return summary stats with confusion matrix data."""
    path = _split_path(split)
    devset = build_devset(path)
    if limit:
        devset = devset[:limit]

    console.print(f"[bold cyan]Running eval on {split} split ({len(devset)} examples from {path.name})...[/]\n")

    agent = CommitmentAgent()
    per_example: list[dict] = []

    for i, ex in enumerate(devset):
        pred = agent(**ex.inputs())

        expected_empty = len(ex.expected_commitments) == 0
        actual_empty = len(pred.commitments) == 0
        mismatches = compare_commitments(ex.expected_commitments, pred.commitments)
        matched = not mismatches
        avi = act_vs_ignore_metric(ex, pred)
        metric_score = commitment_metric(ex, pred)

        # Classify: TP / FP / FN / TN
        if not expected_empty and not actual_empty:
            confusion = "tp"
        elif expected_empty and not actual_empty:
            confusion = "fp"
        elif not expected_empty and actual_empty:
            confusion = "fn"
        else:
            confusion = "tn"

        # Per-field match for TP cases
        field_matches: dict[str, bool] = {}
        if confusion == "tp":
            exp_list = ex.expected_commitments
            act_list = pred.commitments
            for field in _DETAIL_FIELDS:
                for idx_e, exp_c in enumerate(exp_list):
                    act_c = act_list[idx_e] if idx_e < len(act_list) else None
                    ev = getattr(exp_c, field, None)
                    av = getattr(act_c, field, None) if act_c else None
                    ev_s = str(ev).lower() if ev else "—"
                    av_s = str(av).lower() if av else "—"
                    if field == "context":
                        field_matches[field] = _word_overlap(ev_s, av_s)
                    elif field == "required_action":
                        field_matches[field] = _token_f1(ev_s, av_s) >= 0.75
                    else:
                        field_matches[field] = ev_s == av_s

        per_example.append({
            "index": i,
            "category": getattr(ex, "category", "—") or "—",
            "scenario": getattr(ex, "scenario", "—") or "—",
            "difficulty": getattr(ex, "difficulty", "—") or "—",
            "expected_empty": expected_empty,
            "actual_empty": actual_empty,
            "confusion": confusion,
            "matched": matched,
            "avi": avi,
            "metric_score": metric_score,
            "mismatches": mismatches,
            "field_matches": field_matches,
            "messages": ex.messages,
            "pred_commitments": pred.commitments,
            "expected_commitments": ex.expected_commitments,
        })

        if verbose:
            color = "green" if matched else "red"
            status = "MATCH" if matched else "MISMATCH"
            avi_label = (
                "[green]correctly ignored[/]"
                if avi == 1.0 and expected_empty
                else "[green]correctly acted[/]"
                if avi == 1.0
                else "[red]false positive[/]"
                if expected_empty
                else "[red]false negative[/]"
            )

            console.print(Panel(
                f"[dim]Messages:[/]\n{ex.messages}",
                title=f"Example {i} — {status} | Act/Ignore: {avi_label}",
                border_style=color,
            ))
            if mismatches and avi == 1.0:
                console.print(_mismatch_table(mismatches))
                if pred.commitments:
                    console.print(_commitment_details(pred.commitments, "Actual", "red"))
            elif pred.commitments:
                console.print(_commitment_details(pred.commitments, "Actual", "green"))
            console.print()

    # Compute aggregate stats
    tp = sum(1 for e in per_example if e["confusion"] == "tp")
    fp = sum(1 for e in per_example if e["confusion"] == "fp")
    fn = sum(1 for e in per_example if e["confusion"] == "fn")
    tn = sum(1 for e in per_example if e["confusion"] == "tn")

    total_expected = sum(len(e["expected_commitments"]) for e in per_example)
    total_matched = sum(e["metric_score"] * len(e["expected_commitments"]) for e in per_example)

    # Print confusion matrix
    _print_confusion_matrix(split, tp, fp, fn, tn)

    # Print per-category breakdown
    _print_category_breakdown(per_example)

    # Print per-difficulty breakdown
    _print_difficulty_breakdown(per_example)

    # Print category × difficulty matrix
    _print_category_difficulty_matrix(per_example)

    # Print per-field accuracy for TP cases
    _print_field_accuracy(per_example)

    if verbose:
        console.print()

    avi_score = sum(e["avi"] for e in per_example) / len(per_example) if per_example else 0
    commit_metric = total_matched / total_expected if total_expected > 0 else 0.0

    console.print(f"[bold]Act/Ignore Score: {avi_score:.2f} ({sum(e['avi'] for e in per_example):.0f}/{len(per_example)})[/]")
    if total_expected > 0:
        console.print(f"[bold]Commitment Metric: {total_matched:.0f}/{total_expected} ({commit_metric * 100:.1f}%)[/]")
    else:
        console.print(f"[bold]Commitment Metric: N/A[/]")
    console.print()

    return {
        "split": split,
        "n": len(per_example),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "act_vs_ignore": avi_score,
        "commitment_metric": commit_metric,
        "total_expected": total_expected,
        "total_matched": total_matched,
        "per_example": per_example,
    }


def _print_confusion_matrix(split: str, tp: int, fp: int, fn: int, tn: int) -> None:
    """Print a 2x2 confusion matrix table for act vs ignore."""
    total = tp + fp + fn + tn
    table = Table(
        show_header=True,
        header_style="bold",
        title=f"{split.upper()} — Confusion Matrix (Act vs Ignore)",
        show_lines=True,
    )
    table.add_column("", style="dim", width=12)
    table.add_column("Predicted: Act", justify="center", width=16)
    table.add_column("Predicted: Ignore", justify="center", width=16)
    table.add_column("Total", justify="right", width=8)

    table.add_row(
        "[bold]Expected: Act",
        f"[green]{tp}[/]" if tp else "0",
        f"[red]{fn}[/]" if fn else "0",
        str(tp + fn),
    )
    table.add_row(
        "[bold]Expected: Ignore",
        f"[red]{fp}[/]" if fp else "0",
        f"[green]{tn}[/]" if tn else "0",
        str(fp + tn),
    )
    table.add_row(
        "[bold]Total",
        str(tp + fp),
        str(fn + tn),
        str(total),
        style="dim",
    )

    console.print(table)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    console.print(
        f"  Precision: [bold]{precision:.2f}[/]  "
        f"Recall: [bold]{recall:.2f}[/]  "
        f"F1: [bold]{f1:.2f}[/]  "
        f"Accuracy: [bold]{accuracy:.2f}[/]\n"
    )


def _print_category_breakdown(per_example: list[dict]) -> None:
    """Print per-category TP/FP/FN/TN breakdown."""
    categories: dict[str, dict[str, int]] = {}
    for e in per_example:
        cat = e.get("category", "—")
        categories.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        categories[cat][e["confusion"]] += 1

    table = Table(
        show_header=True,
        header_style="bold",
        title="Per-Category Breakdown",
    )
    table.add_column("Category", style="bold", width=28)
    table.add_column("TP", justify="center", width=5)
    table.add_column("FP", justify="center", width=5)
    table.add_column("FN", justify="center", width=5)
    table.add_column("TN", justify="center", width=5)
    table.add_column("Total", justify="right", width=6)

    for cat in sorted(categories):
        c = categories[cat]
        total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        table.add_row(
            cat,
            f"[green]{c['tp']}[/]" if c["tp"] else "0",
            f"[red]{c['fp']}[/]" if c["fp"] else "0",
            f"[red]{c['fn']}[/]" if c["fn"] else "0",
            f"[green]{c['tn']}[/]" if c["tn"] else "0",
            str(total),
        )

    console.print(table)
    console.print()


def _print_difficulty_breakdown(per_example: list[dict]) -> None:
    """Print per-difficulty TP/FP/FN/TN breakdown with precision/recall/F1/accuracy."""
    difficulties: dict[str, dict[str, int]] = {}
    for e in per_example:
        diff = e.get("difficulty", "—")
        difficulties.setdefault(diff, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        difficulties[diff][e["confusion"]] += 1

    table = Table(
        show_header=True,
        header_style="bold",
        title="Per-Difficulty Breakdown",
    )
    table.add_column("Difficulty", style="bold", width=12)
    table.add_column("TP", justify="center", width=5)
    table.add_column("FP", justify="center", width=5)
    table.add_column("FN", justify="center", width=5)
    table.add_column("TN", justify="center", width=5)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=8)
    table.add_column("F1", justify="right", width=8)
    table.add_column("Accuracy", justify="right", width=10)

    for diff in ("easy", "medium", "hard", "—"):
        if diff not in difficulties:
            continue
        c = difficulties[diff]
        total = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (c["tp"] + c["tn"]) / total if total > 0 else 0.0
        color = "green" if accuracy >= 0.8 else "yellow" if accuracy >= 0.5 else "red"
        table.add_row(
            diff,
            f"[green]{c['tp']}[/]" if c["tp"] else "0",
            f"[red]{c['fp']}[/]" if c["fp"] else "0",
            f"[red]{c['fn']}[/]" if c["fn"] else "0",
            f"[green]{c['tn']}[/]" if c["tn"] else "0",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}",
            f"[{color}]{accuracy:.2f}[/{color}]",
        )

    console.print(table)
    console.print()


def _print_category_difficulty_matrix(per_example: list[dict]) -> None:
    """Print category × difficulty matrix with counts and hard accuracy."""
    matrix: dict[str, dict[str, int]] = {}
    hard_correct: dict[str, int] = {}
    hard_total: dict[str, int] = {}

    for e in per_example:
        cat = e.get("category", "—")
        diff = e.get("difficulty", "—")
        matrix.setdefault(cat, {"easy": 0, "medium": 0, "hard": 0, "—": 0})
        matrix[cat][diff] = matrix[cat].get(diff, 0) + 1
        if diff == "hard":
            hard_total.setdefault(cat, 0)
            hard_total[cat] += 1
            if e["confusion"] in ("tp", "tn"):
                hard_correct.setdefault(cat, 0)
                hard_correct[cat] += 1

    table = Table(
        show_header=True,
        header_style="bold",
        title="Category × Difficulty",
    )
    table.add_column("Category", style="bold", width=28)
    table.add_column("Easy", justify="center", width=6)
    table.add_column("Medium", justify="center", width=7)
    table.add_column("Hard", justify="center", width=6)
    table.add_column("Hard Accuracy", justify="right", width=14)

    for cat in sorted(matrix):
        row = matrix[cat]
        h_total = hard_total.get(cat, 0)
        h_correct = hard_correct.get(cat, 0)
        h_acc = h_correct / h_total if h_total > 0 else 0.0
        color = "green" if h_acc >= 0.8 else "yellow" if h_acc >= 0.5 else "red"
        table.add_row(
            cat,
            str(row.get("easy", 0)),
            str(row.get("medium", 0)),
            str(row.get("hard", 0)),
            f"[{color}]{h_acc:.0%}[/{color}]" if h_total > 0 else "—",
        )

    console.print(table)
    console.print()


def _print_field_accuracy(per_example: list[dict]) -> None:
    """Print per-field match accuracy for TP (true positive) examples."""
    tp_examples = [e for e in per_example if e["confusion"] == "tp"]
    if not tp_examples:
        console.print("[dim]No true positives to analyze field accuracy.\n[/]")
        return

    field_counts: dict[str, int] = {f: 0 for f in _DETAIL_FIELDS}
    field_totals: dict[str, int] = {f: 0 for f in _DETAIL_FIELDS}

    for e in tp_examples:
        fm = e.get("field_matches", {})
        for field in _DETAIL_FIELDS:
            if field in fm:
                field_totals[field] += 1
                if fm[field]:
                    field_counts[field] += 1

    table = Table(
        show_header=True,
        header_style="bold",
        title="Field Accuracy (True Positives only)",
    )
    table.add_column("Field", style="bold", width=20)
    table.add_column("Matched", justify="right", width=10)
    table.add_column("Total", justify="right", width=8)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("Bar", width=16)

    for field in _DETAIL_FIELDS:
        matched = field_counts[field]
        total = field_totals[field]
        pct = matched / total * 100 if total > 0 else 0.0
        bar_filled = int(pct / 6.25)
        bar = "█" * bar_filled + "░" * (16 - bar_filled)
        color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
        table.add_row(
            field,
            str(matched),
            str(total),
            f"[{color}]{pct:.0f}%[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)
    console.print()


def _print_aggregated_table(results: list[dict]) -> None:
    """Print a rich aggregated results table across all splits."""
    table = Table(show_header=True, header_style="bold", title="Aggregated Results")
    table.add_column("Split", style="bold cyan", width=8)
    table.add_column("N", justify="right", width=5)
    table.add_column("TP", justify="center", width=5)
    table.add_column("FP", justify="center", width=5)
    table.add_column("FN", justify="center", width=5)
    table.add_column("TN", justify="center", width=5)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1", justify="right", width=8)
    table.add_column("Accuracy", justify="right", width=10)

    for r in results:
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        table.add_row(
            r["split"].upper(),
            str(r["n"]),
            f"[green]{tp}[/]" if tp else "0",
            f"[red]{fp}[/]" if fp else "0",
            f"[red]{fn}[/]" if fn else "0",
            f"[green]{tn}[/]" if tn else "0",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}",
            f"{accuracy:.2f}",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy commitment extraction eval")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test"],
        help="Which split to run (default: dev)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three splits and show aggregated results table",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N examples (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the OpenAI model (default: from settings)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print actual vs expected for each example",
    )
    args = parser.parse_args()

    if args.model:
        settings.openai_model = args.model

    configure_dspy(settings)

    if args.all:
        results = []
        for split in ["train", "dev", "test"]:
            r = _run_split(split, limit=args.limit, verbose=args.verbose)
            results.append(r)
        console.print()
        _print_aggregated_table(results)
        sys.exit(0)

    split = args.split or "dev"
    _run_split(split, limit=args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
