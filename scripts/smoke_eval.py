"""CLI entry point for running a DSPy evaluation smoke test.

Usage:
    # Full devset (10 examples)
    .venv/bin/python scripts/smoke_eval.py

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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import settings
from app.commitments.commitments_agent import CommitmentAgent, configure_dspy
from app.commitments.eval import (
    act_vs_ignore_metric,
    build_devset,
    compare_commitments,
    run_evaluation,
)

console = Console()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy commitment extraction eval")
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

    devset = build_devset()
    if args.limit:
        devset = devset[: args.limit]
        console.print(f"[bold cyan]Running smoke eval on {len(devset)} example(s)...[/]\n")
    else:
        console.print(f"[bold cyan]Running eval on full devset ({len(devset)} examples)...[/]\n")

    if args.verbose:
        agent = CommitmentAgent()
        act_vs_ignore_scores: list[float] = []
        for i, ex in enumerate(devset):
            pred = agent(**ex.inputs())

            mismatches = compare_commitments(ex.expected_commitments, pred.commitments)
            matched = not mismatches
            color = "green" if matched else "red"
            status = "MATCH" if matched else "MISMATCH"

            avi = act_vs_ignore_metric(ex, pred)
            act_vs_ignore_scores.append(avi)
            avi_label = (
                "[green]correctly ignored[/]"
                if avi == 1.0 and len(ex.expected_commitments) == 0
                else "[green]correctly acted[/]"
                if avi == 1.0
                else "[red]false positive[/]"
                if len(ex.expected_commitments) == 0
                else "[red]false negative[/]"
            )

            console.print(Panel(
                f"[dim]Messages:[/]\n{ex.messages}",
                title=f"Example {i} — {status} | Act/Ignore: {avi_label}",
                border_style=color,
            ))
            if mismatches and avi == 1.0:
                console.print(_mismatch_table(mismatches))
            console.print()
        console.print()

        avi_score = sum(act_vs_ignore_scores) / len(act_vs_ignore_scores) if act_vs_ignore_scores else 0
        console.print(f"[bold]Act/Ignore Score: {avi_score:.2f} ({sum(act_vs_ignore_scores)}/{len(act_vs_ignore_scores)})[/]\n")

    result = run_evaluation(devset=devset, display_table=not args.verbose)
    score = result.score if hasattr(result, "score") else result

    console.print(f"\n[bold]{'='*50}[/]")
    console.print(f"[bold]Score: {score}[/]")
    console.print(f"[bold]{'='*50}[/]")

    if score < 1.0:
        pct = score * 100 if isinstance(score, (int, float)) else 0
        console.print(f"\n[yellow]{pct:.0f}% of examples matched expected output.[/]")
        console.print("[dim]Review the table above for mismatches.[/]")
        sys.exit(0)


if __name__ == "__main__":
    main()
