"""Run the full frozen-predictions workflow: eval → regrade (no judge) → regrade (frozen judge) → compare.

One command, zero ambiguity. Agent runs once, then two deterministic regrades
from the same frozen predictions. The delta between them is 100% judge effect.

Usage:
    python scripts/frozen_compare.py
    python scripts/frozen_compare.py --split dev
    python scripts/frozen_compare.py --limit 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run a command and stream output."""
    console.print(f"\n[bold cyan]$ {' '.join(cmd)}[/]\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        console.print(f"[bold red]Command failed with exit code {result.returncode}[/]")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-predictions workflow: eval → regrade → compare")
    parser.add_argument("--split", type=str, default=None, choices=["train", "dev", "test", "challenge"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Step 1: Run eval with --llm-judge --freeze-judge
    console.print("[bold green]═══ Step 1: Eval run with judge + freeze ═══[/]")
    eval_cmd = [PYTHON, "scripts/eval_runner.py", "--llm-judge", "--freeze-judge"]
    if args.split:
        eval_cmd += ["--split", args.split]
    if args.limit:
        eval_cmd += ["--limit", str(args.limit)]
    _run(eval_cmd)

    # Find the run directory (most recent with -judge suffix)
    runs_dir = ROOT / "runs"
    judge_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.endswith("-judge") and "regrade" not in d.name],
        key=lambda d: d.name,
    )
    if not judge_runs:
        console.print("[bold red]No judge run found after eval![/]")
        sys.exit(1)
    run_dir = judge_runs[-1]
    run_id = run_dir.name

    console.print(f"\n[bold green]Found run: {run_id}[/]")

    predictions_path = run_dir / "predictions.jsonl"
    verdicts_path = run_dir / "judge_verdicts.jsonl"

    if not predictions_path.exists():
        console.print(f"[bold red]No predictions.jsonl in {run_dir}[/]")
        sys.exit(1)
    if not verdicts_path.exists():
        console.print(f"[bold red]No judge_verdicts.jsonl in {run_dir}[/]")
        sys.exit(1)

    # Step 2: Regrade without judge (pure deterministic)
    console.print("\n[bold green]═══ Step 2: Regrade without judge (deterministic) ═══[/]")
    regrade_no_judge_cmd = [PYTHON, "scripts/regrade_predictions.py", str(predictions_path)]
    if args.split:
        regrade_no_judge_cmd += ["--split", args.split]
    _run(regrade_no_judge_cmd)

    # Step 3: Regrade with frozen judge (zero LLM calls)
    console.print("\n[bold green]═══ Step 3: Regrade with frozen judge (zero LLM calls) ═══[/]")
    regrade_judge_cmd = [
        PYTHON, "scripts/regrade_predictions.py", str(predictions_path),
        "--llm-judge",
        "--judge-cache", str(verdicts_path),
        "--no-new-judge-calls",
    ]
    if args.split:
        regrade_judge_cmd += ["--split", args.split]
    _run(regrade_judge_cmd)

    # Find the two regrade run directories
    regrade_runs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and f"{run_id}-regrade" in d.name],
        key=lambda d: d.name,
    )
    if len(regrade_runs) < 2:
        console.print(f"[bold red]Expected 2 regrade runs, found {len(regrade_runs)}[/]")
        sys.exit(1)

    # no-judge regrade (no -judge suffix) and judge regrade (has -judge suffix)
    no_judge_run = next((d for d in regrade_runs if not d.name.endswith("-judge")), None)
    judge_regrade_run = next((d for d in regrade_runs if d.name.endswith("-judge")), None)

    if not no_judge_run or not judge_regrade_run:
        console.print("[bold red]Could not identify both regrade runs[/]")
        sys.exit(1)

    # Step 4: Compare
    console.print("\n[bold green]═══ Step 4: Compare (delta = pure judge effect) ═══[/]")
    _run([PYTHON, "scripts/compare_runs.py", str(no_judge_run), str(judge_regrade_run)])

    console.print("\n[bold green]═══ Done ═══[/]")
    console.print(f"  Original run:     {run_dir}")
    console.print(f"  Regrade (no judge): {no_judge_run}")
    console.print(f"  Regrade (judge):   {judge_regrade_run}")
    console.print("\n[dim]The delta between the two regrades is 100% attributable to the judge.[/]")


if __name__ == "__main__":
    main()
