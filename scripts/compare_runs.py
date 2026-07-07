"""Compare two eval runs side by side.

Reads failures.jsonl from each run directory and produces a rich console
table + markdown file showing:
  - Per-run failure counts and error types
  - Failures fixed (in run A but not in run B)
  - New failures (in run B but not in run A)
  - Persistent failures (in both runs)
  - Per-scenario diff table

Usage:
    python scripts/compare_runs.py runs/20260707_043117 runs/20260707_043208
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console(record=True)


def _load_failures(run_dir: Path) -> list[dict]:
    """Load failures.jsonl from a run directory."""
    fpath = run_dir / "failures.jsonl"
    if not fpath.exists():
        raise FileNotFoundError(f"No failures.jsonl in {run_dir}")
    with fpath.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _failure_key(f: dict) -> str:
    """Unique key for a failure: split:scenario (scenario is unique per split)."""
    return f"{f.get('split', '?')}:{f.get('scenario', '?')}"


def _parse_md_stats(run_dir: Path) -> dict[str, str]:
    """Extract key stats from the run's .md files by parsing the header line."""
    stats: dict[str, str] = {}
    for md_file in run_dir.glob("*.md"):
        if md_file.name == "summary.md":
            continue
        split = md_file.stem.upper()
        text = md_file.read_text(encoding="utf-8")
        # Look for: **N=22** | TP=18 FP=0 FN=1 TN=3
        for line in text.splitlines():
            if line.startswith("**N="):
                clean = line.replace("**", "").strip("| ").strip()
                stats[split] = clean
                break
    return stats


def _build_summary_table(
    run_a_id: str, run_a: list[dict],
    run_b_id: str, run_b: list[dict],
) -> Table:
    """High-level comparison: failure counts, error type breakdown."""
    table = Table(
        title="Run Comparison — Summary",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric", style="bold")
    table.add_column(run_a_id, justify="right")
    table.add_column(run_b_id, justify="right")
    table.add_column("Delta", justify="right")

    def _count_by(failures: list[dict], key: str, val: str) -> int:
        return sum(1 for f in failures if f.get(key) == val)

    total_a, total_b = len(run_a), len(run_b)
    table.add_row("Total failures", str(total_a), str(total_b), f"{total_b - total_a:+d}")

    for et in ["FALSE_POSITIVE", "FALSE_NEGATIVE", "FIELD_MISMATCH"]:
        ca = _count_by(run_a, "error_type", et)
        cb = _count_by(run_b, "error_type", et)
        label = et.replace("_", " ").title()
        table.add_row(label, str(ca), str(cb), f"{cb - ca:+d}")

    # Unique scenarios with failures
    scenarios_a = {_failure_key(f) for f in run_a}
    scenarios_b = {_failure_key(f) for f in run_b}
    table.add_row(
        "Unique failing scenarios",
        str(len(scenarios_a)),
        str(len(scenarios_b)),
        f"{len(scenarios_b) - len(scenarios_a):+d}",
    )

    return table


def _build_diff_table(
    run_a_id: str, run_a: list[dict],
    run_b_id: str, run_b: list[dict],
) -> Table:
    """Per-scenario diff: fixed, new, persistent."""
    map_a = {_failure_key(f): f for f in run_a}
    map_b = {_failure_key(f): f for f in run_b}

    keys_a = set(map_a)
    keys_b = set(map_b)
    fixed = sorted(keys_a - keys_b)
    new = sorted(keys_b - keys_a)
    persistent = sorted(keys_a & keys_b)

    table = Table(
        title="Per-Scenario Failure Diff",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Status", style="bold")
    table.add_column("Split")
    table.add_column("Scenario")
    table.add_column("Category")
    table.add_column("Error Type")
    table.add_column("Fields", overflow="fold")

    def _row(status: str, status_style: str, f: dict) -> None:
        table.add_row(
            f"[{status_style}]{status}[/{status_style}]",
            f.get("split", "—"),
            f.get("scenario", "—"),
            f.get("category", "—"),
            f.get("error_type", "—").replace("_", " "),
            ", ".join(f.get("mismatched_fields", [])) or "—",
        )

    for key in fixed:
        _row("FIXED", "green", map_a[key])
    for key in new:
        _row("NEW", "red", map_b[key])
    for key in persistent:
        _row("PERSIST", "yellow", map_b[key])

    return table


def _build_category_table(
    run_a_id: str, run_a: list[dict],
    run_b_id: str, run_b: list[dict],
) -> Table:
    """Per-category failure counts comparison."""
    from collections import Counter

    cats_a = Counter(f.get("category", "—") for f in run_a)
    cats_b = Counter(f.get("category", "—") for f in run_b)
    all_cats = sorted(set(cats_a) | set(cats_b))

    table = Table(
        title="Per-Category Failure Counts",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Category", style="bold")
    table.add_column(run_a_id, justify="right")
    table.add_column(run_b_id, justify="right")
    table.add_column("Delta", justify="right")

    for cat in all_cats:
        ca = cats_a.get(cat, 0)
        cb = cats_b.get(cat, 0)
        delta = cb - ca
        style = "green" if delta < 0 else "red" if delta > 0 else ""
        table.add_row(cat, str(ca), str(cb), f"[{style}]{delta:+d}[/{style}]" if style else str(delta))

    return table


def compare_runs(run_a_dir: Path, run_b_dir: Path) -> None:
    """Compare two run directories and print + save results."""
    run_a_id = run_a_dir.name
    run_b_id = run_b_dir.name

    failures_a = _load_failures(run_a_dir)
    failures_b = _load_failures(run_b_dir)

    stats_a = _parse_md_stats(run_a_dir)
    stats_b = _parse_md_stats(run_b_dir)

    console.print()
    console.print(f"[bold cyan]Comparing runs:[/]")
    console.print(f"  A: {run_a_id}  ({len(failures_a)} failures)")
    console.print(f"  B: {run_b_id}  ({len(failures_b)} failures)")
    console.print()

    # Stats from md headers
    if stats_a or stats_b:
        stats_table = Table(title="Run Headers (from .md files)", show_header=True, header_style="bold")
        stats_table.add_column("Split", style="bold")
        stats_table.add_column(run_a_id)
        stats_table.add_column(run_b_id)
        all_splits = sorted(set(stats_a) | set(stats_b))
        for split in all_splits:
            stats_table.add_row(split, stats_a.get(split, "—"), stats_b.get(split, "—"))
        console.print(stats_table)
        console.print()

    console.print(_build_summary_table(run_a_id, failures_a, run_b_id, failures_b))
    console.print()
    console.print(_build_category_table(run_a_id, failures_a, run_b_id, failures_b))
    console.print()
    console.print(_build_diff_table(run_a_id, failures_a, run_b_id, failures_b))

    # Save markdown
    compares_dir = run_a_dir.parent / "compares"
    compares_dir.mkdir(exist_ok=True)
    out_name = f"{run_a_id}_vs_{run_b_id}.md"
    out_path = compares_dir / out_name

    md = f"# Run Comparison: {run_a_id} vs {run_b_id}\n\n"
    md += f"**A:** {run_a_id} ({len(failures_a)} failures)\n\n"
    md += f"**B:** {run_b_id} ({len(failures_b)} failures)\n\n"

    if stats_a or stats_b:
        md += "## Run Headers\n\n"
        md += "| Split | " + run_a_id + " | " + run_b_id + " |\n"
        md += "|---|---|---|\n"
        all_splits = sorted(set(stats_a) | set(stats_b))
        for split in all_splits:
            md += f"| {split} | {stats_a.get(split, '—')} | {stats_b.get(split, '—')} |\n"
        md += "\n"

    # Summary table as text
    md += "## Summary\n\n"
    md += "```\n"
    md += console.export_text(clear=False, styles=False)
    md += "```\n"

    out_path.write_text(md, encoding="utf-8")
    console.print(f"\n[bold green]Saved to: {out_path}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("run_a", type=str, help="Path to first run directory (baseline)")
    parser.add_argument("run_b", type=str, help="Path to second run directory (new)")
    args = parser.parse_args()

    run_a_dir = Path(args.run_a)
    run_b_dir = Path(args.run_b)

    if not run_a_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_a_dir}")
    if not run_b_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_b_dir}")

    compare_runs(run_a_dir, run_b_dir)


if __name__ == "__main__":
    main()
