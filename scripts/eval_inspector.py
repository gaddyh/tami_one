"""Interactive CLI agent for examining eval failures.

Can either load an existing run from runs/ or run a fresh eval, then enters
an interactive REPL for examining failures with filters, side-by-side diffs,
root-cause localization, and single-example re-runs.

Usage:
    # Pick from recent runs
    python scripts/eval_inspector.py

    # Load a specific run
    python scripts/eval_inspector.py --run 20260707_183434-judge

    # Run fresh eval, then enter inspector
    python scripts/eval_inspector.py --split dev
    python scripts/eval_inspector.py --split dev --limit 5 --llm-judge
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from app.commitments.models import Commitment
from eval.localize import (
    LocalizedFailure,
    _REPAIR_MAP,
    _classify_failure,
    localize,
    print_summary as print_localization_summary,
)

console = Console()

_DETAIL_FIELDS = ["committed_party", "required_action", "deadline", "context", "status"]


# ─── Data Model ──────────────────────────────────────────────────────────────


@dataclass
class FailureRecord:
    index: int
    split: str
    category: str
    scenario: str
    difficulty: str
    error_type: str
    mismatched_fields: list[str]
    messages: str
    existing_commitments: list[dict]
    expected_commitments: list[dict]
    actual_commitments: list[dict]
    chat_id: str = ""
    chat_name: str = ""
    current_datetime: str = ""
    existing_commitments_json: str = "[]"
    _raw_dict: dict | None = None

    def to_raw_dict(self) -> dict:
        if self._raw_dict is not None:
            return self._raw_dict
        return {
            "split": self.split,
            "category": self.category,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "messages": self.messages,
            "existing_commitments": self.existing_commitments,
            "expected_commitments": self.expected_commitments,
            "actual_commitments": self.actual_commitments,
            "error_type": self.error_type,
            "mismatched_fields": self.mismatched_fields,
        }


# ─── Loaders ─────────────────────────────────────────────────────────────────


def _load_from_run(run_dir: Path) -> list[FailureRecord]:
    failures_path = run_dir / "failures.jsonl"
    predictions_path = run_dir / "predictions.jsonl"

    if not failures_path.exists():
        console.print(f"[red]No failures.jsonl in {run_dir}[/]")
        raise SystemExit(1)

    with failures_path.open() as f:
        failure_dicts = [_json.loads(line) for line in f if line.strip()]

    # Build predictions map keyed by (split, category, scenario)
    pred_map: dict[tuple[str, str, str], dict] = {}
    predictions_available = False
    if predictions_path.exists():
        predictions_available = True
        with predictions_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                pred = _json.loads(line)
                key = (pred.get("split", ""), pred.get("category", ""), pred.get("scenario", ""))
                pred_map[key] = pred

    if not predictions_available:
        console.print(
            "[yellow]Warning: predictions.jsonl not found. "
            "rerun will not work for this run.[/]"
        )

    records: list[FailureRecord] = []
    for i, fd in enumerate(failure_dicts):
        split = fd.get("split", "")
        category = fd.get("category", "")
        scenario = fd.get("scenario", "")
        key = (split, category, scenario)

        pred = pred_map.get(key)
        chat_id = ""
        chat_name = ""
        current_datetime = ""
        existing_commitments_json = "[]"
        true_index = i

        if pred:
            inputs = pred.get("inputs", {})
            chat_id = inputs.get("chat_id", "")
            chat_name = inputs.get("chat_name", "") or ""
            current_datetime = inputs.get("current_datetime", "")
            existing_commitments_json = inputs.get("existing_commitments_json", "[]")
            true_index = pred.get("example_index", i)

        records.append(
            FailureRecord(
                index=true_index,
                split=split,
                category=category,
                scenario=scenario,
                difficulty=fd.get("difficulty", ""),
                error_type=fd.get("error_type", ""),
                mismatched_fields=fd.get("mismatched_fields", []),
                messages=fd.get("messages", ""),
                existing_commitments=fd.get("existing_commitments", []),
                expected_commitments=fd.get("expected_commitments", []),
                actual_commitments=fd.get("actual_commitments", []),
                chat_id=chat_id,
                chat_name=chat_name,
                current_datetime=current_datetime,
                existing_commitments_json=existing_commitments_json,
                _raw_dict=fd,
            )
        )

    return records


def _load_from_fresh(
    split: str,
    limit: int | None = None,
    use_llm_judge: bool = False,
) -> list[FailureRecord]:
    from eval_runner import _run_split

    result = _run_split(split, limit=limit, verbose=False, use_llm_judge=use_llm_judge)

    records: list[FailureRecord] = []
    for e in result["per_example"]:
        conf = e["confusion"]
        if conf == "fp":
            error_type = "FALSE_POSITIVE"
            mismatched_fields: list[str] = []
        elif conf == "fn":
            error_type = "FALSE_NEGATIVE"
            mismatched_fields = []
        elif conf == "tp" and e.get("mismatches"):
            error_type = "FIELD_MISMATCH"
            mismatched_fields = [m["field"] for m in e["mismatches"]]
        else:
            continue

        expected_dicts = _commitments_to_dicts(e["expected_commitments"])
        actual_dicts = _commitments_to_dicts(e["pred_commitments"])

        raw_dict = {
            "split": e["split"],
            "category": e["category"],
            "scenario": e["scenario"],
            "difficulty": e["difficulty"],
            "messages": e["messages"],
            "existing_commitments": _json.loads(e.get("existing_commitments_json", "[]")),
            "expected_commitments": expected_dicts,
            "actual_commitments": actual_dicts,
            "error_type": error_type,
            "mismatched_fields": mismatched_fields,
        }

        records.append(
            FailureRecord(
                index=e["index"],
                split=e["split"],
                category=e["category"],
                scenario=e["scenario"],
                difficulty=e["difficulty"],
                error_type=error_type,
                mismatched_fields=mismatched_fields,
                messages=e["messages"],
                existing_commitments=raw_dict["existing_commitments"],
                expected_commitments=expected_dicts,
                actual_commitments=actual_dicts,
                chat_id=e.get("chat_id", ""),
                chat_name=e.get("chat_name", "") or "",
                current_datetime=e.get("current_datetime", ""),
                existing_commitments_json=e.get("existing_commitments_json", "[]"),
                _raw_dict=raw_dict,
            )
        )

    return records


def _commitments_to_dicts(commitments: list) -> list[dict]:
    out = []
    for c in commitments:
        if hasattr(c, "model_dump"):
            out.append(c.model_dump(mode="json"))
        elif isinstance(c, dict):
            out.append(c)
        else:
            out.append(str(c))
    return out


# ─── Display Functions ───────────────────────────────────────────────────────


def _print_failure_list(failures: list[FailureRecord]) -> None:
    if not failures:
        console.print("[green]No failures matching current filters.[/]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Failures ({len(failures)})",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Error Type", style="bold red", width=18)
    table.add_column("Category", width=24)
    table.add_column("Scenario", width=32)
    table.add_column("Diff", justify="center", width=10)
    table.add_column("Fields", width=20)

    for i, f in enumerate(failures):
        fields_str = ", ".join(f.mismatched_fields) if f.mismatched_fields else "—"
        table.add_row(
            str(i),
            f.error_type.replace("_", " "),
            f.category,
            f.scenario,
            f.difficulty,
            fields_str,
        )

    console.print(table)


def _print_failure_detail(failure: FailureRecord, list_position: int | None = None) -> None:
    # 1. Input panel
    input_text = f"[dim]Messages:[/]\n{failure.messages}"
    meta_lines = []
    if failure.chat_id:
        meta_lines.append(f"[dim]chat_id:[/] {failure.chat_id}")
    if failure.chat_name:
        meta_lines.append(f"[dim]chat_name:[/] {failure.chat_name}")
    if failure.current_datetime:
        meta_lines.append(f"[dim]current_datetime:[/] {failure.current_datetime}")
    if meta_lines:
        input_text += "\n\n" + "\n".join(meta_lines)
    if failure.existing_commitments:
        input_text += f"\n\n[dim]Existing commitments:[/]\n{_json.dumps(failure.existing_commitments, indent=2, ensure_ascii=False)}"

    pos_str = f"#{list_position} " if list_position is not None else ""
    run_str = f"(run #{failure.index})" if failure.index != list_position else ""
    title = f"Failure {pos_str}{run_str} — {failure.error_type.replace('_', ' ')} | {failure.category}/{failure.scenario} ({failure.difficulty})"
    console.print(Panel(
        input_text,
        title=title,
        border_style="red",
    ))

    # 2. Diff table
    _print_diff_table(
        failure.expected_commitments,
        failure.actual_commitments,
        failure.mismatched_fields,
    )

    # 3. Localization
    _print_localization(failure)

    console.print()


def _print_diff_table(
    expected: list[dict],
    actual: list[dict],
    mismatched_fields: list[str],
) -> None:
    if not expected and not actual:
        console.print("[dim]No commitments to compare.[/]")
        return

    max_len = max(len(expected), len(actual))
    all_fields = ["id", "committed_party", "required_action", "deadline", "context", "status", "notification"]
    mismatch_set = set(mismatched_fields)

    table = Table(
        show_header=True,
        header_style="bold",
        title="Expected vs Actual",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Field", style="bold", width=18)
    table.add_column("Expected", style="green", width=30)
    table.add_column("Actual", width=30)
    table.add_column("Match", justify="center", width=5)

    for idx in range(max_len):
        exp = expected[idx] if idx < len(expected) else {}
        act = actual[idx] if idx < len(actual) else {}

        for field in all_fields:
            ev = exp.get(field, "—")
            av = act.get(field, "—")
            ev_s = str(ev) if ev is not None else "None"
            av_s = str(av) if av is not None else "None"

            if ev_s == av_s:
                match_str = "[green]✓[/]"
                av_styled = av_s
            elif field in mismatch_set:
                match_str = "[red]✗[/]"
                av_styled = f"[red]{av_s}[/]"
            elif not expected or not actual:
                match_str = "[red]✗[/]"
                av_styled = f"[red]{av_s}[/]"
            else:
                match_str = "[yellow]~[/]"
                av_styled = f"[yellow]{av_s}[/]"

            table.add_row(
                str(idx) if field == all_fields[0] else "",
                field,
                ev_s,
                av_styled,
                match_str,
            )

    console.print(table)


def _print_localization(failure: FailureRecord) -> None:
    raw = failure.to_raw_dict()
    lf = _classify_failure(raw)
    repair_text = _REPAIR_MAP.get(lf.root_cause, ("unknown", "—"))[1]

    panel_content = (
        f"[bold cyan]Root Cause:[/] {lf.root_cause}\n"
        f"[bold]Subcause:[/] {lf.subcause}\n"
        f"[bold]Repair Type:[/] {lf.repair_type}\n"
        f"[bold]Confidence:[/] {lf.confidence:.2f}\n"
        f"[bold]Suggested Repair:[/] {repair_text}"
    )

    console.print(Panel(
        panel_content,
        title="Localization",
        border_style="cyan",
    ))


def _print_summary(failures: list[FailureRecord]) -> None:
    if not failures:
        console.print("[dim]No failures to summarize.[/]")
        return

    from collections import Counter

    by_error = Counter(f.error_type for f in failures)
    by_category = Counter(f.category for f in failures)
    by_difficulty = Counter(f.difficulty for f in failures)

    table = Table(show_header=True, header_style="bold", title="Failure Summary")
    table.add_column("Dimension", style="bold", width=16)
    table.add_column("Value", width=24)
    table.add_column("Count", justify="right", width=8)

    for et, count in by_error.most_common():
        table.add_row("Error Type", et.replace("_", " "), str(count))
    for cat, count in by_category.most_common():
        table.add_row("Category", cat, str(count))
    for diff, count in by_difficulty.most_common():
        table.add_row("Difficulty", diff, str(count))

    console.print(table)


# ─── Filtering ───────────────────────────────────────────────────────────────


def _apply_filters(
    failures: list[FailureRecord],
    filters: dict[str, str],
) -> list[FailureRecord]:
    result = failures
    for key, value in filters.items():
        if key == "error_type":
            result = [f for f in result if f.error_type == value.upper().replace(" ", "_")]
        elif key == "category":
            result = [f for f in result if f.category == value]
        elif key == "difficulty":
            result = [f for f in result if f.difficulty == value]
        elif key == "split":
            result = [f for f in result if f.split == value]
        elif key == "field":
            result = [f for f in result if value in f.mismatched_fields]
    return result


# ─── Re-run ──────────────────────────────────────────────────────────────────


_dspy_configured = False
_rerun_use_judge = False


def _ensure_dspy_configured() -> None:
    global _dspy_configured
    if not _dspy_configured:
        from app.config import settings
        from app.commitments.commitments_agent import configure_dspy

        configure_dspy(settings)
        _dspy_configured = True


def _init_judge_for_rerun() -> None:
    from app.config import settings
    from eval.llm_judge import reset_cache as reset_judge_cache, set_judge_model

    reset_judge_cache()
    set_judge_model(settings.openai_model)
    console.print("[dim]LLM judge enabled for rerun grading[/]\n")


def _rerun_example(failure: FailureRecord) -> None:
    if not failure.chat_id:
        console.print(
            "[red]Cannot rerun — input fields not available for this run "
            "(predictions.jsonl missing).[/]"
        )
        return

    _ensure_dspy_configured()

    from app.commitments.commitments_agent import CommitmentAgent
    from eval.metrics import compare_commitments

    console.print(f"\n[dim]Re-running {failure.category}/{failure.scenario}...[/]")

    agent = CommitmentAgent()
    pred = agent(
        chat_id=failure.chat_id,
        chat_name=failure.chat_name or None,
        existing_commitments_json=failure.existing_commitments_json,
        messages=failure.messages,
        current_datetime=failure.current_datetime or None,
    )

    # Rehydrate expected as Commitment objects
    expected_objs = [
        Commitment.model_validate(c) if isinstance(c, dict) else c
        for c in failure.expected_commitments
    ]

    new_mismatches = compare_commitments(expected_objs, pred.commitments, use_llm_judge=_rerun_use_judge)
    passed = len(new_mismatches) == 0

    if passed:
        console.print("[bold green]✓ PASS — no mismatches![/]")
    else:
        console.print("[bold red]✗ FAIL — still has mismatches:[/]")
        new_mismatch_fields = [m["field"] for m in new_mismatches]
        old_mismatch_fields = set(failure.mismatched_fields)
        new_mismatch_set = set(new_mismatch_fields)

        fixed_fields = old_mismatch_fields - new_mismatch_set
        still_wrong = old_mismatch_fields & new_mismatch_set
        new_problems = new_mismatch_set - old_mismatch_fields

        if fixed_fields:
            console.print(f"  [green]Fixed:[/] {', '.join(sorted(fixed_fields))}")
        if still_wrong:
            console.print(f"  [red]Still wrong:[/] {', '.join(sorted(still_wrong))}")
        if new_problems:
            console.print(f"  [yellow]New problems:[/] {', '.join(sorted(new_problems))}")

        actual_dicts = _commitments_to_dicts(pred.commitments)
        _print_diff_table(
            failure.expected_commitments,
            actual_dicts,
            new_mismatch_fields,
        )

    console.print()


# ─── Run Picker ──────────────────────────────────────────────────────────────


def _pick_run() -> Path | None:
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        console.print(
            "[yellow]No runs/ directory found. "
            "Run `python scripts/eval_runner.py` first, or use --split to run a fresh eval.[/]"
        )
        return None

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and (d / "failures.jsonl").exists()],
        key=lambda d: d.name,
        reverse=True,
    )

    if not run_dirs:
        console.print(
            "[yellow]No runs with failures.jsonl found in runs/. "
            "Run `python scripts/eval_runner.py` first, or use --split to run a fresh eval.[/]"
        )
        return None

    console.print("[bold]Available runs:[/]")
    for i, d in enumerate(run_dirs):
        console.print(f"  {i}: {d.name}")

    choice = Prompt.ask("\nSelect a run", default="0")
    try:
        idx = int(choice)
        return run_dirs[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/]")
        return None


# ─── REPL ────────────────────────────────────────────────────────────────────


_HELP_TEXT = """[bold]Commands:[/]
  [cyan]list[/] / [cyan]ls[/]          Show numbered failure table
  [cyan]<N>[/]              Show detail for failure #N
  [cyan]next[/] / [cyan]prev[/]       Navigate from detail view
  [cyan]filter <k>=<v>[/]   Filter (keys: error_type, category, difficulty, split, field)
  [cyan]filter clear[/]     Clear all filters
  [cyan]filters[/]          Show active filters
  [cyan]localize[/]         Show root-cause localization summary
  [cyan]rerun <N>[/]        Re-run failure #N through fresh agent
  [cyan]summary[/]          Show aggregate stats
  [cyan]help[/] / [cyan]h[/]          Show this help
  [cyan]quit[/] / [cyan]q[/]          Exit"""


def _repl(failures: list[FailureRecord]) -> None:
    filters: dict[str, str] = {}
    current_detail: int | None = None

    console.print(f"\n[bold green]Eval Inspector loaded {len(failures)} failures.[/]\n")
    console.print(_HELP_TEXT)
    console.print()

    _print_failure_list(failures)
    console.print()

    while True:
        try:
            prompt_text = "[bold cyan]inspector>[/]"
            cmd = Prompt.ask(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "q"):
            console.print("[dim]Bye.[/]")
            break

        if command in ("help", "h"):
            console.print(_HELP_TEXT)
            continue

        if command in ("list", "ls"):
            filtered = _apply_filters(failures, filters)
            _print_failure_list(filtered)
            continue

        if command in ("summary",):
            filtered = _apply_filters(failures, filters)
            _print_summary(filtered)
            continue

        if command == "filter":
            if not arg:
                console.print("[dim]Usage: filter <key>=<value>[/]")
                console.print("[dim]Keys: error_type, category, difficulty, split, field[/]")
                continue

            if arg.lower() == "clear":
                filters.clear()
                console.print("[green]Filters cleared.[/]")
                filtered = _apply_filters(failures, filters)
                _print_failure_list(filtered)
                continue

            if "=" not in arg:
                console.print("[red]Invalid filter format. Use key=value[/]")
                continue

            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()
            valid_keys = {"error_type", "category", "difficulty", "split", "field"}
            if key not in valid_keys:
                console.print(f"[red]Invalid filter key: {key}. Valid: {', '.join(sorted(valid_keys))}[/]")
                continue

            filters[key] = value
            filtered = _apply_filters(failures, filters)
            console.print(f"[green]Filter added: {key}={value}[/]")
            _print_failure_list(filtered)
            continue

        if command == "filters":
            if not filters:
                console.print("[dim]No active filters.[/]")
            else:
                for k, v in filters.items():
                    console.print(f"  [cyan]{k}[/] = [bold]{v}[/]")
            continue

        if command == "localize":
            filtered = _apply_filters(failures, filters)
            raw_dicts = [f.to_raw_dict() for f in filtered]
            localized = localize(raw_dicts)
            print_localization_summary(localized)
            continue

        if command == "rerun":
            if not arg:
                console.print("[red]Usage: rerun <N>[/]")
                continue
            try:
                idx = int(arg)
            except ValueError:
                console.print("[red]Invalid number.[/]")
                continue
            filtered = _apply_filters(failures, filters)
            if idx < 0 or idx >= len(filtered):
                console.print(f"[red]Index out of range (0-{len(filtered) - 1}).[/]")
                continue
            _rerun_example(filtered[idx])
            continue

        if command in ("next", "prev"):
            if current_detail is None:
                console.print("[dim]No detail view active. Use <N> first.[/]")
                continue
            filtered = _apply_filters(failures, filters)
            if command == "next":
                current_detail = min(current_detail + 1, len(filtered) - 1)
            else:
                current_detail = max(current_detail - 1, 0)
            _print_failure_detail(filtered[current_detail], list_position=current_detail)
            continue

        # Try parsing as a number (detail view)
        try:
            idx = int(command)
            filtered = _apply_filters(failures, filters)
            if idx < 0 or idx >= len(filtered):
                console.print(f"[red]Index out of range (0-{len(filtered) - 1}).[/]")
                continue
            current_detail = idx
            _print_failure_detail(filtered[idx], list_position=idx)
            continue
        except ValueError:
            pass

        console.print(f"[red]Unknown command: {command}. Type 'help' for commands.[/]")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive eval failure inspector"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name under runs/ (e.g. 20260707_183434-judge)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test", "challenge"],
        help="Run a fresh eval on this split, then enter inspector",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (fresh run only)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM judge for fresh run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the OpenAI model (fresh run only)",
    )
    args = parser.parse_args()

    global _dspy_configured, _rerun_use_judge
    failures: list[FailureRecord]

    if args.split:
        # Fresh run
        if args.model:
            from app.config import settings

            settings.openai_model = args.model

        _dspy_configured = True
        _rerun_use_judge = args.llm_judge

        from app.config import settings
        from app.commitments.commitments_agent import configure_dspy

        configure_dspy(settings)

        if args.llm_judge:
            from eval.llm_judge import reset_cache as reset_judge_cache, set_judge_model

            reset_judge_cache()
            set_judge_model(settings.openai_model)
            console.print("[dim]LLM judge enabled for required_action, deadline, context[/]\n")

        failures = _load_from_fresh(
            args.split,
            limit=args.limit,
            use_llm_judge=args.llm_judge,
        )
    elif args.run:
        run_dir = Path(__file__).resolve().parent.parent / "runs" / args.run
        if not run_dir.is_dir():
            console.print(f"[red]Run directory not found: {run_dir}[/]")
            raise SystemExit(1)
        failures = _load_from_run(run_dir)
        if "-judge" in args.run:
            _rerun_use_judge = True
            _init_judge_for_rerun()
    else:
        # Interactive picker
        run_dir = _pick_run()
        if run_dir is None:
            raise SystemExit(0)
        failures = _load_from_run(run_dir)
        if "-judge" in run_dir.name:
            _rerun_use_judge = True
            _init_judge_for_rerun()

    if not failures:
        console.print("[green]No failures found! All examples passed.[/]")
        raise SystemExit(0)

    _repl(failures)


if __name__ == "__main__":
    main()
