"""CLI entry point for running DSPy commitment extraction evaluations.

Usage:
    # Full devset
    .venv/bin/python scripts/eval_runner.py

    # Run a specific split
    .venv/bin/python scripts/eval_runner.py --split dev
    .venv/bin/python scripts/eval_runner.py --split test
    .venv/bin/python scripts/eval_runner.py --split train

    # Run all splits with aggregated results table
    .venv/bin/python scripts/eval_runner.py --all

    # First 3 examples only
    .venv/bin/python scripts/eval_runner.py --limit 3

    # Use a specific model
    .venv/bin/python scripts/eval_runner.py --model gpt-4o-mini

    # Verbose: show actual vs expected per example
    .venv/bin/python scripts/eval_runner.py --verbose
"""

import argparse
import json as _json
import shutil as _shutil
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config import settings
from app.commitments.commitments_agent import CommitmentAgent, configure_dspy
from eval.dataset import build_devset
from eval.metrics import (
    _token_f1,
    _word_overlap,
    act_vs_ignore_metric,
    compare_commitments,
    commitment_metric,
)
from eval.llm_judge import (
    reset_cache as reset_judge_cache,
    save_verdicts as save_judge_verdicts,
    set_judge_model,
)

console = Console(record=True)

_DETAIL_FIELDS = ["committed_party", "required_action", "deadline", "context", "status"]

_SPLITS = {
    "train": "trainset.json",
    "dev": "devset.json",
    "test": "testset.json",
    "challenge": "challenge_act_ignore.json",
}


def _split_path(split: str) -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "evals" / _SPLITS[split]


def _copy_dataset_to_run(run_dir: Path, splits: list[str]) -> None:
    """Copy the JSON dataset files used by this run into the run directory."""
    dest = run_dir / "dataset"
    dest.mkdir(exist_ok=True)
    for split in splits:
        src = _split_path(split)
        if src.exists():
            _shutil.copy2(src, dest / src.name)


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
    use_llm_judge: bool = False,
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
        mismatches = compare_commitments(ex.expected_commitments, pred.commitments, use_llm_judge=use_llm_judge)
        matched = not mismatches
        avi = act_vs_ignore_metric(ex, pred)
        metric_score = commitment_metric(ex, pred, use_llm_judge=use_llm_judge)

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
                    # LLM judge fallback for semantic fields
                    if not field_matches[field] and use_llm_judge and field in ("required_action", "deadline", "context"):
                        from eval.llm_judge import judge_field_safe
                        field_matches[field] = judge_field_safe(field, ev_s, av_s)

        per_example.append({
            "index": i,
            "split": split,
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
            "existing_commitments_json": getattr(ex, "existing_commitments_json", "[]"),
            "current_datetime": getattr(ex, "current_datetime", ""),
            "chat_id": getattr(ex, "chat_id", ""),
            "chat_name": getattr(ex, "chat_name", ""),
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

    expected_ignore = tn + fp
    over_extraction_rate = fp / expected_ignore if expected_ignore > 0 else 0.0

    console.print(f"[bold]Act/Ignore Score: {avi_score:.2f} ({sum(e['avi'] for e in per_example):.0f}/{len(per_example)})[/]")
    if total_expected > 0:
        console.print(f"[bold]Commitment Metric: {total_matched:.0f}/{total_expected} ({commit_metric * 100:.1f}%)[/]")
    else:
        console.print(f"[bold]Commitment Metric: N/A[/]")
    if expected_ignore > 0:
        console.print(f"[bold]Over-Extraction Rate: {fp}/{expected_ignore} ({over_extraction_rate * 100:.0f}%)[/]")
    else:
        console.print(f"[bold]Over-Extraction Rate: N/A[/]")
    console.print()

    # Print failure table
    _print_failure_table(split, per_example)

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
        "over_extraction_rate": over_extraction_rate,
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


def _print_failure_table(split: str, per_example: list[dict]) -> None:
    """Print a table of all failing examples: FP, FN, and TP with field mismatches."""
    failures: list[dict] = []

    for e in per_example:
        conf = e["confusion"]
        if conf == "fp":
            failures.append({
                "category": e["category"],
                "difficulty": e["difficulty"],
                "scenario": e["scenario"],
                "error_type": "FALSE POSITIVE",
                "detail": f"Expected: [] | Actual: {len(e['pred_commitments'])} commitment(s)",
                "messages": e["messages"],
            })
        elif conf == "fn":
            failures.append({
                "category": e["category"],
                "difficulty": e["difficulty"],
                "scenario": e["scenario"],
                "error_type": "FALSE NEGATIVE",
                "detail": f"Expected: {len(e['expected_commitments'])} | Actual: []",
                "messages": e["messages"],
            })
        elif conf == "tp" and e.get("mismatches"):
            mismatch_fields = ", ".join(m["field"] for m in e["mismatches"])
            failures.append({
                "category": e["category"],
                "difficulty": e["difficulty"],
                "scenario": e["scenario"],
                "error_type": "FIELD MISMATCH",
                "detail": f"Fields: {mismatch_fields}",
                "messages": e["messages"],
            })

    if not failures:
        console.print("[green]No failures in this split.[/]")
        console.print()
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"{split.upper()} — Failures ({len(failures)})",
        show_lines=True,
    )
    table.add_column("Error Type", style="bold red", width=16)
    table.add_column("Category", width=22)
    table.add_column("Difficulty", justify="center", width=10)
    table.add_column("Scenario", width=30)
    table.add_column("Detail", width=40)
    table.add_column("Messages", style="dim", width=50)

    for f in failures:
        table.add_row(
            f["error_type"],
            f["category"],
            f["difficulty"],
            f["scenario"],
            f["detail"],
            f["messages"].replace("\n", " | "),
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
    table.add_column("Over-Ext", justify="right", width=10)

    for r in results:
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        expected_ignore = tn + fp
        over_ext = fp / expected_ignore if expected_ignore > 0 else 0.0

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
            f"[red]{over_ext:.0%}[/]" if expected_ignore > 0 else "—",
        )

    console.print(table)


def _commitments_to_dicts(commitments: list) -> list[dict]:
    """Convert commitment objects (pydantic or dict) to plain dicts."""
    out = []
    for c in commitments:
        if hasattr(c, "model_dump"):
            out.append(c.model_dump(mode="json"))
        elif isinstance(c, dict):
            out.append(c)
        else:
            out.append(str(c))
    return out


def _save_predictions_jsonl(
    run_dir: Path,
    results: list[dict],
    *,
    run_id: str = "",
    agent_model: str = "",
) -> None:
    """Save all predictions (agent outputs) as JSONL for frozen regrading."""
    path = run_dir / "predictions.jsonl"
    input_keys = ["chat_id", "chat_name", "current_datetime", "existing_commitments_json", "messages"]
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            for e in r["per_example"]:
                row = {
                    "run_id": run_id,
                    "split": e["split"],
                    "example_index": e["index"],
                    "example_id": f"{e['category']}/{e['scenario']}",
                    "category": e["category"],
                    "scenario": e["scenario"],
                    "difficulty": e["difficulty"],
                    "input_keys": input_keys,
                    "inputs": {
                        "chat_id": e.get("chat_id", ""),
                        "chat_name": e.get("chat_name", ""),
                        "current_datetime": e.get("current_datetime", ""),
                        "existing_commitments_json": e.get("existing_commitments_json", "[]"),
                        "messages": e["messages"],
                    },
                    "expected_commitments": _commitments_to_dicts(e["expected_commitments"]),
                    "actual_commitments": _commitments_to_dicts(e["pred_commitments"]),
                    "agent_model": agent_model,
                    "created_at": datetime.now().isoformat(),
                }
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _save_failures_jsonl(run_dir: Path, results: list[dict]) -> None:
    """Save all failures across splits as a JSONL file for debugging."""
    path = run_dir / "failures.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            for e in r["per_example"]:
                conf = e["confusion"]
                if conf == "fp":
                    error_type = "FALSE_POSITIVE"
                    mismatched_fields = []
                elif conf == "fn":
                    error_type = "FALSE_NEGATIVE"
                    mismatched_fields = []
                elif conf == "tp" and e.get("mismatches"):
                    error_type = "FIELD_MISMATCH"
                    mismatched_fields = [m["field"] for m in e["mismatches"]]
                else:
                    continue

                row = {
                    "split": e["split"],
                    "category": e["category"],
                    "scenario": e["scenario"],
                    "difficulty": e["difficulty"],
                    "messages": e["messages"],
                    "existing_commitments": _json.loads(e.get("existing_commitments_json", "[]")),
                    "expected_commitments": _commitments_to_dicts(e["expected_commitments"]),
                    "actual_commitments": _commitments_to_dicts(e["pred_commitments"]),
                    "error_type": error_type,
                    "mismatched_fields": mismatched_fields,
                }
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _save_split_md(run_dir: Path, split: str, result: dict) -> None:
    """Save a single split's recorded console output as a markdown file."""
    text = console.export_text(clear=True, styles=False)
    md = f"# {split.upper()} Split\n\n"
    md += f"**N={result['n']}** | TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}\n\n"
    md += f"```\n{text}\n```\n"
    (run_dir / f"{split}.md").write_text(md, encoding="utf-8")


def _save_summary_md(run_dir: Path, run_id: str, results: list[dict], model: str) -> None:
    """Save aggregated summary as a markdown file."""
    text = console.export_text(clear=True, styles=False)
    md = f"# Eval Run {run_id}\n\n"
    md += f"**Model:** {model}\n\n"
    md += f"**Timestamp:** {datetime.now().isoformat()}\n\n"
    md += f"**Splits:** {', '.join(r['split'] for r in results)}\n\n"
    md += f"```\n{text}\n```\n\n"
    md += "## Per-Split Summary\n\n"
    md += "| Split | N | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy | Over-Ext |\n"
    md += "|-------|---|----|----|----|----|-----------|--------|----|----------|----------|\n"
    for r in results:
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        expected_ignore = tn + fp
        over_ext = f"{fp / expected_ignore:.0%}" if expected_ignore > 0 else "—"
        md += f"| {r['split'].upper()} | {r['n']} | {tp} | {fp} | {fn} | {tn} | {precision:.2f} | {recall:.2f} | {f1:.2f} | {accuracy:.2f} | {over_ext} |\n"
    md += "\n"
    (run_dir / "summary.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy commitment extraction eval")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test", "challenge"],
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
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge for semantic field matching (required_action, deadline, context)",
    )
    parser.add_argument(
        "--freeze-judge",
        action="store_true",
        help="Save judge verdicts to judge_verdicts.jsonl (implies --llm-judge)",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save reports as markdown files under runs/<run_id>/ (default: on, use --no-save to disable)",
    )
    args = parser.parse_args()

    if args.model:
        settings.openai_model = args.model

    configure_dspy(settings)

    if args.freeze_judge:
        args.llm_judge = True

    if args.llm_judge:
        reset_judge_cache()
        set_judge_model(settings.openai_model)
        label = "LLM judge enabled (+freeze)" if args.freeze_judge else "LLM judge enabled"
        console.print(f"[dim]{label} for required_action, deadline, context[/]\n")

    run_dir: Path | None = None
    run_id: str | None = None
    if args.save:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.llm_judge:
            run_id += "-judge"
        run_dir = Path(__file__).resolve().parent.parent / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]Saving reports to: {run_dir}[/]\n")

    if args.all:
        results = []
        for split in ["train", "dev", "test"]:
            r = _run_split(split, limit=args.limit, verbose=args.verbose, use_llm_judge=args.llm_judge)
            results.append(r)
            if run_dir:
                _save_split_md(run_dir, split, r)
        console.print()
        _print_aggregated_table(results)
        if run_dir:
            _save_summary_md(run_dir, run_id, results, settings.openai_model)
            _save_failures_jsonl(run_dir, results)
            _save_predictions_jsonl(run_dir, results, run_id=run_id or "", agent_model=settings.openai_model)
            _copy_dataset_to_run(run_dir, ["train", "dev", "test"])
            if args.freeze_judge:
                save_judge_verdicts(run_dir / "judge_verdicts.jsonl")
                console.print(f"[dim]Judge verdicts saved to: {run_dir}/judge_verdicts.jsonl[/]")
            console.print(f"\n[bold green]Reports saved to: {run_dir}/[/]")
            console.print(f"[dim]Predictions saved to: {run_dir}/predictions.jsonl[/]")
        sys.exit(0)

    split = args.split or "dev"
    r = _run_split(split, limit=args.limit, verbose=args.verbose, use_llm_judge=args.llm_judge)
    if run_dir:
        _save_split_md(run_dir, split, r)
        _save_failures_jsonl(run_dir, [r])
        _save_predictions_jsonl(run_dir, [r], run_id=run_id or "", agent_model=settings.openai_model)
        _copy_dataset_to_run(run_dir, [split])
        if args.freeze_judge:
            save_judge_verdicts(run_dir / "judge_verdicts.jsonl")
            console.print(f"[dim]Judge verdicts saved to: {run_dir}/judge_verdicts.jsonl[/]")
        console.print(f"\n[bold green]Report saved to: {run_dir}/{split}.md[/]")
        console.print(f"[dim]Predictions saved to: {run_dir}/predictions.jsonl[/]")


if __name__ == "__main__":
    main()
