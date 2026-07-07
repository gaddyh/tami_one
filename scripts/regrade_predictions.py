"""Regrade frozen predictions to isolate metric/judge changes from agent variance.

Saves agent predictions to JSONL during eval runs, then regrades from frozen
predictions. This eliminates agent stochasticity. Regrading can run with
deterministic metrics only, fresh LLM judge calls, or frozen judge verdicts.

Usage:
    # Pure deterministic metric (zero LLM calls)
    python scripts/regrade_predictions.py runs/<run_id>/predictions.jsonl

    # Frozen agent, fresh judge calls
    python scripts/regrade_predictions.py runs/<run_id>/predictions.jsonl --llm-judge

    # Frozen agent, save judge verdicts
    python scripts/regrade_predictions.py runs/<run_id>/predictions.jsonl --llm-judge --freeze-judge

    # Fully deterministic replay (frozen agent + frozen judge)
    python scripts/regrade_predictions.py runs/<run_id>/predictions.jsonl \
        --llm-judge --judge-cache runs/<run_id>/judge_verdicts.jsonl --no-new-judge-calls
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from datetime import datetime
from pathlib import Path

import dspy
from rich.console import Console

from app.commitments.models import Commitment
from app.config import settings
from app.commitments.commitments_agent import configure_dspy
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
    load_verdicts as load_judge_verdicts,
    set_offline_mode,
    set_judge_model,
    get_cache_size,
    JudgeCacheMiss,
)

# Reuse reporting functions from eval_runner
from scripts.eval_runner import (
    _DETAIL_FIELDS,
    _print_confusion_matrix,
    _print_category_breakdown,
    _print_difficulty_breakdown,
    _print_category_difficulty_matrix,
    _print_field_accuracy,
    _print_failure_table,
    _print_aggregated_table,
    _save_failures_jsonl,
    _save_split_md,
    _save_summary_md,
    _commitments_to_dicts,
)

console = Console(record=True)


def _load_predictions(path: Path) -> list[dict]:
    """Load predictions.jsonl and return list of rows."""
    with path.open("r", encoding="utf-8") as f:
        return [_json.loads(line) for line in f if line.strip()]


def _regrade_split(
    rows: list[dict],
    split: str,
    *,
    use_llm_judge: bool = False,
    verbose: bool = False,
) -> dict:
    """Regrade a single split from frozen predictions."""
    split_rows = [r for r in rows if r["split"] == split]
    if not split_rows:
        raise ValueError(f"No predictions found for split '{split}'")

    console.print(
        f"[bold cyan]Regrading {split} split ({len(split_rows)} examples from frozen predictions)...[/]\n"
    )

    per_example: list[dict] = []

    for i, row in enumerate(split_rows):
        # Reconstruct expected and actual commitments
        expected = [Commitment.model_validate(c) for c in row["expected_commitments"]]
        actual = [Commitment.model_validate(c) for c in row["actual_commitments"]]

        # Reconstruct dspy Example and Prediction
        inputs = row["inputs"]
        ex = dspy.Example(
            chat_id=inputs["chat_id"],
            chat_name=inputs.get("chat_name"),
            current_datetime=inputs.get("current_datetime", ""),
            existing_commitments_json=inputs.get("existing_commitments_json", "[]"),
            messages=inputs["messages"],
            expected_commitments=expected,
            category=row.get("category", ""),
            scenario=row.get("scenario", ""),
            difficulty=row.get("difficulty", ""),
        ).with_inputs(
            "chat_id",
            "chat_name",
            "current_datetime",
            "existing_commitments_json",
            "messages",
        )
        pred = dspy.Prediction(commitments=actual)

        expected_empty = len(expected) == 0
        actual_empty = len(actual) == 0
        mismatches = compare_commitments(expected, actual, use_llm_judge=use_llm_judge)
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
            exp_list = expected
            act_list = actual
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
                    if not field_matches[field] and use_llm_judge and field in ("required_action", "deadline", "context"):
                        from eval.llm_judge import judge_field_safe
                        field_matches[field] = judge_field_safe(field, ev_s, av_s)

        per_example.append({
            "index": i,
            "split": split,
            "category": row.get("category", "—") or "—",
            "scenario": row.get("scenario", "—") or "—",
            "difficulty": row.get("difficulty", "—") or "—",
            "expected_empty": expected_empty,
            "actual_empty": actual_empty,
            "confusion": confusion,
            "matched": matched,
            "avi": avi,
            "metric_score": metric_score,
            "mismatches": mismatches,
            "field_matches": field_matches,
            "messages": inputs["messages"],
            "existing_commitments_json": inputs.get("existing_commitments_json", "[]"),
            "current_datetime": inputs.get("current_datetime", ""),
            "chat_id": inputs.get("chat_id", ""),
            "chat_name": inputs.get("chat_name", ""),
            "pred_commitments": actual,
            "expected_commitments": expected,
        })

    # Compute aggregate stats
    tp = sum(1 for e in per_example if e["confusion"] == "tp")
    fp = sum(1 for e in per_example if e["confusion"] == "fp")
    fn = sum(1 for e in per_example if e["confusion"] == "fn")
    tn = sum(1 for e in per_example if e["confusion"] == "tn")

    total_expected = sum(len(e["expected_commitments"]) for e in per_example)
    total_matched = sum(e["metric_score"] * len(e["expected_commitments"]) for e in per_example)

    _print_confusion_matrix(split, tp, fp, fn, tn)
    _print_category_breakdown(per_example)
    _print_difficulty_breakdown(per_example)
    _print_category_difficulty_matrix(per_example)
    _print_field_accuracy(per_example)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Regrade frozen predictions without calling the LLM agent")
    parser.add_argument(
        "predictions",
        type=str,
        help="Path to predictions.jsonl from a previous eval run",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "dev", "test", "challenge"],
        help="Which split to regrade (default: all splits in the file)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge for semantic field matching",
    )
    parser.add_argument(
        "--freeze-judge",
        action="store_true",
        help="Save judge verdicts to judge_verdicts.jsonl (implies --llm-judge)",
    )
    parser.add_argument(
        "--judge-cache",
        type=str,
        default=None,
        help="Path to judge_verdicts.jsonl to load as cache",
    )
    parser.add_argument(
        "--no-new-judge-calls",
        action="store_true",
        help="Only use cached judge verdicts, never call the LLM (implies offline mode)",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save reports as markdown files under runs/<run_id>-regrade/ (default: on)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print actual vs expected for each example",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")

    if args.no_new_judge_calls:
        args.llm_judge = True
    if args.freeze_judge:
        args.llm_judge = True

    # Configure DSPy (needed for judge, even if agent is frozen)
    configure_dspy(settings)

    # Judge setup
    if args.llm_judge:
        reset_judge_cache()
        set_judge_model(settings.openai_model)

        if args.judge_cache:
            cache_path = Path(args.judge_cache)
            if cache_path.exists():
                count = load_judge_verdicts(cache_path)
                console.print(f"[dim]Loaded {count} judge verdicts from {cache_path}[/]")
            else:
                console.print(f"[yellow]Warning: judge cache not found: {cache_path}[/]")

        if args.no_new_judge_calls:
            set_offline_mode(True)
            console.print("[dim]Judge offline mode: no new LLM calls, cache misses will raise[/]\n")
        else:
            console.print("[dim]LLM judge enabled for required_action, deadline, context[/]\n")
    else:
        console.print("[dim]No judge — pure deterministic metrics[/]\n")

    # Load frozen predictions
    rows = _load_predictions(pred_path)
    console.print(f"[bold green]Loaded {len(rows)} frozen predictions from {pred_path}[/]\n")

    # Determine which splits to run
    available_splits = sorted({r["split"] for r in rows})
    splits_to_run = [args.split] if args.split else available_splits

    # Setup output directory
    run_dir: Path | None = None
    run_id: str | None = None
    if args.save:
        source_name = pred_path.parent.name
        run_id = f"{source_name}-regrade-{datetime.now().strftime('%H%M%S')}"
        if args.llm_judge:
            run_id += "-judge"
        run_dir = Path(__file__).resolve().parent.parent / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]Saving reports to: {run_dir}[/]\n")

    # Regrade each split
    results = []
    for split in splits_to_run:
        try:
            r = _regrade_split(rows, split, use_llm_judge=args.llm_judge, verbose=args.verbose)
        except JudgeCacheMiss as exc:
            console.print(f"\n[bold red]FATAL: JudgeCacheMiss in split {split}[/]")
            console.print(f"[red]{exc}[/]")
            console.print(
                "[dim]The frozen judge cache is incomplete. "
                "Re-run with --freeze-judge to generate a complete cache, "
                "or remove --no-new-judge-calls to allow fresh judge calls.[/]"
            )
            sys.exit(1)
        results.append(r)
        if run_dir:
            _save_split_md(run_dir, split, r)

    if len(results) > 1:
        console.print()
        _print_aggregated_table(results)

    # Save outputs
    if run_dir:
        _save_failures_jsonl(run_dir, results)
        if args.freeze_judge:
            save_judge_verdicts(run_dir / "judge_verdicts.jsonl")
            console.print(f"[dim]Judge verdicts saved to: {run_dir}/judge_verdicts.jsonl[/]")

        if len(results) > 1:
            _save_summary_md(run_dir, run_id, results, settings.openai_model)

        console.print(f"\n[bold green]Reports saved to: {run_dir}/[/]")

    if args.llm_judge:
        console.print(f"[dim]Judge cache size: {get_cache_size()}[/]")


if __name__ == "__main__":
    main()
