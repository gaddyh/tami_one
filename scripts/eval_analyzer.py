"""Eval analyzer agent: analyze failures, propose fixes, apply, and verify.

Three subcommands forming an interactive loop:
  analyze  — Read failures from a run, use LLM to propose concrete fixes.
  apply    — Apply selected fixes (prompt edits interactive, metric edits explicit approval).
  verify   — Confirm improvement: metric changes via frozen regrade, prompt changes via held-out re-run.

Usage:
    python scripts/eval_analyzer.py analyze --run 20260707_224735-judge
    python scripts/eval_analyzer.py apply --proposal runs/20260707_224735-judge/proposal.json
    python scripts/eval_analyzer.py verify --run 20260707_224735-judge --proposal runs/20260707_224735-judge/proposal.json
"""

from __future__ import annotations

import argparse
import ast
import json as _json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from app.commitments.commitments_agent import configure_dspy
from app.config import settings
from eval.localize import localize, _REPAIR_MAP, _REPAIR_COST, LocalizedFailure

console = Console()

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

# Root cause → metric function names (for populating current_metric_code input)
_METRIC_CODE_MAP: dict[str, list[str]] = {
    "deadline_normalization": ["_normalize_deadline", "_deadline_equal"],
    "context_metric_noise": ["_word_overlap"],
    "required_action_normalization": ["_token_f1"],
    "update_vs_new_matching": ["compare_commitments"],
}


# ─── LLM Signatures ──────────────────────────────────────────────────────────


class AnalyzeModelFailure(dspy.Signature):
    """
    You are analyzing evaluation failures from a commitment extraction agent.
    The deterministic classifier says this is a model failure (prompt needs improvement).
    Confirm or dispute this, then propose a concrete prompt edit.

    Edit types:
    - anchored_replace: modify an existing rule. Provide old_text (exact verbatim
      quote from the current prompt) and new_text.
    - insert_after_heading: add a new rule. Provide anchor_heading (an existing
      section heading in the prompt) and new_text (the new bullet(s) to insert).
    """
    root_cause: str = dspy.InputField()
    subcause: str = dspy.InputField()
    repair_type_prior: str = dspy.InputField()
    failures_summary: str = dspy.InputField()
    current_prompt: str = dspy.InputField()

    confirmed: bool = dspy.OutputField(desc="True if this is indeed a model failure")
    edit_type: str = dspy.OutputField(desc="anchored_replace | insert_after_heading")
    anchor_heading: str = dspy.OutputField(desc="For insert_after_heading: existing heading to insert under. Empty for anchored_replace.")
    old_text: str = dspy.OutputField(desc="For anchored_replace: exact text to find. Empty for insert_after_heading.")
    new_text: str = dspy.OutputField(desc="Replacement text or new bullet(s) to insert")
    rationale: str = dspy.OutputField()


class AnalyzeMetricBug(dspy.Signature):
    """
    You are analyzing evaluation failures from a commitment extraction agent.
    The deterministic classifier says this is a metric bug (metric code has a
    normalization issue). Confirm or dispute this, then propose a full
    replacement function body.
    """
    root_cause: str = dspy.InputField()
    subcause: str = dspy.InputField()
    repair_type_prior: str = dspy.InputField()
    failures_summary: str = dspy.InputField()
    current_metric_code: str = dspy.InputField(desc="Current source code of the relevant metric functions")

    confirmed: bool = dspy.OutputField(desc="True if this is indeed a metric bug")
    target_function: str = dspy.OutputField(desc="Function name to replace")
    new_function_body: str = dspy.OutputField(desc="Complete replacement function source")
    rationale: str = dspy.OutputField()


class AnalyzeYamlIssue(dspy.Signature):
    """
    You are analyzing evaluation failures from a commitment extraction agent.
    The deterministic classifier suggests this might be a label issue (the
    expected value in the dataset is wrong). Confirm or dispute this.
    If confirmed, cite the exact message text that contradicts the label.
    """
    root_cause: str = dspy.InputField()
    subcause: str = dspy.InputField()
    failures_summary: str = dspy.InputField()

    confirmed: bool = dspy.OutputField(desc="True if the expected label is wrong")
    evidence_quote: str = dspy.OutputField(desc="Exact message text contradicting the label")
    rationale: str = dspy.OutputField()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_failures(run_dir: Path) -> list[dict[str, Any]]:
    fpath = run_dir / "failures.jsonl"
    if not fpath.exists():
        raise SystemExit(f"No failures.jsonl in {run_dir}")
    with fpath.open() as f:
        return [_json.loads(line) for line in f if line.strip()]


def _load_run_meta(run_dir: Path) -> dict[str, Any]:
    mpath = run_dir / "run_meta.json"
    if mpath.exists():
        return _json.loads(mpath.read_text())
    return {}


def _get_prompt_docstring() -> str:
    """Extract the ExtractCommitments docstring from commitments_agent.py."""
    src = (ROOT / "app" / "commitments" / "commitments_agent.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExtractCommitments":
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                return node.body[0].value.value
    return ""


def _get_metric_code(function_names: list[str]) -> str:
    """Extract source code of named functions from metrics.py."""
    src = (ROOT / "eval" / "metrics.py").read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") else start + 1
            result.append("".join(lines[start:end]))
    return "\n\n".join(result)


def _format_failures_summary(failures: list[dict[str, Any]]) -> str:
    """Format a group of failures into a compact summary for the LLM."""
    parts = []
    for f in failures:
        msgs = f.get("messages", "")[:200]
        exp = f.get("expected_commitments", [])
        act = f.get("actual_commitments", [])
        fields = f.get("mismatched_fields", [])
        parts.append(
            f"  split={f.get('split','')} scenario={f.get('scenario','')}\n"
            f"  messages: {msgs}\n"
            f"  expected: {_json.dumps(exp)[:300]}\n"
            f"  actual: {_json.dumps(act)[:300]}\n"
            f"  mismatched_fields: {fields}"
        )
    return "\n".join(parts)


def _group_failures(
    failures: list[dict[str, Any]], localized: list[LocalizedFailure]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group failures by (root_cause, subcause) using localization results."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for f, lf in zip(failures, localized):
        groups[(lf.root_cause, lf.subcause)].append(f)
    return groups


def _get_repair_type(root_cause: str) -> str:
    return _REPAIR_MAP.get(root_cause, ("unknown", ""))[0]


# ─── Proposal validation ────────────────────────────────────────────────────


def _validate_anchored_replace(proposal: dict[str, Any], file_content: str) -> tuple[bool, str]:
    old_text = proposal.get("old_text", "")
    if not old_text:
        return False, "old_text is empty"
    if old_text not in file_content:
        # Show a snippet to help debug
        snippet = old_text[:80].replace("\n", "\\n")
        return False, f"old_text not found in file: {snippet}..."
    return True, ""


def _validate_insert_after_heading(proposal: dict[str, Any], file_content: str) -> tuple[bool, str]:
    heading = proposal.get("anchor_heading", "")
    if not heading:
        return False, "anchor_heading is empty"
    if heading not in file_content:
        return False, f"anchor_heading not found in file: {heading!r}"
    new_text = proposal.get("new_text", "")
    if not new_text:
        return False, "new_text is empty"
    return True, ""


def _validate_function_replace(proposal: dict[str, Any]) -> tuple[bool, str]:
    body = proposal.get("new_function_body", "")
    if not body:
        return False, "new_function_body is empty"
    try:
        ast.parse(body)
    except SyntaxError as e:
        return False, f"ast.parse failed: {e}"
    func_name = proposal.get("target_function", "")
    if not func_name:
        return False, "target_function is empty"
    # Check function exists in metrics.py
    src = (ROOT / "eval" / "metrics.py").read_text()
    tree = ast.parse(src)
    found = any(
        isinstance(node, ast.FunctionDef) and node.name == func_name
        for node in ast.walk(tree)
    )
    if not found:
        return False, f"function {func_name!r} not found in eval/metrics.py"
    return True, ""


def _validate_proposal(proposal: dict[str, Any]) -> tuple[bool, str]:
    edit_type = proposal.get("edit_type", "")
    target_file = proposal.get("target_file", "")
    file_content = ""
    if target_file:
        fpath = ROOT / target_file
        if fpath.exists():
            file_content = fpath.read_text()

    if edit_type == "anchored_replace":
        return _validate_anchored_replace(proposal, file_content)
    elif edit_type == "insert_after_heading":
        return _validate_insert_after_heading(proposal, file_content)
    elif edit_type == "function_replace":
        return _validate_function_replace(proposal)
    elif edit_type == "flag_only":
        return True, ""
    else:
        return False, f"unknown edit_type: {edit_type!r}"


# ─── Analyze command ─────────────────────────────────────────────────────────


def _route_and_analyze(
    root_cause: str,
    subcause: str,
    repair_type: str,
    failures: list[dict[str, Any]],
    current_prompt: str,
) -> dict[str, Any] | None:
    """Route to the appropriate LLM signature based on repair_type prior."""
    failures_summary = _format_failures_summary(failures)
    failures_addrs = [f"{f.get('split','')}/{f.get('scenario','')}" for f in failures]

    def _base_proposal(cls: str, edit_type: str, **extra) -> dict[str, Any]:
        return {
            "classification": cls,
            "root_cause": root_cause,
            "subcause": subcause,
            "repair_type_prior": repair_type,
            "failures_addressed": failures_addrs,
            "edit_type": edit_type,
            **extra,
        }

    # Routing logic
    if repair_type in ("metric", "metric_or_postprocess", "postprocess"):
        # Try metric bug first
        metric_funcs = _METRIC_CODE_MAP.get(root_cause, [])
        current_metric_code = _get_metric_code(metric_funcs) if metric_funcs else ""

        predictor = dspy.Predict(AnalyzeMetricBug)
        result = predictor(
            root_cause=root_cause,
            subcause=subcause,
            repair_type_prior=repair_type,
            failures_summary=failures_summary,
            current_metric_code=current_metric_code or "(no metric code mapping for this root cause)",
        )
        if result.confirmed:
            return _base_proposal(
                "metric", "function_replace",
                target_file="eval/metrics.py",
                target_function=result.target_function,
                new_function_body=result.new_function_body,
                rationale=result.rationale,
            )
        # Fall through to model failure
        predictor = dspy.Predict(AnalyzeModelFailure)
        result = predictor(
            root_cause=root_cause,
            subcause=subcause,
            repair_type_prior=repair_type,
            failures_summary=failures_summary,
            current_prompt=current_prompt,
        )
        if result.confirmed:
            return _base_proposal(
                "model", result.edit_type,
                target_file="app/commitments/commitments_agent.py",
                anchor_heading=result.anchor_heading,
                old_text=result.old_text,
                new_text=result.new_text,
                rationale=result.rationale,
            )

    elif repair_type == "signature_rule":
        # Try model failure first
        predictor = dspy.Predict(AnalyzeModelFailure)
        result = predictor(
            root_cause=root_cause,
            subcause=subcause,
            repair_type_prior=repair_type,
            failures_summary=failures_summary,
            current_prompt=current_prompt,
        )
        if result.confirmed:
            return _base_proposal(
                "model", result.edit_type,
                target_file="app/commitments/commitments_agent.py",
                anchor_heading=result.anchor_heading,
                old_text=result.old_text,
                new_text=result.new_text,
                rationale=result.rationale,
            )
        # Fall through to yaml issue
        predictor = dspy.Predict(AnalyzeYamlIssue)
        result = predictor(
            root_cause=root_cause,
            subcause=subcause,
            failures_summary=failures_summary,
        )
        if result.confirmed:
            return _base_proposal(
                "yaml", "flag_only",
                target_file=None,
                evidence_quote=result.evidence_quote,
                rationale=result.rationale,
            )

    elif repair_type == "product_decision":
        # Try yaml issue first
        predictor = dspy.Predict(AnalyzeYamlIssue)
        result = predictor(
            root_cause=root_cause,
            subcause=subcause,
            failures_summary=failures_summary,
        )
        if result.confirmed:
            return _base_proposal(
                "yaml", "flag_only",
                target_file=None,
                evidence_quote=result.evidence_quote,
                rationale=result.rationale,
            )

    return None


def cmd_analyze(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    failures = _load_failures(run_dir)
    if not failures:
        console.print("[bold green]No failures to analyze![/]")
        return

    meta = _load_run_meta(run_dir)
    analyzed_splits = sorted({f.get("split", "") for f in failures})

    console.print(f"[bold cyan]Analyzing {len(failures)} failures from {args.run}[/]")
    console.print(f"[dim]Splits: {analyzed_splits}[/]\n")

    # Step 1: deterministic localization
    localized = localize(failures)
    groups = _group_failures(failures, localized)
    console.print(f"[dim]Grouped into {len(groups)} (root_cause, subcause) clusters[/]\n")

    # Step 2: LLM analysis per group
    configure_dspy(settings)
    current_prompt = _get_prompt_docstring()

    proposals = []
    rejected = []
    pid = 0

    for (root_cause, subcause), group_failures in groups.items():
        repair_type = _get_repair_type(root_cause)
        console.print(f"[bold]Analyzing: {root_cause}/{subcause}[/] [dim]({len(group_failures)} failures, repair_type={repair_type})[/]")

        try:
            proposal = _route_and_analyze(
                root_cause, subcause, repair_type, group_failures, current_prompt
            )
        except Exception as e:
            console.print(f"  [red]LLM error: {e}[/]")
            rejected.append({
                "root_cause": root_cause,
                "subcause": subcause,
                "reason": f"LLM error: {e}",
            })
            continue

        if proposal is None:
            console.print(f"  [yellow]No proposal (LLM did not confirm any classification)[/]")
            rejected.append({
                "root_cause": root_cause,
                "subcause": subcause,
                "reason": "LLM did not confirm any classification",
            })
            continue

        pid += 1
        proposal["id"] = f"p{pid}"
        proposal["target_file"] = proposal.get("target_file") or ""

        # Validate
        valid, reason = _validate_proposal(proposal)
        if not valid:
            console.print(f"  [red]Rejected: {reason}[/]")
            proposal["rejection_reason"] = reason
            rejected.append(proposal)
            continue

        proposals.append(proposal)
        console.print(f"  [green]Proposed: {proposal['classification']} / {proposal['edit_type']}[/]")
        console.print(f"  [dim]Rationale: {proposal.get('rationale', '')[:120]}[/]")

    console.print()

    # Step 3: Display proposals table
    if proposals:
        table = Table(title="Proposed Fixes", show_lines=True)
        table.add_column("ID", style="bold")
        table.add_column("Class", style="cyan")
        table.add_column("Root Cause", style="yellow")
        table.add_column("Edit Type", style="magenta")
        table.add_column("# Failures", justify="right")
        table.add_column("Target", style="dim")
        table.add_column("Rationale", max_width=50)

        for p in proposals:
            table.add_row(
                p["id"],
                p["classification"],
                f"{p['root_cause']}/{p['subcause']}",
                p["edit_type"],
                str(len(p["failures_addressed"])),
                p.get("target_file", "—"),
                p.get("rationale", "")[:80],
            )
        console.print(table)

    if rejected:
        rtable = Table(title="Rejected Proposals", show_lines=True)
        rtable.add_column("Root Cause", style="yellow")
        rtable.add_column("Subcause", style="dim")
        rtable.add_column("Reason", style="red")
        for r in rejected:
            rtable.add_row(
                r.get("root_cause", "—"),
                r.get("subcause", "—"),
                r.get("rejection_reason", r.get("reason", "—")),
            )
        console.print(rtable)

    # Step 4: Save proposal.json
    proposal_data = {
        "run_id": args.run,
        "pre_apply_sha": None,
        "analyzed_splits": analyzed_splits,
        "proposals": proposals,
        "rejected": rejected,
    }
    proposal_path = run_dir / "proposal.json"
    proposal_path.write_text(_json.dumps(proposal_data, indent=2, ensure_ascii=False) + "\n")
    console.print(f"\n[bold green]Proposal saved to: {proposal_path}[/]")
    console.print(f"[dim]{len(proposals)} proposals, {len(rejected)} rejected[/]")


# ─── Apply command ───────────────────────────────────────────────────────────


def _check_clean_working_tree() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def _create_branch(run_id: str) -> str:
    branch = f"eval-analyzer/{run_id}"
    subprocess.run(["git", "checkout", "-b", branch], cwd=ROOT, check=True)
    return branch


def _git_commit(message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)


def _get_current_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _apply_anchored_replace(proposal: dict[str, Any]) -> bool:
    target_file = ROOT / proposal["target_file"]
    content = target_file.read_text()
    old_text = proposal["old_text"]
    new_text = proposal["new_text"]

    if old_text not in content:
        console.print(f"  [red]old_text not found — file may have changed[/]")
        return False

    content = content.replace(old_text, new_text, 1)
    target_file.write_text(content)
    return True


def _apply_insert_after_heading(proposal: dict[str, Any]) -> bool:
    target_file = ROOT / proposal["target_file"]
    content = target_file.read_text()
    heading = proposal["anchor_heading"]
    new_text = proposal["new_text"]

    if heading not in content:
        console.print(f"  [red]anchor_heading not found: {heading!r}[/]")
        return False

    # Find the heading line, then find the last bullet under it
    # (last line starting with "    -" before the next blank line or heading)
    lines = content.split("\n")
    heading_idx = None
    for i, line in enumerate(lines):
        if heading in line:
            heading_idx = i
            break

    if heading_idx is None:
        console.print(f"  [red]heading line not found[/]")
        return False

    # Find the last bullet under this heading
    last_bullet_idx = heading_idx
    for i in range(heading_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            break
        if stripped.startswith("- ") or stripped.startswith('"- '):
            last_bullet_idx = i
        # Also handle continuation lines (indented, not a new bullet, not blank)
        elif line.startswith("      ") and stripped:
            last_bullet_idx = i

    # Insert new text after the last bullet
    new_lines = lines[: last_bullet_idx + 1] + new_text.split("\n") + lines[last_bullet_idx + 1 :]
    target_file.write_text("\n".join(new_lines))
    return True


def _apply_function_replace(proposal: dict[str, Any]) -> bool:
    target_file = ROOT / proposal["target_file"]
    content = target_file.read_text()
    func_name = proposal["target_function"]
    new_body = proposal["new_function_body"]

    # Parse to find the function's line range
    tree = ast.parse(content)
    lines = content.splitlines(keepends=True)

    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target_node = node
            break

    if target_node is None:
        console.print(f"  [red]function {func_name!r} not found[/]")
        return False

    start = target_node.lineno - 1
    end = target_node.end_lineno if hasattr(target_node, "end_lineno") else start + 1

    # Replace the function
    new_content = "".join(lines[:start]) + new_body + "\n\n" + "".join(lines[end:])
    target_file.write_text(new_content)
    return True


def _validate_python_file(filepath: Path) -> bool:
    try:
        ast.parse(filepath.read_text())
        return True
    except SyntaxError as e:
        console.print(f"  [red]ast.parse failed: {e}[/]")
        return False


def cmd_apply(args: argparse.Namespace) -> None:
    proposal_path = Path(args.proposal)
    if not proposal_path.exists():
        raise SystemExit(f"Proposal file not found: {proposal_path}")

    data = _json.loads(proposal_path.read_text())
    proposals = data.get("proposals", [])
    run_id = data.get("run_id", "unknown")

    if not proposals:
        console.print("[yellow]No proposals to apply.[/]")
        return

    # Preconditions
    if not _check_clean_working_tree():
        console.print("[red]Working tree is not clean. Commit or stash your changes first.[/]")
        return

    pre_sha = _get_current_sha()
    branch = _create_branch(run_id)
    console.print(f"[bold green]Created branch: {branch}[/]")
    console.print(f"[dim]Pre-apply SHA: {pre_sha}[/]\n")

    # Separate by class
    prompt_proposals = [p for p in proposals if p["classification"] == "model"]
    metric_proposals = [p for p in proposals if p["classification"] == "metric"]
    yaml_proposals = [p for p in proposals if p["classification"] == "yaml"]

    # Display yaml flags (no edits)
    for p in yaml_proposals:
        console.print(Panel(
            f"[bold]YAML flag: {p['root_cause']}/{p['subcause']}[/]\n"
            f"Evidence: {p.get('evidence_quote', '—')}\n"
            f"Rationale: {p.get('rationale', '—')}",
            title=f"{p['id']} — YAML Issue (flag only, no edit)",
            border_style="yellow",
        ))

    # Apply prompt proposals
    applied_prompt = 0
    if prompt_proposals:
        console.print("\n[bold cyan]═══ Prompt Proposals ═══[/]\n")
        for p in prompt_proposals:
            edit_type = p["edit_type"]
            console.print(f"\n[bold]{p['id']}[/] ({edit_type}): {p['root_cause']}/{p['subcause']}")
            console.print(f"[dim]Rationale: {p.get('rationale', '')}[/]")

            if edit_type == "anchored_replace":
                console.print("\n[red]--- old_text ---[/]")
                console.print(Syntax(p.get("old_text", ""), "python", word_wrap=True))
                console.print("\n[green]--- new_text ---[/]")
                console.print(Syntax(p.get("new_text", ""), "python", word_wrap=True))
            elif edit_type == "insert_after_heading":
                console.print(f"\n[dim]Insert after heading: {p.get('anchor_heading', '')!r}[/]")
                console.print("\n[green]--- new_text ---[/]")
                console.print(Syntax(p.get("new_text", ""), "python", word_wrap=True))

            if not Confirm.ask("\nApply this prompt edit?", default=True):
                console.print("[dim]Skipped.[/]")
                continue

            success = False
            if edit_type == "anchored_replace":
                success = _apply_anchored_replace(p)
            elif edit_type == "insert_after_heading":
                success = _apply_insert_after_heading(p)

            if success:
                target_file = ROOT / p["target_file"]
                if _validate_python_file(target_file):
                    console.print("[green]Applied and validated.[/]")
                    applied_prompt += 1
                else:
                    console.print("[red]Validation failed — reverting.[/]")
                    subprocess.run(["git", "checkout", "--", str(target_file)], cwd=ROOT)
            else:
                console.print("[red]Apply failed.[/]")

        if applied_prompt > 0:
            _git_commit(f"eval-analyzer: apply {applied_prompt} prompt fixes for run {run_id}")
            console.print(f"\n[green]Committed {applied_prompt} prompt fixes.[/]")

    # Apply metric proposals (always explicit approval)
    applied_metric = 0
    if metric_proposals:
        console.print("\n[bold cyan]═══ Metric Proposals (human-review-always) ═══[/]\n")
        for p in metric_proposals:
            console.print(f"\n[bold]{p['id']}[/] (function_replace): {p['root_cause']}/{p['subcause']}")
            console.print(f"[dim]Rationale: {p.get('rationale', '')}[/]")
            console.print(f"\n[dim]Target function: {p.get('target_function', '')}[/]")
            console.print("\n[green]--- new function body ---[/]")
            console.print(Syntax(p.get("new_function_body", ""), "python", word_wrap=True))

            console.print("\n[bold yellow]⚠ Metric edits touch the scorekeeper. Review carefully.[/]")
            if not Confirm.ask("\nApply this metric edit?", default=False):
                console.print("[dim]Skipped.[/]")
                continue

            success = _apply_function_replace(p)
            if success:
                target_file = ROOT / p["target_file"]
                if _validate_python_file(target_file):
                    # Run tests
                    console.print("[dim]Running tests...[/]")
                    test_result = subprocess.run(
                        ["python", "-m", "pytest", "tests/", "-x", "--tb=short"],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    if test_result.returncode == 0:
                        console.print("[green]Applied, validated, tests pass.[/]")
                        applied_metric += 1
                    else:
                        console.print(f"[red]Tests failed:\n{test_result.stdout[-500:]}{test_result.stderr[-500:]}[/]")
                        console.print("[red]Reverting.[/]")
                        subprocess.run(["git", "checkout", "--", str(target_file)], cwd=ROOT)
                else:
                    console.print("[red]ast.parse failed — reverting.[/]")
                    subprocess.run(["git", "checkout", "--", str(target_file)], cwd=ROOT)
            else:
                console.print("[red]Apply failed.[/]")

        if applied_metric > 0:
            _git_commit(f"eval-analyzer: apply {applied_metric} metric fixes for run {run_id}")
            console.print(f"\n[green]Committed {applied_metric} metric fixes.[/]")

    # Save pre-apply SHA to proposal
    data["pre_apply_sha"] = pre_sha
    proposal_path.write_text(_json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    console.print(f"\n[bold green]Done. Applied {applied_prompt} prompt + {applied_metric} metric fixes.[/]")
    console.print(f"[dim]Branch: {branch} | Pre-apply SHA: {pre_sha}[/]")
    console.print(f"[dim]To verify: python scripts/eval_analyzer.py verify --run {run_id} --proposal {proposal_path}[/]")


# ─── Verify command ──────────────────────────────────────────────────────────


def _load_failures_set(run_dir: Path) -> set[str]:
    """Load failures as a set of 'split:scenario' keys."""
    failures = _load_failures(run_dir)
    return {f"{f.get('split','')}:{f.get('scenario','')}" for f in failures}


def _run_regrade(predictions_path: Path, judge_cache: Path | None = None) -> Path:
    """Run regrade_predictions.py and return the output run directory."""
    cmd = ["python", "scripts/regrade_predictions.py", str(predictions_path)]
    if judge_cache and judge_cache.exists():
        cmd += ["--llm-judge", "--judge-cache", str(judge_cache), "--no-new-judge-calls"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Regrade failed:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}[/]")
        raise SystemExit(1)
    # Find the regrade run dir (most recent -regrade- dir)
    regrade_dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and "regrade" in d.name],
        key=lambda d: d.name,
    )
    if not regrade_dirs:
        raise SystemExit("No regrade run directory found")
    return regrade_dirs[-1]


def _run_eval_split(split: str, llm_judge: bool = True) -> Path:
    """Run eval_runner.py on a single split and return the run directory."""
    cmd = ["python", "scripts/eval_runner.py", "--split", split]
    if llm_judge:
        cmd += ["--llm-judge"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Eval failed for split {split}:\n{result.stdout[-1000:]}\n{result.stderr[-500:]}[/]")
        raise SystemExit(1)
    # Find the most recent run dir
    judge_suffix = "-judge" if llm_judge else ""
    run_dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.endswith(judge_suffix) and "regrade" not in d.name],
        key=lambda d: d.name,
    )
    if not run_dirs:
        raise SystemExit(f"No run directory found for split {split}")
    return run_dirs[-1]


def _compare_failure_sets(
    before: set[str], after: set[str], targeted: set[str]
) -> dict[str, list[str]]:
    """Compare two failure sets and categorize flips."""
    fixed = before - after  # were failing, now passing
    new_fails = after - before  # were passing, now failing
    persistent = before & after  # still failing

    targeted_fixed = fixed & targeted
    untargeted_fixed = fixed - targeted
    untargeted_new = new_fails - targeted

    return {
        "targeted_fixed": sorted(targeted_fixed),
        "untargeted_fixed": sorted(untargeted_fixed),
        "untargeted_new": sorted(untargeted_new),
        "persistent": sorted(persistent),
        "total_fixed": sorted(fixed),
        "total_new": sorted(new_fails),
    }


def _print_delta_table(delta: dict[str, list[str]], title: str) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right", style="cyan")
    table.add_column("Scenarios", style="dim")

    table.add_row("Targeted fixed", str(len(delta["targeted_fixed"])), ", ".join(delta["targeted_fixed"][:5]))
    table.add_row("Untargeted fixed", str(len(delta["untargeted_fixed"])), ", ".join(delta["untargeted_fixed"][:5]))
    table.add_row("[red]Untargeted NEW failures[/]", str(len(delta["untargeted_new"])), ", ".join(delta["untargeted_new"][:5]))
    table.add_row("Persistent", str(len(delta["persistent"])), ", ".join(delta["persistent"][:5]))

    console.print(table)


def cmd_verify(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    proposal_path = Path(args.proposal)
    if not proposal_path.exists():
        raise SystemExit(f"Proposal file not found: {proposal_path}")

    data = _json.loads(proposal_path.read_text())
    proposals = data.get("proposals", [])
    run_id = data.get("run_id", args.run)
    pre_sha = data.get("pre_apply_sha")

    has_metric = any(p["classification"] == "metric" for p in proposals)
    has_prompt = any(p["classification"] == "model" for p in proposals)
    targeted_keys = set()
    for p in proposals:
        for addr in p.get("failures_addressed", []):
            parts = addr.split("/", 1)
            if len(parts) == 2:
                targeted_keys.add(f"{parts[0]}:{parts[1]}")

    original_failures = _load_failures_set(run_dir)
    meta = _load_run_meta(run_dir)

    # ── Metric verify: regrade frozen predictions with judge held constant ──
    if has_metric:
        console.print("[bold cyan]═══ Metric Verification: Regrading frozen predictions ═══[/]\n")

        predictions_path = run_dir / "predictions.jsonl"
        judge_cache = run_dir / "judge_verdicts.jsonl"

        if judge_cache.exists():
            console.print(f"[dim]Using frozen judge cache: {judge_cache}[/]")
        else:
            console.print("[dim]No judge cache — regrading with deterministic metrics only[/]")

        regrade_dir = _run_regrade(predictions_path, judge_cache if judge_cache.exists() else None)
        regrade_failures = _load_failures_set(regrade_dir)

        delta = _compare_failure_sets(original_failures, regrade_failures, targeted_keys)
        _print_delta_table(delta, "Metric Change Delta (frozen predictions)")

        # Goodhart alarm
        untargeted_new = delta["untargeted_new"]
        untargeted_fixed = delta["untargeted_fixed"]
        if untargeted_new:
            console.print(f"\n[bold red]⚠ GOODHART ALARM: {len(untargeted_new)} untargeted scenarios newly failing![/]")
            for s in untargeted_new:
                console.print(f"  [red]NEW: {s}[/]")
        if untargeted_fixed:
            console.print(f"\n[bold yellow]⚠ {len(untargeted_fixed)} untargeted scenarios newly passing — check if metric loosened:[/]")
            for s in untargeted_fixed:
                console.print(f"  [yellow]FIXED (untargeted): {s}[/]")

        if len(untargeted_new) > len(untargeted_fixed):
            console.print("\n[bold red]Net regression on untargeted scenarios. Recommend rollback.[/]")

    # ── Prompt verify: re-run agent on analyzed + held-out splits ──
    if has_prompt:
        analyzed_splits = data.get("analyzed_splits", ["dev"])
        held_out_splits = [s for s in ["test", "challenge"] if s not in analyzed_splits]

        # Part 1: Analyzed splits (re-run and compare)
        console.print(f"\n[bold cyan]═══ Prompt Verification: Analyzed splits ({analyzed_splits}) ═══[/]\n")
        use_judge = meta.get("llm_judge", False)

        for split in analyzed_splits:
            console.print(f"[dim]Re-running split: {split}[/]")
            new_run_dir = _run_eval_split(split, llm_judge=use_judge)
            new_failures = _load_failures_set(new_run_dir)

            # Compare against original (filter to same split)
            orig_split = {k for k in original_failures if k.startswith(f"{split}:")}
            new_split = {k for k in new_failures if k.startswith(f"{split}:")}
            split_targeted = {k for k in targeted_keys if k.startswith(f"{split}:")}

            delta = _compare_failure_sets(orig_split, new_split, split_targeted)
            n = len(orig_split) + len(new_split - orig_split)
            _print_delta_table(delta, f"Prompt Change Delta — {split} (N≈{n})")

            noise_floor = max(2, n // 10)
            net = len(delta["total_fixed"]) - len(delta["total_new"])
            console.print(f"\n[dim]Noise floor: ±{noise_floor} on N≈{n}. Net change: {net:+d}.[/]")
            if abs(net) <= noise_floor:
                console.print(f"[yellow]Net change is inside noise floor — not statistically meaningful.[/]")

        # Part 2: Held-out splits (produce baseline on main, then run on branch)
        if held_out_splits:
            console.print(f"\n[bold cyan]═══ Prompt Verification: Held-out splits ({held_out_splits}) ═══[/]\n")
            console.print("[dim]Producing baseline on main (old prompt)...[/]")

            current_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
            ).strip()

            # Checkout main and run baseline
            subprocess.run(["git", "checkout", "main"], cwd=ROOT, check=True)
            baseline_runs = {}
            for split in held_out_splits:
                console.print(f"[dim]  Baseline: {split}[/]")
                baseline_dir = _run_eval_split(split, llm_judge=use_judge)
                baseline_runs[split] = _load_failures_set(baseline_dir)

            # Checkout analyzer branch and run with new prompt
            subprocess.run(["git", "checkout", current_branch], cwd=ROOT, check=True)
            console.print("\n[dim]Running with new prompt on held-out splits...[/]")
            for split in held_out_splits:
                console.print(f"[dim]  New: {split}[/]")
                new_dir = _run_eval_split(split, llm_judge=use_judge)
                new_failures = _load_failures_set(new_dir)

                # Filter to this split
                baseline_split = {k for k in baseline_runs[split] if k.startswith(f"{split}:")}
                new_split = {k for k in new_failures if k.startswith(f"{split}:")}
                split_targeted = {k for k in targeted_keys if k.startswith(f"{split}:")}

                delta = _compare_failure_sets(baseline_split, new_split, split_targeted)
                n = len(baseline_split) + len(new_split - baseline_split)
                _print_delta_table(delta, f"Held-out Generalization — {split} (N≈{n})")

                noise_floor = max(2, n // 10)
                net = len(delta["total_fixed"]) - len(delta["total_new"])
                console.print(f"\n[dim]Noise floor: ±{noise_floor} on N≈{n}. Net change: {net:+d}.[/]")

                if len(delta["total_new"]) > len(delta["total_fixed"]):
                    console.print(f"[bold red]Regression on held-out {split}! Consider rollback.[/]")

    # Rollback option
    if pre_sha:
        console.print(f"\n[dim]Pre-apply SHA: {pre_sha}[/]")
        if Confirm.ask("Rollback (discard analyzer branch)?", default=False):
            subprocess.run(["git", "checkout", "main"], cwd=ROOT, check=True)
            branch = f"eval-analyzer/{run_id}"
            subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, check=True)
            console.print(f"[green]Rolled back. Branch {branch} discarded.[/]")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval analyzer agent: analyze failures, propose fixes, apply, and verify."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Read failures and propose fixes")
    p_analyze.add_argument("--run", required=True, help="Run directory name (e.g. 20260707_224735-judge)")

    # apply
    p_apply = subparsers.add_parser("apply", help="Apply proposed fixes")
    p_apply.add_argument("--proposal", required=True, help="Path to proposal.json")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify improvement after applying fixes")
    p_verify.add_argument("--run", required=True, help="Original run directory name")
    p_verify.add_argument("--proposal", required=True, help="Path to proposal.json")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
