# Eval Analyzer Agent — Concepts & Usage Guide

## Overview

The eval analyzer is an LLM-powered agent that automates the analysis of evaluation failures and proposes fixes. It runs as a three-step interactive CLI:

```
analyze → apply → verify
```

Each step is a separate command. You run them in sequence, reviewing at each step.

---

## Prerequisites

### 1. Run an evaluation first

The analyzer works on an existing eval run. You need a run directory under `runs/` containing:

- `failures.jsonl` — detailed failure records (scenarios, expected vs actual commitments, mismatched fields)
- `predictions.jsonl` — agent predictions (frozen, used for metric regrading)
- `run_meta.json` — metadata (model, splits, limit, git SHA)
- `judge_verdicts.jsonl` — (optional) frozen judge verdicts, only if `--freeze-judge` was used

To produce a run with all files:

```bash
# Basic run (no judge)
python scripts/eval_runner.py --split train

# With LLM judge
python scripts/eval_runner.py --split train --llm-judge

# With LLM judge + frozen judge verdicts (recommended for analyzer)
python scripts/eval_runner.py --split train --llm-judge --freeze-judge

# All splits
python scripts/eval_runner.py --all --llm-judge --freeze-judge
```

> **Why `--freeze-judge`?** The verify step needs to hold the judge constant when regrading metric changes. Without `judge_verdicts.jsonl`, metric verification falls back to deterministic-only metrics, which may trigger false Goodhart alarms.

### 2. Clean working tree

The `apply` command creates a git branch and commits changes. You need a clean working tree before running it:

```bash
git status --porcelain  # should be empty
```

---

## Step 1: Analyze

### What it does

1. **Loads failures** from `runs/<run_id>/failures.jsonl`
2. **Deterministic localization** — runs `eval/localize.py` to classify each failure with a root cause, subcause, and repair type. This is rule-based, not LLM-based.
3. **Groups failures** by `(root_cause, subcause)` — each group becomes one proposal
4. **LLM analysis per group** — routes to one of three DSPy signatures based on the deterministic `repair_type`:
   - `AnalyzeModelFailure` — for prompt bugs (repair_type = `signature_rule`)
   - `AnalyzeMetricBug` — for metric code bugs (repair_type = `metric`, `metric_or_postprocess`, `postprocess`)
   - `AnalyzeYamlIssue` — for dataset label issues (repair_type = `product_decision`)
5. **Validates proposals** — checks that anchors exist, function names are real, AST parses
6. **Saves** `proposal.json` in the run directory

### Routing logic

```
repair_type = "metric" → try AnalyzeMetricBug first
  if confirmed → metric proposal (function_replace)
  if not confirmed → fall through to AnalyzeModelFailure

repair_type = "signature_rule" → try AnalyzeModelFailure first
  if confirmed → model proposal (anchored_replace or insert_after_heading)
  if not confirmed → fall through to AnalyzeYamlIssue

repair_type = "product_decision" → try AnalyzeYamlIssue
  if confirmed → yaml flag (flag_only, no edit)
```

### Edit types

| Edit type | Description | Target file |
|---|---|---|
| `anchored_replace` | Replace existing text in the prompt docstring. Requires `old_text` (exact verbatim quote) and `new_text`. | `app/commitments/commitments_agent.py` |
| `insert_after_heading` | Insert new bullet(s) after an existing heading in the prompt. Requires `anchor_heading` and `new_text`. | `app/commitments/commitments_agent.py` |
| `function_replace` | Replace an entire function in the metric code. Requires `target_function` and `new_function_body`. | `eval/metrics.py` |
| `flag_only` | No edit. Just flags a YAML/dataset issue with evidence. | N/A |

### How to run

```bash
python scripts/eval_analyzer.py analyze --run <run_id>
```

Example:
```bash
python scripts/eval_analyzer.py analyze --run 20260707_225919-judge
```

### Output

- Console: table of proposals (ID, classification, root cause, edit type, # failures, target, rationale) + table of rejected proposals
- File: `runs/<run_id>/proposal.json` containing all proposals and rejections

### Proposal JSON structure

```json
{
  "run_id": "20260707_225919-judge",
  "pre_apply_sha": null,
  "analyzed_splits": ["train"],
  "proposals": [
    {
      "id": "p1",
      "classification": "model",
      "root_cause": "act_vs_ignore_over_extraction",
      "subcause": "request_without_acceptance",
      "repair_type_prior": "signature_rule",
      "failures_addressed": ["train/scenario_name", ...],
      "edit_type": "insert_after_heading",
      "target_file": "app/commitments/commitments_agent.py",
      "anchor_heading": "Act vs Ignore rules (critical — do NOT over-extract):",
      "old_text": "",
      "new_text": "    - New rule here",
      "rationale": "..."
    }
  ],
  "rejected": [...]
}
```

---

## Step 2: Apply

### What it does

1. **Checks clean working tree** — refuses if dirty
2. **Creates git branch** `eval-analyzer/<run_id>` (or reuses if already on it)
3. **Records pre-apply SHA** — for rollback
4. **Applies proposals by class:**

   **Prompt proposals (model):**
   - Shows diff (old_text → new_text for anchored_replace, or new bullet for insert_after_heading)
   - Asks apply/skip interactively (default: yes)
   - Validates with `ast.parse`
   - Commits all prompt fixes in one commit
   - Idempotent: skips if `new_text` already present in file

   **Metric proposals (metric):**
   - Shows full new function body
   - Asks apply/skip interactively (default: **no** — requires explicit approval)
   - Validates with `ast.parse`
   - Runs `pytest tests/` — compares against known pre-existing failures (baseline exclusion set)
   - Only fails on **new** test regressions, not pre-existing ones
   - Commits all metric fixes in one commit

   **YAML proposals (yaml):**
   - Displayed as flags with evidence quotes
   - No edits applied — these are for human review

5. **Updates `proposal.json`** with `pre_apply_sha`

### How to run

```bash
python scripts/eval_analyzer.py apply --proposal runs/<run_id>/proposal.json
```

Example:
```bash
python scripts/eval_analyzer.py apply --proposal runs/20260707_225919-judge/proposal.json
```

### Interaction

For each proposal you'll see a diff and be asked:

```
Apply this prompt edit? [y/n] (y):     ← prompt edits default to yes
Apply this metric edit? [y/n] (n):     ← metric edits default to no (review carefully)
```

### After apply

You'll be on branch `eval-analyzer/<run_id>` with two commits:
1. Prompt fixes (if any were applied)
2. Metric fixes (if any were applied)

The console will show you the verify command to run next.

---

## Step 3: Verify

### What it does

The verify step checks whether the applied fixes actually improved results. It has two parts:

### Part A: Metric Verification (if metric proposals were applied)

**Goal:** Confirm that metric code changes fix targeted failures without breaking others.

**Method:** Regrade frozen predictions — reuse the exact same agent predictions from the original run, but score them with the **new metric code**. This isolates the metric change from agent variance.

**Judge handling:**
- If `judge_verdicts.jsonl` exists in the run dir → regrade with `--llm-judge --judge-cache ... --no-new-judge-calls` (frozen judge, no new LLM calls)
- If no judge cache → regrade with deterministic metrics only

**Output:**
- Delta table: targeted fixed, untargeted fixed, untargeted NEW failures, persistent
- **Goodhart alarm** — if untargeted scenarios newly fail, flags them
- **Net regression check** — if more untargeted new failures than untargeted fixes, recommends rollback

### Part B: Prompt Verification (if prompt proposals were applied)

**Goal:** Confirm that prompt changes fix targeted failures on analyzed splits AND don't regress on held-out splits.

**Part B1: Analyzed splits (re-run agent)**

Re-runs `eval_runner.py` on each analyzed split with the new prompt (on the analyzer branch). Compares against original failures.

- Shows delta table per split
- Reports noise floor (±N/10) and net change
- Flags if net change is inside noise floor (not statistically meaningful)

**Part B2: Held-out splits (generalization check)**

This is the key guardrail against overfitting to the training data.

1. **Checkout `main`** (old prompt, no fixes)
2. **Run baseline** on held-out splits (test, challenge) — produces reference failure set
3. **Checkout analyzer branch** (new prompt with fixes)
4. **Run new eval** on held-out splits
5. **Compare** — shows delta table per held-out split

If the new prompt causes regressions on held-out splits (scenarios that were passing on main but now fail on the analyzer branch), it flags them.

### Flags

| Flag | Description |
|---|---|
| `--llm-judge` | Use LLM judge for prompt re-runs. If omitted, reads from `run_meta.json`. |
| `--no-held-out` | Skip held-out baseline runs (faster, but no generalization check). |

### How to run

```bash
# Full verify (with held-out generalization check)
python scripts/eval_analyzer.py verify --run <run_id> --proposal runs/<run_id>/proposal.json --llm-judge

# Fast verify (skip held-out)
python scripts/eval_analyzer.py verify --run <run_id> --proposal runs/<run_id>/proposal.json --no-held-out
```

### Rollback

At the end of verify, you're offered a rollback option:

```
Rollback (discard analyzer branch)? [y/n] (n):
```

If you say yes:
- Checks out main
- Deletes the analyzer branch
- All changes are discarded

---

## Complete Workflow Example

```bash
# 1. Run eval with judge + freeze judge
python scripts/eval_runner.py --split train --llm-judge --freeze-judge

# 2. Analyze failures
python scripts/eval_analyzer.py analyze --run 20260707_225919-judge

# 3. Review proposals
cat runs/20260707_225919-judge/proposal.json | python -m json.tool | less

# 4. Apply (interactive)
python scripts/eval_analyzer.py apply --proposal runs/20260707_225919-judge/proposal.json

# 5. Verify (full, with held-out)
python scripts/eval_analyzer.py verify --run 20260707_225919-judge --proposal runs/20260707_225919-judge/proposal.json --llm-judge

# 6. If happy: merge branch to main
git checkout main
git merge eval-analyzer/20260707_225919-judge

# 7. If not happy: rollback
# (offered at end of verify, or manually:)
git checkout main
git branch -D eval-analyzer/20260707_225919-judge
```

---

## Key Concepts

### Frozen Predictions Regrading

When verifying metric changes, we don't re-run the agent. Instead, we reuse the exact same `predictions.jsonl` from the original run and re-score them with the new metric code. This eliminates agent stochasticity — the only thing that changes is the scoring logic.

### Judge Cache Replay

The LLM judge adds noise (it's another LLM call). To isolate metric changes, we replay frozen judge verdicts from `judge_verdicts.jsonl` using `regrade_predictions.py --judge-cache ... --no-new-judge-calls`. This means the judge's decisions are held constant — only the metric code changes.

### Held-Out Generalization

Prompt changes can overfit to the training data. To check generalization:
1. Run baseline on held-out splits (test, challenge) with the **old** prompt (on main)
2. Run the same splits with the **new** prompt (on analyzer branch)
3. Compare — if the new prompt causes new failures on held-out data, it's overfitting

### Noise Floor

On small datasets (N≈15-20), a single scenario flip is ±1, which is within noise. The noise floor is calculated as `max(2, N//10)`. If the net change (fixed - new) is within this range, the result is flagged as "not statistically meaningful."

### Goodhart Alarm

When metric code changes, there's a risk of "Goodharting" — making the metric easier to pass without actually improving the agent. The alarm fires when untargeted scenarios (ones the LLM didn't propose fixes for) newly pass, suggesting the metric may have loosened. It also fires when untargeted scenarios newly fail, suggesting the metric may have tightened in a harmful way.

### Root Cause → Metric Code Mapping

The analyzer maps deterministic root causes to the relevant metric functions so the LLM has context:

| Root cause | Metric functions |
|---|---|
| `deadline_normalization` | `_normalize_deadline`, `_deadline_equal` |
| `context_metric_noise` | `_word_overlap` |
| `required_action_normalization` | `_token_f1` |
| `update_vs_new_matching` | `compare_commitments` |

---

## Troubleshooting

### "No judge cache — regrading with deterministic metrics only"

Your original run didn't use `--freeze-judge`. Re-run with:
```bash
python scripts/eval_runner.py --split train --llm-judge --freeze-judge
```

### "Working tree is not clean"

Commit or stash your changes before running `apply`:
```bash
git add . && git commit -m "wip"
```

### "a branch named 'eval-analyzer/...' already exists"

You're re-running apply on an existing branch. The fixed code handles this — it'll reuse the branch. If you want to start fresh:
```bash
git checkout main
git branch -D eval-analyzer/<run_id>
```

### Metric edit reverted due to test failure

The apply command runs `pytest tests/` after each metric edit. It excludes known pre-existing failures (currently `test_localize.py::test_deadline_subcauses`). If your edit causes a **new** test failure, it reverts. Check the error output to see which test failed.

### Verify is slow

The held-out baseline runs require running the full agent on test + challenge splits, twice (once on main, once on analyzer branch). Use `--no-held-out` to skip:
```bash
python scripts/eval_analyzer.py verify --run <run_id> --proposal <path> --no-held-out
```

### Verify left me on the wrong branch

If you Ctrl-C during verify while it's running held-out baselines, it may leave you on `main`. To recover:
```bash
git checkout eval-analyzer/<run_id>
```
