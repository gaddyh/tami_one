# Tami One

A WhatsApp-native assistant for a mortgage/loan advisor. It reads the advisor's real WhatsApp conversations (personal number, via Green API) and automatically extracts **commitments** — who owes whom what, and by when — so the advisor never has to re-read a thread to remember what's still open.

Separately, it can run as a **360dialog-based business bot** that replies to customers directly with an OpenAI agent (text + voice notes). The two capabilities share one FastAPI app and one database but don't otherwise interact.

**Status:** active prototype, ~3 days old (first commit 2026-07-05). The commitment-extraction pipeline is validated against a 93-example eval set; it has not been run against real production traffic yet, and there are two known reliability gaps before it should be (see [Known Limitations](#known-limitations--roadmap)).

---

## Contents

- [Why this exists](#why-this-exists)
- [How a message becomes a commitment](#how-a-message-becomes-a-commitment)
- [Project layout](#project-layout)
- [Commitment extraction & evals](#commitment-extraction--evals)
- [Interactive failure inspector](#interactive-failure-inspector)
- [Data model](#data-model)
- [Setup](#setup)
- [Configuration](#configuration)
- [Run locally](#run-locally)
- [Seed the database](#seed-the-database)
- [Tests](#tests)
- [Deploy (Render)](#deploy-render)
- [Endpoints](#endpoints)
- [Known limitations / roadmap](#known-limitations--roadmap)

---

## Why this exists

Advisors run their business out of dozens of parallel WhatsApp threads — clients, banks, lawyers. Nothing is ever explicitly "closed," so the mental load of tracking *who's waiting on what* falls entirely on the advisor. Tami One turns that implicit tracking into an explicit, queryable list, extracted automatically from ordinary conversation:

```json
{
  "committed_party": "me",
  "required_action": "Send the material",
  "deadline": "in one hour",
  "status": "waiting",
  "context": "Message: 'אני אשלח לך את החומר עוד שעה' ..."
}
```

Given a real, mixed-Hebrew/English WhatsApp thread, the pipeline correctly separates a fulfilled promise (`status=done`) from a still-waiting one (`status=waiting`), attributes each to the right party, and keeps a plain-language `context` trail for why. That's the core proof-of-concept the project has validated so far.

## How a message becomes a commitment

1. **Ingest** — Green API POSTs to `/webhook/green-api`. `normalize_green_api_message_event()` turns the raw payload into a `MessageEvent` (chat id, sender, direction, text, timestamp).
2. **Upsert + buffer** — `upsert_contact_and_chat()`:
   - Creates a `Contact` row if this `(tenant_id, chat_id)` hasn't been seen before (insert-only; existing contacts are never updated).
   - Skips chats with more distinct senders than `MAX_GROUP_PARTICIPANTS` (this is a 1:1/small-group tool, not a broadcast-list reader).
   - Appends the event to an **in-memory** `MessageBuffer`, keyed by `(tenant_id, chat_id)`.
3. **Drain** — a background `asyncio` task inside the running app process wakes up every 2 minutes, atomically drains the whole buffer, and hands each chat's batch of messages to the commitment extractor.
4. **Extract** — `extract_commitments()` sends the batch, plus that chat's currently-open commitments, through a DSPy `CommitmentAgent` (`ExtractCommitments` signature + `dspy.Predict`), configured at startup with `dspy.JSONAdapter()` so it behaves like a structured-output call while staying measurable and optimizable.
5. **Persist** — `CommitmentItem` rows are upserted into Postgres/SQLite, tagged with the `source_message_ids` that produced them.

The 360dialog path (`/webhook/360dialog`) is unrelated to this flow — it's a direct reply bot for business-number customers, using `agents/core.py` for in-memory per-thread conversation and `services/transcription.py` for voice notes.

## Project layout

```
app/
├── main.py                 # FastAPI app; startup wires init_db + load_cache + drain loop
├── config.py                # Settings, loaded from env vars at import time
├── routers/
│   ├── green_api.py          # Green API payload → normalized MessageEvent
│   ├── personal_webhook.py   # POST /webhook/green-api, /admin/seed, /debug/settings
│   └── business_webhook.py   # POST /webhook/360dialog — agent-driven customer replies
├── db/
│   ├── engine.py              # SQLAlchemy engine (Postgres or SQLite)
│   ├── models.py              # SQLModel tables
│   ├── cache.py                # In-memory account/contact cache + MessageBuffer
│   ├── upsert.py                # Insert-only Contact creation + buffering
│   └── seed.py                  # Demo data seeding
├── commitments/
│   ├── models.py                 # Canonical Commitment / CommitmentList schema
│   ├── commitments_agent.py       # DSPy Signature + Module for commitment extraction
│   ├── extractor.py                # Adapter: messages + existing commitments → DSPy agent
│   └── processor.py                 # Drains buffer, calls extractor, upserts CommitmentItem rows
├── agents/
│   ├── core.py                       # OpenAI agent used by the 360dialog business bot
│   └── memory.py                      # Per-thread in-memory conversation history
└── services/
    ├── whatsapp.py                     # 360dialog API client
    └── transcription.py                 # Voice-note transcription via OpenAI
eval/
├── dataset.py               # Example construction + devset loading
├── metrics.py                # commitment_metric, act_vs_ignore_metric, token-F1, word-overlap
├── localize.py                # Failure → root cause / subcause / repair-type / priority scoring
└── llm_judge.py               # Optional LLM-as-judge for semantic field matching
scripts/
├── seed.py                    # CLI seed entry point
├── eval_runner.py               # Local commitment-extraction eval runner (train/dev/test/challenge)
├── eval_inspector.py             # Interactive CLI for examining eval failures (load or run fresh, then REPL)
└── compare_runs.py             # Compare two eval runs side by side
tests/
├── evals/
│   ├── data/*.yaml             # Hand-authored example definitions, 6 categories + challenge split
│   ├── generate_devset.py       # YAML → train/dev/test JSON splits
│   ├── trainset.json / devset.json / testset.json / challenge_act_ignore.json
│   └── localize_cases/           # Fixtures for testing the localizer itself
└── test_*.py                    # upsert, webhook, commitment extraction/eval, localizer
```

## Commitment extraction & evals

The extractor is a DSPy `Signature + Module`, not a one-off prompt string:

- `Commitment` / `CommitmentList` (`app/commitments/models.py`) are the canonical domain schema.
- `ExtractCommitments(dspy.Signature)` in `app/commitments/commitments_agent.py` encodes the extraction rules directly in the docstring — most importantly a set of **act-vs-ignore rules** ("we should do X" is an opinion, not a commitment; a request needs explicit acceptance; "started"/"almost done" are progress reports, not completions).
- `eval/metrics.py` scores predictions against a hand-labeled devset: exact match on structured fields, token-F1 on `required_action` (so `settle the invoice` vs `pay the invoice` isn't an automatic fail), and word-overlap on `context`. Three semantic fields (`required_action`, `deadline`, `context`) can optionally use an **LLM-as-judge** fallback (`eval/llm_judge.py`) when the deterministic check fails — enabled with `--llm-judge` on the eval runner.
- `eval/dataset.py` builds train/dev/test splits from YAML fixtures in `tests/evals/data/`, each tagged by category (`act_vs_ignore`, `args_party`, `args_deadline`, `args_required_action`, `lifecycle_update_vs_new`, `lifecycle_completion`), difficulty (`easy`/`medium`/`hard`), and a specific contrastive scenario name.

Current split sizes: **train 40 · dev 22 · test 31** (93 total), generated with:

```bash
python tests/evals/generate_devset.py
```

### Latest eval run

Model: `gpt-5.4-mini`, run `20260707_023551`.

```bash
python scripts/eval_runner.py --all              # deterministic metrics (default)
python scripts/eval_runner.py --all --llm-judge   # with LLM-as-judge fallback
```

Runs are saved to `runs/<run_id>/` by default (use `--no-save` to disable). Runs with `--llm-judge` get a `-judge` suffix in the run ID.

| Split | N | TP | FP | FN | TN | Precision | Recall | F1 | Act/Ignore Acc. | Full-Commitment | Over-Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 40 | 28 | 3 | 4 | 5 | 0.90 | 0.88 | 0.89 | 0.82 | 50.0% (16/32) | 38% |
| Dev | 22 | 18 | 1 | 1 | 2 | 0.95 | 0.95 | 0.95 | 0.91 | 52.6% (10/19) | 33% |
| Test | 31 | 25 | 0 | 2 | 4 | 1.00 | 0.93 | 0.96 | 0.94 | 48.1% (13/27) | 0% |

Field accuracy, measured only on true-positive commitment detections:

| Field | Train | Dev | Test |
|---|---:|---:|---:|
| `committed_party` | 100% | 94% | 96% |
| `required_action` | 71% | 78% | 76% |
| `deadline` | 86% | 78% | 80% |
| `context` | 82% | 89% | 84% |
| `status` | 100% | 100% | 96% |

**Reading these numbers:** the binary act-vs-ignore decision (should this message produce a commitment at all?) is solid across all three splits. The full-commitment metric is intentionally much stricter — it requires the right party, action, deadline, context, and status simultaneously — and sits around 50% everywhere, which is the real signal on where to invest next.

With `--llm-judge` enabled, the dev split's full-commitment metric rises from 52.6% to 68.4% — the gap is almost entirely semantic equivalence (`reserve` vs `book`, `by Friday` vs `Friday`) that the deterministic metric penalizes but a judge correctly accepts. This confirms the localizer's top-2 repair recommendation: fix `required_action` and `context` matching in the metric/postprocessing layer first.

### Failure localization

`eval/localize.py` takes the `failures.jsonl` from an eval run and buckets every failure into a root cause, subcause, repair type, and a `priority = impact × confidence / cost` score, so effort goes to the highest-leverage fix first rather than the loudest one:

```bash
python scripts/eval_runner.py --all --save        # writes runs/<run_id>/{train,dev,test,summary}.md + failures.jsonl
python -m eval.localize runs/<run_id>/failures.jsonl          # rich table
python -m eval.localize runs/<run_id>/failures.jsonl --json    # machine-readable
```

Across all 45 failures in the latest run:

| Root Cause | Count | Top Subcause | Repair Type | Priority |
|---|---:|---|---|---:|
| `required_action_normalization` | 15 | verb/object too specific vs. expected | metric/postprocess | 6.8 |
| `context_metric_noise` | 7 | paraphrase, same meaning | metric | 6.3 |
| `deadline_normalization` | 7 | missing/extra `by` prefix | metric/postprocess | 3.1 |
| `under_extraction_policy` | 6 | external party / group obligation | signature rule | 1.7 |
| `update_vs_new_matching` | 4 | new commitment mismatched to existing one | postprocess | 1.5 |
| `over_extraction_policy` | 2 | refusal read as acceptance | signature rule | 0.6 |
| `lifecycle_policy` | 2 | "started"/"almost done" read as done | signature rule | 0.6 |
| `party_resolution` | 2 | third-party obligation implied by role | signature rule | 0.6 |

The top two repairs are both about **normalization, not the model's judgment** — `required_action` and `context` mismatches are usually semantically correct but worded differently than the reference (`send over the docs` vs `send the documents`), which the current exact/near-exact matching penalizes. That's cheap to fix in the metric or a postprocessing step, before touching the prompt at all.

One pattern worth calling out even though it scores lower on priority: several `under_extraction_policy` and `party_resolution` misses across *all three* splits are the same shape — a commitment implied by someone's role or an external party rather than stated directly (*"I'm waiting for the bank to process the loan"*, *"we need to send the documents"*, *"the contractor needs to finish by next week"*). These recur often enough that they're a real, cross-split gap even though today's failure count per category is small.

The localizer is validated with 16 deterministic tests (`tests/test_localize.py`) against fixed fake failures in `tests/evals/localize_cases/`, so its scoring logic isn't just eyeballed.

### Comparing runs

`scripts/compare_runs.py` diffs two eval runs side by side — showing failures fixed, new failures, and persistent failures per scenario:

```bash
python scripts/compare_runs.py runs/<run_a> runs/<run_b>
```

Output is a rich console table (summary, per-category, per-scenario diff) saved as markdown to `runs/compares/<run_a>_vs_<run_b>.md`.

## Interactive failure inspector

`scripts/eval_inspector.py` is an interactive CLI agent that either loads an existing run or runs a fresh eval, then drops into a REPL for examining each failure without leaving the terminal:

```bash
# Load an existing run (interactive picker lists recent runs)
python scripts/eval_inspector.py

# Load a specific run directly
python scripts/eval_inspector.py --run 20260707_183434-judge

# Run a fresh eval on a split, then enter inspector
python scripts/eval_inspector.py --split dev
python scripts/eval_inspector.py --split dev --limit 5 --llm-judge
```

When loading an existing run, the inspector joins `failures.jsonl` with `predictions.jsonl` (on `split + category + scenario`) to recover the original inputs (`chat_id`, `current_datetime`, `existing_commitments_json`) needed for re-runs.

### REPL commands

| Command | Description |
|---|---|
| `list` / `ls` | Numbered table of all failures (or filtered subset) |
| `<N>` | Show full detail for failure #N: input messages, expected vs actual side-by-side diff, root-cause localization |
| `next` / `prev` | Navigate to next/previous failure from detail view |
| `filter <key>=<value>` | Filter by `error_type`, `category`, `difficulty`, `split`, or `field` |
| `filter clear` | Clear all filters |
| `filters` | Show active filters |
| `localize` | Run failure localization on the current (filtered) failure set — shows root-cause summary table + top suggested repairs |
| `rerun <N>` | Re-run failure #N through a fresh `CommitmentAgent` — shows pass/fail, what was fixed, what's still wrong, and any new problems |
| `summary` | Aggregate counts by error type, category, and difficulty |
| `help` / `h` | Show available commands |
| `quit` / `q` | Exit |

The detail view shows three sections: (1) the input messages and metadata, (2) a field-by-field expected vs actual diff table with ✓/✗/~ indicators, and (3) root cause, subcause, repair type, confidence, and suggested repair text from the localizer.

The `rerun` command is useful after making code changes — it re-runs a single failing example through a fresh agent instance and shows whether the fix worked, without re-running the entire eval set.

## Data model

| Table | Status | Notes |
|---|---|---|
| `Tenant` | in use | multi-tenant root |
| `WhatsAppAccount` | in use | one row per Green API instance / 360dialog number; stores `provider` + `provider_instance_id`, **not** API credentials |
| `Contact` | in use | one row per `(tenant_id, chat_id)`; `kind` field (`client`/`bank`/`lawyer`/`internal`/`family`/`unknown`) exists but nothing populates it yet |
| `CommitmentItem` | in use | the actual product surface — party, action, deadline, status, context, source message ids |
| `Chat`, `ChatMessage` | **defined, not wired up** | raw per-message history is not persisted anywhere; messages only exist transiently in the in-memory buffer until drained, then are discarded once extraction runs |
| `WaitingStatus`, `WaitingParty`, `Urgency`, `WaitingItemStatus` enums | **defined, no table** | scaffolding for a more general waiting/urgency model beyond commitments |

**In-memory only (lost on process restart):**
- `accounts_by_instance`, `contacts_by_tenant_chat_id` — rebuilt from the DB on startup via `load_cache()`, so these are safe.
- `MessageBuffer._messages_by_chat` — **not** rebuilt from anywhere. Any message ingested but not yet drained when the process restarts (deploy, crash, autoscale) is gone permanently, and there's no raw-message table to recover it from either. This is the single biggest reliability gap right now.

**Also unwired:** `app/services/green_api_client.py` defines a `GreenApiClient` (get chats / group data / chat history) but nothing in the app instantiates it, and `WhatsAppAccount` has no field to store the per-instance token it would need anyway — this is scaffolding for a future feature (e.g. backfilling chat history), not a bug in the current flow.

## Setup

There's no `requirements.txt` — dependencies are pinned in `pyproject.toml` / `uv.lock`.

```bash
# recommended (uv.lock is committed, so this is reproducible)
uv sync

# or with plain pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp example.env .env
```

## Configuration

Edit `.env` (`example.env` currently only lists a subset of these — worth syncing it up):

| Var | Required | Default | Purpose |
|---|---|---|---|
| `D360_API_KEY` | **yes** — app raises on startup without it | — | needed even if you're only using the Green API path |
| `D360_API_BASE_URL` | no | `https://waba-v2.360dialog.io` | use the sandbox URL for testing |
| `OPENAI_API_KEY` | yes, for either pipeline to function | — | used by both the commitment extractor and the 360dialog agent |
| `OPENAI_MODEL` | no | `gpt-5.4-mini` | model for commitment extraction and agent replies |
| `OPENAI_TRANSCRIBE_MODEL` | no | `gpt-4o-transcribe` | voice-note transcription |
| `WEBHOOK_AUTH_MODE` | no | `none` | `none` / `bearer` / `basic` — protects `/webhook/360dialog` only, not `/webhook/green-api` |
| `WEBHOOK_BEARER_TOKEN` | if `bearer` mode | — | |
| `WEBHOOK_BASIC_USER` / `WEBHOOK_BASIC_PASS` | if `basic` mode | — | |
| `MAX_GROUP_PARTICIPANTS` | no | `3` | group chats with more distinct senders than this are skipped entirely |
| `ALLOWED_CHAT_IDS` | no | empty (no filter) | parsed into settings and exposed via `/debug/settings`, but **not currently enforced** in the Green API webhook handler |
| `DATABASE_PATH` | no | `tami.db` | SQLite file, used when `DATABASE_URL` is unset |
| `DATABASE_URL` | no | — | Postgres URL; takes priority over `DATABASE_PATH` when set |
| `LOG_LEVEL` | no | `INFO` | |
| `LANGSMITH_TRACING_V2` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | no | tracing off | optional observability |
| `CONVERSATION_MAX_MESSAGES` | no | `20` | per-thread history cap for the 360dialog agent |

`GREEN_API_BASE_URL` / `GREEN_API_TOKEN` currently appear in `example.env` but aren't read anywhere in the code — Green API credentials live in the `WhatsAppAccount` table lookup path, not a global env var. Safe to ignore or remove.

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
```

## Seed the database

```bash
# Local, against Postgres
DATABASE_URL="postgresql://user:password@host:5432/dbname" .venv/bin/python scripts/seed.py

# On Render, after first deploy
curl -X POST https://your-app.onrender.com/admin/seed
```

⚠️ `run_seed(overwrite=True)` drops the entire `public` schema before recreating tables. The `/admin/seed` route currently calls it with `overwrite=False`, but the route has no auth of its own — don't leave it reachable on a public URL longer than you need to.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Covers Green API payload normalization, contact upsert + buffering, webhook handling, commitment extraction helpers, commitment-eval utilities, the failure localizer (16 deterministic tests), and the LLM judge (11 tests with mocked DSPy responses).

Run the commitment eval, optionally with LLM-as-judge, and regenerate the deterministic train/dev/test splits if the YAML fixtures change:

```bash
python scripts/eval_runner.py --all              # deterministic (default)
python scripts/eval_runner.py --all --llm-judge   # with LLM judge
python tests/evals/generate_devset.py
```

## Deploy (Render)

1. Connect the repo.
2. Set env vars (`D360_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, etc. — see [Configuration](#configuration)).
3. Build: `pip install -e .`
4. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Seed once: `curl -X POST https://your-app.onrender.com/admin/seed`

Because the drain loop and message buffer live inside the app process, **any redeploy or restart between ingestion and the 2-minute drain cycle loses buffered-but-undrained messages**, and Render's free tier spinning an idle instance down has the same effect. This is the top item in the roadmap below.

## Endpoints

| Endpoint | Provider | Purpose |
|---|---|---|
| `POST /webhook/green-api` | Green API | Ingest personal WhatsApp messages, buffer for commitment extraction |
| `POST /webhook/360dialog` | 360dialog | Business WhatsApp — AI agent replies, text + voice |
| `POST /admin/seed` | — | Seed demo data (unauthenticated) |
| `GET /health` | — | Health check |
| `GET /debug/settings` | — | Dev debug info (no secrets) |

## Known limitations / roadmap

Roughly in priority order:

1. **Durable ingestion.** Persist `ChatMessage` rows on arrival (the table already exists) instead of relying solely on the in-memory buffer. This removes the "lose messages on restart" risk and gives a permanent audit trail independent of what the LLM does with them.
2. **Green API webhook auth is disabled** (`verify_green_api_authorization` is defined but commented out in `personal_webhook.py`). Re-enable before this is exposed on a public URL for real.
3. **`required_action` and `context` wording drive most eval failures** (15 + 7 of 45, see [Failure localization](#failure-localization)) — mostly verb/object synonyms and paraphrasing that the current metric treats as wrong even when semantically correct. Highest-leverage fix available, and it's in the metric/postprocessing layer, not the prompt.
4. **`deadline` is free text** (`"in one hour"`, `"by Friday"`), not a resolved timestamp — 7 failures trace to missing/extra `by`-style phrasing. Needs normalizing before it can drive urgency sorting or scheduled digests.
5. **Commitment dedup/update matching is still partly model-driven** (matching by `id` inside one extraction call); 4 failures are the model creating a near-duplicate instead of updating the existing row. Worth a deterministic fallback matcher.
6. **Under-extraction on implied/external-party commitments** — "waiting on the bank," "the contractor needs to," "we need to" — recurs across all three splits. Loosening the act-vs-ignore rule for implied-party cases is the natural next prompt change.
7. **`Contact.kind` (client/bank/lawyer/…) is modeled but not populated.** No classifier sets it yet, so nothing downstream can filter or route by relationship type.
8. **No digest job yet.** `CommitmentItem` data is good enough to query; a scheduled per-tenant job rendering waiting commitments into a daily WhatsApp message is the natural next milestone.
9. **`/admin/seed` has no auth of its own** — fine for a private dev deploy, worth gating before wider use.
