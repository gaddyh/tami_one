# Tami One

A WhatsApp-native assistant for a mortgage/loan advisor. It watches the advisor's real WhatsApp conversations (personal number, via Green API) and automatically extracts **commitments** — who owes whom what, and by when — so the advisor never has to re-read a thread to remember what's still open.

Separately, it can run as a **360dialog-based business bot** that replies to customers directly with an OpenAI agent (text + voice notes).

These are two independent capabilities sharing one FastAPI app and one database.

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

This is the core proof-of-concept the project has already validated: given a real, mixed-Hebrew/English WhatsApp thread, the pipeline correctly separates a fulfilled promise (`status=DONE`) from a still-waiting one (`status=waiting`), attributes each to the right party, and keeps a plain-language `context` trail for why.

---

## Architecture

```
app/
├── main.py                    # FastAPI app; startup wires init_db + load_cache
│                               # + a background drain loop (every 10 min)
├── config.py                  # Settings, loaded from env vars at import time
├── routers/
│   ├── green_api.py            # Green API payload → normalized MessageEvent
│   ├── personal_webhook.py     # POST /webhook/green-api, /admin/seed, /debug/settings
│   └── business_webhook.py     # POST /webhook/360dialog — agent-driven customer replies
├── db/
│   ├── engine.py                # SQLAlchemy engine (Postgres or SQLite)
│   ├── models.py                # SQLModel tables (see below)
│   ├── cache.py                 # In-memory account/contact cache + MessageBuffer
│   ├── upsert.py                 # Insert-only Contact creation + buffering
│   └── seed.py                   # Demo data seeding
├── commitments/
│   ├── models.py                 # Canonical Commitment / CommitmentList schema
│   ├── commitments_agent.py      # DSPy Signature + Module for commitment extraction
│   ├── extractor.py              # Adapter: messages + existing commitments → DSPy agent
│   └── processor.py              # Drains buffer, calls extractor, upserts CommitmentItem rows
├── eval/
│   ├── metrics.py                # DSPy eval metrics: commitment_metric, act_vs_ignore, token-F1
│   └── dataset.py                # Example construction + devset loading
├── agents/
│   ├── core.py                    # OpenAI agent used by the 360dialog business bot
│   └── memory.py                   # Per-thread in-memory conversation history
└── services/
    ├── whatsapp.py                 # 360dialog API client
    └── transcription.py             # Voice-note transcription via OpenAI
scripts/seed.py                      # CLI seed entry point
scripts/eval_runner.py               # Local commitment-extraction eval runner
tests/evals/
├── data/                             # YAML example definitions (6 categories + challenge + schema)
│   ├── _schema.yaml                  # Shared constants: chat_id, chat_name, defaults
│   ├── act_vs_ignore.yaml            # Category A: act vs ignore boundary (18 examples)
│   ├── args_party.yaml               # Category B: committed_party extraction (15)
│   ├── args_deadline.yaml            # Category C: deadline extraction (15)
│   ├── args_required_action.yaml     # Category D: required_action extraction (15)
│   ├── lifecycle_update_vs_new.yaml  # Category E: update vs new commitment (15)
│   ├── lifecycle_completion.yaml     # Category F: done/dismissed status (15)
│   └── challenge_act_ignore.yaml     # Challenge split: hard act/ignore pairs (36)
├── generate_devset.py                # Data-driven generator: YAML → train/dev/test JSON splits
├── seed_examples.json                # Legacy hand-written reference examples
├── trainset.json                     # Generated (40 examples)
├── devset.json                       # Generated (22 examples)
├── testset.json                      # Generated (31 examples)
└── challenge_act_ignore.json         # Generated challenge split (36 examples)
tests/                               # upsert, webhook, commitment extraction/eval tests
```

## How a message becomes a commitment

1. **Ingest** — Green API POSTs to `/webhook/green-api`. `normalize_green_api_message_event()` turns the raw payload into a `MessageEvent` (chat id, sender, direction, text, timestamp).
2. **Upsert + buffer** — `upsert_contact_and_chat()`:
   - Creates a `Contact` row if this `(tenant_id, chat_id)` hasn't been seen before (insert-only; existing contacts are never updated).
   - Filters out chats with more distinct senders than `MAX_GROUP_PARTICIPANTS` (large groups are skipped — this is a 1:1/small-group tool, not a broadcast-list reader).
   - Appends the event to an **in-memory** `MessageBuffer`, keyed by `(tenant_id, chat_id)`.
3. **Drain** — a background `asyncio` task inside the running app process wakes up every 10 minutes, atomically drains the whole buffer, and hands each chat's batch of messages to the commitment extractor.
4. **Extract** — `extract_commitments()` sends the batch, plus that chat's currently-open commitments, through a DSPy `CommitmentAgent` (`ExtractCommitments` signature + `dspy.Predict`). DSPy is configured at startup with `dspy.JSONAdapter()` so the extractor behaves like a structured-output call while staying measurable and optimizable.
5. **Persist** — `CommitmentItem` rows are upserted into Postgres/SQLite, tagged with the `source_message_ids` that produced them.

The 360dialog path (`/webhook/360dialog`) is unrelated to this flow — it's a direct reply bot for business-number customers, using `agents/core.py` for in-memory per-thread conversation and `services/transcription.py` for voice notes.


## Commitment extraction evals

The commitment extractor has been refactored from a one-off raw OpenAI `responses.parse` call into a DSPy `Signature + Module` pattern:

- `Commitment` / `CommitmentList` remain the canonical domain schema in `app/commitments/models.py`.
- `app/commitments/commitments_agent.py` defines `ExtractCommitments(dspy.Signature)` and `CommitmentAgent(dspy.Module)`.
- `extract_commitments()` keeps the same public interface used by `processor.py`, but internally calls the DSPy agent and normalizes `chat_id` / `chat_name` after the LLM response.
- `eval/metrics.py` contains a strict full-commitment metric, with token-F1 matching for `required_action` so small wording differences are not always counted as failures.
- `eval/dataset.py` handles example construction and devset loading.

The generated eval dataset is made of controlled probes rather than random examples. Each example is tagged by:

| Dimension | Values |
|---|---|
| Category | `act_vs_ignore`, `args_party`, `args_deadline`, `args_required_action`, `lifecycle_update_vs_new`, `lifecycle_completion` |
| Difficulty | `easy`, `medium`, `hard` |
| Scenario | A specific contrastive case, such as `request_without_acceptance_ignore`, `almost_done_not_done`, or `party_implied_by_role` |

The hand-written examples are preserved separately as `tests/evals/seed_examples.json` (a legacy reference dataset, not used by the eval runner). The data-driven generator reads YAML definitions from `tests/evals/data/` and writes JSON splits:

```bash
python tests/evals/generate_devset.py
```

Current generated split sizes:

| Split | Examples |
|---|---:|
| Train | 40 |
| Dev | 22 |
| Test | 31 |
| Total | 93 |

### Latest eval results

Command:

```bash
python scripts/eval_runner.py --all
```

Latest run (model: `gpt-5.4-mini`, 2026-07-07):

| Split | N | TP | FP | FN | TN | Precision | Recall | F1 | Act/Ignore Accuracy | Full Commitment Metric | Over-Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 40 | 28 | 3 | 4 | 5 | 0.90 | 0.88 | 0.89 | 0.82 | 43.8% | 38% |
| Dev | 22 | 18 | 1 | 1 | 2 | 0.95 | 0.95 | 0.95 | 0.91 | 57.9% | 33% |
| Test | 31 | 25 | 0 | 2 | 4 | 1.00 | 0.93 | 0.96 | 0.94 | 48.1% | 0% |

Field accuracy on the test split, measured only on true-positive commitment detections:

| Field | Accuracy |
|---|---:|
| `committed_party` | 96% |
| `required_action` | 76% |
| `deadline` | 80% |
| `context` | 84% |
| `status` | 96% |

Interpretation:

- The extractor is already strong at the binary decision of whether a message should produce a commitment (`Act/Ignore Accuracy = 0.94` on test).
- The stricter full-object metric is intentionally harder (`48.1%` on test), because it requires the right party, action, deadline, context, status, and update/new behavior.
- Current weaknesses are mostly around `required_action` wording, deadline phrasing, and lifecycle edge cases such as update-vs-new or conditional completion.
- The eval now separates obvious examples from hard contrastive probes, so regressions show up by difficulty instead of being hidden by easy cases.

### Failure localization

After each eval run, `eval/localize.py` classifies every failure into a root cause with a subcause and computes a priority score (`impact * confidence / cost`) to rank suggested repairs.

```bash
python -m eval.localize runs/20260707_023551/failures.jsonl          # rich table
python -m eval.localize runs/20260707_023551/failures.jsonl --json   # JSON output
```

Top 2 suggested repairs from the latest run:

| # | Root Cause | Failures | Priority | Repair |
|---|---|---:|---:|---|
| 1 | `required_action_normalization` | 15 | 6.8 | Normalize/match action semantically |
| 2 | `context_metric_noise` | 7 | 6.3 | Soften context match (word overlap, not exact) |

The localizer is validated with 16 deterministic tests (`tests/test_localize.py`) using controlled fake failures in `tests/evals/localize_cases/`.

## Data model

| Table | Status | Notes |
|---|---|---|
| `Tenant` | in use | multi-tenant root |
| `WhatsAppAccount` | in use | one row per Green API instance / 360dialog number |
| `Contact` | in use | one row per `(tenant_id, chat_id)`; `kind` field (`client`/`bank`/`lawyer`/`internal`/`family`/`unknown`) exists but isn't populated by any classifier yet |
| `CommitmentItem` | in use | the actual product surface — party, action, deadline, status, context, source message ids |
| `Chat`, `ChatMessage` | **defined, not yet wired up** | raw per-message history is not currently persisted anywhere; messages only exist transiently in the in-memory buffer until drained, then are discarded once extraction runs |
| `WaitingStatus`, `WaitingParty`, `Urgency`, `WaitingItemStatus` enums | **defined, no table yet** | scaffolding for a more general waiting/urgency model beyond commitments |

**In-memory only (lost on process restart):**
- `accounts_by_instance`, `contacts_by_tenant_chat_id` — rebuilt from DB on startup via `load_cache()`, so these are safe.
- `MessageBuffer._messages_by_chat` — **not** rebuilt from anywhere. Any message ingested but not yet drained when the process restarts (deploy, crash, autoscale) is gone permanently, and there's no raw-message table to recover it from either.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env
```

Edit `.env` (see `example.env` for the full list). Notably:

| Var | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string — takes priority over SQLite when set |
| `DATABASE_PATH` | SQLite file path for local dev (default `tami.db`) |
| `OPENAI_API_KEY` | required — both the commitment extractor and the 360dialog agent use it |
| `OPENAI_MODEL` | model used for commitment extraction / agent replies |
| `D360_API_KEY` | required to start the app at all — `Settings.from_env()` raises if missing, even if you're only using the Green API path |
| `WEBHOOK_AUTH_MODE` | `none` / `bearer` / `basic` — protects `/webhook/360dialog` only |
| `ALLOWED_CHAT_IDS` | whitelist, referenced in settings (not currently enforced in the Green API webhook handler) |
| `MAX_GROUP_PARTICIPANTS` | group chats with more distinct senders than this are skipped entirely |

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

⚠️ `run_seed(overwrite=True)` drops the entire `public` schema before recreating tables. The `/admin/seed` route currently calls it with `overwrite=False`, but the route has no auth of its own — treat it as sensitive and don't leave it reachable on a public URL longer than you need to.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

Covers: Green API payload normalization, contact upsert + buffering, webhook handling, commitment extraction helpers, and commitment-eval utilities.

Run the commitment eval separately:

```bash
python scripts/eval_runner.py --all
```

Regenerate the deterministic train/dev/test eval splits:

```bash
python tests/evals/generate_devset.py
```

## Deploy (Render)

1. Connect the repo.
2. Set env vars (`DATABASE_URL`, `D360_API_KEY`, `OPENAI_API_KEY`, etc.).
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Seed once: `curl -X POST https://your-app.onrender.com/admin/seed`

Because the drain loop and message buffer live inside the app process, **any redeploy or restart on a 10-minute drain cycle can lose buffered-but-undrained messages.** This is the single biggest reliability gap right now — see below.

## Endpoints

| Endpoint | Provider | Purpose |
|---|---|---|
| `POST /webhook/green-api` | Green API | Ingest personal WhatsApp messages, buffer for commitment extraction |
| `POST /webhook/360dialog` | 360dialog | Business WhatsApp — AI agent replies, text + voice |
| `POST /admin/seed` | — | Seed demo data (unauthenticated) |
| `GET /health` | — | Health check |
| `GET /debug/settings` | — | Dev debug info (no secrets) |

## Known limitations / next up

Roughly in priority order:

1. **Durable ingestion.** Persist `ChatMessage` rows on arrival (the table already exists) instead of relying solely on the in-memory buffer. This alone removes the "lose messages on restart" risk and gives you a permanent audit trail independent of what the LLM does with them.
2. **Green API webhook auth is currently disabled** (`verify_green_api_authorization` is defined but commented out in `personal_webhook.py`). Re-enable before this is exposed on a public URL for real.
3. **Commitment dedup/update matching is still partly model-driven** (matching by `id` inside one extraction call). The eval now includes update-vs-new probes, but production should still add a deterministic fallback if the model creates a near-duplicate commitment.
4. **`deadline` is free text** (`"in one hour"`, `"by Friday"`, `"next Monday"`), not a resolved timestamp. The latest eval shows deadline extraction is one of the weaker fields, so this should be normalized before urgency sorting or scheduled digests.
5. **`required_action` wording is not yet stable enough.** The eval uses token-F1 instead of exact equality, but action normalization remains an important quality target.
6. **`Contact.kind` (client/bank/lawyer/…) is modeled but not populated.** No classifier currently sets it, so nothing downstream can yet filter or route by relationship type.
7. **No digest job yet.** The `CommitmentItem` data is now good enough to query; a scheduled per-tenant job that renders waiting commitments into a daily WhatsApp message is the natural next milestone.
8. **`/admin/seed` has no auth of its own** — fine for a private dev deploy, worth gating before wider use.
