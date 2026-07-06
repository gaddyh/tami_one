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
  "status": "open",
  "context": "Message: 'אני אשלח לך את החומר עוד שעה' ..."
}
```

This is the core proof-of-concept the project has already validated: given a real, mixed-Hebrew/English WhatsApp thread, the pipeline correctly separates a fulfilled promise (`status=DONE`) from a still-open one (`status=OPEN`), attributes each to the right party, and keeps a plain-language `context` trail for why.

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
│   ├── models.py                 # Commitment / CommitmentList (LLM output schema)
│   ├── extractor.py               # OpenAI call: messages + existing commitments → commitments
│   └── processor.py                # Drains buffer, calls extractor, upserts CommitmentItem rows
├── agents/
│   ├── core.py                    # OpenAI agent used by the 360dialog business bot
│   └── memory.py                   # Per-thread in-memory conversation history
└── services/
    ├── whatsapp.py                 # 360dialog API client
    └── transcription.py             # Voice-note transcription via OpenAI
scripts/seed.py                      # CLI seed entry point
tests/                                 # upsert, webhook, commitment extraction tests
```

## How a message becomes a commitment

1. **Ingest** — Green API POSTs to `/webhook/green-api`. `normalize_green_api_message_event()` turns the raw payload into a `MessageEvent` (chat id, sender, direction, text, timestamp).
2. **Upsert + buffer** — `upsert_contact_and_chat()`:
   - Creates a `Contact` row if this `(tenant_id, chat_id)` hasn't been seen before (insert-only; existing contacts are never updated).
   - Filters out chats with more distinct senders than `MAX_GROUP_PARTICIPANTS` (large groups are skipped — this is a 1:1/small-group tool, not a broadcast-list reader).
   - Appends the event to an **in-memory** `MessageBuffer`, keyed by `(tenant_id, chat_id)`.
3. **Drain** — a background `asyncio` task inside the running app process wakes up every 10 minutes, atomically drains the whole buffer, and hands each chat's batch of messages to the commitment extractor.
4. **Extract** — `extract_commitments()` sends the batch, plus that chat's currently-open commitments, to an OpenAI structured-output call. The model itself decides whether a message opens a new commitment, updates an existing one (matched by `id`), or changes nothing.
5. **Persist** — `CommitmentItem` rows are upserted into Postgres/SQLite, tagged with the `source_message_ids` that produced them.

The 360dialog path (`/webhook/360dialog`) is unrelated to this flow — it's a direct reply bot for business-number customers, using `agents/core.py` for in-memory per-thread conversation and `services/transcription.py` for voice notes.

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

Covers: Green API payload normalization, contact upsert + buffering, webhook handling, and commitment extraction against a real fixture payload (`tests/test_payload.json`).

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
3. **Commitment dedup relies entirely on the LLM's judgment** (matching by `id` inside one extraction call). There's no deterministic fallback if the model creates a near-duplicate commitment for something already tracked — worth a periodic dedup pass or a stricter matching heuristic.
4. **`deadline` is free text** (`"in one hour"`), not a resolved timestamp — fine for display, not usable yet for urgency sorting or a scheduled digest.
5. **`Contact.kind` (client/bank/lawyer/…) is modeled but not populated.** No classifier currently sets it, so nothing downstream can yet filter or route by relationship type.
6. **No digest job yet.** The `CommitmentItem` data is now good enough to query; a scheduled per-tenant job that renders open commitments into a daily WhatsApp message is the natural next milestone.
7. **`/admin/seed` has no auth of its own** — fine for a private dev deploy, worth gating before wider use.
