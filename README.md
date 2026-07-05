# Tami One

WhatsApp assistant backend built with FastAPI. Receives messages via Green API webhooks, stores contacts and messages, and processes them through an AI agent. Also supports 360dialog Direct API for business WhatsApp.

## Architecture

```
app/
├── main.py                  # FastAPI app, startup (init_db + load_cache)
├── config.py                # Settings from env vars
├── routers/
│   ├── green_api.py         # Green API webhook normalization → MessageEvent
│   └── personal_webhook.py  # /webhook/green-api endpoint, /admin/seed
│   └── business_webhook.py  # /webhook/360dialog endpoint (360dialog)
├── db/
│   ├── engine.py            # SQLAlchemy engine (PostgreSQL or SQLite)
│   ├── models.py            # SQLModel tables: Tenant, WhatsAppAccount, Contact, Chat, ChatMessage
│   ├── cache.py             # In-memory cache + message queue
│   ├── upsert.py            # Insert-only contact creation + message queueing
│   └── seed.py              # Demo data seeding
├── agents/
│   ├── core.py              # OpenAI agent with conversation memory
│   └── memory.py            # Per-user in-memory conversation history
└── services/
    ├── whatsapp.py          # 360dialog API client
    └── transcription.py     # Audio transcription via OpenAI
scripts/
└── seed.py                  # CLI seed entry point
tests/
└── test_upsert.py           # Upsert + message queue tests
```

## Data Flow

1. **Green API webhook** → `POST /webhook/green-api`
2. Payload normalized into `MessageEvent` (chat_id, direction, text, etc.)
3. `upsert_contact_and_chat()`:
   - Looks up `WhatsAppAccount` in cache by `idInstance`
   - Looks up `Contact` in cache by `(tenant_id, chat_id)` — inserts if new
   - Appends `MessageEvent` to in-memory queue `messages_by_chat[(tenant_id, chat_id)]`
4. Messages are queued for later processing (pop + process step coming next)

## Database

Supports **PostgreSQL** (production, Render) and **SQLite** (local dev).

**Tables:**
- **Tenant** — multi-tenant root
- **WhatsAppAccount** — Green API instance (provider, instance_id, chat_id)
- **Contact** — WhatsApp contact (tenant_id, chat_id, display_name)
- **Chat** — conversation (tenant_id, provider_chat_id, is_group, primary_contact_id)
- **ChatMessage** — individual message (tenant_id, chat_id, provider_message_id, direction, text)

**In-memory cache** (`app/db/cache.py`):
- `accounts_by_instance` — provider_instance_id → WhatsAppAccount
- `contacts_by_tenant_chat_id` — (tenant_id, chat_id) → Contact
- `messages_by_chat` — (tenant_id, chat_id) → list[MessageEvent] (pending processing)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env
```

Edit `.env` with your keys (see `example.env` for all options).

Key env vars:
- `DATABASE_URL` — PostgreSQL connection string (takes priority over SQLite)
- `DATABASE_PATH` — SQLite file path (local dev default: `tami.db`)
- `D360_API_KEY` — 360dialog API key
- `OPENAI_API_KEY` — OpenAI API key
- `ALLOWED_CHAT_IDS` — comma-separated whitelist of chat IDs

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Check health:
```bash
curl http://localhost:8000/health
```

## Seed the database

Local (with PostgreSQL):
```bash
DATABASE_URL="postgresql://user:password@host:5432/dbname" .venv/bin/python scripts/seed.py
```

On Render:
```bash
curl -X POST https://your-app.onrender.com/admin/seed
```

Seeding with `overwrite=True` drops the entire `public` schema and recreates all tables from scratch.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Deploy (Render)

1. Connect repo to Render
2. Set env vars (DATABASE_URL, D360_API_KEY, OPENAI_API_KEY, etc.)
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. After first deploy, seed: `curl -X POST https://your-app.onrender.com/admin/seed`

## Webhook endpoints

| Endpoint | Provider | Purpose |
|---|---|---|
| `POST /webhook/green-api` | Green API | Ingest WhatsApp messages, queue for processing |
| `POST /webhook/360dialog` | 360dialog | Business WhatsApp with AI agent replies |
| `POST /admin/seed` | — | Seed demo data (drops all tables first) |
| `GET /health` | — | Health check |
| `GET /debug/settings` | — | Dev debug (no secrets) |
