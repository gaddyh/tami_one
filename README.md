# 360dialog Echo Bot

Tiny FastAPI WhatsApp bot for 360dialog Direct API.

It receives inbound WhatsApp text messages at `/webhook/360dialog` and replies to the sender with:

```text
echo: <their message>
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env
```

Edit `.env`:

```env
D360_API_KEY=your_360dialog_phone_number_api_key
D360_API_BASE_URL=https://waba-v2.360dialog.io
WEBHOOK_AUTH_MODE=none
```

For the 360dialog sandbox:

```env
D360_API_BASE_URL=https://waba-sandbox.360dialog.io/v1
```

## 2. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Check:

```bash
curl http://localhost:8000/health
```

## 3. Expose publicly

Use Render, Railway, Fly.io, ngrok, Cloudflare Tunnel, etc.

Your webhook URL will be:

```text
https://your-domain.com/webhook/360dialog
```

## 4. Configure 360dialog webhook

In 360dialog Hub, configure the **Phone Number / Channel webhook** for the registered WhatsApp number:

```text
https://your-domain.com/webhook/360dialog
```

Use the phone-number/channel webhook, not the WABA-level fallback, unless you intentionally want one webhook for all numbers in the WABA.

## 5. Test

Send a WhatsApp text message to the registered bot number.

Expected reply:

```text
echo: your message
```

## Notes

- Free-form text replies only work when the WhatsApp customer-service window is open. The easiest test is to message the bot number first from your personal WhatsApp.
- For production, return quickly and move heavier work to a queue/background worker. This echo bot sends inline to keep the repo tiny.
- Keep your `D360_API_KEY` secret. Do not commit `.env`.
