# Hebrew Whisper Transcription — Modal Service

Standalone Modal Web Function for Hebrew audio transcription using
`ivrit-ai/whisper-large-v3-turbo-ct2` (Faster-Whisper / CTranslate2).

## Architecture

```
Render FastAPI app
  └── ModalWhisperTranscriber
        └── ModalTranscriptionClient (HTTP)
              └── Modal Web Function (this service)
                    └── HebrewWhisper GPU class (T4, FP16)
                          └── faster-whisper + CTranslate2
```

The Render app sends raw audio bytes via HTTPS POST with Modal proxy
authentication.  This service transcribes and returns JSON.

## Setup

### 1. Virtual environment

```bash
cd modal_transcription
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade modal "fastapi[standard]"
```

### 2. Developer CLI authentication

This authenticates **you** as a developer to deploy to Modal:

```bash
modal setup
```

This creates `~/.modal.toml` with your deploy token.

### 3. Proxy authentication credentials

These are **different** from your CLI deploy token.  Proxy auth credentials
are used by Render to authenticate HTTP calls to your Modal endpoint.

Modal proxy auth uses two headers:
- `Modal-Key`
- `Modal-Secret`

Create proxy credentials via the Modal dashboard or CLI.  Save the key
and secret — you will set them as environment variables on Render:

```
MODAL_TRANSCRIPTION_KEY=<your-modal-key>
MODAL_TRANSCRIPTION_SECRET=<your-modal-secret>
```

The endpoint is deployed with `requires_proxy_auth=True`, so unauthorized
requests are rejected by Modal's proxy layer **before** your Python handler
runs or a GPU container starts.

### 4. Development

```bash
modal serve app.py
```

Modal builds the container image, starts the app, and prints temporary
public URLs.  It watches for source changes and live-reloads.

You should see URLs resembling:

```
Created web function health:
  https://your-workspace--hebrew-whisper-transcription-health-dev.modal.run

Created web function HebrewWhisper.transcribe:
  https://your-workspace--hebrew-whisper-transcription-hebrewwhisper-transcribe-dev.modal.run
```

### 5. Test the health endpoint

```bash
curl "https://YOUR-HEALTH-URL.modal.run"
```

Expected:

```json
{"status": "ok", "service": "hebrew-transcription"}
```

This only proves the deployment exists and the HTTP route responds.
It does **not** prove the model is loaded or the GPU is available.

### 6. Test transcription with a Hebrew audio file

```bash
export MODAL_TRANSCRIPTION_URL="https://YOUR-TRANSCRIBE-URL.modal.run"
export MODAL_TRANSCRIPTION_KEY="your-modal-key"
export MODAL_TRANSCRIPTION_SECRET="your-modal-secret"

# Using the smoke test script:
python scripts/smoke_test.py --audio path/to/hebrew-audio.ogg

# Or with curl:
curl -L \
  -X POST \
  "$MODAL_TRANSCRIPTION_URL?beam_size=1&vad_filter=true" \
  -H "Modal-Key: $MODAL_TRANSCRIPTION_KEY" \
  -H "Modal-Secret: $MODAL_TRANSCRIPTION_SECRET" \
  -H "Content-Type: audio/ogg" \
  -H "X-Filename: voice-note.ogg" \
  --data-binary "@voice-note.ogg"
```

`-L` follows redirects.  Modal Web Functions return 303 redirects for
requests exceeding 150 seconds (e.g., during cold model load).

Expected response:

```json
{
  "text": "שלום, רציתי לבדוק אם אפשר לקבוע פגישה למחר בבוקר.",
  "language": "he",
  "language_probability": 1.0,
  "audio_duration_seconds": 5.82,
  "duration_after_vad_seconds": 5.31,
  "processing_seconds": 1.42,
  "model": "ivrit-ai/whisper-large-v3-turbo-ct2",
  "segments": [...]
}
```

### 7. Deploy to production

```bash
modal deploy app.py
```

This creates persistent URLs that remain available after your local
process exits.  Save the transcription URL.

### 8. Configure Render

Set these environment variables on Render:

```
TRANSCRIPTION_PROVIDER=modal
MODAL_TRANSCRIPTION_URL=https://your-production-transcribe-url.modal.run
MODAL_TRANSCRIPTION_KEY=your-modal-key
MODAL_TRANSCRIPTION_SECRET=your-modal-secret
MODAL_TRANSCRIPTION_TIMEOUT_SECONDS=180
```

## Timeout and retry policy

- **Connect timeout**: 20 seconds
- **Overall timeout**: 180–240 seconds
- **Follow redirects**: enabled (Modal returns 303 for requests >150s)
- **Retry**: connection reset/connect failures, 502/503/504
- **Retry 500**: once only if request is known idempotent
- **Do not retry**: 400, 401, 413, 422
- **Backoff**: exponential with jitter

## GPU and scaling

- **GPU**: NVIDIA T4 (~$0.000164/sec, ~$0.59/hour)
- **`min_containers=0`**: scales to zero when idle (no idle cost)
- **`scaledown_window=60`**: container stays warm 60s after last request
- **`max_containers=2`**: concurrent request cap

Model weights persist in a Modal Volume (`/model-cache`), so cold starts
after the first only need to load weights into GPU memory, not re-download.

## Model settings

- `language="he"` — skip language detection
- `beam_size=1` — fast, sufficient for short voice notes
- `vad_filter=True` — remove silent regions
- `compute_type="float16"` — GPU-optimized

## Logging

Structured logs include: `event`, `audio_bytes`, `audio_duration_seconds`,
`processing_seconds`, `segment_count`.

**Not logged**: transcript text, authentication credentials, raw audio.
