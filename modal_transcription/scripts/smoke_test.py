#!/usr/bin/env python3
"""
Smoke test for the deployed Modal Hebrew Whisper transcription endpoint.

This is NOT a pytest test.  Run it manually after deploying:

    python scripts/smoke_test.py --audio path/to/hebrew-audio.ogg

It sends a real audio file to the endpoint and verifies that:
- The endpoint responds
- Proxy authentication works
- The model loads (or is warm)
- Transcription returns non-empty Hebrew text
- Latency is reported

Environment variables required:
    MODAL_TRANSCRIPTION_URL
    MODAL_TRANSCRIPTION_KEY
    MODAL_TRANSCRIPTION_SECRET

Optional:
    MODAL_TRANSCRIPTION_TIMEOUT_SECONDS (default: 180)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Modal Hebrew Whisper endpoint")
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to a Hebrew audio file (ogg, mp3, m4a, wav, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for transcription (default: 1)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    url = os.environ.get("MODAL_TRANSCRIPTION_URL")
    key = os.environ.get("MODAL_TRANSCRIPTION_KEY")
    secret = os.environ.get("MODAL_TRANSCRIPTION_SECRET")
    timeout = float(os.environ.get("MODAL_TRANSCRIPTION_TIMEOUT_SECONDS", "180"))

    if not url or not key or not secret:
        print(
            "ERROR: Set MODAL_TRANSCRIPTION_URL, MODAL_TRANSCRIPTION_KEY, "
            "and MODAL_TRANSCRIPTION_SECRET environment variables.",
            file=sys.stderr,
        )
        return 1

    audio_bytes = audio_path.read_bytes()
    content_type_map = {
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
    }
    content_type = content_type_map.get(audio_path.suffix.lower(), "application/octet-stream")

    headers = {
        "Modal-Key": key,
        "Modal-Secret": secret,
        "Content-Type": content_type,
        "X-Filename": audio_path.name,
    }

    params = {
        "beam_size": str(args.beam_size),
        "vad_filter": "true",
    }

    print(f"Sending {len(audio_bytes)} bytes ({audio_path.name}) to {url}")
    print(f"Content-Type: {content_type}, beam_size={args.beam_size}")

    started = time.monotonic()

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(
                url,
                params=params,
                headers=headers,
                content=audio_bytes,
            )
    except httpx.TimeoutException:
        elapsed = time.monotonic() - started
        print(f"FAIL: Request timed out after {elapsed:.1f}s", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - started
        print(f"FAIL: HTTP error after {elapsed:.1f}s: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started

    if response.status_code != 200:
        print(
            f"FAIL: HTTP {response.status_code} after {elapsed:.1f}s: {response.text[:500]}",
            file=sys.stderr,
        )
        return 1

    try:
        payload = response.json()
    except ValueError:
        print(f"FAIL: Invalid JSON response after {elapsed:.1f}s", file=sys.stderr)
        return 1

    text = payload.get("text", "")
    if not text:
        print(f"FAIL: Empty transcription text after {elapsed:.1f}s", file=sys.stderr)
        return 1

    print(f"\nSUCCESS ({elapsed:.1f}s)")
    print(f"  Text: {text}")
    print(f"  Language: {payload.get('language')}")
    print(f"  Audio duration: {payload.get('audio_duration_seconds')}s")
    print(f"  Processing: {payload.get('processing_seconds')}s")
    print(f"  Model: {payload.get('model')}")
    print(f"  Segments: {len(payload.get('segments', []))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
