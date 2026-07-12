"""
Modal Web Function for Hebrew Whisper transcription.

Deployed separately from the Render FastAPI application.
The Render app calls this endpoint via HTTP with Modal proxy authentication.

Model: ivrit-ai/whisper-large-v3-turbo-ct2 (Faster-Whisper / CTranslate2)
GPU:   NVIDIA T4 (sufficient for FP16 large-v3-turbo)
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
from fastapi import HTTPException, Request, status

APP_NAME = "hebrew-whisper-transcription"

MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
MODEL_CACHE_DIR = "/model-cache"

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MiB

model_cache = modal.Volume.from_name(
    "hebrew-whisper-model-cache",
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "fastapi[standard]",
        "faster-whisper",
        "huggingface-hub[hf-xet]",
    )
    .env(
        {
            "HF_HOME": MODEL_CACHE_DIR,
            "HF_HUB_CACHE": f"{MODEL_CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _safe_suffix(content_type: str | None, filename: str | None) -> str:
    """Choose a safe temporary-file suffix without trusting the supplied path."""

    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in {
            ".ogg",
            ".oga",
            ".opus",
            ".mp3",
            ".m4a",
            ".mp4",
            ".wav",
            ".webm",
            ".aac",
            ".flac",
        }:
            return suffix

    content_type_map = {
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "audio/aac": ".aac",
        "audio/flac": ".flac",
        "application/ogg": ".ogg",
    }

    if content_type:
        normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
        return content_type_map.get(normalized, ".bin")

    return ".bin"


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> dict[str, str]:
    """CPU-only health check. Proves the deployment exists, not that the model is ready."""
    return {
        "status": "ok",
        "service": "hebrew-transcription",
    }


@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=300,
    scaledown_window=60,
    min_containers=0,
    max_containers=2,
)
class HebrewWhisper:
    @modal.enter()
    def load_model(self) -> None:
        """Run once whenever Modal starts a new container."""
        from faster_whisper import WhisperModel

        started = time.monotonic()

        self.model = WhisperModel(
            MODEL_ID,
            device="cuda",
            compute_type="float16",
            download_root=MODEL_CACHE_DIR,
        )

        print(
            {
                "event": "model_loaded",
                "model": MODEL_ID,
                "seconds": round(time.monotonic() - started, 2),
            }
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def transcribe(self, request: Request) -> dict[str, Any]:
        """
        Accept raw audio bytes as the HTTP request body.

        Expected headers:
          Content-Type: audio/ogg
          X-Filename: voice-note.ogg

        Optional query parameters:
          beam_size=1
          vad_filter=true
        """
        audio_bytes = await request.body()

        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body contains no audio",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio exceeds {MAX_AUDIO_BYTES} bytes",
            )

        try:
            beam_size = int(request.query_params.get("beam_size", "1"))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="beam_size must be an integer",
            ) from exc

        if not 1 <= beam_size <= 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="beam_size must be between 1 and 10",
            )

        vad_filter = (
            request.query_params.get("vad_filter", "true").lower() == "true"
        )

        filename = request.headers.get("x-filename")
        content_type = request.headers.get("content-type")
        suffix = _safe_suffix(content_type, filename)

        started = time.monotonic()
        temporary_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = temporary_file.name

            segments_generator, info = self.model.transcribe(
                temporary_path,
                language="he",
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=True,
            )

            segments = list(segments_generator)

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

            segment_results: list[dict[str, Any]] = []

            for segment in segments:
                item: dict[str, Any] = {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                }
                segment_results.append(item)

            elapsed = round(time.monotonic() - started, 3)

            print(
                {
                    "event": "transcription_complete",
                    "audio_bytes": len(audio_bytes),
                    "audio_duration_seconds": round(info.duration, 3),
                    "processing_seconds": elapsed,
                    "segment_count": len(segments),
                }
            )

            return {
                "text": text,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "audio_duration_seconds": round(info.duration, 3),
                "duration_after_vad_seconds": round(
                    getattr(info, "duration_after_vad", info.duration),
                    3,
                ),
                "processing_seconds": elapsed,
                "model": MODEL_ID,
                "segments": segment_results,
            }

        except HTTPException:
            raise
        except Exception as exc:
            print(
                {
                    "event": "transcription_failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio transcription failed",
            ) from exc

        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
