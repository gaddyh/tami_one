import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

from app.services.whatsapp import Dialog360Client

logger = logging.getLogger(__name__)

SUPPORTED_TRANSCRIPTION_EXTS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
}


def suffix_from_mime(mime_type: str) -> str:
    normalized = mime_type.split(";")[0].strip().lower()

    mapping = {
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/mpeg": ".mpeg",
        "audio/mpga": ".mpga",
        "audio/mp4": ".mp4",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
    }

    return mapping.get(normalized, ".bin")


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    audio_duration_seconds: float | None = None
    processing_seconds: float | None = None
    model: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


class TranscriptionError(RuntimeError):
    """Provider-neutral transcription failure."""


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult: ...

    async def close(self) -> None: ...


class OpenAITranscriber:
    """Transcriber implementation using the OpenAI Audio API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self._model = model
        self._openai = wrap_openai(AsyncOpenAI(api_key=api_key))

    async def transcribe_bytes(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> TranscriptionResult:
        suffix = Path(filename).suffix or suffix_from_mime(content_type)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            with tmp_path.open("rb") as audio_file:
                transcription = await self._openai.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                )

            return TranscriptionResult(
                text=transcription.text.strip(),
                model=self._model,
            )
        finally:
            safe_unlink(tmp_path)

    async def close(self) -> None:
        pass


async def handle_360dialog_audio_message(
    *,
    wa: Dialog360Client,
    transcriber: Transcriber,
    media_id: str,
    mime_type: str,
) -> str:
    """
    Download a 360dialog WhatsApp audio/voice message and transcribe it.

    The transcriber is injected — this function does not select the provider.
    WhatsApp voice notes are often audio/ogg with opus codec.
    Unsupported formats are converted to 16kHz mono WAV through ffmpeg.
    """
    suffix = suffix_from_mime(mime_type)
    raw_path = await wa.download_media_to_tempfile(
        media_id=media_id,
        suffix=suffix,
    )

    try:
        transcribable_path = ensure_transcribable_audio(raw_path)
        audio_bytes = transcribable_path.read_bytes()
        result = await transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=transcribable_path.name,
            content_type=mime_type,
        )
        logger.info(
            "Transcription complete: provider=%s model=%s language=%s "
            "audio_bytes=%d audio_duration=%.2fs processing=%.2fs text=%s",
            type(transcriber).__name__,
            result.model,
            result.language,
            len(audio_bytes),
            result.audio_duration_seconds or 0.0,
            result.processing_seconds or 0.0,
            result.text[:200],
        )
        return result.text
    finally:
        safe_unlink(raw_path)
        if "transcribable_path" in locals() and transcribable_path != raw_path:
            safe_unlink(transcribable_path)


async def handle_direct_audio_download_url(
    *,
    transcriber: Transcriber,
    download_url: str,
    file_name: str = "voice-message",
    mime_type: str = "",
) -> str:
    """
    Optional fallback for providers/payloads that already give a direct downloadUrl.
    360dialog usually gives media_id, so the main path is handle_360dialog_audio_message().
    """
    suffix = Path(file_name).suffix or suffix_from_mime(mime_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / f"audio{suffix}"
        await download_file(download_url, raw_path)
        transcribable_path = ensure_transcribable_audio(raw_path)
        audio_bytes = transcribable_path.read_bytes()
        result = await transcriber.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=transcribable_path.name,
            content_type=mime_type,
        )
        return result.text


async def download_file(url: str, target_path: Path) -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        target_path.write_bytes(response.content)


def ensure_transcribable_audio(path: Path) -> Path:
    if path.suffix.lower() in SUPPORTED_TRANSCRIPTION_EXTS:
        return path

    converted = path.with_suffix(".wav")

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(converted),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return converted


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
