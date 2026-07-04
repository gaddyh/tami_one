import subprocess
import tempfile
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

from app.config import Settings, settings
from app.services.whatsapp import Dialog360Client

_openai = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

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


async def handle_360dialog_audio_message(
    wa: Dialog360Client,
    settings: Settings,
    media_id: str,
    mime_type: str,
) -> str:
    """
    Download a 360dialog WhatsApp audio/voice message and transcribe it.

    WhatsApp voice notes are often audio/ogg with opus codec.
    Unsupported formats are converted to 16kHz mono WAV through ffmpeg.
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    suffix = suffix_from_mime(mime_type)
    raw_path = await wa.download_media_to_tempfile(
        media_id=media_id,
        suffix=suffix,
    )

    try:
        transcribable_path = ensure_transcribable_audio(raw_path)
        return await transcribe_audio(settings, transcribable_path)
    finally:
        safe_unlink(raw_path)
        if "transcribable_path" in locals() and transcribable_path != raw_path:
            safe_unlink(transcribable_path)


async def handle_direct_audio_download_url(
    settings: Settings,
    download_url: str,
    file_name: str = "voice-message",
    mime_type: str = "",
) -> str:
    """
    Optional fallback for providers/payloads that already give a direct downloadUrl.
    360dialog usually gives media_id, so the main path is handle_360dialog_audio_message().
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    suffix = Path(file_name).suffix or suffix_from_mime(mime_type)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / f"audio{suffix}"
        await download_file(download_url, raw_path)
        transcribable_path = ensure_transcribable_audio(raw_path)
        return await transcribe_audio(settings, transcribable_path)


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


async def transcribe_audio(settings: Settings, path: Path) -> str:
    with path.open("rb") as audio_file:
        transcription = await _openai.audio.transcriptions.create(
            model=settings.openai_transcribe_model,
            file=audio_file,
        )

    return transcription.text.strip()


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
