"""Tests for _safe_suffix helper in modal_transcription/app.py."""

import sys
from pathlib import Path

# Add the modal_transcription directory to the path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import _safe_suffix


class TestSafeSuffix:
    def test_ogg_from_filename(self):
        assert _safe_suffix("audio/ogg", "voice-note.ogg") == ".ogg"

    def test_ogg_from_content_type_only(self):
        assert _safe_suffix("audio/ogg", None) == ".ogg"

    def test_ogg_with_codecs_param(self):
        assert _safe_suffix("audio/ogg; codecs=opus", None) == ".ogg"

    def test_opus_content_type(self):
        assert _safe_suffix("audio/opus", None) == ".opus"

    def test_application_ogg(self):
        assert _safe_suffix("application/ogg", None) == ".ogg"

    def test_mpeg_content_type(self):
        assert _safe_suffix("audio/mpeg", None) == ".mp3"

    def test_mp4_content_type(self):
        assert _safe_suffix("audio/mp4", None) == ".m4a"

    def test_x_m4a_content_type(self):
        assert _safe_suffix("audio/x-m4a", None) == ".m4a"

    def test_wav_content_type(self):
        assert _safe_suffix("audio/wav", None) == ".wav"

    def test_x_wav_content_type(self):
        assert _safe_suffix("audio/x-wav", None) == ".wav"

    def test_webm_content_type(self):
        assert _safe_suffix("audio/webm", None) == ".webm"

    def test_aac_content_type(self):
        assert _safe_suffix("audio/aac", None) == ".aac"

    def test_flac_content_type(self):
        assert _safe_suffix("audio/flac", None) == ".flac"

    def test_unknown_content_type_returns_bin(self):
        assert _safe_suffix("audio/unknown", None) == ".bin"

    def test_no_content_type_no_filename_returns_bin(self):
        assert _safe_suffix(None, None) == ".bin"

    def test_filename_overrides_unknown_content_type(self):
        assert _safe_suffix("audio/unknown", "file.mp3") == ".mp3"

    def test_filename_with_unrecognized_suffix_falls_back_to_content_type(self):
        assert _safe_suffix("audio/ogg", "file.xyz") == ".ogg"

    def test_mixed_case_filename_suffix(self):
        assert _safe_suffix(None, "voice.OGG") == ".ogg"

    def test_mixed_case_content_type(self):
        assert _safe_suffix("Audio/OGG", None) == ".ogg"

    def test_content_type_with_extra_whitespace(self):
        assert _safe_suffix("  audio/ogg  ", None) == ".ogg"

    def test_opus_filename_suffix(self):
        assert _safe_suffix(None, "voice.opus") == ".opus"

    def test_m4a_filename_suffix(self):
        assert _safe_suffix(None, "voice.m4a") == ".m4a"

    def test_wav_filename_suffix(self):
        assert _safe_suffix(None, "voice.wav") == ".wav"

    def test_webm_filename_suffix(self):
        assert _safe_suffix(None, "voice.webm") == ".webm"

    def test_aac_filename_suffix(self):
        assert _safe_suffix(None, "voice.aac") == ".aac"

    def test_flac_filename_suffix(self):
        assert _safe_suffix(None, "voice.flac") == ".flac"

    def test_oga_filename_suffix(self):
        assert _safe_suffix(None, "voice.oga") == ".oga"
