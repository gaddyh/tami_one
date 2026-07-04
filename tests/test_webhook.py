import base64
import json
from pathlib import Path

from app.services.whatsapp import (
    expected_basic_auth_header,
    iter_incoming_messages,
    normalize_phone,
    to_360dialog_media_url,
)

PAYLOAD = json.loads((Path(__file__).parent / "test_payload.json").read_text())


def test_iter_incoming_messages_parses_text():
    messages = list(iter_incoming_messages(PAYLOAD))
    assert len(messages) == 1
    msg = messages[0]
    assert msg["type"] == "text"
    assert msg["from"] == "972509999999"
    assert msg["text"] == "hello"
    assert msg["name"] == "Test User"
    assert msg["id"] == "wamid.test"


def test_iter_incoming_messages_empty_payload():
    assert list(iter_incoming_messages({})) == []


def test_iter_incoming_messages_no_messages():
    payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
    assert list(iter_incoming_messages(payload)) == []


def test_normalize_phone_strips_plus():
    assert normalize_phone("+972509999999") == "972509999999"


def test_normalize_phone_strips_suffix():
    assert normalize_phone("972509999999@c.us") == "972509999999"


def test_normalize_phone_strips_formatting():
    assert normalize_phone("+972 50-9999999") == "972509999999"


def test_expected_basic_auth_header():
    header = expected_basic_auth_header("user", "pass")
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header[len("Basic "):]).decode()
    assert decoded == "user:pass"


def test_to_360dialog_media_url_replaces_host():
    base = "https://waba-v2.360dialog.io"
    original = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=123&ext=456"
    result = to_360dialog_media_url(base, original)
    assert result.startswith(base)
    assert "mid=123" in result
    assert "lookaside.fbsbx.com" not in result
