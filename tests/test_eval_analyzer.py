"""Tests for eval_analyzer proposal validators and helpers."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ─── Validator tests (pure functions, no dspy import needed) ────────────────


def _validate_anchored_replace(proposal: dict, file_content: str) -> tuple[bool, str]:
    old_text = proposal.get("old_text", "")
    if not old_text:
        return False, "old_text is empty"
    if old_text not in file_content:
        return False, f"old_text not found in file: {old_text[:80]}..."
    return True, ""


def _validate_insert_after_heading(proposal: dict, file_content: str) -> tuple[bool, str]:
    heading = proposal.get("anchor_heading", "")
    if not heading:
        return False, "anchor_heading is empty"
    if heading not in file_content:
        return False, f"anchor_heading not found in file: {heading!r}"
    new_text = proposal.get("new_text", "")
    if not new_text:
        return False, "new_text is empty"
    return True, ""


def _validate_function_replace(proposal: dict) -> tuple[bool, str]:
    body = proposal.get("new_function_body", "")
    if not body:
        return False, "new_function_body is empty"
    try:
        ast.parse(body)
    except SyntaxError as e:
        return False, f"ast.parse failed: {e}"
    func_name = proposal.get("target_function", "")
    if not func_name:
        return False, "target_function is empty"
    src = (ROOT / "eval" / "metrics.py").read_text()
    tree = ast.parse(src)
    found = any(isinstance(n, ast.FunctionDef) and n.name == func_name for n in ast.walk(tree))
    if not found:
        return False, f"function {func_name!r} not found in eval/metrics.py"
    return True, ""


class TestAnchoredReplace:
    def test_valid_text(self):
        content = "    Waiting commitments:\n    - some rule"
        ok, reason = _validate_anchored_replace(
            {"old_text": "    Waiting commitments:"}, content
        )
        assert ok
        assert reason == ""

    def test_invalid_text(self):
        content = "some other content"
        ok, reason = _validate_anchored_replace(
            {"old_text": "nonexistent"}, content
        )
        assert not ok
        assert "not found" in reason

    def test_empty_old_text(self):
        ok, reason = _validate_anchored_replace({"old_text": ""}, "content")
        assert not ok
        assert "empty" in reason

    def test_real_prompt_text(self):
        prompt = (ROOT / "app" / "commitments" / "commitments_agent.py").read_text()
        ok, reason = _validate_anchored_replace(
            {"old_text": "    Act vs Ignore rules"}, prompt
        )
        assert ok


class TestInsertAfterHeading:
    def test_valid_heading(self):
        content = "    Act vs Ignore rules (critical — do NOT over-extract):\n    - rule"
        ok, reason = _validate_insert_after_heading(
            {"anchor_heading": "Act vs Ignore rules (critical — do NOT over-extract):",
             "new_text": "    - new rule"},
            content,
        )
        assert ok

    def test_invalid_heading(self):
        ok, reason = _validate_insert_after_heading(
            {"anchor_heading": "Nonexistent:", "new_text": "    - rule"},
            "some content",
        )
        assert not ok
        assert "not found" in reason

    def test_empty_new_text(self):
        ok, reason = _validate_insert_after_heading(
            {"anchor_heading": "Rules:", "new_text": ""},
            "Rules:\n- something",
        )
        assert not ok
        assert "empty" in reason

    def test_real_prompt_heading(self):
        prompt = (ROOT / "app" / "commitments" / "commitments_agent.py").read_text()
        ok, reason = _validate_insert_after_heading(
            {"anchor_heading": "Dismiss + new:", "new_text": "    - new rule"},
            prompt,
        )
        assert ok


class TestFunctionReplace:
    def test_valid_function(self):
        ok, reason = _validate_function_replace({
            "target_function": "_normalize_deadline",
            "new_function_body": "def _normalize_deadline(value):\n    return value",
        })
        assert ok

    def test_invalid_syntax(self):
        ok, reason = _validate_function_replace({
            "target_function": "_normalize_deadline",
            "new_function_body": "def _normalize_deadline(",
        })
        assert not ok
        assert "ast.parse" in reason

    def test_nonexistent_function(self):
        ok, reason = _validate_function_replace({
            "target_function": "nonexistent_func",
            "new_function_body": "def nonexistent_func():\n    pass",
        })
        assert not ok
        assert "not found" in reason

    def test_empty_body(self):
        ok, reason = _validate_function_replace({
            "target_function": "_normalize_deadline",
            "new_function_body": "",
        })
        assert not ok
        assert "empty" in reason


class TestInsertAfterHeadingApplication:
    """Test the actual insert logic without dspy imports."""

    def test_insert_after_last_bullet(self):
        content = """    Rules:
    - first rule
    - second rule

    Other section:
    - something else
"""
        heading = "    Rules:"
        new_text = "    - third rule"

        lines = content.split("\n")
        heading_idx = None
        for i, line in enumerate(lines):
            if heading in line:
                heading_idx = i
                break

        last_bullet_idx = heading_idx
        for i in range(heading_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if stripped == "":
                break
            if stripped.startswith("- "):
                last_bullet_idx = i

        new_lines = lines[:last_bullet_idx + 1] + new_text.split("\n") + lines[last_bullet_idx + 1:]
        result = "\n".join(new_lines)

        assert "- third rule" in result
        assert result.index("- third rule") > result.index("- second rule")
        assert result.index("- third rule") < result.index("Other section:")
