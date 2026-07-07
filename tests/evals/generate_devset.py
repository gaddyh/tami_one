"""Data-driven generator for commitment extraction evaluation examples.

Loads example definitions from YAML files in tests/evals/data/, fills defaults
from _schema.yaml, splits into train/dev/test, and writes JSON output files.

Run: python tests/evals/generate_devset.py
Run validators only: python tests/evals/generate_devset.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent

_CATEGORY_FILES = [
    "act_vs_ignore.yaml",
    "args_party.yaml",
    "args_deadline.yaml",
    "args_required_action.yaml",
    "lifecycle_update_vs_new.yaml",
    "lifecycle_completion.yaml",
    "cardinality.yaml",
]

_CHALLENGE_FILE = "challenge_act_ignore.yaml"

_ALL_CATEGORIES = [
    "act_vs_ignore",
    "args_party",
    "args_deadline",
    "args_required_action",
    "lifecycle_update_vs_new",
    "lifecycle_completion",
    "cardinality",
]

_SPLIT_RATIOS = {
    "easy": (0.70, 0.15, 0.15),
    "medium": (0.50, 0.25, 0.25),
    "hard": (0.35, 0.32, 0.33),
}

_SELF_CHAT_ID = os.getenv("SELF_CHAT_ID", "972546610653@c.us")
_SENTINEL_CHAT_ID = "972500000000@c.us"
_SENTINEL_CHAT_NAME = "DEFAULT CONTACT — SHOULD NOT APPEAR"

_ROLE_WORDS = [
    "supplier",
    "bank",
    "contractor",
    "landlord",
    "lawyer",
    "notary",
    "accountant",
]

_LAYER_MAP = {
    "act_vs_ignore": 1,
    "args_party": 2,
    "args_deadline": 2,
    "args_required_action": 2,
    "lifecycle_update_vs_new": 3,
    "lifecycle_completion": 3,
    "cardinality": 4,
}

_LAYER_NAMES = {
    1: "Act vs Ignore",
    2: "Args",
    3: "Lifecycle",
    4: "Cardinality",
}


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> set[str]:
    """Tokenize normalized text into a set of tokens."""
    return set(_normalize(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _example_id(raw: dict[str, Any]) -> str:
    """Stable example ID for hashing: category/scenario."""
    return f"{raw['category']}/{raw['scenario']}"


def _get_context_fragments(raw: dict[str, Any]) -> list[str]:
    """Extract context fragments from raw YAML entry.

    Handles both context_fragments (list) and legacy context (string).
    """
    if "context_fragments" in raw:
        return raw["context_fragments"]
    if "context" in raw:
        return [raw["context"]]
    return []


def _get_context_fragments_from_commitment(c: dict[str, Any]) -> list[str]:
    """Extract context fragments from a commitment dict (raw YAML)."""
    if "context_fragments" in c:
        return c["context_fragments"]
    if "context" in c:
        return [c["context"]]
    return []


def load_raw_examples() -> list[dict[str, Any]]:
    """Load all raw YAML entries from category files (no schema filling)."""
    examples: list[dict[str, Any]] = []
    for filename in _CATEGORY_FILES:
        with (DATA_DIR / filename).open() as f:
            raw_list = yaml.safe_load(f)
        for raw in raw_list:
            raw["_source_file"] = filename
            examples.append(raw)
    return examples


def load_raw_challenge_examples() -> list[dict[str, Any]]:
    """Load raw challenge YAML entries."""
    with (DATA_DIR / _CHALLENGE_FILE).open() as f:
        raw_list = yaml.safe_load(f)
    for raw in raw_list:
        raw["_source_file"] = _CHALLENGE_FILE
    return raw_list


# ── Build-failing validators ──────────────────────────────────────────


def _validate_no_self_chat(examples: list[dict[str, Any]]) -> list[str]:
    """No example uses the user's own chat_id."""
    errors: list[str] = []
    for ex in examples:
        chat_id = ex.get("chat_id", "")
        if chat_id == _SELF_CHAT_ID:
            errors.append(
                f"{_example_id(ex)}: chat_id is self-chat ({_SELF_CHAT_ID})"
            )
    return errors


def _validate_schema(examples: list[dict[str, Any]]) -> list[str]:
    """Required fields present, no sentinel values."""
    errors: list[str] = []
    for ex in examples:
        eid = _example_id(ex)
        for field in ("category", "scenario", "difficulty", "chat_id", "chat_name", "messages"):
            if field not in ex:
                errors.append(f"{eid}: missing required field '{field}'")
        chat_id = ex.get("chat_id", "")
        chat_name = ex.get("chat_name", "")
        if chat_id == _SENTINEL_CHAT_ID:
            errors.append(f"{eid}: sentinel chat_id (missing per-example chat_id)")
        if chat_name == _SENTINEL_CHAT_NAME:
            errors.append(f"{eid}: sentinel chat_name (missing per-example chat_name)")
    return errors


def _validate_context_is_quote(examples: list[dict[str, Any]]) -> list[str]:
    """Each context fragment must be a substring of some message after normalization.

    Skips commitments that are unchanged from existing (same id + same status)
    since their context is inherited from the original conversation.
    Also emits migration warnings for legacy 'context' field usage.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for ex in examples:
        eid = _example_id(ex)
        messages = ex.get("messages", "")
        normalized_messages = _normalize(messages)
        existing = ex.get("existing_commitments", [])
        existing_by_id = {c.get("id"): c for c in existing if c.get("id")}
        commitments = ex.get("expected_commitments", [])
        for i, c in enumerate(commitments):
            # Skip context validation for unchanged commitments (inherited context)
            cid = c.get("id")
            if cid and cid in existing_by_id:
                orig = existing_by_id[cid]
                if c.get("status", "waiting") == orig.get("status", "waiting"):
                    continue
            fragments = _get_context_fragments_from_commitment(c)
            if not fragments:
                errors.append(f"{eid}: commitment {i} has no context")
                continue
            for j, frag in enumerate(fragments):
                norm_frag = _normalize(frag)
                if not norm_frag:
                    errors.append(f"{eid}: commitment {i} fragment {j} is empty")
                    continue
                if norm_frag not in normalized_messages:
                    errors.append(
                        f"{eid}: commitment {i} fragment {j} not found in messages: "
                        f"'{frag[:60]}'"
                    )
            if "context" in c and "context_fragments" not in c:
                warnings.append(
                    f"{eid}: commitment {i} uses legacy 'context' field — migrate to 'context_fragments'"
                )
    for w in warnings:
        print(f"  MIGRATION WARNING: {w}", file=sys.stderr)
    return errors


def _validate_chat_name_role(examples: list[dict[str, Any]]) -> list[str]:
    """Role-word denylist check for 1-on-1 chats.

    For @c.us chats, fail if a message mentions a role word in third person
    AND chat_name contains that role word.
    """
    errors: list[str] = []
    for ex in examples:
        chat_id = ex.get("chat_id", "")
        if not chat_id.endswith("@c.us"):
            continue
        chat_name = ex.get("chat_name", "").lower()
        messages = ex.get("messages", "").lower()
        for role in _ROLE_WORDS:
            if role not in chat_name:
                continue
            if role not in messages:
                continue
            errors.append(
                f"{_example_id(ex)}: 1-on-1 chat_name '{ex['chat_name']}' contains role word "
                f"'{role}' which is also mentioned in messages in third person"
            )
    return errors


def _outputs_near_identical(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two examples have near-identical expected outputs."""
    ea = a.get("expected_commitments", [])
    eb = b.get("expected_commitments", [])
    if len(ea) != len(eb):
        return False
    for ca, cb in zip(ea, eb):
        if ca.get("required_action") != cb.get("required_action"):
            return False
        if ca.get("committed_party") != cb.get("committed_party"):
            return False
        if ca.get("deadline") != cb.get("deadline"):
            return False
    return True


def _validate_no_near_duplicates(
    examples: list[dict[str, Any]],
    split_map: dict[str, str] | None = None,
) -> list[str]:
    """Check no near-duplicate messages across different splits.

    Jaccard > 0.80 on tokens (min 6 tokens per message), AND expected outputs near-identical.
    Honors allow_similar_to declarations.
    """
    errors: list[str] = []
    if split_map is None:
        split_map = {}

    allow_map: dict[str, set[str]] = {}
    for ex in examples:
        eid = _example_id(ex)
        allow = ex.get("allow_similar_to")
        if allow:
            if isinstance(allow, str):
                allow = [allow]
            allow_map.setdefault(eid, set()).update(allow)

    tokenized = []
    for ex in examples:
        tokens = _tokenize(ex.get("messages", ""))
        tokenized.append((_example_id(ex), tokens, ex))

    for i in range(len(tokenized)):
        eid_a, tokens_a, ex_a = tokenized[i]
        split_a = split_map.get(eid_a, "?")
        for j in range(i + 1, len(tokenized)):
            eid_b, tokens_b, ex_b = tokenized[j]
            split_b = split_map.get(eid_b, "?")
            if split_a == split_b:
                continue
            if len(tokens_a) < 6 or len(tokens_b) < 6:
                continue
            if eid_b in allow_map.get(eid_a, set()) or eid_a in allow_map.get(eid_b, set()):
                continue
            sim = _jaccard(tokens_a, tokens_b)
            if sim > 0.80 and _outputs_near_identical(ex_a, ex_b):
                errors.append(
                    f"Near-duplicate across splits: {eid_a} ({split_a}) <-> {eid_b} ({split_b})\n"
                    f"  A: '{_normalize(ex_a.get('messages', ''))[:80]}'\n"
                    f"  B: '{_normalize(ex_b.get('messages', ''))[:80]}'\n"
                    f"  Jaccard: {sim:.2f}, expected outputs identical: yes"
                )
    return errors


# ── Warning validators ────────────────────────────────────────────────


def _warn_action_diversity(examples: list[dict[str, Any]]) -> None:
    """Print action frequency distribution."""
    from collections import Counter

    actions: list[str] = []
    for ex in examples:
        for c in ex.get("expected_commitments", []):
            actions.append(c.get("required_action", "?"))
    if not actions:
        return
    counts = Counter(actions)
    total = len(actions)
    print(f"  Action diversity ({total} total):", file=sys.stderr)
    for action, count in counts.most_common(5):
        pct = count / total * 100
        print(f"    {action}: {count} ({pct:.0f}%)", file=sys.stderr)


def _warn_monologue(examples: list[dict[str, Any]]) -> None:
    """Print count of 1-on-1 examples with single speaker."""
    monologue_count = 0
    one_on_one_count = 0
    for ex in examples:
        chat_id = ex.get("chat_id", "")
        if not chat_id.endswith("@c.us"):
            continue
        one_on_one_count += 1
        messages = ex.get("messages", "")
        speakers = set()
        for line in messages.split("\n"):
            if ":" in line:
                speaker = line.split(":")[0].strip()
                if speaker:
                    speakers.add(speaker)
        if len(speakers) <= 1:
            monologue_count += 1
    if one_on_one_count:
        print(
            f"  Monologue: {monologue_count}/{one_on_one_count} 1-on-1 chats have single speaker",
            file=sys.stderr,
        )


def _warn_layer_coverage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Print layer distribution per split."""
    for layer_num in sorted(_LAYER_NAMES):
        parts = []
        for split_name in ("train", "dev", "test", "challenge"):
            examples = splits.get(split_name, [])
            count = sum(
                1 for ex in examples if _LAYER_MAP.get(ex.get("category", "")) == layer_num
            )
            if count:
                parts.append(f"{count} {split_name}")
        if parts:
            print(f"  Layer {layer_num} ({_LAYER_NAMES[layer_num]}): {' / '.join(parts)}", file=sys.stderr)


def run_validators(
    raw_examples: list[dict[str, Any]],
    raw_challenge: list[dict[str, Any]],
    split_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Run all validators. Returns (errors, warnings)."""
    all_raw = raw_examples + raw_challenge
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_validate_no_self_chat(all_raw))
    errors.extend(_validate_schema(all_raw))
    errors.extend(_validate_context_is_quote(all_raw))
    errors.extend(_validate_chat_name_role(all_raw))
    errors.extend(_validate_no_near_duplicates(all_raw, split_map))

    print("--- Warnings ---", file=sys.stderr)
    _warn_action_diversity(all_raw)
    _warn_monologue(all_raw)

    return errors, warnings


def _load_schema() -> dict[str, Any]:
    with (DATA_DIR / "_schema.yaml").open() as f:
        return yaml.safe_load(f)


def _fill_commitment(
    c: dict[str, Any],
    schema: dict[str, Any],
    chat_id: str,
    chat_name: str,
) -> dict[str, Any]:
    """Fill in chat_id, chat_name, status, notification defaults."""
    fragments = c.get("context_fragments", [])
    context = ", ".join(fragments) if fragments else c.get("context", "")
    filled = {
        "id": c.get("id"),
        "chat_id": chat_id,
        "chat_name": chat_name,
        "committed_party": c.get("committed_party"),
        "required_action": c["required_action"],
        "deadline": c.get("deadline"),
        "context": context,
        "status": c.get("status", schema["default_status"]),
        "notification": c.get("notification", schema["default_notification"]),
    }
    return filled


def _build_example(raw: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a YAML example entry to the JSON output format."""
    chat_id = raw.get("chat_id", schema["chat_id"])
    chat_name = raw.get("chat_name", schema["chat_name"])

    existing = raw.get("existing_commitments", [])
    existing_json = json.dumps(
        [_fill_commitment(c, schema, chat_id, chat_name) for c in existing],
        ensure_ascii=False,
    )

    expected = raw.get("expected_commitments", [])
    expected_filled = [_fill_commitment(c, schema, chat_id, chat_name) for c in expected]

    out: dict[str, Any] = {
        "category": raw["category"],
        "scenario": raw["scenario"],
        "difficulty": raw["difficulty"],
        "chat_id": chat_id,
        "chat_name": chat_name,
        "current_datetime": schema.get("current_datetime", "2025-01-06T10:00:00Z"),
        "existing_commitments_json": existing_json,
        "messages": raw["messages"],
        "expected_commitments": expected_filled,
    }

    if raw.get("policy_note"):
        out["policy_note"] = raw["policy_note"]

    return out


def load_examples() -> list[dict[str, Any]]:
    """Load all category examples from YAML files."""
    schema = _load_schema()
    examples: list[dict[str, Any]] = []
    for filename in _CATEGORY_FILES:
        with (DATA_DIR / filename).open() as f:
            raw_list = yaml.safe_load(f)
        for raw in raw_list:
            examples.append(_build_example(raw, schema))
    return examples


def load_challenge_examples() -> list[dict[str, Any]]:
    """Load challenge split examples from YAML."""
    schema = _load_schema()
    with (DATA_DIR / _CHALLENGE_FILE).open() as f:
        raw_list = yaml.safe_load(f)
    return [_build_example(raw, schema) for raw in raw_list]


def _assign_split(example_id: str, layer: int) -> str:
    """Deterministic split assignment based on example ID hash."""
    h = hashlib.sha256(example_id.encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    if layer == 1:
        return "train" if bucket < 60 else ("dev" if bucket < 80 else "test")
    if layer == 2:
        return "train" if bucket < 50 else ("dev" if bucket < 75 else "test")
    if layer == 3:
        return "train" if bucket < 35 else ("dev" if bucket < 65 else "test")
    if layer == 4:
        return "train" if bucket < 25 else ("dev" if bucket < 50 else "test")
    return "test"


def _load_raw_split_map(raw_examples: list[dict[str, Any]]) -> dict[str, str]:
    """Build a map of example_id -> split from raw YAML entries.

    Uses explicit `split:` pin if present, otherwise hash-based assignment.
    """
    split_map: dict[str, str] = {}
    for raw in raw_examples:
        eid = _example_id(raw)
        if raw.get("split"):
            split_map[eid] = raw["split"]
        else:
            layer = _LAYER_MAP.get(raw.get("category", ""), 3)
            split_map[eid] = _assign_split(eid, layer)
    return split_map


def _assert_has_hard_per_category(
    examples: list[dict[str, Any]], split_name: str
) -> None:
    hard_cats = {ex["category"] for ex in examples if ex.get("difficulty") == "hard"}
    missing = set(_ALL_CATEGORIES) - hard_cats
    if missing:
        raise AssertionError(
            f"{split_name} missing hard examples for categories: {sorted(missing)}"
        )


def _assert_split_has_all_categories(
    examples: list[dict[str, Any]], split_name: str
) -> None:
    cats = {ex["category"] for ex in examples}
    missing = set(_ALL_CATEGORIES) - cats
    if missing:
        raise AssertionError(
            f"{split_name} missing categories: {sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval dataset JSONs from YAML sources.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validators against YAML sources without generating JSONs.",
    )
    args = parser.parse_args()

    raw_examples = load_raw_examples()
    raw_challenge = load_raw_challenge_examples()

    # Build split map from raw YAML (explicit pins or hash-based)
    split_map = _load_raw_split_map(raw_examples)

    if args.validate_only:
        errors, _warnings = run_validators(raw_examples, raw_challenge, split_map)
        if errors:
            print(f"\n{len(errors)} VALIDATION ERRORS:", file=sys.stderr)
            for err in errors:
                print(f"  ERROR: {err}", file=sys.stderr)
            sys.exit(1)
        print("All validators passed.")
        return

    all_examples = load_examples()

    # Assign splits using hash-based or explicit pin
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for ex in all_examples:
        eid = f"{ex['category']}/{ex['scenario']}"
        split = split_map.get(eid, "test")
        if split == "train":
            train.append(ex)
        elif split == "dev":
            dev.append(ex)
        elif split == "test":
            test.append(ex)

    _assert_split_has_all_categories(train, "train")
    _assert_split_has_all_categories(dev, "dev")
    _assert_split_has_all_categories(test, "test")
    _assert_has_hard_per_category(dev, "dev")
    _assert_has_hard_per_category(test, "test")

    for ex in all_examples:
        assert ex["chat_id"] != _SELF_CHAT_ID, (
            f"Example {ex['category']}/{ex['scenario']} still has old self-chat chat_id"
        )
        assert ex["chat_id"] != _SENTINEL_CHAT_ID, (
            f"Example {ex['category']}/{ex['scenario']} still has sentinel chat_id "
            "(missing per-example chat_id in YAML)"
        )

    challenge = load_challenge_examples()

    for ex in challenge:
        assert ex["chat_id"] != _SELF_CHAT_ID, (
            f"Challenge example {ex['category']}/{ex['scenario']} still has old self-chat chat_id"
        )
        assert ex["chat_id"] != _SENTINEL_CHAT_ID, (
            f"Challenge example {ex['category']}/{ex['scenario']} still has sentinel chat_id "
            "(missing per-example chat_id in YAML)"
        )

    # Run validators with split map for near-duplicate check
    errors, _warnings = run_validators(raw_examples, raw_challenge, split_map)
    if errors:
        print(f"\n{len(errors)} VALIDATION ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    for name, data in [
        ("trainset.json", train),
        ("devset.json", dev),
        ("testset.json", test),
        ("challenge_act_ignore.json", challenge),
    ]:
        path = OUTPUT_DIR / name
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    challenge_ignore = sum(1 for e in challenge if not e["expected_commitments"])
    challenge_act = len(challenge) - challenge_ignore
    print(f"Total: {len(all_examples)} examples")
    print(f"  train: {len(train)}")
    print(f"  dev:   {len(dev)}")
    print(f"  test:  {len(test)}")
    print(f"  challenge: {len(challenge)} ({challenge_act} act / {challenge_ignore} ignore)")
    print()

    # Layer coverage report
    splits = {"train": train, "dev": dev, "test": test, "challenge": challenge}
    print("Layer coverage:")
    _warn_layer_coverage(splits)
    print()

    for cat_name in _ALL_CATEGORIES:
        tr = [e for e in train if e["category"] == cat_name]
        dv = [e for e in dev if e["category"] == cat_name]
        te = [e for e in test if e["category"] == cat_name]
        print(f"  {cat_name}: {len(tr)}t/{len(dv)}d/{len(te)}t")


if __name__ == "__main__":
    main()
