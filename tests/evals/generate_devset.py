"""Data-driven generator for commitment extraction evaluation examples.

Loads example definitions from YAML files in tests/evals/data/, fills defaults
from _schema.yaml, splits into train/dev/test, and writes JSON output files.

Run: python tests/evals/generate_devset.py
"""

from __future__ import annotations

import json
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
]

_CHALLENGE_FILE = "challenge_act_ignore.yaml"

_ALL_CATEGORIES = [
    "act_vs_ignore",
    "args_party",
    "args_deadline",
    "args_required_action",
    "lifecycle_update_vs_new",
    "lifecycle_completion",
]

_SPLIT_RATIOS = {
    "easy": (0.70, 0.15, 0.15),
    "medium": (0.50, 0.25, 0.25),
    "hard": (0.35, 0.32, 0.33),
}


def _load_schema() -> dict[str, Any]:
    with (DATA_DIR / "_schema.yaml").open() as f:
        return yaml.safe_load(f)


def _fill_commitment(c: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Fill in chat_id, chat_name, status, notification defaults."""
    filled = {
        "id": c.get("id"),
        "chat_id": schema["chat_id"],
        "chat_name": schema["chat_name"],
        "committed_party": c.get("committed_party"),
        "required_action": c["required_action"],
        "deadline": c.get("deadline"),
        "context": c["context"],
        "status": c.get("status", schema["default_status"]),
        "notification": c.get("notification", schema["default_notification"]),
    }
    return filled


def _build_example(raw: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a YAML example entry to the JSON output format."""
    existing = raw.get("existing_commitments", [])
    existing_json = json.dumps(
        [_fill_commitment(c, schema) for c in existing],
        ensure_ascii=False,
    )

    expected = raw.get("expected_commitments", [])
    expected_filled = [_fill_commitment(c, schema) for c in expected]

    out: dict[str, Any] = {
        "category": raw["category"],
        "scenario": raw["scenario"],
        "difficulty": raw["difficulty"],
        "chat_id": schema["chat_id"],
        "chat_name": schema["chat_name"],
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


def _split_bucket(
    examples: list[dict[str, Any]], difficulty: str
) -> tuple[list, list, list]:
    """Split a single (category, difficulty) bucket by target ratios."""
    r_train, r_dev, _ = _SPLIT_RATIOS[difficulty]
    n = len(examples)
    train_end = max(1, int(n * r_train)) if n > 1 else 1
    dev_end = train_end + max(1, int(n * r_dev)) if n > 2 else train_end
    if n <= 1:
        return examples, [], []
    if n == 2:
        return examples[:1], examples[1:], []
    train = examples[:train_end]
    dev = examples[train_end:dev_end]
    test = examples[dev_end:]
    return train, dev, test


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
    all_examples = load_examples()

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ex in all_examples:
        key = (ex["category"], ex["difficulty"])
        buckets.setdefault(key, []).append(ex)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for (cat, diff), bucket_examples in sorted(buckets.items()):
        tr, dv, te = _split_bucket(bucket_examples, diff)
        train.extend(tr)
        dev.extend(dv)
        test.extend(te)

    _assert_split_has_all_categories(train, "train")
    _assert_split_has_all_categories(dev, "dev")
    _assert_split_has_all_categories(test, "test")
    _assert_has_hard_per_category(dev, "dev")
    _assert_has_hard_per_category(test, "test")

    challenge = load_challenge_examples()

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

    for cat_name in _ALL_CATEGORIES:
        parts = []
        for diff in ("easy", "medium", "hard"):
            bucket = buckets.get((cat_name, diff), [])
            tr = [e for e in bucket if e in train]
            dv = [e for e in bucket if e in dev]
            te = [e for e in bucket if e in test]
            parts.append(f"{diff}: {len(tr)}t/{len(dv)}d/{len(te)}t")
        print(f"  {cat_name}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
