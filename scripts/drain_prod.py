"""Run drain_and_process against the prod DB to test extraction end-to-end.

Usage:
  DATABASE_URL="postgresql://..." .venv/bin/python scripts/drain_prod.py
"""

import asyncio
import os
import sys

from app.commitments.processor import drain_and_process


def main():
    if not os.getenv("DATABASE_URL"):
        print("ERROR: Set DATABASE_URL env var first.")
        sys.exit(1)

    print("Running drain_and_process against prod DB...")
    results = asyncio.run(drain_and_process())

    total = sum(len(v) for v in results.values())
    print(f"\nDone: {total} commitment(s) across {len(results)} chat(s)")

    for key, items in results.items():
        tenant_id, chat_id = key
        print(f"\n  Chat: {chat_id}")
        for item in items:
            print(f"    [{item.status}] {item.committed_party} → {item.required_action}")
            print(f"      deadline={item.deadline}  context={item.context[:60] if item.context else ''}")
            print(f"      conversation_id={item.conversation_id[:8] if item.conversation_id else 'None'}...")


if __name__ == "__main__":
    main()
