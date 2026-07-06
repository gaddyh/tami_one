"""CLI entry point for running a DSPy evaluation smoke test.

Usage:
    # Full devset (10 examples)
    .venv/bin/python scripts/smoke_eval.py

    # First 3 examples only
    .venv/bin/python scripts/smoke_eval.py --limit 3

    # Use a specific model
    .venv/bin/python scripts/smoke_eval.py --model gpt-4o-mini
"""

import argparse
import sys

from app.config import settings
from app.commitments.commitments_agent import configure_dspy
from app.commitments.eval import build_devset, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy commitment extraction eval")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N examples (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the OpenAI model (default: from settings)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print actual vs expected for each example",
    )
    args = parser.parse_args()

    if args.model:
        settings.openai_model = args.model

    configure_dspy(settings)

    devset = build_devset()
    if args.limit:
        devset = devset[: args.limit]
        print(f"Running smoke eval on {len(devset)} example(s)...\n")
    else:
        print(f"Running eval on full devset ({len(devset)} examples)...\n")

    if args.verbose:
        import json as _json

        from app.commitments.commitments_agent import CommitmentAgent

        agent = CommitmentAgent()
        for i, ex in enumerate(devset):
            pred = agent(**ex.inputs())
            actual = [c.model_dump(mode="json") for c in pred.commitments]
            expected = [c.model_dump(mode="json") for c in ex.expected_commitments]
            print(f"\n--- Example {i} ---")
            print(f"Messages: {ex.messages[:80]}...")
            print(f"Expected: {_json.dumps(expected, ensure_ascii=False, indent=2)}")
            print(f"Actual:   {_json.dumps(actual, ensure_ascii=False, indent=2)}")
        print()

    result = run_evaluation(devset=devset)
    score = result.score if hasattr(result, "score") else result
    print(f"\n{'='*50}")
    print(f"Score: {score}")
    print(f"{'='*50}")

    if score < 1.0:
        print(f"\n{score*100:.0f}% of examples matched expected output.")
        print("Review the table above for mismatches.")
        sys.exit(0)


if __name__ == "__main__":
    main()
