"""
main.py

Entrypoint. Run with: python3 code/main.py, from the repository root,
so relative paths (dataset/, log.txt, evaluation/) resolve correctly.
"""

import os
import sys

from orchestrator import Orchestrator


def main() -> int:
    """
    Input: none
    Output: int, process exit code. 0 on success, 1 on a fatal error
        that prevented output.csv from being written at all.

    Per-request failures are already handled inside Orchestrator
    (retry, then a conservative fallback row), so any exception that
    reaches here is a whole-run problem: a missing API key, a missing
    dataset/ directory, a disk write failure, and so on.
    """
    if "ANTH_API_KEY" not in os.environ:
        print(
            "Fatal error: ANTH_API_KEY is not set. Set it in your .env "
            "or environment before running this pipeline.",
            file=sys.stderr,
        )
        return 1

    try:
        orchestrator = Orchestrator()
        orchestrator.run()
    except Exception as exc:  # noqa: BLE001 - top-level catch-all, see docstring
        print(f"Fatal error: pipeline failed before producing output.csv: {exc}", file=sys.stderr)
        return 1

    print("Done: output.csv and evaluation/usage_report.md written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
