"""Hermes-Lite — CLI entry point.

Run with::

    python -m hermes_lite          # Interactive shell
    python -m hermes_lite stats    # Per-turn observability table
"""

import sys


def _main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        from hermes_lite.observability import print_stats
        print_stats()
        return

    # Default: launch the orchestrator shell
    from hermes_lite.orchestrator import HermesOrchestrator

    orchestrator = HermesOrchestrator()
    orchestrator.start()


if __name__ == "__main__":
    _main()
