"""Hermes-Lite — CLI entry point.

Run with::

    hermes-lite              # Interactive shell
    hermes-lite stats        # Per-turn observability table
    hermes-lite --version    # Print version and exit
    python -m hermes_lite    # Same as hermes-lite
"""

import sys


def _main() -> None:
    # --version flag
    if "--version" in sys.argv or "-V" in sys.argv:
        try:
            from importlib.metadata import version as _pkg_version
            v = _pkg_version("hermes-lite")
        except Exception:
            v = "0.5.0"
        print(f"hermes-lite {v}")
        return

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
