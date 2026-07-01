"""Hermes-Lite — CLI entry point.

Run with::

    hermes-lite              # Interactive shell
    hermes-lite stats        # Per-turn observability table
    hermes-lite --version    # Print version and exit
    hermes-lite --auth-token <token>  # Set auth token for dangerous tools
    python -m hermes_lite    # Same as hermes-lite
"""

import sys
import os


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

    # --auth-token flag
    auth_token = None
    if "--auth-token" in sys.argv:
        try:
            idx = sys.argv.index("--auth-token")
            if idx + 1 < len(sys.argv):
                auth_token = sys.argv[idx + 1]
                # Remove the flag and its value from sys.argv
                sys.argv.pop(idx)  # --auth-token
                sys.argv.pop(idx)  # token value
            else:
                print("Error: --auth-token requires a value", file=sys.stderr)
                sys.exit(1)
        except ValueError:
            pass

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        from hermes_lite.observability import print_stats
        print_stats()
        return

    # Default: launch the orchestrator shell
    from hermes_lite.orchestrator import HermesOrchestrator

    orchestrator = HermesOrchestrator(auth_token=auth_token)
    orchestrator.start()


if __name__ == "__main__":
    _main()
