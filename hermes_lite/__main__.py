"""Hermes-Lite — CLI entry point.

Run with::

    python -m hermes_lite
"""

from hermes_lite.orchestrator import HermesOrchestrator


if __name__ == "__main__":
    orchestrator = HermesOrchestrator()
    orchestrator.start()