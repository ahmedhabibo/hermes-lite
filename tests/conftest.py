"""Pytest configuration for Hermes-Lite.

Prevents the .env loader from polluting test environment by setting
HERMES_LITE_NO_DOTENV=1 before any test module imports hermes_lite.
"""

import os

os.environ.setdefault("HERMES_LITE_NO_DOTENV", "1")
