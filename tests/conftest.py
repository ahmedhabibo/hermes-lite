"""Pytest configuration for Hermes-Lite.

Prevents the .env loader from polluting test environment by setting
HERMES_LITE_NO_DOTENV=1 before any test module imports hermes_lite.

Also resets the cached config singleton between tests so tests that
``monkeypatch.setenv()`` see a fresh build every time (matches the
runtime ``reload_config()`` pattern documented in ``hermes_lite.config``).
"""

import os

os.environ.setdefault("HERMES_LITE_NO_DOTENV", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Force a fresh ``get_config()`` build per test.

    Tests that mutate ``HERMES_LITE_*`` env vars via ``monkeypatch``
    need ``get_config()`` to re-resolve. Without this, the first
    test that runs freezes the singleton and subsequent env-override
    tests get stale values.
    """
    from hermes_lite import config as _cfg

    _cfg._config_instance = None
    yield
    _cfg._config_instance = None
