from __future__ import annotations

import logging

from helpdesk_rag.logging_config import configure_logging

CLIENT_LIBRARIES = ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic")


def test_client_libraries_are_quiet_unless_the_app_runs_at_debug() -> None:
    root = logging.getLogger()
    previous = root.level
    try:
        configure_logging("INFO")
        assert {logging.getLogger(n).getEffectiveLevel() for n in CLIENT_LIBRARIES} == {logging.WARNING}
        configure_logging("DEBUG")
        assert {logging.getLogger(n).getEffectiveLevel() for n in CLIENT_LIBRARIES} == {logging.DEBUG}
        configure_logging("WARNING")
        assert {logging.getLogger(n).getEffectiveLevel() for n in CLIENT_LIBRARIES} == {logging.WARNING}
    finally:
        configure_logging(logging.getLevelName(previous) if previous else "INFO")
