"""Stdlib logging: one line per event, every line tagged with the current request id."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

# A ContextVar is per-task (and copied into asyncio.to_thread), not shared global state.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [req=%(request_id)s] %(message)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    # Replace only our own handler on repeated calls (tests create many apps).
    for existing in list(root.handlers):
        if getattr(existing, "_helpdesk_rag", False):
            root.removeHandler(existing)
    handler._helpdesk_rag = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Our middleware logs every request; uvicorn's access log would duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
    # Keep noisy client libraries at WARNING, unless the app itself runs at DEBUG:
    # then they inherit DEBUG (NOTSET), which shows the HTTP traffic.
    debugging = root.level <= logging.DEBUG
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic"):
        logging.getLogger(name).setLevel(logging.NOTSET if debugging else logging.WARNING)
