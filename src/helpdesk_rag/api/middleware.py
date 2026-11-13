"""Pure-ASGI middleware: request id, access log line, and last-resort 500 handler."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from helpdesk_rag.logging_config import request_id_var

logger = logging.getLogger("helpdesk_rag.access")

REQUEST_ID_HEADER = "X-Request-ID"
# Accept caller-supplied ids only if they are short and log-safe; otherwise generate one.
_VALID_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _log_safe(text: str) -> str:
    """Escape control characters (the ASGI path is percent-decoded, so %0A would forge log lines)."""
    return text.encode("unicode_escape").decode("ascii")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _VALID_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        status_code = 500
        response_started = False
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled error")
            if response_started:
                raise
            body = json.dumps(
                {"error": {"code": "internal_error", "message": "Internal server error", "request_id": request_id}}
            ).encode()
            await send_wrapper(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
        finally:
            logger.info(
                "%s %s -> %d in %.1fms",
                scope.get("method"),
                _log_safe(scope.get("path", "")),
                status_code,
                (time.perf_counter() - started) * 1000,
            )
            request_id_var.reset(token)
