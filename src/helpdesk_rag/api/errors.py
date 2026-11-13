"""Exception -> uniform JSON error mapping: {"error": {"code", "message", "request_id"}}."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from helpdesk_rag.domain import (
    IndexNotReadyError,
    InvalidQuestionError,
    LLMError,
    LLMTimeoutError,
)
from helpdesk_rag.logging_config import request_id_var

logger = logging.getLogger(__name__)

_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def error_response(status_code: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id_var.get()}},
        headers=headers,
    )


def _describe_validation(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body"
        if err.get("type") == "json_invalid":
            parts.append("request body is not valid JSON")
        else:
            parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts) or "invalid request"


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = _describe_validation(exc)
    logger.info("validation error: %s", message)
    return error_response(422, "validation_error", message)


async def _invalid_question_handler(request: Request, exc: InvalidQuestionError) -> JSONResponse:
    return error_response(422, "validation_error", f"question: {exc}")


async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))


async def _llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    logger.error("LLM timeout: %s", exc)  # one line, no traceback
    return error_response(504, "llm_timeout", "The language model did not answer in time. Please retry.")


async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    # Expected operational failure: one log line with the provider cause, no traceback.
    cause = f" (cause: {type(exc.__cause__).__name__})" if exc.__cause__ is not None else ""
    logger.error("LLM error: %s%s", exc, cause)
    return error_response(502, "llm_error", "The language model call failed. Please retry later.")


async def _index_not_ready_handler(request: Request, exc: IndexNotReadyError) -> JSONResponse:
    logger.warning("index not ready: %s", exc)
    # No Retry-After: a running process never picks up a later index (loaded at startup only).
    return error_response(503, "index_not_ready", str(exc))


def register_exception_handlers(app: FastAPI) -> None:
    # Starlette resolves handlers by walking the exception MRO, so LLMTimeoutError wins over
    # its parent LLMError. Unhandled exceptions are turned into 500s by the middleware.
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(InvalidQuestionError, _invalid_question_handler)
    app.add_exception_handler(StarletteHTTPException, _http_handler)
    app.add_exception_handler(LLMTimeoutError, _llm_timeout_handler)
    app.add_exception_handler(LLMError, _llm_error_handler)
    app.add_exception_handler(IndexNotReadyError, _index_not_ready_handler)
