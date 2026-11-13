"""FastAPI app factory: wires settings -> adapters -> graph -> service once, at startup.

Run with:  uvicorn helpdesk_rag.api.app:create_app --factory
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request

from helpdesk_rag.api.errors import register_exception_handlers
from helpdesk_rag.api.middleware import RequestContextMiddleware
from helpdesk_rag.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from helpdesk_rag.config import Settings
from helpdesk_rag.logging_config import configure_logging
from helpdesk_rag.ports import Embedder, LLMClient, VectorStore
from helpdesk_rag.rag.graph import build_graph
from helpdesk_rag.service import QAService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppContainer:
    """Everything built at startup. Read-only afterwards; shared by all requests."""

    settings: Settings
    llm: LLMClient
    embedder: Embedder
    store: VectorStore | None
    service: QAService


async def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> AppContainer:
    from helpdesk_rag.integrations import factory

    llm = llm or factory.build_llm(settings)
    # Model load (and first-time download) is blocking; keep it off the event loop.
    embedder = embedder or await asyncio.to_thread(factory.build_embedder, settings)
    if store is None:
        store = factory.load_store(settings)
    if store is None and settings.ingest_on_startup:
        from helpdesk_rag.ingest import build_index

        logger.info("INGEST_ON_STARTUP=true and no index found: building index")
        try:
            store = await asyncio.to_thread(build_index, settings, embedder)
        except (OSError, ValueError) as exc:  # incl. FileNotFoundError, UnicodeDecodeError
            # Keep serving: /health reports index_ready=false and /ask returns 503.
            logger.error("startup ingest failed, continuing without an index: %s", exc)
            store = None

    graph = None
    if store is not None:
        graph = build_graph(
            embedder=embedder,
            store=store,
            llm=llm,
            min_score=settings.min_relevance_score,
            llm_timeout_seconds=settings.llm_timeout_seconds,
            llm_semaphore=asyncio.Semaphore(settings.max_concurrent_llm_calls),
        )
    service = QAService(graph, default_top_k=settings.top_k)
    logger.info(
        "app ready provider=%s model=%s index_ready=%s chunks=%d",
        settings.llm_provider,
        settings.active_llm_model,
        service.ready,
        len(store) if store is not None else 0,
    )
    return AppContainer(settings=settings, llm=llm, embedder=embedder, store=store, service=service)


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


Container = Annotated[AppContainer, Depends(get_container)]

router = APIRouter()

_ERRORS = {
    422: {"model": ErrorResponse, "description": "Invalid request (`validation_error`)"},
    502: {"model": ErrorResponse, "description": "LLM provider failed or refused (`llm_error`)"},
    503: {"model": ErrorResponse, "description": "Index not built (`index_not_ready`)"},
    504: {"model": ErrorResponse, "description": "LLM call timed out (`llm_timeout`)"},
}


@router.post(
    "/ask",
    response_model=AskResponse,
    responses=_ERRORS,
    summary="Ask the IT helpdesk knowledge base a question",
)
async def ask(
    body: AskRequest,
    request: Request,
    container: Container,
) -> AskResponse:
    answer = await container.service.ask(body.question, body.top_k)
    return AskResponse.from_answer(answer, request_id=request.state.request_id)


@router.get("/health", response_model=HealthResponse, summary="Liveness and index readiness")
async def health(container: Container) -> HealthResponse:
    settings = container.settings
    return HealthResponse(
        index_ready=container.service.ready,
        indexed_chunks=len(container.store) if container.store is not None else 0,
        llm_provider=settings.llm_provider,
        llm_model=settings.active_llm_model,
        embedding_model=container.embedder.model_name,
    )


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> FastAPI:
    """App factory. Tests inject fakes via the keyword arguments; production passes nothing."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(settings, llm=llm, embedder=embedder, store=store)
        app.state.container = container
        try:
            yield
        finally:
            close = getattr(container.llm, "aclose", None)
            if close is not None:
                await close()

    app = FastAPI(
        title="Helpdesk RAG",
        version="0.1.0",
        description=(
            "Internal IT helpdesk Q&A over a markdown knowledge base. "
            "Retrieval-augmented answers with cited sources, powered by Claude or a local Ollama model."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app
