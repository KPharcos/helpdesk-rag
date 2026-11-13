"""Application logic: one entry point the API calls, independent of HTTP and of providers."""

from __future__ import annotations

from typing import Any

from helpdesk_rag.domain import Answer, IndexNotReadyError


class QAService:
    def __init__(self, graph: Any | None, *, default_top_k: int) -> None:
        # `graph` is None when no index is loaded; /ask then reports index_not_ready.
        self._graph = graph
        self._default_top_k = default_top_k

    @property
    def ready(self) -> bool:
        return self._graph is not None

    async def ask(self, question: str, top_k: int | None = None) -> Answer:
        if self._graph is None:
            raise IndexNotReadyError(
                "The document index is not loaded. Run the ingest command "
                "(helpdesk-rag-ingest), then restart the app."
            )
        state = await self._graph.ainvoke(
            {"question": question, "top_k": top_k or self._default_top_k}
        )
        return Answer(
            answer=state["answer"],
            status=state["status"],
            sources=list(state.get("retrieved", [])),
            truncated=bool(state.get("truncated", False)),
        )
