# syntax=docker/dockerfile:1

# ---- build stage: resolve the locked dependencies into /app/.venv ----------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime stage --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app data/docs /app/data/docs
RUN mkdir -p /app/data/index /app/models && chown -R app:app /app/data /app/models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DOCS_DIR=/app/data/docs \
    INDEX_DIR=/app/data/index \
    EMBEDDING_CACHE_DIR=/app/models \
    INGEST_ON_STARTUP=true \
    HF_HUB_DISABLE_TELEMETRY=1

USER app

# Pre-download the embedding model at build time so containers start without internet.
RUN python -c "from helpdesk_rag.config import Settings; \
from helpdesk_rag.integrations.factory import build_embedder; \
e = build_embedder(Settings()); print('embedding model ready:', e.model_name)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "helpdesk_rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
