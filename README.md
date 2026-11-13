# helpdesk-rag

An internal **IT helpdesk knowledge-base Q&A service**. Employees ask questions in plain English
("How do I reset my password?", "The VPN says my certificate expired"), and the service answers
from the company's IT articles and returns the passages it gave the model as sources.

It is a small backend built to be easy to reproduce: FastAPI HTTP API, a LangGraph
retrieve → generate flow, swappable model/embedding/vector-store adapters, **Claude** as the
primary (paid, cloud) LLM, and a **fully free local mode** with Ollama. Embeddings always run
locally, so local mode needs no API key at all.

---

## Contents

1. [Problem, users and value](#1-problem-users-and-value)
2. [Features and limits](#2-features-and-limits)
3. [Quick start](#3-quick-start)
4. [Project structure](#4-project-structure)
5. [Architecture and design decisions](#5-architecture-and-design-decisions)
6. [Install, ingest, run, test: exact steps](#6-install-ingest-run-test-exact-steps)
7. [API and sample requests](#7-api-and-sample-requests)
8. [Configuration](#8-configuration)
9. [Primary model: Claude (cloud)](#9-primary-model-claude-cloud)
10. [Free local alternative: Ollama](#10-free-local-alternative-ollama)
11. [Docker](#11-docker)
12. [Concurrent users and scaling](#12-concurrent-users-and-scaling)
13. [Known limits and simplifications](#13-known-limits-and-simplifications)
14. [Next steps](#14-next-steps)

---

## 1. Problem, users and value

**Users:** employees of a fictional mid-size company, "Brightfield Ltd.", who have everyday IT
questions: passwords, VPN, MFA, Wi-Fi, laptops, software, printing, phishing, onboarding.

**Need:** answers to these questions are in the IT knowledge base, but people don't read long
articles. They open tickets or ping the Service Desk instead, which costs time on both sides.

**How the app helps:** it answers in a few sentences or numbered steps, **based only on the
knowledge base**, and returns the excerpts given to the model as sources so the employee (or a Service Desk
agent) can check the source. When the knowledge base has nothing relevant, it says so instead
of guessing.

**Sample data:** `data/docs/` contains 10 synthetic, non-confidential markdown articles written
for this project (all names, URLs and numbers are fictional; domains use the reserved `.example`
TLD). You can replace them with your own markdown files and re-run ingest.

## 2. Features and limits

Implemented:

- `POST /ask`: question → retrieval → grounded answer with sources (`doc_id`, `title`,
  `chunk_id`, `score`, `snippet`) and a `status` (`answered` | `no_relevant_context`).
- `GET /health`: liveness, index readiness, active LLM provider/model and embedding model.
- Interactive OpenAPI docs at `/docs` with request/response examples.
- Ingest as a CLI (`helpdesk-rag-ingest`) and optionally at startup (`INGEST_ON_STARTUP=true`).
- LangGraph flow: `validate → retrieve → (score threshold) → generate | no_context`.
- Two LLM providers, switched by `LLM_PROVIDER=anthropic|ollama` only (no code change).
- Input validation, one JSON error shape for every error, request id on every request
  (`X-Request-ID` echoed or generated) in the response and in every log line written while
  handling it (startup and ingest lines show `req=-`).
- Handling for empty/invalid requests (422), no relevant context (200 with explanation, LLM
  not called), LLM failure or refusal (502), LLM timeout (504), missing index (503).
- Per-process cap on concurrent LLM calls, overall LLM timeout, SDK retries with backoff.
- Offline test suite with fakes (no key, no Ollama, no model download) and separate
  integration smoke tests against real Claude and real Ollama.
- Dockerfile (non-root, embedding model baked in) and docker-compose (app + Ollama).

Deliberately **not** included (out of scope for this prototype): frontend, authentication,
user management, chat history / follow-up questions, document upload API, RAG quality
metrics or evaluation sets, streaming responses.

## 3. Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
git clone https://github.com/KPharcos/helpdesk-rag.git && cd helpdesk-rag
uv sync                              # install locked dependencies (Python 3.12)
uv run helpdesk-rag-ingest           # build the index (downloads a 67 MB embedding model on first use)
cp .env.example .env                 # then pick a provider in .env:
#   cloud: LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=sk-ant-...
#   local: LLM_PROVIDER=ollama   (see section 10 for installing Ollama)
uv run uvicorn helpdesk_rag.api.app:create_app --factory --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How do I reset my password?"}'
```

Or everything in Docker, local and free: `docker compose up --build` (section 11). Compose also reads `.env`:
if you ran `cp .env.example .env` above, set `LLM_PROVIDER=ollama` in it first, otherwise Docker starts in
Claude mode.

## 4. Project structure

```
src/helpdesk_rag/
├── config.py               # Settings (pydantic-settings): the single configuration surface
├── domain.py               # Chunk, RetrievedChunk, Answer + error hierarchy (provider-neutral)
├── ports.py                # Protocols: LLMClient, Embedder, VectorStore (the swap points)
├── service.py              # QAService.ask(): application logic, runs the compiled graph
├── ingest.py               # CLI + function: docs -> chunks -> embeddings -> saved index
├── logging_config.py       # stdlib logging, request id on every line (ContextVar)
├── rag/
│   ├── chunking.py         # heading-aware markdown chunker with size/overlap
│   ├── prompts.py          # grounded system prompt, user prompt builder, no-context answer
│   └── graph.py            # LangGraph StateGraph: validate -> retrieve -> generate | no_context
├── integrations/           # ALL provider-specific code lives here
│   ├── anthropic_llm.py    # Claude via the official `anthropic` SDK
│   ├── ollama_llm.py       # Ollama via the official `ollama` client
│   ├── fastembed_embedder.py  # local ONNX embeddings (BAAI/bge-small-en-v1.5)
│   ├── numpy_store.py      # cosine-similarity store persisted as .npy + JSON
│   └── factory.py          # Settings -> concrete adapters (LLM, embedder, vector store)
└── api/
    ├── app.py              # app factory, lifespan (build deps once), /ask and /health
    ├── schemas.py          # request/response models with OpenAPI examples
    ├── errors.py           # exception -> uniform JSON error mapping
    └── middleware.py       # request id, access log line, last-resort 500 handler
data/docs/                  # 10 synthetic IT helpdesk articles (markdown)
data/index/                 # generated index (git-ignored)
tests/                      # offline tests with fakes; tests/integration/ = real providers
Dockerfile, docker-compose.yml, docker-compose.gpu.yml, .env.example, uv.lock
```

## 5. Architecture and design decisions

### Components and layers

```
            HTTP
             │
┌────────────▼─────────────┐  api/        FastAPI: validation, error mapping, request id,
│  API layer               │              logging. Knows nothing about providers.
└────────────┬─────────────┘
             │ QAService.ask(question, top_k)
┌────────────▼─────────────┐  service.py  Application logic: readiness check, runs the graph,
│  Service                 │              returns a domain Answer.
└────────────┬─────────────┘
             │ graph.ainvoke(state)
┌────────────▼─────────────────────────────────────────────────────────┐  rag/
│  LangGraph flow                                                      │
│  START → validate → retrieve ──best ≥ MIN_RELEVANCE_SCORE?───┐       │
│                         │ no                            yes  ▼       │
│                         ▼                               generate → END
│                    no_context → END                                  │
└──────┬──────────────────┬────────────────────────────┬───────────────┘
       │ Embedder         │ VectorStore                │ LLMClient       ports.py (Protocols)
┌──────▼───────┐  ┌───────▼────────┐  ┌────────────────▼──────────────┐
│ fastembed    │  │ numpy cosine   │  │ AnthropicLLM  |  OllamaLLM    │  integrations/
│ (local CPU)  │  │ (.npy + JSON)  │  │ (Claude API)  |  (local HTTP) │  chosen by factory.py
└──────────────┘  └────────────────┘  └───────────────────────────────┘  from Settings
```

### Data flow

**Ingest** (`helpdesk-rag-ingest`, or at startup if enabled and no index exists):
`data/docs/*.md` → split at markdown headings → long sections split into ≤ `CHUNK_SIZE`
characters with `CHUNK_OVERLAP` → each chunk embedded as `"<title> - <section>\n<text>"` →
vectors written to `data/index/vectors.npy`, chunk metadata + embedding model name to
`data/index/chunks.json`. Each file is replaced atomically (unique temp file + rename), but the
two files are swapped one after the other; the app loads the index only at startup, so
**restart the app after re-ingesting**.

**Question** (`POST /ask`):
1. FastAPI validates the body (1–1000 chars after trimming, `top_k` 1–10).
2. `validate` node normalises whitespace.
3. `retrieve` node embeds the question (in a worker thread) and takes the `top_k` most similar
   chunks. The threshold gates the *question*: if the best chunk's cosine score is
   ≥ `MIN_RELEVANCE_SCORE`, all `top_k` chunks are kept; otherwise none are.
4. Route: no chunk kept → `no_context` returns a fixed explanatory answer **without calling the
   LLM**; otherwise → `generate` builds a grounded prompt (excerpts tagged with their
   `chunk_id`) and calls the configured LLM under a concurrency limit and timeout.
5. The API returns the answer, the status and, as `sources`, the excerpts given to the model
   (all `top_k` retrieved chunks once the question passed the gate).

### Main design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Heading-aware chunking**, 800 chars, 120 overlap | Helpdesk articles are organised by heading; a chunk = one sub-topic keeps answers focused and makes `section` a meaningful source label. | Character-based limits are approximate in tokens; good enough for short English articles. |
| **fastembed + `BAAI/bge-small-en-v1.5`** in both modes | Runs on CPU via ONNX, 67 MB, 384-dim, no PyTorch, no API key. One index serves both LLM providers, so switching provider never requires re-ingest, and local mode is free end to end. | English only; weaker than large hosted embedding models. Index records the model name and refuses to load with a different one. |
| **NumPy brute-force cosine store** | 62 chunks; an exact search is microseconds. No extra service to run, trivially testable, persisted as two files. | Linear in corpus size, whole index in memory, no live updates. Replace behind the `VectorStore` port with Qdrant/pgvector when the corpus grows (section 12). |
| **Relevance threshold before the LLM** (`MIN_RELEVANCE_SCORE=0.62`) | Off-topic questions get a clear `no_relevant_context` answer, cost nothing and cannot be "answered" from irrelevant text. Calibrated on the sample data (8 on-topic, 8 off-topic questions, best-chunk score): on-topic 0.645–0.847; off-topic (HR policy, trivia, greetings) 0.417–0.6195, the highest being "How do I book a meeting room?" at 0.6195, just below the gate (`score >= 0.62`). The gate applies to the **best** chunk only; once it passes, all `top_k` chunks go to the LLM. A chunk that holds the answer often scores a little below a broader one (e.g. "I spilled coffee on my laptop" → liquid-damage chunk 0.6065, lost/stolen chunk 0.6279), so filtering chunk by chunk would drop the answer. | A fixed cosine threshold is model- and corpus-specific; re-check it when changing either. Weaker chunks behind a passing best chunk may be off-topic; the prompt tells the model to answer only from the excerpts and to say "I don't know" when they do not cover the question. |
| **LangGraph** with a per-request `TypedDict` state | Makes the flow explicit and the routing testable node by node; easy to add steps (query rewrite, re-ranking). | Small overhead for a 4-node graph. |
| **Ports + adapters**, provider code only in `integrations/` | Providers are swapped by config and replaced by fakes in tests; `tests/test_architecture.py` fails if core code imports a provider SDK or a concrete adapter instead of going through `integrations/factory.py`. No LangChain LLM wrappers: the official SDKs are called directly, so error mapping and request parameters are fully under our control. | A bit more code than a framework wrapper. |
| **Claude Opus 5.5 at `effort: low`** | Current default Opus model: strong instruction following, so it sticks to the excerpts and cites `[chunk_id]`s reliably. Grounded Q&A over a few short excerpts is a simple task, so low effort keeps latency and cost down (estimated at roughly 1–2 cents per question with Opus 5.5 at $4/$20 per million input/output tokens; thinking tokens are billed as output). Effort is configurable (`ANTHROPIC_EFFORT`). | Paid, network-dependent. |
| **Server-side refusal fallbacks** (`fallbacks: "default"`) | Security content (phishing, MFA attacks) can occasionally trip a model's safety classifier. With fallbacks the API re-runs a declined request on a fallback model within the same call, so users rarely see a refusal; if the whole chain refuses we return `llm_error`. | Beta feature (`server-side-fallback-2026-07-01`); only on the first-party Claude API. |
| **Llama 3.1 8B via Ollama** for local mode | Free model, good instruction following, 128k context, fits an 8 GB GPU in the default 4-bit quantisation and still runs on CPU. | Slower than Claude and expected to follow the grounding rules less reliably (not measured; in spot checks on the sample data it answered and cited `[chunk_id]`s correctly). |

## 6. Install, ingest, run, test: exact steps

```bash
# 0. Get the code
git clone https://github.com/KPharcos/helpdesk-rag.git
cd helpdesk-rag

# 1. Install (Python 3.12 is pinned in .python-version; uv downloads it if needed)
uv sync

# 2. Prepare data: build the vector index from data/docs/*.md
#    (first run downloads the embedding model, ~67 MB, into fastembed's cache: the system temp dir by default, so
#    it is fetched again after a reboot that clears /tmp; set EMBEDDING_CACHE_DIR to keep it)
uv run helpdesk-rag-ingest
#    -> "Indexed 62 chunks into data/index"
#    Options: --docs-dir PATH --index-dir PATH (or DOCS_DIR / INDEX_DIR env vars)
#    The app loads the index at startup only: restart it after re-ingesting.

# 3. Configure
cp .env.example .env        # choose LLM_PROVIDER and fill in what it needs (sections 9, 10)

# 4. Run
uv run uvicorn helpdesk_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000
#    OpenAPI UI: http://localhost:8000/docs

# 5. Test (offline: no API key, no Ollama, no model download needed)
uv run pytest
#    Integration smoke tests (the Claude and Ollama tests skip if their provider is unavailable;
#    the embedder gate test needs the embedding-model download):
ANTHROPIC_API_KEY=sk-ant-... uv run pytest -m integration    # Claude
uv run pytest -m integration                                 # Ollama, if running with the model pulled
```

The default test suite covers every row of the error matrix (happy path, each invalid input,
no relevant context, LLM error, refusal, timeout, missing index), the chunker, the NumPy store
round-trip, graph routing, concurrency limiting, request isolation, provider error mapping and
the config-based provider switch, plus an architecture check that provider and storage adapters
are only reached through `integrations/factory.py`.

## 7. API and sample requests

| Method & path | Purpose |
|---|---|
| `POST /ask` | Body `{"question": str (1–1000 chars), "top_k": int 1–10 (optional)}` |
| `GET /health` | `{"status":"ok","index_ready":bool,"indexed_chunks":int,"llm_provider":..,"llm_model":..,"embedding_model":..}` |
| `GET /docs` | Swagger UI with examples |

```bash
# Happy path
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How do I reset my password?"}'

# Pass your own request id (echoed in the X-Request-ID header, the body and the logs)
curl -si -X POST localhost:8000/ask -H 'Content-Type: application/json' -H 'X-Request-ID: demo-42' \
     -d '{"question": "The VPN says my certificate expired. What should I do?", "top_k": 3}'

# More questions the sample data answers
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "I got a new phone. How do I move my authenticator?"}'
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "I clicked a link in a suspicious email, what now?"}'

# No relevant context -> 200, status "no_relevant_context", LLM not called
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is the capital of France?"}'

# Invalid input -> 422 validation_error
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "   "}'
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "x", "top_k": 50}'
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' -d 'not json'

# Health
curl -s localhost:8000/health
```

Example success response (Claude; answer text shortened):

```json
{
  "answer": "1. Open https://password.brightfield.example and select \"Forgot password\" [password-reset#2].\n2. Approve the MFA request, choose a new password and sign in on your laptop while connected to the office network or VPN [password-reset#2].",
  "status": "answered",
  "sources": [
    {"doc_id": "password-reset", "title": "Resetting Your Password", "chunk_id": "password-reset#3",
     "score": 0.8106, "snippet": "On a Windows laptop press Ctrl+Alt+Del and choose \"Change a password\". ..."},
    {"doc_id": "password-reset", "title": "Resetting Your Password", "chunk_id": "password-reset#2",
     "score": 0.7631, "snippet": "1. On any device, open the self-service portal at https://password.brightfield.example. ..."}
  ],
  "request_id": "demo-42"
}
```

Every error has the same shape:

```json
{"error": {"code": "llm_timeout", "message": "The language model did not answer in time. Please retry.", "request_id": "3f2b9c0e..."}}
```

| HTTP | `code` | When |
|---|---|---|
| 422 | `validation_error` | empty / whitespace / > 1000 chars / missing question, wrong types, `top_k` outside 1–10, non-JSON or missing body |
| 502 | `llm_error` | provider error, rate limit after retries, unreachable provider, model missing in Ollama, refusal (`stop_reason == "refusal"` after fallbacks), empty answer |
| 503 | `index_not_ready` | no usable index was loaded at startup (`INGEST_ON_STARTUP=false`, or startup ingest failed). Run `helpdesk-rag-ingest`, then **restart the app**: a running process never picks up a new index |
| 504 | `llm_timeout` | LLM call (including waiting for a concurrency slot) exceeded `LLM_TIMEOUT_SECONDS` |
| 404 / 405 | `not_found` / `method_not_allowed` | unknown route / wrong method |
| 500 | `internal_error` | unexpected bug (logged with traceback and request id) |

Provider details (exception types, status codes) are logged but never returned to the client.

Logs are one line per event, e.g.:

```
2026-11-12 16:50:21,522 INFO helpdesk_rag.rag.graph [req=demo-1] retrieved hits=4 top_score=0.811 threshold=0.620 gate=pass
2026-11-12 16:50:21,524 ERROR helpdesk_rag.api.errors [req=demo-1] LLM error: Could not reach Ollama: ... (cause: ConnectionError)
2026-11-12 16:50:21,526 INFO helpdesk_rag.access [req=demo-1] POST /ask -> 502 in 7.6ms
```

## 8. Configuration

All settings are environment variables (or a `.env` file in the working directory); see
`.env.example`. The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` (Claude) or `ollama` (local) |
| `ANTHROPIC_API_KEY` | – | Claude API key (optional if the SDK finds credentials otherwise) |
| `ANTHROPIC_MODEL` / `ANTHROPIC_EFFORT` | `claude-opus-5-5` / `low` | Claude model and effort (`low`…`max`) |
| `ANTHROPIC_MAX_TOKENS` | `16000` | output cap (includes the model's internal reasoning) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.1:8b` | local model server and model |
| `OLLAMA_NUM_CTX` / `OLLAMA_TEMPERATURE` | `8192` / `0.1` | context window and sampling for Ollama |
| `LLM_TIMEOUT_SECONDS` | `60` | overall LLM budget per request, including the concurrency queue |
| `LLM_MAX_RETRIES` | `2` | Claude SDK retries (429, 5xx, connection errors, with backoff) |
| `MAX_CONCURRENT_LLM_CALLS` | `8` | per-process cap on in-flight LLM calls |
| `EMBEDDING_MODEL` / `EMBEDDING_CACHE_DIR` | `BAAI/bge-small-en-v1.5` / fastembed default (system temp dir) | embedding model and its download cache |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | characters; overlap must be < half the size |
| `TOP_K` / `MIN_RELEVANCE_SCORE` | `4` / `0.62` | retrieval depth and cosine threshold |
| `DOCS_DIR` / `INDEX_DIR` | `data/docs` / `data/index` | input documents and index location |
| `INGEST_ON_STARTUP` | `false` | build the index at startup when none exists |
| `LOG_LEVEL` | `INFO` | standard logging level |

Changing `LLM_PROVIDER` and restarting is all it takes to switch modes; `/health` shows the
active provider and model. Changing `EMBEDDING_MODEL` or the chunking settings requires
re-running ingest (the app refuses to load an index built with a different embedding model).

## 9. Primary model: Claude (cloud)

1. Create an API key in the [Claude Console](https://platform.claude.com/).
2. Configure:
   ```bash
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   # optional: ANTHROPIC_MODEL=claude-opus-5-5  ANTHROPIC_EFFORT=low
   ```
3. Ingest (once) and start the app as in section 6. `curl localhost:8000/health` should show
   `"llm_provider":"anthropic","llm_model":"claude-opus-5-5"`.

Request details (`integrations/anthropic_llm.py`): async client with `timeout` and
`max_retries` from config; `client.beta.messages.create(model, max_tokens, system, messages,
output_config={"effort": ...}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")`.
No sampling parameters are sent (the model doesn't accept them). The adapter checks
`stop_reason` before reading content, treats `refusal` as `llm_error`, and reads only `text`
blocks. SDK exceptions are mapped: `APITimeoutError` → `llm_timeout`; `RateLimitError`,
`APIStatusError`, `APIConnectionError` → `llm_error`.

## 10. Free local alternative: Ollama

In this mode the whole pipeline runs on your machine without any paid key: embeddings with
fastembed (CPU) and generation with Ollama.

**Install and start Ollama** (needs internet once, for the installer and the model):

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh     # installs and starts the `ollama` service
# macOS / Windows: download the app from https://ollama.com/download and start it

ollama pull llama3.1:8b          # ~4.9 GB download (4-bit quantised)
ollama list                      # verify the model is there
# if the service is not running: ollama serve   (listens on http://localhost:11434)
```

**Configure and run the app:**

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
LLM_TIMEOUT_SECONDS=120          # local generation is slower, especially on CPU
```

```bash
uv run helpdesk-rag-ingest       # if not done yet
uv run uvicorn helpdesk_rag.api.app:create_app --factory --port 8000
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How do I connect to the VPN?"}'
```

No `ANTHROPIC_API_KEY` is needed or read in this mode.

**Hardware and memory (llama3.1:8b, Q4, `OLLAMA_NUM_CTX=8192`):**

| Setup | Memory | Expected speed |
|---|---|---|
| NVIDIA GPU with 8 GB VRAM (the reference machine, RTX 5070 Laptop) | 5.8 GB VRAM (≈4.9 GB weights + KV cache; `ollama ps`) | ≈ 68 tokens/s measured (one 185-token answer) → 1–4 s per answer |
| CPU only | ~6–8 GB free RAM, 16 GB system RAM recommended | estimate, not measured: 3–10 tokens/s → 20–90 s per answer |
| Apple Silicon (unified memory) | 16 GB recommended | estimate, not measured: 15–30 tokens/s |

Plus the app itself: about 330 MiB RSS as a local process with the embedding model and index
loaded, after a few questions (`ps`), and about 555 MiB for the app container, which also
builds the index at startup (`docker stats`). Disk: ~5 GB for the model.
Smaller GPUs or RAM: use a smaller model, e.g. `OLLAMA_MODEL=llama3.2:3b` (~2 GB), with lower
answer quality.

**Limits of local mode:** slower, especially on CPU; answer quality and citation discipline are
expected to be weaker than Claude's (not measured; spot checks on the sample data were fine); one Ollama server
processes only a few requests in parallel (`OLLAMA_NUM_PARALLEL`), the rest queue; the first
request after idle includes model load time (seconds, longer on CPU). No retries are
configured for Ollama because failures are usually local (not running, model not pulled) and
would not be fixed by retrying.

**Reaching Ollama from the app:** the app talks to Ollama's HTTP API at `OLLAMA_BASE_URL`.
Locally that's `http://localhost:11434`; in docker-compose it's the service name,
`http://ollama:11434` (both tested). If the app runs in Docker and Ollama on the host, use
`http://host.docker.internal:11434` (on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]`).
On Linux two more things are needed, because the Ollama service listens on 127.0.0.1 only and
host firewalls usually drop traffic from Docker networks: make Ollama listen on all interfaces
(`sudo systemctl edit ollama`, then `Environment="OLLAMA_HOST=0.0.0.0:11434"`) and allow Docker
subnets to reach the port (with ufw: `sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp`).
Checked on the development machine: with Ollama on 127.0.0.1, or on 0.0.0.0 while ufw had no
such rule, the container could not connect; the full recipe with the firewall rule was not run.
The compose setup in section 11 needs none of this.

## 11. Docker

The image (`Dockerfile`) installs the locked dependencies with uv in a build stage, runs as a
non-root user (`app`, uid 1000), downloads the embedding model at build time (so containers
start without internet access), and builds the index at startup (`INGEST_ON_STARTUP=true`,
< 1 s for the sample docs). Its `HEALTHCHECK` calls `/health`, which checks liveness only
(the endpoint always returns 200; readiness is the `index_ready` field).

**Configuration:** compose reads an optional `.env` next to `docker-compose.yml`, both for
`${...}` interpolation and as the app's `env_file`, so every setting in `.env.example` can be
tuned. Entries under `environment:` in `docker-compose.yml` win (e.g. `OLLAMA_BASE_URL` is
always `http://ollama:11434` inside compose). **Note: `LLM_PROVIDER` in `.env` also controls
the Docker mode.** The default local mode only applies when `.env` doesn't set it; after
`cp .env.example .env` (which sets `LLM_PROVIDER=anthropic`) change it to `ollama` for the
free local mode. `.env.example` also sets `LLM_TIMEOUT_SECONDS=60`, which overrides compose's default of
120: raise it to 120 or more for Ollama on a CPU. Ollama's port is published on `127.0.0.1:11434` only,
because its API has no authentication.

**Local, free mode (default), app + Ollama:**

```bash
docker compose up --build
# first start: the one-shot `ollama-pull` service downloads llama3.1:8b (~4.9 GB) into the
# named volume `ollama-models`; the app starts after the pull has finished.
curl -s localhost:8000/health
```

With an NVIDIA GPU (requires the NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**Cloud mode (Claude), app only:**

```bash
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... docker compose up --build --no-deps app
# or without compose:
docker build -t helpdesk-rag .
docker run --rm -p 8000:8000 -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=sk-ant-... helpdesk-rag
```

**Your own documents:** mount a folder of markdown files over `/app/data/docs`, e.g.
`docker run -v "$PWD/my-docs:/app/data/docs:ro" ...`; the index is built from them when a
new container starts (a restarted container keeps its existing index; recreate it, or run
`helpdesk-rag-ingest` inside it and restart, after changing documents).

## 12. Concurrent users and scaling

This section describes what the current code does (**implemented**) and what I'd do next
(**next step**) for each concern.

### Request isolation

- **Implemented:** everything shared between requests is built once at startup and only read
  afterwards: settings, adapters, the compiled graph, and the vector index (the NumPy array is
  marked read-only). Request data lives only in the per-request LangGraph state (a fresh
  `TypedDict` per `ainvoke`), local variables and the response. The request id is a
  `ContextVar`, which is per task, not a global. There is no chat history, no answer cache and
  no per-user data in memory, so one user's question or answer cannot leak into another's.
  A test fires concurrent requests on different topics and checks that the sources don't mix.
- **Next step:** if chat history or personalisation is added, store it outside the process
  (Redis/Postgres) keyed by the authenticated user id, and pass only that user's data into the
  graph state. With per-department documents, filter retrieval by the caller's permissions in
  the vector store query, not after generation.

### Parallel execution: what blocks and what doesn't

- **Implemented:**
  - LLM calls (the slow part, seconds) are fully async (`AsyncAnthropic`, `ollama.AsyncClient`),
    so one event loop serves many waiting requests.
  - Query embedding is CPU-bound ONNX inference (a few ms), so it runs in a worker thread via
    `asyncio.to_thread` and doesn't block the loop.
  - Vector search is a NumPy matrix-vector product, microseconds at this size, so it runs inline.
  - Model loading and startup ingest run in a thread during startup, never in a request.
  - Ingest is a separate command, not part of the request path.
- **Next step:** run several uvicorn workers (`--workers N`, roughly one per CPU core) to use
  more cores. Each worker loads its own embedding model and index (~0.35 GB), which is fine at
  this size. When documents can be uploaded or change often, ingest becomes a background job
  on a queue (e.g. a worker consuming "document changed" events) that writes a new index
  version, which the API switches to atomically. Move vector search into a thread or a vector
  DB once the corpus reaches hundreds of thousands of chunks.

### Resource limits: concurrency, API limits, timeouts, transient errors, overload

- **Implemented:**
  - `MAX_CONCURRENT_LLM_CALLS` (an `asyncio.Semaphore` per process) caps in-flight provider
    calls, which protects the Claude rate limit and a local Ollama server from bursts.
  - `LLM_TIMEOUT_SECONDS` is enforced around the whole LLM step with `asyncio.wait_for`,
    **including the time spent waiting for a semaphore slot**, plus the SDK's own per-request
    timeout. A slow or queued request returns `504 llm_timeout` instead of hanging.
  - The Anthropic SDK retries transient failures (429, 5xx, connection errors) with
    exponential backoff (`LLM_MAX_RETRIES`); the outer timeout bounds the total time.
  - Refusals are retried server-side on a fallback model (fallbacks), then mapped to 502.
  - Off-topic questions never reach the LLM (relevance threshold), saving capacity.
  - Errors are mapped to clear statuses (502/503/504) so clients and load balancers can react.
- **Next step:** load shedding: when the semaphore queue is longer than a limit, reject
  immediately with `503` + `Retry-After` instead of letting requests time out; per-user rate
  limits at the API gateway; a circuit breaker that fails fast for a while after repeated
  provider errors; reading Anthropic's rate-limit response headers to slow down before hitting
  429s. With several instances the per-process semaphore multiplies (N instances × limit), so
  the global limit would move to a shared limiter (e.g. Redis token bucket).

### Multiple instances

- **Implemented / stateless already:** the API process holds no session or user state, all
  configuration comes from the environment, and the index is rebuilt deterministically from the
  documents (or loaded from `INDEX_DIR`). Any number of identical containers can run behind a
  load balancer today. `/health` always returns HTTP 200, so its status code (and the Docker
  `HEALTHCHECK`) only checks liveness; readiness is the `index_ready` field in its body, which
  a readiness probe has to inspect.
- **Needs shared, durable storage when scaling out:**
  - The **source documents**: object storage or a docs repository, not files in the image.
  - The **index**: build it once in a CI/ingest job and publish it as a versioned artifact
    (object storage) that instances download at startup, or move to a shared vector database
    (Qdrant, pgvector) when the corpus is large or must update without redeploys.
  - **Rate-limit / concurrency counters** across instances (Redis), see above.
  - **Ollama** runs as a separate service (one per GPU node) behind its own load balancer;
    app instances share it via `OLLAMA_BASE_URL`.
  - Future chat history / feedback / audit logs: a database.

### Bottlenecks, metrics, order of improvements

- **Main limit:** LLM generation. With Claude it's latency (seconds per answer) and the
  organisation's rate limits (requests and tokens per minute); with Ollama on a single 8 GB GPU
  it's throughput (a few parallel generations at best, far fewer on CPU). Embedding (ms) and
  retrieval (µs) are negligible in comparison.
- **Metrics I'd watch:** request rate and p50/p95/p99 latency per graph node (retrieve vs.
  generate); LLM error, timeout, refusal and 429 rates; semaphore wait time and queue depth;
  in-flight LLM calls; input/output tokens and cost per request; share of
  `no_relevant_context` answers and the top-score distribution (content gaps or a mistuned
  threshold); event-loop lag, CPU and memory; for Ollama, GPU utilisation and memory.
- **Order of improvements:**
  1. Metrics and tracing (Prometheus + OpenTelemetry, request id as trace attribute), so the
     next steps are driven by data.
  2. Load shedding and per-user rate limits (cheap, prevents cascading timeouts).
  3. Streaming responses (SSE) to cut perceived latency.
  4. Horizontal scaling: more workers/instances plus a shared rate limiter.
  5. Shared index storage / vector database when the corpus grows or needs live updates.
  6. For local mode: a dedicated GPU server (vLLM or Ollama with more parallel slots) if many
     users need the free mode.

## 13. Known limits and simplifications

- Single-turn Q&A only: no conversation history or follow-up questions.
- No authentication, authorisation or per-user document permissions.
- The index is rebuilt from scratch (no incremental updates) and loaded only at startup:
  restart the app after re-ingesting. No upload API.
- English only (embedding model and prompts).
- The relevance threshold is a fixed cosine value calibrated on the sample data.
- Citations in the answer come from the model and are not verified (sources are always
  returned separately from retrieval, so they are reliable).
- The answer can still be wrong if the model misreads an excerpt; there is no answer-quality
  evaluation.
- `/health` always returns HTTP 200 (liveness); index readiness is only in the `index_ready`
  field. It reports the configured provider but doesn't call the LLM.
- Per-process concurrency limit only; no global rate limiting, load shedding or circuit breaker.
- No streaming responses.
- The Docker, Ollama and Claude paths were each run end-to-end on the development machine (Docker 29,
  RTX 5070 Laptop with the GPU compose override, `claude-opus-5-5`). Not run: CPU-only Ollama, Apple
  Silicon, compose without the GPU override, and the full host-Ollama-from-Docker recipe in section 10. The speed figures for those setups in section 10 are estimates.

## 14. Next steps

1. Observability: metrics and traces per graph node, token/cost accounting.
2. Load shedding, rate limiting and a circuit breaker (section 12).
3. Streaming answers over SSE.
4. Hybrid retrieval (BM25 + vectors) and a re-ranker for better recall on exact terms
   (error codes, product names).
5. Incremental ingest as a background job, a shared vector DB, document permissions.
6. Conversation memory with question rewriting for follow-ups.
7. A small evaluation set for regression-testing retrieval and answers when prompts,
   models or documents change.
