"""Local Ollama adapter (official `ollama` Python client). No API key, no paid service."""

from __future__ import annotations

import httpx
import ollama

from helpdesk_rag.domain import LLMError, LLMTimeoutError


class OllamaLLM:
    def __init__(
        self,
        client: ollama.AsyncClient,
        *,
        model: str,
        num_ctx: int,
        temperature: float,
    ) -> None:
        self._client = client
        self.model = model
        self._options = {"num_ctx": num_ctx, "temperature": temperature}

    async def generate(self, system: str, prompt: str) -> str:
        try:
            response = await self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options=self._options,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Ollama request timed out") from exc
        except ollama.ResponseError as exc:
            # e.g. 404 "model not found" when `ollama pull` was not run.
            raise LLMError(f"Ollama error: {exc.error} (HTTP {exc.status_code})") from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMError(f"Could not reach Ollama: {exc}") from exc

        text = (response.message.content or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty answer")
        return text

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
