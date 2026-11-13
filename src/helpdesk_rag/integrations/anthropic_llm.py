"""Claude adapter (official `anthropic` SDK). All Anthropic-specific details live here."""

from __future__ import annotations

import logging

import anthropic

from helpdesk_rag.domain import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)

# Server-side refusal fallbacks: if the primary model declines on policy grounds, the API
# re-runs the same request on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str,
        max_tokens: int,
        effort: str,
    ) -> None:
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._effort = effort

    async def generate(self, system: str, prompt: str) -> str:
        try:
            response = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        # APITimeoutError subclasses APIConnectionError, so it must be caught first.
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError("Claude request timed out") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit exceeded (retries exhausted)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API returned HTTP {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Claude API") from exc
        # The SDK raises TypeError before sending when no credentials resolve
        # (no ANTHROPIC_API_KEY, auth token or profile).
        except TypeError as exc:
            if "authentication" not in str(exc).lower():
                raise  # a genuine bug (e.g. SDK signature change) must surface as a 500
            logger.error("Claude client could not build the request: %s", exc)
            raise LLMError("Claude client is not configured (missing API key?)") from exc
        # Any other SDK error (e.g. APIResponseValidationError) is still a provider failure.
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {type(exc).__name__}") from exc

        # Always check stop_reason before reading content.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMError(f"Claude declined to answer (category={category})")
        if response.stop_reason == "max_tokens":
            logger.warning("Claude answer truncated at max_tokens=%s", self._max_tokens)

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError(f"Claude returned no text (stop_reason={response.stop_reason})")
        return text

    async def aclose(self) -> None:
        await self._client.close()
