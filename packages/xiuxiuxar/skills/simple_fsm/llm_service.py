# ------------------------------------------------------------------------------
#
#   Copyright 2025 xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""LLM Abstraction Layer for OpenAI-compatible APIs."""

import os
import time
import logging

from openai import OpenAI, OpenAIError, APIStatusError, RateLimitError, APITimeoutError, APIConnectionError


logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""


class LLMRateLimitError(LLMServiceError):
    """Raised on rate limit errors."""


class LLMAPIError(LLMServiceError):
    """Raised on API errors."""


class LLMContentFilterError(LLMServiceError):
    """Raised on content filtering errors."""


class LLMInvalidResponseError(LLMServiceError):
    """Raised on invalid or unexpected responses."""


class LLMService:
    """LLM Service for OpenAI-compatible APIs."""

    def __init__(self):
        self.primary_model = os.environ["LLM_PRIMARY_MODEL"]
        self.fallback_model = os.environ["LLM_FALLBACK_MODEL"]
        self.api_key = os.environ["LLM_API_KEY"]
        self.base_url = os.environ["LLM_BASE_URL"]
        self.max_retries = int(os.environ["LLM_MAX_RETRIES"])
        self.backoff_factor = float(os.environ["LLM_BACKOFF_FACTOR"])
        self.timeout = int(os.environ["LLM_TIMEOUT"])

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_summary(self, prompt: str, model_config: dict) -> dict:
        """Generate a summary using the configured LLM backend.
        Returns dict: {
            "content": str,
            "llm_model_used": str,
            "generation_time_ms": int,
            "token_usage": dict
        }.
        """
        models_to_try = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models_to_try.append(self.fallback_model)

        last_exception = None
        for model in models_to_try:
            try:
                return self._call_llm(prompt, model, model_config)
            except (LLMRateLimitError, LLMAPIError, LLMContentFilterError, LLMInvalidResponseError) as e:
                logger.warning(f"LLM call failed for model '{model}': {e}")
                last_exception = e
                continue
        raise last_exception or LLMServiceError("LLM call failed for all models.")

    def _call_llm(self, prompt: str, model: str, model_config: dict) -> dict:
        retries = 0
        while retries <= self.max_retries:
            try:
                start_time = time.monotonic()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=model_config.get("temperature", 0.7),
                    max_tokens=model_config.get("max_tokens", 512),
                    top_p=model_config.get("top_p", 1.0),
                    frequency_penalty=model_config.get("frequency_penalty", 0.0),
                    presence_penalty=model_config.get("presence_penalty", 0.0),
                    timeout=self.timeout,
                )
                latency = time.monotonic() - start_time
                generation_time_ms = int(latency * 1000)

                content = response.choices[0].message.content.strip()
                usage = getattr(response, "usage", None)
                token_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

                logger.info(
                    f"LLM call success | model={model} | latency={latency:.2f}s | "
                    f"prompt_tokens={token_usage['prompt_tokens']} | "
                    f"completion_tokens={token_usage['completion_tokens']} | "
                    f"total_tokens={token_usage['total_tokens']}"
                )
                return {
                    "content": content,
                    "llm_model_used": model,
                    "generation_time_ms": generation_time_ms,
                    "token_usage": token_usage,
                }

            except RateLimitError as e:
                logger.warning(f"Rate limit error (429): {e}")
                if retries == self.max_retries:
                    msg = "Rate limit exceeded"
                    raise LLMRateLimitError(msg) from e
                self._backoff(retries)
                retries += 1
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                logger.warning(f"Transient API error: {e}")
                if retries == self.max_retries:
                    msg = "API connection or timeout error"
                    raise LLMAPIError(msg) from e
                self._backoff(retries)
                retries += 1
            except OpenAIError as e:
                # Content filtering or other OpenAI errors
                if "content_filter" in str(e).lower():
                    msg = "Content filtered by LLM provider"
                    raise LLMContentFilterError(msg) from e
                msg = f"OpenAI API error: {e}"
                raise LLMAPIError(msg) from e
            except Exception as e:
                logger.exception(f"Unexpected error from LLM: {e}")
                msg = f"Unexpected LLM error: {e}"
                raise LLMInvalidResponseError(msg) from e

        msg = "Max retries exceeded for LLM call."
        raise LLMServiceError(msg)

    def _backoff(self, retries: int):
        delay = self.backoff_factor * (2**retries)
        logger.info(f"Retrying after {delay:.2f}s (retry {retries + 1})")
        time.sleep(delay)

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        # Static pricing table (USD per 1K tokens). Extend as needed.
        pricing = {
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.001,
            "Meta-Llama-3-1-8B-Instruct-FP8": 0.0005,
            "Meta-Llama-3-70B-Instruct": 0.002,
            "DeepSeek-R1": 0.001,
            "DeepSeek-R1-Distill-Llama-70B": 0.001,
            "DeepSeek-R1-Distill-Qwen-14B": 0.001,
            "DeepSeek-R1-Distill-Qwen-32B": 0.001,
            "Meta-Llama-3-2-3B-Instruct": 0.0005,
            "Meta-Llama-4-Maverick-17B-128E-Instruct-FP8": 0.001,
        }
        price_per_1k = pricing.get(model, 0.001)
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        return (total_tokens / 1000.0) * price_per_1k


# Singleton instance
llm_service = LLMService()


def generate_summary(prompt: str, model_config: dict) -> dict:
    """Unified interface for LLM summary generation. Returns dict with content, model, timing, tokens."""
    return llm_service.generate_summary(prompt, model_config)
