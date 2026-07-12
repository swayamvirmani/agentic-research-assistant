"""
LLM abstraction layer supporting OpenAI GPT-4o and Anthropic Claude.
Provides consistent streaming and non-streaming interfaces.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import structlog

from utils.config import settings

logger = structlog.get_logger()


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMResponse:
    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_ms = latency_ms


class OpenAILLM:
    """
    OpenAI GPT-4o wrapper with:
    - Structured output support
    - Streaming
    - Automatic retry
    - Token tracking
    """

    def __init__(
        self,
        model: str = settings.openai_model,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=settings.openai_api_key)

    def chat(
        self,
        messages: list[Message | dict],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Non-streaming completion."""
        msg_dicts = []
        if system:
            msg_dicts.append({"role": "system", "content": system})

        for m in messages:
            msg_dicts.append(m.to_dict() if isinstance(m, Message) else m)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msg_dicts,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                latency = (time.perf_counter() - start) * 1000
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=round(latency, 2),
                )
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("llm_retry", attempt=attempt, error=str(e))
                time.sleep(2 ** attempt)

    def stream(
        self,
        messages: list[Message | dict],
        system: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Streaming completion — yields token chunks."""
        msg_dicts = []
        if system:
            msg_dicts.append({"role": "system", "content": system})
        for m in messages:
            msg_dicts.append(m.to_dict() if isinstance(m, Message) else m)

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=msg_dicts,
            temperature=temperature if temperature is not None else self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def json_chat(
        self,
        messages: list[Message | dict],
        system: str | None = None,
    ) -> dict:
        """Force JSON output and parse it."""
        import json

        response = self.chat(
            messages,
            system=system,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.error("json_parse_failed", content=response.content[:200])
            return {}


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.1) -> OpenAILLM:
    return OpenAILLM(temperature=temperature)
