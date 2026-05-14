from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from kvserve.api.schemas import ChatCompletionRequest
from kvserve.models.schemas import ModelSpec


@dataclass(slots=True)
class GenerationChunk:
    text: str
    token_count: int = 1
    finish_reason: str | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    prompt_tokens: int


@dataclass(slots=True)
class RerankResult:
    scores: list[float]
    prompt_tokens: int


class Backend(Protocol):
    spec: ModelSpec

    async def generate(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        ...

    async def generate_stream(
        self, prompt: str, request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        ...

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        ...

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        ...

    def token_count(self, text: str) -> int:
        ...

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        ...
