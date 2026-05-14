from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Any, AsyncIterator

from kvserve.api.schemas import ChatCompletionRequest
from kvserve.backends.base import EmbeddingResult, GenerationChunk, GenerationResult, RerankResult
from kvserve.models.schemas import ModelSpec


class DevBackend:
    """Deterministic local backend for smoke tests and control-plane validation."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.embedding_dimension = int(spec.backend_config.get("dimension", 384))

    def token_count(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\S+", text))

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if isinstance(content, list):
                text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            else:
                text = "" if content is None else str(content)
            rendered.append(f"<|{role}|>\n{text}")
        rendered.append("<|assistant|>\n")
        return "\n".join(rendered)

    async def generate(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        text = self._completion_text(prompt, request)
        limit = request.output_token_limit
        words = text.split()
        if len(words) > limit:
            text = " ".join(words[:limit])
            finish_reason = "length"
        else:
            finish_reason = "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=self.token_count(prompt),
            completion_tokens=self.token_count(text),
            finish_reason=finish_reason,
        )

    async def generate_stream(
        self, prompt: str, request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        result = await self.generate(prompt, request)
        tokens = result.text.split()
        for index, token in enumerate(tokens):
            suffix = " " if index < len(tokens) - 1 else ""
            yield GenerationChunk(text=token + suffix, token_count=1)
            await asyncio.sleep(0)
        yield GenerationChunk(text="", token_count=0, finish_reason=result.finish_reason)

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        dim = dimensions or self.embedding_dimension
        vectors = [self._hash_embedding(text, dim) for text in inputs]
        return EmbeddingResult(vectors=vectors, prompt_tokens=sum(self.token_count(x) for x in inputs))

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        query_terms = set(self._terms(query))
        scores: list[float] = []
        for document in documents:
            doc_terms = set(self._terms(document))
            if not query_terms or not doc_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & doc_terms)
            denom = math.sqrt(len(query_terms) * len(doc_terms))
            scores.append(float(overlap / denom))
        return RerankResult(scores=scores, prompt_tokens=self.token_count(query) + sum(map(self.token_count, documents)))

    def _completion_text(self, prompt: str, request: ChatCompletionRequest) -> str:
        last_user = ""
        for message in reversed(request.messages):
            if message.role == "user":
                last_user = str(message.content or "")
                break
        if request.response_format and request.response_format.type in {"json_object", "json_schema"}:
            payload: dict[str, Any] = {
                "answer": last_user[: max(1, min(240, len(last_user)))],
                "model": self.spec.model_id,
            }
            return json.dumps(payload, separators=(",", ":"))
        if request.tools and request.tool_choice not in (None, "none"):
            tool = request.tools[0].function
            payload = {
                "tool_calls": [
                    {
                        "id": "call_dev_0001",
                        "type": "function",
                        "function": {"name": tool.name, "arguments": "{}"},
                    }
                ]
            }
            return json.dumps(payload, separators=(",", ":"))
        return f"KV-optimized response: {last_user}".strip()

    def _hash_embedding(self, text: str, dimensions: int) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < dimensions:
            digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == dimensions:
                    break
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def _terms(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
