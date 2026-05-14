from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator

from kvserve.api.schemas import ChatCompletionRequest
from kvserve.backends.base import EmbeddingResult, GenerationChunk, GenerationResult, RerankResult
from kvserve.models.schemas import KVCompressionMode, ModelSpec


class VLLMBackend:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("vLLM backend requires vllm and transformers") from exc

        self.SamplingParams = SamplingParams
        model_name = spec.backend_config.get("model_name_or_path", spec.model_id)
        engine_kwargs = dict(spec.backend_config.get("engine_args", {}))
        if spec.kv_compression_mode == KVCompressionMode.FP8:
            engine_kwargs.setdefault("kv_cache_dtype", "fp8")
        engine_kwargs.setdefault("enable_prefix_caching", True)
        if spec.quantization_mode in {"fp8", "int8", "int4"}:
            engine_kwargs.setdefault("quantization", spec.quantization_mode.value)
        args = AsyncEngineArgs(model=model_name, **engine_kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        rendered = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    async def generate(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        text_parts: list[str] = []
        completion_tokens = 0
        finish_reason = "stop"
        async for chunk in self.generate_stream(prompt, request):
            text_parts.append(chunk.text)
            completion_tokens += chunk.token_count
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        return GenerationResult(
            text="".join(text_parts),
            prompt_tokens=self.token_count(prompt),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    async def generate_stream(
        self, prompt: str, request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        params = self.SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.output_token_limit,
            stop=request.stop,
        )
        request_id = f"cmpl-{uuid.uuid4().hex}"
        previous_text = ""
        async for output in self.engine.generate(prompt, params, request_id=request_id):
            candidate = output.outputs[0]
            current_text = candidate.text
            delta = current_text[len(previous_text) :]
            previous_text = current_text
            if delta:
                yield GenerationChunk(text=delta, token_count=max(1, len(delta.split())))
            if candidate.finish_reason:
                yield GenerationChunk(text="", token_count=0, finish_reason=candidate.finish_reason)
                return
            await asyncio.sleep(0)

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        raise RuntimeError("register embedding models with a dedicated embedding backend")

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        raise RuntimeError("register reranking models with a dedicated reranking backend")
