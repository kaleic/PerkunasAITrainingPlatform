from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, AsyncIterator

import numpy as np
from fastapi import HTTPException, Request, status

from kvserve.api.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatMessage,
    ChatChoice,
    ChatChunkChoice,
    CompletionUsage,
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from kvserve.api.security import TenantContext
from kvserve.backends.manager import BackendManager
from kvserve.kv.control_plane import KVAdmission, KVControlPlane
from kvserve.models.registry import ModelRegistry
from kvserve.models.schemas import KVCompressionMode, ModelSpec, TaskType
from kvserve.observability.metrics import Metrics
from kvserve.quantization.planner import QuantizationPlanner


class InferenceOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        backend_manager: BackendManager,
        kv_control: KVControlPlane,
        metrics: Metrics,
        quantization_planner: QuantizationPlanner,
    ):
        self.registry = registry
        self.backend_manager = backend_manager
        self.kv_control = kv_control
        self.metrics = metrics
        self.quantization_planner = quantization_planner
        self.active_requests = 0

    async def complete_chat(
        self, tenant: TenantContext, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        model = self._model_for(request.model, TaskType.GENERATE)
        self._enforce_kv_contract(model)
        backend = self.backend_manager.get(model)
        prompt = self._render_prompt(backend, request)
        admission = self._admit(tenant, model, prompt, backend.token_count(prompt))
        self._materialize_control_plane_kv_if_supported(model, admission, prompt)
        self.active_requests += 1
        try:
            result = await backend.generate(prompt, request)
        finally:
            self.active_requests -= 1
            self.kv_control.record_prefix(tenant.tenant_id, model.model_id, prompt, admission.allocation)
            self.kv_control.release(admission.allocation.allocation_id)
        message = self._message_from_generation(result.text, request)
        usage = CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )
        self.metrics.tokens_generated.labels(model_id=model.model_id).inc(result.completion_tokens)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model.model_id,
            choices=[ChatChoice(index=0, message=message, finish_reason=result.finish_reason)],
            usage=usage,
        )

    async def stream_chat(
        self, tenant: TenantContext, request: ChatCompletionRequest, raw_request: Request
    ) -> AsyncIterator[str]:
        model = self._model_for(request.model, TaskType.GENERATE)
        self._enforce_kv_contract(model)
        if not model.streaming_supported:
            raise HTTPException(status_code=400, detail={"message": "model does not support streaming"})
        backend = self.backend_manager.get(model)
        prompt = self._render_prompt(backend, request)
        admission = self._admit(tenant, model, prompt, backend.token_count(prompt))
        self._materialize_control_plane_kv_if_supported(model, admission, prompt)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_tokens = backend.token_count(prompt)
        completion_tokens = 0

        first = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model.model_id,
            choices=[ChatChunkChoice(index=0, delta=ChatDelta(role="assistant"))],
        )
        yield sse_event(first.model_dump_json(exclude_none=True))

        self.active_requests += 1
        finish_reason = "stop"
        try:
            async for delta in backend.generate_stream(prompt, request):
                if await raw_request.is_disconnected():
                    finish_reason = "cancelled"
                    return
                completion_tokens += delta.token_count
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                    break
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model.model_id,
                    choices=[
                        ChatChunkChoice(index=0, delta=ChatDelta(content=delta.text), finish_reason=None)
                    ],
                )
                yield sse_event(chunk.model_dump_json(exclude_none=True))
        finally:
            self.active_requests -= 1
            self.metrics.tokens_generated.labels(model_id=model.model_id).inc(completion_tokens)
            self.kv_control.record_prefix(tenant.tenant_id, model.model_id, prompt, admission.allocation)
            self.kv_control.release(admission.allocation.allocation_id)

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        final = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model.model_id,
            choices=[ChatChunkChoice(index=0, delta=ChatDelta(), finish_reason=finish_reason)],
            usage=usage,
        )
        yield sse_event(final.model_dump_json(exclude_none=True))
        yield sse_event("[DONE]")

    async def embeddings(
        self, tenant: TenantContext, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        del tenant
        model = self._model_for(request.model, TaskType.EMBED)
        backend = self.backend_manager.get(model)
        inputs = normalize_embedding_inputs(request.input)
        result = await backend.embed(inputs, request.dimensions)
        objects: list[EmbeddingObject] = []
        for index, vector in enumerate(result.vectors):
            embedding: list[float] | str
            if request.encoding_format == "base64":
                embedding = base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")
            else:
                embedding = vector
            objects.append(EmbeddingObject(embedding=embedding, index=index))
        usage = CompletionUsage(prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens)
        return EmbeddingsResponse(data=objects, model=model.model_id, usage=usage)

    async def rerank(self, tenant: TenantContext, request: RerankRequest) -> RerankResponse:
        del tenant
        model = self._model_for(request.model, TaskType.RERANK)
        backend = self.backend_manager.get(model)
        result = await backend.rerank(request.query, request.documents)
        ranked = sorted(enumerate(result.scores), key=lambda item: item[1], reverse=True)
        if request.top_n is not None:
            ranked = ranked[: request.top_n]
        objects = [
            RerankResult(
                index=index,
                relevance_score=float(score),
                document=request.documents[index] if request.return_documents else None,
            )
            for index, score in ranked
        ]
        usage = CompletionUsage(prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens)
        return RerankResponse(
            id=f"rerank-{uuid.uuid4().hex}", model=model.model_id, results=objects, usage=usage
        )

    def _model_for(self, model_id: str, task_type: TaskType) -> ModelSpec:
        try:
            model = self.registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": str(exc), "type": "not_found"}) from exc
        if model.task_type != task_type:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"model {model_id} is registered for {model.task_type}, not {task_type}",
                    "type": "invalid_request_error",
                },
            )
        return model

    def _enforce_kv_contract(self, model: ModelSpec) -> None:
        if not model.kv_required:
            return
        if model.backend == "dev":
            return
        if model.backend == "vllm" and model.kv_compression_mode == KVCompressionMode.FP8:
            return
        if model.backend_config.get("force_unoptimized_kv", False):
            return
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": (
                    f"model {model.model_id} backend={model.backend} cannot prove optimized KV for "
                    f"{model.kv_compression_mode}; set force_unoptimized_kv=true only for explicit fallback"
                ),
                "type": "kv_policy_error",
            },
        )

    def _admit(
        self, tenant: TenantContext, model: ModelSpec, prompt: str, context_tokens: int
    ) -> KVAdmission:
        admission = self.kv_control.admit(
            tenant.tenant_id,
            model,
            prefix_text=prompt,
            context_tokens=context_tokens,
            active_requests=self.active_requests,
        )
        plan = self.quantization_planner.plan(model)
        self.metrics.quantization_mode_active.labels(
            model_id=model.model_id, mode=plan.selected.value
        ).set(1)
        return admission

    def _render_prompt(self, backend: Any, request: ChatCompletionRequest) -> str:
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        control_parts: list[str] = []
        if request.tools:
            control_parts.append(
                "Available tools must be called with JSON: "
                + json.dumps([tool.model_dump(exclude_none=True) for tool in request.tools])
            )
        if request.response_format and request.response_format.type in {"json_object", "json_schema"}:
            control_parts.append(
                "Respond with JSON that conforms to: "
                + json.dumps(request.response_format.model_dump(exclude_none=True))
            )
        if control_parts:
            messages.insert(0, {"role": "system", "content": "\n".join(control_parts)})
        return backend.apply_chat_template(messages)

    def _message_from_generation(self, text: str, request: ChatCompletionRequest) -> ChatMessage:
        if request.tools:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
                return ChatMessage(role="assistant", content=None, tool_calls=payload["tool_calls"])
        if request.response_format and request.response_format.type in {"json_object", "json_schema"}:
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "backend returned invalid JSON for structured output request",
                        "type": "backend_error",
                    },
                ) from exc
        return ChatMessage(role="assistant", content=text)

    def _materialize_control_plane_kv_if_supported(
        self, model: ModelSpec, admission: KVAdmission, prompt: str
    ) -> None:
        if model.backend != "dev":
            return
        tokens = max(1, min(admission.allocation.token_count, model.max_context))
        key, value, attention = synthetic_kv(prompt, tokens=tokens)
        self.kv_control.store_kv(admission.allocation, key, value, admission.policy, attention)


def synthetic_kv(prompt: str, tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = int.from_bytes(__import__("hashlib").sha256(prompt.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    layers, heads, dim = 2, 4, 32
    key = rng.standard_normal((layers, heads, tokens, dim), dtype=np.float32)
    value = rng.standard_normal((layers, heads, tokens, dim), dtype=np.float32)
    position = np.linspace(0.0, 1.0, tokens, dtype=np.float32)
    energy = np.mean(np.abs(key) + np.abs(value), axis=(0, 1, 3))
    attention = 0.70 * position + 0.30 * (energy / (np.max(energy) or 1.0))
    return key, value, attention.astype(np.float32)


def normalize_embedding_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, int) for item in value):
            return [" ".join(str(item) for item in value)]
        if all(isinstance(item, list) for item in value):
            return [" ".join(str(token) for token in item) for item in value]
        return [str(item) for item in value]
    return [str(value)]


def sse_event(data: str) -> str:
    return f"data: {data}\n\n"
