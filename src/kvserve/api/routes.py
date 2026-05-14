from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from kvserve.api.rate_limit import rate_limited
from kvserve.api.schemas import (
    ChatCompletionRequest,
    EmbeddingsRequest,
    ListModelsResponse,
    ModelCard,
    RerankRequest,
)
from kvserve.api.security import TenantContext
from kvserve.models.schemas import TaskType
from kvserve.observability.metrics import render_metrics
from kvserve.orchestrator import InferenceOrchestrator


router = APIRouter()


def get_orchestrator(request: Request) -> InferenceOrchestrator:
    return request.app.state.orchestrator


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(render_metrics(), media_type="text/plain; version=0.0.4")


@router.get("/v1/models", response_model=ListModelsResponse)
async def list_models(
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
) -> ListModelsResponse:
    del tenant
    now = int(time.time())
    cards = [
        ModelCard(id=model.model_id, created=now)
        for model in orchestrator.registry.list()
        if model.task_type in {TaskType.GENERATE, TaskType.EMBED, TaskType.RERANK}
    ]
    return ListModelsResponse(data=cards)


@router.post("/v1/chat/completions")
async def chat_completions(
    request_body: ChatCompletionRequest,
    raw_request: Request,
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    if request_body.stream:
        return StreamingResponse(
            orchestrator.stream_chat(tenant, request_body, raw_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await orchestrator.complete_chat(tenant, request_body)


@router.post("/v1/chat")
async def chat_alias(
    request_body: ChatCompletionRequest,
    raw_request: Request,
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    return await chat_completions(request_body, raw_request, tenant, orchestrator)


@router.post("/v1/embeddings")
async def embeddings(
    request_body: EmbeddingsRequest,
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    return await orchestrator.embeddings(tenant, request_body)


@router.post("/v1/rerank")
async def rerank(
    request_body: RerankRequest,
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    return await orchestrator.rerank(tenant, request_body)


@router.post("/v1/reranking")
async def reranking_alias(
    request_body: RerankRequest,
    tenant: TenantContext = Depends(rate_limited),
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    return await rerank(request_body, tenant, orchestrator)
