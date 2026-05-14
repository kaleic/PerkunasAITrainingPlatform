from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from kvserve.api.middleware import audit_middleware, configure_logging
from kvserve.api.rate_limit import InMemoryRateLimiter
from kvserve.api.routes import router
from kvserve.backends.manager import BackendManager
from kvserve.config import get_settings
from kvserve.gpu import configure_cuda_runtime
from kvserve.kv.control_plane import KVControlPlane
from kvserve.kv.paging import KVPager
from kvserve.kv.policy import KVPolicyEngine
from kvserve.kv.prefix import PrefixKVIndex
from kvserve.mcp.server import MCPHttpServer
from kvserve.models.registry import ModelRegistry
from kvserve.observability.metrics import create_metrics
from kvserve.orchestrator import InferenceOrchestrator
from kvserve.quantization.planner import QuantizationPlanner


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    gpu_diagnostics = configure_cuda_runtime(settings)
    metrics = create_metrics()
    registry = ModelRegistry(settings.model_registry)
    backend_manager = BackendManager(registry)
    kv_control = KVControlPlane(
        pager=KVPager(settings.nvme_cache_dir),
        policy_engine=KVPolicyEngine(),
        prefix_index=PrefixKVIndex(settings.prefix_near_match_hamming),
        metrics=metrics,
    )
    quantization_planner = QuantizationPlanner()
    orchestrator = InferenceOrchestrator(
        registry=registry,
        backend_manager=backend_manager,
        kv_control=kv_control,
        metrics=metrics,
        quantization_planner=quantization_planner,
    )

    app = FastAPI(
        title="kvserve",
        version="0.1.0",
        description="KV-memory-optimized OpenAI-compatible AI inference platform",
    )
    app.state.settings = settings
    app.state.gpu_diagnostics = gpu_diagnostics
    if "warm_cuda_tensor" in gpu_diagnostics:
        app.state.warm_cuda_tensor = gpu_diagnostics["warm_cuda_tensor"]
    app.state.metrics = metrics
    app.state.registry = registry
    app.state.backend_manager = backend_manager
    app.state.kv_control = kv_control
    app.state.orchestrator = orchestrator
    app.state.rate_limiter = InMemoryRateLimiter(settings.rate_limit_rpm)

    app.middleware("http")(audit_middleware(metrics))
    app.exception_handler(HTTPException)(openai_http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.include_router(router)
    app.include_router(MCPHttpServer(registry).router)
    FastAPIInstrumentor.instrument_app(app)
    return app


async def openai_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    del request
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail:
        body = {"error": {"message": detail["message"], "type": detail.get("type", "api_error")}}
    else:
        body = {"error": {"message": str(detail), "type": "api_error"}}
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": str(exc.errors()),
                "type": "invalid_request_error",
            }
        },
    )
