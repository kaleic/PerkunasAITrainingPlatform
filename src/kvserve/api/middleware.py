from __future__ import annotations

import json
import logging
import time
from typing import Callable

from fastapi import Request, Response

from kvserve.observability.metrics import Metrics


logger = logging.getLogger("kvserve.audit")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def audit_middleware(metrics: Metrics) -> Callable:
    async def middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.url.path
            metrics.request_latency.labels(route=route).observe(elapsed)
            metrics.requests_total.labels(route=route, status=status).inc()
            tenant = getattr(getattr(request, "state", None), "tenant", None)
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "method": request.method,
                        "path": route,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 3),
                        "tenant_id": getattr(tenant, "tenant_id", None),
                        "client": request.client.host if request.client else None,
                    },
                    separators=(",", ":"),
                )
            )

    return middleware
