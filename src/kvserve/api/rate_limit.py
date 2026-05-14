from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from kvserve.api.security import TenantContext, authenticate
from kvserve.config import Settings, get_settings


@dataclass(slots=True)
class Bucket:
    tokens: float
    updated_at: float


class InMemoryRateLimiter:
    def __init__(self, rpm: int):
        self.capacity = float(rpm)
        self.refill_per_second = float(rpm) / 60.0
        self.buckets: dict[str, Bucket] = defaultdict(lambda: Bucket(self.capacity, time.time()))
        self.lock = asyncio.Lock()

    async def allow(self, tenant_id: str, cost: float = 1.0) -> bool:
        async with self.lock:
            bucket = self.buckets[tenant_id]
            now = time.time()
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
            bucket.updated_at = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True
            return False


def get_rate_limiter(
    request: Request, settings: Settings = Depends(get_settings)
) -> InMemoryRateLimiter:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        limiter = InMemoryRateLimiter(settings.rate_limit_rpm)
        request.app.state.rate_limiter = limiter
    return limiter


async def rate_limited(
    tenant: TenantContext = Depends(authenticate),
    limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
) -> TenantContext:
    if not await limiter.allow(tenant.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"message": "rate limit exceeded", "type": "rate_limit_error"},
        )
    return tenant
