from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kvserve.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: str
    token_hash: str


bearer = HTTPBearer(auto_error=False)


async def authenticate(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> TenantContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "missing bearer token", "type": "authentication_error"},
        )
    token_map = settings.token_map()
    supplied = credentials.credentials
    for tenant_id, expected in token_map.items():
        if hmac.compare_digest(supplied, expected):
            context = TenantContext(tenant_id=tenant_id, token_hash=_hash_token(supplied))
            request.state.tenant = context
            return context
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"message": "invalid bearer token", "type": "authentication_error"},
    )


def _hash_token(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
