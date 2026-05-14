from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from kvserve.api.rate_limit import rate_limited
from kvserve.api.security import TenantContext
from kvserve.models.registry import ModelRegistry


class MCPHttpServer:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.router = APIRouter(prefix="/mcp", tags=["mcp"])
        self.router.add_api_route("", self.handle_json_rpc, methods=["POST"])
        self.router.add_api_route("/tools/list", self.tools_list, methods=["GET"])
        self.router.add_api_route("/tools/call", self.tools_call, methods=["POST"])
        self.router.add_api_route("/resources/list", self.resources_list, methods=["GET"])
        self.router.add_api_route("/resources/read", self.resources_read, methods=["POST"])
        self.router.add_api_route("/prompts/list", self.prompts_list, methods=["GET"])
        self.router.add_api_route("/prompts/get", self.prompts_get, methods=["POST"])

    async def handle_json_rpc(
        self, payload: dict[str, Any], request: Request, tenant: TenantContext = Depends(rate_limited)
    ) -> dict[str, Any]:
        method = payload.get("method")
        rpc_id = payload.get("id")
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {},
                    },
                    "serverInfo": {"name": "kvserve-mcp", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = await self.tools_list(tenant)
            elif method == "tools/call":
                result = await self.tools_call(payload.get("params", {}), request, tenant)
            elif method == "resources/list":
                result = await self.resources_list(tenant)
            elif method == "resources/read":
                result = await self.resources_read(payload.get("params", {}), request, tenant)
            elif method == "prompts/list":
                result = await self.prompts_list(tenant)
            elif method == "prompts/get":
                result = await self.prompts_get(payload.get("params", {}), tenant)
            else:
                return self._error(rpc_id, -32601, f"unknown MCP method: {method}")
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:
            return self._error(rpc_id, -32000, str(exc))

    async def tools_list(self, tenant: TenantContext = Depends(rate_limited)) -> dict[str, Any]:
        del tenant
        return {
            "tools": [
                {
                    "name": "models.list",
                    "description": "List tenant-visible model registry entries.",
                    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                },
                {
                    "name": "kv.stats",
                    "description": "Return current KV memory and prefix reuse counters.",
                    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            ]
        }

    async def tools_call(
        self,
        payload: dict[str, Any],
        request: Request,
        tenant: TenantContext = Depends(rate_limited),
    ) -> dict[str, Any]:
        name = payload.get("name")
        if name == "models.list":
            models = [model.model_dump(mode="json") for model in self.registry.list()]
            return {"content": [{"type": "json", "json": {"models": models}}], "isError": False}
        if name == "kv.stats":
            kv_control = request.app.state.kv_control
            stats = {
                "tenant_id": tenant.tenant_id,
                "gpu_bytes": kv_control.pager.tier_bytes("gpu"),
                "cpu_bytes": kv_control.pager.tier_bytes("cpu"),
                "nvme_bytes": kv_control.pager.tier_bytes("nvme"),
                "prefix": kv_control.prefix_index.stats(),
            }
            return {"content": [{"type": "json", "json": stats}], "isError": False}
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}

    async def resources_list(self, tenant: TenantContext = Depends(rate_limited)) -> dict[str, Any]:
        del tenant
        return {
            "resources": [
                {
                    "uri": "kvserve://models",
                    "name": "Model registry",
                    "mimeType": "application/json",
                },
                {
                    "uri": "kvserve://kv/policy",
                    "name": "Current KV policy state",
                    "mimeType": "application/json",
                },
            ]
        }

    async def resources_read(
        self,
        payload: dict[str, Any],
        request: Request,
        tenant: TenantContext = Depends(rate_limited),
    ) -> dict[str, Any]:
        uri = payload.get("uri")
        if uri == "kvserve://models":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": str([model.model_dump(mode="json") for model in self.registry.list()]),
                    }
                ]
            }
        if uri == "kvserve://kv/policy":
            kv_control = request.app.state.kv_control
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": str(
                            {
                                "tenant_id": tenant.tenant_id,
                                "prefix": kv_control.prefix_index.stats(),
                                "allocations": len(kv_control.allocations),
                            }
                        ),
                    }
                ]
            }
        return {"contents": []}

    async def prompts_list(self, tenant: TenantContext = Depends(rate_limited)) -> dict[str, Any]:
        del tenant
        return {
            "prompts": [
                {
                    "name": "kv_memory_first_system",
                    "description": "System prompt that biases model output toward concise, cache-friendly context.",
                    "arguments": [],
                }
            ]
        }

    async def prompts_get(
        self, payload: dict[str, Any], tenant: TenantContext = Depends(rate_limited)
    ) -> dict[str, Any]:
        del tenant
        if payload.get("name") != "kv_memory_first_system":
            return {"messages": []}
        return {
            "messages": [
                {
                    "role": "system",
                    "content": {
                        "type": "text",
                        "text": "Prefer concise answers and preserve stable prefixes for KV reuse.",
                    },
                }
            ]
        }

    def _error(self, rpc_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}
