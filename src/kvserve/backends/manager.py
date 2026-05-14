from __future__ import annotations

from kvserve.backends.base import Backend
from kvserve.backends.dev import DevBackend
from kvserve.backends.hf import TransformersBackend
from kvserve.backends.perkunas_backend import PerkunasBackend
from kvserve.backends.vllm_backend import VLLMBackend
from kvserve.models.registry import ModelRegistry
from kvserve.models.schemas import ModelSpec


class BackendManager:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._instances: dict[str, Backend] = {}

    def get(self, model: ModelSpec) -> Backend:
        existing = self._instances.get(model.model_id)
        if existing is not None:
            return existing
        if model.backend == "dev":
            backend: Backend = DevBackend(model)
        elif model.backend == "vllm":
            backend = VLLMBackend(model)
        elif model.backend == "transformers":
            backend = TransformersBackend(model)
        elif model.backend == "perkunas":
            backend = PerkunasBackend(model)
        else:
            raise ValueError(f"unsupported backend: {model.backend}")
        self._instances[model.model_id] = backend
        return backend
