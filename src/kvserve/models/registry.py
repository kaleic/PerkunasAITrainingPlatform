from __future__ import annotations

import json
from pathlib import Path

from kvserve.models.schemas import ModelRegistryDocument, ModelSpec, TaskType


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._models: dict[str, ModelSpec] = {}
        self.reload()

    def reload(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        document = ModelRegistryDocument.model_validate(data)
        seen: set[str] = set()
        models: dict[str, ModelSpec] = {}
        for model in document.models:
            if model.model_id in seen:
                raise ValueError(f"duplicate model_id in registry: {model.model_id}")
            seen.add(model.model_id)
            models[model.model_id] = model
        self._models = models

    def list(self) -> list[ModelSpec]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model: {model_id}") from exc

    def first_for_task(self, task_type: TaskType) -> ModelSpec:
        for model in self._models.values():
            if model.task_type == task_type:
                return model
        raise KeyError(f"no model registered for task {task_type}")
