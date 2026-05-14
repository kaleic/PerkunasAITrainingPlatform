from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(StrEnum):
    GENERATE = "generate"
    EMBED = "embed"
    RERANK = "rerank"


class QuantizationMode(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    AUTO = "auto"


class KVCompressionMode(StrEnum):
    STANDARD = "standard"
    FP8 = "fp8"
    TURBOQUANT = "turboquant"


class PolicyMode(StrEnum):
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    MEMORY_FIRST = "memory_first"
    THROUGHPUT_FIRST = "throughput_first"


class HardwareConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_gpu_memory_gb: float = 0
    supports_fp8_kv: bool = False
    requires_cuda: bool = False
    allowed_gpu_names: list[str] | None = None


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_id: str
    task_type: TaskType
    backend: Literal["dev", "vllm", "transformers", "perkunas"] = "dev"
    backend_config: dict[str, Any] = Field(default_factory=dict)
    quantization_mode: QuantizationMode
    kv_compression_mode: KVCompressionMode
    kv_required: bool = True
    max_context: int
    streaming_supported: bool
    chat_template_required: bool
    hardware_constraints: HardwareConstraints = Field(default_factory=HardwareConstraints)
    policy_mode: PolicyMode = PolicyMode.BALANCED

    @field_validator("kv_required")
    @classmethod
    def generation_models_require_kv(cls, value: bool, info):
        task_type = info.data.get("task_type")
        if task_type == TaskType.GENERATE and not value:
            raise ValueError("generation models must set kv_required=true")
        return value


class ModelRegistryDocument(BaseModel):
    models: list[ModelSpec]
