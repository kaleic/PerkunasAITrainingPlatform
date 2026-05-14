from __future__ import annotations

from kvserve.models.schemas import (
    HardwareConstraints,
    KVCompressionMode,
    ModelSpec,
    PolicyMode,
    QuantizationMode,
    TaskType,
)
from kvserve.quantization.planner import HardwareKind, HardwareProfile, QuantizationPlanner


def test_auto_quantization_selects_fp8_on_supported_hardware() -> None:
    model = ModelSpec(
        model_id="m",
        task_type=TaskType.GENERATE,
        quantization_mode=QuantizationMode.AUTO,
        kv_compression_mode=KVCompressionMode.FP8,
        max_context=8192,
        streaming_supported=True,
        chat_template_required=True,
        hardware_constraints=HardwareConstraints(min_gpu_memory_gb=8, supports_fp8_kv=True),
        policy_mode=PolicyMode.BALANCED,
    )
    hardware = HardwareProfile(
        kind=HardwareKind.CUDA,
        gpu_name="test",
        gpu_memory_gb=24,
        compute_capability=(9, 0),
        supports_bf16=True,
        supports_fp8=True,
        supports_int4=True,
        device_count=1,
    )
    plan = QuantizationPlanner().plan(model, hardware=hardware)
    assert plan.selected == QuantizationMode.FP8
    assert plan.kv_cache_dtype == "fp8"
