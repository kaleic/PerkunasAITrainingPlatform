from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kvserve.models.schemas import ModelSpec, PolicyMode, QuantizationMode


class HardwareKind(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    kind: HardwareKind
    gpu_name: str | None
    gpu_memory_gb: float
    compute_capability: tuple[int, int] | None
    supports_bf16: bool
    supports_fp8: bool
    supports_int4: bool
    device_count: int


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    model_id: str
    selected: QuantizationMode
    weight_dtype: str
    load_format: str
    kv_cache_dtype: str
    online_quantization: bool
    prequantized: bool
    reason: str


class QuantizationPlanner:
    def detect_hardware(self) -> HardwareProfile:
        try:
            import torch
        except ImportError:
            return HardwareProfile(
                kind=HardwareKind.CPU,
                gpu_name=None,
                gpu_memory_gb=0.0,
                compute_capability=None,
                supports_bf16=False,
                supports_fp8=False,
                supports_int4=False,
                device_count=0,
            )

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            capability = torch.cuda.get_device_capability(0)
            supports_bf16 = bool(torch.cuda.is_bf16_supported())
            supports_fp8 = capability >= (8, 9)
            return HardwareProfile(
                kind=HardwareKind.CUDA,
                gpu_name=props.name,
                gpu_memory_gb=props.total_memory / 1024**3,
                compute_capability=capability,
                supports_bf16=supports_bf16,
                supports_fp8=supports_fp8,
                supports_int4=True,
                device_count=torch.cuda.device_count(),
            )
        if getattr(torch.version, "hip", None):
            return HardwareProfile(
                kind=HardwareKind.ROCM,
                gpu_name=platform.processor() or "rocm",
                gpu_memory_gb=0.0,
                compute_capability=None,
                supports_bf16=True,
                supports_fp8=False,
                supports_int4=True,
                device_count=1,
            )
        return HardwareProfile(
            kind=HardwareKind.CPU,
            gpu_name=None,
            gpu_memory_gb=0.0,
            compute_capability=None,
            supports_bf16=False,
            supports_fp8=False,
            supports_int4=False,
            device_count=0,
        )

    def plan(
        self,
        model: ModelSpec,
        policy_mode: PolicyMode | None = None,
        hardware: HardwareProfile | None = None,
    ) -> QuantizationPlan:
        hardware = hardware or self.detect_hardware()
        policy = policy_mode or model.policy_mode
        requested = model.quantization_mode
        prequantized = bool(model.backend_config.get("prequantized", False))

        if requested != QuantizationMode.AUTO:
            selected = self._validate_requested(requested, hardware)
            return self._build_plan(model, selected, hardware, prequantized, "registry requested mode")

        if policy == PolicyMode.QUALITY_FIRST:
            if hardware.supports_bf16:
                selected = QuantizationMode.BF16
            elif hardware.kind == HardwareKind.CUDA:
                selected = QuantizationMode.FP16
            else:
                selected = QuantizationMode.INT8
            return self._build_plan(model, selected, hardware, prequantized, "quality_first auto policy")

        if policy == PolicyMode.MEMORY_FIRST:
            selected = QuantizationMode.INT4 if hardware.supports_int4 else QuantizationMode.INT8
            return self._build_plan(model, selected, hardware, prequantized, "memory_first auto policy")

        if policy == PolicyMode.THROUGHPUT_FIRST:
            if hardware.supports_fp8:
                selected = QuantizationMode.FP8
            elif hardware.supports_int4:
                selected = QuantizationMode.INT4
            else:
                selected = QuantizationMode.INT8
            return self._build_plan(model, selected, hardware, prequantized, "throughput_first auto policy")

        if hardware.supports_fp8 and hardware.gpu_memory_gb >= model.hardware_constraints.min_gpu_memory_gb:
            selected = QuantizationMode.FP8
        elif hardware.supports_int4 and hardware.gpu_memory_gb < max(1, model.hardware_constraints.min_gpu_memory_gb * 1.25):
            selected = QuantizationMode.INT4
        elif hardware.supports_bf16:
            selected = QuantizationMode.BF16
        elif hardware.kind == HardwareKind.CUDA:
            selected = QuantizationMode.FP16
        else:
            selected = QuantizationMode.INT8
        return self._build_plan(model, selected, hardware, prequantized, "balanced auto policy")

    def _validate_requested(
        self, requested: QuantizationMode, hardware: HardwareProfile
    ) -> QuantizationMode:
        if requested == QuantizationMode.FP8 and not hardware.supports_fp8:
            return QuantizationMode.INT8 if hardware.kind == HardwareKind.CPU else QuantizationMode.FP16
        if requested == QuantizationMode.INT4 and not hardware.supports_int4:
            return QuantizationMode.INT8
        if requested == QuantizationMode.BF16 and not hardware.supports_bf16:
            return QuantizationMode.FP16 if hardware.kind == HardwareKind.CUDA else QuantizationMode.INT8
        return requested

    def _build_plan(
        self,
        model: ModelSpec,
        selected: QuantizationMode,
        hardware: HardwareProfile,
        prequantized: bool,
        reason: str,
    ) -> QuantizationPlan:
        if selected == QuantizationMode.BF16:
            return QuantizationPlan(
                model.model_id, selected, "bfloat16", "safetensors", "auto", False, prequantized, reason
            )
        if selected == QuantizationMode.FP16:
            return QuantizationPlan(
                model.model_id, selected, "float16", "safetensors", "auto", False, prequantized, reason
            )
        if selected == QuantizationMode.FP8:
            return QuantizationPlan(
                model.model_id,
                selected,
                "float8",
                "fp8" if prequantized else "safetensors",
                "fp8",
                not prequantized,
                prequantized,
                reason,
            )
        if selected == QuantizationMode.INT4:
            return QuantizationPlan(
                model.model_id,
                selected,
                "int4_awq",
                "awq" if prequantized else "safetensors",
                "auto",
                not prequantized,
                prequantized,
                reason,
            )
        return QuantizationPlan(
            model.model_id,
            QuantizationMode.INT8,
            "int8",
            "bitsandbytes" if not prequantized else "int8",
            "auto",
            not prequantized,
            prequantized,
            reason,
        )

    def as_backend_config(self, plan: QuantizationPlan) -> dict[str, Any]:
        return {
            "quantization_mode": plan.selected.value,
            "weight_dtype": plan.weight_dtype,
            "load_format": plan.load_format,
            "kv_cache_dtype": plan.kv_cache_dtype,
            "online_quantization": plan.online_quantization,
            "prequantized": plan.prequantized,
        }
