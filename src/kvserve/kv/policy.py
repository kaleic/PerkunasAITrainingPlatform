from __future__ import annotations

from dataclasses import dataclass

from kvserve.kv.compression import SelectiveCompressionConfig
from kvserve.models.schemas import KVCompressionMode, PolicyMode


@dataclass(frozen=True, slots=True)
class KVPressure:
    gpu_used_bytes: int
    gpu_budget_bytes: int
    cpu_used_bytes: int = 0
    cpu_budget_bytes: int = 0
    active_requests: int = 0
    context_tokens: int = 0
    target_latency_ms: int = 250

    @property
    def gpu_pressure(self) -> float:
        if self.gpu_budget_bytes <= 0:
            return 0.0
        return min(2.0, self.gpu_used_bytes / self.gpu_budget_bytes)


@dataclass(frozen=True, slots=True)
class KVPolicyDecision:
    mode: KVCompressionMode
    bit_width: int
    residual_ratio: float
    recent_window: int
    high_attention_fraction: float
    prune_fraction: float
    page_to_cpu: bool
    page_to_nvme: bool
    prefetch_tokens: int
    reason: str

    def selective_config(self) -> SelectiveCompressionConfig:
        return SelectiveCompressionConfig(
            recent_window=self.recent_window,
            high_attention_fraction=self.high_attention_fraction,
            cold_bit_width=self.bit_width,
            hot_mode=KVCompressionMode.FP8,
            residual_ratio=self.residual_ratio,
        )


class KVPolicyEngine:
    def decide(self, mode: PolicyMode, pressure: KVPressure, supports_fp8_kv: bool) -> KVPolicyDecision:
        gpu_pressure = pressure.gpu_pressure
        long_context = pressure.context_tokens >= 8192
        high_concurrency = pressure.active_requests >= 16

        if mode == PolicyMode.QUALITY_FIRST and gpu_pressure < 0.72 and not high_concurrency:
            return KVPolicyDecision(
                mode=KVCompressionMode.FP8 if supports_fp8_kv else KVCompressionMode.TURBOQUANT,
                bit_width=4,
                residual_ratio=0.004,
                recent_window=1024,
                high_attention_fraction=0.20,
                prune_fraction=0.0,
                page_to_cpu=False,
                page_to_nvme=False,
                prefetch_tokens=2048,
                reason="quality_first under available GPU memory",
            )

        if mode == PolicyMode.MEMORY_FIRST or gpu_pressure >= 0.90:
            return KVPolicyDecision(
                mode=KVCompressionMode.TURBOQUANT,
                bit_width=2 if gpu_pressure >= 0.98 else 3,
                residual_ratio=0.001,
                recent_window=256,
                high_attention_fraction=0.08,
                prune_fraction=0.20 if gpu_pressure >= 0.98 else 0.10,
                page_to_cpu=True,
                page_to_nvme=gpu_pressure >= 0.98,
                prefetch_tokens=512,
                reason="memory pressure requires aggressive KV compression and pruning",
            )

        if mode == PolicyMode.THROUGHPUT_FIRST or high_concurrency:
            return KVPolicyDecision(
                mode=KVCompressionMode.TURBOQUANT,
                bit_width=3,
                residual_ratio=0.001,
                recent_window=384,
                high_attention_fraction=0.10,
                prune_fraction=0.05 if long_context else 0.0,
                page_to_cpu=gpu_pressure >= 0.80,
                page_to_nvme=False,
                prefetch_tokens=768,
                reason="throughput/concurrency favors smaller KV and predictable paging",
            )

        return KVPolicyDecision(
            mode=KVCompressionMode.TURBOQUANT if long_context else KVCompressionMode.FP8,
            bit_width=4 if gpu_pressure < 0.75 else 3,
            residual_ratio=0.002,
            recent_window=512 if long_context else 768,
            high_attention_fraction=0.12,
            prune_fraction=0.05 if gpu_pressure >= 0.82 else 0.0,
            page_to_cpu=gpu_pressure >= 0.86,
            page_to_nvme=False,
            prefetch_tokens=1024,
            reason="balanced policy",
        )
