from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from kvserve.kv.compression import SelectiveCompressedTensor, SelectiveKVCompressor
from kvserve.kv.paging import KVPager, KVTier
from kvserve.kv.policy import KVPolicyDecision, KVPolicyEngine, KVPressure
from kvserve.kv.prefix import PrefixHandle, PrefixKVIndex
from kvserve.kv.pruning import prune_kv_tokens
from kvserve.models.schemas import ModelSpec
from kvserve.observability.metrics import Metrics


@dataclass(slots=True)
class KVAllocation:
    allocation_id: str
    tenant_id: str
    model_id: str
    request_id: str
    token_count: int
    page_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class KVAdmission:
    request_id: str
    policy: KVPolicyDecision
    prefix_hit: PrefixHandle | None
    allocation: KVAllocation


class KVControlPlane:
    def __init__(
        self,
        pager: KVPager,
        policy_engine: KVPolicyEngine,
        prefix_index: PrefixKVIndex,
        metrics: Metrics,
        gpu_budget_bytes: int = 24 * 1024**3,
        cpu_budget_bytes: int = 128 * 1024**3,
    ):
        self.pager = pager
        self.policy_engine = policy_engine
        self.prefix_index = prefix_index
        self.metrics = metrics
        self.gpu_budget_bytes = gpu_budget_bytes
        self.cpu_budget_bytes = cpu_budget_bytes
        self.allocations: dict[str, KVAllocation] = {}

    def admit(
        self,
        tenant_id: str,
        model: ModelSpec,
        prefix_text: str,
        context_tokens: int,
        active_requests: int,
    ) -> KVAdmission:
        request_id = f"req_{uuid.uuid4().hex}"
        pressure = KVPressure(
            gpu_used_bytes=self.pager.tier_bytes(KVTier.GPU),
            gpu_budget_bytes=self.gpu_budget_bytes,
            cpu_used_bytes=self.pager.tier_bytes(KVTier.CPU),
            cpu_budget_bytes=self.cpu_budget_bytes,
            active_requests=active_requests,
            context_tokens=context_tokens,
        )
        policy = self.policy_engine.decide(
            model.policy_mode, pressure, model.hardware_constraints.supports_fp8_kv
        )
        prefix_hit = self.prefix_index.lookup(tenant_id, model.model_id, prefix_text)
        if prefix_hit is not None:
            self.metrics.prefix_reuse_hits.inc()
        self.metrics.prefix_reuse_lookups.inc()
        prefix_stats = self.prefix_index.stats()
        lookups = float(prefix_stats.get("lookups", 0) or 0)
        hits = float(prefix_stats.get("hits", 0) or 0)
        self.metrics.kv_prefix_reuse_rate.set(hits / lookups if lookups else 0.0)
        self.metrics.active_policy.labels(model_id=model.model_id, policy=policy.reason).set(1)
        allocation = KVAllocation(
            allocation_id=f"kva_{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            model_id=model.model_id,
            request_id=request_id,
            token_count=context_tokens,
        )
        self.allocations[allocation.allocation_id] = allocation
        return KVAdmission(request_id=request_id, policy=policy, prefix_hit=prefix_hit, allocation=allocation)

    def store_kv(
        self,
        allocation: KVAllocation,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        decision: KVPolicyDecision,
        attention_scores: np.ndarray | None = None,
    ) -> list[str]:
        compressor = SelectiveKVCompressor(model_seed=_seed_for_model(allocation.model_id))
        config = decision.selective_config()
        pruned = prune_kv_tokens(
            key_tensor,
            value_tensor,
            decision.prune_fraction,
            attention_scores,
            decision.recent_window,
        )
        allocation.token_count = int(pruned.retained_indices.size)
        if pruned.pruned_tokens:
            self.metrics.kv_pruned_tokens.inc(pruned.pruned_tokens)
            if attention_scores is not None:
                attention_scores = np.asarray(attention_scores)[pruned.retained_indices]
        compressed_key = compressor.compress(pruned.key, config, attention_scores)
        compressed_value = compressor.compress(pruned.value, config, attention_scores)
        merged = merge_key_value_pages(compressed_key, compressed_value)
        tier = KVTier.CPU if decision.page_to_cpu else KVTier.GPU
        page = self.pager.put(
            allocation.tenant_id,
            allocation.model_id,
            allocation.request_id,
            merged,
            token_start=0,
            token_end=allocation.token_count,
            tier=tier,
        )
        allocation.page_ids.append(page.page_id)
        original_bytes = int(key_tensor.nbytes + value_tensor.nbytes)
        compressed_bytes = int(merged.actual_nbytes)
        if compressed_bytes > 0:
            self.metrics.kv_compression_ratio.labels(model_id=allocation.model_id).set(
                original_bytes / compressed_bytes
            )
        self._publish_tier_metrics()
        return allocation.page_ids

    def record_prefix(self, tenant_id: str, model_id: str, prefix_text: str, allocation: KVAllocation) -> None:
        kv_ref = allocation.page_ids[0] if allocation.page_ids else allocation.allocation_id
        self.prefix_index.insert(tenant_id, model_id, prefix_text, kv_ref)

    def release(self, allocation_id: str) -> None:
        allocation = self.allocations.pop(allocation_id, None)
        if allocation is None:
            return
        for page_id in allocation.page_ids:
            if page_id in self.pager.pages:
                self.pager.delete(page_id)
        self._publish_tier_metrics()

    def enforce_pressure(self, bytes_needed: int) -> None:
        evicted = self.pager.evict_lru(KVTier.GPU, bytes_needed)
        self.metrics.kv_evictions.inc(len(evicted))
        self._publish_tier_metrics()

    def _publish_tier_metrics(self) -> None:
        self.metrics.kv_memory_gpu_bytes.set(self.pager.tier_bytes(KVTier.GPU))
        self.metrics.kv_memory_cpu_bytes.set(self.pager.tier_bytes(KVTier.CPU))
        self.metrics.kv_memory_nvme_bytes.set(self.pager.tier_bytes(KVTier.NVME))


def merge_key_value_pages(
    key: SelectiveCompressedTensor, value: SelectiveCompressedTensor
) -> SelectiveCompressedTensor:
    segments = []
    for key_segment, value_segment in zip(key.segments, value.segments, strict=True):
        if not np.array_equal(key_segment.indices, value_segment.indices):
            raise ValueError("key/value selective segment indices differ")
        key_segment.tensor.metadata["kv_part"] = "key"
        value_segment.tensor.metadata["kv_part"] = "value"
        segments.append(key_segment)
        segments.append(value_segment)
    return SelectiveCompressedTensor(shape=key.shape, token_axis=key.token_axis, segments=segments)


def _seed_for_model(model_id: str) -> int:
    digest = hashlib.sha256(model_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)
