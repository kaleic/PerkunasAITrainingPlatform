from __future__ import annotations

from kvserve.kv.policy import KVPolicyEngine, KVPressure
from kvserve.kv.prefix import PrefixKVIndex
from kvserve.models.schemas import PolicyMode


def test_prefix_reuse_is_tenant_isolated() -> None:
    index = PrefixKVIndex(near_match_hamming=4)
    index.insert("tenant_a", "model", "System prompt and shared docs", "kv_a")
    assert index.lookup("tenant_a", "model", "System prompt and shared docs").kv_ref == "kv_a"
    assert index.lookup("tenant_b", "model", "System prompt and shared docs") is None


def test_memory_first_policy_is_aggressive() -> None:
    decision = KVPolicyEngine().decide(
        PolicyMode.MEMORY_FIRST,
        KVPressure(
            gpu_used_bytes=98,
            gpu_budget_bytes=100,
            active_requests=32,
            context_tokens=32768,
        ),
        supports_fp8_kv=True,
    )
    assert decision.bit_width <= 3
    assert decision.prune_fraction > 0
    assert decision.page_to_cpu is True
