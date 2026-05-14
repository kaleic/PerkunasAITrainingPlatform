# Risk Analysis

## KV Compression Quality Loss

Risk: low-bit KV reconstruction can perturb attention.

Mitigation:

- default selective retention protects recent and high-attention tokens;
- residual correction is available;
- quality gates compare compressed modes against BF16/FP16 baselines;
- policy can move to `quality_first` per model.

## Backend KV Access

Risk: some backends do not expose KV tensors for external compression.

Mitigation:

- orchestrator refuses unsupported optimized-KV combinations;
- vLLM FP8 KV is configured through native engine arguments;
- explicit `force_unoptimized_kv=true` is required for fallback paths.

## Tenant Leakage

Risk: prefix cache reuse could leak data across tenants.

Mitigation:

- prefix index keys include tenant id;
- auth maps one token to one tenant;
- no cross-tenant lookup path exists;
- audit logs include tenant id.

## Paging Latency

Risk: CPU/NVMe paging can harm tail latency.

Mitigation:

- policy prefetches ahead of token position;
- GPU demotion uses LRU;
- NVMe is only selected under extreme pressure.

## Quantization Mismatch

Risk: selected quantization may be unsupported by hardware or backend.

Mitigation:

- hardware detection validates FP8/BF16/INT4 support;
- planner falls back safely;
- registry carries hardware constraints.

## Operational Scale

Risk: in-memory rate limit and prefix metadata are single-process.

Mitigation:

- Redis/Valkey can replace in-memory counters for multi-process deployment;
- single-process behavior remains deterministic and deployable.
