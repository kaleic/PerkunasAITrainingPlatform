# Phased Implementation Plan

## Phase 1: Deployable KV-first Core

Included in this repository:

- OpenAI-compatible FastAPI gateway.
- SSE streaming with final usage chunk.
- Bearer auth, tenant isolation, rate limiting, audit logging.
- Model registry with mandatory KV and quantization metadata.
- KV control plane with prefix reuse, compression, pruning, paging, and policy.
- Auto quantization planner.
- vLLM, Transformers, and deterministic dev backend adapters.
- MCP-over-HTTP tools, resources, and prompts.
- Prometheus metrics.

## Phase 2: Backend Hardening

- Add production embedding and reranking adapters for the selected model families.
- Add GPU integration tests on the target vLLM version.
- Add model-specific quality gates for each compression policy.
- Add Redis/Valkey rate limiter and distributed prefix metadata when scaling
  beyond one API process.

## Phase 3: Cluster Scheduling

- Add multi-GPU and multi-node routing.
- Use per-backend queue depth, KV pressure, and hardware profile for routing.
- Add warm model pools and rolling model version promotion.

## Phase 4: Advanced KV Research Path

- Replace NumPy validation codec with fused CUDA kernels for TurboQuant.
- Add per-layer adaptive bit width.
- Add attention-sink aware retention policies.
- Add continuous quality monitoring for compression decisions.
