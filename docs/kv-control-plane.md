# KV Control Plane Design

The KV control plane owns KV admission, reuse, compression, pruning, paging, and
policy enforcement. It is implemented in `src/kvserve/kv`.

## Prefix KV Reuse

`PrefixKVIndex` stores entries under `(tenant_id, model_id, prefix_hash)`.
Identical prefixes use SHA-256 over normalized text. Near-identical prefixes use
64-bit SimHash with a configurable Hamming-distance threshold.

Isolation rule: the lookup key always includes tenant id. No candidate from one
tenant can be returned to another tenant.

Metrics:

- `kv_prefix_reuse_lookups_total`
- `kv_prefix_reuse_hits_total`
- `kv_prefix_reuse_rate`

## Compression Modes

### Standard

Stores KV as FP16 bytes. This is a fallback/storage baseline and is not selected
for generation unless policy or explicit registry settings permit it.

### FP8 KV

`FP8E4M3Codec` implements scaled E4M3 encode/decode in NumPy for validation and
CPU-side page storage. vLLM models with `kv_compression_mode=fp8` also set native
`kv_cache_dtype=fp8`.

### TurboQuant-style Advanced Compression

`TurboQuantCodec` implements:

- deterministic orthogonal rotation of the head dimension;
- symmetric 2-bit, 3-bit, or 4-bit group quantization;
- actual bit packing into byte payloads;
- per-row/per-group FP16 scales;
- optional top residual correction stored as sparse FP16 deltas;
- reversible reconstruction for attention.

The codec does not require retraining. Reconstruction error is bounded by the
selected bit width, group size, and residual ratio.

## Selective Compression

`SelectiveKVCompressor` divides token positions into:

- recent tier: preserved as FP16;
- hot tier: high-attention old tokens, stored as FP8 by default;
- cold tier: old low-attention tokens, stored with TurboQuant low-bit packing.

Recent tokens are always retained at the highest precision selected by policy.
Older low-attention tokens receive stronger compression.

## KV Pruning

`prune_kv_tokens` removes low-value old tokens under pressure. It always protects
the recent window and ranks older candidates by provided attention scores or
fallback KV energy. The policy controls prune fraction.

Metric:

- `kv_pruned_tokens_total`

## Paging

`KVPager` owns page residency:

- GPU: hot, active pages.
- CPU: warm demoted pages.
- NVMe: cold serialized pages.

Eviction is LRU by tier. GPU pressure demotes to CPU; CPU pressure demotes to
NVMe. Prefetch promotes pages that overlap the next token window.

Metrics:

- `kv_memory_gpu_bytes`
- `kv_memory_cpu_bytes`
- `kv_memory_nvme_bytes`
- `kv_evictions_total`

## Policy Engine

Inputs:

- GPU memory pressure;
- CPU memory pressure;
- active request count;
- context length;
- latency target;
- model FP8 support;
- selected policy mode.

Outputs:

- compression mode;
- low-bit width;
- residual ratio;
- recent-token window;
- high-attention fraction;
- prune fraction;
- CPU/NVMe paging behavior;
- prefetch window.

Policy modes:

- `quality_first`
- `balanced`
- `memory_first`
- `throughput_first`
