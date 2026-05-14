# Benchmark Plan

## Goals

- Maximize tokens/sec per GPU.
- Minimize KV bytes per active token.
- Preserve answer quality within model-specific tolerance.
- Quantify latency impact of compression, pruning, and paging.

## Workloads

1. Short chat: 1k prompt, 256 output.
2. Long chat: 16k prompt, 512 output.
3. Retrieval chat: shared 8k system/retrieval prefix with many user tails.
4. Tool calling: JSON tool selection with constrained output.
5. Structured output: JSON schema response.
6. Embeddings: batch sizes 1, 16, 128.
7. Reranking: 10, 100, 1000 documents.

## Matrix

Run each workload across:

- policies: `quality_first`, `balanced`, `memory_first`, `throughput_first`;
- KV modes: FP8, TurboQuant 4-bit, TurboQuant 3-bit, TurboQuant 2-bit;
- concurrency: 1, 4, 16, 64, saturation;
- context length: 1k, 4k, 16k, 32k;
- prefix reuse: 0%, 50%, 90%.

## Metrics

Collect:

- `kv_memory_gpu_bytes`
- `kv_memory_cpu_bytes`
- `kv_memory_nvme_bytes`
- `kv_compression_ratio`
- `kv_prefix_reuse_rate`
- `kv_pruned_tokens_total`
- `kv_evictions_total`
- `kv_policy_active`
- `quantization_mode_active`
- request latency histograms
- generated token throughput
- backend GPU utilization

## Quality Evaluation

Use a held-out evaluation set per model:

- exact JSON validity for structured outputs;
- tool-call name and argument accuracy;
- pairwise preference versus BF16 baseline;
- retrieval answer faithfulness;
- perplexity or logprob delta when available.

Accept a compressed mode only when quality delta stays inside the model owner
tolerance and latency/throughput gains justify the mode.
