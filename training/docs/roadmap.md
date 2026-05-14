# Recommended Roadmap After First Successful Run

## Immediately After Smoke

1. Inspect `training/runs/smoke/train_log.jsonl`.
2. Confirm loss decreases on the tiny run.
3. Run checkpoint evaluation and inspect generated samples.
4. Export the checkpoint and verify artifact structure.

## First Pilot

1. Run normalization and dedup on the full local shard.
2. Train tokenizer with `vocab_size=32000`.
3. Tokenize at sequence length 1024.
4. Train `perkunas-small` for a short pilot run.
5. Compare validation loss across checkpoints.

## Before Long Training

1. Add more parquet shards.
2. Re-run inspection and schema drift checks.
3. Revisit language/license/date filters.
4. Recompute duplicate reports.
5. Benchmark dataloader throughput and tokens/sec.

## Medium-Term

1. Move to `perkunas-medium`.
2. Add external evaluation harness hooks.
3. Implement fused or flash attention options when GPUs are available.
4. Validate export with the KV-optimized serving platform.
5. Add extension training for embedding, classification, and reranking heads.
