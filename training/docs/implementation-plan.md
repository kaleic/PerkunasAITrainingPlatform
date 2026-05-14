# Phased Implementation Plan

## Stage A: Single-Shard Bring-Up

1. Inspect `D:\LLMProject\0000.parquet`.
2. Normalize with permissive language/license settings.
3. Deduplicate with exact + SimHash.
4. Train a 32k byte-level BPE tokenizer.
5. Tokenize into 1024-token blocks with 0.5% validation split.
6. Run `perkunas-small` smoke training for 100 steps.
7. Export a checkpoint and test loading into the serving artifact layout.

## Stage B: Pilot Pretraining

1. Increase model to `perkunas-medium`.
2. Use sequence length 2048.
3. Train on all clean rows from the first shard.
4. Track loss curves, tokenizer stats, duplicate rate, and generation samples.
5. Validate checkpoint resume after an intentional interruption.

## Stage C: Multi-Shard Corpus

1. Add additional PleIAs/common_corpus parquet shards to `data.yaml`.
2. Re-run inspection to catch schema drift.
3. Normalize all shards with the same thresholds.
4. Deduplicate globally across shards.
5. Retrain tokenizer if domain composition changes materially.

## Stage D: Scale Training

1. Move to torchrun DDP on multi-GPU.
2. Increase batch size via gradient accumulation.
3. Enable BF16 mixed precision on supported GPUs.
4. Add periodic exports for serving compatibility checks.
5. Add external benchmark hooks.

## Stage E: Extension Heads and Modalities

1. Freeze a stable base checkpoint.
2. Add modality encoders and projection layers into token space.
3. Add auxiliary embedding/classification/reranking heads.
4. Train extensions without full base retraining where possible.
