# From-Scratch Training Risk Analysis

## Data Quality

Risk: the shard mixes natural language, code, court decisions, tables, and very
long documents.

Mitigation: inspect schema, preserve provenance, filter low-value rows, cap max
length, deduplicate, and keep manifests for later audit.

## Tokenizer Mismatch

Risk: a tokenizer trained on one shard may underperform when many more shards are
added.

Mitigation: treat the first tokenizer as a pilot artifact. Retrain or validate it
when shard composition changes. Track chars/token, fertility, unknown rate, and
segmentation examples.

## Undertraining

Risk: from-scratch models are poor until they see enough tokens.

Mitigation: use staged runs: smoke, tiny overfit, pilot, then long training. Do
not judge capability from the smoke checkpoint.

## Instability

Risk: random initialization, high learning rate, or batch-size changes can cause
loss spikes.

Mitigation: warmup, gradient clipping, AdamW beta control, deterministic seed
option, and frequent checkpointing.

## Resume Failure

Risk: long pretraining runs can be interrupted.

Mitigation: checkpoint model, optimizer, scaler, config, step, and metadata.
Update `latest` only after a successful checkpoint write.

## Multi-GPU Drift

Risk: a single-GPU pipeline may hide DDP shape or sampler issues.

Mitigation: the trainer uses PyTorch DDP and `DistributedSampler`; run a short
torchrun job before scale-up.

## Serving Compatibility

Risk: custom model architectures need serving integration.

Mitigation: export standard tokenizer/config/weights, preserve architecture code,
and generate a serving registration template for the existing KV-optimized
serving system.
