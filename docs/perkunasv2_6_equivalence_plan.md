# Perkunas_v2.6 Equivalence and Stability Plan

Version: draft 0.1  
Date: 2026-04-27

## Purpose

Perkunas_v2.6 is not a bigger-model run. It is a 50M-parameter correctness and stability gate for the shard-native trainer.

The coworker feedback is correct: shard-local clipping, deterministic recomputation, untied embeddings, zero dropout, and reduced training-data shuffle are not mere performance details. They can affect stability, convergence, and final quality. v2.6 therefore starts with an equivalence test before spending long wall-clock time on another full run.

## Questions v2.6 Should Answer

1. Does shard-native global clipping produce the same update as full-resident training on a tiny GPT?
2. Do per-step loss, global gradient norm, clip scale, and final weights match under controlled conditions?
3. Does a 50M Perkunasv2 model learn cleanly under the production shard-native path?
4. Does sequential training-data streaming remain acceptable, or does reduced shuffling harm validation?
5. Does the 50M run provide enough evidence to justify returning to 291M+ models?

## Implemented Equivalence Gate

The test suite now includes:

```text
test_shard_native_global_clip_matches_full_resident_one_step
```

This test builds:

- one tiny full-resident Perkunasv2 model
- one shard-native Perkunasv2 run
- same initialization seed
- same packed training rows
- same fp32 precision
- same AdamW optimizer implementation
- same learning rate and weight decay
- same full-model global gradient clipping
- same gradient accumulation
- same no-shuffle data order

It compares:

- reported training loss
- full-model global gradient norm
- global clip scale
- final shard weights against final full-resident weights

Current result:

```text
python -m pytest training\tests\test_perkunasv2_shard_native.py::test_shard_native_global_clip_matches_full_resident_one_step -q
1 passed
```

The full shard-native suite also passes:

```text
python -m pytest training\tests\test_perkunasv2_shard_native.py -q
27 passed
```

## v2.6 Main Run Recommendation

Use a 50M-parameter model so both shard-native and full-resident behavior remain practical to compare:

```text
config: training/configs/perkunasv2_6_50m.json
parameters: 49,840,448
hidden size: 448
layers: 9
heads: 7
head dim: 64
intermediate size: 1152
sequence length: 512
dtype: fp16 active compute
master weights: fp32
optimizer: AdamW
gradient clipping: global
training order: no shuffle for current mmap performance
validation: required every 100 steps
```

The first decision point is not train loss. It is validation at steps 100, 200, 300, and 500. Because the model is much smaller than v2.5, it should move faster per step and should reveal convergence or validation problems sooner.

## Risks to Track

- Global clipping is correct but slower than shard-local clipping.
- Zero dropout is required for deterministic recomputation, but it removes one regularization tool.
- Untied embeddings avoid cross-shard gradient merging, but change model parameterization.
- No-shuffle training improves throughput but may reduce stochasticity.
- The current persistence policy rewrites full shard state every step and is write-heavy.

## Exit Criteria

Continue v2.6 if:

- train loss falls smoothly,
- validation loss falls at steps 100 and 200,
- all 12 logical shards update every step,
- `global_grad_clip_scale` is not pinned extremely low for long periods,
- `data_load_seconds` remains near zero,
- no commit or transaction errors appear.

Pause or adjust if:

- validation rises while train loss falls,
- global clip scale collapses below 0.1 for many steps,
- update time regresses unexpectedly,
- any shard update count is below 12,
- generated validation is accidentally using train shards.
