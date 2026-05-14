# Perkunasv2 Shard-Native Active-Parameter Streaming Training

Perkunasv2 is a shard-native training path for decoder-only pretraining. It does
not instantiate a full `nn.Module` containing all layers at once. A run consists
of architecture metadata, tokenizer artifacts, parameter shards, optimizer
shards, and one active trainable module at a time.

## Why This Reduces Memory

Conventional training keeps every parameter and every optimizer state resident
for the whole model. AdamW usually adds two full-size optimizer tensors per
parameter, so optimizer memory can dominate small-GPU training.

Perkunasv2 bounds resident trainable state to:

- the current parameter shard;
- the current optimizer shard during update;
- current microbatch boundary activations;
- small metadata and trainer state.

This trades extra compute and disk I/O for lower active parameter and optimizer
memory.

## Checkpoint Layout

```text
training/runs/perkunasv2/
  config.json
  trainer_state.json
  tokenizer/
  shards/
    params/
      embeddings.pt
      block_000.pt
      block_001.pt
      block_002.pt
      ...
      final_norm.pt
      lm_head.pt
    optim/
      embeddings.pt
      block_000.pt
      block_001.pt
      block_002.pt
      ...
      final_norm.pt
      lm_head.pt
    metadata.json
```

There is no required monolithic checkpoint. `trainer_state.json` tracks
`global_step`, `tokens_seen`, `optimizer_step`, scheduler state, RNG state where
practical, latest validation loss, config hash, and shard metadata version.

## Forward Flow

For each microbatch:

1. Load `embeddings.pt`, build only the embedding module, move it to the target
   device, compute token embeddings, save the boundary activation on CPU, and
   unload the module.
2. For each transformer block, load `block_N.pt`, build only that block, compute
   its output, save the next boundary activation, and unload the block.
3. Load `final_norm.pt`, compute normalized hidden states, save the lm-head
   boundary activation, and unload final norm.
4. Load `lm_head.pt`, compute logits and loss, and unload lm head.

The forward pass runs without retaining autograd graphs.

## Backward And Update Flow

Normal end-to-end autograd cannot survive module unloading, so Perkunasv2 uses
explicit shard-local recomputation:

1. Reload `lm_head`, recompute logits from the saved norm boundary activation,
   backpropagate the loss, update only `lm_head` with its optimizer shard, save
   both shards, and release them.
2. Reload `final_norm`, recompute final normalization, backpropagate the saved
   gradient from the lm head, update final norm, save, and release.
3. Walk transformer blocks in reverse order. For each block, reload the block,
   recompute local forward from the saved input boundary, backpropagate the
   incoming gradient, update the block shard, save, and release.
4. Reload embeddings, recompute token embeddings from `input_ids`, backpropagate
   the final incoming gradient, update embeddings, save, and release.

Gradient accumulation is supported by collecting several microbatch traces,
accumulating gradients inside the active shard, and applying one AdamW update per
shard for the global step.

## Optimizer Shards

AdamW state is stored per parameter shard. There is no global optimizer over all
model parameters. During a shard update:

1. Load the matching optimizer shard from `shards/optim`.
2. Initialize missing `exp_avg` and `exp_avg_sq` tensors for first use.
3. Apply decoupled AdamW using the current scheduled learning rate.
4. Save the updated parameter shard.
5. Save the updated optimizer shard.
6. Release optimizer tensors immediately.

## Runtime Enforcement

The `ParameterShardStore` tracks active parameter shards and active optimizer
shards independently. It raises if the active shard count exceeds
`max_resident_shards` or if active parameters match full-model residency.

Training logs include:

- active shard;
- step;
- loss;
- shard update time;
- tokens/sec;
- CUDA allocated/reserved/peak memory when using CUDA;
- current residency snapshot.

## Commands

Initialize random sharded weights:

```powershell
python .\training\scripts\train_perkunasv2.py `
  --init-shards `
  --config .\training\configs\perkunasv2_280m.json `
  --run-dir training\runs\perkunasv2
```

Train from packed token shards:

```powershell
python .\training\scripts\train_perkunasv2.py `
  --train `
  --run-dir training\runs\perkunasv2 `
  --data-dir training\data\perkunas_pilot\tokenized `
  --seq-len 512 `
  --micro-batch-size 1 `
  --gradient-accumulation-steps 32 `
  --dtype fp16 `
  --device cuda
```

Validate from shards:

```powershell
python .\training\scripts\train_perkunasv2.py `
  --validate `
  --run-dir training\runs\perkunasv2 `
  --data-dir training\data\perkunas_pilot\tokenized `
  --seq-len 512 `
  --dtype fp16 `
  --device cuda
```

Resume is automatic when `--train` is used with an existing `run_dir` containing
`trainer_state.json`.

## Known Tradeoffs

- Training is slower than fully resident training because each shard is loaded
  and saved repeatedly.
- Boundary activations are stored for each accumulated microbatch. This is much
  smaller than full optimizer residency, but it still scales with sequence
  length, hidden size, layers, and gradient accumulation.
- `dropout` is required to be `0.0` because shard-local backward recomputes each
  stage after unloading modules.
- `tied_embeddings=true` is rejected for now because exact tied-weight training
  requires cross-stage gradient merging before a single shared optimizer update.

## Debugging Memory Leaks

Use `max_resident_shards: 1` during development. If residency checks fail, inspect
`train_log.jsonl` for the last `active_shard` and residency snapshot.

On CUDA, watch:

- `allocated_mb`;
- `reserved_mb`;
- `peak_allocated_mb`;
- whether those values return near the expected baseline after each shard.

If memory grows monotonically, check for retained tensors in microbatch traces,
module references held outside context managers, or optimizer state that was
loaded but not saved/released.

## Future Improvements

- async prefetch of the next parameter shard;
- NVMe mmap shard reads and writes;
- quantized optimizer states;
- blockwise gradient checkpoint compression;
- shard-level scheduling across multiple devices;
- exact tied embedding gradient merge support.
