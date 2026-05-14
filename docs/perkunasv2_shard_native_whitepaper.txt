# Perkunasv2: Shard-Native Training With Exported High-Throughput Serving

Version: draft 1.0  
Date: 2026-05-01  
Project: Perkunas / D:\LLMProject

## Abstract

Perkunasv2 is a shard-native runtime for training decoder-only language models under strict memory constraints, paired with a post-training export path for standard high-throughput serving runtimes. Its central design decision is to make parameter shards and optimizer shards the canonical training representation of the model. The runtime does not treat sharding as a wrapper around a full resident `nn.Module`; instead, during training, the shards are the model.

The current implementation trains GPT-style decoder models on a single NVIDIA GeForce RTX 3050 8 GB by streaming one active shard at a time through GPU memory. It avoids full-model residency during training, avoids a resident global optimizer object, stages parameter and optimizer updates transactionally, supports fp32 master-weight storage with fp16 active compute, supports shard-local clipping, full-model global gradient clipping, a periodic scalar-only global optimizer normalizer, and an experimental guarded step replay mode that can reject and retry harmful same-batch updates before publishing them. Validation runs against separate held-out tokenized data.

The original Perkunasv2 run demonstrated a 291,351,424 parameter model reaching validation loss 7.0371 by step 1000 on held-out C4 validation data, from 9.2588 at step 25. A later Perkunas_v2.1 run initialized a larger 373,867,520 parameter model from step 0 with fp32 master shards, fp16 active compute, AdamW, and shard-local updates. The Perkunas_v2.5 line returned to the 291M-parameter family and added full-model global gradient clipping while preserving shard-native execution. The Perkunas_v2.6 50M line exposed a tokenizer-vocabulary mismatch and plateau-recovery behavior. The Perkunas_v2.8 50M-correct-vocab line fixed the model/data vocabulary contract at 8,000 tokens. The current Perkunas_v2.9 100M TinyStories line has trained a 100,487,424 parameter model under the same memory-first design and reached validation loss 3.5135 at step 3000 on 100 held-out validation batches.

This paper exposes the system mechanics, training loop, validation evidence, serving/export path, current limitations, and the engineering roadmap required to turn Perkunasv2 from prototype into a robust platform.

## Executive Summary

Perkunasv2 is a memory-first model training system with two deliberate serving modes.

Its claim is not that memory-constrained GPUs become faster than datacenter accelerators. The claim is narrower, testable, and commercially useful:

> Perkunasv2 expands the practical training envelope of commodity GPUs by making model shards, optimizer shards, and fp32 master weights durable first-class training state.

Serving is treated as a separate product phase. The shard-native host proves that a run directory can be queried directly without assembling a conventional checkpoint. For responsive production-style inference, the preferred path is now to export the trained streamable checkpoint into a Hugging Face / Llama-compatible safetensors package and serve that package through a mature runtime such as vLLM.

That separation does not weaken the training result. It clarifies it. Training and serving have different optimization targets:

```text
training: minimize active residency while preserving optimizer state and update correctness
serving: maximize token latency/throughput from a frozen checkpoint
```

Perkunasv2's core contribution is on the training side. The export path lets the system use standard inference infrastructure instead of reinventing high-throughput serving.

The current system demonstrates:

- Shard-native initialization of GPT-style decoder models.
- Training without constructing the full model in GPU memory.
- Per-shard parameter and optimizer loading.
- Per-shard AdamW, Lion, and experimental Adafactor-style update paths.
- fp32 canonical master-weight storage while active compute remains fp16.
- Forward tracing without retaining the complete autograd graph.
- Reverse-order backward replay with local recomputation.
- Chunked language-model-head loss to avoid full logit materialization.
- Shard-local gradient clipping, full-model global gradient clipping, and periodic global gradient-RMS normalization without persistent global optimizer state.
- Experimental guarded step replay: candidate updates can be staged, evaluated on the same batch, retried with safer learning-rate or gradient-norm scales, and either accepted or skipped without publishing rejected shards.
- Token-based learning-rate scheduling.
- Separate validation data directories.
- Held-out validation learning on the 55,963,520-parameter v2.8 corrected-vocabulary 50M line, including validation loss 7.6752 at step 1300 on 100 held-out batches.
- Historical plateau-recovery evidence on the 49,840,448-parameter v2.6 50M line, now treated as an instructive 32k-vocabulary mismatch run rather than the recommended 50M baseline.
- Transaction-staged shard writes.
- Async shard writes with bounded queueing.
- Optional RAM-disk active shard stores, with the NVMe run directory retained as the durable canonical archive.
- Locality-preserving train shuffle that avoids PyTorch's full-dataset `RandomSampler` permutation and keeps mmap reads local within token shards.
- Fine-grained timing buckets for load, module rebuild, host-to-device copy, kernels, CPU activation/gradient copy, optimizer math, and shard save staging.
- CPU, GPU, and secondary-GPU shard prefetch modes.
- A shard-native local inference server with `/health`, `/generate`, `/compare`, `/models`, `/v1/models`, `/v1/chat/completions`, and local preload/cache options for test generation.
- An OpenAI-style chat/completions compatibility layer for local testing.
- A Hugging Face / vLLM exporter that converts the training shard store into a Llama-compatible `model.safetensors` package, including tokenizer files and generation config.
- A second serving route through `kvserve` for exported full-checkpoint Perkunas artifacts.

The system is not yet production complete. It still needs stronger crash recovery, immutable checkpoint snapshots, snapshot-pinned shard-native serving, broader export validation, and controlled benchmark comparisons. But it is no longer just a memory-management experiment. It is a functioning training runtime with measurable learning and a credible path to standard serving infrastructure.

## Strategic Context

Modern language-model development is bottlenecked by memory availability as much as by raw compute. High-memory accelerator access is expensive, scarce, supply-chain constrained, and often unavailable to smaller organizations. Enterprise GPU supply is a strategic resource.

Perkunasv2 targets that constraint directly. It asks:

> How much useful language-model training can be extracted from commodity silicon that is already widely deployed?

The answer matters to:

- Independent research labs.
- Universities.
- Small companies.
- Air-gapped organizations.
- Domestic semiconductor programs.
- Edge and workstation training appliances.
- Organizations that need architecture tests before reserving cluster time.

An 8 GB RTX 3050 is not an H100. Perkunasv2 does not pretend otherwise. The commercial opportunity is different: use commodity hardware for model-design iteration, early pretraining, ablation studies, continued pretraining experiments, and constrained local model development that would normally be blocked by VRAM.

## Scientific Claim

The testable Perkunasv2 claim is:

> A decoder language model can be trained from a canonical shard store by streaming one active parameter shard and one active optimizer shard through GPU memory, without ever requiring the full model and optimizer to be resident during training.

This implies four measurable subclaims:

1. Correctness: every shard receives gradients and optimizer updates.
2. Persistence: parameter and optimizer state survive across steps and resumes.
3. Learning: held-out validation loss falls over training.
4. Memory expansion: model shapes larger than conventional full-resident training on the same GPU become trainable.
5. Exportability: the trained shard store can be packaged into a standard inference artifact when low-latency serving matters more than low active residency.

The current evidence supports the first three claims. The fourth requires a formal benchmark suite against full-resident PyTorch, ZeRO-Offload, and FSDP baselines.

## The Core Design

Conventional single-GPU training usually looks like this:

```text
instantiate full model
move full model to GPU
create global optimizer over all parameters
run forward pass
retain autograd graph
run backward pass
update all parameters
checkpoint model and optimizer
```

Perkunasv2 instead looks like this:

```text
load architecture config
load one parameter shard
build one active module
run local forward or backward work
load the matching optimizer shard
apply local optimizer update
stage parameter and optimizer writes
release the active shard
continue to the next shard
```

The invariant is:

```text
active GPU state = current shard + local activations + temporary tensors
```

The full model exists as:

```text
config.json
shards/params/*.pt or *.safetensors
shards/optim/*.pt or *.safetensors
trainer_state.json
tokenizer metadata
```

That is the major departure from normal PyTorch training. In Perkunasv2, the durable shard store is not merely a checkpoint. It is the canonical model.

## Current Model Families

### Perkunasv2.0

The original shard-native run used:

```text
vocab_size:              32000
hidden_size:             896
num_layers:              24
num_attention_heads:     14
intermediate_size:       2432
max_position_embeddings: 2048
sequence_length:         512 during training
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        291,351,424
```

### Perkunas_v2.1

The current fresh run initializes a larger model:

```text
vocab_size:              32000
hidden_size:             1024
num_layers:              24
num_attention_heads:     16
intermediate_size:       2816
max_position_embeddings: 2048
sequence_length:         512 during current run
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        373,867,520
```

Both models are split into 27 canonical parameter shards:

```text
embeddings
block_000
block_001
...
block_023
final_norm
lm_head
```

There are 24 transformer block shards, one embedding shard, one final normalization shard, and one language-model-head shard.

### Perkunas_v2.5

Perkunas_v2.5 is a stability-test family. It returns to the 291M-parameter geometry while incorporating lessons from v2.1-v2.4:

```text
vocab_size:              32000
hidden_size:             896
num_layers:              24
num_attention_heads:     14
intermediate_size:       2432
max_position_embeddings: 2048
sequence_length:         512 during current run
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        291,351,424
```

The v2.5 training recipe uses fp16 active compute, fp32 master shards, AdamW, sequential training-data streaming, CPU shard prefetch, chunked `lm_head`, and full-model global gradient clipping. The goal is not to maximize early tokens/sec at all costs. The goal is to determine whether a globally controlled update path improves validation stability after earlier runs plateaued or drifted upward.

### Perkunas_v2.6 50M Historical 32k-Vocab Run

Perkunas_v2.6 is a smaller 50M-parameter experimental line intended for faster optimizer and training-schedule exploration on the same RTX 3050 8 GB hardware. It is now classified as historical because it used a 32,000-token model head against tokenized data whose observed token IDs fit an 8,000-token tokenizer:

```text
vocab_size:              32000
hidden_size:             448
num_layers:              9
num_attention_heads:     7
intermediate_size:       1152
max_position_embeddings: 2048
sequence_length:         512 during current run
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        49,840,448
storage shards:          12
```

The 12 canonical shards are:

```text
embeddings
block_000
block_001
...
block_008
final_norm
lm_head
```

This model is not intended to compete with larger v2.x runs on final quality. Its purpose is to compress the experiment cycle: test global clipping, token-scheduled learning-rate floors, batch-size changes, shuffling behavior, and throughput under the same shard-native invariants. The run remains useful because it exposed optimizer behavior quickly while still validating on held-out C4. The vocabulary mismatch, however, means it spent substantial parameter and softmax capacity on unreachable token IDs, so it should not be the recommended baseline for new 50M runs.

### Perkunas_v2.8 50M Correct-Vocab Run

Perkunas_v2.8 is the corrected 50M-class experimental baseline. It corrects the tokenizer/model contract by setting `vocab_size=8000`, matching the project tokenizer and the observed packed-token range. To keep the model in the same experimental size class after reducing vocabulary parameters, the hidden width was increased:

```text
vocab_size:              8000
hidden_size:             640
num_layers:              9
num_attention_heads:     10
intermediate_size:       1792
max_position_embeddings: 2048
sequence_length:         512 during current run
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        55,963,520
storage shards:          12
```

The corrected run keeps the same canonical shard list:

```text
embeddings
block_000
block_001
...
block_008
final_norm
lm_head
```

The important lesson is that the training stack was compatible with the old 32k-vocabulary model, but the recipe was wasteful. With 8k data, the random cross-entropy baseline is `ln(8000) = 8.99`, not `ln(32000) = 10.37`. New runs should use the tokenizer's real vocabulary size unless deliberately testing an expanded output vocabulary.

### Perkunas_v2.9 100M TinyStories Run

Perkunas_v2.9 is the current 100M-class TinyStories line. It keeps the corrected 8,000-token vocabulary and increases model capacity while preserving the same shard-native training invariant:

```text
vocab_size:              8000
hidden_size:             768
num_layers:              13
num_attention_heads:     12
intermediate_size:       1920
max_position_embeddings: 2048
sequence_length:         512 during current run
normalization:           RMSNorm
position encoding:       RoPE
activation:              SwiGLU
dropout:                 0.0
tied_embeddings:         false
total parameters:        100,487,424
storage shards:          16
```

The canonical shard list is:

```text
embeddings
block_000
block_001
...
block_012
final_norm
lm_head
```

The run demonstrates two useful properties:

- The 100M-class model trains under the same memory-first streaming design.
- A stopped checkpoint can be exported into a Hugging Face / Llama-compatible safetensors package for vLLM-style serving.

The best observed TinyStories validation point in this line is:

```text
step:                  3000
validation loss:       3.5135
validation perplexity: 33.5658
validation batches:    100
```

Later recipe branches produced lower short-window train-loss floors but did not immediately improve validation. That distinction is important: local train-loss dips prove the model can find easier batches or sharper immediate fits, while held-out validation decides whether a recipe is improving the checkpoint.

## Runtime Components

### `PerkunasV2Config`

The model configuration defines the architecture:

- Vocabulary size.
- Hidden size.
- Number of layers.
- Number of attention heads.
- SwiGLU intermediate size.
- Maximum position embeddings.
- RoPE theta.
- RMSNorm epsilon.
- Token IDs.

The config validates structural constraints, including:

- `hidden_size % num_heads == 0`
- RMSNorm only in the current implementation.
- SwiGLU only in the current implementation.
- No tied embeddings, because tied embeddings require cross-shard gradient merging.
- Dropout must be zero, because shard-local recomputation must be deterministic.

### `ParameterShardStore`

`ParameterShardStore` is the canonical shard manager. It owns:

- Parameter shard paths.
- Optimizer shard paths.
- Metadata.
- Active shard residency counters.
- Prefetch caches.
- Pending async writes.
- Transaction directories.
- Trainer state.

This component is the heart of the runtime. It is responsible for making the model durable while allowing only a narrow shard window to be active.

### Module Builders

The runtime constructs modules on demand:

```text
build_embeddings
build_transformer_block
build_final_norm
build_lm_head
```

There is intentionally no normal training path that constructs the full model.

### `ShardStreamingTrainer`

The trainer coordinates:

- Data loading.
- Forward tracing.
- Boundary activation capture.
- Reverse backward replay.
- Local shard recomputation.
- Gradient clipping.
- Optimizer updates.
- Transaction commit.
- Validation.
- Logging.
- Trainer state persistence.
- Resume repair.
- Shard prefetching.

### Tokenized Data Runtime

Training consumes pre-tokenized `.npy` shards. Each row stores `seq_len + 1` tokens:

```text
input_ids = tokens[:-1]
labels    = tokens[1:]
```

This removes tokenizer cost from the training loop and makes training behavior easier to profile.

The current C4 tokenized train and validation sets use the project tokenizer's 8,000-token vocabulary. Packed shards are `int32` arrays with rows shaped as `513` columns for `seq_len=512`, and observed token IDs remain within `0..7999`. This is now a hard recipe constraint: a model may choose a larger `vocab_size`, but doing so spends parameters and language-model-head compute on output classes that the dataset never targets. The v2.8 corrected-vocabulary run therefore uses `vocab_size=8000`.

The training corpus is very large, so data order is also a systems concern. A naive PyTorch map-style shuffle delegates to `RandomSampler`, which can build a full random permutation for the whole dataset and then issue random mmap reads across tens of thousands of `.npy` files. Perkunasv2 now uses a locality-preserving iterable data path for shuffled training: it shuffles token-shard order, then streams rows sequentially within each shard. That preserves most of the useful stochasticity while avoiding the worst startup-memory and filesystem-locality costs.

## Training Lifecycle

### 1. Initialization

`--init-shards` creates:

```text
run_dir/config.json
run_dir/shards/metadata.json
run_dir/shards/params/*.pt or *.safetensors
run_dir/shards/optim/
run_dir/trainer_state.json
```

Each parameter shard is initialized independently from the model config. Optimizer shards are created lazily when training first touches each shard.

### 2. Forward Trace

The forward pass does not retain a full autograd graph. It records boundary activations:

```text
load embeddings
compute hidden state
save boundary activation
release embeddings

for each transformer block:
  load block
  compute hidden state
  save boundary activation
  release block

load final_norm
compute normalized hidden state
save final boundary
release final_norm
```

For validation or direct loss computation, the model then loads `lm_head` and computes cross-entropy.

### 3. Backward Replay

Backward is performed in reverse shard order. Each shard is loaded again, recomputed locally, backpropagated, updated, and released:

```text
load lm_head
recompute logits from normalized hidden state
compute cross-entropy
backpropagate into normalized hidden state
update lm_head
release lm_head

load final_norm
recompute local output
backpropagate downstream gradient
update final_norm
release final_norm

for block_023 down to block_000:
  load block
  recompute output from saved boundary
  backpropagate downstream gradient
  update block
  release block

load embeddings
recompute embeddings
backpropagate downstream gradient
update embeddings
release embeddings
```

This is why the end-of-step logs can show:

```text
active_param_shards: {}
active_optimizer_shards: {}
resident_parameter_count: 0
resident_optimizer_state_count: 0
```

Those counters are zero because the active shard has already been released at log time. The important step activity counters are:

```text
updated_shards: 27
optimizer_shards_touched: 27
max_active_param_shards_observed: 1
max_active_optimizer_shards_observed: 1
```

That means every shard was updated while the maximum active residency remained one parameter shard and one optimizer shard.

## Optimizer Sharding

Perkunasv2 uses optimizers, but not a global optimizer object.

For each trainable shard:

```text
load optimizer state for this shard
move optimizer tensors for this shard to the active device
clip gradients with either shard-local or global-model policy
apply AdamW, Lion, or Adafactor-style update
stage updated parameter shard
stage updated optimizer shard
release optimizer tensors
```

Current optimizer modes:

- `adamw`: default and recommended.
- `lion`: implemented for experimentation.
- `adafactor`: experimental, with factored state for matrix-like tensors.

This design avoids resident AdamW state for the entire model. AdamW normally requires multiple tensors per parameter: weights, gradients, first moment, second moment, and often master weights. Perkunasv2 only loads that state for the current shard.

## Global Gradient Clipping

Earlier Perkunasv2 runs clipped gradients independently inside each shard. That is memory-efficient, but it is not mathematically identical to normal full-model gradient clipping. If every shard is clipped separately, the effective full-model update can be larger than intended because each shard gets its own norm budget.

Perkunas_v2.5 adds an explicit global clipping mode:

```text
--grad-clip-mode global
```

In global mode, a training step becomes:

```text
run forward trace
run backward replay shard by shard
collect each shard's CPU gradient payload
compute one full-model gradient norm
compute one global clip scale
apply the same scale to every shard gradient
run per-shard AdamW updates
commit updated parameter and optimizer shards
```

The clip rule is:

```text
global_clip_scale = min(1.0, max_grad_norm / global_grad_norm)
```

This gives the shard-native trainer the same global clipping behavior expected from a resident PyTorch model, without making the full model resident on GPU. It does require extra CPU gradient payload handling, so it is slower than pure shard-local clipping. The tradeoff is scientific: if a run stalls, diverges, or plateaus, global clipping removes one possible source of mismatch from standard training.

The logs expose this directly:

```text
global_grad_norm
global_grad_clip_scale
grad_clip_mode
```

That makes gradient control auditable. The v2.5 fresh run shows global clipping active from step 1, with `global_grad_clip_scale` starting near 0.70 and tightening toward 0.50 as the gradients grow.

## Periodic Global Optimizer Normalizer

Full-model global clipping improves standard-training equivalence, but it is expensive when run every step because gradients must be retained long enough to compute the full-model norm. The newer compromise is a periodic scalar-only global optimizer normalizer:

```text
--global-optimizer-every <N>
--global-optimizer-blend <0..1>
--global-optimizer-min-scale <scale>
--global-optimizer-max-scale <scale>
```

This feature is not a resident global AdamW object and it does not keep whole-model optimizer state in memory. On scheduled steps, the trainer runs a global gradient-statistics pass, computes one target gradient RMS across all shards, computes one scalar normalization factor per shard, clamps and blends those factors, then replays the normal shard-local optimizer update with the shard's scalar applied.

In operational terms:

```text
ordinary step:
  shard-local backward replay and optimizer update

every Nth step:
  collect scalar global gradient statistics
  compute per-shard gradient-RMS normalization factors
  recompute shard-local gradients
  apply clipped/blended scalar factors during per-shard optimizer updates
```

The design goal is memory discipline: preserve the per-shard optimizer storage model while periodically correcting shard-to-shard update imbalance. It is slower on scheduled steps because those steps perform extra gradient-statistics work, but it avoids the persistent memory balloon of a conventional full-model optimizer. The recommended pairing for this mode is `--grad-clip-mode shard` plus periodic global normalization; using `--grad-clip-mode global` simultaneously keeps the expensive full-model gradient path active every step.

The training log records the normalizer under `global_optimizer`, including `target_grad_rms`, raw scale range, and applied scale range, so the effect can be audited.

## fp32 Master Weights With fp16 Active Compute

The current runtime supports:

```text
--dtype fp16
--master-weight-dtype fp32
```

This is important.

`--dtype fp16` controls the active module dtype used during forward and backward compute. It protects VRAM.

`--master-weight-dtype fp32` controls canonical shard storage and update precision. The optimizer update is applied to fp32 master tensors, then copied back into the active fp16 module for continued compute.

This prevents tiny late-stage updates from being rounded away by fp16 storage. It does not recover precision already lost by previous fp16 shard saves, but for a fresh v2.1 run the canonical shard weights begin and remain fp32.

In practical terms:

```text
GPU active compute: fp16
canonical parameter shards: fp32
optimizer state: fp32
```

That is the current recommended training mode on RTX-class hardware.

## Chunked Language-Model Head

The language-model head can create huge temporary logits:

```text
batch_size * sequence_length * vocab_size
```

For `batch=32`, `sequence_length=512`, and `vocab=32000`:

```text
32 * 512 * 32000 = 524,288,000 logits
```

Perkunasv2 supports chunking this path:

```text
--lm-head-chunk-tokens 4096
```

The runtime flattens hidden states and labels, computes logits for one token chunk at a time, accumulates cross-entropy, and backpropagates scaled chunk losses. This is one of the practical changes that makes the runtime usable on an 8 GB card.

## Transaction-Staged Updates

Each training step stages writes into:

```text
shards/transactions/step_<N>/params/
shards/transactions/step_<N>/optim/
```

A step can only commit if all expected parameter shards and optimizer shards were staged. This prevents intentionally publishing a step where only some shards were updated.

Current commit behavior:

```text
flush pending async saves
verify all parameter shards staged
verify all optimizer shards staged
replace canonical shard files one by one
delete transaction directory
save trainer_state.json
```

Important limitation: `os.replace` is atomic per file, not across all shard files. A process or machine crash during the replacement loop can still leave mixed-step canonical shards. This is the remaining crash-recovery gap. The roadmap calls for a commit manifest or write-ahead-log style protocol so the runtime can detect and repair partial commits.

## Guarded Step Replay

The transaction system now supports an experimental safety mode called guarded step replay. The purpose is not to force every training batch to show monotonically lower loss. Batch-to-batch loss naturally jumps because some batches are harder than others. Instead, the guard evaluates the same batch before and after a candidate update and rejects updates that are clearly worse than the configured tolerance.

The high-level protocol is:

```text
run forward trace for the current batch
stage candidate parameter and optimizer shard updates
replay the same batch through the staged candidate state
if loss_after <= loss_before + tolerance:
  commit transaction
else:
  abort transaction
  retry with safer LR / grad-norm scales
```

The command surface is:

```text
--guarded-step-replay
--guard-replay-max-replays 2
--guard-replay-loss-tolerance 0.02
--guard-replay-loss-tolerance-ratio 0.002
--guard-replay-lr-scales 1.0,0.5,0.25
--guard-replay-grad-norm-scales 1.0,0.75,0.5
--guard-replay-on-exhaust accept|skip
```

`accept` commits the last attempted update if all replay attempts fail. This is the safer default for normal long-running training because it avoids starving the model of updates. `skip` advances the data stream without publishing an optimizer step if every attempt is rejected. This is useful for aggressive experiments, but it can burn batches without learning and should be tracked carefully.

The log records the intervention:

```json
"guarded_step_replay": {
  "active": true,
  "accepted": true,
  "attempts": 1,
  "loss_before": 7.52,
  "accepted_loss_after": 7.51,
  "accepted_lr_scale": 1.0,
  "accepted_grad_norm_scale": 1.0
}
```

This is a direct consequence of shard-native state management: because the runtime already stages a complete candidate step before publishing it, Perkunasv2 can evaluate, abort, and retry the update without corrupting the canonical shard store.

## Active Store and Durable Flush

Perkunasv2 now supports a two-tier shard store:

```text
NVMe run_dir       = durable canonical archive
RAM active_run_dir = active working shard store
durable flush      = checkpoint publication boundary
```

When `--active-run-dir` is provided, the trainer copies the initialized durable run directory into the active directory and then reads and writes shards from the active store. The original `--run-dir` remains the durable archive. A durable flush copies the active store back to the archive using atomic file replacement and writes `durable_flush_manifest.json` with the published step.

The command surface is:

```text
--run-dir <NVMe archive>
--active-run-dir <RAM disk working copy>
--durable-flush-every <N>
```

If `--durable-flush-every` is `0`, active-store training publishes at `--save-every` and at the final step. This keeps the shard-native design intact: the canonical representation is still a shard store, but the high-churn working copy can live on volatile memory while the SSD receives coherent checkpoint publications.

This does not eliminate the need for crash recovery. If power is lost before a durable flush, only the previous durable publication is guaranteed. The benefit is separating high-frequency shard churn from lower-frequency durable checkpoint publication.

One operational caveat is now explicit: the current `checkpoints/` entries are step markers around the live shard store, not immutable rewind snapshots. A run can resume from its current coherent shard directory, but it cannot safely rewind to an arbitrary earlier marker unless a full snapshot of that step's shards and `trainer_state.json` was preserved. The next checkpoint design should create real immutable snapshots, for example:

```text
checkpoints/step_00001800/
  trainer_state.json
  shards/
    metadata.json
    params/
    optim/
```

Until that exists, best-validation preservation should be done by copying or flushing a full run directory at the chosen step.

## Trainer State and Resume Repair

`trainer_state.json` tracks:

- `global_step`
- `optimizer_step`
- `tokens_seen`
- scheduler state
- latest validation loss
- RNG state
- config hash

The trainer now saves this state after every committed step. It also audits optimizer shard step counters on resume. If `trainer_state.json` lags behind optimizer shards, the trainer repairs the global step forward:

```text
WARNING: trainer_state.json lagged optimizer shards; repairing global_step 2125 -> 2309
```

This warning means the optimizer shards were ahead of trainer metadata, usually because durable shard writes succeeded but trainer-state metadata lagged. The repair moves `global_step` to the consistent optimizer-shard step. If optimizer shard steps disagree with each other, the trainer refuses to resume and asks for audit/repair.

## Prefetch and Residency

`max_resident_shards` started as a safety ceiling. With prefetch enabled, it also becomes a cache budget.

Supported modes:

```text
--prefetch-shards off
--prefetch-shards cpu
--prefetch-shards gpu
--prefetch-shards secondary-gpu
```

Mode behavior:

```text
off:
  no prefetching; safest baseline

cpu:
  stage upcoming parameter and optimizer payloads in host RAM

gpu:
  stage upcoming payloads on the active training GPU

secondary-gpu:
  stage upcoming payloads on a second CUDA device
```

CPU prefetch is the safest current default because it reduces disk latency without consuming active VRAM. GPU prefetch can reduce host-to-device transfer latency but spends VRAM on the training card. Secondary-GPU prefetch is the natural path for two-GPU systems, where one GPU computes and the other stages upcoming shards.

Current logs may show pending prefetches:

```text
pending_param_prefetches: [...]
pending_optimizer_prefetches: [...]
cached_param_shards: []
cached_optimizer_shards: []
```

This means the prefetch thread has scheduled or is loading payloads. The cache is consumed as active modules request shards.

## Async Writes and the I/O Wall

When async shard writes are enabled:

```text
--async-shard-writes
--max-pending-shard-writes 4
```

the store copies updated tensors to CPU and enqueues serialization on a background writer thread. Writes use temporary files followed by atomic file replace.

The runtime avoids direct same-path read/write races:

1. Before loading a parameter shard, it waits for any pending write to that parameter path.
2. Before loading an optimizer shard, it waits for any pending write to that optimizer path.
3. The write queue is bounded.
4. Transaction commit flushes pending writes before publishing the step.

This lets shard serialization overlap with later shard compute inside a step. It does not eliminate the I/O wall. As the model, optimizer state, and shard count grow, throughput will depend heavily on:

- NVMe read bandwidth.
- NVMe write bandwidth.
- CPU serialization cost.
- PCIe transfer cost.
- PyTorch `.pt` load/save overhead when using the legacy storage backend.
- Queue backpressure.

The runtime now supports an explicit safetensors storage backend:

```text
--shard-storage-format safetensors
```

This writes pickle-free `.safetensors` parameter and optimizer shards while still reading legacy `.pt` shards for migration. The remaining storage roadmap is to benchmark safetensors under long runs and then evaluate a raw tensor layout with pinned CPU staging if safetensors is still not enough.

## Timing Instrumentation

The step log now splits wall time into the pieces that matter for shard-native training:

```text
param_load_seconds
module_build_seconds
h2d_seconds
forward_kernel_seconds
backward_kernel_seconds
activation_cpu_copy_seconds
gradient_cpu_copy_seconds
optimizer_load_seconds
optimizer_math_seconds
param_save_stage_seconds
optimizer_save_stage_seconds
```

This changed the diagnosis of the next speed lever. RAM-disk active stores reduce durable-storage churn, but current 50M-class logs show that module rebuild and host-to-device transfer can dominate once parameter files are cached. In other words, the next speed lever is not only "make disk faster." It is reducing rebuild, copy, recomputation, and serialization overhead while keeping the active memory window bounded.

For diagnostic runs, `--timing-sync-cuda` forces CUDA synchronization around timing regions. It should not be used for throughput numbers because it slows training, but it is useful when separating CPU queueing from actual kernel time.

## Validation Design

Validation is now treated as a held-out signal:

- `--val-data-dir` points to a separate validation token directory.
- Validation does not silently fall back to training data unless explicitly allowed.
- Validation uses `drop_last=False`.
- Validation errors if no batches run.
- Validation can use the same chunked `lm_head` loss path.

This matters because earlier validation could accidentally reuse training shards. The current validation signal is much more credible.

## Empirical Evidence: Perkunasv2.0

The original v2.0 run learned from a near-random baseline and reached its best validation near step 1200 before flattening and drifting upward.

Selected held-out validation results:

| Step | Validation Loss | Validation Perplexity | Batches |
| ---: | ---: | ---: | ---: |
| 25 | 9.2588 | 10,496.36 | 12 |
| 100 | 8.8844 | 7,218.77 | 12 |
| 250 | 8.1509 | 3,466.40 | 12 |
| 500 | 7.4352 | 1,694.55 | 12 |
| 750 | 7.0984 | 1,210.04 | 12 |
| 1000 | 7.0371 | 1,138.08 | 128 |
| 1200 | 7.0133 | 1,111.29 | 128 |
| 1500 | 7.0625 | 1,167.40 | 128 |
| 1800 | 7.1315 | 1,250.72 | 128 |
| 2100 | 7.1623 | 1,289.93 | 128 |

Interpretation:

- The model definitely learned.
- Validation was no longer random.
- The run plateaued and then began to overfit or drift.
- The plateau motivated the v2.1 changes: larger hidden width, fp32 master weights, fresh initialization, lower starting learning rate than the early `1e-4` smoke run, and a less fragile precision path.

## Empirical Evidence: Perkunas_v2.1

The fresh v2.1 run uses:

```text
model parameters:          373,867,520
hidden size:               1024
layers:                    24
heads:                     16
intermediate size:         2816
dtype:                     fp16 active compute
master weight dtype:       fp32 canonical shards
optimizer:                 AdamW
learning rate:             3e-5
weight decay:              0.05
max grad norm:             1.0
sequence length:           512
micro_batch_size:          32
gradient accumulation:     1
prefetch:                  cpu
prefetch window:           500
```

Early training results:

| Step | Train Loss | Tokens/Sec | Shard Update Seconds |
| ---: | ---: | ---: | ---: |
| 1 | 10.5788 | 450.0 | 23.03 |
| 2 | 10.4874 | 473.0 | 23.26 |
| 58 | 7.8467 | 483.3 | 23.17 |

Interpretation:

- The model begins near the expected random-token range.
- The loss drop by step 58 shows active learning.
- Throughput is lower than the smaller v2.0 run because the model is wider and fp32 master storage increases data movement.
- Memory remains stable enough for the RTX 3050 class.
- Validation is still required before claiming v2.1 beats v2.0.

## Empirical Evidence: Perkunas_v2.5 Fresh Global-Clip Run

Perkunas_v2.5 returns to the 291M-parameter family and tests a more conservative training recipe:

```text
model parameters:          291,351,424
dtype:                     fp16 active compute
master weight dtype:       fp32 canonical shards
optimizer:                 AdamW
learning rate:             1e-5
weight decay:              0.05
max grad norm:             1.0
gradient clip mode:        global
sequence length:           512
micro_batch_size:          32
gradient accumulation:     4
effective tokens/update:   65,536
prefetch:                  cpu
storage format:            torch shard files
training order:            sequential data streaming
```

The first clean run from step 1 shows falling training loss while updating every canonical shard:

| Step | Train Loss | Tokens/Sec | Forward Sec | Backward/Update Sec | Global Grad Norm | Clip Scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.5728 | 1,187.6 | 22.118 | 31.065 | 1.430 | 0.699 |
| 2 | 10.5250 | 1,208.9 | 25.065 | 28.381 | 1.378 | 0.726 |
| 4 | 10.4542 | 1,269.0 | 23.246 | 27.652 | 1.423 | 0.703 |
| 8 | 10.2461 | 1,264.0 | 23.074 | 27.846 | 1.818 | 0.550 |
| 10 | 10.1660 | 1,225.4 | 24.420 | 28.320 | 1.844 | 0.542 |
| 12 | 9.9918 | 1,118.7 | 24.923 | 32.829 | 2.011 | 0.497 |

Interpretation:

- The run begins near the random-token baseline and immediately moves below it.
- `updated_shards=27` and `optimizer_shards_touched=27` confirm full-model coverage.
- `max_active_param_shards_observed=1` and `max_active_optimizer_shards_observed=1` confirm the shard-native memory invariant.
- `data_load_seconds` remains approximately 0.006-0.011 seconds, so the data loader is not the early bottleneck in this configuration.
- Global clipping is active and conservative: by step 12 the update is scaled to roughly half strength.
- This is not yet a validation result. The scientific test is whether held-out validation improves past the earlier low-7 plateau without drifting upward.

## Empirical Evidence: Perkunas_v2.8 50M Correct-Vocab Run

Perkunas_v2.8 is the current corrected 50M-class run. It uses the 8k tokenizer-compatible vocabulary, fp16 active compute, fp32 master shards, AdamW, RAM-disk active store, CPU prefetch, locality-preserving train shuffle, and 100-batch held-out validation.

The run uses:

```text
model parameters:          55,963,520
storage shards:            12
dtype:                     fp16 active compute
master weight dtype:       fp32 canonical shards
optimizer:                 AdamW
learning rate:             3e-5 initial recipe
weight decay:              0.05 initial recipe
gradient clip mode:        global in the initial run
sequence length:           512
validation batches:        100
active store:              RAM disk
durable archive:           NVMe run directory
storage format:            torch shard files
prefetch:                  cpu
prefetch window:           12
```

Selected held-out validation results:

| Step | Validation Loss | Validation Perplexity | Batches |
| ---: | ---: | ---: | ---: |
| 100 | 8.2274 | 3,742.22 | 100 |
| 200 | 8.0152 | 3,026.62 | 100 |
| 300 | 7.9033 | 2,706.16 | 100 |
| 400 | 7.8105 | 2,466.27 | 100 |
| 500 | 7.7313 | 2,278.49 | 100 |
| 600 | 7.8040 | 2,450.34 | 100 |
| 700 | 7.7639 | 2,354.01 | 100 |
| 800 | 7.7426 | 2,304.56 | 100 |
| 900 | 7.7232 | 2,260.10 | 100 |
| 1000 | 7.7207 | 2,254.58 | 100 |
| 1100 | 7.7159 | 2,243.79 | 100 |
| 1200 | 7.7082 | 2,226.53 | 100 |
| 1300 | 7.6752 | 2,154.16 | 100 |

Interpretation:

- The corrected-vocabulary model starts below the old 32k-vocabulary random baseline because its correct random baseline is `ln(8000) = 8.99`.
- Validation improves from 8.2274 at step 100 to 7.6752 at step 1300, a 0.5523 loss-point improvement.
- Step 600 is a visible bump, but the run recovers and continues improving by step 1300.
- The result is not yet a good text-generation model; it is evidence that the corrected model/data contract learns under the shard-native runtime.
- Because vocabulary size changed, v2.8 loss is not a clean apples-to-apples comparison with v2.6. It is the correct baseline going forward.

## Empirical Evidence: Perkunas_v2.6 50M Historical Plateau Recovery

Perkunas_v2.6 50M was the rapid-iteration training line before the vocabulary correction. Unlike a locked benchmark recipe, this run intentionally used interactive continuation experiments to test whether the shard-native trainer could recover from an apparent validation plateau. It should therefore be treated as system-learning evidence and optimizer-search evidence, not as a final controlled comparison against other training stacks.

The run uses:

```text
model parameters:          49,840,448
storage shards:            12
dtype:                     fp16 active compute
master weight dtype:       fp32 canonical shards
optimizer:                 AdamW
gradient clip mode:        global
sequence length:           512
validation batches:        24
storage format:            torch shard files
prefetch:                  cpu
prefetch window:           12
```

The initial training recipe reached a useful descent path and then flattened near validation loss 7.97. Later continuation with stronger learning-rate floors, larger effective token batches, global clipping, and higher-confidence 99-batch validation broke that plateau. By step 1800, the run reached validation loss 7.6628 on held-out data.

Selected held-out validation results:

| Step | Validation Loss | Validation Perplexity | Batches |
| ---: | ---: | ---: | ---: |
| 25 | 10.2502 | 28,288.66 | 24 |
| 100 | 9.3680 | 11,707.35 | 24 |
| 300 | 8.3640 | 4,289.84 | 24 |
| 550 | 8.0784 | 3,224.16 | 24 |
| 700 | 7.9921 | 2,957.57 | 24 |
| 750 | 7.9670 | 2,884.16 | 24 |
| 900 | 7.9860 | 2,939.53 | 24 |
| 950 | 7.9373 | 2,799.82 | 24 |
| 1050 | 7.8904 | 2,671.39 | 24 |
| 1200 | 7.8681 | 2,612.64 | 24 |
| 1300 | 7.8479 | 2,560.46 | 24 |
| 1400 | 7.8305 | 2,516.07 | 24 |
| 1450 | 7.8172 | 2,483.06 | 24 |
| 1475 | 7.8114 | 2,468.58 | 24 |
| 1500 | 7.8159 | 2,479.64 | 24 |
| 1600 | 7.7737 | 2,377.24 | 24 |
| 1625 | 7.7396 | 2,297.67 | 24 |
| 1700 | 7.6640 | 2,130.28 | 99 |
| 1800 | 7.6628 | 2,127.69 | 99 |

Interpretation:

- The run begins near the expected random-token range for a 32,000-token vocabulary and then moves steadily below it.
- The first apparent plateau occurred around steps 750-900, with validation hovering around 7.97-7.99.
- The plateau was not terminal. Later continuation reached a new best validation loss of 7.6628 by step 1800.
- The improvement from step 750 to step 1800 is approximately 0.3042 validation-loss points and about a 26.2% reduction in validation perplexity.
- Training remained shard-native: logs around the new best show `updated_shards=12`, `optimizer_shards_touched=12`, `max_active_param_shards_observed=1`, and `max_active_optimizer_shards_observed=1`.
- The late continuation was aggressive. Global clipping acted as a hard update governor, not merely as an occasional safety check.
- The move to 99 validation batches at steps 1700 and 1800 makes the latest validation points more credible than the earlier 24-batch probes.

This evidence strengthens the learning claim for the shard-native system. It also demonstrates why held-out validation and continuation discipline matter: an apparent flat region was a training-recipe plateau, not proof that the model or architecture had failed. The run also demonstrates why the corrected v2.8 baseline matters: a compatible but oversized output vocabulary can still learn, but it adds avoidable loss-head work and makes early loss interpretation misleading.

## Why Perplexity Is High

A uniform random baseline depends on the model's output vocabulary. The corrected v2.8 runs use an 8,000-token vocabulary:

```text
ln(8000) = 8.99
```

The historical 32,000-token runs used:

```text
ln(32000) = 10.37
```

Early losses around 10 are expected. Loss around 7 is much better than random, but still corresponds to high perplexity. That does not mean the architecture failed. It means the model is early in training and has not yet acquired strong language modeling ability.

At this stage the model first learns:

- Token frequency.
- Punctuation.
- Common whitespace patterns.
- Common words such as "the", "of", and "and".
- Local syntactic rhythm.
- C4-specific distribution shape.

Durable factual knowledge and coherent generation require far more training.

## Serving and Hosting

Perkunasv2 has three serving paths:

1. Shard-native test serving directly from a Perkunasv2 run directory.
2. Hugging Face / vLLM serving of an exported Llama-compatible safetensors package.
3. `kvserve` serving of exported full-checkpoint Perkunas artifacts.

These are intentionally different.

The design principle is:

```text
train in the low-memory streaming format
freeze or copy a checkpoint
export to a standard serving package when latency matters
serve with the best available inference runtime
```

This split is not a concession that the training design failed. It is an explicit separation of concerns. Training needs optimizer state, replay, validation, guarded updates, and bounded residency. Serving needs fast decode, batching, KV cache, and mature API behavior. A single runtime does not have to optimize both phases to prove the training architecture.

### Shard-Native Test Host

The shard-native host lives in:

```text
training/src/perkunas_training/perkunasv2/serve.py
training/src/perkunas_training/perkunasv2/inference.py
```

It uses `PerkunasV2ShardGenerator`, which reads:

```text
run_dir/config.json
run_dir/shards/params/*.pt or *.safetensors
tokenizer_dir/tokenizer.json
```

For each generated token, inference streams through the shard sequence:

```text
load embeddings
compute hidden states
release embeddings

for block_000 to block_023:
  load block
  compute hidden states
  release block

load final_norm
compute final hidden state
release final_norm

load lm_head
compute last-token logits
sample next token
release lm_head
```

This is the same philosophical model as training: no full resident model is required for shard-native inference.

The test host loads two named models:

```text
primary
backup
```

Supported endpoints:

```text
GET  /health
GET  /models
GET  /v1/models
POST /generate
POST /compare
POST /v1/chat/completions
```

The `/generate` endpoint takes:

```json
{
  "model": "primary",
  "prompt": "The future of AI is",
  "max_new_tokens": 16,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "seed": 42
}
```

The `/v1/chat/completions` endpoint accepts OpenAI-style chat bodies. For raw TinyStories-style base models, the server includes a lightweight prompt adapter for common story requests so that chat-shaped input can be converted into a model-native story prefix. This improves test ergonomics but does not make the checkpoint instruction-tuned.

Earlier versions rendered messages literally as:

```text
system: ...
user: ...
assistant:
```

and returns an OpenAI-style response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1777063098,
  "model": "primary",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### Shard-Native Host Command

Example local host command:

```powershell
python training/scripts/serve_perkunasv2.py `
  --primary-run-dir training/runs/Perkunas_v2.9_100m_tinystories `
  --backup-run-dir training/runs/Perkunas_v2.9_100m_tinystories `
  --primary-tokenizer-dir training/tokenizer/perkunas-hf-blend-tokenizer `
  --backup-tokenizer-dir training/tokenizer/perkunas-hf-blend-tokenizer `
  --device cuda `
  --dtype fp16 `
  --max-resident-shards 16 `
  --cache-active-modules `
  --preload-modules `
  --host 127.0.0.1 `
  --port 8010
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/health" -Method Get |
  ConvertTo-Json -Depth 4
```

Generation check:

```powershell
$body = @{
  model = "primary"
  prompt = "Hello, my name is"
  max_new_tokens = 16
  temperature = 0.8
  top_k = 50
  top_p = 0.95
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8010/generate" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 240 |
  ConvertTo-Json -Depth 8
```

Chat completion check:

```powershell
$body = @{
  model = "primary"
  messages = @(
    @{ role = "system"; content = "You are a test model." },
    @{ role = "user"; content = "Say hello." }
  )
  max_tokens = 16
  temperature = 0.8
  top_p = 0.95
  top_k = 50
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:8010/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  -TimeoutSec 240 |
  ConvertTo-Json -Depth 10
```

### Serving Limitations

The current shard-native host is a test host, not a high-throughput production server.

Current limitations:

- Without module caching and preload, it reloads/rebuilds modules repeatedly and is slow.
- A functional KV-cache path exists for local generation, but it is not yet a production decode engine with batching, streaming, paging, or scheduler support.
- It does not stream tokens from `/v1/chat/completions`.
- It has a lightweight chat prompt adapter, not an instruction-tuned chat template.
- Serving directly from a mutable training run is unsafe if training commits while generation is in progress.
- Generation quality is limited by base-model training quality and the absence of instruction tuning.

The important serving safety rule is:

> Serve a stopped run or a copied snapshot, not a run directory actively being mutated by training.

Without snapshot pinning, a request could load some shards from step N and later shards from step N+1 if training commits mid-generation. The production fix is a pinned current-step manifest or immutable snapshot directory.

### Hugging Face / vLLM Export

The preferred high-throughput serving path for models that fit the target inference device is now:

```text
training shard store
  -> export_perkunasv2_hf.py
  -> Hugging Face Llama-compatible safetensors package
  -> vLLM or another standard inference runtime
```

The exporter lives in:

```text
training/src/perkunas_training/perkunasv2/hf_export.py
training/scripts/export_perkunasv2_hf.py
```

It reads `.pt` or `.safetensors` Perkunas parameter shards and writes:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
perkunas_export_manifest.json
README.md
```

The v2.9 100M TinyStories export maps the Perkunas module names to Llama-compatible names:

```text
embeddings.weight                     -> model.embed_tokens.weight
block_i.self_attn.q_proj.weight       -> model.layers.i.self_attn.q_proj.weight
block_i.self_attn.k_proj.weight       -> model.layers.i.self_attn.k_proj.weight
block_i.self_attn.v_proj.weight       -> model.layers.i.self_attn.v_proj.weight
block_i.self_attn.o_proj.weight       -> model.layers.i.self_attn.o_proj.weight
block_i.mlp.gate_up.weight[:mid]      -> model.layers.i.mlp.gate_proj.weight
block_i.mlp.gate_up.weight[mid:]      -> model.layers.i.mlp.up_proj.weight
block_i.mlp.down.weight               -> model.layers.i.mlp.down_proj.weight
final_norm.weight                     -> model.norm.weight
lm_head.weight                        -> lm_head.weight
```

Example export command:

```powershell
python training/scripts/export_perkunasv2_hf.py `
  --run-dir training/runs/Perkunas_v2.9_100m_tinystories `
  --tokenizer-dir training/tokenizer/perkunas-hf-blend-tokenizer `
  --output-dir exports/Perkunas_v2.9_100m_tinystories_vllm `
  --dtype fp16 `
  --overwrite
```

The training run may have used legacy `.pt` shard storage. That does not matter. The exporter is a conversion step: it reads the training storage format and emits `safetensors` because that is the right packaging format for modern serving runtimes.

Example vLLM command in a Linux or WSL environment with vLLM installed:

```bash
vllm serve /mnt/d/LLMProject/exports/Perkunas_v2.9_100m_tinystories_vllm \
  --dtype float16 \
  --served-model-name perkunas-v2.9 \
  --host 127.0.0.1 \
  --port 8011
```

This route may require more inference VRAM than shard-native serving, because vLLM is optimized around resident weights and a high-performance decode engine. That is acceptable within the architecture: Perkunas lowers the memory barrier for training, while exported serving can choose speed over minimal residency when the deployment environment allows it.

### `kvserve` Exported-Artifact Backend

The second serving path is the broader `kvserve` application:

```text
src/kvserve/
```

It exposes OpenAI-compatible routes through FastAPI:

```text
GET  /health
GET  /metrics
GET  /v1/models
POST /v1/chat/completions
POST /v1/chat
POST /v1/embeddings
POST /v1/rerank
POST /v1/reranking
```

The `perkunas` backend in `src/kvserve/backends/perkunas_backend.py` loads exported Perkunas checkpoints through:

```text
PerkunasForCausalLM.from_pretrained(...)
```

This route is not shard-native serving. It is a full-model PyTorch backend intended for exported artifacts. It is useful for integration with the production API layer, model registry, authentication, rate limiting, metrics, and streaming response plumbing.

The strategic direction is:

- Use shard-native serving for diagnostic access to a stopped run directory and for future non-resident serving research.
- Use exported vLLM serving when the model can fit and latency matters.
- Use exported `kvserve` serving when Perkunas needs to integrate with the broader local API layer.
- Add snapshot-pinned shard-native serving to `kvserve` as a future backend.

## Relationship to FSDP

FSDP and Perkunasv2 both reduce memory pressure, but their architecture differs.

FSDP:

```text
starts from a full logical model
shards parameters across distributed ranks
uses collectives
is designed for multi-GPU throughput
benefits from high-bandwidth interconnect
```

Perkunasv2:

```text
does not construct the full model for normal training
does not require distributed ranks
does not require collectives
does not create a global optimizer
makes shard files canonical
streams one active module at a time
stores optimizer state per shard
```

Short version:

> FSDP shards a full model across devices. Perkunasv2 makes shards the model.

FSDP should win on mature multi-GPU datacenter systems. Perkunasv2 is aimed at the hardware regimes where the question is not "how fast can the cluster train?" but "can this model train here at all?"

## Relationship to ZeRO and Offload Systems

DeepSpeed ZeRO, ZeRO-Offload, and ZeRO-Infinity are important prior art. They reduce memory by partitioning or offloading optimizer state, gradients, and parameters.

Perkunasv2 is different in operational shape:

- The shard store is canonical.
- The full model is not constructed during normal training.
- Optimizer state is loaded and saved per shard.
- Training explicitly replays local modules instead of retaining a full graph.
- The runtime is designed first for commodity local hardware.

Perkunasv2 should not be described as "inventing sharding." It is better described as a compact shard-native active-parameter runtime.

## What Is Innovative

The innovation is the combination of constraints and implementation:

1. Shards are canonical model state, not checkpoint fragments.
2. The GPU is treated as a narrow execution window.
3. Optimizer state is local to the active shard.
4. Backward replay makes full-graph residency unnecessary.
5. fp32 master weights preserve update precision while fp16 active compute preserves VRAM.
6. Transaction-staged updates make guarded replay, rollback, and recipe safety practical inside the training loop.
7. The same shard-native philosophy can serve models without full residency.
8. The system is intentionally commodity-first rather than cluster-first.

None of these pieces alone is magic. The value is in the disciplined composition.

## Commercial and Research Value

The defensible value proposition is:

> Perkunasv2 turns low-memory commodity GPUs into useful participants in language-model training and evaluation.

Potential applications:

- Architecture prototyping before cluster training.
- Continued pretraining experiments.
- Small model pretraining.
- Private local model training.
- Workstation AI appliances.
- Academic research without premium accelerator access.
- Domestic hardware utilization.
- Edge and constrained-site adaptation.

Potential products:

- A local shard-native training runtime.
- A hosted benchmark suite for low-memory training.
- A workstation appliance for model experimentation.
- A cloud service that uses mixed commodity GPU fleets.
- A shard-native serving backend for models that exceed single-GPU VRAM.
- A training-operations layer for safe recipe branching, guarded updates, rewind, audit, and recovery around expensive model runs.

The investor-facing claim should remain disciplined:

> Perkunasv2 does not replace H100 training clusters. It expands the market of hardware capable of doing useful model training work.

## Current Limitations

The current limitations are real:

- The best v2.0, v2.6, v2.8, and v2.9 validation losses are still high relative to polished text-generation quality.
- v2.1 has early train evidence but needs validation evidence.
- v2.5 global-clip training has strong early mechanical evidence, v2.8 has held-out validation evidence on a corrected-vocabulary 50M-class model, and v2.9 extends the corrected-vocabulary line to 100M parameters on TinyStories.
- The v2.6 run is an interactive continuation experiment with a now-known 32k-vocabulary mismatch, not a controlled one-recipe benchmark.
- Full global clipping currently requires extra CPU gradient payload handling and is slower than shard-local clipping.
- Periodic global optimizer normalization is newly implemented and needs controlled ablation.
- Guarded step replay is experimental. It can prevent clearly harmful same-batch updates, but if configured too tightly it can overfit to batch noise or skip too many updates.
- Shard-native serving is still a test host; module caching, preload, and KV cache improve latency but do not make it a production decode engine.
- Serving mutable training directories needs snapshot pinning.
- Exported vLLM serving needs a Linux/WSL or server environment with vLLM installed and enough inference memory for resident weights and KV cache.
- Current checkpoint files are markers, not immutable rewindable snapshots.
- Transaction commit can still publish mixed-step shards if the machine crashes during file replacement.
- `.pt` files are convenient but not ideal for high-throughput shard storage; safetensors support now exists and needs long-run benchmarking.
- Prefetching exists but needs systematic benchmarking.
- Locality-preserving shuffle is implemented, but the platform still needs lazy shard caching or larger repacked token shards for the 35k-file Windows workload.
- Boundary activations grow with batch size and sequence length.
- Equivalence tests now cover important shard-native global-clipping behavior, but the suite still needs broader end-to-end parity coverage against full-resident training.
- No mature distributed training mode exists yet.

These are not fatal flaws. They are the engineering agenda.

## Roadmap

Near-term:

1. Continue Perkunas_v2.8 from the corrected-vocabulary baseline and record controlled ablations for learning rate, weight decay, batch size, clipping mode, and periodic global normalization.
2. Benchmark `--grad-clip-mode shard`, `--grad-clip-mode global`, and `--grad-clip-mode shard` plus `--global-optimizer-every` on matched token budgets.
3. Run controlled guarded-step-replay ablations with loose, moderate, and strict tolerances; report intervention rate, skipped updates, tokens/sec, train loss, and validation loss.
4. Preserve full best-validation snapshots manually until immutable checkpoint directories exist.
5. Add real immutable checkpoints under `checkpoints/step_<N>/shards/...` and a `--resume-from-step` path.
6. Add a step commit manifest to detect mixed-step shard states.
7. Add recovery tooling for incomplete commits.
8. Add snapshot-pinned serving and immutable best-validation snapshots.
9. Harden the shard-native KV-cache path and add streamed token responses.
10. Benchmark CPU, GPU, and secondary-GPU prefetch modes.
11. Benchmark safetensors shard storage against legacy `.pt` and evaluate a raw tensor layout if needed.
12. Add pinned CPU staging buffers.
13. Add a lazy LRU packed-shard cache or a repacking tool for fewer/larger token shards.
14. Expand formal tiny-model equivalence tests against full-resident PyTorch.
15. Add better generated text evaluation once loss reaches usable ranges.
16. Validate the HF/vLLM exporter under WSL/Linux vLLM and add a round-trip parity test against the shard-native generator.

Medium-term:

1. Add a `kvserve` shard-native backend.
2. Add model import paths from standard checkpoints back into the shard-native training store.
3. Add multi-GPU staging where GPU0 computes and GPU1 prefetches.
4. Add hardware-profile-based auto tuning.
5. Add benchmark comparisons against PyTorch, ZeRO-Offload, and FSDP.

Long-term:

1. Distributed shard-native training.
2. Remote shard stores.
3. Mixed hardware execution.
4. Production-grade serving for non-resident models.
5. A reproducible public benchmark suite for memory-expanded training.

## Recommended Benchmark Plan

A credible benchmark suite should measure:

- Maximum trainable model size.
- Maximum stable micro-batch at fixed sequence length.
- Tokens/sec.
- Step time.
- Peak CUDA allocation.
- CUDA reserved memory.
- Host RAM usage.
- Disk read bandwidth.
- Disk write bandwidth.
- Validation loss over matched token budgets.
- Resume correctness.
- Crash recovery correctness.
- Serving latency per generated token.
- Serving memory use.

Baselines:

1. Perkunasv2 on RTX 3050 8 GB.
2. Standard PyTorch full-resident training on RTX 3050 8 GB.
3. Standard PyTorch on 16 GB and 24 GB GPUs.
4. DeepSpeed ZeRO-Offload.
5. FSDP on multi-GPU hardware.
6. Perkunasv2 with and without async writes.
7. Perkunasv2 with CPU, GPU, and secondary-GPU prefetch.
8. Perkunasv2 shard-native serving versus exported vLLM serving.

## Appendix A: Perkunas_v2.1 Config

```json
{
  "vocab_size": 32000,
  "hidden_size": 1024,
  "num_layers": 24,
  "num_heads": 16,
  "intermediate_size": 2816,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "norm_type": "rmsnorm",
  "rms_norm_eps": 1e-5,
  "activation_function": "swiglu",
  "tied_embeddings": false,
  "dropout": 0.0,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

## Appendix B: Perkunas_v2.1 Training Command

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/Perkunas_v2.1 `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size 32 `
  --gradient-accumulation-steps 1 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format safetensors `
  --device cuda `
  --optimizer adamw `
  --learning-rate 3e-5 `
  --weight-decay 0.05 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 1.0 `
  --lr-schedule tokens `
  --warmup-tokens 0 `
  --decay-tokens 100000000 `
  --min-lr-ratio 1.0 `
  --max-steps 3000 `
  --save-every 500 `
  --validate-every 100 `
  --max-validation-batches 128 `
  --max-resident-shards 500 `
  --prefetch-shards cpu `
  --prefetch-window 500 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 25 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 4
```

## Appendix C: Perkunas_v2.5 Config

```json
{
  "vocab_size": 32000,
  "hidden_size": 896,
  "num_layers": 24,
  "num_heads": 14,
  "intermediate_size": 2432,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "norm_type": "rmsnorm",
  "rms_norm_eps": 1e-5,
  "activation_function": "swiglu",
  "tied_embeddings": false,
  "dropout": 0.0,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

## Appendix D: Perkunas_v2.5 Global-Clip Training Command

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/Perkunas_v2.5_test `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size 32 `
  --gradient-accumulation-steps 4 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format torch `
  --device cuda `
  --optimizer adamw `
  --learning-rate 1e-5 `
  --weight-decay 0.05 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 1.0 `
  --grad-clip-mode global `
  --lr-schedule tokens `
  --warmup-tokens 0 `
  --decay-tokens 500000000 `
  --min-lr-ratio 0.5 `
  --max-steps 25000 `
  --save-every 100 `
  --validate-every 100 `
  --max-validation-batches 50 `
  --no-shuffle-train `
  --max-resident-shards 27 `
  --prefetch-shards cpu `
  --prefetch-window 27 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 25 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 8
```

## Appendix E: Perkunas_v2.6 50M Config

```json
{
  "vocab_size": 32000,
  "hidden_size": 448,
  "num_layers": 9,
  "num_heads": 7,
  "intermediate_size": 1152,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "norm_type": "rmsnorm",
  "rms_norm_eps": 1e-5,
  "activation_function": "swiglu",
  "tied_embeddings": false,
  "dropout": 0.0,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

## Appendix F: Perkunas_v2.6 Plateau-Recovery Continuation Command

This command captures the representative plateau-recovery continuation used after the 50M run had flattened near validation loss 7.97. The exact run directory contains an interactive history of prior settings, so this should be treated as a reproducible continuation recipe rather than the original step-1 command.

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/Perkunas_v2.6_50m_run4_lr3e5_agile `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size 64 `
  --gradient-accumulation-steps 24 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format torch `
  --device cuda `
  --optimizer adamw `
  --learning-rate 6e-5 `
  --weight-decay 0.05 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 1.5 `
  --grad-clip-mode global `
  --lr-schedule tokens `
  --warmup-tokens 26214400 `
  --decay-tokens 500000000 `
  --min-lr-ratio 0.6 `
  --max-steps 5000 `
  --save-every 25 `
  --validate-every 25 `
  --max-validation-batches 24 `
  --shuffle-train `
  --max-resident-shards 12 `
  --prefetch-shards cpu `
  --prefetch-window 12 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 25 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 16
```

## Appendix G: Perkunas_v2.8 Correct-Vocab Config

```json
{
  "vocab_size": 8000,
  "hidden_size": 640,
  "num_layers": 9,
  "num_heads": 10,
  "intermediate_size": 1792,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "norm_type": "rmsnorm",
  "rms_norm_eps": 1e-5,
  "activation_function": "swiglu",
  "tied_embeddings": false,
  "dropout": 0.0,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

## Appendix H: Perkunas_v2.8 Correct-Vocab Training Commands

Initial corrected-vocabulary run:

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/Perkunas_v2.8_50m_correct_vocab `
  --active-run-dir E:\Perkunas_v2.8_50m_correct_vocab_active `
  --durable-flush-every 1000 `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size 192 `
  --gradient-accumulation-steps 4 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format torch `
  --device cuda `
  --optimizer adamw `
  --learning-rate 3e-5 `
  --weight-decay 0.05 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 0.5 `
  --grad-clip-mode global `
  --lr-schedule tokens `
  --warmup-tokens 26214400 `
  --decay-tokens 5000000000 `
  --min-lr-ratio 0.75 `
  --max-steps 20000 `
  --save-every 100 `
  --validate-every 100 `
  --max-validation-batches 100 `
  --shuffle-train `
  --max-resident-shards 12 `
  --prefetch-shards cpu `
  --prefetch-window 12 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 100 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 16
```

Experimental periodic global-normalizer continuation:

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/Perkunas_v2.8_50m_correct_vocab `
  --active-run-dir E:\Perkunas_v2.8_50m_correct_vocab_active `
  --durable-flush-every 1000 `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size 32 `
  --gradient-accumulation-steps 8 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format torch `
  --device cuda `
  --optimizer adamw `
  --learning-rate 2e-5 `
  --weight-decay 0.01 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 0.5 `
  --grad-clip-mode shard `
  --lr-schedule tokens `
  --warmup-tokens 26214400 `
  --decay-tokens 5000000000 `
  --min-lr-ratio 0.75 `
  --max-steps 20000 `
  --save-every 100 `
  --validate-every 100 `
  --max-validation-batches 100 `
  --shuffle-train `
  --max-resident-shards 12 `
  --prefetch-shards cpu `
  --prefetch-window 12 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 100 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 16 `
  --global-optimizer-every 50 `
  --global-optimizer-blend 0.25 `
  --global-optimizer-min-scale 0.5 `
  --global-optimizer-max-scale 2.0
```

## Appendix I: Shard-Native Serving Command

```powershell
python training/scripts/serve_perkunasv2.py `
  --primary-run-dir training/runs/Perkunas_v2.9_100m_tinystories `
  --backup-run-dir training/runs/Perkunas_v2.9_100m_tinystories `
  --primary-tokenizer-dir training/tokenizer/perkunas-hf-blend-tokenizer `
  --backup-tokenizer-dir training/tokenizer/perkunas-hf-blend-tokenizer `
  --device cuda `
  --dtype fp16 `
  --max-resident-shards 16 `
  --cache-active-modules `
  --preload-modules `
  --host 127.0.0.1 `
  --port 8010
```

## Conclusion

Perkunasv2 is a practical attempt to change the memory economics of language-model training.

The core result is not that a small GPU beats a datacenter cluster. The core result is that a shard-native runtime can train model shapes that would otherwise be blocked by full-model training residency. The v2.0 run proved the training loop learns on held-out data. The v2.1 run tested a larger model shape with fp32 master shards and safer precision. The v2.5 line tested the most important scientific control added so far: full-model global gradient clipping while still keeping the model shard-native. The v2.6 50M run exposed useful plateau-recovery behavior and the cost of a tokenizer-vocabulary mismatch. The v2.8 corrected-vocabulary 50M run became the recommended small-model baseline. The v2.9 100M TinyStories run demonstrates a larger corrected-vocabulary model reaching validation loss 3.5135 by step 3000, while also proving that the trained shard store can be exported into a standard Hugging Face / vLLM serving package.

The architecture is scientifically testable, commercially relevant, and technically unfinished in the right ways. The next proof points are controlled v2.9 ablations, periodic global-normalizer benchmarking, real immutable checkpoints, crash recovery, snapshot-pinned serving, vLLM export validation, and benchmarked comparisons.

Perkunasv2's contribution can be stated simply:

> Make the shards the model, make memory a first-class design boundary, and let commodity hardware do training work it otherwise could not do.
