# Perkunasv2 H200 One-Hour Burn Plan

This plan is for a short x8 H200 rental where the goal is to learn as much as possible quickly. The current trainer is single-GPU per process, so the correct way to use an x8 box today is parallel experimentation: several isolated training runs, each bound to its own GPU, plus one two-GPU run to test secondary-GPU prefetch.

## What This Tests

- Whether the shard-native trainer runs cleanly on H200-class CUDA hardware.
- How throughput changes as micro-batch size increases beyond RTX 3050 limits.
- Whether CPU, same-GPU, or secondary-GPU prefetch wins on datacenter I/O.
- Whether async shard writes and transactional shard updates remain stable under higher throughput.
- Whether validation continues decreasing under the same held-out validation directory.

## Before Starting The Rental Clock

Have these ready before renting:

- The repo on the cloud machine.
- The current Perkunasv2 run directory copied to fast scratch NVMe.
- Training and validation tokenized data copied to fast scratch NVMe.
- A shell with the package import path working.

Recommended scratch layout:

```text
/scratch/perkunasv2_base
/scratch/data/perkunasv2_c4_tokenized_bigshards
/scratch/data/perkunasv2_c4_val_small
/scratch/perkunasv2_h200_burn
```

## Launch Command

From the repo root:

```bash
BASE_RUN_DIR=/scratch/perkunasv2_base \
DATA_DIR=/scratch/data/perkunasv2_c4_tokenized_bigshards \
VAL_DATA_DIR=/scratch/data/perkunasv2_c4_val_small \
WORK_ROOT=/scratch/perkunasv2_h200_burn \
TARGET_STEP=1000 \
bash training/scripts/h200_burn_sweep.sh
```

If the source checkpoint is already past step 1000, raise `TARGET_STEP`. This value is absolute, not "additional steps."

## Experiment Layout

The script uses all eight GPUs:

| Physical GPU | Experiment | Purpose |
| --- | --- | --- |
| 0 | baseline, micro-batch 128, no prefetch | Control run |
| 1 | CPU prefetch, micro-batch 128 | Hide disk/deserialization without using VRAM for staging |
| 2 | GPU parameter prefetch, micro-batch 128 | Test same-GPU parameter staging |
| 3 | GPU parameter+optimizer prefetch, micro-batch 128 | Test full same-GPU staging pressure |
| 4 | CPU prefetch, micro-batch 256 | Larger batch CPU-prefetch throughput |
| 5 | GPU parameter prefetch, micro-batch 256 | Larger batch same-GPU staging |
| 6 + 7 | secondary-GPU prefetch, micro-batch 256 | Stage future shards on a spare H200 |

Each experiment clones the base run into a separate run directory. Do not run multiple trainers against the same run directory.

## What To Watch

Good signs:

- `validation loss` continues decreasing.
- `updated_shards` and `optim_shards` remain near the full shard count.
- `tokens/sec` increases versus RTX 3050.
- `update_sec` decreases or stays stable as batch size increases.
- `cached_*` and `pending_*` fields are populated when prefetch is enabled.

Bad signs:

- OOM on high micro-batch runs.
- Same-GPU optimizer prefetch slower than CPU prefetch. This can happen because AdamW state is large and competes with compute/allocator traffic.
- Validation stalls or rises across multiple checkpoints.
- Any run reports zero validation batches.

## Quick Decision Rule

After 15-20 minutes, keep the best two or three runs and stop the rest if cost matters. Favor the run with the best combination of:

- lowest validation loss at the same step,
- highest sustained `tokens/sec`,
- lowest `update_sec`,
- no memory instability,
- clean checkpoint writes.

If secondary-GPU prefetch wins clearly, it becomes the next architecture direction. If CPU prefetch wins, the next priority is pinned-memory staging and safetensors/raw tensor loading. If no prefetch mode wins, the bottleneck is likely not disk deserialization anymore; inspect compute and Python overhead next.
