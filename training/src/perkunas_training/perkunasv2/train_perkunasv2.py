from __future__ import annotations

import argparse
import json
from pathlib import Path

from perkunas_training.perkunasv2.configuration import (
    PerkunasShardTrainingConfig,
    PerkunasV2Config,
)
from perkunas_training.perkunasv2.trainer import ShardStreamingTrainer, initialize_run


def parse_float_tuple(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Perkunasv2 shard-native active-parameter trainer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init-shards", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--config", help="Perkunasv2 model config JSON for --init-shards")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--active-run-dir",
        help=(
            "Optional RAM-disk or fast working copy. --run-dir remains the durable archive; "
            "training reads and writes this active directory, then publishes durable flushes."
        ),
    )
    parser.add_argument(
        "--durable-flush-every",
        type=int,
        default=0,
        help=(
            "Publish active shards back to --run-dir every N steps. "
            "0 publishes at --save-every and at the final step when --active-run-dir is set."
        ),
    )
    parser.add_argument("--data-dir", default="training/data/perkunas_pilot/tokenized")
    parser.add_argument("--val-data-dir")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--master-weight-dtype",
        choices=["compute", "fp32", "fp16", "bf16"],
        default="compute",
        help=(
            "Canonical shard storage dtype. compute preserves historical behavior; "
            "fp32 keeps master weights in fp32 while --dtype controls active compute."
        ),
    )
    parser.add_argument(
        "--shard-storage-format",
        choices=["torch", "safetensors"],
        default="torch",
        help=(
            "On-disk parameter/optimizer shard format. torch keeps legacy .pt files; "
            "safetensors writes pickle-free .safetensors shards while still reading legacy .pt."
        ),
    )
    parser.add_argument(
        "--storage-shard-count",
        type=int,
        default=0,
        help=(
            "Target number of physical safetensors files for parameters and optimizer state. "
            "0 preserves one storage file per logical compute shard."
        ),
    )
    parser.add_argument(
        "--init-weight-dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Initial parameter shard dtype used by --init-shards.",
    )
    parser.add_argument("--optimizer", choices=["adamw", "lion", "adafactor"], default="adamw")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument(
        "--grad-clip-mode",
        choices=["shard", "global"],
        default="shard",
        help=(
            "shard clips each active shard independently; global first measures the "
            "full-model gradient norm across all shards, then applies one shared clip scale."
        ),
    )
    parser.add_argument(
        "--global-optimizer-every",
        type=int,
        default=0,
        help=(
            "Every N steps, run a memory-light global optimizer controller pass. "
            "0 disables it. The pass collects shard gradients on CPU, computes scalar "
            "per-shard normalization factors, then still applies the regular shard-local optimizer."
        ),
    )
    parser.add_argument(
        "--global-optimizer-blend",
        type=float,
        default=0.25,
        help="Blend strength for periodic global optimizer shard normalization; 0 disables scaling, 1 applies full normalization.",
    )
    parser.add_argument(
        "--global-optimizer-min-scale",
        type=float,
        default=0.5,
        help="Lower clamp for per-shard global optimizer gradient scale.",
    )
    parser.add_argument(
        "--global-optimizer-max-scale",
        type=float,
        default=2.0,
        help="Upper clamp for per-shard global optimizer gradient scale.",
    )
    parser.add_argument(
        "--guarded-step-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: stage each update, replay the same batch loss, and retry with "
            "safer LR/grad-norm scales if the candidate update exceeds the configured loss guard."
        ),
    )
    parser.add_argument(
        "--guard-replay-max-replays",
        type=int,
        default=0,
        help="Maximum guarded step retries after the first candidate update.",
    )
    parser.add_argument(
        "--guard-replay-loss-tolerance",
        type=float,
        default=0.0,
        help="Absolute same-batch loss increase allowed before a guarded replay is rejected.",
    )
    parser.add_argument(
        "--guard-replay-loss-tolerance-ratio",
        type=float,
        default=0.0,
        help="Relative same-batch loss increase allowed before a guarded replay is rejected.",
    )
    parser.add_argument(
        "--guard-replay-lr-scales",
        type=parse_float_tuple,
        default=(1.0, 0.5, 0.25),
        help="Comma-separated LR scales used by guarded replay attempts, e.g. 1.0,0.5,0.25.",
    )
    parser.add_argument(
        "--guard-replay-grad-norm-scales",
        type=parse_float_tuple,
        default=(1.0,),
        help="Comma-separated max-grad-norm scales used by guarded replay attempts.",
    )
    parser.add_argument(
        "--guard-replay-on-exhaust",
        choices=["accept", "skip"],
        default="accept",
        help="What to do when all guarded replay attempts are rejected.",
    )
    parser.add_argument("--lr-schedule", choices=["steps", "tokens"], default="steps")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--decay-tokens", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--max-validation-batches", type=int, default=8)
    parser.add_argument("--allow-train-validation-fallback", action="store_true")
    parser.add_argument(
        "--shuffle-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shuffle training samples with PyTorch's RandomSampler. Disable for very large "
            "many-file mmap datasets to stream shards sequentially and avoid random I/O."
        ),
    )
    parser.add_argument("--max-resident-shards", type=int, default=1)
    parser.add_argument(
        "--cache-active-modules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep constructed shard modules resident between passes. Experimental; "
            "can increase VRAM and slow small GPUs if the cache churns."
        ),
    )
    parser.add_argument(
        "--prefetch-shards",
        choices=["off", "cpu", "gpu", "secondary-gpu"],
        default="off",
        help=(
            "Prefetch upcoming parameter shards. cpu stages tensors in host RAM; "
            "gpu stages on the training GPU; secondary-gpu stages on another CUDA device."
        ),
    )
    parser.add_argument(
        "--prefetch-window",
        type=int,
        default=0,
        help="Number of upcoming shards to prefetch; 0 uses --max-resident-shards.",
    )
    parser.add_argument(
        "--prefetch-optimizer-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also prefetch optimizer shards during backward/update.",
    )
    parser.add_argument(
        "--prefetch-device",
        help="Explicit staging device for --prefetch-shards secondary-gpu, e.g. cuda:1.",
    )
    parser.add_argument(
        "--trace-storage",
        choices=["cpu", "gpu", "secondary-gpu"],
        default="cpu",
        help=(
            "Where to stage forward boundaries and backward boundary gradients. "
            "cpu preserves the low-VRAM default; gpu stages on the training GPU; "
            "secondary-gpu stages on another CUDA device."
        ),
    )
    parser.add_argument(
        "--trace-storage-device",
        help=(
            "Explicit staging device for --trace-storage secondary-gpu, e.g. cuda:1. "
            "Use --trace-storage gpu to enable same-GPU trace staging."
        ),
    )
    parser.add_argument("--clear-cuda-cache-between-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shard-log-every",
        type=int,
        default=1,
        help="Log every shard update every N steps; use 0 to disable per-shard logs.",
    )
    parser.add_argument(
        "--trainer-state-every",
        type=int,
        default=1,
        help=(
            "Compatibility option retained for older commands; trainer_state.json is now "
            "written after every committed step so resume metadata tracks shard updates."
        ),
    )
    parser.add_argument(
        "--lm-head-chunk-tokens",
        type=int,
        default=0,
        help="Chunk lm_head loss/backward by token count; 0 keeps the full logits path.",
    )
    parser.add_argument(
        "--timing-sync-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize CUDA around timed kernel regions for more exact timing. "
            "Useful for short diagnostics; it can slow training."
        ),
    )
    parser.add_argument(
        "--async-shard-writes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write parameter/optimizer shards on a background writer with atomic replace.",
    )
    parser.add_argument(
        "--max-pending-shard-writes",
        type=int,
        default=4,
        help="Maximum queued async shard writes before throttling.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.init_shards:
        if not args.config:
            raise SystemExit("--config is required with --init-shards")
        result = initialize_run(
            args.run_dir,
            args.config,
            seed=args.seed,
            shard_storage_format=args.shard_storage_format,
            storage_shard_count=args.storage_shard_count,
            initial_weight_dtype=args.init_weight_dtype,
        )
        print(json.dumps(result, indent=2))
        return

    model_config = PerkunasV2Config.from_json(Path(args.run_dir) / "config.json")
    train_config = PerkunasShardTrainingConfig(
        run_dir=args.run_dir,
        active_run_dir=args.active_run_dir,
        durable_flush_every=args.durable_flush_every,
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dtype=args.dtype,
        master_weight_dtype=args.master_weight_dtype,
        shard_storage_format=args.shard_storage_format,
        storage_shard_count=args.storage_shard_count,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        max_grad_norm=args.max_grad_norm,
        grad_clip_mode=args.grad_clip_mode,
        global_optimizer_every=args.global_optimizer_every,
        global_optimizer_blend=args.global_optimizer_blend,
        global_optimizer_min_scale=args.global_optimizer_min_scale,
        global_optimizer_max_scale=args.global_optimizer_max_scale,
        guarded_step_replay=args.guarded_step_replay,
        guard_replay_max_replays=args.guard_replay_max_replays,
        guard_replay_loss_tolerance=args.guard_replay_loss_tolerance,
        guard_replay_loss_tolerance_ratio=args.guard_replay_loss_tolerance_ratio,
        guard_replay_lr_scales=args.guard_replay_lr_scales,
        guard_replay_grad_norm_scales=args.guard_replay_grad_norm_scales,
        guard_replay_on_exhaust=args.guard_replay_on_exhaust,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        warmup_tokens=args.warmup_tokens,
        decay_tokens=args.decay_tokens,
        min_lr_ratio=args.min_lr_ratio,
        max_steps=args.max_steps,
        save_every=args.save_every,
        validate_every=args.validate_every,
        max_validation_batches=args.max_validation_batches,
        allow_train_validation_fallback=args.allow_train_validation_fallback,
        shuffle_train=args.shuffle_train,
        max_resident_shards=args.max_resident_shards,
        cache_active_modules=args.cache_active_modules,
        prefetch_mode=args.prefetch_shards,
        prefetch_window=args.prefetch_window,
        prefetch_optimizer_shards=args.prefetch_optimizer_shards,
        prefetch_device=args.prefetch_device,
        trace_storage_mode=args.trace_storage,
        trace_storage_device=args.trace_storage_device,
        clear_cuda_cache_between_shards=args.clear_cuda_cache_between_shards,
        shard_log_every=args.shard_log_every,
        trainer_state_every=args.trainer_state_every,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        timing_sync_cuda=args.timing_sync_cuda,
        async_shard_writes=args.async_shard_writes,
        max_pending_shard_writes=args.max_pending_shard_writes,
        device=args.device,
        seed=args.seed,
    )
    trainer = ShardStreamingTrainer(model_config, train_config)
    if args.train:
        print(json.dumps(trainer.train(), indent=2))
    elif args.validate:
        print(json.dumps(trainer.validate(max_batches=args.max_validation_batches), indent=2))


if __name__ == "__main__":
    main()
