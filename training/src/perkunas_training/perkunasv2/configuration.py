from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PerkunasV2Config:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"
    rms_norm_eps: float = 1e-5
    activation_function: str = "swiglu"
    tied_embeddings: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_json(cls, path: str | Path) -> "PerkunasV2Config":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"model_type": "perkunasv2-shard-native"}

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.norm_type != "rmsnorm":
            raise ValueError("Perkunasv2 shard-native training currently implements rmsnorm")
        if self.activation_function != "swiglu":
            raise ValueError("Perkunasv2 shard-native training currently implements swiglu")
        if self.tied_embeddings:
            raise ValueError(
                "tied_embeddings=true requires cross-shard gradient merging; set tied_embeddings=false"
            )
        if self.dropout != 0.0 or self.attention_dropout != 0.0:
            raise ValueError(
                "dropout must be 0 for deterministic shard-local recomputation in Perkunasv2"
            )


@dataclass(slots=True)
class PerkunasShardTrainingConfig:
    run_dir: str
    data_dir: str
    val_data_dir: str | None = None
    active_run_dir: str | None = None
    durable_flush_every: int = 0
    seq_len: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    dtype: str = "fp16"
    master_weight_dtype: str = "compute"
    shard_storage_format: str = "torch"
    storage_shard_count: int = 0
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 0.0
    grad_clip_mode: str = "shard"
    global_optimizer_every: int = 0
    global_optimizer_blend: float = 0.25
    global_optimizer_min_scale: float = 0.5
    global_optimizer_max_scale: float = 2.0
    guarded_step_replay: bool = False
    guard_replay_max_replays: int = 0
    guard_replay_loss_tolerance: float = 0.0
    guard_replay_loss_tolerance_ratio: float = 0.0
    guard_replay_lr_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    guard_replay_grad_norm_scales: tuple[float, ...] = (1.0,)
    guard_replay_on_exhaust: str = "accept"
    lr_schedule: str = "steps"
    warmup_steps: int = 100
    warmup_tokens: int = 0
    decay_tokens: int = 0
    min_lr_ratio: float = 0.1
    max_steps: int = 1000
    save_every: int = 100
    validate_every: int = 100
    max_validation_batches: int = 8
    allow_train_validation_fallback: bool = False
    shuffle_train: bool = True
    max_resident_shards: int = 1
    cache_active_modules: bool = False
    prefetch_mode: str = "off"
    prefetch_window: int = 0
    prefetch_optimizer_shards: bool = True
    prefetch_device: str | None = None
    trace_storage_mode: str = "cpu"
    trace_storage_device: str | None = None
    clear_cuda_cache_between_shards: bool = True
    shard_log_every: int = 1
    trainer_state_every: int = 1
    lm_head_chunk_tokens: int = 0
    timing_sync_cuda: bool = False
    async_shard_writes: bool = False
    max_pending_shard_writes: int = 4
    device: str = "cuda"
    seed: int = 1337

    def validate(self) -> None:
        if self.dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be one of fp32, fp16, bf16")
        if self.master_weight_dtype not in {"compute", "fp32", "fp16", "bf16"}:
            raise ValueError("master_weight_dtype must be one of compute, fp32, fp16, bf16")
        if self.shard_storage_format not in {"torch", "safetensors"}:
            raise ValueError("shard_storage_format must be one of torch, safetensors")
        if self.storage_shard_count < 0:
            raise ValueError("storage_shard_count must be >= 0")
        if self.optimizer not in {"adamw", "lion", "adafactor"}:
            raise ValueError("optimizer must be one of adamw, lion, adafactor")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.grad_clip_mode not in {"shard", "global"}:
            raise ValueError("grad_clip_mode must be one of shard, global")
        if self.global_optimizer_every < 0:
            raise ValueError("global_optimizer_every must be >= 0")
        if not 0.0 <= self.global_optimizer_blend <= 1.0:
            raise ValueError("global_optimizer_blend must be between 0 and 1")
        if self.global_optimizer_min_scale <= 0:
            raise ValueError("global_optimizer_min_scale must be > 0")
        if self.global_optimizer_max_scale < self.global_optimizer_min_scale:
            raise ValueError("global_optimizer_max_scale must be >= global_optimizer_min_scale")
        if self.guard_replay_max_replays < 0:
            raise ValueError("guard_replay_max_replays must be >= 0")
        if self.guard_replay_loss_tolerance < 0:
            raise ValueError("guard_replay_loss_tolerance must be >= 0")
        if self.guard_replay_loss_tolerance_ratio < 0:
            raise ValueError("guard_replay_loss_tolerance_ratio must be >= 0")
        if not self.guard_replay_lr_scales:
            raise ValueError("guard_replay_lr_scales must not be empty")
        if not self.guard_replay_grad_norm_scales:
            raise ValueError("guard_replay_grad_norm_scales must not be empty")
        if any(scale <= 0 for scale in self.guard_replay_lr_scales):
            raise ValueError("guard_replay_lr_scales values must be > 0")
        if any(scale <= 0 for scale in self.guard_replay_grad_norm_scales):
            raise ValueError("guard_replay_grad_norm_scales values must be > 0")
        if self.guard_replay_on_exhaust not in {"accept", "skip"}:
            raise ValueError("guard_replay_on_exhaust must be one of accept, skip")
        if self.lr_schedule not in {"steps", "tokens"}:
            raise ValueError("lr_schedule must be one of steps, tokens")
        if self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be >= 0")
        if self.decay_tokens < 0:
            raise ValueError("decay_tokens must be >= 0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        if self.lr_schedule == "tokens" and self.decay_tokens <= 0:
            raise ValueError("decay_tokens must be > 0 when lr_schedule=tokens")
        if self.max_resident_shards < 1:
            raise ValueError("max_resident_shards must be >= 1")
        if self.prefetch_mode not in {"off", "cpu", "gpu", "secondary-gpu"}:
            raise ValueError("prefetch_mode must be one of off, cpu, gpu, secondary-gpu")
        if self.prefetch_window < 0:
            raise ValueError("prefetch_window must be >= 0")
        if self.trace_storage_mode not in {"cpu", "gpu", "secondary-gpu"}:
            raise ValueError("trace_storage_mode must be one of cpu, gpu, secondary-gpu")
        if self.trace_storage_mode == "cpu" and self.trace_storage_device is not None:
            raise ValueError("trace_storage_device requires trace_storage_mode gpu or secondary-gpu")
        if self.shard_log_every < 0:
            raise ValueError("shard_log_every must be >= 0")
        if self.trainer_state_every < 1:
            raise ValueError("trainer_state_every must be >= 1")
        if self.lm_head_chunk_tokens < 0:
            raise ValueError("lm_head_chunk_tokens must be >= 0")
        if self.max_pending_shard_writes < 1:
            raise ValueError("max_pending_shard_writes must be >= 1")
        if self.durable_flush_every < 0:
            raise ValueError("durable_flush_every must be >= 0")
