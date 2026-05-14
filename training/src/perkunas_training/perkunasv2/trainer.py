from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from perkunas_training.perkunasv2.configuration import (
    PerkunasShardTrainingConfig,
    PerkunasV2Config,
)
from perkunas_training.perkunasv2.shard_store import (
    ParameterShardStore,
    ShardStepTransaction,
    shard_names,
)
from perkunas_training.train.dataset import LocalityPreservingPackedTokenDataset, PackedTokenDataset
from perkunas_training.train.device import print_cuda_diagnostics
from perkunas_training.utils.io import ensure_dir
from perkunas_training.utils.random import seed_everything


@dataclass(slots=True)
class MicroBatchTrace:
    input_ids: torch.Tensor
    labels: torch.Tensor
    boundaries: list[torch.Tensor]
    norm_output: torch.Tensor
    loss: float | None


@dataclass(slots=True)
class StepActivity:
    updated_shards: int = 0
    optimizer_shards_touched: int = 0
    max_active_param_shards_observed: int = 0
    max_active_optimizer_shards_observed: int = 0

    def observe(self, residency: dict[str, Any]) -> None:
        self.max_active_param_shards_observed = max(
            self.max_active_param_shards_observed,
            len(residency["active_param_shards"]),
        )
        self.max_active_optimizer_shards_observed = max(
            self.max_active_optimizer_shards_observed,
            len(residency["active_optimizer_shards"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "updated_shards": self.updated_shards,
            "optimizer_shards_touched": self.optimizer_shards_touched,
            "max_active_param_shards_observed": self.max_active_param_shards_observed,
            "max_active_optimizer_shards_observed": self.max_active_optimizer_shards_observed,
        }


@dataclass(slots=True)
class ShardGradientPayload:
    shard_name: str
    param_grads: dict[str, torch.Tensor]
    grad_norm: float
    grad_norm_sq: float
    parameter_count: int


@dataclass(slots=True)
class StepUpdateResult:
    train_loss: float
    backward_elapsed: float
    commit_elapsed: float
    prefetch_elapsed: float
    committed_update: bool
    effective_lr: float


class ShardStreamingTrainer:
    def __init__(
        self,
        model_config: PerkunasV2Config,
        train_config: PerkunasShardTrainingConfig,
    ) -> None:
        model_config.validate()
        train_config.validate()
        self.model_config = model_config
        self.train_config = train_config
        self.device = select_shard_device(train_config.device)
        self.dtype = dtype_for_device(train_config.dtype, self.device)
        self.trace_storage_device = self._resolve_trace_storage_device()
        self.store = ParameterShardStore(
            train_config.run_dir,
            active_run_dir=train_config.active_run_dir,
            durable_flush_every=train_config.durable_flush_every,
            config=model_config,
            max_resident_shards=train_config.max_resident_shards,
            clear_cuda_cache_between_shards=train_config.clear_cuda_cache_between_shards,
            async_shard_writes=train_config.async_shard_writes,
            max_pending_shard_writes=train_config.max_pending_shard_writes,
            prefetch_mode=train_config.prefetch_mode,
            prefetch_window=train_config.prefetch_window,
            prefetch_optimizer_shards=train_config.prefetch_optimizer_shards,
            prefetch_device=train_config.prefetch_device,
            storage_format=train_config.shard_storage_format,
            storage_shard_count=train_config.storage_shard_count,
            cache_active_modules=train_config.cache_active_modules,
        )
        seed_everything(train_config.seed, deterministic=False)
        self.run_dir = ensure_dir(self.store.run_dir)
        self.log_path = self.run_dir / "train_log.jsonl"
        self._last_step_grad_norms: list[float] = []
        self._last_step_global_grad_norm: float | None = None
        self._last_step_global_grad_clip_scale: float | None = None
        self._last_global_optimizer_stats: dict[str, Any] | None = None
        self._last_guarded_step_replay_stats: dict[str, Any] | None = None
        self._last_step_activity = StepActivity()

    def train(self) -> dict[str, Any]:
        state = self.store.load_trainer_state()
        state = self._reconcile_state_with_optimizer_shards(state)
        restore_rng_state_json(state.get("rng_state"))
        self.store.discard_stale_transactions()
        start_step = int(state.get("global_step", 0))
        loader = self._loader("train", shuffle=self.train_config.shuffle_train)
        iterator = infinite_loader(loader)
        latest_validation_loss = state.get("latest_validation_loss")
        last_log = time.perf_counter()

        for step in range(start_step + 1, self.train_config.max_steps + 1):
            self.store.reset_timings()
            step_start = time.perf_counter()
            tokens_seen_before = int(state.get("tokens_seen", 0))
            traces: list[MicroBatchTrace] = []
            tokens_this_step = 0
            data_load_elapsed = 0.0
            forward_compute_elapsed = 0.0
            forward_trace_start = time.perf_counter()
            for _ in range(self.train_config.gradient_accumulation_steps):
                data_load_start = time.perf_counter()
                input_ids, labels = next(iterator)
                data_load_elapsed += time.perf_counter() - data_load_start
                input_ids = input_ids[:, : self.train_config.seq_len]
                labels = labels[:, : self.train_config.seq_len]
                forward_compute_start = time.perf_counter()
                trace = self.forward_trace(input_ids, labels, training=True, compute_loss=False)
                forward_compute_elapsed += time.perf_counter() - forward_compute_start
                traces.append(trace)
                tokens_this_step += int(input_ids.numel())
            forward_trace_elapsed = time.perf_counter() - forward_trace_start

            tokens_seen_after = tokens_seen_before + tokens_this_step
            lr = learning_rate_for_update(self.train_config, step, tokens_seen_after)
            update_start = time.perf_counter()
            update_result = self._run_step_update(traces, lr=lr, step=step)
            update_elapsed = time.perf_counter() - update_start
            train_loss = update_result.train_loss
            now = time.perf_counter()
            step_elapsed = now - step_start
            elapsed = max(1e-9, now - last_log)
            tokens_per_sec = tokens_this_step / elapsed
            last_log = now

            state = {
                "global_step": step,
                "tokens_seen": tokens_seen_after,
                "optimizer_step": int(state.get("optimizer_step", 0))
                + (1 if update_result.committed_update else 0),
                "scheduler_state": {
                    "learning_rate": update_result.effective_lr,
                    "base_learning_rate": lr,
                    "lr_schedule": self.train_config.lr_schedule,
                    "tokens_seen": tokens_seen_after,
                },
                "latest_validation_loss": latest_validation_loss,
                "config_hash": self.model_config.stable_hash(),
            }
            self.store.save_trainer_state(state)
            timing_breakdown = self.store.timing_snapshot()

            latest_validation_loss = maybe_float(latest_validation_loss)
            if step % self.train_config.validate_every == 0:
                metrics = self.validate(max_batches=self.train_config.max_validation_batches)
                latest_validation_loss = metrics["val_loss"]
                state["latest_validation_loss"] = latest_validation_loss
                self.store.save_trainer_state(state)
                self._append_log({"step": step, **metrics})
            if step % self.train_config.save_every == 0:
                self._write_checkpoint_marker(step)

            self._append_log(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "lr": update_result.effective_lr,
                    "base_lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "step_seconds": step_elapsed,
                    "forward_trace_seconds": forward_trace_elapsed,
                    "data_load_seconds": data_load_elapsed,
                    "forward_compute_seconds": forward_compute_elapsed,
                    "shard_update_seconds": update_elapsed,
                    "backward_update_seconds": update_result.backward_elapsed,
                    "commit_seconds": update_result.commit_elapsed,
                    "prefetch_prime_seconds": update_result.prefetch_elapsed,
                    "timing_breakdown": timing_breakdown,
                    **timing_breakdown,
                    "grad_norm": grad_norm_summary(self._last_step_grad_norms),
                    "global_grad_norm": self._last_step_global_grad_norm,
                    "global_grad_clip_scale": self._last_step_global_grad_clip_scale,
                    "global_optimizer": self._last_global_optimizer_stats,
                    "guarded_step_replay": self._last_guarded_step_replay_stats,
                    "grad_clip_mode": self.train_config.grad_clip_mode,
                    "optimizer": self.train_config.optimizer,
                    "trace_storage": self._trace_storage_snapshot(),
                    **self._last_step_activity.to_dict(),
                    "memory": memory_snapshot(self.device),
                    "residency": self.store.residency_snapshot(),
                }
            )
            print(
                "step="
                f"{step} loss={train_loss:.4f} "
                f"tokens/sec={tokens_per_sec:.1f} update_sec={update_elapsed:.3f} "
                f"trace_sec={forward_trace_elapsed:.3f} "
                f"data_sec={data_load_elapsed:.3f} "
                f"forward_sec={forward_compute_elapsed:.3f} "
                f"backward_sec={update_result.backward_elapsed:.3f} "
                f"commit_sec={update_result.commit_elapsed:.3f} "
                f"param_load={timing_breakdown['param_load_seconds']:.3f} "
                f"module_build={timing_breakdown['module_build_seconds']:.3f} "
                f"h2d={timing_breakdown['h2d_seconds']:.3f} "
                f"fwd_kernel={timing_breakdown['forward_kernel_seconds']:.3f} "
                f"bwd_kernel={timing_breakdown['backward_kernel_seconds']:.3f} "
                f"opt_load={timing_breakdown['optimizer_load_seconds']:.3f} "
                f"opt_math={timing_breakdown['optimizer_math_seconds']:.3f} "
                f"save_stage={(timing_breakdown['param_save_stage_seconds'] + timing_breakdown['optimizer_save_stage_seconds']):.3f} "
                f"updated_shards={self._last_step_activity.updated_shards} "
                f"optim_shards={self._last_step_activity.optimizer_shards_touched} "
                f"max_active_params={self._last_step_activity.max_active_param_shards_observed} "
                f"max_active_optims={self._last_step_activity.max_active_optimizer_shards_observed} "
                f"trace_storage={self._trace_storage_snapshot()} "
                f"resident={self.store.residency_snapshot()}",
                flush=True,
            )
            durable_flush = self._maybe_flush_active_store(step)
            if durable_flush is not None:
                print(
                    "durable_flush "
                    f"step={step} seconds={durable_flush['durable_flush_seconds']:.3f} "
                    f"active={durable_flush['active_run_dir']} "
                    f"durable={durable_flush['durable_run_dir']}",
                    flush=True,
                )

        self.store.flush_pending_saves()
        return {
            "run_dir": str(self.store.durable_run_dir),
            "active_run_dir": str(self.run_dir),
            "global_step": self.train_config.max_steps,
        }

    def _run_step_update(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
    ) -> StepUpdateResult:
        self._last_guarded_step_replay_stats = None
        if self.train_config.guarded_step_replay:
            return self._run_guarded_step_update(traces, lr=lr, step=step)
        return self._run_single_step_update(traces, lr=lr, step=step)

    def _run_single_step_update(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        loss: float | None = None,
    ) -> StepUpdateResult:
        transaction = self.store.begin_step_transaction(step)
        try:
            backward_start = time.perf_counter()
            train_loss = self.backward_update(traces, lr=lr, step=step, transaction=transaction, loss=loss)
            backward_elapsed = time.perf_counter() - backward_start
            commit_start = time.perf_counter()
            transaction.commit()
            commit_elapsed = time.perf_counter() - commit_start
            prefetch_start = time.perf_counter()
            self._prime_next_step_prefetch()
            prefetch_elapsed = time.perf_counter() - prefetch_start
        except BaseException:
            transaction.abort()
            raise
        return StepUpdateResult(
            train_loss=train_loss,
            backward_elapsed=backward_elapsed,
            commit_elapsed=commit_elapsed,
            prefetch_elapsed=prefetch_elapsed,
            committed_update=True,
            effective_lr=lr,
        )

    def _run_guarded_step_update(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
    ) -> StepUpdateResult:
        max_attempts = self.train_config.guard_replay_max_replays + 1
        base_max_grad_norm = self.train_config.max_grad_norm
        attempts: list[dict[str, Any]] = []
        loss_before: float | None = None
        total_backward_elapsed = 0.0
        total_commit_elapsed = 0.0
        total_prefetch_elapsed = 0.0
        transaction: ShardStepTransaction | None = None

        try:
            for attempt_index in range(max_attempts):
                lr_scale = replay_scale_for_attempt(
                    self.train_config.guard_replay_lr_scales, attempt_index
                )
                grad_norm_scale = replay_scale_for_attempt(
                    self.train_config.guard_replay_grad_norm_scales, attempt_index
                )
                effective_lr = lr * lr_scale
                effective_max_grad_norm = base_max_grad_norm * grad_norm_scale
                transaction = self.store.begin_step_transaction(step)
                self.train_config.max_grad_norm = effective_max_grad_norm
                backward_start = time.perf_counter()
                train_loss = self.backward_update(
                    traces,
                    lr=effective_lr,
                    step=step,
                    transaction=transaction,
                    loss=loss_before,
                )
                backward_elapsed = time.perf_counter() - backward_start
                total_backward_elapsed += backward_elapsed
                if loss_before is None:
                    loss_before = train_loss

                loss_after = self._batch_loss_for_traces(traces)
                accepted = self._guarded_replay_accepts(loss_before, loss_after)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "lr_scale": lr_scale,
                        "effective_lr": effective_lr,
                        "grad_norm_scale": grad_norm_scale,
                        "effective_max_grad_norm": effective_max_grad_norm,
                        "loss_before": loss_before,
                        "loss_after": loss_after,
                        "accepted": accepted,
                        "global_grad_norm": self._last_step_global_grad_norm,
                        "global_grad_clip_scale": self._last_step_global_grad_clip_scale,
                    }
                )
                if accepted:
                    commit_start = time.perf_counter()
                    transaction.commit()
                    total_commit_elapsed += time.perf_counter() - commit_start
                    transaction = None
                    prefetch_start = time.perf_counter()
                    self._prime_next_step_prefetch()
                    total_prefetch_elapsed += time.perf_counter() - prefetch_start
                    self._last_guarded_step_replay_stats = {
                        "active": True,
                        "accepted": True,
                        "attempts": len(attempts),
                        "max_attempts": max_attempts,
                        "loss_before": loss_before,
                        "accepted_loss_after": loss_after,
                        "accepted_lr_scale": lr_scale,
                        "accepted_grad_norm_scale": grad_norm_scale,
                        "loss_tolerance": self.train_config.guard_replay_loss_tolerance,
                        "loss_tolerance_ratio": self.train_config.guard_replay_loss_tolerance_ratio,
                        "attempt_log": attempts,
                    }
                    return StepUpdateResult(
                        train_loss=loss_before,
                        backward_elapsed=total_backward_elapsed,
                        commit_elapsed=total_commit_elapsed,
                        prefetch_elapsed=total_prefetch_elapsed,
                        committed_update=True,
                        effective_lr=effective_lr,
                    )

                if (
                    attempt_index == max_attempts - 1
                    and self.train_config.guard_replay_on_exhaust == "accept"
                ):
                    commit_start = time.perf_counter()
                    transaction.commit()
                    total_commit_elapsed += time.perf_counter() - commit_start
                    transaction = None
                    prefetch_start = time.perf_counter()
                    self._prime_next_step_prefetch()
                    total_prefetch_elapsed += time.perf_counter() - prefetch_start
                    self._last_guarded_step_replay_stats = {
                        "active": True,
                        "accepted": False,
                        "attempts": len(attempts),
                        "max_attempts": max_attempts,
                        "loss_before": loss_before,
                        "exhausted_action": "accept",
                        "accepted_loss_after": loss_after,
                        "accepted_lr_scale": lr_scale,
                        "accepted_grad_norm_scale": grad_norm_scale,
                        "loss_tolerance": self.train_config.guard_replay_loss_tolerance,
                        "loss_tolerance_ratio": self.train_config.guard_replay_loss_tolerance_ratio,
                        "attempt_log": attempts,
                    }
                    return StepUpdateResult(
                        train_loss=loss_before,
                        backward_elapsed=total_backward_elapsed,
                        commit_elapsed=total_commit_elapsed,
                        prefetch_elapsed=total_prefetch_elapsed,
                        committed_update=True,
                        effective_lr=effective_lr,
                    )

                transaction.abort()
                transaction = None

            if self.train_config.guard_replay_on_exhaust == "skip":
                self._last_step_grad_norms = []
                self._last_step_global_grad_norm = None
                self._last_step_global_grad_clip_scale = None
                self._last_global_optimizer_stats = None
                self._last_step_activity = StepActivity()
                prefetch_start = time.perf_counter()
                self._prime_next_step_prefetch()
                total_prefetch_elapsed += time.perf_counter() - prefetch_start
                self._last_guarded_step_replay_stats = {
                    "active": True,
                    "accepted": False,
                    "attempts": len(attempts),
                    "max_attempts": max_attempts,
                    "loss_before": loss_before,
                    "exhausted_action": "skip",
                    "loss_tolerance": self.train_config.guard_replay_loss_tolerance,
                    "loss_tolerance_ratio": self.train_config.guard_replay_loss_tolerance_ratio,
                    "attempt_log": attempts,
                }
                return StepUpdateResult(
                    train_loss=loss_before if loss_before is not None else float("nan"),
                    backward_elapsed=total_backward_elapsed,
                    commit_elapsed=total_commit_elapsed,
                    prefetch_elapsed=total_prefetch_elapsed,
                    committed_update=False,
                    effective_lr=0.0,
                )
            raise RuntimeError("guarded step replay exhausted without a configured action")
        finally:
            self.train_config.max_grad_norm = base_max_grad_norm
            if transaction is not None:
                transaction.abort()

    @torch.no_grad()
    def _batch_loss_for_traces(self, traces: list[MicroBatchTrace]) -> float:
        losses: list[float] = []
        for trace in traces:
            replay_trace = self.forward_trace(
                trace.input_ids,
                trace.labels,
                training=False,
                compute_loss=True,
            )
            if replay_trace.loss is None:
                raise RuntimeError("guarded step replay expected a replay loss")
            losses.append(replay_trace.loss)
            del replay_trace
        return sum(losses) / max(1, len(losses))

    def _guarded_replay_accepts(self, loss_before: float, loss_after: float) -> bool:
        tolerance = self.train_config.guard_replay_loss_tolerance
        tolerance += abs(loss_before) * self.train_config.guard_replay_loss_tolerance_ratio
        return loss_after <= loss_before + tolerance

    @torch.no_grad()
    def forward_trace(
        self,
        input_ids_cpu: torch.Tensor,
        labels_cpu: torch.Tensor,
        *,
        training: bool,
        compute_loss: bool = True,
    ) -> MicroBatchTrace:
        if input_ids_cpu.shape[1] > self.model_config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        input_ids = self._to_device_timed(input_ids_cpu, dtype=torch.long)
        labels = self._to_device_timed(labels_cpu, dtype=torch.long)
        boundaries: list[torch.Tensor] = []
        forward_sequence = self._forward_shard_sequence(compute_loss=compute_loss)
        sequence_index = 0

        self._prefetch_shard_window(forward_sequence, sequence_index, include_optimizer=False)
        with self.store.active_module(
            "embeddings", device=self.device, dtype=self.dtype, training=training
        ) as module:
            kernel_start = self._timing_start(sync_cuda=True)
            x = module(input_ids)
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
        boundaries.append(self._stage_trace_timed(x.detach()))
        sequence_index += 1

        for block_index in range(self.model_config.num_layers):
            shard_name = f"block_{block_index:03d}"
            self._prefetch_shard_window(forward_sequence, sequence_index, include_optimizer=False)
            with self.store.active_module(
                shard_name, device=self.device, dtype=self.dtype, training=training
            ) as module:
                kernel_start = self._timing_start(sync_cuda=True)
                x = module(x)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
            boundaries.append(self._stage_trace_timed(x.detach()))
            sequence_index += 1

        self._prefetch_shard_window(forward_sequence, sequence_index, include_optimizer=False)
        with self.store.active_module(
            "final_norm", device=self.device, dtype=self.dtype, training=training
        ) as module:
            kernel_start = self._timing_start(sync_cuda=True)
            norm_output = module(x)
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
        sequence_index += 1
        loss: float | None = None
        if compute_loss:
            self._prefetch_shard_window(forward_sequence, sequence_index, include_optimizer=False)
            with self.store.active_module(
                "lm_head", device=self.device, dtype=self.dtype, training=training
            ) as module:
                loss_tensor = self._lm_head_loss_for_trace(module, norm_output, labels)
                loss = float(self._to_cpu_timed(loss_tensor.detach()))

        trace_input_ids = (
            input_ids_cpu.detach().cpu()
            if self.trace_storage_device.type == "cpu"
            else self._stage_trace_timed(input_ids.detach())
        )
        trace_labels = (
            labels_cpu.detach().cpu()
            if self.trace_storage_device.type == "cpu"
            else self._stage_trace_timed(labels.detach())
        )

        return MicroBatchTrace(
            input_ids=trace_input_ids,
            labels=trace_labels,
            boundaries=boundaries,
            norm_output=self._stage_trace_timed(norm_output.detach()),
            loss=loss,
        )

    def backward_update(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float | None = None,
    ) -> float:
        self._last_step_grad_norms = []
        self._last_step_global_grad_norm = None
        self._last_step_global_grad_clip_scale = None
        self._last_global_optimizer_stats = None
        self._last_step_activity = StepActivity()
        if self._global_optimizer_should_run(step):
            return self._backward_update_periodic_global_optimizer(
                traces,
                lr=lr,
                step=step,
                transaction=transaction,
                loss=loss,
            )
        if self.train_config.grad_clip_mode == "global":
            return self._backward_update_global_clip(
                traces,
                lr=lr,
                step=step,
                transaction=transaction,
                loss=loss,
            )
        return self._backward_update_shard_clip(
            traces,
            lr=lr,
            step=step,
            transaction=transaction,
            loss=loss,
        )

    def _backward_update_shard_clip(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float | None,
        gradient_scales: dict[str, float] | None = None,
    ) -> float:
        gradient_scales = gradient_scales or {}
        backward_sequence = self._backward_shard_sequence()
        sequence_index = 0
        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=True)
        grad, train_loss = self._update_lm_head(
            traces,
            lr=lr,
            step=step,
            transaction=transaction,
            loss=loss,
            gradient_scale=gradient_scales.get("lm_head", 1.0),
            clip_grad="lm_head" not in gradient_scales,
        )
        sequence_index += 1
        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=True)
        grad = self._update_final_norm(
            traces,
            grad,
            lr=lr,
            step=step,
            transaction=transaction,
            loss=train_loss,
            gradient_scale=gradient_scales.get("final_norm", 1.0),
            clip_grad="final_norm" not in gradient_scales,
        )
        sequence_index += 1
        for block_index in reversed(range(self.model_config.num_layers)):
            self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=True)
            grad = self._update_block(
                block_index,
                traces,
                grad,
                lr=lr,
                step=step,
                transaction=transaction,
                loss=train_loss,
                gradient_scale=gradient_scales.get(f"block_{block_index:03d}", 1.0),
                clip_grad=f"block_{block_index:03d}" not in gradient_scales,
            )
            sequence_index += 1
        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=True)
        self._update_embeddings(
            traces,
            grad,
            lr=lr,
            step=step,
            transaction=transaction,
            loss=train_loss,
            gradient_scale=gradient_scales.get("embeddings", 1.0),
            clip_grad="embeddings" not in gradient_scales,
        )
        total_norm_sq = sum(value * value for value in self._last_step_grad_norms)
        self._last_step_global_grad_norm = math.sqrt(total_norm_sq)
        self._last_step_global_grad_clip_scale = None
        return train_loss

    def _backward_update_global_clip(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float | None,
    ) -> float:
        backward_sequence = self._backward_shard_sequence()
        gradients: list[ShardGradientPayload] = []
        sequence_index = 0

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        grad, train_loss, shard_gradient = self._collect_lm_head_gradient(
            traces,
            loss=loss,
        )
        gradients.append(shard_gradient)
        sequence_index += 1

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        grad, shard_gradient = self._collect_final_norm_gradient(
            traces,
            grad,
        )
        gradients.append(shard_gradient)
        sequence_index += 1

        for block_index in reversed(range(self.model_config.num_layers)):
            self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
            grad, shard_gradient = self._collect_block_gradient(
                block_index,
                traces,
                grad,
            )
            gradients.append(shard_gradient)
            sequence_index += 1

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        gradients.append(self._collect_embeddings_gradient(traces, grad))

        total_norm_sq = sum(item.grad_norm_sq for item in gradients)
        global_grad_norm = math.sqrt(total_norm_sq)
        clip_scale = global_clip_scale(global_grad_norm, self.train_config.max_grad_norm)
        self._last_step_global_grad_norm = global_grad_norm
        self._last_step_global_grad_clip_scale = clip_scale

        for sequence_index, shard_gradient in enumerate(gradients):
            self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=True)
            start = time.perf_counter()
            self._apply_deferred_gradient(
                shard_gradient,
                lr,
                transaction=transaction,
                gradient_scale=clip_scale,
            )
            self._log_shard_update(shard_gradient.shard_name, step, train_loss, start)
            shard_gradient.param_grads.clear()

        return train_loss

    def _backward_update_periodic_global_optimizer(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float | None,
    ) -> float:
        gradients, train_loss = self._collect_global_gradient_stats(traces, loss=loss)
        total_norm_sq = sum(item.grad_norm_sq for item in gradients)
        global_grad_norm = math.sqrt(total_norm_sq)
        clip_scale = global_clip_scale(global_grad_norm, self.train_config.max_grad_norm)
        self._last_step_global_grad_norm = global_grad_norm
        self._last_step_global_grad_clip_scale = clip_scale
        shard_scales = self._global_optimizer_shard_scales(gradients)
        gradient_scales = {
            name: clip_scale * shard_scale for name, shard_scale in shard_scales.items()
        }
        return self._backward_update_shard_clip(
            traces,
            lr=lr,
            step=step,
            transaction=transaction,
            loss=train_loss,
            gradient_scales=gradient_scales,
        )

    def _collect_global_gradient_stats(
        self,
        traces: list[MicroBatchTrace],
        *,
        loss: float | None,
    ) -> tuple[list[ShardGradientPayload], float]:
        backward_sequence = self._backward_shard_sequence()
        gradients: list[ShardGradientPayload] = []
        sequence_index = 0

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        grad, train_loss, shard_gradient = self._collect_lm_head_gradient(
            traces,
            loss=loss,
            capture_param_grads=False,
        )
        gradients.append(shard_gradient)
        sequence_index += 1

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        grad, shard_gradient = self._collect_final_norm_gradient(
            traces,
            grad,
            capture_param_grads=False,
        )
        gradients.append(shard_gradient)
        sequence_index += 1

        for block_index in reversed(range(self.model_config.num_layers)):
            self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
            grad, shard_gradient = self._collect_block_gradient(
                block_index,
                traces,
                grad,
                capture_param_grads=False,
            )
            gradients.append(shard_gradient)
            sequence_index += 1

        self._prefetch_shard_window(backward_sequence, sequence_index, include_optimizer=False)
        gradients.append(
            self._collect_embeddings_gradient(
                traces,
                grad,
                capture_param_grads=False,
            )
        )
        return gradients, train_loss

    def _update_lm_head(
        self,
        traces: list[MicroBatchTrace],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float | None,
        gradient_scale: float = 1.0,
        clip_grad: bool = True,
    ) -> tuple[list[torch.Tensor], float]:
        shard_name = "lm_head"
        start = time.perf_counter()
        grads: list[torch.Tensor] = []
        loss_total = 0.0
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace in traces:
                x = self._trace_to_device_timed(trace.norm_output, dtype=self.dtype).detach().requires_grad_(True)
                labels = self._trace_to_device_timed(trace.labels, dtype=torch.long)
                local_loss = self._lm_head_backward_for_trace(
                    module,
                    x,
                    labels,
                    loss_scale=1.0 / len(traces),
                )
                loss_total += float(self._to_cpu_timed(local_loss.detach()))
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            self._optimizer_step_and_save(
                shard_name,
                module,
                lr,
                transaction=transaction,
                parameter_payload=payload,
                gradient_scale=gradient_scale,
                clip_grad=clip_grad,
            )
        mean_loss = loss if loss is not None else loss_total / max(1, len(traces))
        self._log_shard_update(shard_name, step, mean_loss, start)
        return grads, mean_loss

    def _lm_head_backward_for_trace(
        self,
        module: torch.nn.Module,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss_scale: float,
    ) -> torch.Tensor:
        chunk_tokens = self.train_config.lm_head_chunk_tokens
        x_flat = x.reshape(-1, x.shape[-1])
        labels_flat = labels.reshape(-1)
        token_count = max(1, int(labels_flat.numel()))
        if chunk_tokens <= 0 or token_count <= chunk_tokens:
            kernel_start = self._timing_start(sync_cuda=True)
            logits = module(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels_flat)
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
            backward_start = self._timing_start(sync_cuda=True)
            (loss * loss_scale).backward()
            self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
            return loss.detach()

        loss_sum = x_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, token_count, chunk_tokens):
            stop = min(start + chunk_tokens, token_count)
            kernel_start = self._timing_start(sync_cuda=True)
            logits = module(x_flat[start:stop])
            chunk_loss_sum = F.cross_entropy(
                logits.float(),
                labels_flat[start:stop],
                reduction="sum",
            )
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
            loss_sum = loss_sum + chunk_loss_sum.detach()
            backward_start = self._timing_start(sync_cuda=True)
            (chunk_loss_sum * (loss_scale / token_count)).backward()
            self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
        return (loss_sum / token_count).detach()

    def _lm_head_loss_for_trace(
        self,
        module: torch.nn.Module,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        chunk_tokens = self.train_config.lm_head_chunk_tokens
        x_flat = x.reshape(-1, x.shape[-1])
        labels_flat = labels.reshape(-1)
        token_count = max(1, int(labels_flat.numel()))
        if chunk_tokens <= 0 or token_count <= chunk_tokens:
            kernel_start = self._timing_start(sync_cuda=True)
            logits = module(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels_flat)
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
            return loss

        loss_sum = x_flat.new_zeros((), dtype=torch.float32)
        for start in range(0, token_count, chunk_tokens):
            stop = min(start + chunk_tokens, token_count)
            kernel_start = self._timing_start(sync_cuda=True)
            logits = module(x_flat[start:stop])
            loss_sum = loss_sum + F.cross_entropy(
                logits.float(),
                labels_flat[start:stop],
                reduction="sum",
            )
            self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
        return loss_sum / token_count

    def _update_final_norm(
        self,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float,
        gradient_scale: float = 1.0,
        clip_grad: bool = True,
    ) -> list[torch.Tensor]:
        shard_name = "final_norm"
        start = time.perf_counter()
        grads: list[torch.Tensor] = []
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                x = self._trace_to_device_timed(trace.boundaries[-1], dtype=self.dtype).detach().requires_grad_(True)
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(x)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            self._optimizer_step_and_save(
                shard_name,
                module,
                lr,
                transaction=transaction,
                parameter_payload=payload,
                gradient_scale=gradient_scale,
                clip_grad=clip_grad,
            )
        self._log_shard_update(shard_name, step, loss, start)
        return grads

    def _update_block(
        self,
        block_index: int,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float,
        gradient_scale: float = 1.0,
        clip_grad: bool = True,
    ) -> list[torch.Tensor]:
        shard_name = f"block_{block_index:03d}"
        start = time.perf_counter()
        grads: list[torch.Tensor] = []
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                x = (
                    self._trace_to_device_timed(trace.boundaries[block_index], dtype=self.dtype)
                    .detach()
                    .requires_grad_(True)
                )
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(x)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            self._optimizer_step_and_save(
                shard_name,
                module,
                lr,
                transaction=transaction,
                parameter_payload=payload,
                gradient_scale=gradient_scale,
                clip_grad=clip_grad,
            )
        self._log_shard_update(shard_name, step, loss, start)
        return grads

    def _update_embeddings(
        self,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        lr: float,
        step: int,
        transaction: ShardStepTransaction,
        loss: float,
        gradient_scale: float = 1.0,
        clip_grad: bool = True,
    ) -> None:
        shard_name = "embeddings"
        start = time.perf_counter()
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                input_ids = self._trace_to_device_timed(trace.input_ids, dtype=torch.long)
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(input_ids)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
            self._optimizer_step_and_save(
                shard_name,
                module,
                lr,
                transaction=transaction,
                parameter_payload=payload,
                gradient_scale=gradient_scale,
                clip_grad=clip_grad,
            )
        self._log_shard_update(shard_name, step, loss, start)

    def _collect_lm_head_gradient(
        self,
        traces: list[MicroBatchTrace],
        *,
        loss: float | None,
        capture_param_grads: bool = True,
    ) -> tuple[list[torch.Tensor], float, ShardGradientPayload]:
        shard_name = "lm_head"
        grads: list[torch.Tensor] = []
        loss_total = 0.0
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, _payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace in traces:
                x = self._trace_to_device_timed(trace.norm_output, dtype=self.dtype).detach().requires_grad_(True)
                labels = self._trace_to_device_timed(trace.labels, dtype=torch.long)
                local_loss = self._lm_head_backward_for_trace(
                    module,
                    x,
                    labels,
                    loss_scale=1.0 / len(traces),
                )
                loss_total += float(self._to_cpu_timed(local_loss.detach()))
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            shard_gradient = self._capture_module_gradients(
                shard_name,
                module,
                capture_param_grads=capture_param_grads,
            )
        mean_loss = loss if loss is not None else loss_total / max(1, len(traces))
        return grads, mean_loss, shard_gradient

    def _collect_final_norm_gradient(
        self,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        capture_param_grads: bool = True,
    ) -> tuple[list[torch.Tensor], ShardGradientPayload]:
        shard_name = "final_norm"
        grads: list[torch.Tensor] = []
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, _payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                x = self._trace_to_device_timed(trace.boundaries[-1], dtype=self.dtype).detach().requires_grad_(True)
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(x)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            shard_gradient = self._capture_module_gradients(
                shard_name,
                module,
                capture_param_grads=capture_param_grads,
            )
        return grads, shard_gradient

    def _collect_block_gradient(
        self,
        block_index: int,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        capture_param_grads: bool = True,
    ) -> tuple[list[torch.Tensor], ShardGradientPayload]:
        shard_name = f"block_{block_index:03d}"
        grads: list[torch.Tensor] = []
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, _payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                x = (
                    self._trace_to_device_timed(trace.boundaries[block_index], dtype=self.dtype)
                    .detach()
                    .requires_grad_(True)
                )
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(x)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
                grads.append(self._stage_gradient_timed(x.grad.detach()))
            shard_gradient = self._capture_module_gradients(
                shard_name,
                module,
                capture_param_grads=capture_param_grads,
            )
        return grads, shard_gradient

    def _collect_embeddings_gradient(
        self,
        traces: list[MicroBatchTrace],
        grad_outputs: list[torch.Tensor],
        *,
        capture_param_grads: bool = True,
    ) -> ShardGradientPayload:
        shard_name = "embeddings"
        with self.store.active_module_with_payload(
            shard_name, device=self.device, dtype=self.dtype, training=True
        ) as (module, _payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for trace, grad_output in zip(traces, grad_outputs, strict=True):
                input_ids = self._trace_to_device_timed(trace.input_ids, dtype=torch.long)
                kernel_start = self._timing_start(sync_cuda=True)
                y = module(input_ids)
                self._record_timing("forward_kernel_seconds", kernel_start, sync_cuda=True)
                grad_output_device = self._trace_to_device_timed(grad_output, dtype=self.dtype)
                backward_start = self._timing_start(sync_cuda=True)
                y.backward(grad_output_device)
                self._record_timing("backward_kernel_seconds", backward_start, sync_cuda=True)
            return self._capture_module_gradients(
                shard_name,
                module,
                capture_param_grads=capture_param_grads,
            )

    def _capture_module_gradients(
        self,
        shard_name: str,
        module: torch.nn.Module,
        *,
        capture_param_grads: bool = True,
    ) -> ShardGradientPayload:
        param_grads: dict[str, torch.Tensor] = {}
        grad_norm_sq = 0.0
        parameter_count = 0
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            parameter_count += int(parameter.numel())
            grad = parameter.grad.detach()
            norm_start = self._timing_start(sync_cuda=True)
            grad_norm_sq += float(grad.float().square().sum().detach().cpu())
            self._record_timing("gradient_cpu_copy_seconds", norm_start, sync_cuda=True)
            if capture_param_grads:
                param_grads[name] = self._to_cpu_timed(
                    grad,
                    bucket="gradient_cpu_copy_seconds",
                ).clone()
            parameter.grad = None
        return ShardGradientPayload(
            shard_name=shard_name,
            param_grads=param_grads,
            grad_norm=math.sqrt(grad_norm_sq),
            grad_norm_sq=grad_norm_sq,
            parameter_count=parameter_count,
        )

    def _apply_deferred_gradient(
        self,
        shard_gradient: ShardGradientPayload,
        lr: float,
        *,
        transaction: ShardStepTransaction,
        gradient_scale: float,
    ) -> None:
        with self.store.active_module_with_payload(
            shard_gradient.shard_name,
            device=self.device,
            dtype=self.dtype,
            training=True,
        ) as (module, payload):
            self._observe_residency()
            module.zero_grad(set_to_none=True)
            for name, parameter in module.named_parameters():
                grad = shard_gradient.param_grads.get(name)
                if grad is None:
                    continue
                parameter.grad = self._to_device_timed(grad, dtype=parameter.dtype)
                if gradient_scale != 1.0:
                    kernel_start = self._timing_start(sync_cuda=True)
                    parameter.grad.mul_(gradient_scale)
                    self._record_timing("optimizer_math_seconds", kernel_start, sync_cuda=True)
            self._optimizer_step_and_save(
                shard_gradient.shard_name,
                module,
                lr,
                transaction=transaction,
                parameter_payload=payload,
                clip_grad=False,
                precomputed_grad_norm=shard_gradient.grad_norm,
            )

    def _global_optimizer_should_run(self, step: int) -> bool:
        every = self.train_config.global_optimizer_every
        return every > 0 and step % every == 0

    def _global_optimizer_shard_scales(
        self,
        gradients: list[ShardGradientPayload],
    ) -> dict[str, float]:
        total_parameters = sum(item.parameter_count for item in gradients)
        total_norm_sq = sum(item.grad_norm_sq for item in gradients)
        if total_parameters <= 0 or total_norm_sq <= 0:
            self._last_global_optimizer_stats = {
                "active": True,
                "reason": "zero_gradient",
                "every": self.train_config.global_optimizer_every,
            }
            return {item.shard_name: 1.0 for item in gradients}

        target_rms = math.sqrt(total_norm_sq / total_parameters)
        blend = self.train_config.global_optimizer_blend
        min_scale = self.train_config.global_optimizer_min_scale
        max_scale = self.train_config.global_optimizer_max_scale
        result: dict[str, float] = {}
        raw_scales: list[float] = []
        applied_scales: list[float] = []

        for item in gradients:
            if item.parameter_count <= 0 or item.grad_norm_sq <= 0:
                raw_scale = 1.0
            else:
                shard_rms = math.sqrt(item.grad_norm_sq / item.parameter_count)
                raw_scale = target_rms / max(shard_rms, 1e-12)
            clamped_scale = min(max(raw_scale, min_scale), max_scale)
            applied_scale = 1.0 + blend * (clamped_scale - 1.0)
            result[item.shard_name] = float(applied_scale)
            raw_scales.append(float(raw_scale))
            applied_scales.append(float(applied_scale))

        self._last_global_optimizer_stats = {
            "active": True,
            "every": self.train_config.global_optimizer_every,
            "mode": "gradient_rms_shard_normalization",
            "blend": blend,
            "min_scale": min_scale,
            "max_scale": max_scale,
            "target_grad_rms": target_rms,
            "raw_scale_min": min(raw_scales),
            "raw_scale_mean": sum(raw_scales) / len(raw_scales),
            "raw_scale_max": max(raw_scales),
            "applied_scale_min": min(applied_scales),
            "applied_scale_mean": sum(applied_scales) / len(applied_scales),
            "applied_scale_max": max(applied_scales),
        }
        return result

    def _optimizer_step_and_save(
        self,
        shard_name: str,
        module: torch.nn.Module,
        lr: float,
        *,
        transaction: ShardStepTransaction,
        parameter_payload: dict[str, Any],
        clip_grad: bool = True,
        precomputed_grad_norm: float | None = None,
        gradient_scale: float = 1.0,
    ) -> None:
        optim_state = self.store.load_optimizer_state(
            shard_name,
            module,
            device=self.device,
            optimizer=self.train_config.optimizer,
        )
        master_state_dict = self._master_state_dict_for_update(parameter_payload)
        self._observe_residency()
        optimizer_math_start = self._timing_start(sync_cuda=True)
        if gradient_scale != 1.0:
            scale_module_gradients(module, gradient_scale)
        grad_norm = (
            clip_module_grad_norm(module, self.train_config.max_grad_norm)
            if clip_grad
            else maybe_precomputed_grad_norm(precomputed_grad_norm, module)
        )
        self._last_step_grad_norms.append(grad_norm)
        optimizer_update_module(
            module,
            optim_state,
            master_state_dict=master_state_dict,
            optimizer=self.train_config.optimizer,
            lr=lr,
            beta1=self.train_config.beta1,
            beta2=self.train_config.beta2,
            eps=self.train_config.adam_eps,
            weight_decay=self.train_config.weight_decay,
        )
        self._record_timing("optimizer_math_seconds", optimizer_math_start, sync_cuda=True)
        updated_files = self.store.save_parameter_shard(
            shard_name,
            module,
            transaction=transaction,
            state_dict=self._master_state_dict_for_save(master_state_dict),
        )
        optimizer_files = self.store.save_optimizer_state(
            shard_name, optim_state, transaction=transaction
        )
        self._last_step_activity.updated_shards += updated_files
        self._last_step_activity.optimizer_shards_touched += optimizer_files

    def _observe_residency(self) -> None:
        self._last_step_activity.observe(self.store.residency_snapshot())

    def _resolve_trace_storage_device(self) -> torch.device:
        mode = self.train_config.trace_storage_mode
        if mode == "cpu":
            return torch.device("cpu")
        if self.device.type != "cuda":
            raise RuntimeError("--trace-storage gpu modes require a CUDA training device")
        if not torch.cuda.is_available():
            raise RuntimeError("--trace-storage gpu modes require CUDA")

        active_index = self.device.index if self.device.index is not None else 0
        if mode == "gpu":
            if self.train_config.trace_storage_device:
                device = torch.device(self.train_config.trace_storage_device)
                if device.type != "cuda":
                    raise RuntimeError("--trace-storage-device must be a CUDA device")
                device_index = device.index if device.index is not None else 0
                if device_index != active_index:
                    raise RuntimeError(
                        "--trace-storage gpu stages on the training GPU; "
                        "use --trace-storage secondary-gpu for another CUDA device"
                    )
            return torch.device(f"cuda:{active_index}")
        if mode == "secondary-gpu":
            if self.train_config.trace_storage_device:
                device = torch.device(self.train_config.trace_storage_device)
                if device.type != "cuda":
                    raise RuntimeError("--trace-storage-device must be a CUDA device")
                device_index = device.index if device.index is not None else 0
                if device_index == active_index:
                    raise RuntimeError("--trace-storage secondary-gpu must differ from the training GPU")
                if device_index >= torch.cuda.device_count():
                    raise RuntimeError(f"trace storage CUDA device is unavailable: cuda:{device_index}")
                return torch.device(f"cuda:{device_index}")
            for index in range(torch.cuda.device_count()):
                if index != active_index:
                    return torch.device(f"cuda:{index}")
            raise RuntimeError("--trace-storage secondary-gpu requires a second CUDA device")
        raise RuntimeError(f"unknown trace storage mode: {mode}")

    def _trace_storage_snapshot(self) -> dict[str, str]:
        return {
            "mode": self.train_config.trace_storage_mode,
            "device": str(self.trace_storage_device),
        }

    def _timing_start(self, *, sync_cuda: bool = False) -> float:
        if sync_cuda and self.train_config.timing_sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _record_timing(self, bucket: str, start: float, *, sync_cuda: bool = False) -> None:
        if sync_cuda and self.train_config.timing_sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.store.add_timing(bucket, time.perf_counter() - start)

    def _to_device_timed(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        bucket: str = "h2d_seconds",
    ) -> torch.Tensor:
        start = self._timing_start(sync_cuda=True)
        result = tensor.to(self.device, dtype=dtype)
        self._record_timing(bucket, start, sync_cuda=True)
        return result

    def _trace_to_device_timed(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bucket = "h2d_seconds" if tensor.device.type == "cpu" else "trace_restore_seconds"
        return self._to_device_timed(tensor, dtype=dtype, bucket=bucket)

    def _stage_trace_timed(
        self,
        tensor: torch.Tensor,
        *,
        cpu_bucket: str = "activation_cpu_copy_seconds",
    ) -> torch.Tensor:
        bucket = cpu_bucket if self.trace_storage_device.type == "cpu" else "trace_stage_seconds"
        start = self._timing_start(sync_cuda=True)
        if self.trace_storage_device.type == "cpu":
            result = tensor.cpu()
        else:
            result = tensor.to(device=self.trace_storage_device, non_blocking=True)
        self._record_timing(bucket, start, sync_cuda=True)
        return result

    def _stage_gradient_timed(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._stage_trace_timed(
            tensor,
            cpu_bucket="gradient_cpu_copy_seconds",
        )

    def _to_cpu_timed(
        self,
        tensor: torch.Tensor,
        *,
        bucket: str = "activation_cpu_copy_seconds",
    ) -> torch.Tensor:
        start = self._timing_start(sync_cuda=True)
        result = tensor.cpu()
        self._record_timing(bucket, start, sync_cuda=True)
        return result

    def _master_state_dict_for_update(
        self, parameter_payload: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        if self.train_config.master_weight_dtype == "compute":
            return None
        state_dict = parameter_payload.get("state_dict", {})
        result: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            start = self._timing_start(sync_cuda=True)
            result[key] = value.detach().to(device=self.device, dtype=torch.float32).clone()
            self._record_timing("h2d_seconds", start, sync_cuda=True)
        return result

    def _master_state_dict_for_save(
        self, master_state_dict: dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor] | None:
        if master_state_dict is None:
            return None
        save_dtype = dtype_for_storage(self.train_config.master_weight_dtype)
        result: dict[str, torch.Tensor] = {}
        for key, value in master_state_dict.items():
            start = self._timing_start(sync_cuda=True)
            result[key] = value.detach().to(device="cpu", dtype=save_dtype)
            self._record_timing("param_save_stage_seconds", start, sync_cuda=True)
        return result

    def _reconcile_state_with_optimizer_shards(self, state: dict[str, Any]) -> dict[str, Any]:
        optimizer_step_min, optimizer_step_max = self.store.optimizer_step_bounds()
        if optimizer_step_min is None or optimizer_step_max is None:
            return state
        if optimizer_step_min != optimizer_step_max:
            raise RuntimeError(
                "optimizer shard steps are inconsistent; run audit_perkunasv2.py before resuming "
                f"(min={optimizer_step_min}, max={optimizer_step_max})"
            )
        state_step = int(state.get("global_step", 0))
        if optimizer_step_max <= state_step:
            return state

        repaired = dict(state)
        old_tokens_seen = int(repaired.get("tokens_seen", 0))
        if state_step > 0 and old_tokens_seen > 0:
            tokens_per_step = old_tokens_seen / state_step
        else:
            tokens_per_step = (
                self.train_config.micro_batch_size
                * self.train_config.gradient_accumulation_steps
                * self.train_config.seq_len
            )
        repaired["global_step"] = optimizer_step_max
        repaired["optimizer_step"] = max(int(repaired.get("optimizer_step", 0)), optimizer_step_max)
        repaired["tokens_seen"] = max(old_tokens_seen, int(round(tokens_per_step * optimizer_step_max)))
        repaired["resume_repair"] = {
            "reason": "trainer_state_lagged_optimizer_shards",
            "old_global_step": state_step,
            "optimizer_step": optimizer_step_max,
            "old_tokens_seen": old_tokens_seen,
            "new_tokens_seen": repaired["tokens_seen"],
        }
        print(
            "WARNING: trainer_state.json lagged optimizer shards; "
            f"repairing global_step {state_step} -> {optimizer_step_max}",
            flush=True,
        )
        self.store.save_trainer_state(repaired)
        return repaired

    def _forward_shard_sequence(self, *, compute_loss: bool) -> list[str]:
        sequence = [
            "embeddings",
            *[f"block_{block_index:03d}" for block_index in range(self.model_config.num_layers)],
            "final_norm",
        ]
        if compute_loss:
            sequence.append("lm_head")
        return sequence

    def _backward_shard_sequence(self) -> list[str]:
        return [
            "lm_head",
            "final_norm",
            *[
                f"block_{block_index:03d}"
                for block_index in reversed(range(self.model_config.num_layers))
            ],
            "embeddings",
        ]

    def _prefetch_shard_window(
        self,
        sequence: list[str],
        index: int,
        *,
        include_optimizer: bool,
    ) -> None:
        window = self.train_config.prefetch_window or self.train_config.max_resident_shards
        upcoming = sequence[index : index + max(1, window)]
        self.store.prefetch_shards(
            param_shards=upcoming,
            optimizer_shards=upcoming if include_optimizer else (),
            active_device=self.device,
        )

    def _prime_next_step_prefetch(self) -> None:
        forward_sequence = self._forward_shard_sequence(compute_loss=False)
        backward_sequence = self._backward_shard_sequence()
        self.store.prefetch_shards(
            param_shards=[*forward_sequence, *backward_sequence],
            optimizer_shards=backward_sequence,
            active_device=self.device,
        )

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict[str, float]:
        loader = self._loader("val", shuffle=False)
        losses: list[float] = []
        for batch_index, (input_ids, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            trace = self.forward_trace(
                input_ids[:, : self.train_config.seq_len],
                labels[:, : self.train_config.seq_len],
                training=False,
            )
            if trace.loss is None:
                raise RuntimeError("validation trace did not compute loss")
            losses.append(trace.loss)
        if not losses:
            raise RuntimeError("validation produced zero batches; lower micro_batch_size or add val shards")
        mean_loss = sum(losses) / max(1, len(losses))
        metrics = {
            "val_loss": mean_loss,
            "val_perplexity": float(math.exp(min(20, mean_loss))),
            "validation_batches": float(len(losses)),
        }
        print(f"validation loss={mean_loss:.4f} memory={memory_snapshot(self.device)}", flush=True)
        return metrics

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        data_dir = Path(
            self.train_config.data_dir
            if split == "train" or self.train_config.val_data_dir is None
            else self.train_config.val_data_dir
        )
        shard_prefix = "train" if split == "train" else "val"
        glob_pattern = str(data_dir / f"{shard_prefix}_*.npy")
        if split == "val" and not list(data_dir.glob("val_*.npy")):
            if not self.train_config.allow_train_validation_fallback:
                raise FileNotFoundError(
                    f"no validation shards matched {data_dir / 'val_*.npy'}; "
                    "use --val-data-dir or copy val_*.npy into --data-dir"
                )
            glob_pattern = str(data_dir / "train_*.npy")
        if split == "train" and shuffle:
            dataset = LocalityPreservingPackedTokenDataset(
                glob_pattern,
                seed=self.train_config.seed,
                shuffle_shards=True,
            )
            loader_shuffle = False
        else:
            dataset = PackedTokenDataset(glob_pattern)
            loader_shuffle = False
        if dataset.sequence_length < self.train_config.seq_len:
            raise ValueError(
                f"requested seq_len={self.train_config.seq_len}, "
                f"but packed shards provide sequence length {dataset.sequence_length}"
            )
        return DataLoader(
            dataset,
            batch_size=self.train_config.micro_batch_size,
            shuffle=loader_shuffle,
            drop_last=(split == "train"),
            num_workers=0,
        )

    def _write_checkpoint_marker(self, step: int) -> None:
        marker_dir = ensure_dir(self.run_dir / "checkpoints" / f"step_{step:08d}")
        self.store.flush_pending_saves()
        (marker_dir / "_SHARDED_CHECKPOINT").write_text(
            "Perkunasv2 checkpoint is stored in run_dir/shards plus trainer_state.json\n",
            encoding="utf-8",
        )

    def write_checkpoint_marker(self, step: int) -> None:
        self._write_checkpoint_marker(step)

    def _maybe_flush_active_store(self, step: int) -> dict[str, Any] | None:
        if not self.store.uses_active_run_dir:
            return None
        if self.train_config.durable_flush_every > 0:
            should_flush = step % self.train_config.durable_flush_every == 0
        else:
            should_flush = self.train_config.save_every > 0 and step % self.train_config.save_every == 0
        if step == self.train_config.max_steps:
            should_flush = True
        if not should_flush:
            return None
        return self.store.flush_active_to_durable(step=step)

    def _append_log(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    def append_log(self, row: dict[str, Any]) -> None:
        self._append_log(row)

    def _log_shard_update(self, shard_name: str, step: int, loss: float, start: float) -> None:
        if self.train_config.shard_log_every == 0 or step % self.train_config.shard_log_every != 0:
            return
        elapsed = time.perf_counter() - start
        row = {
            "step": step,
            "active_shard": shard_name,
            "loss": loss,
            "shard_update_seconds": elapsed,
            "memory": memory_snapshot(self.device),
            "residency": self.store.residency_snapshot(),
        }
        self._append_log(row)
        print(
            f"active_shard={shard_name} step={step} loss={loss:.4f} "
            f"shard_update_sec={elapsed:.4f} memory={row['memory']}",
            flush=True,
        )


def optimizer_update_module(
    module: torch.nn.Module,
    state: dict[str, dict[str, Any]],
    *,
    master_state_dict: dict[str, torch.Tensor] | None = None,
    optimizer: str,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    if optimizer == "adamw":
        adamw_update_module(
            module,
            state,
            master_state_dict=master_state_dict,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        return
    if optimizer == "lion":
        lion_update_module(
            module,
            state,
            master_state_dict=master_state_dict,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
        return
    if optimizer == "adafactor":
        adafactor_update_module(
            module,
            state,
            master_state_dict=master_state_dict,
            lr=lr,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        return
    raise ValueError(f"unknown optimizer: {optimizer}")


def master_parameter_tensor(
    master_state_dict: dict[str, torch.Tensor] | None,
    name: str,
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    if master_state_dict is None:
        return parameter.data.float()
    value = master_state_dict.get(name)
    if value is None:
        value = parameter.data.detach()
    return value.to(device=parameter.device, dtype=torch.float32)


def adamw_update_module(
    module: torch.nn.Module,
    state: dict[str, dict[str, Any]],
    *,
    master_state_dict: dict[str, torch.Tensor] | None = None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            item = state[name]
            grad = parameter.grad.detach().float()
            param_fp32 = master_parameter_tensor(master_state_dict, name, parameter)
            item["step"] += 1
            if weight_decay:
                param_fp32.mul_(1.0 - lr * weight_decay)
            exp_avg = item["exp_avg"]
            exp_avg_sq = item["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1 ** item["step"]
            bias_correction2 = 1.0 - beta2 ** item["step"]
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            denom = exp_avg_sq.sqrt().add_(eps)
            param_fp32.addcdiv_(exp_avg, denom, value=-step_size)
            parameter.data.copy_(param_fp32.to(parameter.dtype))
            if master_state_dict is not None:
                master_state_dict[name] = param_fp32.detach()
            parameter.grad = None


def lion_update_module(
    module: torch.nn.Module,
    state: dict[str, dict[str, Any]],
    *,
    master_state_dict: dict[str, torch.Tensor] | None = None,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            item = state[name]
            grad = parameter.grad.detach().float()
            param_fp32 = master_parameter_tensor(master_state_dict, name, parameter)
            item["step"] += 1
            if weight_decay:
                param_fp32.mul_(1.0 - lr * weight_decay)
            exp_avg = item["exp_avg"]
            update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
            param_fp32.add_(update.sign(), alpha=-lr)
            exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
            parameter.data.copy_(param_fp32.to(parameter.dtype))
            if master_state_dict is not None:
                master_state_dict[name] = param_fp32.detach()
            parameter.grad = None


def adafactor_update_module(
    module: torch.nn.Module,
    state: dict[str, dict[str, Any]],
    *,
    master_state_dict: dict[str, torch.Tensor] | None = None,
    lr: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            item = state[name]
            grad = parameter.grad.detach().float()
            param_fp32 = master_parameter_tensor(master_state_dict, name, parameter)
            item["step"] += 1
            if weight_decay:
                param_fp32.mul_(1.0 - lr * weight_decay)
            grad_sq = grad.square()
            if grad.ndim == 2 and "exp_avg_sq_row" in item and "exp_avg_sq_col" in item:
                row_state = item["exp_avg_sq_row"]
                col_state = item["exp_avg_sq_col"]
                row_state.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2)
                col_state.mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=1.0 - beta2)
                row_factor = (row_state / row_state.mean().clamp_min(eps)).rsqrt().unsqueeze(-1)
                col_factor = col_state.clamp_min(eps).rsqrt().unsqueeze(0)
                update = grad * row_factor * col_factor
            else:
                exp_avg_sq = item["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
                update = grad * exp_avg_sq.clamp_min(eps).rsqrt()
            update_rms = update.square().mean().sqrt().clamp_min(1.0)
            param_fp32.add_(update / update_rms, alpha=-lr)
            parameter.data.copy_(param_fp32.to(parameter.dtype))
            if master_state_dict is not None:
                master_state_dict[name] = param_fp32.detach()
            parameter.grad = None


def clip_module_grad_norm(module: torch.nn.Module, max_norm: float) -> float:
    parameters = [parameter for parameter in module.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0
    if max_norm <= 0:
        norms = [parameter.grad.detach().float().norm(2) for parameter in parameters]
        return float(torch.stack(norms).norm(2).detach().cpu())
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total_norm.detach().cpu())


def scale_module_gradients(module: torch.nn.Module, scale: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)


def maybe_precomputed_grad_norm(
    value: float | None,
    module: torch.nn.Module,
) -> float:
    if value is not None:
        return float(value)
    return module_grad_norm(module)


def module_grad_norm(module: torch.nn.Module) -> float:
    total_norm_sq = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total_norm_sq += float(parameter.grad.detach().float().square().sum().detach().cpu())
    return math.sqrt(total_norm_sq)


def global_clip_scale(grad_norm: float, max_norm: float) -> float:
    if max_norm <= 0 or grad_norm <= max_norm:
        return 1.0
    return max_norm / max(grad_norm, 1e-12)


def grad_norm_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(sum(values) / len(values)), "max": float(max(values))}


def learning_rate_for_update(
    config: PerkunasShardTrainingConfig,
    step: int,
    tokens_seen: int,
) -> float:
    if config.lr_schedule == "tokens":
        warmup_tokens = config.warmup_tokens
        if warmup_tokens > 0 and tokens_seen <= warmup_tokens:
            return config.learning_rate * tokens_seen / max(1, warmup_tokens)
        progress = (tokens_seen - warmup_tokens) / max(1, config.decay_tokens - warmup_tokens)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return config.learning_rate * max(config.min_lr_ratio, cosine)
    return learning_rate_for_step(config, step)


def learning_rate_for_step(config: PerkunasShardTrainingConfig, step: int) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.learning_rate * max(config.min_lr_ratio, cosine)


def replay_scale_for_attempt(scales: tuple[float, ...], attempt_index: int) -> float:
    if not scales:
        raise ValueError("guard replay scales must not be empty")
    return float(scales[min(attempt_index, len(scales) - 1)])


def restore_rng_state_json(data: Any) -> None:
    if not isinstance(data, dict):
        return
    cpu_state = data.get("torch_cpu")
    if isinstance(cpu_state, str):
        torch.set_rng_state(torch.tensor(list(bytes.fromhex(cpu_state)), dtype=torch.uint8))
    cuda_states = data.get("torch_cuda_all")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        tensors = [
            torch.tensor(list(bytes.fromhex(item)), dtype=torch.uint8)
            for item in cuda_states
            if isinstance(item, str)
        ]
        if tensors:
            torch.cuda.set_rng_state_all(tensors)


def infinite_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def select_shard_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        print_cuda_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError("Perkunasv2 requested CUDA but torch.cuda.is_available() is false")
        device = torch.device("cuda:0" if device_name == "cuda" else device_name)
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device.index)
        print_cuda_diagnostics(device)
        return device
    if torch.cuda.is_available() and device_name == "cpu":
        print("Perkunasv2 explicit CPU mode selected while CUDA is available", flush=True)
    return torch.device(device_name)


def dtype_for_device(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda" and dtype_name != "fp32":
        raise ValueError("fp16/bf16 shard-native training requires CUDA; use dtype=fp32 for CPU")
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def dtype_for_storage(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported storage dtype: {dtype_name}")


def memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def initialize_run(
    run_dir: str | Path,
    config_path: str | Path,
    *,
    seed: int = 1337,
    shard_storage_format: str = "torch",
    storage_shard_count: int = 0,
    initial_weight_dtype: str = "fp32",
) -> dict[str, Any]:
    model_config = PerkunasV2Config.from_json(config_path)
    store = ParameterShardStore.initialize_random_shards(
        run_dir,
        model_config,
        seed=seed,
        storage_format=shard_storage_format,
        storage_shard_count=storage_shard_count,
        initial_weight_dtype=initial_weight_dtype,
    )
    return {
        "run_dir": str(store.run_dir),
        "shards": shard_names(model_config),
        "metadata": str(store.metadata_path),
    }
