from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import shutil
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file

from perkunas_training.perkunasv2.configuration import PerkunasV2Config
from perkunas_training.perkunasv2.modules import (
    build_embeddings,
    build_final_norm,
    build_lm_head,
    build_transformer_block,
    initialize_module,
)


SHARD_METADATA_VERSION = 1
SHARD_EXTENSIONS = {"torch": ".pt", "safetensors": ".safetensors"}
SHARD_FORMATS_BY_EXTENSION = {value: key for key, value in SHARD_EXTENSIONS.items()}
ACTIVE_STORE_MANIFEST_NAME = "active_store_manifest.json"
DURABLE_FLUSH_MANIFEST_NAME = "durable_flush_manifest.json"
TIMING_BUCKETS = (
    "param_load_seconds",
    "module_build_seconds",
    "h2d_seconds",
    "forward_kernel_seconds",
    "backward_kernel_seconds",
    "activation_cpu_copy_seconds",
    "gradient_cpu_copy_seconds",
    "trace_stage_seconds",
    "trace_restore_seconds",
    "optimizer_load_seconds",
    "optimizer_math_seconds",
    "param_save_stage_seconds",
    "optimizer_save_stage_seconds",
)


def shard_names(config: PerkunasV2Config) -> list[str]:
    return [
        "embeddings",
        *[f"block_{index:03d}" for index in range(config.num_layers)],
        "final_norm",
        "lm_head",
    ]


def build_module_for_shard(config: PerkunasV2Config, shard_name: str) -> nn.Module:
    if shard_name == "embeddings":
        return build_embeddings(config)
    if shard_name == "final_norm":
        return build_final_norm(config)
    if shard_name == "lm_head":
        return build_lm_head(config)
    if shard_name.startswith("block_"):
        return build_transformer_block(config, int(shard_name.split("_", maxsplit=1)[1]))
    raise ValueError(f"unknown Perkunasv2 shard: {shard_name}")


class ParameterShardStore:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        active_run_dir: str | Path | None = None,
        durable_flush_every: int = 0,
        config: PerkunasV2Config | None = None,
        max_resident_shards: int = 1,
        clear_cuda_cache_between_shards: bool = True,
        async_shard_writes: bool = False,
        max_pending_shard_writes: int = 4,
        prefetch_mode: str = "off",
        prefetch_window: int = 0,
        prefetch_optimizer_shards: bool = True,
        prefetch_device: str | None = None,
        storage_format: str = "torch",
        storage_shard_count: int = 0,
        cache_active_modules: bool = False,
    ) -> None:
        self.durable_run_dir = Path(run_dir)
        self.run_dir = Path(active_run_dir) if active_run_dir is not None else self.durable_run_dir
        self.active_run_dir = self.run_dir
        self.uses_active_run_dir = not same_path(self.active_run_dir, self.durable_run_dir)
        if durable_flush_every < 0:
            raise ValueError("durable_flush_every must be >= 0")
        self.durable_flush_every = durable_flush_every
        if self.uses_active_run_dir:
            if path_contains(self.durable_run_dir, self.active_run_dir) or path_contains(
                self.active_run_dir,
                self.durable_run_dir,
            ):
                raise ValueError("active_run_dir and run_dir must not be nested")
            self._prepare_active_store_from_durable()
        self.config = config or PerkunasV2Config.from_json(self.run_dir / "config.json")
        self.config.validate()
        if storage_format not in SHARD_EXTENSIONS:
            raise ValueError("storage_format must be one of torch, safetensors")
        if storage_shard_count < 0:
            raise ValueError("storage_shard_count must be >= 0")
        if storage_shard_count > 0 and storage_format != "safetensors":
            raise ValueError("storage_shard_count requires storage_format=safetensors")
        self.max_resident_shards = max_resident_shards
        self.clear_cuda_cache_between_shards = clear_cuda_cache_between_shards
        self.async_shard_writes = async_shard_writes
        self.max_pending_shard_writes = max_pending_shard_writes
        self.prefetch_mode = prefetch_mode
        self.prefetch_window = prefetch_window
        self.prefetch_optimizer_shards = prefetch_optimizer_shards
        self.prefetch_device = prefetch_device
        self.storage_format = storage_format
        self.storage_shard_count = storage_shard_count
        self.cache_active_modules = cache_active_modules
        self.params_dir = self.run_dir / "shards" / "params"
        self.optim_dir = self.run_dir / "shards" / "optim"
        self.transactions_dir = self.run_dir / "shards" / "transactions"
        self.metadata_path = self.run_dir / "shards" / "metadata.json"
        self.active_param_shards: dict[str, int] = {}
        self.active_optimizer_shards: dict[str, int] = {}
        self.metadata = self._load_metadata_if_present()
        self.storage_shard_plan = self._resolve_storage_shard_plan(storage_shard_count)
        self._save_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="perkunasv2-shard-writer")
            if async_shard_writes
            else None
        )
        self._pending_saves: dict[Path, concurrent.futures.Future[None]] = {}
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="perkunasv2-shard-prefetch")
            if prefetch_mode != "off"
            else None
        )
        self._cached_param_payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cached_optimizer_payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cached_modules: OrderedDict[tuple[str, str, str], nn.Module] = OrderedDict()
        self._cached_module_parameter_counts: dict[tuple[str, str, str], int] = {}
        self._pending_prefetches: dict[
            tuple[str, str], concurrent.futures.Future[dict[str, Any]]
        ] = {}
        self._timings: dict[str, float] = {bucket: 0.0 for bucket in TIMING_BUCKETS}

    @classmethod
    def initialize_random_shards(
        cls,
        run_dir: str | Path,
        config: PerkunasV2Config,
        *,
        seed: int = 1337,
        max_resident_shards: int = 1,
        storage_format: str = "torch",
        storage_shard_count: int = 0,
        initial_weight_dtype: str = "fp32",
    ) -> "ParameterShardStore":
        config.validate()
        if storage_format not in SHARD_EXTENSIONS:
            raise ValueError("storage_format must be one of torch, safetensors")
        if storage_shard_count < 0:
            raise ValueError("storage_shard_count must be >= 0")
        if storage_shard_count > 0 and storage_format != "safetensors":
            raise ValueError("storage_shard_count requires storage_format=safetensors")
        initial_dtype = storage_dtype_for_name(initial_weight_dtype)
        run_dir = Path(run_dir)
        params_dir = run_dir / "shards" / "params"
        optim_dir = run_dir / "shards" / "optim"
        params_dir.mkdir(parents=True, exist_ok=True)
        optim_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        config.save_json(run_dir / "config.json")

        logical_shards = shard_names(config)
        parameter_counts: dict[str, int] = {}
        for shard_name in logical_shards:
            module = build_module_for_shard(config, shard_name)
            parameter_counts[shard_name] = count_module_parameters(module)
            del module
            gc.collect()
        storage_shard_plan = allocate_storage_shard_plan(
            parameter_counts,
            storage_shard_count or len(logical_shards),
        )

        shards: list[dict[str, Any]] = []
        total_parameters = 0
        largest_shard_parameters = 0
        for shard_index, shard_name in enumerate(logical_shards):
            torch.manual_seed(seed + shard_index)
            module = build_module_for_shard(config, shard_name)
            initialize_module(module, config)
            state_dict = {
                key: value.detach().cpu().to(dtype=initial_dtype)
                for key, value in module.state_dict().items()
            }
            parameter_count = parameter_counts[shard_name]
            total_parameters += parameter_count
            largest_shard_parameters = max(largest_shard_parameters, parameter_count)
            part_count = storage_shard_plan[shard_name]
            params_paths = save_payload_to_storage_atomic(
                {
                    "shard_name": shard_name,
                    "state_dict": state_dict,
                    "num_parameters": parameter_count,
                    "created_unix": time.time(),
                },
                params_dir,
                shard_name,
                storage_format=storage_format,
                part_count=part_count,
            )
            params_path = params_paths[0]
            optim_path = shard_path(optim_dir, shard_name, storage_format)
            shards.append(
                {
                    "name": shard_name,
                    "params_path": str(params_path),
                    "optim_path": str(optim_path),
                    "num_parameters": parameter_count,
                    "storage_parts": part_count,
                }
            )
            del module, state_dict
            gc.collect()

        metadata = {
            "version": SHARD_METADATA_VERSION,
            "model_config": config.to_dict(),
            "config_hash": config.stable_hash(),
            "shards": shards,
            "total_parameters": total_parameters,
            "largest_shard_parameters": largest_shard_parameters,
            "max_resident_shards": max_resident_shards,
            "storage_format": storage_format,
            "storage_shard_count": sum(storage_shard_plan.values()),
            "storage_shard_plan": storage_shard_plan,
            "initial_weight_dtype": initial_weight_dtype,
        }
        metadata_path = run_dir / "shards" / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        trainer_state = {
            "global_step": 0,
            "tokens_seen": 0,
            "optimizer_step": 0,
            "scheduler_state": {"learning_rate": 0.0},
            "rng_state": rng_state_json(),
            "latest_validation_loss": None,
            "config_hash": config.stable_hash(),
            "shard_metadata_version": SHARD_METADATA_VERSION,
        }
        (run_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2) + "\n", encoding="utf-8"
        )
        return cls(
            run_dir,
            config=config,
            max_resident_shards=max_resident_shards,
            storage_format=storage_format,
            storage_shard_count=storage_shard_count,
        )

    @contextmanager
    def active_module(
        self,
        shard_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        training: bool,
    ) -> Iterator[nn.Module]:
        cache_key = self._module_cache_key(shard_name, device, dtype)
        module = self._cached_modules.get(cache_key) if self.cache_active_modules else None
        if module is not None:
            if (
                shard_name not in self.active_param_shards
                and len(self.active_param_shards) >= self.max_resident_shards
            ):
                raise RuntimeError(
                    f"resident parameter shards {list(self.active_param_shards) + [shard_name]} "
                    f"exceed max_resident_shards={self.max_resident_shards}"
                )
            self._cached_modules.move_to_end(cache_key)
            module.train(training)
            parameter_count = self._cached_module_parameter_counts.get(
                cache_key, count_module_parameters(module)
            )
            self._mark_param_active(shard_name, parameter_count)
            try:
                yield module
            finally:
                self._mark_param_released(shard_name)
            return
        with self.active_module_with_payload(
            shard_name, device=device, dtype=dtype, training=training
        ) as (module, _payload):
            yield module

    @contextmanager
    def active_module_with_payload(
        self,
        shard_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        training: bool,
    ) -> Iterator[tuple[nn.Module, dict[str, Any]]]:
        if (
            shard_name not in self.active_param_shards
            and len(self.active_param_shards) >= self.max_resident_shards
        ):
            raise RuntimeError(
                f"resident parameter shards {list(self.active_param_shards) + [shard_name]} "
                f"exceed max_resident_shards={self.max_resident_shards}"
            )
        payload = self._load_parameter_payload(shard_name)
        cache_key = self._module_cache_key(shard_name, device, dtype)
        module = self._cached_modules.get(cache_key) if self.cache_active_modules else None
        if module is None:
            module = self._build_active_module_from_payload(shard_name, payload, device, dtype)
            if self.cache_active_modules:
                self._put_cached_module(cache_key, module)
        elif self.cache_active_modules:
            self._cached_modules.move_to_end(cache_key)
        module.train(training)
        parameter_count = self._cached_module_parameter_counts.get(
            cache_key, count_module_parameters(module)
        )
        self._mark_param_active(shard_name, parameter_count)
        try:
            yield module, payload
        finally:
            self._mark_param_released(shard_name)
            if not self.cache_active_modules:
                del module
            del payload
            gc.collect()
            if self.clear_cuda_cache_between_shards and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def begin_step_transaction(self, step: int) -> "ShardStepTransaction":
        self.flush_pending_saves()
        return ShardStepTransaction(self, step, shard_names(self.config))

    def discard_stale_transactions(self) -> None:
        self.flush_pending_saves()
        if self.transactions_dir.exists():
            shutil.rmtree(self.transactions_dir)

    def reset_timings(self) -> None:
        self._timings = {bucket: 0.0 for bucket in TIMING_BUCKETS}

    def add_timing(self, bucket: str, elapsed: float) -> None:
        if elapsed < 0:
            return
        self._timings[bucket] = self._timings.get(bucket, 0.0) + float(elapsed)

    def timing_snapshot(self) -> dict[str, float]:
        return {bucket: float(self._timings.get(bucket, 0.0)) for bucket in TIMING_BUCKETS}

    def save_parameter_shard(
        self,
        shard_name: str,
        module: nn.Module,
        *,
        transaction: "ShardStepTransaction | None" = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> int:
        start = time.perf_counter()
        try:
            self._invalidate_cached_payload("param", shard_name)
            directory = self.params_dir if transaction is None else transaction.params_dir
            payload = (
                parameter_shard_payload(shard_name, module)
                if state_dict is None
                else parameter_shard_payload_from_state_dict(shard_name, state_dict, module)
            )
            paths = self._save_payload_to_storage("param", shard_name, payload, directory)
            if transaction is None:
                self._remove_alternate_payload_paths("param", shard_name, set(paths))
            self._put_cached_payload("param", shard_name, payload)
            if transaction is not None:
                transaction.mark_parameter_staged(shard_name)
            return len(paths)
        finally:
            self.add_timing("param_save_stage_seconds", time.perf_counter() - start)

    def load_optimizer_state(
        self,
        shard_name: str,
        module: nn.Module,
        *,
        device: torch.device,
        optimizer: str = "adamw",
    ) -> dict[str, dict[str, Any]]:
        start = time.perf_counter()
        try:
            payload = self._load_optimizer_payload(shard_name)
            state = payload["state"]
            result: dict[str, dict[str, Any]] = {}
            for name, parameter in module.named_parameters():
                item = state.get(name)
                if item is None:
                    item = {}
                result[name] = optimizer_state_for_parameter(
                    item,
                    parameter,
                    device=device,
                    optimizer=optimizer,
                )
            self._mark_optimizer_active(shard_name, optimizer_state_numel(result))
            return result
        finally:
            self.add_timing("optimizer_load_seconds", time.perf_counter() - start)

    def save_optimizer_state(
        self,
        shard_name: str,
        state: dict[str, dict[str, Any]],
        *,
        transaction: "ShardStepTransaction | None" = None,
    ) -> int:
        start = time.perf_counter()
        try:
            self._invalidate_cached_payload("optim", shard_name)
            directory = self.optim_dir if transaction is None else transaction.optim_dir
            payload = optimizer_state_payload(shard_name, state)
            paths = self._save_payload_to_storage(
                "optim",
                shard_name,
                payload,
                directory,
            )
            if transaction is None:
                self._remove_alternate_payload_paths("optim", shard_name, set(paths))
            self._put_cached_payload("optim", shard_name, payload)
            if transaction is not None:
                transaction.mark_optimizer_staged(shard_name)
            self._mark_optimizer_released(shard_name)
            return len(paths)
        finally:
            self.add_timing("optimizer_save_stage_seconds", time.perf_counter() - start)

    def save_trainer_state(self, state: dict[str, Any]) -> None:
        self.flush_pending_saves()
        state = dict(state)
        state["rng_state"] = rng_state_json()
        state["config_hash"] = self.config.stable_hash()
        state["shard_metadata_version"] = SHARD_METADATA_VERSION
        write_json_atomic(self.run_dir / "trainer_state.json", state)

    def load_trainer_state(self) -> dict[str, Any]:
        path = self.run_dir / "trainer_state.json"
        if not path.exists():
            return {
                "global_step": 0,
                "tokens_seen": 0,
                "optimizer_step": 0,
                "scheduler_state": {"learning_rate": 0.0},
                "latest_validation_loss": None,
                "config_hash": self.config.stable_hash(),
                "shard_metadata_version": SHARD_METADATA_VERSION,
            }
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config_hash") and state["config_hash"] != self.config.stable_hash():
            raise RuntimeError("trainer_state.json config hash does not match run config")
        return state

    def flush_active_to_durable(self, *, step: int | None = None) -> dict[str, Any]:
        self.flush_pending_saves()
        if not self.uses_active_run_dir:
            return {"durable_flush_enabled": False}

        start = time.perf_counter()
        copy_run_tree_atomic(
            self.active_run_dir,
            self.durable_run_dir,
            excluded_dirs={"transactions"},
            excluded_files={ACTIVE_STORE_MANIFEST_NAME, DURABLE_FLUSH_MANIFEST_NAME},
        )
        manifest = {
            "version": 1,
            "published_step": step,
            "published_unix": time.time(),
            "active_run_dir": str(self.active_run_dir),
            "durable_run_dir": str(self.durable_run_dir),
            "checkpoint_publication_boundary": True,
            "config_hash": self.config.stable_hash(),
        }
        write_json_atomic(self.durable_run_dir / DURABLE_FLUSH_MANIFEST_NAME, manifest)
        elapsed = time.perf_counter() - start
        return {
            "durable_flush_enabled": True,
            "durable_flush_seconds": elapsed,
            "active_run_dir": str(self.active_run_dir),
            "durable_run_dir": str(self.durable_run_dir),
            "published_step": step,
        }

    def optimizer_step_bounds(self) -> tuple[int | None, int | None]:
        steps: list[int] = []
        for shard_name in shard_names(self.config):
            paths = self._existing_optim_paths(shard_name)
            self._wait_for_pending_paths(paths)
            if not paths:
                continue
            payload = load_payload_from_paths(paths, kind="optim", shard_name=shard_name)
            for item in payload.get("state", {}).values():
                if "step" in item:
                    steps.append(int(item["step"]))
        if not steps:
            return None, None
        return min(steps), max(steps)

    def param_path(self, shard_name: str) -> Path:
        return shard_path(self.params_dir, shard_name, self.storage_format)

    def optim_path(self, shard_name: str) -> Path:
        return shard_path(self.optim_dir, shard_name, self.storage_format)

    def param_paths(self, shard_name: str) -> list[Path]:
        return payload_paths(
            self.params_dir,
            shard_name,
            self.storage_format,
            self.storage_part_count(shard_name),
        )

    def optim_paths(self, shard_name: str) -> list[Path]:
        return payload_paths(
            self.optim_dir,
            shard_name,
            self.storage_format,
            self.storage_part_count(shard_name),
        )

    def storage_part_count(self, shard_name: str) -> int:
        return max(1, int(self.storage_shard_plan.get(shard_name, 1)))

    def residency_snapshot(self) -> dict[str, Any]:
        return {
            "active_param_shards": dict(self.active_param_shards),
            "active_optimizer_shards": dict(self.active_optimizer_shards),
            "cached_module_shards": [key[0] for key in self._cached_modules],
            "cached_module_parameter_count": sum(
                self._cached_module_parameter_counts.get(key, 0)
                for key in self._cached_modules
            ),
            "cached_param_shards": list(self._cached_param_payloads),
            "cached_optimizer_shards": list(self._cached_optimizer_payloads),
            "pending_param_prefetches": [
                shard_name for kind, shard_name in self._pending_prefetches if kind == "param"
            ],
            "pending_optimizer_prefetches": [
                shard_name for kind, shard_name in self._pending_prefetches if kind == "optim"
            ],
            "resident_parameter_count": sum(self.active_param_shards.values()),
            "resident_optimizer_state_count": sum(self.active_optimizer_shards.values()),
            "max_resident_shards": self.max_resident_shards,
            "prefetch_mode": self.prefetch_mode,
            "prefetch_window": self.effective_prefetch_window(),
            "prefetch_device": self.prefetch_device,
            "storage_shard_count": sum(self.storage_shard_plan.values()),
            "active_run_dir": str(self.active_run_dir),
            "durable_run_dir": str(self.durable_run_dir),
            "uses_active_run_dir": self.uses_active_run_dir,
        }

    def effective_prefetch_window(self) -> int:
        if self.prefetch_mode == "off":
            return 0
        requested = self.prefetch_window if self.prefetch_window > 0 else self.max_resident_shards
        return max(0, min(requested, self.max_resident_shards))

    def prefetch_shards(
        self,
        *,
        param_shards: list[str] | tuple[str, ...] = (),
        optimizer_shards: list[str] | tuple[str, ...] = (),
        active_device: torch.device,
    ) -> None:
        if self._prefetch_executor is None:
            return
        window = self.effective_prefetch_window()
        if window <= 0:
            return
        self._reap_completed_prefetches()
        for shard_name in unique_prefix(param_shards, window):
            self._schedule_prefetch("param", shard_name, active_device)
        if self.prefetch_optimizer_shards:
            for shard_name in unique_prefix(optimizer_shards, window):
                self._schedule_prefetch("optim", shard_name, active_device)

    def _mark_param_active(self, shard_name: str, num_parameters: int) -> None:
        if shard_name in self.active_param_shards:
            raise RuntimeError(f"parameter shard already active: {shard_name}")
        self.active_param_shards[shard_name] = num_parameters
        try:
            self._enforce_limits()
        except Exception:
            self.active_param_shards.pop(shard_name, None)
            raise

    def _mark_param_released(self, shard_name: str) -> None:
        self.active_param_shards.pop(shard_name, None)

    def _mark_optimizer_active(self, shard_name: str, num_values: int) -> None:
        if shard_name in self.active_optimizer_shards:
            raise RuntimeError(f"optimizer shard already active: {shard_name}")
        self.active_optimizer_shards[shard_name] = num_values
        try:
            self._enforce_limits()
        except Exception:
            self.active_optimizer_shards.pop(shard_name, None)
            raise

    def _mark_optimizer_released(self, shard_name: str) -> None:
        self.active_optimizer_shards.pop(shard_name, None)

    def _enforce_limits(self) -> None:
        if len(self.active_param_shards) > self.max_resident_shards:
            raise RuntimeError(
                f"resident parameter shards {list(self.active_param_shards)} exceed "
                f"max_resident_shards={self.max_resident_shards}"
            )
        if len(self.active_optimizer_shards) > self.max_resident_shards:
            raise RuntimeError(
                f"resident optimizer shards {list(self.active_optimizer_shards)} exceed "
                f"max_resident_shards={self.max_resident_shards}"
            )
        resident_parameters = sum(self.active_param_shards.values())
        total_parameters = int(self.metadata.get("total_parameters", 0))
        largest = int(self.metadata.get("largest_shard_parameters", 0))
        if total_parameters and resident_parameters >= total_parameters and total_parameters > largest:
            raise RuntimeError(
                "full-model parameter residency detected; shard-native training refuses this state"
            )

    def _load_metadata_if_present(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return {
            "version": SHARD_METADATA_VERSION,
            "model_config": self.config.to_dict(),
            "config_hash": self.config.stable_hash(),
            "shards": [],
            "total_parameters": 0,
            "largest_shard_parameters": 0,
        }

    def _prepare_active_store_from_durable(self) -> None:
        durable_config_path = self.durable_run_dir / "config.json"
        durable_metadata_path = self.durable_run_dir / "shards" / "metadata.json"
        if not durable_config_path.exists() or not durable_metadata_path.exists():
            raise FileNotFoundError(
                "active_run_dir requires an initialized durable run_dir with config.json "
                "and shards/metadata.json"
            )

        active_config_path = self.active_run_dir / "config.json"
        active_metadata_path = self.active_run_dir / "shards" / "metadata.json"
        if active_config_path.exists() and active_metadata_path.exists():
            self._validate_or_write_active_store_manifest()
            return

        copy_run_tree_atomic(
            self.durable_run_dir,
            self.active_run_dir,
            excluded_dirs={"transactions"},
            excluded_files={DURABLE_FLUSH_MANIFEST_NAME},
        )
        self._validate_or_write_active_store_manifest()

    def _validate_or_write_active_store_manifest(self) -> None:
        manifest_path = self.active_run_dir / ACTIVE_STORE_MANIFEST_NAME
        durable_config = PerkunasV2Config.from_json(self.durable_run_dir / "config.json")
        active_config = PerkunasV2Config.from_json(self.active_run_dir / "config.json")
        if durable_config.stable_hash() != active_config.stable_hash():
            raise RuntimeError("active_run_dir config hash does not match durable run_dir")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("durable_run_dir") and not same_path(
                Path(str(manifest["durable_run_dir"])),
                self.durable_run_dir,
            ):
                raise RuntimeError("active_run_dir is already bound to a different durable run_dir")
        write_json_atomic(
            manifest_path,
            {
                "version": 1,
                "active_run_dir": str(self.active_run_dir),
                "durable_run_dir": str(self.durable_run_dir),
                "config_hash": durable_config.stable_hash(),
                "updated_unix": time.time(),
            },
        )

    def _resolve_storage_shard_plan(self, requested_count: int) -> dict[str, int]:
        logical_shards = shard_names(self.config)
        metadata_plan = self.metadata.get("storage_shard_plan")
        if requested_count <= 0 and isinstance(metadata_plan, dict):
            return {name: max(1, int(metadata_plan.get(name, 1))) for name in logical_shards}
        if requested_count <= 0:
            return {name: 1 for name in logical_shards}
        counts = {
            str(item["name"]): int(item.get("num_parameters", 0))
            for item in self.metadata.get("shards", [])
            if "name" in item
        }
        if any(counts.get(name, 0) <= 0 for name in logical_shards):
            counts = compute_module_parameter_counts(self.config)
        return allocate_storage_shard_plan(counts, requested_count)

    def flush_pending_saves(self) -> None:
        for path in list(self._pending_saves):
            self._wait_for_pending_path(path)

    def flush_pending_prefetches(self) -> None:
        for key in list(self._pending_prefetches):
            future = self._pending_prefetches.pop(key)
            kind, shard_name = key
            self._put_cached_payload(kind, shard_name, future.result())

    def sync_storage_metadata(self) -> None:
        metadata = dict(self.metadata)
        metadata["storage_format"] = self.storage_format
        metadata["storage_shard_count"] = sum(self.storage_shard_plan.values())
        metadata["storage_shard_plan"] = dict(self.storage_shard_plan)

        previous_shards = {
            str(item["name"]): dict(item)
            for item in metadata.get("shards", [])
            if isinstance(item, dict) and "name" in item
        }
        shards: list[dict[str, Any]] = []
        for shard_name in shard_names(self.config):
            item = previous_shards.get(shard_name, {"name": shard_name})
            item["params_path"] = str(self.param_paths(shard_name)[0])
            item["optim_path"] = str(self.optim_paths(shard_name)[0])
            item["storage_parts"] = self.storage_part_count(shard_name)
            shards.append(item)
        metadata["shards"] = shards
        write_json_atomic(self.metadata_path, metadata)
        self.metadata = metadata

    def _save_payload(self, payload: dict[str, Any], path: Path) -> None:
        path = Path(path)
        if self._save_executor is None:
            save_payload_atomic(payload, path)
            return
        self._reap_completed_saves()
        self._wait_for_pending_path(path)
        while self._pending_save_job_count() >= self.max_pending_shard_writes:
            oldest_path = next(iter(self._pending_saves))
            self._wait_for_pending_path(oldest_path)
        self._pending_saves[path] = self._save_executor.submit(save_payload_atomic, payload, path)

    def _build_active_module_from_payload(
        self,
        shard_name: str,
        payload: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        start = time.perf_counter()
        module = build_module_for_shard(self.config, shard_name)
        if payload_first_tensor_device(payload).type == "cpu":
            module.load_state_dict(payload["state_dict"])
            self.add_timing("module_build_seconds", time.perf_counter() - start)
            transfer_start = time.perf_counter()
            module = module.to(device=device)
            if dtype != torch.float32:
                module = module.to(dtype=dtype)
            self.add_timing("h2d_seconds", time.perf_counter() - transfer_start)
        else:
            self.add_timing("module_build_seconds", time.perf_counter() - start)
            transfer_start = time.perf_counter()
            module = module.to(device=device)
            if dtype != torch.float32:
                module = module.to(dtype=dtype)
            self.add_timing("h2d_seconds", time.perf_counter() - transfer_start)
            start = time.perf_counter()
            module.load_state_dict(payload["state_dict"])
            self.add_timing("module_build_seconds", time.perf_counter() - start)
        return module

    def _save_payload_to_storage(
        self,
        kind: str,
        shard_name: str,
        payload: dict[str, Any],
        directory: Path,
    ) -> list[Path]:
        part_count = self.storage_part_count(shard_name)
        paths = payload_paths(directory, shard_name, self.storage_format, part_count)
        if part_count == 1:
            self._save_payload(payload, paths[0])
            return paths
        if self._save_executor is None:
            save_safetensors_payload_parts_atomic(payload, paths)
            return paths

        self._reap_completed_saves()
        for path in paths:
            self._wait_for_pending_path(path)
        while self._pending_save_job_count() >= self.max_pending_shard_writes:
            oldest_path = next(iter(self._pending_saves))
            self._wait_for_pending_path(oldest_path)

        future = self._save_executor.submit(save_safetensors_payload_parts_atomic, payload, paths)
        for path in paths:
            self._pending_saves[path] = future
        return paths

    def _wait_for_pending_path(self, path: Path) -> None:
        path = Path(path)
        future = self._pending_saves.pop(path, None)
        if future is not None:
            future.result()
            for other_path, other_future in list(self._pending_saves.items()):
                if other_future is future:
                    self._pending_saves.pop(other_path)

    def _wait_for_pending_paths(self, paths: list[Path]) -> None:
        for path in paths:
            self._wait_for_pending_path(path)

    def _reap_completed_saves(self) -> None:
        for path, future in list(self._pending_saves.items()):
            if future.done():
                self._pending_saves.pop(path)
                future.result()

    def _pending_save_job_count(self) -> int:
        return len({id(future) for future in self._pending_saves.values()})

    def _load_parameter_payload(self, shard_name: str) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            payload = self._get_cached_payload("param", shard_name)
            if payload is not None:
                return payload
            preferred_paths = self.param_paths(shard_name)
            self._wait_for_pending_paths(preferred_paths)
            paths = self._existing_param_paths(shard_name)
            self._wait_for_pending_paths(paths)
            return load_payload_from_paths(paths, kind="param", shard_name=shard_name)
        finally:
            self.add_timing("param_load_seconds", time.perf_counter() - start)

    def _load_optimizer_payload(self, shard_name: str) -> dict[str, Any]:
        payload = self._get_cached_payload("optim", shard_name)
        if payload is not None:
            return payload
        preferred_paths = self.optim_paths(shard_name)
        self._wait_for_pending_paths(preferred_paths)
        paths = self._existing_optim_paths(shard_name)
        self._wait_for_pending_paths(paths)
        if not paths:
            return {"shard_name": shard_name, "state": {}}
        return load_payload_from_paths(paths, kind="optim", shard_name=shard_name)

    def _schedule_prefetch(self, kind: str, shard_name: str, active_device: torch.device) -> None:
        key = (kind, shard_name)
        if key in self._pending_prefetches:
            return
        cache = self._cache_for_kind(kind)
        if shard_name in cache:
            cache.move_to_end(shard_name)
            return
        if len(cache) + pending_prefetch_count(self._pending_prefetches, kind) >= self.effective_prefetch_window():
            return
        preferred_paths = self.param_paths(shard_name) if kind == "param" else self.optim_paths(shard_name)
        self._wait_for_pending_paths(preferred_paths)
        paths = (
            self._existing_param_paths(shard_name)
            if kind == "param"
            else self._existing_optim_paths(shard_name)
        )
        self._wait_for_pending_paths(paths)
        target_device = self._prefetch_target_device(active_device)
        self._pending_prefetches[key] = self._prefetch_executor.submit(
            load_payload_for_prefetch,
            kind,
            shard_name,
            paths,
            target_device,
        )

    def _existing_param_paths(self, shard_name: str) -> list[Path]:
        return self._existing_payload_paths(self.params_dir, shard_name, self.param_paths(shard_name))

    def _existing_optim_paths(self, shard_name: str) -> list[Path]:
        return self._existing_payload_paths(self.optim_dir, shard_name, self.optim_paths(shard_name))

    def _existing_payload_paths(
        self,
        directory: Path,
        shard_name: str,
        preferred_paths: list[Path],
    ) -> list[Path]:
        existing_preferred = [path for path in preferred_paths if path.exists()]
        if len(existing_preferred) == len(preferred_paths):
            return existing_preferred
        part_paths = sorted(directory.glob(f"{shard_name}.part_*.safetensors"))
        if part_paths:
            return part_paths
        for storage_format in ("safetensors", "torch"):
            path = shard_path(directory, shard_name, storage_format)
            if path.exists():
                return [path]
        return []

    def _remove_alternate_payload_paths(
        self,
        kind: str,
        shard_name: str,
        kept_paths: set[Path],
    ) -> None:
        directory = self.params_dir if kind == "param" else self.optim_dir
        all_paths: list[Path] = []
        for storage_format in SHARD_EXTENSIONS:
            path = shard_path(directory, shard_name, storage_format)
            all_paths.append(path)
        all_paths.extend(directory.glob(f"{shard_name}.part_*.safetensors"))
        for path in all_paths:
            if path not in kept_paths:
                path.unlink(missing_ok=True)

    def _get_cached_payload(self, kind: str, shard_name: str) -> dict[str, Any] | None:
        self._reap_completed_prefetches()
        key = (kind, shard_name)
        future = self._pending_prefetches.pop(key, None)
        if future is not None:
            payload = future.result()
            self._put_cached_payload(kind, shard_name, payload)
            return payload
        cache = self._cache_for_kind(kind)
        payload = cache.get(shard_name)
        if payload is not None:
            cache.move_to_end(shard_name)
        return payload

    def _put_cached_payload(self, kind: str, shard_name: str, payload: dict[str, Any]) -> None:
        cache = self._cache_for_kind(kind)
        cache[shard_name] = payload
        cache.move_to_end(shard_name)
        while len(cache) > self.effective_prefetch_window():
            cache.popitem(last=False)

    def _module_cache_key(
        self,
        shard_name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[str, str, str]:
        device_index = device.index if device.index is not None else 0
        device_label = f"{device.type}:{device_index}" if device.type == "cuda" else device.type
        return (shard_name, device_label, str(dtype))

    def _put_cached_module(
        self,
        key: tuple[str, str, str],
        module: nn.Module,
    ) -> None:
        self._cached_modules[key] = module
        self._cached_modules.move_to_end(key)
        self._cached_module_parameter_counts[key] = count_module_parameters(module)
        while len(self._cached_modules) > self.max_resident_shards:
            evicted_key = next(
                (
                    candidate
                    for candidate in self._cached_modules
                    if candidate[0] not in self.active_param_shards
                ),
                None,
            )
            if evicted_key is None:
                break
            evicted_module = self._cached_modules.pop(evicted_key)
            self._cached_module_parameter_counts.pop(evicted_key, None)
            evicted_module.to(device="cpu")
            del evicted_module
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _invalidate_cached_module(self, shard_name: str) -> None:
        for key in list(self._cached_modules):
            if key[0] != shard_name:
                continue
            module = self._cached_modules.pop(key)
            self._cached_module_parameter_counts.pop(key, None)
            module.to(device="cpu")
            del module

    def _invalidate_cached_payload(self, kind: str, shard_name: str) -> None:
        self._cache_for_kind(kind).pop(shard_name, None)
        future = self._pending_prefetches.pop((kind, shard_name), None)
        if future is not None:
            future.cancel()

    def _reap_completed_prefetches(self) -> None:
        for key, future in list(self._pending_prefetches.items()):
            if future.done():
                self._pending_prefetches.pop(key)
                kind, shard_name = key
                self._put_cached_payload(kind, shard_name, future.result())

    def _cache_for_kind(self, kind: str) -> OrderedDict[str, dict[str, Any]]:
        if kind == "param":
            return self._cached_param_payloads
        if kind == "optim":
            return self._cached_optimizer_payloads
        raise ValueError(f"unknown shard payload kind: {kind}")

    def _prefetch_target_device(self, active_device: torch.device) -> torch.device:
        if self.prefetch_mode == "cpu":
            return torch.device("cpu")
        if self.prefetch_mode == "gpu":
            if active_device.type != "cuda":
                raise RuntimeError("--prefetch-shards gpu requires CUDA training device")
            return active_device
        if self.prefetch_mode == "secondary-gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("--prefetch-shards secondary-gpu requires CUDA")
            if self.prefetch_device:
                device = torch.device(self.prefetch_device)
                if device.type != "cuda":
                    raise RuntimeError("--prefetch-device for secondary-gpu must be a CUDA device")
                return device
            active_index = active_device.index if active_device.index is not None else 0
            for index in range(torch.cuda.device_count()):
                if index != active_index:
                    return torch.device(f"cuda:{index}")
            raise RuntimeError("--prefetch-shards secondary-gpu requires a second CUDA device")
        raise RuntimeError(f"prefetch is disabled or unknown: {self.prefetch_mode}")


def count_module_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def unique_prefix(values: list[str] | tuple[str, ...], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def pending_prefetch_count(
    pending: dict[tuple[str, str], concurrent.futures.Future[dict[str, Any]]],
    kind: str,
) -> int:
    return sum(1 for pending_kind, _ in pending if pending_kind == kind)


def load_payload_for_prefetch(
    kind: str,
    shard_name: str,
    paths: list[Path],
    target_device: torch.device,
) -> dict[str, Any]:
    if kind == "optim" and not paths:
        return {"shard_name": shard_name, "state": {}}
    payload = load_payload_from_paths(paths, kind=kind, shard_name=shard_name)
    if target_device.type != "cpu":
        payload = move_payload_tensors(payload, target_device)
    return payload


def move_payload_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_payload_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_payload_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_payload_tensors(item, device) for item in value)
    return value


def payload_first_tensor_device(payload: dict[str, Any]) -> torch.device:
    tensor = first_tensor(payload)
    if tensor is None:
        return torch.device("cpu")
    return tensor.device


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def optimizer_state_numel(state: dict[str, dict[str, Any]]) -> int:
    total = 0
    for item in state.values():
        for value in item.values():
            if torch.is_tensor(value):
                total += int(value.numel())
    return total


def optimizer_state_for_parameter(
    item: dict[str, Any],
    parameter: nn.Parameter,
    *,
    device: torch.device,
    optimizer: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"step": int(item.get("step", 0))}
    if optimizer in {"adamw", "lion"}:
        result["exp_avg"] = tensor_or_zeros_like(
            item.get("exp_avg"),
            parameter,
            device=device,
        )
    if optimizer == "adamw":
        result["exp_avg_sq"] = tensor_or_zeros_like(
            item.get("exp_avg_sq"),
            parameter,
            device=device,
        )
    elif optimizer == "adafactor":
        if parameter.ndim == 2:
            result["exp_avg_sq_row"] = tensor_or_zeros(
                item.get("exp_avg_sq_row"),
                parameter.shape[:-1],
                device=device,
            )
            result["exp_avg_sq_col"] = tensor_or_zeros(
                item.get("exp_avg_sq_col"),
                parameter.shape[-1:],
                device=device,
            )
        else:
            result["exp_avg_sq"] = tensor_or_zeros_like(
                item.get("exp_avg_sq"),
                parameter,
                device=device,
            )
    return result


def tensor_or_zeros_like(value: Any, parameter: nn.Parameter, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.zeros_like(parameter, dtype=torch.float32, device=device)


def tensor_or_zeros(value: Any, shape: torch.Size, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.zeros(shape, dtype=torch.float32, device=device)


class ShardStepTransaction:
    def __init__(self, store: ParameterShardStore, step: int, expected_shards: list[str]) -> None:
        self.store = store
        self.step = step
        self.expected_shards = expected_shards
        self.root = store.transactions_dir / f"step_{step:08d}"
        self.params_dir = self.root / "params"
        self.optim_dir = self.root / "optim"
        self.staged_params: set[str] = set()
        self.staged_optim: set[str] = set()
        if self.root.exists():
            shutil.rmtree(self.root)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.optim_dir.mkdir(parents=True, exist_ok=True)

    def param_path(self, shard_name: str) -> Path:
        return self.param_paths(shard_name)[0]

    def optim_path(self, shard_name: str) -> Path:
        return self.optim_paths(shard_name)[0]

    def param_paths(self, shard_name: str) -> list[Path]:
        return payload_paths(
            self.params_dir,
            shard_name,
            self.store.storage_format,
            self.store.storage_part_count(shard_name),
        )

    def optim_paths(self, shard_name: str) -> list[Path]:
        return payload_paths(
            self.optim_dir,
            shard_name,
            self.store.storage_format,
            self.store.storage_part_count(shard_name),
        )

    def mark_parameter_staged(self, shard_name: str) -> None:
        self.staged_params.add(shard_name)

    def mark_optimizer_staged(self, shard_name: str) -> None:
        self.staged_optim.add(shard_name)

    def commit(self) -> None:
        self.store.flush_pending_saves()
        missing_params = [name for name in self.expected_shards if name not in self.staged_params]
        missing_optim = [name for name in self.expected_shards if name not in self.staged_optim]
        if missing_params or missing_optim:
            raise RuntimeError(
                "cannot commit incomplete shard step transaction: "
                f"missing_params={missing_params}, missing_optim={missing_optim}"
            )
        for shard_name in self.expected_shards:
            param_sources = self.param_paths(shard_name)
            optim_sources = self.optim_paths(shard_name)
            missing_param_sources = [path for path in param_sources if not path.exists()]
            missing_optim_sources = [path for path in optim_sources if not path.exists()]
            if missing_param_sources or missing_optim_sources:
                raise RuntimeError(
                    "cannot commit incomplete physical shard files: "
                    f"missing_params={missing_param_sources}, "
                    f"missing_optim={missing_optim_sources}"
                )
            self.store._remove_alternate_payload_paths("param", shard_name, set())
            self.store._remove_alternate_payload_paths("optim", shard_name, set())
            param_targets = self.store.param_paths(shard_name)
            optim_targets = self.store.optim_paths(shard_name)
            for source, target in zip(param_sources, param_targets, strict=True):
                os.replace(source, target)
            for source, target in zip(optim_sources, optim_targets, strict=True):
                os.replace(source, target)
        self.store.sync_storage_metadata()
        shutil.rmtree(self.root, ignore_errors=True)
        remove_empty_dir(self.store.transactions_dir)

    def abort(self) -> None:
        self.store.flush_pending_saves()
        for shard_name in self.staged_params:
            self.store._invalidate_cached_payload("param", shard_name)
            self.store._invalidate_cached_module(shard_name)
        for shard_name in self.staged_optim:
            self.store._invalidate_cached_payload("optim", shard_name)
        shutil.rmtree(self.root, ignore_errors=True)
        remove_empty_dir(self.store.transactions_dir)


def parameter_shard_payload(shard_name: str, module: nn.Module) -> dict[str, Any]:
    state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    return parameter_shard_payload_from_state_dict(shard_name, state_dict, module)


def parameter_shard_payload_from_state_dict(
    shard_name: str,
    state_dict: dict[str, torch.Tensor],
    module: nn.Module,
) -> dict[str, Any]:
    cpu_state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
    return {
        "shard_name": shard_name,
        "state_dict": cpu_state_dict,
        "num_parameters": count_module_parameters(module),
        "updated_unix": time.time(),
    }


def optimizer_state_payload(shard_name: str, state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cpu_state: dict[str, dict[str, Any]] = {}
    for name, item in state.items():
        cpu_item: dict[str, Any] = {}
        for key, value in item.items():
            if torch.is_tensor(value):
                cpu_item[key] = value.detach().cpu()
            elif key == "step":
                cpu_item[key] = int(value)
            else:
                cpu_item[key] = value
        cpu_state[name] = cpu_item
    return {"shard_name": shard_name, "state": cpu_state, "updated_unix": time.time()}


def remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def shard_path(directory: Path, shard_name: str, storage_format: str) -> Path:
    return directory / f"{shard_name}{SHARD_EXTENSIONS[storage_format]}"


def shard_part_path(directory: Path, shard_name: str, part_index: int) -> Path:
    return directory / f"{shard_name}.part_{part_index:03d}.safetensors"


def payload_paths(
    directory: Path,
    shard_name: str,
    storage_format: str,
    part_count: int,
) -> list[Path]:
    if storage_format == "safetensors" and part_count > 1:
        return [shard_part_path(directory, shard_name, index) for index in range(part_count)]
    return [shard_path(directory, shard_name, storage_format)]


def load_payload(path: Path) -> dict[str, Any]:
    suffix = Path(path).suffix
    if suffix == ".safetensors":
        return load_safetensors_payload(path)
    return torch.load(path, map_location="cpu")


def load_payload_from_paths(paths: list[Path], *, kind: str, shard_name: str) -> dict[str, Any]:
    if not paths:
        if kind == "optim":
            return {"shard_name": shard_name, "state": {}}
        raise FileNotFoundError(f"missing parameter payload for shard {shard_name}")
    if len(paths) == 1 and ".part_" not in paths[0].name:
        return load_payload(paths[0])
    return load_safetensors_payload_parts(paths)


def save_payload_atomic(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        if path.suffix == ".safetensors":
            save_safetensors_payload(payload, tmp_path)
        else:
            torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            # On Windows, a failed torch.save can leave the temp file briefly locked.
            # Preserve the original write failure instead of masking it with cleanup noise.
            pass
        raise


def save_payload_to_storage_atomic(
    payload: dict[str, Any],
    directory: Path,
    shard_name: str,
    *,
    storage_format: str,
    part_count: int,
) -> list[Path]:
    paths = payload_paths(directory, shard_name, storage_format, part_count)
    if len(paths) == 1:
        save_payload_atomic(payload, paths[0])
        return paths
    save_safetensors_payload_parts_atomic(payload, paths)
    return paths


def save_safetensors_payload(payload: dict[str, Any], path: Path) -> None:
    if "state_dict" in payload:
        tensors = {
            f"state_dict::{key}": tensor.detach().cpu()
            for key, tensor in payload["state_dict"].items()
        }
        metadata = {
            "kind": "param",
            "shard_name": str(payload["shard_name"]),
            "num_parameters": str(int(payload.get("num_parameters", 0))),
        }
    elif "state" in payload:
        tensors: dict[str, torch.Tensor] = {}
        metadata = {"kind": "optim", "shard_name": str(payload["shard_name"])}
        for name, item in payload["state"].items():
            metadata[f"state::{name}::step"] = str(int(item.get("step", 0)))
            for key, value in item.items():
                if torch.is_tensor(value):
                    tensors[f"state::{name}::{key}"] = value.detach().cpu()
    else:
        raise ValueError("unsupported safetensors payload")
    timestamp = payload.get("updated_unix", payload.get("created_unix", time.time()))
    metadata["timestamp_unix"] = str(float(timestamp))
    save_safetensors_file(tensors, str(path), metadata=metadata)


def load_safetensors_payload(path: Path) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    kind = metadata.get("kind")
    if kind == "param":
        state_dict = {
            key.split("::", maxsplit=1)[1]: tensor
            for key, tensor in tensors.items()
            if key.startswith("state_dict::")
        }
        return {
            "shard_name": metadata["shard_name"],
            "state_dict": state_dict,
            "num_parameters": int(metadata.get("num_parameters", 0)),
            "updated_unix": float(metadata.get("timestamp_unix", 0.0)),
        }
    if kind == "optim":
        state: dict[str, dict[str, Any]] = {}
        for meta_key, value in metadata.items():
            if meta_key.startswith("state::") and meta_key.endswith("::step"):
                _prefix, name, _field = meta_key.split("::", maxsplit=2)
                state.setdefault(name, {})["step"] = int(value)
        for key, tensor in tensors.items():
            if not key.startswith("state::"):
                continue
            _prefix, name, field = key.split("::", maxsplit=2)
            state.setdefault(name, {})[field] = tensor
        return {
            "shard_name": metadata["shard_name"],
            "state": state,
            "updated_unix": float(metadata.get("timestamp_unix", 0.0)),
        }
    raise ValueError(f"unsupported safetensors shard kind in {path}: {kind}")


def save_safetensors_payload_parts_atomic(payload: dict[str, Any], paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths = [
        path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}.{index}")
        for index, path in enumerate(paths)
    ]
    try:
        parts = safetensors_payload_parts(payload, len(paths))
        for tmp_path, part in zip(tmp_paths, parts, strict=True):
            save_safetensors_file(part["tensors"], str(tmp_path), metadata=part["metadata"])
        for tmp_path, path in zip(tmp_paths, paths, strict=True):
            os.replace(tmp_path, path)
    except Exception:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)
        raise


def safetensors_payload_parts(payload: dict[str, Any], part_count: int) -> list[dict[str, Any]]:
    if part_count < 1:
        raise ValueError("part_count must be >= 1")
    kind, shard_name, tensors, base_metadata = flatten_safetensors_payload(payload)
    chunks = split_flat_tensors(tensors, part_count)
    timestamp = payload.get("updated_unix", payload.get("created_unix", time.time()))
    parts: list[dict[str, Any]] = []
    for part_index, part_chunks in enumerate(chunks):
        part_tensors: dict[str, torch.Tensor] = {}
        metadata = {
            **base_metadata,
            "kind": kind,
            "shard_name": shard_name,
            "timestamp_unix": str(float(timestamp)),
            "storage_part_index": str(part_index),
            "storage_part_count": str(part_count),
        }
        for local_index, chunk in enumerate(part_chunks):
            tensor_key = f"tensor_{local_index:05d}"
            part_tensors[tensor_key] = chunk["tensor"].detach().cpu().contiguous()
            metadata[f"{tensor_key}::payload_key"] = str(chunk["payload_key"])
            metadata[f"{tensor_key}::chunk_index"] = str(int(chunk["chunk_index"]))
            metadata[f"{tensor_key}::chunks"] = str(int(chunk["chunks"]))
        if not part_tensors:
            part_tensors["__empty__"] = torch.empty(0, dtype=torch.uint8)
        parts.append({"tensors": part_tensors, "metadata": metadata})
    return parts


def flatten_safetensors_payload(
    payload: dict[str, Any],
) -> tuple[str, str, list[tuple[str, torch.Tensor]], dict[str, str]]:
    if "state_dict" in payload:
        shard_name = str(payload["shard_name"])
        tensors = [
            (key, tensor.detach().cpu())
            for key, tensor in payload["state_dict"].items()
            if torch.is_tensor(tensor)
        ]
        metadata = {"num_parameters": str(int(payload.get("num_parameters", 0)))}
        return "param", shard_name, tensors, metadata
    if "state" in payload:
        shard_name = str(payload["shard_name"])
        tensors: list[tuple[str, torch.Tensor]] = []
        metadata: dict[str, str] = {}
        for name, item in payload["state"].items():
            metadata[f"state::{name}::step"] = str(int(item.get("step", 0)))
            for key, value in item.items():
                if torch.is_tensor(value):
                    tensors.append((f"state::{name}::{key}", value.detach().cpu()))
        return "optim", shard_name, tensors, metadata
    raise ValueError("unsupported safetensors payload")


def split_flat_tensors(
    tensors: list[tuple[str, torch.Tensor]],
    part_count: int,
) -> list[list[dict[str, Any]]]:
    if not tensors:
        return [[] for _ in range(part_count)]
    split_counts = [1 for _ in tensors]
    total_splits = len(split_counts)
    while total_splits < part_count:
        best_index: int | None = None
        best_score = -1.0
        for index, (_key, tensor) in enumerate(tensors):
            if tensor.ndim == 0 or split_counts[index] >= int(tensor.shape[0]):
                continue
            score = float(tensor.numel()) / float(split_counts[index])
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            break
        split_counts[best_index] += 1
        total_splits += 1

    flat_chunks: list[dict[str, Any]] = []
    for (payload_key, tensor), chunks in zip(tensors, split_counts, strict=True):
        if chunks <= 1 or tensor.ndim == 0:
            flat_chunks.append(
                {
                    "payload_key": payload_key,
                    "chunk_index": 0,
                    "chunks": 1,
                    "tensor": tensor,
                }
            )
            continue
        for chunk_index, chunk in enumerate(torch.tensor_split(tensor, chunks, dim=0)):
            flat_chunks.append(
                {
                    "payload_key": payload_key,
                    "chunk_index": chunk_index,
                    "chunks": chunks,
                    "tensor": chunk,
                }
            )

    parts: list[list[dict[str, Any]]] = [[] for _ in range(part_count)]
    for index, chunk in enumerate(flat_chunks):
        parts[index % part_count].append(chunk)
    return parts


def load_safetensors_payload_parts(paths: list[Path]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
    metadata_steps: dict[str, int] = {}
    kind: str | None = None
    shard_name: str | None = None
    num_parameters = 0
    timestamp = 0.0
    for path in sorted(paths):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            kind = metadata.get("kind", kind)
            shard_name = metadata.get("shard_name", shard_name)
            if metadata.get("num_parameters"):
                num_parameters = int(metadata["num_parameters"])
            if metadata.get("timestamp_unix"):
                timestamp = max(timestamp, float(metadata["timestamp_unix"]))
            for meta_key, value in metadata.items():
                if meta_key.startswith("state::") and meta_key.endswith("::step"):
                    _prefix, name, _field = meta_key.split("::", maxsplit=2)
                    metadata_steps[name] = int(value)
            for tensor_key in handle.keys():
                if not tensor_key.startswith("tensor_"):
                    continue
                payload_key = metadata[f"{tensor_key}::payload_key"]
                chunk_index = int(metadata[f"{tensor_key}::chunk_index"])
                chunks = int(metadata[f"{tensor_key}::chunks"])
                grouped.setdefault(payload_key, []).append(
                    (chunk_index, chunks, handle.get_tensor(tensor_key))
                )
    if kind == "param":
        state_dict = {
            key: reconstruct_tensor_chunks(values)
            for key, values in grouped.items()
        }
        return {
            "shard_name": shard_name,
            "state_dict": state_dict,
            "num_parameters": num_parameters,
            "updated_unix": timestamp,
        }
    if kind == "optim":
        state: dict[str, dict[str, Any]] = {
            name: {"step": step} for name, step in metadata_steps.items()
        }
        for key, values in grouped.items():
            if not key.startswith("state::"):
                continue
            _prefix, name, field = key.split("::", maxsplit=2)
            state.setdefault(name, {})[field] = reconstruct_tensor_chunks(values)
        return {
            "shard_name": shard_name,
            "state": state,
            "updated_unix": timestamp,
        }
    raise ValueError(f"unsupported safetensors shard kind in parts {paths}: {kind}")


def reconstruct_tensor_chunks(values: list[tuple[int, int, torch.Tensor]]) -> torch.Tensor:
    values = sorted(values, key=lambda item: item[0])
    expected_chunks = values[0][1]
    if expected_chunks <= 1:
        return values[0][2]
    if len(values) != expected_chunks:
        raise RuntimeError(
            f"incomplete safetensors tensor chunks: have {len(values)}, expected {expected_chunks}"
        )
    return torch.cat([tensor for _index, _chunks, tensor in values], dim=0)


def compute_module_parameter_counts(config: PerkunasV2Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard_name in shard_names(config):
        module = build_module_for_shard(config, shard_name)
        counts[shard_name] = count_module_parameters(module)
        del module
    return counts


def allocate_storage_shard_plan(
    parameter_counts: dict[str, int],
    requested_count: int,
) -> dict[str, int]:
    logical_shards = list(parameter_counts)
    if requested_count <= len(logical_shards):
        return {name: 1 for name in logical_shards}
    total_parameters = max(1, sum(max(0, count) for count in parameter_counts.values()))
    remaining = requested_count - len(logical_shards)
    plan = {name: 1 for name in logical_shards}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for name in logical_shards:
        raw = remaining * max(0, parameter_counts[name]) / total_parameters
        whole = int(raw)
        plan[name] += whole
        assigned += whole
        remainders.append((raw - whole, name))
    for _remainder, name in sorted(remainders, reverse=True)[: remaining - assigned]:
        plan[name] += 1
    return plan


def storage_dtype_for_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported initial shard dtype: {dtype_name}")


def same_path(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left).resolve(strict=False)
    right_path = Path(right).resolve(strict=False)
    return os.path.normcase(str(left_path)) == os.path.normcase(str(right_path))


def path_contains(parent: str | Path, child: str | Path) -> bool:
    parent_path = Path(parent).resolve(strict=False)
    child_path = Path(child).resolve(strict=False)
    if same_path(parent_path, child_path):
        return False
    try:
        child_path.relative_to(parent_path)
        return True
    except ValueError:
        return False


def copy_run_tree_atomic(
    source: str | Path,
    destination: str | Path,
    *,
    excluded_dirs: set[str] | None = None,
    excluded_files: set[str] | None = None,
) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if same_path(source_path, destination_path):
        return
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    ensure_destination_volume_exists(destination_path)
    excluded_dirs = excluded_dirs or set()
    excluded_files = excluded_files or set()
    destination_path.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(source_path):
        dirs[:] = [
            name
            for name in dirs
            if name not in excluded_dirs and not name.endswith(".tmp")
        ]
        root_path = Path(root)
        relative_root = root_path.relative_to(source_path)
        target_root = destination_path / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name in excluded_files or ".tmp." in name:
                continue
            copy_file_atomic(root_path / name, target_root / name)


def copy_file_atomic(source: Path, destination: Path) -> None:
    ensure_destination_volume_exists(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, destination)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def ensure_destination_volume_exists(path: Path) -> None:
    anchor = Path(path.anchor) if path.anchor else None
    if os.name == "nt" and anchor is not None and not anchor.exists():
        raise FileNotFoundError(
            f"destination volume is not mounted: {anchor}. "
            "Create or mount the RAM disk first, or choose an existing active_run_dir path."
        )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def rng_state_json() -> dict[str, Any]:
    data: dict[str, Any] = {"torch_cpu": torch.get_rng_state().cpu().numpy().tobytes().hex()}
    if torch.cuda.is_available():
        data["torch_cuda_all"] = [
            state.cpu().numpy().tobytes().hex() for state in torch.cuda.get_rng_state_all()
        ]
    return data
