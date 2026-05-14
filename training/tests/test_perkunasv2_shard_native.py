from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from perkunas_training.perkunasv2.modules import (
    build_embeddings,
    build_final_norm,
    build_lm_head,
    build_transformer_block,
)
import perkunas_training.perkunasv2.shard_store as shard_store
from perkunas_training.perkunasv2.configuration import (
    PerkunasShardTrainingConfig,
    PerkunasV2Config,
)
from perkunas_training.perkunasv2.hf_export import export_perkunasv2_to_hf
from perkunas_training.train.dataset import LocalityPreservingPackedTokenDataset, PackedTokenDataset
from perkunas_training.perkunasv2.shard_store import ParameterShardStore
from perkunas_training.perkunasv2.trainer import (
    ShardStreamingTrainer,
    global_clip_scale,
    learning_rate_for_update,
    optimizer_update_module,
    select_shard_device,
)


def tiny_config() -> PerkunasV2Config:
    return PerkunasV2Config(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
        max_position_embeddings=32,
        tied_embeddings=False,
    )


def write_tiny_data(path: Path, *, rows: int = 8, seq_len: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    train = rng.integers(0, 256, size=(rows, seq_len + 1), dtype=np.int32)
    val = rng.integers(0, 256, size=(max(2, rows // 2), seq_len + 1), dtype=np.int32)
    np.save(path / "train_00000.npy", train)
    np.save(path / "val_00000.npy", val)


def write_identifiable_shards(path: Path, *, shards: int = 4, rows: int = 3, seq_len: int = 4) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for shard_index in range(shards):
        data = np.zeros((rows, seq_len + 1), dtype=np.int32)
        data[:, 0] = shard_index
        data[:, 1] = np.arange(rows, dtype=np.int32)
        data[:, 2:] = shard_index * 100 + np.arange(2, seq_len + 1, dtype=np.int32)
        np.save(path / f"train_{shard_index:05d}.npy", data)
    np.save(path / "val_00000.npy", np.zeros((rows, seq_len + 1), dtype=np.int32))


def train_config(
    tmp_path: Path,
    *,
    max_steps: int = 2,
    lm_head_chunk_tokens: int = 0,
    async_shard_writes: bool = False,
    optimizer: str = "adamw",
    max_grad_norm: float = 0.0,
    grad_clip_mode: str = "shard",
    max_resident_shards: int = 1,
    prefetch_mode: str = "off",
    trainer_state_every: int = 1,
    master_weight_dtype: str = "compute",
    shard_storage_format: str = "torch",
    storage_shard_count: int = 0,
    shuffle_train: bool = True,
    active_run_dir: str | None = None,
    durable_flush_every: int = 0,
    global_optimizer_every: int = 0,
    global_optimizer_blend: float = 0.25,
    guarded_step_replay: bool = False,
    guard_replay_max_replays: int = 0,
    guard_replay_loss_tolerance: float = 0.0,
    guard_replay_loss_tolerance_ratio: float = 0.0,
    guard_replay_lr_scales: tuple[float, ...] = (1.0, 0.5, 0.25),
    guard_replay_grad_norm_scales: tuple[float, ...] = (1.0,),
    guard_replay_on_exhaust: str = "accept",
) -> PerkunasShardTrainingConfig:
    return PerkunasShardTrainingConfig(
        run_dir=str(tmp_path / "run"),
        active_run_dir=active_run_dir,
        durable_flush_every=durable_flush_every,
        data_dir=str(tmp_path / "data"),
        seq_len=32,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        dtype="fp32",
        master_weight_dtype=master_weight_dtype,
        shard_storage_format=shard_storage_format,
        storage_shard_count=storage_shard_count,
        optimizer=optimizer,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=max_grad_norm,
        grad_clip_mode=grad_clip_mode,
        global_optimizer_every=global_optimizer_every,
        global_optimizer_blend=global_optimizer_blend,
        guarded_step_replay=guarded_step_replay,
        guard_replay_max_replays=guard_replay_max_replays,
        guard_replay_loss_tolerance=guard_replay_loss_tolerance,
        guard_replay_loss_tolerance_ratio=guard_replay_loss_tolerance_ratio,
        guard_replay_lr_scales=guard_replay_lr_scales,
        guard_replay_grad_norm_scales=guard_replay_grad_norm_scales,
        guard_replay_on_exhaust=guard_replay_on_exhaust,
        warmup_steps=1,
        max_steps=max_steps,
        save_every=1,
        validate_every=10,
        max_validation_batches=2,
        shuffle_train=shuffle_train,
        max_resident_shards=max_resident_shards,
        prefetch_mode=prefetch_mode,
        clear_cuda_cache_between_shards=False,
        lm_head_chunk_tokens=lm_head_chunk_tokens,
        async_shard_writes=async_shard_writes,
        max_pending_shard_writes=2,
        trainer_state_every=trainer_state_every,
        device="cpu",
        seed=7,
    )


def initialize_run(tmp_path: Path) -> ParameterShardStore:
    return ParameterShardStore.initialize_random_shards(
        tmp_path / "run",
        tiny_config(),
        seed=7,
        max_resident_shards=1,
    )


def initialize_safetensors_run(tmp_path: Path) -> ParameterShardStore:
    return ParameterShardStore.initialize_random_shards(
        tmp_path / "run",
        tiny_config(),
        seed=7,
        max_resident_shards=1,
        storage_format="safetensors",
    )


class FullResidentPerkunasV2(nn.Module):
    def __init__(self, config: PerkunasV2Config) -> None:
        super().__init__()
        self.embeddings = build_embeddings(config)
        self.blocks = nn.ModuleList(
            build_transformer_block(config, index) for index in range(config.num_layers)
        )
        self.final_norm = build_final_norm(config)
        self.lm_head = build_lm_head(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def load_full_resident_model_from_shards(
    store: ParameterShardStore,
    config: PerkunasV2Config,
) -> FullResidentPerkunasV2:
    model = FullResidentPerkunasV2(config)
    model.embeddings.load_state_dict(torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"])
    for index, block in enumerate(model.blocks):
        block.load_state_dict(
            torch.load(store.param_path(f"block_{index:03d}"), map_location="cpu")["state_dict"]
        )
    model.final_norm.load_state_dict(torch.load(store.param_path("final_norm"), map_location="cpu")["state_dict"])
    model.lm_head.load_state_dict(torch.load(store.param_path("lm_head"), map_location="cpu")["state_dict"])
    return model


def full_resident_optimizer_state(model: nn.Module) -> dict[str, dict[str, torch.Tensor | int]]:
    return {
        name: {
            "step": 0,
            "exp_avg": torch.zeros_like(parameter.detach(), dtype=torch.float32),
            "exp_avg_sq": torch.zeros_like(parameter.detach(), dtype=torch.float32),
        }
        for name, parameter in model.named_parameters()
    }


def run_full_resident_one_step(
    model: FullResidentPerkunasV2,
    config: PerkunasShardTrainingConfig,
) -> tuple[float, float, float]:
    loader = DataLoader(
        PackedTokenDataset(str(Path(config.data_dir) / "train_*.npy")),
        batch_size=config.micro_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    state = full_resident_optimizer_state(model)
    model.train()
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(config.gradient_accumulation_steps):
        input_ids, labels = next(iterator)
        logits = model(input_ids[:, : config.seq_len])
        labels = labels[:, : config.seq_len]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        losses.append(float(loss.detach().cpu()))
        (loss / config.gradient_accumulation_steps).backward()
    global_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            global_norm_sq += float(parameter.grad.detach().float().square().sum().detach().cpu())
    global_norm = global_norm_sq**0.5
    clip_scale = global_clip_scale(global_norm, config.max_grad_norm)
    if clip_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(clip_scale)
    optimizer_update_module(
        model,
        state,
        optimizer=config.optimizer,
        lr=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    return sum(losses) / len(losses), global_norm, clip_scale


def assert_model_matches_shards(
    model: FullResidentPerkunasV2,
    store: ParameterShardStore,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    shard_pairs: list[tuple[str, dict[str, torch.Tensor]]] = [
        ("embeddings", model.embeddings.state_dict()),
        ("final_norm", model.final_norm.state_dict()),
        ("lm_head", model.lm_head.state_dict()),
    ]
    shard_pairs.extend(
        (f"block_{index:03d}", block.state_dict()) for index, block in enumerate(model.blocks)
    )
    for shard_name, expected_state in shard_pairs:
        actual_state = torch.load(store.param_path(shard_name), map_location="cpu")["state_dict"]
        for name, expected in expected_state.items():
            assert torch.allclose(actual_state[name], expected, atol=atol, rtol=rtol), (
                f"{shard_name}.{name} diverged: "
                f"max_abs={(actual_state[name] - expected).abs().max().item()}"
            )


def test_locality_preserving_shuffle_keeps_rows_contiguous_within_shards(tmp_path: Path) -> None:
    write_identifiable_shards(tmp_path / "data", shards=4, rows=3, seq_len=4)
    dataset = LocalityPreservingPackedTokenDataset(
        str(tmp_path / "data" / "train_*.npy"),
        seed=3,
        shuffle_shards=True,
    )

    observed = [(int(input_ids[0]), int(input_ids[1])) for input_ids, _ in dataset]
    shard_order = [observed[offset][0] for offset in range(0, len(observed), 3)]

    assert shard_order == [3, 2, 1, 0]
    for offset in range(0, len(observed), 3):
        group = observed[offset : offset + 3]
        assert len({shard for shard, _ in group}) == 1
        assert [row for _, row in group] == [0, 1, 2]


def test_shuffled_train_loader_uses_locality_preserving_dataset(tmp_path: Path) -> None:
    write_identifiable_shards(tmp_path / "data", shards=3, rows=4, seq_len=32)
    initialize_run(tmp_path)
    trainer = ShardStreamingTrainer(tiny_config(), train_config(tmp_path, shuffle_train=True))

    loader = trainer._loader("train", shuffle=True)

    assert isinstance(loader.dataset, LocalityPreservingPackedTokenDataset)
    assert loader.batch_size == trainer.train_config.micro_batch_size
    assert len(loader.dataset.paths) == 3


def test_shard_initialization_from_config(tmp_path: Path) -> None:
    store = initialize_run(tmp_path)

    assert (tmp_path / "run" / "config.json").exists()
    assert store.param_path("embeddings").exists()
    assert store.param_path("block_000").exists()
    assert store.param_path("block_001").exists()
    assert store.param_path("final_norm").exists()
    assert store.param_path("lm_head").exists()
    metadata = json.loads((tmp_path / "run" / "shards" / "metadata.json").read_text())
    assert metadata["total_parameters"] > 0
    assert metadata["version"] == 1


def test_safetensors_shard_initialization_from_config(tmp_path: Path) -> None:
    store = initialize_safetensors_run(tmp_path)

    assert store.param_path("embeddings").suffix == ".safetensors"
    assert store.param_path("embeddings").exists()
    assert not (tmp_path / "run" / "shards" / "params" / "embeddings.pt").exists()
    tensors = load_file(store.param_path("embeddings"))
    assert "state_dict::weight" in tensors
    metadata = json.loads((tmp_path / "run" / "shards" / "metadata.json").read_text())
    assert metadata["storage_format"] == "safetensors"


def test_hf_export_writes_llama_compatible_tensors(tmp_path: Path) -> None:
    initialize_run(tmp_path)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    export_dir = tmp_path / "export"

    stats = export_perkunasv2_to_hf(
        run_dir=tmp_path / "run",
        tokenizer_dir=tokenizer_dir,
        output_dir=export_dir,
        dtype="fp16",
        max_shard_size="1GB",
    )

    assert stats.files == ["model.safetensors"]
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"
    tensors = load_file(export_dir / "model.safetensors")
    block_payload = torch.load(
        tmp_path / "run" / "shards" / "params" / "block_000.pt",
        map_location="cpu",
    )["state_dict"]
    gate, up = block_payload["mlp.gate_up.weight"].chunk(2, dim=0)
    assert torch.equal(
        tensors["model.layers.0.mlp.gate_proj.weight"],
        gate.to(torch.float16),
    )
    assert torch.equal(
        tensors["model.layers.0.mlp.up_proj.weight"],
        up.to(torch.float16),
    )
    assert tensors["model.embed_tokens.weight"].shape == (256, 64)
    assert tensors["lm_head.weight"].shape == (256, 64)


def test_resident_shard_limit_is_enforced(tmp_path: Path) -> None:
    store = initialize_run(tmp_path)
    device = torch.device("cpu")

    with store.active_module("embeddings", device=device, dtype=torch.float32, training=True):
        with pytest.raises(RuntimeError, match="exceed"):
            with store.active_module("block_000", device=device, dtype=torch.float32, training=True):
                pass

    assert store.residency_snapshot()["active_param_shards"] == {}


def test_cpu_prefetch_caches_and_consumes_parameter_shards(tmp_path: Path) -> None:
    initialize_run(tmp_path)
    store = ParameterShardStore(
        tmp_path / "run",
        config=tiny_config(),
        max_resident_shards=2,
        clear_cuda_cache_between_shards=False,
        prefetch_mode="cpu",
    )
    device = torch.device("cpu")

    store.prefetch_shards(
        param_shards=["embeddings", "block_000"],
        active_device=device,
    )
    store.flush_pending_prefetches()

    snapshot = store.residency_snapshot()
    assert snapshot["cached_param_shards"] == ["embeddings", "block_000"]
    assert snapshot["prefetch_mode"] == "cpu"

    with store.active_module("embeddings", device=device, dtype=torch.float32, training=True):
        pass

    snapshot = store.residency_snapshot()
    assert "embeddings" in snapshot["cached_param_shards"]
    assert "block_000" in snapshot["cached_param_shards"]

    with store.active_module(
        "embeddings",
        device=device,
        dtype=torch.float32,
        training=True,
    ) as module:
        store.save_parameter_shard("embeddings", module)

    snapshot = store.residency_snapshot()
    assert "embeddings" in snapshot["cached_param_shards"]


def test_module_cache_is_opt_in(tmp_path: Path) -> None:
    initialize_run(tmp_path)
    device = torch.device("cpu")
    store = ParameterShardStore(
        tmp_path / "run",
        config=tiny_config(),
        max_resident_shards=5,
        clear_cuda_cache_between_shards=False,
    )

    with store.active_module("embeddings", device=device, dtype=torch.float32, training=True):
        pass

    assert store.residency_snapshot()["cached_module_shards"] == []

    cached_store = ParameterShardStore(
        tmp_path / "run",
        config=tiny_config(),
        max_resident_shards=5,
        clear_cuda_cache_between_shards=False,
        cache_active_modules=True,
    )
    with cached_store.active_module("embeddings", device=device, dtype=torch.float32, training=True):
        pass

    assert cached_store.residency_snapshot()["cached_module_shards"] == ["embeddings"]


def test_training_two_steps_updates_shards_and_optimizer(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    store = initialize_run(tmp_path)
    before = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"][
        "weight"
    ].clone()

    trainer = ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=2))
    result = trainer.train()

    after = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"]["weight"]
    optim_payload = torch.load(store.optim_path("embeddings"), map_location="cpu")
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert result["global_step"] == 2
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after)
    assert optim_payload["state"]["weight"]["step"] >= 1
    assert state["global_step"] == 2
    assert state["optimizer_step"] == 2
    assert not (tmp_path / "run" / "shards" / "transactions").exists()
    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    timing_buckets = [
        "param_load_seconds",
        "module_build_seconds",
        "h2d_seconds",
        "forward_kernel_seconds",
        "backward_kernel_seconds",
        "activation_cpu_copy_seconds",
        "gradient_cpu_copy_seconds",
        "optimizer_load_seconds",
        "optimizer_math_seconds",
        "param_save_stage_seconds",
        "optimizer_save_stage_seconds",
    ]
    assert set(timing_buckets).issubset(row["timing_breakdown"])
    for bucket in timing_buckets:
        assert bucket in row
        assert row[bucket] >= 0.0
        assert row["timing_breakdown"][bucket] == row[bucket]


def test_active_run_dir_flushes_to_durable_archive(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)
    active_run_dir = tmp_path / "active_run"

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=1,
            active_run_dir=str(active_run_dir),
            durable_flush_every=1,
        ),
    )
    result = trainer.train()

    active_state = json.loads((active_run_dir / "trainer_state.json").read_text())
    durable_state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    flush_manifest = json.loads((tmp_path / "run" / "durable_flush_manifest.json").read_text())
    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])

    assert result["run_dir"] == str(tmp_path / "run")
    assert result["active_run_dir"] == str(active_run_dir)
    assert active_state["global_step"] == 1
    assert durable_state["global_step"] == 1
    assert flush_manifest["published_step"] == 1
    assert (active_run_dir / "active_store_manifest.json").exists()
    assert not (tmp_path / "run" / "active_store_manifest.json").exists()
    assert row["residency"]["uses_active_run_dir"] is True


def test_safetensors_training_writes_parameter_and_optimizer_shards(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    store = initialize_safetensors_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, shard_storage_format="safetensors"),
    )
    trainer.train()

    assert store.param_path("embeddings").exists()
    assert store.optim_path("embeddings").exists()
    param_tensors = load_file(store.param_path("embeddings"))
    optim_tensors = load_file(store.optim_path("embeddings"))
    assert "state_dict::weight" in param_tensors
    assert "state::weight::exp_avg" in optim_tensors
    assert "state::weight::exp_avg_sq" in optim_tensors


def test_safetensors_storage_shard_count_splits_physical_files_and_logs(
    tmp_path: Path,
) -> None:
    write_tiny_data(tmp_path / "data")
    store = ParameterShardStore.initialize_random_shards(
        tmp_path / "run",
        tiny_config(),
        seed=7,
        max_resident_shards=8,
        storage_format="safetensors",
        storage_shard_count=8,
    )

    metadata = json.loads((tmp_path / "run" / "shards" / "metadata.json").read_text())
    assert metadata["storage_shard_count"] == 8
    assert sum(metadata["storage_shard_plan"].values()) == 8
    assert len(list((tmp_path / "run" / "shards" / "params").glob("*.safetensors"))) == 8

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=1,
            shard_storage_format="safetensors",
            storage_shard_count=8,
            max_resident_shards=8,
        ),
    )
    trainer.train()

    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    assert row["updated_shards"] == 8
    assert row["optimizer_shards_touched"] == 8
    assert len(list((tmp_path / "run" / "shards" / "optim").glob("*.safetensors"))) == 8
    with store.active_module("embeddings", device=torch.device("cpu"), dtype=torch.float32, training=False):
        pass


def test_initial_weight_dtype_can_store_fp16_shards(tmp_path: Path) -> None:
    store = ParameterShardStore.initialize_random_shards(
        tmp_path / "run",
        tiny_config(),
        storage_format="torch",
        initial_weight_dtype="fp16",
    )

    payload = torch.load(store.param_path("embeddings"), map_location="cpu")
    metadata = json.loads((tmp_path / "run" / "shards" / "metadata.json").read_text())

    assert payload["state_dict"]["weight"].dtype == torch.float16
    assert metadata["initial_weight_dtype"] == "fp16"


def test_multipart_safetensors_save_uses_async_writer_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ParameterShardStore.initialize_random_shards(
        tmp_path / "run",
        tiny_config(),
        seed=7,
        storage_format="safetensors",
        storage_shard_count=8,
    )
    release_save = threading.Event()
    original_save = shard_store.save_safetensors_payload_parts_atomic

    def blocking_save(payload: dict[str, object], paths: list[Path]) -> None:
        release_save.wait(timeout=5)
        original_save(payload, paths)

    monkeypatch.setattr(shard_store, "save_safetensors_payload_parts_atomic", blocking_save)
    store = ParameterShardStore(
        tmp_path / "run",
        config=tiny_config(),
        async_shard_writes=True,
        storage_format="safetensors",
        storage_shard_count=8,
    )

    with store.active_module("block_000", device=torch.device("cpu"), dtype=torch.float32, training=True) as module:
        saved_files = store.save_parameter_shard("block_000", module)

    assert saved_files == 2
    assert store._pending_save_job_count() == 1
    release_save.set()
    store.flush_pending_saves()
    assert all(path.exists() for path in store.param_paths("block_000"))


def test_safetensors_training_can_migrate_legacy_torch_shards(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, shard_storage_format="safetensors"),
    )
    trainer.train()

    assert (tmp_path / "run" / "shards" / "params" / "embeddings.safetensors").exists()
    assert not (tmp_path / "run" / "shards" / "params" / "embeddings.pt").exists()
    assert (tmp_path / "run" / "shards" / "optim" / "embeddings.safetensors").exists()
    metadata = json.loads((tmp_path / "run" / "shards" / "metadata.json").read_text())
    assert metadata["storage_format"] == "safetensors"
    assert metadata["storage_shard_count"] == 5


def test_chunked_lm_head_training_updates_shards(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)
    before = torch.load(store.param_path("lm_head"), map_location="cpu")["state_dict"][
        "weight"
    ].clone()

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, lm_head_chunk_tokens=7),
    )
    result = trainer.train()

    after = torch.load(store.param_path("lm_head"), map_location="cpu")["state_dict"]["weight"]
    assert result["global_step"] == 1
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after)


def test_master_weight_dtype_saves_fp32_parameter_shards(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=4, seq_len=32)
    initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, master_weight_dtype="fp32"),
    )
    trainer.train()

    payload = torch.load(
        tmp_path / "run" / "shards" / "params" / "lm_head.pt",
        map_location="cpu",
    )
    assert {tensor.dtype for tensor in payload["state_dict"].values()} == {torch.float32}


def test_fp32_master_keeps_update_too_small_for_fp16_parameter() -> None:
    module = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.float16)
    module.weight.data.fill_(1.0)
    module.weight.grad = torch.tensor([[1.0e-4]], dtype=torch.float16)
    state = {
        "weight": {
            "step": 0,
            "exp_avg": torch.zeros((1, 1), dtype=torch.float32),
            "exp_avg_sq": torch.zeros((1, 1), dtype=torch.float32),
        }
    }
    master_state_dict = {"weight": torch.ones((1, 1), dtype=torch.float32)}

    optimizer_update_module(
        module,
        state,
        master_state_dict=master_state_dict,
        optimizer="adamw",
        lr=1.0e-5,
        beta1=0.0,
        beta2=0.0,
        eps=1.0e-8,
        weight_decay=0.0,
    )

    assert master_state_dict["weight"].item() == pytest.approx(0.99999, abs=1.0e-7)
    assert module.weight.dtype == torch.float16
    assert module.weight.item() == 1.0


def test_async_shard_writes_flush_to_disk(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, async_shard_writes=True),
    )
    trainer.train()

    payload = torch.load(store.optim_path("lm_head"), map_location="cpu")
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert payload["state"]["weight"]["step"] >= 1
    assert state["global_step"] == 1


def test_transaction_abort_does_not_publish_staged_shard(tmp_path: Path) -> None:
    store = initialize_run(tmp_path)
    device = torch.device("cpu")
    before = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"][
        "weight"
    ].clone()
    transaction = store.begin_step_transaction(1)

    with store.active_module("embeddings", device=device, dtype=torch.float32, training=True) as module:
        with torch.no_grad():
            module.weight.add_(1.0)
        store.save_parameter_shard("embeddings", module, transaction=transaction)

    transaction.abort()

    after = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"]["weight"]
    assert torch.allclose(before, after)


def test_lion_optimizer_updates_shards(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, optimizer="lion", max_grad_norm=1.0),
    )
    trainer.train()

    payload = torch.load(store.optim_path("lm_head"), map_location="cpu")
    item = payload["state"]["weight"]
    assert item["step"] == 1
    assert "exp_avg" in item
    assert "exp_avg_sq" not in item


def test_global_grad_clip_updates_shards_and_logs_global_norm(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=1,
            max_grad_norm=0.25,
            grad_clip_mode="global",
        ),
    )
    trainer.train()

    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    payload = torch.load(store.optim_path("lm_head"), map_location="cpu")
    assert row["grad_clip_mode"] == "global"
    assert row["global_grad_norm"] > 0
    assert 0 < row["global_grad_clip_scale"] <= 1.0
    assert payload["state"]["weight"]["step"] == 1


def test_periodic_global_optimizer_normalizer_logs_scalar_stats(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=2,
            max_grad_norm=0.25,
            grad_clip_mode="shard",
            global_optimizer_every=2,
            global_optimizer_blend=0.5,
        ),
    )
    trainer.train()

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
        if "train_loss" in json.loads(line)
    ]
    assert rows[0]["step"] == 1
    assert rows[0]["global_optimizer"] is None
    assert rows[1]["step"] == 2
    stats = rows[1]["global_optimizer"]
    assert stats["active"] is True
    assert stats["every"] == 2
    assert stats["mode"] == "gradient_rms_shard_normalization"
    assert stats["target_grad_rms"] > 0
    assert 0.5 <= stats["applied_scale_min"] <= stats["applied_scale_max"] <= 2.0


def test_guarded_step_replay_accepts_candidate_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)
    before = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"][
        "weight"
    ].clone()

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=1,
            guarded_step_replay=True,
            guard_replay_max_replays=1,
            guard_replay_on_exhaust="skip",
        ),
    )
    monkeypatch.setattr(trainer, "_batch_loss_for_traces", lambda traces: 0.0)
    trainer.train()

    after = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"]["weight"]
    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert not torch.allclose(before, after)
    assert state["optimizer_step"] == 1
    assert row["guarded_step_replay"]["accepted"] is True
    assert row["guarded_step_replay"]["attempts"] == 1


def test_guarded_step_replay_can_skip_rejected_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)
    before = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"][
        "weight"
    ].clone()

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(
            tmp_path,
            max_steps=1,
            guarded_step_replay=True,
            guard_replay_max_replays=1,
            guard_replay_lr_scales=(1.0, 0.5),
            guard_replay_on_exhaust="skip",
        ),
    )
    monkeypatch.setattr(trainer, "_batch_loss_for_traces", lambda traces: 999.0)
    trainer.train()

    after = torch.load(store.param_path("embeddings"), map_location="cpu")["state_dict"]["weight"]
    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert torch.allclose(before, after)
    assert state["global_step"] == 1
    assert state["optimizer_step"] == 0
    assert row["updated_shards"] == 0
    assert row["optimizer_shards_touched"] == 0
    assert row["guarded_step_replay"]["accepted"] is False
    assert row["guarded_step_replay"]["exhausted_action"] == "skip"
    assert row["guarded_step_replay"]["attempts"] == 2


def test_shard_native_global_clip_matches_full_resident_one_step(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    store = initialize_run(tmp_path)
    config = train_config(
        tmp_path,
        max_steps=1,
        max_grad_norm=0.25,
        grad_clip_mode="global",
        shuffle_train=False,
    )
    full_model = load_full_resident_model_from_shards(store, tiny_config())

    full_loss, full_global_norm, full_clip_scale = run_full_resident_one_step(full_model, config)
    ShardStreamingTrainer(tiny_config(), config).train()

    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    assert row["train_loss"] == pytest.approx(full_loss, abs=1e-6)
    assert row["global_grad_norm"] == pytest.approx(full_global_norm, rel=1e-5, abs=1e-7)
    assert row["global_grad_clip_scale"] == pytest.approx(full_clip_scale, rel=1e-5, abs=1e-7)
    assert_model_matches_shards(full_model, store)


def test_step_activity_reports_updated_shards_with_larger_residency_limit(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, max_resident_shards=5),
    )
    trainer.train()

    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    assert row["updated_shards"] == 5
    assert row["optimizer_shards_touched"] == 5
    assert row["max_active_param_shards_observed"] == 1
    assert row["max_active_optimizer_shards_observed"] == 1


def test_training_log_reports_prefetch_mode(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data", rows=8, seq_len=32)
    initialize_run(tmp_path)

    trainer = ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=1, max_resident_shards=5, prefetch_mode="cpu"),
    )
    trainer.train()

    row = json.loads((tmp_path / "run" / "train_log.jsonl").read_text().splitlines()[-1])
    assert row["residency"]["prefetch_mode"] == "cpu"
    assert row["residency"]["prefetch_window"] == 5
    assert row["residency"]["cached_param_shards"] or row["residency"]["pending_param_prefetches"]
    assert (
        row["residency"]["cached_optimizer_shards"]
        or row["residency"]["pending_optimizer_prefetches"]
    )


def test_token_lr_schedule_uses_tokens_seen(tmp_path: Path) -> None:
    config = train_config(tmp_path)
    config.lr_schedule = "tokens"
    config.learning_rate = 1e-3
    config.warmup_tokens = 100
    config.decay_tokens = 1000
    config.min_lr_ratio = 0.1

    assert learning_rate_for_update(config, step=1, tokens_seen=50) == pytest.approx(5e-4)
    assert learning_rate_for_update(config, step=100, tokens_seen=1000) == pytest.approx(1e-4)


def test_validation_runs_from_shards_with_finite_loss(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)
    trainer = ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=1))

    metrics = trainer.validate(max_batches=2)

    assert metrics["validation_batches"] == 2
    assert np.isfinite(metrics["val_loss"])


def test_validation_requires_val_shards_without_fallback(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    (tmp_path / "data" / "val_00000.npy").unlink()
    initialize_run(tmp_path)
    trainer = ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=1))

    with pytest.raises(FileNotFoundError, match="no validation shards"):
        trainer.validate(max_batches=2)


def test_resume_from_existing_trainer_state(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)
    ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=1)).train()
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert state["global_step"] == 1

    ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=2)).train()
    resumed = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert resumed["global_step"] == 2


def test_trainer_state_is_saved_every_committed_step(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)

    ShardStreamingTrainer(
        tiny_config(),
        train_config(tmp_path, max_steps=3, trainer_state_every=100),
    ).train()

    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert state["global_step"] == 3
    assert state["optimizer_step"] == 3


def test_stale_trainer_state_repairs_from_optimizer_steps(tmp_path: Path) -> None:
    write_tiny_data(tmp_path / "data")
    initialize_run(tmp_path)
    ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=2)).train()

    state_path = tmp_path / "run" / "trainer_state.json"
    stale_state = json.loads(state_path.read_text())
    stale_state["global_step"] = 1
    stale_state["optimizer_step"] = 1
    state_path.write_text(json.dumps(stale_state, indent=2) + "\n")

    ShardStreamingTrainer(tiny_config(), train_config(tmp_path, max_steps=3)).train()

    store = ParameterShardStore(tmp_path / "run", config=tiny_config())
    assert store.optimizer_step_bounds() == (3, 3)
    repaired = json.loads(state_path.read_text())
    assert repaired["global_step"] == 3
    assert repaired["optimizer_step"] == 3


def test_cuda_device_without_index_normalizes_to_cuda_zero(monkeypatch) -> None:
    calls: list[int | None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test CUDA")
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(index))
    monkeypatch.setattr(torch.version, "cuda", "test", raising=False)

    device = select_shard_device("cuda")

    assert str(device) == "cuda:0"
    assert calls == [0]
