from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tokenizers import Tokenizer

from perkunas_training.config import TrainConfig
from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM, count_parameters
from perkunas_training.train.checkpoint import load_checkpoint, save_checkpoint
from perkunas_training.train.dataset import PackedTokenDataset, dataset_summary
from perkunas_training.train.device import (
    assert_model_device,
    log_first_batch_verification,
    log_gpu_memory,
    select_device,
    verify_batch_devices,
)
from perkunas_training.utils.io import ensure_dir
from perkunas_training.utils.random import seed_everything


def train(config: TrainConfig) -> dict[str, Any]:
    dist = setup_distributed()
    is_main = dist["rank"] == 0
    seed_everything(config.seed + dist["rank"], deterministic=config.deterministic)
    device = select_device(require_gpu=config.require_gpu, local_rank=dist["local_rank"])
    run_dir = ensure_dir(config.run_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_path = run_dir / "train_log.jsonl"

    model_config = PerkunasConfig.from_yaml(config.model_config)
    tokenizer = Tokenizer.from_file(str(Path(config.tokenizer_dir) / "tokenizer.json"))
    model_config.vocab_size = tokenizer.get_vocab_size()
    model = PerkunasForCausalLM(model_config)
    model = model.to(device)
    assert_model_device(model, require_gpu=config.require_gpu)
    log_gpu_memory("GPU memory allocated after model.to(device)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16" and device.type == "cuda")

    start_step = 0
    if config.resume_from:
        state = load_checkpoint(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        start_step = int(state["step"])

    if dist["world_size"] > 1:
        model = DistributedDataParallel(model, device_ids=[dist["local_rank"]] if device.type == "cuda" else None)
    assert_model_device(model, require_gpu=config.require_gpu)

    train_dataset = PackedTokenDataset(config.train_shards_glob)
    val_dataset = PackedTokenDataset(config.val_shards_glob)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dist["world_size"],
            rank=dist["rank"],
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
        if dist["world_size"] > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    train_iter = infinite_loader(train_loader, train_sampler)

    if is_main:
        metadata = {
            "model_config": model_config.to_dict(),
            "parameters": count_parameters(model.module if hasattr(model, "module") else model),
            "train_dataset": dataset_summary(config.train_shards_glob),
            "val_dataset": dataset_summary(config.val_shards_glob),
            "train_config": asdict(config),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_log = time.perf_counter()
    tokens_since_log = 0
    samples_since_log = 0
    last_loss = 0.0
    first_batch_logged = False

    for step in range(start_step + 1, config.max_steps + 1):
        lr = learning_rate_for_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            input_ids, labels = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            verify_batch_devices(
                input_ids=input_ids,
                labels=labels,
                expected_device=device,
                require_gpu=config.require_gpu,
            )
            if not first_batch_logged:
                log_first_batch_verification(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    batch_size=int(input_ids.shape[0]),
                    require_gpu=config.require_gpu,
                )
                first_batch_logged = True
            with autocast_context(device, config.mixed_precision):
                output = model(input_ids=input_ids, labels=labels)
                loss = output["loss"] / config.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += float(loss.detach().cpu()) * config.gradient_accumulation_steps
            tokens_since_log += int(input_ids.numel())
            samples_since_log += int(input_ids.shape[0])
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        last_loss = accumulated_loss

        if is_main and step % config.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(1e-9, now - last_log)
            row = {
                "step": step,
                "train_loss": last_loss,
                "lr": lr,
                "tokens_per_sec": tokens_since_log / elapsed,
                "samples_per_sec": samples_since_log / elapsed,
                "gpu_memory_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            }
            append_jsonl(log_path, row)
            last_log = now
            tokens_since_log = 0
            samples_since_log = 0

        if is_main and step % config.eval_interval == 0:
            metrics = evaluate_loss(model, val_loader, device, config.mixed_precision, max_batches=20)
            append_jsonl(log_path, {"step": step, **metrics})
            model.train()

        if is_main and step % config.save_interval == 0:
            save_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=model_config,
                step=step,
                metadata={"train_loss": last_loss, "lr": lr},
            )

    if is_main:
        save_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=model_config,
            step=config.max_steps,
            metadata={"train_loss": last_loss, "final": True},
        )
    cleanup_distributed(dist)
    return {"run_dir": str(run_dir), "final_step": config.max_steps}


def learning_rate_for_step(config: TrainConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + cosine * (config.learning_rate - min_lr)


def infinite_loader(loader: DataLoader, sampler: DistributedSampler | None = None) -> Iterator:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, mixed_precision):
            loss = model(input_ids=input_ids, labels=labels)["loss"]
        losses.append(float(loss.detach().cpu()))
    mean_loss = sum(losses) / max(1, len(losses))
    return {"val_loss": mean_loss, "val_perplexity": float(math.exp(min(20, mean_loss)))}


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def setup_distributed() -> dict[str, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def cleanup_distributed(dist: dict[str, int]) -> None:
    if dist["world_size"] > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Perkunas from scratch")
    parser.add_argument("--config", default="training/configs/train.yaml")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    if args.resume_from:
        config.resume_from = args.resume_from
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
