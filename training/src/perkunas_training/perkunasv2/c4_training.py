from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from perkunas_training.perkunasv2.configuration import (
    PerkunasShardTrainingConfig,
    PerkunasV2Config,
)
from perkunas_training.perkunasv2.shard_store import ParameterShardStore
from perkunas_training.perkunasv2.trainer import (
    ShardStreamingTrainer,
    learning_rate_for_update,
    memory_snapshot,
)
from perkunas_training.utils.io import write_json


PARQUET_INDEX_RE = re.compile(r"\.(\d+)-of-(\d+)\.parquet$", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\f\v]+")
BLANK_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class C4StreamPosition:
    epoch: int = 0
    file_index: int = 0
    batch_index: int = 0
    row_offset: int = 0
    current_file: str | None = None
    token_buffer_remainder: list[int] | None = None
    approximate_resume: bool = True

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "C4StreamPosition":
        if not state:
            return cls()
        return cls(
            epoch=int(state.get("epoch", 0)),
            file_index=int(state.get("current_train_file_index", state.get("file_index", 0))),
            batch_index=int(state.get("current_row_group_index", state.get("batch_index", 0))),
            row_offset=int(state.get("current_row_offset", state.get("row_offset", 0))),
            current_file=state.get("current_file"),
            token_buffer_remainder=list(state.get("token_buffer_remainder") or []),
            approximate_resume=bool(state.get("approximate_resume", True)),
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "current_train_file_index": self.file_index,
            "current_row_group_index": self.batch_index,
            "current_row_offset": self.row_offset,
            "current_file": self.current_file,
            "token_buffer_remainder": self.token_buffer_remainder or [],
            "approximate_resume": self.approximate_resume,
            "resume_note": (
                "C4 parquet resume is safe approximate resume from the latest file index; "
                "row-level exact replay is not guaranteed."
            ),
        }


class C4ParquetTokenStream:
    def __init__(
        self,
        files: list[str | Path],
        tokenizer: Tokenizer,
        *,
        seq_len: int,
        text_column: str = "text",
        parquet_batch_rows: int = 1024,
        min_text_chars: int = 50,
        enable_basic_filter: bool = True,
        strip_excessive_whitespace: bool = True,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.files = natural_sort_paths(files)
        if not self.files:
            raise FileNotFoundError("no C4 parquet files were discovered")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.parquet_batch_rows = parquet_batch_rows
        self.min_text_chars = min_text_chars
        self.enable_basic_filter = enable_basic_filter
        self.strip_excessive_whitespace = strip_excessive_whitespace
        self.position = C4StreamPosition.from_state(initial_state)
        self.token_buffer: list[int] = list(self.position.token_buffer_remainder or [])
        if self.position.file_index >= len(self.files):
            self.position.file_index = 0
            self.position.epoch += 1

    def iter_sequences(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            for file_index in range(self.position.file_index, len(self.files)):
                path = self.files[file_index]
                self.position.file_index = file_index
                self.position.current_file = str(path)
                pf = pq.ParquetFile(path)
                if self.text_column not in pf.schema_arrow.names:
                    raise ValueError(f"text column {self.text_column!r} not found in {path}")
                row_offset = 0
                for batch_index, batch in enumerate(
                    pf.iter_batches(batch_size=self.parquet_batch_rows, columns=[self.text_column])
                ):
                    self.position.batch_index = batch_index
                    data = batch.to_pydict()
                    texts = data[self.text_column]
                    for raw_text in texts:
                        self.position.row_offset = row_offset
                        row_offset += 1
                        text = normalize_c4_text(raw_text, self.strip_excessive_whitespace)
                        if not should_keep_text(
                            text,
                            min_text_chars=self.min_text_chars,
                            enable_basic_filter=self.enable_basic_filter,
                        ):
                            continue
                        self.token_buffer.extend(self.tokenizer.encode(text).ids)
                        yield from self._drain_sequences()
                self.position.batch_index = 0
                self.position.row_offset = 0
                self.token_buffer = self.token_buffer[-self.seq_len :]
            self.position.file_index = 0
            self.position.epoch += 1

    def iter_batches(self, micro_batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        iterator = self.iter_sequences()
        while True:
            inputs: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            for _ in range(micro_batch_size):
                input_ids, target_ids = next(iterator)
                inputs.append(input_ids)
                labels.append(target_ids)
            yield torch.stack(inputs), torch.stack(labels)

    def state_dict(self) -> dict[str, Any]:
        self.position.token_buffer_remainder = self.token_buffer[-self.seq_len :]
        return self.position.to_state()

    def current_file_label(self) -> str:
        return self.position.current_file or str(self.files[self.position.file_index])

    def _drain_sequences(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        chunk = self.seq_len + 1
        while len(self.token_buffer) >= chunk:
            tokens = self.token_buffer[:chunk]
            del self.token_buffer[: self.seq_len]
            tensor = torch.tensor(tokens, dtype=torch.long)
            self.position.token_buffer_remainder = self.token_buffer[-self.seq_len :]
            yield tensor[:-1], tensor[1:]


def discover_c4_parquet_files(data_dir: str | Path) -> list[Path]:
    return natural_sort_paths(Path(data_dir).glob("*.parquet"))


def natural_sort_paths(paths: Any) -> list[Path]:
    return sorted((Path(path) for path in paths), key=natural_sort_key)


def natural_sort_key(path: Path) -> tuple[str, int, str]:
    match = PARQUET_INDEX_RE.search(path.name)
    if match:
        return (path.name[: match.start()], int(match.group(1)), path.name)
    numbers = re.findall(r"\d+", path.name)
    return (path.name, int(numbers[-1]) if numbers else -1, path.name)


def detect_numbering_gaps(paths: list[Path]) -> list[str]:
    warnings: list[str] = []
    grouped: dict[str, tuple[int, set[int]]] = {}
    for path in paths:
        match = PARQUET_INDEX_RE.search(path.name)
        if not match:
            continue
        prefix = path.name[: match.start()]
        index = int(match.group(1))
        total = int(match.group(2))
        if prefix not in grouped:
            grouped[prefix] = (total, set())
        grouped[prefix][1].add(index)
    for prefix, (total, seen) in grouped.items():
        missing = sorted(set(range(total)) - seen)
        if missing:
            preview = ", ".join(str(item) for item in missing[:10])
            more = "..." if len(missing) > 10 else ""
            warnings.append(
                f"{prefix}: found {len(seen)} of {total} parquet shards; missing {preview}{more}"
            )
    return warnings


def normalize_c4_text(raw_text: Any, strip_excessive_whitespace: bool = True) -> str:
    if raw_text is None:
        return ""
    text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if strip_excessive_whitespace:
        text = SPACE_RE.sub(" ", text)
        text = BLANK_RE.sub("\n\n", text)
    return text


def should_keep_text(text: str, *, min_text_chars: int, enable_basic_filter: bool) -> bool:
    if len(text) < min_text_chars:
        return False
    if not enable_basic_filter:
        return True
    non_space = sum(not char.isspace() for char in text)
    if non_space == 0:
        return False
    alpha = sum(char.isalpha() for char in text)
    alnum = sum(char.isalnum() for char in text)
    return alpha / non_space >= 0.20 and alnum / non_space >= 0.35


def load_project_tokenizer(run_dir: str | Path, tokenizer_path: str | Path | None) -> Tokenizer:
    run_tokenizer = Path(run_dir) / "tokenizer" / "tokenizer.json"
    if run_tokenizer.exists():
        return Tokenizer.from_file(str(run_tokenizer))
    if tokenizer_path is None:
        raise FileNotFoundError(
            f"missing tokenizer: expected {run_tokenizer} or pass --tokenizer-path"
        )
    path = Path(tokenizer_path)
    tokenizer_file = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_file}")
    return Tokenizer.from_file(str(tokenizer_file))


def copy_tokenizer_to_run_dir(run_dir: str | Path, tokenizer_path: str | Path | None) -> None:
    if tokenizer_path is None:
        return
    source = Path(tokenizer_path)
    source_dir = source if source.is_dir() else source.parent
    target = Path(run_dir) / "tokenizer"
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, target / filename)


def ensure_perkunasv2_shards(run_dir: str | Path, config_path: str | Path, *, seed: int) -> None:
    run_dir = Path(run_dir)
    metadata = run_dir / "shards" / "metadata.json"
    if metadata.exists():
        print(f"Resuming existing Perkunasv2 shard run: {run_dir}", flush=True)
        return
    resolved_config = resolve_config_path(config_path)
    config = PerkunasV2Config.from_json(resolved_config)
    ParameterShardStore.initialize_random_shards(run_dir, config, seed=seed)
    print(f"Initialized Perkunasv2 shards in {run_dir}", flush=True)


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.exists():
        return path
    fallback = Path("training") / "configs" / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Perkunasv2 config not found: {config_path}")


def validate_tokenizer_vocab(tokenizer: Tokenizer, model_config: PerkunasV2Config) -> None:
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size={vocab_size} exceeds model vocab_size={model_config.vocab_size}"
        )
    if vocab_size < model_config.vocab_size:
        print(
            f"WARNING: tokenizer vocab_size={vocab_size} is smaller than model "
            f"vocab_size={model_config.vocab_size}; extra embeddings will be unused until tokenizer changes.",
            flush=True,
        )


def train_perkunasv2_c4(args: argparse.Namespace) -> dict[str, Any]:
    train_files = discover_c4_parquet_files(args.train_data_dir)
    val_files = discover_c4_parquet_files(args.val_data_dir)
    print(f"Discovered {len(train_files)} train parquet files in {args.train_data_dir}", flush=True)
    print(f"Discovered {len(val_files)} validation parquet files in {args.val_data_dir}", flush=True)
    for warning in [*detect_numbering_gaps(train_files), *detect_numbering_gaps(val_files)]:
        print(f"WARNING: {warning}", flush=True)

    ensure_perkunasv2_shards(args.run_dir, args.config, seed=args.seed)
    copy_tokenizer_to_run_dir(args.run_dir, args.tokenizer_path)
    model_config = PerkunasV2Config.from_json(Path(args.run_dir) / "config.json")
    tokenizer = load_project_tokenizer(args.run_dir, args.tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_config)

    train_config = PerkunasShardTrainingConfig(
        run_dir=args.run_dir,
        data_dir=str(args.train_data_dir),
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dtype=args.dtype,
        optimizer=getattr(args, "optimizer", "adamw"),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=getattr(args, "beta1", 0.9),
        beta2=getattr(args, "beta2", 0.95),
        adam_eps=getattr(args, "adam_eps", 1e-8),
        max_grad_norm=getattr(args, "max_grad_norm", 0.0),
        lr_schedule=getattr(args, "lr_schedule", "steps"),
        warmup_steps=args.warmup_steps,
        warmup_tokens=getattr(args, "warmup_tokens", 0),
        decay_tokens=getattr(args, "decay_tokens", 0),
        min_lr_ratio=getattr(args, "min_lr_ratio", 0.1),
        max_steps=2 if args.smoke_test else args.max_steps,
        save_every=args.save_every,
        validate_every=1 if args.smoke_test else args.validate_every,
        max_validation_batches=2 if args.smoke_test else args.val_batches,
        max_resident_shards=args.max_resident_shards,
        cache_active_modules=getattr(args, "cache_active_modules", False),
        prefetch_mode=getattr(args, "prefetch_shards", "off"),
        prefetch_window=getattr(args, "prefetch_window", 0),
        prefetch_optimizer_shards=getattr(args, "prefetch_optimizer_shards", True),
        prefetch_device=getattr(args, "prefetch_device", None),
        trace_storage_mode=getattr(args, "trace_storage", "cpu"),
        trace_storage_device=getattr(args, "trace_storage_device", None),
        clear_cuda_cache_between_shards=args.clear_cuda_cache_between_shards,
        shard_log_every=getattr(args, "shard_log_every", 1),
        trainer_state_every=getattr(args, "trainer_state_every", 1),
        lm_head_chunk_tokens=getattr(args, "lm_head_chunk_tokens", 0),
        async_shard_writes=getattr(args, "async_shard_writes", False),
        max_pending_shard_writes=getattr(args, "max_pending_shard_writes", 4),
        device=args.device,
        seed=args.seed,
    )
    trainer = ShardStreamingTrainer(model_config, train_config)
    state = trainer.store.load_trainer_state()
    start_step = int(state.get("global_step", 0))
    tokens_seen = int(state.get("tokens_seen", 0))
    start_data_state = state.get("c4_train_stream")
    if start_data_state:
        print(
            "C4 resume is approximate: restarting at latest recorded file index, "
            "not exact parquet row replay.",
            flush=True,
        )

    train_stream = C4ParquetTokenStream(
        train_files,
        tokenizer,
        seq_len=args.seq_len,
        text_column=args.text_column,
        parquet_batch_rows=args.parquet_batch_rows,
        min_text_chars=args.min_text_chars,
        enable_basic_filter=args.enable_basic_filter,
        initial_state=start_data_state,
    )
    train_batches = train_stream.iter_batches(args.micro_batch_size)
    last_log = time.perf_counter()
    latest_val_loss = state.get("latest_validation_loss")
    final_train_loss = math.nan
    final_metrics: dict[str, float] = {}
    before_smoke = load_smoke_reference_shard(args.run_dir) if args.smoke_test else None
    trainer.store.discard_stale_transactions()

    for step in range(start_step + 1, train_config.max_steps + 1):
        lr = learning_rate_for_update(train_config, step, tokens_seen)
        traces = []
        tokens_this_step = 0
        write_progress(
            args.run_dir,
            {
                "phase": "forward_accumulation",
                "step": step,
                "global_step_completed": start_step if step == start_step + 1 else step - 1,
                "microbatch": 0,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "tokens_seen": tokens_seen,
                "current_parquet_file": train_stream.current_file_label(),
                "learning_rate": lr,
                "memory": memory_snapshot(trainer.device),
            },
        )
        for _ in range(args.gradient_accumulation_steps):
            microbatch_index = len(traces) + 1
            input_ids, labels = next(train_batches)
            trace = trainer.forward_trace(input_ids, labels, training=True, compute_loss=False)
            traces.append(trace)
            tokens_this_step += int(input_ids.numel())
            if microbatch_index % args.progress_every_microbatches == 0:
                write_progress(
                    args.run_dir,
                    {
                        "phase": "forward_accumulation",
                        "step": step,
                        "global_step_completed": start_step if step == start_step + 1 else step - 1,
                        "microbatch": microbatch_index,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "partial_train_loss": None,
                        "tokens_this_step": tokens_this_step,
                        "tokens_seen": tokens_seen,
                        "current_parquet_file": train_stream.current_file_label(),
                        "learning_rate": lr,
                        "memory": memory_snapshot(trainer.device),
                    },
                )
        write_progress(
            args.run_dir,
            {
                "phase": "shard_backward_update",
                "step": step,
                "microbatch": len(traces),
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "train_loss": None,
                "tokens_this_step": tokens_this_step,
                "tokens_seen": tokens_seen,
                "current_parquet_file": train_stream.current_file_label(),
                "learning_rate": lr,
                "memory": memory_snapshot(trainer.device),
            },
        )
        lr = learning_rate_for_update(train_config, step, tokens_seen + tokens_this_step)
        transaction = trainer.store.begin_step_transaction(step)
        try:
            final_train_loss = trainer.backward_update(
                traces,
                lr=lr,
                step=step,
                transaction=transaction,
            )
            transaction.commit()
            trainer._prime_next_step_prefetch()
        except BaseException:
            transaction.abort()
            raise
        tokens_seen += tokens_this_step

        if step % train_config.validate_every == 0:
            write_progress(
                args.run_dir,
                {
                    "phase": "validation",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "current_parquet_file": train_stream.current_file_label(),
                    "memory": memory_snapshot(trainer.device),
                },
            )
            final_metrics = validate_c4_stream(
                trainer,
                val_files,
                tokenizer,
                args=args,
                max_batches=train_config.max_validation_batches,
            )
            latest_val_loss = final_metrics["val_loss"]
            trainer.append_log({"step": step, **final_metrics})

        trainer_state = {
            "global_step": step,
            "tokens_seen": tokens_seen,
            "optimizer_step": int(state.get("optimizer_step", 0)) + (step - start_step),
            "scheduler_state": {
                "learning_rate": lr,
                "lr_schedule": train_config.lr_schedule,
                "tokens_seen": tokens_seen,
            },
            "latest_validation_loss": latest_val_loss,
            "config_hash": model_config.stable_hash(),
            "c4_train_stream": train_stream.state_dict(),
        }
        if (
            step % train_config.trainer_state_every == 0
            or step % train_config.save_every == 0
            or step == train_config.max_steps
        ):
            trainer.store.save_trainer_state(trainer_state)
        if step % train_config.save_every == 0:
            trainer.write_checkpoint_marker(step)

        now = time.perf_counter()
        elapsed = max(1e-9, now - last_log)
        tokens_per_sec = tokens_this_step / elapsed
        last_log = now
        log_row = {
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss": final_train_loss,
            "val_loss": latest_val_loss,
            "val_perplexity": (
                float(math.exp(min(20, latest_val_loss))) if latest_val_loss is not None else None
            ),
            "learning_rate": lr,
            "optimizer": train_config.optimizer,
            "grad_norm": trainer._last_step_grad_norms,
            **trainer._last_step_activity.to_dict(),
            "tokens_per_sec": tokens_per_sec,
            "current_parquet_file": train_stream.current_file_label(),
            "trace_storage": trainer._trace_storage_snapshot(),
            "memory": memory_snapshot(trainer.device),
            "residency": trainer.store.residency_snapshot(),
        }
        trainer.append_log(log_row)
        if step % args.log_every == 0 or args.smoke_test:
            print_training_log(log_row)
        write_progress(
            args.run_dir,
            {
                "phase": "step_complete",
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": final_train_loss,
                "val_loss": latest_val_loss,
                "learning_rate": lr,
                "tokens_per_sec": tokens_per_sec,
                **trainer._last_step_activity.to_dict(),
                "current_parquet_file": train_stream.current_file_label(),
                "trace_storage": trainer._trace_storage_snapshot(),
                "memory": memory_snapshot(trainer.device),
                "residency": trainer.store.residency_snapshot(),
            },
        )

    if args.smoke_test:
        after_smoke = load_smoke_reference_shard(args.run_dir)
        trainer_state_path = Path(args.run_dir) / "trainer_state.json"
        if not math.isfinite(final_train_loss):
            raise RuntimeError("smoke test train loss is not finite")
        if not final_metrics or not math.isfinite(final_metrics["val_loss"]):
            raise RuntimeError("smoke test validation loss is not finite")
        if not trainer_state_path.exists():
            raise RuntimeError("smoke test did not write trainer_state.json")
        if before_smoke is not None and torch.allclose(before_smoke, after_smoke):
            raise RuntimeError("smoke test did not update embeddings shard")
        print("Perkunasv2 C4 smoke test passed", flush=True)

    return {
        "run_dir": str(args.run_dir),
        "global_step": train_config.max_steps,
        "tokens_seen": tokens_seen,
        "train_loss": final_train_loss,
        "latest_validation_loss": latest_val_loss,
    }


def validate_c4_stream(
    trainer: ShardStreamingTrainer,
    val_files: list[Path],
    tokenizer: Tokenizer,
    *,
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, float]:
    stream = C4ParquetTokenStream(
        val_files,
        tokenizer,
        seq_len=args.seq_len,
        text_column=args.text_column,
        parquet_batch_rows=args.parquet_batch_rows,
        min_text_chars=args.min_text_chars,
        enable_basic_filter=args.enable_basic_filter,
    )
    losses: list[float] = []
    for batch_index, (input_ids, labels) in enumerate(stream.iter_batches(args.micro_batch_size)):
        if batch_index >= max_batches:
            break
        trace = trainer.forward_trace(input_ids, labels, training=False)
        losses.append(trace.loss)
    if not losses:
        raise RuntimeError("validation produced no batches")
    mean_loss = sum(losses) / len(losses)
    metrics = {
        "val_loss": mean_loss,
        "val_perplexity": float(math.exp(min(20, mean_loss))),
        "validation_batches": float(len(losses)),
    }
    print(
        f"validation loss={mean_loss:.4f} ppl={metrics['val_perplexity']:.2f} "
        f"memory={memory_snapshot(trainer.device)}",
        flush=True,
    )
    return metrics


def print_training_log(row: dict[str, Any]) -> None:
    memory = row["memory"]
    print(
        f"step={row['step']} tokens_seen={row['tokens_seen']} "
        f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']} "
        f"ppl={row['val_perplexity']} lr={row['learning_rate']:.6g} "
        f"tokens/sec={row['tokens_per_sec']:.1f} "
        f"vram_alloc={memory['allocated_mb']:.2f}MB "
        f"vram_reserved={memory['reserved_mb']:.2f}MB "
        f"vram_peak={memory['peak_allocated_mb']:.2f}MB "
        f"file={row['current_parquet_file']}",
        flush=True,
    )


def write_progress(run_dir: str | Path, row: dict[str, Any]) -> None:
    payload = dict(row)
    payload["updated_unix"] = time.time()
    write_json(Path(run_dir) / "c4_progress.json", payload)


def load_smoke_reference_shard(run_dir: str | Path) -> torch.Tensor:
    payload = torch.load(Path(run_dir) / "shards" / "params" / "embeddings.pt", map_location="cpu")
    return payload["state_dict"]["weight"].detach().clone()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Perkunasv2 shard-native from C4 parquet")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", choices=["adamw", "lion", "adafactor"], default="adamw")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=["steps", "tokens"], default="steps")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--decay-tokens", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--parquet-batch-rows", type=int, default=1024)
    parser.add_argument("--min-text-chars", type=int, default=50)
    parser.add_argument("--enable-basic-filter", type=parse_bool, default=True)
    parser.add_argument("--max-resident-shards", type=int, default=1)
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
        help="Write trainer_state.json every N steps; larger values trade resume precision for speed.",
    )
    parser.add_argument(
        "--lm-head-chunk-tokens",
        type=int,
        default=0,
        help="Chunk lm_head loss/backward by token count; 0 keeps the full logits path.",
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--progress-every-microbatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = train_perkunasv2_c4(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
