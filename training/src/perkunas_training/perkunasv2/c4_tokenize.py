from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from perkunas_training.perkunasv2.c4_training import (
    detect_numbering_gaps,
    discover_c4_parquet_files,
    normalize_c4_text,
    parse_bool,
    should_keep_text,
)
from perkunas_training.utils.io import ensure_dir, read_json, write_json


class C4PackedTokenWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        split: str,
        seq_len: int,
        blocks_per_shard: int,
        shard_index: int = 0,
        token_buffer: list[int] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.seq_len = seq_len
        self.blocks_per_shard = blocks_per_shard
        self.shard_index = max(shard_index, next_shard_index(output_dir, split))
        self.token_buffer: list[int] = list(token_buffer or [])
        self.blocks: list[list[int]] = []
        self.blocks_written = 0
        self.tokens_written = 0

    def add_document(self, token_ids: list[int]) -> None:
        if len(token_ids) < 2:
            return
        self.token_buffer.extend(token_ids)
        block_size = self.seq_len + 1
        while len(self.token_buffer) >= block_size:
            self.blocks.append(self.token_buffer[:block_size])
            del self.token_buffer[: self.seq_len]
            if len(self.blocks) >= self.blocks_per_shard:
                self.flush()

    def flush(self) -> None:
        if not self.blocks:
            return
        array = np.asarray(self.blocks, dtype=np.int32)
        path = self.output_dir / f"{self.split}_{self.shard_index:05d}.npy"
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, path)
        self.blocks_written += int(array.shape[0])
        self.tokens_written += int(array.size)
        self.shard_index += 1
        self.blocks = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seq_len": self.seq_len,
            "blocks_per_shard": self.blocks_per_shard,
            "shard_index": self.shard_index,
            "token_buffer": self.token_buffer[-self.seq_len :],
            "pending_blocks": len(self.blocks),
            "blocks_written_this_process": self.blocks_written,
            "tokens_written_this_process": self.tokens_written,
        }


def tokenize_c4_parquet(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.exists() and not args.overwrite:
        manifest = read_json(manifest_path)
        if manifest.get("complete"):
            print(f"Tokenized C4 manifest already complete: {manifest_path}", flush=True)
            return manifest
    if args.overwrite:
        remove_tokenized_outputs(output_dir)

    tokenizer = load_tokenizer(args.tokenizer_path)
    train_files = limited_files(discover_c4_parquet_files(args.train_data_dir), args.max_train_files)
    val_files = limited_files(discover_c4_parquet_files(args.val_data_dir), args.max_val_files)
    if args.smoke_test:
        train_files = train_files[:1]
        val_files = val_files[:1]
        args.max_train_docs = min(args.max_train_docs or 256, 256)
        args.max_val_docs = min(args.max_val_docs or 128, 128)
        args.blocks_per_shard = min(args.blocks_per_shard, 16)

    print(f"Discovered {len(train_files)} train parquet files", flush=True)
    print(f"Discovered {len(val_files)} validation parquet files", flush=True)
    for warning in [*detect_numbering_gaps(train_files), *detect_numbering_gaps(val_files)]:
        print(f"WARNING: {warning}", flush=True)

    train_result = tokenize_split(
        split="train",
        files=train_files,
        tokenizer=tokenizer,
        args=args,
        max_docs=args.max_train_docs,
    )
    val_result = tokenize_split(
        split="val",
        files=val_files,
        tokenizer=tokenizer,
        args=args,
        max_docs=args.max_val_docs,
    )
    shards = describe_output_shards(output_dir)
    manifest = {
        "stage": "tokenize_c4_parquet",
        "complete": True,
        "output_dir": str(output_dir),
        "tokenizer_path": str(Path(args.tokenizer_path)),
        "seq_len": args.seq_len,
        "text_column": args.text_column,
        "blocks_per_shard": args.blocks_per_shard,
        "train_data_dir": str(args.train_data_dir),
        "val_data_dir": str(args.val_data_dir),
        "train_files": [str(path) for path in train_files],
        "val_files": [str(path) for path in val_files],
        "stats": {
            "train": train_result["stats"],
            "val": val_result["stats"],
            "total_blocks": sum(int(shard["blocks"]) for shard in shards),
            "total_tokens": sum(int(shard["tokens"]) for shard in shards),
        },
        "shards": shards,
    }
    write_json(manifest_path, manifest)
    (output_dir / "tokenization_report.md").write_text(render_report(manifest), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}", flush=True)
    print(f"Output shards: {len(shards)}", flush=True)
    return manifest


def tokenize_split(
    *,
    split: str,
    files: list[Path],
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    max_docs: int | None,
) -> dict[str, Any]:
    if not files:
        raise FileNotFoundError(f"no parquet files available for split {split}")
    output_dir = ensure_dir(args.output_dir)
    progress_path = output_dir / f"{split}_progress.json"
    progress = read_json(progress_path) if args.resume and progress_path.exists() else {}
    stats = Counter(progress.get("stats", {}))
    writer_state = progress.get("writer", {})
    writer = C4PackedTokenWriter(
        output_dir,
        split=split,
        seq_len=args.seq_len,
        blocks_per_shard=args.blocks_per_shard,
        shard_index=int(writer_state.get("shard_index", 0)),
        token_buffer=writer_state.get("token_buffer") or [],
    )
    start_file_index = int(progress.get("next_file_index", 0))
    start_batch_index = int(progress.get("next_batch_index", 0))
    docs_seen_limit_base = int(stats.get("docs_seen", 0))
    text_batch: list[str] = []

    def process_text_batch() -> None:
        nonlocal text_batch
        if not text_batch:
            return
        encodings = tokenizer.encode_batch(text_batch)
        for encoding in encodings:
            token_ids = encoding.ids
            if len(token_ids) < 2:
                stats["skipped_too_few_tokens"] += 1
                continue
            writer.add_document(token_ids)
            stats["documents_tokenized"] += 1
            stats["tokens_encoded"] += len(token_ids)
        text_batch = []

    for file_index, path in enumerate(files[start_file_index:], start=start_file_index):
        pf = pq.ParquetFile(path)
        if args.text_column not in pf.schema_arrow.names:
            raise ValueError(f"text column {args.text_column!r} not found in {path}")
        for batch_index, batch in enumerate(
            pf.iter_batches(batch_size=args.parquet_batch_rows, columns=[args.text_column])
        ):
            if file_index == start_file_index and batch_index < start_batch_index:
                continue
            data = batch.to_pydict()
            for raw_text in data[args.text_column]:
                if max_docs is not None and stats["docs_seen"] - docs_seen_limit_base >= max_docs:
                    process_text_batch()
                    writer.flush()
                    write_split_progress(
                        progress_path,
                        split=split,
                        next_file_index=file_index,
                        next_batch_index=batch_index,
                        current_file=str(path),
                        stats=stats,
                        writer=writer,
                        complete=False,
                    )
                    return {"stats": dict(stats), "complete": False}
                stats["docs_seen"] += 1
                text = normalize_c4_text(raw_text, strip_excessive_whitespace=True)
                if not should_keep_text(
                    text,
                    min_text_chars=args.min_text_chars,
                    enable_basic_filter=args.enable_basic_filter,
                ):
                    stats["docs_filtered"] += 1
                    continue
                text_batch.append(text)
                stats["docs_kept"] += 1
                if len(text_batch) >= args.tokenization_batch_size:
                    process_text_batch()
            process_text_batch()
            writer.flush()
            write_split_progress(
                progress_path,
                split=split,
                next_file_index=file_index,
                next_batch_index=batch_index + 1,
                current_file=str(path),
                stats=stats,
                writer=writer,
                complete=False,
            )
        start_batch_index = 0
        write_split_progress(
            progress_path,
            split=split,
            next_file_index=file_index + 1,
            next_batch_index=0,
            current_file=str(path),
            stats=stats,
            writer=writer,
            complete=False,
        )

    process_text_batch()
    writer.flush()
    write_split_progress(
        progress_path,
        split=split,
        next_file_index=len(files),
        next_batch_index=0,
        current_file=str(files[-1]),
        stats=stats,
        writer=writer,
        complete=True,
    )
    return {"stats": dict(stats), "complete": True}


def write_split_progress(
    path: Path,
    *,
    split: str,
    next_file_index: int,
    next_batch_index: int,
    current_file: str,
    stats: Counter,
    writer: C4PackedTokenWriter,
    complete: bool,
) -> None:
    write_json(
        path,
        {
            "split": split,
            "complete": complete,
            "next_file_index": next_file_index,
            "next_batch_index": next_batch_index,
            "current_file": current_file,
            "stats": dict(stats),
            "writer": writer.state_dict(),
        },
    )


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    path = Path(tokenizer_path)
    tokenizer_file = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_file}")
    return Tokenizer.from_file(str(tokenizer_file))


def limited_files(files: list[Path], limit: int | None) -> list[Path]:
    return files if limit is None else files[:limit]


def next_shard_index(output_dir: Path, split: str) -> int:
    indexes = []
    for path in output_dir.glob(f"{split}_*.npy"):
        try:
            indexes.append(int(path.stem.rsplit("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    return max(indexes, default=-1) + 1


def describe_output_shards(output_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted([*output_dir.glob("train_*.npy"), *output_dir.glob("val_*.npy")]):
        array = np.load(path, mmap_mode="r")
        split = "val" if path.name.startswith("val_") else "train"
        result.append(
            {
                "path": str(path),
                "split": split,
                "blocks": int(array.shape[0]),
                "sequence_length": int(array.shape[1] - 1),
                "tokens": int(array.size),
            }
        )
    return result


def remove_tokenized_outputs(output_dir: Path) -> None:
    for pattern in ("train_*.npy", "val_*.npy", "*.tmp.npy", "*_progress.json", "manifest.json"):
        for path in output_dir.glob(pattern):
            path.unlink()


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Perkunasv2 C4 Tokenization Report",
        "",
        f"Tokenizer: `{manifest['tokenizer_path']}`",
        f"Sequence length: `{manifest['seq_len']}`",
        f"Output dir: `{manifest['output_dir']}`",
        "",
        "## Stats",
        "",
        f"- Train: `{manifest['stats']['train']}`",
        f"- Validation: `{manifest['stats']['val']}`",
        f"- Total blocks: `{manifest['stats']['total_blocks']}`",
        f"- Total tokens: `{manifest['stats']['total_tokens']}`",
        "",
        "## Shards",
        "",
    ]
    lines.extend(
        f"- `{shard['path']}` split={shard['split']} blocks={shard['blocks']} tokens={shard['tokens']}"
        for shard in manifest["shards"]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize C4 parquet into packed Perkunasv2 shards")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", default="training/data/perkunasv2_c4_tokenized")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--blocks-per-shard", type=int, default=4096)
    parser.add_argument("--parquet-batch-rows", type=int, default=1024)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    parser.add_argument("--min-text-chars", type=int, default=50)
    parser.add_argument("--enable-basic-filter", type=parse_bool, default=True)
    parser.add_argument("--max-train-files", type=int)
    parser.add_argument("--max-val-files", type=int)
    parser.add_argument("--max-train-docs", type=int)
    parser.add_argument("--max-val-docs", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = tokenize_c4_parquet(args)
    print(json.dumps({"output_dir": manifest["output_dir"], "shards": len(manifest["shards"])}, indent=2))


if __name__ == "__main__":
    main()
