from __future__ import annotations

import argparse
import os
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from perkunas_training.config import DataConfig
from perkunas_training.utils.hashing import deterministic_split
from perkunas_training.utils.io import ensure_dir, iter_jsonl, read_json, write_json


class PackedShardWriter:
    def __init__(self, output_dir: Path, split: str, sequence_length: int, blocks_per_shard: int):
        self.output_dir = output_dir
        self.split = split
        self.sequence_length = sequence_length
        self.blocks_per_shard = blocks_per_shard
        self.buffer: list[int] = []
        self.blocks: list[list[int]] = []
        self.shard_index = 0
        self.manifest: list[dict[str, Any]] = []

    def add_document(self, ids: list[int]) -> None:
        self.buffer.extend(ids)
        block_size = self.sequence_length + 1
        while len(self.buffer) >= block_size:
            self.blocks.append(self.buffer[:block_size])
            del self.buffer[:block_size]
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
        self.manifest.append(
            {
                "path": str(path),
                "split": self.split,
                "blocks": int(array.shape[0]),
                "sequence_length": self.sequence_length,
                "tokens": int(array.size),
            }
        )
        self.shard_index += 1
        self.blocks = []


def tokenize_corpus(config: DataConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.tokenized_dir)
    manifest_path = output_dir / "manifest.json"
    if config.resume and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("complete"):
            return manifest

    input_paths = sorted(glob(str(Path(config.dedup_dir) / "*.jsonl")))
    if not input_paths:
        raise FileNotFoundError(f"no deduplicated corpus shards found in {config.dedup_dir}")
    tokenizer_path = Path(config.tokenizer_path) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    blocks_per_shard = max(1, config.output_shard_rows)
    writers = {
        "train": PackedShardWriter(output_dir, "train", config.sequence_length, blocks_per_shard),
        "val": PackedShardWriter(output_dir, "val", config.sequence_length, blocks_per_shard),
    }
    stats = Counter()
    batch_records: list[dict[str, Any]] = []

    def process_batch() -> None:
        nonlocal batch_records
        if not batch_records:
            return
        encodings = tokenizer.encode_batch([record["text"] for record in batch_records])
        for record, encoding in zip(batch_records, encodings, strict=True):
            split = (
                "val"
                if record.get("source_split") == "validation"
                else deterministic_split(record["text_sha256"], config.validation_fraction)
            )
            ids = encoding.ids
            if len(ids) < 2:
                stats["skipped_too_few_tokens"] += 1
                continue
            writers[split].add_document(ids)
            stats[f"{split}_documents"] += 1
            stats[f"{split}_tokens"] += len(ids)
            stats["documents"] += 1
            stats["tokens"] += len(ids)
        batch_records = []

    for path in input_paths:
        for record in iter_jsonl(path):
            batch_records.append(record)
            if len(batch_records) >= config.tokenization_batch_size:
                process_batch()
    process_batch()

    for writer in writers.values():
        writer.flush()

    manifest = {
        "stage": "tokenize",
        "complete": True,
        "input_paths": input_paths,
        "tokenizer_path": str(tokenizer_path),
        "sequence_length": config.sequence_length,
        "validation_fraction": config.validation_fraction,
        "stats": dict(stats),
        "shards": writers["train"].manifest + writers["val"].manifest,
    }
    write_json(manifest_path, manifest)
    (output_dir / "tokenization_report.md").write_text(render_report(manifest), encoding="utf-8")
    return manifest


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Perkunas Tokenization Report",
        "",
        f"Tokenizer: `{manifest['tokenizer_path']}`",
        f"Sequence length: `{manifest['sequence_length']}`",
        f"Validation fraction: `{manifest['validation_fraction']}`",
        "",
        "## Stats",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in manifest["stats"].items())
    lines.extend(["", "## Shards", ""])
    lines.extend(
        f"- `{shard['path']}` split={shard['split']} blocks={shard['blocks']} tokens={shard['tokens']}"
        for shard in manifest["shards"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize and pack Perkunas corpus shards")
    parser.add_argument("--config", default="training/configs/data.yaml")
    args = parser.parse_args()
    manifest = tokenize_corpus(DataConfig.from_yaml(args.config))
    print(f"Tokenized documents: {manifest['stats'].get('documents', 0)}")
    print(f"Tokenized shards: {len(manifest['shards'])}")


if __name__ == "__main__":
    main()
