from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from perkunas_training.config import TokenizerConfig
from perkunas_training.perkunasv2.c4_tokenize import tokenize_c4_parquet
from perkunas_training.tokenizer.train_tokenizer import train_perkunas_tokenizer
from perkunas_training.train.dataset import PackedTokenDataset
from perkunas_training.utils.io import write_jsonl


def write_parquet(path: Path, rows: int) -> None:
    text = [
        (
            "Perkunasv2 offline C4 tokenization turns parquet documents into packed "
            f"token shards for stable shard native training sample {index}. "
        )
        * 12
        for index in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": text}), path)


def make_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus" / "dedup_00000.jsonl"
    rows = [
        {
            "id": f"doc-{index}",
            "text": "Perkunasv2 offline C4 tokenization test corpus with repeated text.",
            "text_sha256": f"hash-{index}",
        }
        for index in range(40)
    ]
    write_jsonl(corpus, rows)
    train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "corpus" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=256,
            min_frequency=1,
            sample_size=5,
        )
    )
    return tmp_path / "tokenizer"


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        train_data_dir=str(tmp_path / "c4" / "training"),
        val_data_dir=str(tmp_path / "c4" / "validation"),
        tokenizer_path=str(make_tokenizer(tmp_path)),
        output_dir=str(tmp_path / "tokenized"),
        text_column="text",
        seq_len=32,
        blocks_per_shard=4,
        parquet_batch_rows=2,
        tokenization_batch_size=2,
        min_text_chars=50,
        enable_basic_filter=True,
        max_train_files=None,
        max_val_files=None,
        max_train_docs=None,
        max_val_docs=None,
        resume=True,
        overwrite=False,
        smoke_test=False,
    )


def test_tokenize_c4_parquet_writes_packed_shards(tmp_path: Path) -> None:
    write_parquet(tmp_path / "c4" / "training" / "c4-train.00000-of-00001.parquet", 6)
    write_parquet(tmp_path / "c4" / "validation" / "c4-validation.00000-of-00001.parquet", 3)
    args = make_args(tmp_path)

    manifest = tokenize_c4_parquet(args)

    assert manifest["complete"] is True
    assert manifest["stats"]["train"]["documents_tokenized"] > 0
    assert manifest["stats"]["val"]["documents_tokenized"] > 0
    assert any(shard["split"] == "train" for shard in manifest["shards"])
    assert any(shard["split"] == "val" for shard in manifest["shards"])
    train_dataset = PackedTokenDataset(str(tmp_path / "tokenized" / "train_*.npy"))
    val_dataset = PackedTokenDataset(str(tmp_path / "tokenized" / "val_*.npy"))
    assert train_dataset.sequence_length == 32
    assert val_dataset.sequence_length == 32
    first = np.load(manifest["shards"][0]["path"], mmap_mode="r")
    assert first.shape[1] == 33


def test_tokenize_c4_parquet_resume_complete_manifest(tmp_path: Path) -> None:
    write_parquet(tmp_path / "c4" / "training" / "c4-train.00000-of-00001.parquet", 4)
    write_parquet(tmp_path / "c4" / "validation" / "c4-validation.00000-of-00001.parquet", 2)
    args = make_args(tmp_path)

    first_manifest = tokenize_c4_parquet(args)
    second_manifest = tokenize_c4_parquet(args)

    assert second_manifest["complete"] is True
    assert len(second_manifest["shards"]) == len(first_manifest["shards"])
