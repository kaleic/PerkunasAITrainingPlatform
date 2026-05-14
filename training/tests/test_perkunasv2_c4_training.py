from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from perkunas_training.config import TokenizerConfig
from perkunas_training.perkunasv2.c4_training import (
    C4ParquetTokenStream,
    detect_numbering_gaps,
    discover_c4_parquet_files,
    train_perkunasv2_c4,
)
from perkunas_training.perkunasv2.configuration import PerkunasV2Config
from perkunas_training.tokenizer.train_tokenizer import train_perkunas_tokenizer
from perkunas_training.utils.io import write_jsonl


def write_c4_parquet(path: Path, rows: int = 8) -> None:
    text = []
    for index in range(rows):
        text.append(
            (
                "Perkunasv2 streams AllenAI C4 parquet text into shard native training. "
                f"This document {index} has enough alphabetic content for tokenizer packing. "
            )
            * 10
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": text}), path)


def make_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus" / "dedup_00000.jsonl"
    rows = [
        {
            "id": f"doc-{idx}",
            "text": (
                "Perkunasv2 streams C4 parquet text for shard native active parameter "
                f"training sample {idx}."
            ),
            "text_sha256": f"hash-{idx}",
        }
        for idx in range(60)
    ]
    write_jsonl(corpus, rows)
    train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "corpus" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=256,
            min_frequency=1,
            sample_size=10,
        )
    )
    return tmp_path / "tokenizer"


def write_v2_config(path: Path) -> Path:
    config = PerkunasV2Config(
        vocab_size=512,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=256,
        max_position_embeddings=32,
        tied_embeddings=False,
    )
    config.save_json(path)
    return path


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=str(tmp_path / "run"),
        train_data_dir=str(tmp_path / "c4" / "training"),
        val_data_dir=str(tmp_path / "c4" / "validation"),
        config=str(write_v2_config(tmp_path / "perkunasv2.json")),
        tokenizer_path=str(make_tokenizer(tmp_path)),
        text_column="text",
        seq_len=32,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        dtype="fp32",
        device="cpu",
        max_steps=10,
        save_every=1,
        validate_every=1,
        val_batches=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=1,
        parquet_batch_rows=2,
        min_text_chars=50,
        enable_basic_filter=True,
        max_resident_shards=1,
        clear_cuda_cache_between_shards=False,
        log_every=1,
        progress_every_microbatches=1,
        seed=9,
        smoke_test=True,
    )


def test_c4_parquet_discovery_and_gap_warning(tmp_path: Path) -> None:
    train_dir = tmp_path / "c4" / "training"
    write_c4_parquet(train_dir / "c4-train.00000-of-00003.parquet")
    write_c4_parquet(train_dir / "c4-train.00002-of-00003.parquet")

    files = discover_c4_parquet_files(train_dir)
    warnings = detect_numbering_gaps(files)

    assert [path.name for path in files] == [
        "c4-train.00000-of-00003.parquet",
        "c4-train.00002-of-00003.parquet",
    ]
    assert warnings
    assert "missing 1" in warnings[0]


def test_c4_token_stream_packs_fixed_sequences(tmp_path: Path) -> None:
    data_dir = tmp_path / "c4" / "training"
    write_c4_parquet(data_dir / "c4-train.00000-of-00001.parquet", rows=4)
    tokenizer = train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(make_small_jsonl(tmp_path) / "*.jsonl"),
            output_dir=str(tmp_path / "tok"),
            vocab_size=256,
            min_frequency=1,
            sample_size=5,
        )
    )
    token_stream = C4ParquetTokenStream(
        discover_c4_parquet_files(data_dir),
        tokenizer=Tokenizer.from_file(tokenizer["tokenizer_json"]),
        seq_len=32,
        parquet_batch_rows=2,
    )

    input_ids, labels = next(token_stream.iter_batches(2))

    assert input_ids.shape == (2, 32)
    assert labels.shape == (2, 32)
    assert token_stream.state_dict()["current_file"].endswith(".parquet")


def make_small_jsonl(tmp_path: Path) -> Path:
    corpus = tmp_path / "small_corpus"
    rows = [
        {
            "id": f"small-{idx}",
            "text": "Perkunasv2 C4 streaming tokenizer test text with repeated language.",
            "text_sha256": f"small-hash-{idx}",
        }
        for idx in range(20)
    ]
    write_jsonl(corpus / "dedup_00000.jsonl", rows)
    return corpus


def test_perkunasv2_c4_smoke_training_updates_shards(tmp_path: Path) -> None:
    write_c4_parquet(
        tmp_path / "c4" / "training" / "c4-train.00000-of-00001.parquet",
        rows=8,
    )
    write_c4_parquet(
        tmp_path / "c4" / "validation" / "c4-validation.00000-of-00001.parquet",
        rows=4,
    )
    args = make_args(tmp_path)

    result = train_perkunasv2_c4(args)

    state_path = Path(args.run_dir) / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    embeddings = torch.load(
        Path(args.run_dir) / "shards" / "params" / "embeddings.pt",
        map_location="cpu",
    )["state_dict"]["weight"]
    assert result["global_step"] == 2
    assert np.isfinite(result["train_loss"])
    assert np.isfinite(result["latest_validation_loss"])
    assert state["global_step"] == 2
    assert state["tokens_seen"] > 0
    assert state["c4_train_stream"]["current_file"].endswith(".parquet")
    assert torch.isfinite(embeddings).all()
