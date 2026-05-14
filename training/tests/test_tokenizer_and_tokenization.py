from __future__ import annotations

from pathlib import Path

import numpy as np

from perkunas_training.config import DataConfig, TokenizerConfig
from perkunas_training.tokenization.tokenize_corpus import tokenize_corpus
from perkunas_training.tokenizer.train_tokenizer import train_perkunas_tokenizer
from perkunas_training.utils.io import write_jsonl


def make_corpus(path: Path, count: int = 30) -> None:
    rows = []
    for idx in range(count):
        rows.append(
            {
                "id": f"doc-{idx}",
                "text": (
                    "Perkunas trains from scratch on carefully prepared text. "
                    f"This document number {idx} gives the tokenizer repeated structure and variation."
                ),
                "text_sha256": f"hash-{idx}",
                "metadata": {"language": "English"},
            }
        )
    write_jsonl(path, rows)


def test_tokenizer_training_smoke(tmp_path: Path) -> None:
    corpus = tmp_path / "dedup" / "dedup_00000.jsonl"
    make_corpus(corpus)
    result = train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "dedup" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=300,
            min_frequency=1,
            sample_size=10,
        )
    )
    assert Path(result["tokenizer_json"]).exists()
    assert result["evaluation"]["sample_count"] == 10
    assert result["evaluation"]["average_chars_per_token"] > 0


def test_tokenization_pipeline_smoke(tmp_path: Path) -> None:
    corpus = tmp_path / "dedup" / "dedup_00000.jsonl"
    make_corpus(corpus, count=50)
    train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "dedup" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=300,
            min_frequency=1,
            sample_size=10,
        )
    )
    manifest = tokenize_corpus(
        DataConfig(
            input_paths=[],
            dedup_dir=str(tmp_path / "dedup"),
            tokenized_dir=str(tmp_path / "tokenized"),
            tokenizer_path=str(tmp_path / "tokenizer"),
            sequence_length=16,
            output_shard_rows=4,
            validation_fraction=0.4,
            tokenization_batch_size=8,
            resume=False,
        )
    )
    assert manifest["stats"]["documents"] == 50
    assert any(shard["split"] == "train" for shard in manifest["shards"])
    assert any(shard["split"] == "val" for shard in manifest["shards"])
    first = np.load(manifest["shards"][0]["path"], mmap_mode="r")
    assert first.shape[1] == 17
