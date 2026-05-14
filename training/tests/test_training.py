from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from perkunas_training.config import TokenizerConfig, TrainConfig, write_yaml
from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
from perkunas_training.tokenizer.train_tokenizer import train_perkunas_tokenizer
from perkunas_training.train.checkpoint import load_checkpoint, save_checkpoint
from perkunas_training.train.train_perkunas import train
from perkunas_training.utils.io import write_jsonl


def make_tokenizer(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus" / "dedup_00000.jsonl"
    rows = [
        {
            "id": f"doc-{idx}",
            "text": f"Perkunas miniature training sample {idx}. The model learns tiny patterns.",
            "text_sha256": f"hash-{idx}",
        }
        for idx in range(40)
    ]
    write_jsonl(corpus, rows)
    train_perkunas_tokenizer(
        TokenizerConfig(
            input_glob=str(tmp_path / "corpus" / "*.jsonl"),
            output_dir=str(tmp_path / "tokenizer"),
            vocab_size=300,
            min_frequency=1,
            sample_size=5,
        )
    )
    return tmp_path / "tokenizer"


def write_shards(tmp_path: Path, vocab_size: int = 300) -> tuple[str, str]:
    rng = np.random.default_rng(3)
    train = rng.integers(0, vocab_size, size=(8, 17), dtype=np.int32)
    val = rng.integers(0, vocab_size, size=(4, 17), dtype=np.int32)
    train_dir = tmp_path / "tok"
    train_dir.mkdir()
    np.save(train_dir / "train_00000.npy", train)
    np.save(train_dir / "val_00000.npy", val)
    return str(train_dir / "train_*.npy"), str(train_dir / "val_*.npy")


def write_tiny_model_config(tmp_path: Path) -> Path:
    path = tmp_path / "model_tiny.yaml"
    write_yaml(
        path,
        {
            "model_name": "perkunas-tiny",
            "vocab_size": 300,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "activation": "swiglu",
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "tie_word_embeddings": True,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
            "modality_extension": {"enabled": True, "projection_dim": 32},
            "auxiliary_heads": {"embedding": False},
        },
    )
    return path


def test_miniature_training_smoke(tmp_path: Path) -> None:
    tokenizer_dir = make_tokenizer(tmp_path)
    train_glob, val_glob = write_shards(tmp_path)
    model_config = write_tiny_model_config(tmp_path)
    result = train(
        TrainConfig(
            run_dir=str(tmp_path / "run"),
            model_config=str(model_config),
            train_shards_glob=train_glob,
            val_shards_glob=val_glob,
            tokenizer_dir=str(tokenizer_dir),
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            learning_rate=1e-3,
            warmup_steps=1,
            mixed_precision="none",
            require_gpu=False,
        )
    )
    assert result["final_step"] == 2
    assert (tmp_path / "run" / "checkpoints" / "latest" / "_SUCCESS").exists()


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    config = PerkunasConfig(
        model_name="roundtrip",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
    )
    model = PerkunasForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ckpt = save_checkpoint(
        tmp_path / "checkpoints",
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=config,
        step=1,
        metadata={"test": True},
    )
    for parameter in model.parameters():
        parameter.data.add_(1.0)
    state = load_checkpoint(ckpt, model=model, optimizer=optimizer)
    assert state["step"] == 1
    for key, value in model.state_dict().items():
        assert torch.allclose(value, before[key])
