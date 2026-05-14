from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def project_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [dataclass_to_dict(v) for v in value]
    return value


@dataclass(slots=True)
class DatasetSourceConfig:
    name: str
    type: str = "parquet_local"
    weight: float = 1.0
    paths: list[str] = field(default_factory=list)
    dataset_name: str | None = None
    dataset_config: str | None = None
    split: str | None = None
    streaming: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    data_files: Any | None = None
    trust_remote_code: bool = False
    text_field: str | None = None
    metadata_fields: list[str] = field(default_factory=list)
    default_language: str | None = None
    max_records_per_path: int | None = None
    max_records: int | None = None


@dataclass(slots=True)
class ChunkingConfig:
    enabled: bool = True
    target_chars: int = 12000
    max_chars: int = 16000
    overlap_chars: int = 512


@dataclass(slots=True)
class DedupConfig:
    exact: bool = True
    approximate: bool = False
    hamming_threshold: int = 3
    max_simhash_tokens: int = 1024
    max_simhash_chars: int = 60000


@dataclass(slots=True)
class DataConfig:
    input_paths: list[str]
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    validation_datasets: list[DatasetSourceConfig] = field(default_factory=list)
    reports_dir: str = "training/reports"
    prepared_dir: str = "training/data/prepared"
    dedup_dir: str = "training/data/dedup"
    tokenized_dir: str = "training/data/tokenized"
    text_field: str | None = None
    metadata_fields: list[str] = field(default_factory=list)
    batch_size: int = 1024
    min_chars: int = 200
    max_chars: int = 500_000
    min_words: int = 20
    allowed_languages: list[str] | None = None
    language_allowlist: list[str] | None = None
    allowed_licenses: list[str] | None = None
    license_allowlist: list[str] | None = None
    license_blocklist: list[str] | None = None
    collection_allowlist: list[str] | None = None
    collection_blocklist: list[str] | None = None
    min_date: int | None = None
    max_date: int | None = None
    output_shard_rows: int = 5000
    resume: bool = True
    validation_fraction: float = 0.005
    sequence_length: int = 1024
    tokenization_batch_size: int = 256
    tokenizer_path: str = "training/tokenizer/perkunas-tokenizer"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        data = load_yaml(path)
        data["datasets"] = [DatasetSourceConfig(**item) for item in data.get("datasets", [])]
        data["validation_datasets"] = [
            DatasetSourceConfig(**item) for item in data.get("validation_datasets", [])
        ]
        if isinstance(data.get("chunking"), dict):
            data["chunking"] = ChunkingConfig(**data["chunking"])
        if isinstance(data.get("dedup"), dict):
            data["dedup"] = DedupConfig(**data["dedup"])
        data.setdefault("input_paths", [])
        return cls(**data)


@dataclass(slots=True)
class TokenizerConfig:
    input_glob: str = "training/data/dedup/*.jsonl"
    output_dir: str = "training/tokenizer/perkunas-tokenizer"
    vocab_size: int = 32000
    min_frequency: int = 2
    limit_files: int | None = None
    sample_size: int = 2000
    special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TokenizerConfig":
        data = load_yaml(path)
        return cls(**data)


@dataclass(slots=True)
class TrainConfig:
    run_dir: str = "training/runs/smoke"
    model_config: str = "training/configs/model_small.yaml"
    train_shards_glob: str = "training/data/tokenized/train_*.npy"
    val_shards_glob: str = "training/data/tokenized/val_*.npy"
    tokenizer_dir: str = "training/tokenizer/perkunas-tokenizer"
    seed: int = 1337
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 100
    log_interval: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    mixed_precision: str = "bf16"
    resume_from: str | None = None
    require_gpu: bool = True
    num_workers: int = 0
    deterministic: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = load_yaml(path)
        return cls(**data)


@dataclass(slots=True)
class EvalConfig:
    checkpoint: str
    tokenizer_dir: str
    val_shards_glob: str
    output_dir: str = "training/reports/eval"
    max_batches: int = 20
    batch_size: int = 2
    generation_prompts: list[str] = field(default_factory=list)
    max_new_tokens: int = 80

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        data = load_yaml(path)
        return cls(**data)
