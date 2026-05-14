from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from perkunas_training.config import load_yaml


@dataclass(slots=True)
class PerkunasConfig:
    model_name: str = "perkunas"
    vocab_size: int = 32000
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    activation: str = "swiglu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    modality_extension: dict[str, Any] = field(default_factory=dict)
    auxiliary_heads: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PerkunasConfig":
        data = load_yaml(path)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "PerkunasConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "model_type": "perkunas",
                "architectures": ["PerkunasForCausalLM"],
                "transformers_version": "custom",
            }
        )
        return data

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.activation != "swiglu":
            raise ValueError("only swiglu activation is implemented")
