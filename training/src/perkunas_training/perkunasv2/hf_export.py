from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import save_file as save_safetensors_file

from perkunas_training.perkunasv2.configuration import PerkunasV2Config
from perkunas_training.perkunasv2.shard_store import (
    ParameterShardStore,
    load_payload_from_paths,
    shard_names,
)


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(slots=True)
class ExportStats:
    output_dir: Path
    files: list[str]
    tensor_count: int
    total_bytes: int
    dtype: str
    format: str


def export_perkunasv2_to_hf(
    *,
    run_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "fp16",
    max_shard_size: str = "2GB",
    overwrite: bool = False,
) -> ExportStats:
    run_dir = Path(run_dir)
    tokenizer_dir = Path(tokenizer_dir)
    output_dir = Path(output_dir)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}")
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(tokenizer_dir / "tokenizer.json")
    prepare_output_dir(output_dir, overwrite=overwrite)

    config = PerkunasV2Config.from_json(run_dir / "config.json")
    config.validate()
    store = ParameterShardStore(run_dir, config=config)
    export_dtype = DTYPES[dtype]
    max_bytes = parse_size_bytes(max_shard_size)

    writer = SafetensorShardWriter(output_dir, max_bytes=max_bytes)
    for name, tensor in iter_llama_tensors(store, config):
        writer.add(name, tensor.detach().cpu().to(dtype=export_dtype).contiguous())
    files, tensor_count, total_bytes = writer.finish()

    copy_tokenizer(tokenizer_dir, output_dir)
    write_json(output_dir / "config.json", llama_config(config, dtype=dtype))
    write_json(output_dir / "generation_config.json", generation_config(config))
    write_export_manifest(
        output_dir / "perkunas_export_manifest.json",
        run_dir=run_dir,
        tokenizer_dir=tokenizer_dir,
        config=config,
        files=files,
        tensor_count=tensor_count,
        total_bytes=total_bytes,
        dtype=dtype,
    )
    write_readme(output_dir / "README.md", config=config)
    return ExportStats(
        output_dir=output_dir,
        files=files,
        tensor_count=tensor_count,
        total_bytes=total_bytes,
        dtype=dtype,
        format="huggingface-llama-safetensors",
    )


def iter_llama_tensors(
    store: ParameterShardStore,
    config: PerkunasV2Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    yield "model.embed_tokens.weight", load_state_dict(store, "embeddings")["weight"]
    for index in range(config.num_layers):
        state = load_state_dict(store, f"block_{index:03d}")
        prefix = f"model.layers.{index}"
        yield f"{prefix}.input_layernorm.weight", state["input_layernorm.weight"]
        yield f"{prefix}.self_attn.q_proj.weight", state["self_attn.q_proj.weight"]
        yield f"{prefix}.self_attn.k_proj.weight", state["self_attn.k_proj.weight"]
        yield f"{prefix}.self_attn.v_proj.weight", state["self_attn.v_proj.weight"]
        yield f"{prefix}.self_attn.o_proj.weight", state["self_attn.o_proj.weight"]
        yield f"{prefix}.post_attention_layernorm.weight", state[
            "post_attention_layernorm.weight"
        ]
        gate, up = state["mlp.gate_up.weight"].chunk(2, dim=0)
        yield f"{prefix}.mlp.gate_proj.weight", gate
        yield f"{prefix}.mlp.up_proj.weight", up
        yield f"{prefix}.mlp.down_proj.weight", state["mlp.down.weight"]
    yield "model.norm.weight", load_state_dict(store, "final_norm")["weight"]
    yield "lm_head.weight", load_state_dict(store, "lm_head")["weight"]


def load_state_dict(store: ParameterShardStore, shard_name: str) -> dict[str, torch.Tensor]:
    paths = store._existing_param_paths(shard_name)  # package-local export helper
    payload = load_payload_from_paths(paths, kind="param", shard_name=shard_name)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"parameter shard {shard_name} did not contain a state_dict")
    return state


class SafetensorShardWriter:
    def __init__(self, output_dir: Path, *, max_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.files: list[str] = []
        self.tensor_count = 0
        self.total_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor_bytes = tensor_nbytes(tensor)
        if self.pending and self.pending_bytes + tensor_bytes > self.max_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += tensor_bytes
        self.tensor_count += 1
        self.total_bytes += tensor_bytes
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        file_name = f"model-{len(self.files) + 1:05d}-of-PLACEHOLDER.safetensors"
        save_safetensors_file(
            self.pending,
            self.output_dir / file_name,
            metadata={"format": "pt"},
        )
        for name in self.pending:
            self.weight_map[name] = file_name
        self.files.append(file_name)
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> tuple[list[str], int, int]:
        self.flush()
        if not self.files:
            raise ValueError("no tensors were exported")
        if len(self.files) == 1:
            original = self.output_dir / self.files[0]
            final_name = "model.safetensors"
            final = self.output_dir / final_name
            original.replace(final)
            self.files = [final_name]
            self.weight_map = {name: final_name for name in self.weight_map}
            return self.files, self.tensor_count, self.total_bytes

        final_files: list[str] = []
        total = len(self.files)
        rename_map: dict[str, str] = {}
        for index, old_name in enumerate(self.files, start=1):
            new_name = f"model-{index:05d}-of-{total:05d}.safetensors"
            (self.output_dir / old_name).replace(self.output_dir / new_name)
            rename_map[old_name] = new_name
            final_files.append(new_name)
        self.weight_map = {
            name: rename_map[file_name] for name, file_name in self.weight_map.items()
        }
        write_json(
            self.output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": dict(sorted(self.weight_map.items())),
            },
        )
        self.files = final_files
        return self.files, self.tensor_count, self.total_bytes


def llama_config(config: PerkunasV2Config, *, dtype: str) -> dict[str, Any]:
    head_dim = config.hidden_size // config.num_heads
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "num_key_value_heads": config.num_heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_position_embeddings,
        "initializer_range": config.initializer_range,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": config.tied_embeddings,
        "pretraining_tp": 1,
        "use_cache": True,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "torch_dtype": {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[dtype],
        "perkunas_source_model_type": "perkunasv2-shard-native",
    }


def generation_config(config: PerkunasV2Config) -> dict[str, Any]:
    return {
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
    }


def write_export_manifest(
    path: Path,
    *,
    run_dir: Path,
    tokenizer_dir: Path,
    config: PerkunasV2Config,
    files: list[str],
    tensor_count: int,
    total_bytes: int,
    dtype: str,
) -> None:
    write_json(
        path,
        {
            "format": "huggingface-llama-safetensors",
            "source_run_dir": str(run_dir),
            "source_tokenizer_dir": str(tokenizer_dir),
            "source_shards": shard_names(config),
            "dtype": dtype,
            "files": files,
            "tensor_count": tensor_count,
            "total_bytes": total_bytes,
            "parameter_count": sum(
                tensor_count_for_config(config).values()
            ),
            "notes": [
                "Perkunas fused mlp.gate_up.weight is exported as Llama gate_proj/up_proj.",
                "The exported directory is intended for HF/vLLM causal-LM serving.",
            ],
        },
    )


def tensor_count_for_config(config: PerkunasV2Config) -> dict[str, int]:
    return {
        "embeddings": config.vocab_size * config.hidden_size,
        "blocks": config.num_layers
        * (
            2 * config.hidden_size
            + 4 * config.hidden_size * config.hidden_size
            + 3 * config.hidden_size * config.intermediate_size
        ),
        "final_norm": config.hidden_size,
        "lm_head": config.vocab_size * config.hidden_size,
    }


def write_readme(path: Path, *, config: PerkunasV2Config) -> None:
    text = f"""---
library_name: transformers
pipeline_tag: text-generation
tags:
- text-generation
- vllm
- llama
- perkunas
---

# Perkunas v2.9 HF/vLLM Export

This directory is a post-training Hugging Face export of a Perkunas v2.9 causal language model.
It uses a Llama-compatible tensor layout for high-performance inference runtimes.

## Shape

- Layers: {config.num_layers}
- Hidden size: {config.hidden_size}
- Attention heads: {config.num_heads}
- Intermediate size: {config.intermediate_size}
- Context length: {config.max_position_embeddings}
- Vocabulary size: {config.vocab_size}

## vLLM

```bash
vllm serve . --dtype float16 --served-model-name perkunas-v2.9
```

For local smoke tests, use completion-style prompts. This checkpoint is a base TinyStories-style model, not an instruction-tuned chat model.
"""
    path.write_text(text, encoding="utf-8")


def copy_tokenizer(tokenizer_dir: Path, output_dir: Path) -> None:
    for item in tokenizer_dir.iterdir():
        if item.is_file():
            # Preserve file contents only. Metadata preservation can fail when exporting
            # between WSL/ext4 and Windows-mounted paths.
            shutil.copyfile(item, output_dir / item.name)
    normalize_tokenizer_for_generation(output_dir)


def normalize_tokenizer_for_generation(output_dir: Path) -> None:
    tokenizer_json_path = output_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        tokenizer_data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        post_processor = tokenizer_data.get("post_processor")
        if isinstance(post_processor, dict) and post_processor.get("type") == "TemplateProcessing":
            # The training tokenizer stores examples as <s> text </s>, but a generation
            # prompt must not end with EOS. Leaving EOS in the HF tokenizer makes vLLM
            # start a fresh story instead of continuing the user's prompt.
            post_processor["single"] = [
                item
                for item in post_processor.get("single", [])
                if item.get("SpecialToken", {}).get("id") != "</s>"
            ]
            post_processor["pair"] = [
                item
                for index, item in enumerate(post_processor.get("pair", []))
                if not (
                    item.get("SpecialToken", {}).get("id") == "</s>"
                    and index == len(post_processor.get("pair", [])) - 1
                )
            ]
            tokenizer_json_path.write_text(
                json.dumps(tokenizer_data, indent=2) + "\n",
                encoding="utf-8",
            )

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config: dict[str, Any] = {}
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["add_bos_token"] = True
    tokenizer_config["add_eos_token"] = False
    tokenizer_config.setdefault("clean_up_tokenization_spaces", False)
    tokenizer_config.setdefault(
        "chat_template",
        (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'].strip() }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "{{ message['content'].strip() }}\n\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'].strip() }}{{ eos_token }}\n\n"
            "{% endif %}"
            "{% endfor %}"
        ),
    )
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_size_bytes(value: str) -> int:
    text = value.strip().upper().replace(" ", "")
    units = {
        "B": 1,
        "KB": 1000,
        "KIB": 1024,
        "MB": 1000**2,
        "MIB": 1024**2,
        "GB": 1000**3,
        "GIB": 1024**3,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            number = float(text[: -len(suffix)])
            return max(1, math.ceil(number * units[suffix]))
    return max(1, int(text))


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Perkunasv2 streaming checkpoint to a HF/vLLM Llama-style package"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = export_perkunasv2_to_hf(
        run_dir=args.run_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(stats.output_dir),
                "files": stats.files,
                "tensor_count": stats.tensor_count,
                "total_bytes": stats.total_bytes,
                "dtype": stats.dtype,
                "format": stats.format,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
