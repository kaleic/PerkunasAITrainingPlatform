from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
from perkunas_training.train.checkpoint import load_checkpoint
from perkunas_training.utils.io import ensure_dir, read_json, write_json


def export_hf(
    checkpoint: str | Path,
    output: str | Path,
    *,
    tokenizer_dir: str | Path = "training/tokenizer/perkunas-tokenizer",
    data_manifest: str | Path | None = "training/data/tokenized/manifest.json",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    output = ensure_dir(output)
    config = PerkunasConfig.from_json(checkpoint / "config.json")
    model = PerkunasForCausalLM(config)
    load_checkpoint(checkpoint, model=model, map_location="cpu")
    model.save_pretrained(output)
    enrich_config_for_handoff(output / "config.json")
    copy_tokenizer_files(Path(tokenizer_dir), output)
    write_json(
        output / "generation_config.json",
        {
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
        },
    )
    provenance = build_provenance_summary(checkpoint, data_manifest)
    write_json(output / "training_metadata.json", provenance)
    (output / "README.md").write_text(render_model_card(config, provenance), encoding="utf-8")
    serving = serving_registration_template(config, output)
    write_json(output / "serving_registration_template.json", serving)
    copy_custom_code(output)
    return {
        "output": str(output),
        "checkpoint": str(checkpoint),
        "artifacts": sorted(path.name for path in output.iterdir()),
    }


def copy_tokenizer_files(tokenizer_dir: Path, output: Path) -> None:
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = tokenizer_dir / name
        if source.exists():
            shutil.copy2(source, output / name)


def enrich_config_for_handoff(path: Path) -> None:
    config = read_json(path)
    config["auto_map"] = {
        "AutoConfig": "configuration_perkunas.PerkunasConfig",
        "AutoModelForCausalLM": "modeling_perkunas.PerkunasForCausalLM",
    }
    config["serving_notes"] = {
        "kv_cache": "standard decoder-only causal attention; register with KV-optimized serving platform after vLLM architecture adapter is available",
        "from_scratch": True,
    }
    write_json(path, config)


def build_provenance_summary(checkpoint: Path, data_manifest: str | Path | None) -> dict[str, Any]:
    state = read_json(checkpoint / "trainer_state.json")
    provenance: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "trainer_state": state,
        "from_scratch": True,
        "base_model": None,
    }
    if data_manifest and Path(data_manifest).exists():
        provenance["tokenized_manifest"] = read_json(data_manifest)
    return provenance


def render_model_card(config: PerkunasConfig, provenance: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Perkunas",
            "",
            "Perkunas is a decoder-only language model trained from scratch.",
            "",
            "## Architecture",
            "",
            f"- Hidden size: `{config.hidden_size}`",
            f"- Layers: `{config.num_hidden_layers}`",
            f"- Attention heads: `{config.num_attention_heads}`",
            f"- KV heads: `{config.num_key_value_heads}`",
            f"- Context length: `{config.max_position_embeddings}`",
            f"- Activation: `{config.activation}`",
            f"- Positional strategy: `RoPE theta={config.rope_theta}`",
            "",
            "## Training",
            "",
            f"- From scratch: `{provenance['from_scratch']}`",
            f"- Base model: `{provenance['base_model']}`",
            "",
            "## Serving Handoff",
            "",
            "Use `serving_registration_template.json` as the starting point for registration in the",
            "KV-memory-optimized inference platform.",
            "",
        ]
    )


def serving_registration_template(config: PerkunasConfig, output: Path) -> dict[str, Any]:
    return {
        "model_id": output.name,
        "task_type": "generate",
        "backend": "vllm",
        "backend_config": {
            "model_name_or_path": str(output),
            "prequantized": False,
            "engine_args": {
                "max_model_len": config.max_position_embeddings,
                "gpu_memory_utilization": 0.9,
                "enable_prefix_caching": True,
            },
        },
        "quantization_mode": "auto",
        "kv_compression_mode": "fp8",
        "kv_required": True,
        "max_context": config.max_position_embeddings,
        "streaming_supported": True,
        "chat_template_required": False,
        "hardware_constraints": {
            "min_gpu_memory_gb": 24,
            "supports_fp8_kv": True,
            "requires_cuda": True,
        },
        "policy_mode": "balanced",
    }


def copy_custom_code(output: Path) -> None:
    source_root = Path(__file__).parents[1] / "model"
    shutil.copy2(source_root / "configuration.py", output / "configuration_perkunas.py")
    shutil.copy2(source_root / "modeling_perkunas.py", output / "modeling_perkunas.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Perkunas checkpoint to HF-style artifacts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-dir", default="training/tokenizer/perkunas-tokenizer")
    args = parser.parse_args()
    result = export_hf(args.checkpoint, args.output, tokenizer_dir=args.tokenizer_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
