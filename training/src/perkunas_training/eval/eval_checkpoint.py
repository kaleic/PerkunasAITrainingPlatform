from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from perkunas_training.config import EvalConfig
from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
from perkunas_training.train.checkpoint import load_checkpoint
from perkunas_training.train.dataset import PackedTokenDataset
from perkunas_training.train.train_perkunas import evaluate_loss
from perkunas_training.utils.io import ensure_dir, write_json


def evaluate_checkpoint(config: EvalConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    checkpoint = Path(config.checkpoint)
    model_config = PerkunasConfig.from_json(checkpoint / "config.json")
    model = PerkunasForCausalLM(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state = load_checkpoint(checkpoint, model=model, map_location=device)
    dataset = PackedTokenDataset(config.val_shards_glob)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    loss_metrics = evaluate_loss(model, loader, device, "fp16", max_batches=config.max_batches)
    tokenizer = Tokenizer.from_file(str(Path(config.tokenizer_dir) / "tokenizer.json"))
    generations = generate_samples(model, tokenizer, config.generation_prompts, config.max_new_tokens, device)
    long_context = long_context_smoke(model, dataset, device)
    tokenization = tokenization_behavior(tokenizer, config.generation_prompts)
    report = {
        "checkpoint": str(checkpoint),
        "trainer_state": state,
        "loss": loss_metrics,
        "generations": generations,
        "long_context_smoke": long_context,
        "tokenization_behavior": tokenization,
    }
    write_json(output_dir / "eval_report.json", report)
    (output_dir / "eval_report.md").write_text(render_report(report), encoding="utf-8")
    return report


@torch.no_grad()
def generate_samples(
    model: PerkunasForCausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
        text = tokenizer.decode(output[0].tolist())
        results.append({"prompt": prompt, "completion": text})
    return results


@torch.no_grad()
def long_context_smoke(
    model: PerkunasForCausalLM, dataset: PackedTokenDataset, device: torch.device
) -> dict[str, Any]:
    input_ids, labels = dataset[0]
    input_ids = input_ids[: model.config.max_position_embeddings].unsqueeze(0).to(device)
    labels = labels[: model.config.max_position_embeddings].unsqueeze(0).to(device)
    output = model(input_ids=input_ids, labels=labels)
    return {
        "sequence_length": int(input_ids.shape[1]),
        "loss": float(output["loss"].detach().cpu()),
        "passed": True,
    }


def tokenization_behavior(tokenizer: Tokenizer, prompts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        encoding = tokenizer.encode(prompt)
        rows.append(
            {
                "prompt": prompt,
                "token_count": len(encoding.ids),
                "tokens": encoding.tokens,
            }
        )
    return rows


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Perkunas Checkpoint Evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## Loss",
        "",
        f"- Validation loss: `{report['loss']['val_loss']:.6f}`",
        f"- Validation perplexity: `{report['loss']['val_perplexity']:.6f}`",
        "",
        "## Long Context Smoke",
        "",
        f"- Sequence length: `{report['long_context_smoke']['sequence_length']}`",
        f"- Loss: `{report['long_context_smoke']['loss']:.6f}`",
        "",
        "## Generations",
        "",
    ]
    for row in report["generations"]:
        lines.extend([f"### {row['prompt']}", "", row["completion"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Perkunas checkpoint")
    parser.add_argument("--config", default="training/configs/eval.yaml")
    args = parser.parse_args()
    report = evaluate_checkpoint(EvalConfig.from_yaml(args.config))
    print(json.dumps(report["loss"], indent=2))


if __name__ == "__main__":
    main()
