from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import Any, Iterator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from tokenizers.pre_tokenizers import ByteLevel

from perkunas_training.config import TokenizerConfig
from perkunas_training.utils.io import ensure_dir, iter_jsonl, write_json
from perkunas_training.utils.text import word_count


def train_perkunas_tokenizer(config: TokenizerConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    files = sorted(glob(config.input_glob))
    if config.limit_files is not None:
        files = files[: config.limit_files]
    if not files:
        raise FileNotFoundError(f"no corpus files matched {config.input_glob}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", fuse_unk=False))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", config.special_tokens.index("<s>")),
            ("</s>", config.special_tokens.index("</s>")),
        ],
    )

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_text(files), trainer=trainer, length=count_records(files))
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 1024,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "clean_up_tokenization_spaces": False,
    }
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    write_json(output_dir / "tokenizer_config.json", tokenizer_config)
    write_json(output_dir / "special_tokens_map.json", special_tokens_map)
    evaluation = evaluate_tokenizer(tokenizer, files, sample_size=config.sample_size)
    write_json(output_dir / "tokenizer_evaluation.json", evaluation)
    (output_dir / "README.md").write_text(render_readme(config, evaluation), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "tokenizer_json": str(tokenizer_path),
        "files": files,
        "evaluation": evaluation,
    }


def iter_text(files: list[str]) -> Iterator[str]:
    for path in files:
        for record in iter_jsonl(path):
            yield record["text"]


def count_records(files: list[str]) -> int:
    total = 0
    for path in files:
        with Path(path).open("r", encoding="utf-8") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def evaluate_tokenizer(tokenizer: Tokenizer, files: list[str], sample_size: int = 2000) -> dict[str, Any]:
    sample_count = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = 0
    single_char_tokens = 0
    examples: list[dict[str, Any]] = []

    for text in iter_text(files):
        if sample_count >= sample_size:
            break
        encoding = tokenizer.encode(text)
        ids = encoding.ids
        tokens = encoding.tokens
        sample_count += 1
        total_chars += len(text)
        total_words += max(1, word_count(text))
        total_tokens += len(ids)
        if unk_id is not None:
            unk_count += sum(token_id == unk_id for token_id in ids)
        single_char_tokens += sum(len(token.replace("Ġ", "")) <= 1 for token in tokens)
        if len(examples) < 20:
            examples.append(
                {
                    "text_preview": text[:200],
                    "tokens": tokens[:80],
                    "token_count": len(ids),
                    "chars_per_token": len(text) / max(1, len(ids)),
                }
            )

    return {
        "sample_count": sample_count,
        "vocab_size": tokenizer.get_vocab_size(),
        "average_chars_per_token": total_chars / max(1, total_tokens),
        "fertility_tokens_per_word": total_tokens / max(1, total_words),
        "unk_token_rate": unk_count / max(1, total_tokens),
        "single_char_token_rate": single_char_tokens / max(1, total_tokens),
        "total_sample_chars": total_chars,
        "total_sample_tokens": total_tokens,
        "segmentation_examples": examples,
    }


def render_readme(config: TokenizerConfig, evaluation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Perkunas Tokenizer",
            "",
            "Tokenizer type: byte-level BPE.",
            "",
            "Rationale:",
            "",
            "- Byte-level coverage avoids hard OOV failures for multilingual text and code.",
            "- BPE keeps the pipeline compatible with Hugging Face fast tokenizer artifacts.",
            "- Special tokens are fixed at the start of the vocabulary for stable training.",
            "",
            "## Configuration",
            "",
            f"- Vocab size target: `{config.vocab_size}`",
            f"- Min frequency: `{config.min_frequency}`",
            f"- Special tokens: `{', '.join(config.special_tokens)}`",
            "",
            "## Evaluation",
            "",
            f"- Actual vocab size: `{evaluation['vocab_size']}`",
            f"- Average chars/token: `{evaluation['average_chars_per_token']:.4f}`",
            f"- Fertility tokens/word: `{evaluation['fertility_tokens_per_word']:.4f}`",
            f"- Unknown token rate: `{evaluation['unk_token_rate']:.8f}`",
            f"- Single-char token rate: `{evaluation['single_char_token_rate']:.4f}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Perkunas tokenizer")
    parser.add_argument("--config", default="training/configs/tokenizer.yaml")
    args = parser.parse_args()
    result = train_perkunas_tokenizer(TokenizerConfig.from_yaml(args.config))
    print(f"Tokenizer: {result['tokenizer_json']}")
    print(f"Average chars/token: {result['evaluation']['average_chars_per_token']:.4f}")


if __name__ == "__main__":
    main()
