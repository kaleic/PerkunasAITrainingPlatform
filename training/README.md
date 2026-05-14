# Perkunas Training System

This directory contains the from-scratch pretraining pipeline for **Perkunas**.
It is separate from the inference platform in the repository root.

Perkunas is trained from data, not fine-tuned from another model:

- streaming parquet inspection;
- corpus filtering and normalization;
- exact and SimHash-based approximate deduplication;
- Perkunas-specific tokenizer training;
- packed training shard creation;
- GPT-style decoder-only pretraining from random initialization;
- checkpoint save/resume;
- evaluation;
- Hugging Face-compatible export.

## Quick Start

From `D:\LLMProject`:

```powershell
pip install -e .\training

python .\training\scripts\inspect_parquet.py --config .\training\configs\data.yaml
python .\training\scripts\normalize_corpus.py --config .\training\configs\data.yaml
python .\training\scripts\dedup_corpus.py --config .\training\configs\data.yaml
python .\training\scripts\train_tokenizer.py --config .\training\configs\tokenizer.yaml
python .\training\scripts\tokenize_corpus.py --config .\training\configs\data.yaml
python .\training\scripts\train_perkunas.py --config .\training\configs\train.yaml
python .\training\scripts\eval_checkpoint.py --config .\training\configs\eval.yaml
python .\training\scripts\export_hf.py --checkpoint .\training\runs\smoke\checkpoints\latest --output .\training\artifacts\perkunas-smoke
```

Training requires CUDA by default (`require_gpu: true`). To verify CUDA placement
before a real run:

```powershell
python .\training\scripts\gpu_smoke.py
```

For distributed training, use PyTorch launch:

```powershell
torchrun --standalone --nproc_per_node 2 .\training\scripts\train_perkunas.py --config .\training\configs\train.yaml
```

## Documentation

- [Pipeline guide](docs/pipeline.md)
- [Implementation plan](docs/implementation-plan.md)
- [Risk analysis](docs/risk-analysis.md)
- [Roadmap](docs/roadmap.md)

## Observed Local Shard Schema

`D:\LLMProject\0000.parquet` has:

- text: `text`
- provenance: `identifier`, `collection`, `open_type`, `curator`, `license`,
  `date`, `title`, `creator`, `language`, `language_type`, `word_count`,
  `token_count`

The pipeline does not hard-code this schema blindly. Inspection identifies text
and metadata candidates for every shard set, then configs can override.
