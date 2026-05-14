# Perkunas

Perkunas is a memory-aware language-model training and serving project.

The core idea is simple: make meaningful model training possible when the active
memory budget is the hard limit. Perkunas v2 uses a streaming training runtime
that keeps only the needed model pieces active, records compact forward
boundaries, replays local work during the backward/update phase, and writes
durable checkpoints with telemetry.

This repository contains two connected systems:

- **Perkunas Training**: data preparation, tokenizer training, packed shard
  creation, shard-native streaming pretraining, validation, telemetry, checkpoint
  recovery, and Hugging Face/vLLM export.
- **kvserve**: an OpenAI-compatible inference control plane focused on model
  registration, KV-cache policy, prefix reuse, compression, pruning, paging, and
  observability.

The current public milestone is a 100M parameter TinyStories run trained with the
Perkunas streaming runtime under an 8GB VRAM limit. The run moved held-out
validation loss from `6.8512` to `3.5135`, with validation perplexity falling to
`33.57`.

> Perkunas is a training systems project first. Frozen checkpoints can be exported
> into standard inference stacks when low-latency serving matters.

## Status

This is an active research and engineering repository.

Working today:

- packed `.npy` train/validation dataset creation from parquet text corpora;
- Perkunas v2 shard-native training from random initialization;
- active/durable run directories for fast local work plus durable persistence;
- AdamW, Lion, and Adafactor optimizer paths;
- global and shard-local gradient clipping modes;
- guarded step replay for safer staged updates;
- CPU/GPU/secondary-GPU prefetch and trace staging options;
- JSONL training telemetry plus a self-contained HTML dashboard generator;
- validation during training and standalone validation;
- Hugging Face/vLLM-style export to a Llama-compatible package;
- OpenAI-style local serving endpoint for Perkunas v2 checkpoints;
- root `kvserve` API and tests for inference control-plane primitives.

Still evolving:

- training recipes and convergence behavior;
- throughput optimization;
- public benchmark harnesses;
- larger model-scale validation;
- production hardening around export and serving.

## Repository Layout

```text
.
+-- docs/                         # Architecture notes, public writeups, visual HTML decks
+-- scripts/                      # Convenience scripts and prompt tests
+-- src/kvserve/                  # OpenAI-compatible inference/control-plane package
+-- tests/                        # kvserve tests
+-- training/
|   +-- configs/                  # Model/data/tokenizer/training configs
|   +-- docs/                     # Training pipeline documentation
|   +-- scripts/                  # Perkunas training, tokenization, export, serving CLIs
|   +-- src/perkunas_training/    # Perkunas training package
|   +-- tests/                    # Training pipeline tests
+-- README.md
```

Large generated files are intentionally not part of the source distribution:

- raw datasets;
- tokenized packed shards;
- active/durable training runs;
- model exports;
- telemetry dashboards;
- local server logs;
- virtual environments.

See [Publishing Checklist](#publishing-checklist) before pushing.

## Requirements

Recommended:

- Python `3.11+`
- CUDA-capable PyTorch for training on GPU
- NVIDIA GPU for Perkunas v2 CUDA training
- PowerShell on Windows or Bash on Linux/WSL
- Optional: vLLM for high-throughput serving of exported checkpoints

The project has two Python packages:

- root package: `kvserve`
- training package: `perkunas-training`

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install -e ".\training[dev]"
```

On Linux/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install -e "./training[dev]"
```

For GPU serving extras:

```powershell
python -m pip install -e ".[gpu]"
```

For vLLM, prefer a dedicated Linux/WSL environment:

```bash
python3 -m venv ~/venvs/perkunas-vllm
source ~/venvs/perkunas-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install vllm
```

## Quick Start: Training Pipeline

### 1. Inspect and Prepare Data

Perkunas v2 training expects packed token shards. For TinyStories-style parquet
data:

```powershell
python training/scripts/tokenize_perkunasv2_c4.py `
  --train-data-dir TrainingData/roneneldan/TinyStories/training `
  --val-data-dir TrainingData/roneneldan/TinyStories/validation `
  --tokenizer-path training/tokenizer/perkunas-tinystories-32k-tokenizer `
  --output-dir training/data/perkunasv2_tinystories_tokenized_512 `
  --text-column text `
  --seq-len 512 `
  --blocks-per-shard 4096 `
  --parquet-batch-rows 1024 `
  --tokenization-batch-size 256 `
  --min-text-chars 0 `
  --enable-basic-filter false
```

The tokenizer path must contain a `tokenizer.json`.

### 2. Initialize Shards

Initialize a Perkunas v2 run from a model config:

```powershell
python training/scripts/train_perkunasv2.py --init-shards `
  --config training/configs/perkunasv2_9_5m_tinystories_32k.json `
  --run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --shard-storage-format torch `
  --init-weight-dtype fp32
```

Use a config that exists in `training/configs/`, or add your own JSON config.
The public 100M TinyStories milestone used the same runtime path with a larger
configuration saved in that run's `config.json`.

### 3. Train

Example low-memory streaming training command:

```powershell
python training/scripts/train_perkunasv2.py --train `
  --run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --active-run-dir training/active/perkunasv2_9_5m_tinystories_32k_smoke `
  --durable-flush-every 1000 `
  --data-dir training/data/perkunasv2_tinystories_tokenized_512 `
  --val-data-dir training/data/perkunasv2_tinystories_tokenized_512 `
  --seq-len 512 `
  --micro-batch-size 8 `
  --gradient-accumulation-steps 2 `
  --dtype fp16 `
  --master-weight-dtype fp32 `
  --shard-storage-format torch `
  --device cuda `
  --optimizer adamw `
  --learning-rate 1.0e-6 `
  --weight-decay 0.02 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 0.15 `
  --grad-clip-mode global `
  --lr-schedule tokens `
  --warmup-tokens 13107200 `
  --decay-tokens 3000000000 `
  --min-lr-ratio 0.40 `
  --max-steps 20000 `
  --save-every 100 `
  --validate-every 100 `
  --max-validation-batches 100 `
  --shuffle-train `
  --max-resident-shards 12 `
  --prefetch-shards cpu `
  --prefetch-window 12 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 100 `
  --lm-head-chunk-tokens 4096 `
  --async-shard-writes `
  --max-pending-shard-writes 12 `
  --guarded-step-replay `
  --guard-replay-max-replays 12 `
  --guard-replay-loss-tolerance 0.004 `
  --guard-replay-loss-tolerance-ratio 0.0005 `
  --guard-replay-lr-scales 1.0,0.85,0.7,0.5,0.35,0.25,0.1 `
  --guard-replay-grad-norm-scales 1.0,0.85,0.7,0.5,0.35,0.25,0.1 `
  --guard-replay-on-exhaust accept
```

Notes:

- `--run-dir` is the durable run archive.
- `--active-run-dir` is an optional fast working copy used during training.
- `--durable-flush-every` publishes the active run back to the durable run.
- `--max-resident-shards` and `--prefetch-window` control active residency.
- `--trace-storage cpu` is the low-memory default.
- `--trace-storage gpu` can reduce CPU transfer overhead if there is enough VRAM.
- `--trace-storage secondary-gpu --trace-storage-device cuda:1` stages traces on
  a second CUDA device.

### 4. Validate

```powershell
python training/scripts/train_perkunasv2.py --validate `
  --run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --active-run-dir training/active/perkunasv2_9_5m_tinystories_32k_smoke `
  --val-data-dir training/data/perkunasv2_tinystories_tokenized_512 `
  --seq-len 512 `
  --micro-batch-size 8 `
  --dtype fp16 `
  --device cuda `
  --max-validation-batches 100
```

## Telemetry Dashboard

Training writes `train_log.jsonl` and `trainer_state.json` into the run
directory. Generate a self-contained HTML dashboard:

```powershell
python training/scripts/build_train_telemetry_dashboard.py `
  -input training/active/perkunasv2_9_5m_tinystories_32k_smoke/train_log.jsonl `
  -output perkunas_train_telemetry.html `
  --title "Perkunas v2.9 TinyStories Telemetry"
```

The dashboard visualizes:

- train and validation loss;
- perplexity;
- learning rate and accepted guard scales;
- gradient norm and clip scale;
- throughput and step timing;
- shard residency and prefetch behavior;
- memory and timing breakdowns.

## Export to Hugging Face / vLLM Format

The streaming checkpoint can be packaged into a standard inference artifact:

```powershell
python training/scripts/export_perkunasv2_hf.py `
  --run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --tokenizer-dir training/tokenizer/perkunas-tinystories-32k-tokenizer `
  --output-dir exports/perkunasv2_9_5m_tinystories_32k_smoke_hf `
  --dtype fp16 `
  --overwrite
```

The exporter writes a Llama-style package with:

- `config.json`
- `generation_config.json`
- tokenizer files
- `model.safetensors`
- `perkunas_export_manifest.json`

## Serve a Perkunas v2 Checkpoint Locally

The native development server exposes OpenAI-style routes:

```powershell
python training/scripts/serve_perkunasv2.py `
  --primary-run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --backup-run-dir training/runs/perkunasv2_9_5m_tinystories_32k_smoke `
  --primary-tokenizer-dir training/tokenizer/perkunas-tinystories-32k-tokenizer `
  --backup-tokenizer-dir training/tokenizer/perkunas-tinystories-32k-tokenizer `
  --device cuda `
  --dtype fp16 `
  --max-resident-shards 12 `
  --preload-modules `
  --host 127.0.0.1 `
  --port 8010
```

Query it:

```powershell
$body = @{
  model = "primary"
  messages = @(
    @{ role = "system"; content = "You write simple stories." }
    @{ role = "user"; content = "Write a short story about a dog who lost a red ball." }
  )
  max_tokens = 120
  temperature = 0.8
  top_p = 0.95
  top_k = 50
  stream = $false
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod http://127.0.0.1:8010/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

$response.choices[0].message.content
```

For faster production-style serving, export the model and serve it with vLLM:

```bash
source ~/venvs/perkunas-vllm/bin/activate

vllm serve ~/models/perkunas-v2.9 \
  --served-model-name perkunas-v2.9 \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8011 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 4
```

## kvserve API

The root package provides an OpenAI-compatible control plane for registered
models and KV-memory policy work:

```powershell
$env:KV_SERVE_ENV = "dev"
$env:KV_API_TOKENS = "dev:dev-token"
uvicorn kvserve.app:create_app --factory --host 0.0.0.0 --port 8000
```

Try it:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models `
  -Headers @{ Authorization = "Bearer dev-token" }
```

Model registration lives in:

```text
config/model_registry.json
```

## Documentation

Start here:

- [Perkunas streaming public note](docs/perkunas_streaming_public_hf_paper.md)
- [Perkunas v2 training LLD](docs/perkunasv2_training_lld.md)
- [Perkunas v2 shard-native whitepaper](docs/perkunasv2_shard_native_whitepaper.md)
- [Training flow design HTML](docs/perkunas_training_flow_design.html)
- [Training runtime graphics HTML](docs/perkunas_training_runtime_graphics.html)
- [kvserve architecture](docs/architecture.md)
- [KV control plane](docs/kv-control-plane.md)
- [Training package README](training/README.md)

## Testing

Root package:

```powershell
pytest tests
```

Training package:

```powershell
pytest training/tests
```

Lint:

```powershell
ruff check src tests training/src training/tests
```

## Publishing Checklist

Before pushing to GitHub, make sure generated assets and private/local files are
not staged.

Common paths to keep out of source control:

```text
TrainingData/
training/data/
training/active/
training/runs/
training/artifacts/
exports/
reports/
*.log
*.out.log
*.err.log
*.parquet
*.npy
*.safetensors
*.pt
.venv/
.venv-vllm/
```

Recommended pre-push check:

```powershell
git status --short
git ls-files | Select-String -Pattern 'TrainingData/|training/data/|training/active/|training/runs/|exports/|reports/|\\.safetensors$|\\.pt$|\\.npy$|\\.parquet$|\\.log$'
```

If large files are already tracked, remove them from the Git index without
deleting the local files:

```powershell
git rm --cached <path>
```

## Project Philosophy

Perkunas is built around a practical split:

- **Train** with a streaming runtime that is designed around memory pressure.
- **Measure** every step with enough telemetry to understand stability and cost.
- **Package** frozen checkpoints into standard formats.
- **Serve** with the best available inference stack for the target machine.

This keeps the training runtime focused on making learning possible, while
letting deployment use mature inference infrastructure when speed is the main
goal.

## Citation

If referencing the public TinyStories systems milestone:

```text
Perkunas Streaming Training Runtime, TinyStories 100M Parameter 8GB GPU Experiment, 2026.
```

## License

Licensed under the Apache License, Version 2.0. 
