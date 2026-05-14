# Perkunas Training Pipeline

## Phase 1: Data Inspection

Local parquet command:

Command:

```powershell
python .\training\scripts\inspect_parquet.py --config .\training\configs\data.yaml
```

Outputs:

- `training/reports/parquet_profile.md`
- `training/reports/parquet_profile.json`

The inspector reads parquet metadata first and then streams record batches. It
does not load the full shard into memory. It identifies candidate text fields,
metadata fields, nulls, short rows, duplicate normalized text hashes, control
characters, and length distributions.

Observed local shard:

- rows: 69,907
- text field: `text`
- provenance fields: `identifier`, `collection`, `open_type`, `curator`,
  `license`, `date`, `title`, `creator`, `language`, `language_type`,
  `word_count`, `token_count`

Hugging Face Common Corpus sample inspection:

```powershell
python .\training\scripts\inspect_hf_dataset.py --config .\training\configs\data_hf_common_corpus.yaml --max-samples 1000
```

Use cached/non-streaming mode when the dataset is already available locally:

```powershell
python .\training\scripts\inspect_hf_dataset.py --config .\training\configs\data_hf_common_corpus.yaml --no-streaming --cache-dir D:\LLMProject\.hf_cache --max-samples 1000
```

The HF inspector uses `datasets.load_dataset(...)` and samples a bounded number
of records. It records the selected text field, feature/schema information when
available, nulls, top language/license/collection values, duplicate hashes in
the sample, and length distributions. It does not materialize the full dataset.

## Phase 2: Corpus Filtering and Normalization

Command:

```powershell
python .\training\scripts\normalize_corpus.py --config .\training\configs\data.yaml
```

Outputs:

- `training/data/prepared/normalized_*.jsonl`
- `training/data/prepared/manifest.json`
- `training/data/prepared/normalization_report.md`

Each JSONL record contains:

- `id`
- normalized `text`
- `text_sha256`
- `char_count`
- `word_count`
- `source_name`
- `source_type`
- `source_path_or_dataset`
- top-level `language`, `license`, `date`, `collection`, and `url` when present
- `source`
- `metadata`

Filtering is configurable in `data.yaml`: min/max chars, min words, language,
license, collection, and date bounds.

Normalize Hugging Face Common Corpus only:

```powershell
python .\training\scripts\normalize_corpus.py --config .\training\configs\data_hf_common_corpus.yaml
```

Normalize a blended Common Corpus HF + local C4 corpus:

```powershell
python .\training\scripts\normalize_corpus.py --config .\training\configs\data_hf_blend.yaml
```

Both configs keep chunking before dedup enabled. The HF source supports
`streaming: true` for remote streaming and `streaming: false` for local cached
dataset loading via Hugging Face Datasets.

## Phase 3: Deduplication

Command:

```powershell
python .\training\scripts\dedup_corpus.py --config .\training\configs\data.yaml
```

For the HF blend, use:

```powershell
python .\training\scripts\dedup_corpus.py --config .\training\configs\data_hf_blend.yaml
```

Outputs:

- `training/data/dedup/dedup_*.jsonl`
- `training/data/dedup/manifest.json`
- `training/data/dedup/dedup_report.md`

Exact dedup uses SHA-256 over canonicalized normalized text. Approximate dedup
uses 64-bit SimHash with LSH bands and a configurable Hamming threshold. This is
a real first-pass near-duplicate detector and can later be swapped for MinHash or
embedding-based dedup behind the same stage boundary.

## Phase 4: Tokenizer Training

Command:

```powershell
python .\training\scripts\train_tokenizer.py --config .\training\configs\tokenizer.yaml
```

For the HF blend:

```powershell
python .\training\scripts\train_tokenizer.py --config .\training\configs\tokenizer_hf_blend.yaml
```

Outputs:

- `training/tokenizer/perkunas-tokenizer/tokenizer.json`
- `training/tokenizer/perkunas-tokenizer/tokenizer_config.json`
- `training/tokenizer/perkunas-tokenizer/special_tokens_map.json`
- `training/tokenizer/perkunas-tokenizer/tokenizer_evaluation.json`
- `training/tokenizer/perkunas-tokenizer/README.md`

Perkunas uses a byte-level BPE tokenizer so multilingual and code-heavy shards do
not produce hard OOV failures.

## Phase 5: Tokenization and Packing

Command:

```powershell
python .\training\scripts\tokenize_corpus.py --config .\training\configs\data.yaml
```

For the HF blend:

```powershell
python .\training\scripts\tokenize_corpus.py --config .\training\configs\data_hf_blend.yaml
```

Outputs:

- `training/data/tokenized/train_*.npy`
- `training/data/tokenized/val_*.npy`
- `training/data/tokenized/manifest.json`
- `training/data/tokenized/tokenization_report.md`

The tokenization stage creates fixed-size blocks of `sequence_length + 1` token
IDs. The trainer uses all but the last token as inputs and all but the first as
labels.

## Phase 6: Model Architecture

Configs:

- `training/configs/model_small.yaml`
- `training/configs/model_medium.yaml`
- `training/configs/model_large.yaml`

The first architecture is a practical GPT-style decoder-only model:

- RoPE positional strategy;
- RMSNorm;
- SwiGLU MLP;
- tied token embeddings by default;
- configurable grouped-query attention;
- extension points for modality encoders, projection layers, and auxiliary
  heads without retraining the base model.

## Phase 7: Pretraining

Command:

```powershell
python .\training\scripts\train_perkunas.py --config .\training\configs\train.yaml
```

For the HF blend on the RTX 3050 profile:

```powershell
python .\training\scripts\train_perkunas.py --config .\training\configs\train_hf_blend_3050.yaml
```

Distributed:

```powershell
torchrun --standalone --nproc_per_node 2 .\training\scripts\train_perkunas.py --config .\training\configs\train.yaml
```

The trainer supports:

- random initialization from config;
- next-token prediction;
- gradient accumulation;
- mixed precision on CUDA;
- DDP under `torchrun`;
- checkpoint save/resume;
- JSONL logs for loss, throughput, and memory.

GPU is strict by default. `training/configs/train.yaml` sets:

```yaml
require_gpu: true
```

At startup the trainer prints:

- `torch.cuda.is_available`
- `torch.version.cuda`
- `torch.cuda.device_count`
- `torch.cuda.get_device_name(0)`
- selected device

If `require_gpu=true` and CUDA is unavailable, or if CUDA is available but the
model is not on CUDA, training raises immediately. CPU fallback is only for
intentional smoke tests with `require_gpu: false`.

Run the standalone CUDA smoke check:

```powershell
python .\training\scripts\gpu_smoke.py
```

Resume:

```powershell
python .\training\scripts\train_perkunas.py --config .\training\configs\train.yaml --resume-from .\training\runs\smoke\checkpoints\latest
```

To resume via config, set `resume_from` in `train.yaml`.

## Phase 8: Evaluation

Command:

```powershell
python .\training\scripts\eval_checkpoint.py --config .\training\configs\eval.yaml
```

Outputs:

- validation loss and perplexity;
- sample generations;
- long-context smoke check;
- tokenization behavior report.

## Phase 9: Export

Command:

```powershell
python .\training\scripts\export_hf.py --checkpoint .\training\runs\smoke\checkpoints\latest --output .\training\artifacts\perkunas-smoke
```

Outputs:

- `config.json`
- `model.safetensors`
- tokenizer files
- `generation_config.json`
- `README.md`
- `training_metadata.json`
- `serving_registration_template.json`

The export is prepared for later registration into the existing KV-optimized
serving platform.

## Scaling to Many Shards

Add parquet paths to `input_paths` in `data.yaml`. Every stage consumes globs or
manifests and produces deterministic shard names, so additional shards can be
processed without changing the model or tokenizer code.

For Hugging Face datasets, add more `hf_dataset` entries, set
`dataset_config`/`split` when targeting future Common Corpus subsets, and use
`max_records` only for bounded pilots. Keep local parquet entries in the same
`datasets` list to blend C4 or other local shards with Common Corpus.
