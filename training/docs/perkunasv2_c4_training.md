# Perkunasv2 C4 Parquet Training

This path streams AllenAI/C4 parquet files directly into the Perkunasv2
shard-native active-parameter trainer. It does not pre-load C4, instantiate a
full Perkunasv2 model, or create a global optimizer.

## Expected Layout

Training shards:

```text
D:/LLMProject/TrainingData/allenai/c4/training/
  c4-train.00000-of-01024.parquet
  ...
  c4-train.01023-of-01024.parquet
```

Validation shards:

```text
D:/LLMProject/TrainingData/allenai/c4/validation/
  c4-validation.00000-of-00008.parquet
  ...
  c4-validation.00007-of-00008.parquet
```

The script discovers existing `*.parquet` files and sorts them by shard number.
It warns about numbering gaps but continues with available files. This handles
the current case where validation may have 7 of 8 possible shards.

Each parquet file must contain a `text` column unless `--text-column` is set.

## Streaming Pipeline

`C4ParquetTokenStream` reads one parquet file at a time through PyArrow batch
iteration. It skips null or empty text, normalizes line endings, optionally
collapses excessive whitespace, applies light C4 filtering, tokenizes text with
the project tokenizer, and maintains a rolling token buffer.

It emits fixed `seq_len + 1` token chunks:

- `input_ids = tokens[:-1]`
- `labels = tokens[1:]`

The trainer consumes those batches through the Perkunasv2 shard-native forward
and reverse recompute/update path.

## Offline Tokenization

For real C4 runs, pre-tokenize once and train from packed token shards. This
avoids repeated tokenizer CPU cost on every restart and gives exact packed-shard
resume behavior.

```powershell
python scripts/tokenize_perkunasv2_c4.py `
  --train-data-dir D:/LLMProject/TrainingData/allenai/c4/training `
  --val-data-dir D:/LLMProject/TrainingData/allenai/c4/validation `
  --tokenizer-path training/tokenizer/perkunas-pilot-tokenizer `
  --output-dir training/data/perkunasv2_c4_tokenized `
  --seq-len 512 `
  --blocks-per-shard 4096 `
  --parquet-batch-rows 1024 `
  --tokenization-batch-size 256
```

Outputs:

```text
training/data/perkunasv2_c4_tokenized/
  train_00000.npy
  train_00001.npy
  ...
  val_00000.npy
  manifest.json
  tokenization_report.md
  train_progress.json
  val_progress.json
```

Each `.npy` shard has shape `[blocks, seq_len + 1]`.

## Tokenizer

The script loads `run_dir/tokenizer/tokenizer.json` when present. Otherwise pass
`--tokenizer-path` pointing to either a tokenizer directory or a `tokenizer.json`
file. It fails clearly if no tokenizer exists. It never trains or creates a new
tokenizer during C4 training.

If `--tokenizer-path` is provided, tokenizer files are copied into
`run_dir/tokenizer` for future resumes.

## Shard Initialization And Resume

If `run_dir/shards/metadata.json` is missing, the script initializes random
Perkunasv2 parameter shards from `--config`, writes `config.json`, creates the
shard layout, and creates `trainer_state.json`.

If shards already exist, the script resumes from `trainer_state.json`. Model and
optimizer state remain sharded. Resume stores the latest C4 file index, row-ish
batch position, epoch, and token-buffer remainder. Row-level replay is safe but
approximate: the script resumes near the latest file position and never rewrites
or corrupts model state.

## Validation

Validation uses the same C4 parquet streaming path and the same Perkunasv2
shard-streaming forward path. It loads no full model and updates no parameters.
Use `--val-batches` to control validation cost.

## Example Smoke Test

```powershell
python scripts/train_perkunasv2_c4.py `
  --run-dir training/runs/perkunasv2_c4_smoke `
  --train-data-dir D:/LLMProject/TrainingData/allenai/c4/training `
  --val-data-dir D:/LLMProject/TrainingData/allenai/c4/validation `
  --config training/configs/perkunasv2_tiny.json `
  --tokenizer-path training/tokenizer/perkunas-pilot-tokenizer `
  --seq-len 32 `
  --micro-batch-size 1 `
  --gradient-accumulation-steps 1 `
  --dtype fp32 `
  --device cpu `
  --smoke-test
```

For GPU smoke, use `--dtype fp16 --device cuda` and a config whose
`max_position_embeddings` is at least the chosen `--seq-len`.

## RTX 3050-Style Training Command

This exact command assumes `training/runs/perkunasv2/tokenizer/tokenizer.json`
already exists. If it does not, add `--tokenizer-path
training/tokenizer/perkunas-pilot-tokenizer`.

```powershell
python scripts/train_perkunasv2_c4.py `
  --run-dir training/runs/perkunasv2 `
  --train-data-dir D:/LLMProject/TrainingData/allenai/c4/training `
  --val-data-dir D:/LLMProject/TrainingData/allenai/c4/validation `
  --config configs/perkunasv2_280m.json `
  --seq-len 512 `
  --micro-batch-size 1 `
  --gradient-accumulation-steps 32 `
  --dtype fp16 `
  --device cuda `
  --learning-rate 1e-4 `
  --weight-decay 0.1 `
  --warmup-steps 2000 `
  --save-every 1000 `
  --validate-every 1000 `
  --val-batches 64
```

After offline tokenization, train from packed token shards:

```powershell
python training/scripts/train_perkunasv2.py `
  --train `
  --run-dir training/runs/perkunasv2 `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --seq-len 512 `
  --micro-batch-size 1 `
  --gradient-accumulation-steps 32 `
  --dtype fp16 `
  --device cuda `
  --learning-rate 1e-4 `
  --weight-decay 0.1 `
  --warmup-steps 2000 `
  --save-every 1000 `
  --validate-every 1000 `
  --max-validation-batches 64
```

The script resolves `configs/perkunasv2_280m.json` to
`training/configs/perkunasv2_280m.json` when the shorter path does not exist.

## Memory Notes

C4 is large. Always train by token budget and checkpoint cadence, not by "one
full epoch" expectations. With Perkunasv2 shard-native training, resident
trainable state is bounded by the active shard, active optimizer shard, boundary
activations, token buffers, and metadata.

Logs include:

- step;
- tokens seen;
- train loss;
- validation loss and perplexity when available;
- learning rate;
- tokens/sec;
- current parquet file;
- CUDA allocated/reserved/peak memory;
- active shard updates from the shard trainer.

## Filtering

Default filtering is intentionally light:

- skip text shorter than `--min-text-chars` default `50`;
- skip text with too little alphabetic content;
- skip mostly numeric or punctuation-heavy garbage.

Disable this with:

```powershell
--enable-basic-filter false
```

Do not use aggressive filtering in this C4 path until distribution-level corpus
metrics are measured.
