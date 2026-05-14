# Perkunasv2 C4 Parquet Training

The implementation lives in `training/src/perkunas_training/perkunasv2/c4_training.py`
and the runnable wrapper is `scripts/train_perkunasv2_c4.py`.

Expected C4 layout:

```text
D:/LLMProject/TrainingData/allenai/c4/training/
  c4-train.00000-of-01024.parquet
  ...
D:/LLMProject/TrainingData/allenai/c4/validation/
  c4-validation.00000-of-00008.parquet
  ...
```

The script discovers existing parquet files, sorts by shard number, warns about
missing shard numbers, and continues with the files present. It streams parquet
rows with PyArrow, reads the `text` column, applies light filtering, tokenizes
with the project tokenizer, packs `seq_len + 1` chunks, and feeds batches into
the Perkunasv2 shard-native trainer.

## Offline Tokenization

For real runs, pre-tokenize C4 once and train from packed token shards. This
keeps tokenizer CPU work out of every training resume and gives exact token-shard
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

Each `.npy` shard has shape `[blocks, seq_len + 1]`; the trainer uses
`tokens[:-1]` as inputs and `tokens[1:]` as labels.

Smoke test:

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

RTX 3050-style run:

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

After offline tokenization, train from packed token shards with:

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

That command assumes `training/runs/perkunasv2/tokenizer/tokenizer.json` already
exists. Add `--tokenizer-path training/tokenizer/perkunas-pilot-tokenizer` for a
fresh run. The shorter config path resolves to `training/configs/perkunasv2_280m.json`
when needed.

Resume is automatic from `trainer_state.json`. C4 row-level replay is approximate:
the state records file index, batch-ish offset, epoch, and token-buffer
remainder, but exact parquet row replay is not guaranteed. This never corrupts
model or optimizer state because those are saved through Perkunasv2 shards.

Validation streams C4 validation parquet files and uses shard-native forward
only. It does not update parameters or instantiate a full model.

C4 is large. Train by token budget and checkpoint cadence. Logs include step,
tokens seen, train loss, validation loss/perplexity, learning rate, tokens/sec,
current parquet file, memory usage, and active shard update records.

Full operational notes are also in
`training/docs/perkunasv2_c4_training.md`.
