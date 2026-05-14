param(
    [string]$RunDir = "training/runs/Perkunas_v2.7_2.5b_96s",
    [string]$ActiveRunDir = "E:\Perkunas_v2.7_2.5b_96s_active",
    [int]$MaxSteps = 20,
    [int]$MicroBatchSize = 1,
    [int]$GradientAccumulationSteps = 16,
    [int]$MinActiveFreeGB = 14
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

$activeRoot = [System.IO.Path]::GetPathRoot($ActiveRunDir)
$driveName = $activeRoot.TrimEnd("\").TrimEnd(":")
$drive = Get-PSDrive -Name $driveName -ErrorAction Stop
$freeGB = [math]::Round($drive.Free / 1GB, 2)
if ($freeGB -lt $MinActiveFreeGB) {
    throw "Active run drive $activeRoot has $freeGB GB free; need at least $MinActiveFreeGB GB. Recreate the RAM disk at 16 GB or point -ActiveRunDir at a larger fast drive."
}

$configPath = "training/configs/perkunasv2_7_2_5b_96s.json"
$runConfigPath = Join-Path $RunDir "config.json"

if (-not (Test-Path $runConfigPath)) {
    python training/scripts/train_perkunasv2.py --init-shards `
      --config $configPath `
      --run-dir $RunDir `
      --shard-storage-format safetensors `
      --storage-shard-count 96 `
      --init-weight-dtype fp16 `
      --seed 2700
}

python training/scripts/train_perkunasv2.py --train `
  --run-dir $RunDir `
  --active-run-dir $ActiveRunDir `
  --durable-flush-every 10 `
  --data-dir training/data/perkunasv2_c4_tokenized `
  --val-data-dir D:\LLMProject\training\data\perkunasv2_c4_tokenized_val `
  --seq-len 512 `
  --micro-batch-size $MicroBatchSize `
  --gradient-accumulation-steps $GradientAccumulationSteps `
  --dtype fp16 `
  --master-weight-dtype fp16 `
  --shard-storage-format safetensors `
  --storage-shard-count 96 `
  --device cuda `
  --optimizer adafactor `
  --learning-rate 7e-5 `
  --weight-decay 0.05 `
  --beta1 0.9 `
  --beta2 0.95 `
  --adam-eps 1e-8 `
  --max-grad-norm 1.0 `
  --grad-clip-mode shard `
  --lr-schedule tokens `
  --warmup-tokens 26214400 `
  --decay-tokens 500000000 `
  --min-lr-ratio 0.75 `
  --max-steps $MaxSteps `
  --save-every 10 `
  --validate-every 10 `
  --max-validation-batches 4 `
  --shuffle-train `
  --max-resident-shards 4 `
  --cache-active-modules `
  --prefetch-shards cpu `
  --prefetch-window 4 `
  --prefetch-optimizer-shards `
  --no-clear-cuda-cache-between-shards `
  --shard-log-every 0 `
  --trainer-state-every 25 `
  --lm-head-chunk-tokens 128 `
  --async-shard-writes `
  --max-pending-shard-writes 16
