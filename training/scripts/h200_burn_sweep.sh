#!/usr/bin/env bash
set -euo pipefail

# One-hour H200 burn harness for Perkunasv2.
#
# Expected environment variables:
#   BASE_RUN_DIR   Existing run/checkpoint directory to clone per experiment.
#   DATA_DIR       Training tokenized data directory on fast local NVMe.
#   VAL_DATA_DIR   Validation tokenized data directory on fast local NVMe.
#
# Optional environment variables:
#   WORK_ROOT      Scratch working directory for cloned runs and logs.
#   TARGET_STEP    Absolute max step. If BASE_RUN_DIR is at step 650 and TARGET_STEP=1000,
#                  each job trains roughly 350 additional steps.
#   PYTHON         Python executable.
#
# Example:
#   BASE_RUN_DIR=/scratch/perkunasv2_base \
#   DATA_DIR=/scratch/data/perkunasv2_c4_tokenized_bigshards \
#   VAL_DATA_DIR=/scratch/data/perkunasv2_c4_val_small \
#   TARGET_STEP=1000 \
#   bash training/scripts/h200_burn_sweep.sh

: "${BASE_RUN_DIR:?Set BASE_RUN_DIR to an existing Perkunasv2 run directory.}"
: "${DATA_DIR:?Set DATA_DIR to the tokenized training data directory.}"
: "${VAL_DATA_DIR:?Set VAL_DATA_DIR to the tokenized validation data directory.}"

WORK_ROOT="${WORK_ROOT:-/scratch/perkunasv2_h200_burn}"
TARGET_STEP="${TARGET_STEP:-1000}"
PYTHON_BIN="${PYTHON:-python}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${WORK_ROOT}/runs"
LOG_ROOT="${WORK_ROOT}/logs"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

clone_run() {
  local name="$1"
  local dst="${RUN_ROOT}/${name}"
  if [[ ! -d "${dst}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    rsync -a --delete "${BASE_RUN_DIR}/" "${dst}/"
  fi
  printf '%s\n' "${dst}"
}

common_args() {
  local run_dir="$1"
  local micro_batch="$2"
  local prefetch_mode="$3"
  local prefetch_window="$4"
  local prefetch_optimizer="$5"
  local prefetch_device="${6:-}"

  local args=(
    "${REPO_ROOT}/training/scripts/train_perkunasv2.py"
    --train
    --run-dir "${run_dir}"
    --data-dir "${DATA_DIR}"
    --val-data-dir "${VAL_DATA_DIR}"
    --seq-len 512
    --micro-batch-size "${micro_batch}"
    --gradient-accumulation-steps 1
    --dtype fp16
    --device cuda
    --optimizer adamw
    --learning-rate 1e-4
    --weight-decay 0.1
    --beta1 0.9
    --beta2 0.95
    --adam-eps 1e-8
    --max-grad-norm 1.0
    --lr-schedule tokens
    --warmup-tokens 5000
    --decay-tokens 200000
    --min-lr-ratio 0.1
    --max-steps "${TARGET_STEP}"
    --save-every 1000
    --validate-every 25
    --max-validation-batches 24
    --max-resident-shards "${prefetch_window}"
    --prefetch-shards "${prefetch_mode}"
    --prefetch-window "${prefetch_window}"
    --no-clear-cuda-cache-between-shards
    --shard-log-every 0
    --trainer-state-every 25
    --lm-head-chunk-tokens 4096
    --async-shard-writes
    --max-pending-shard-writes 8
  )

  if [[ "${prefetch_optimizer}" == "yes" ]]; then
    args+=(--prefetch-optimizer-shards)
  else
    args+=(--no-prefetch-optimizer-shards)
  fi

  if [[ -n "${prefetch_device}" ]]; then
    args+=(--prefetch-device "${prefetch_device}")
  fi

  printf '%q ' "${args[@]}"
}

launch_one_gpu() {
  local physical_gpu="$1"
  local name="$2"
  local micro_batch="$3"
  local prefetch_mode="$4"
  local prefetch_window="$5"
  local prefetch_optimizer="$6"

  local run_dir
  run_dir="$(clone_run "${name}")"
  local log_file="${LOG_ROOT}/${name}.log"
  local cmd
  cmd="$(common_args "${run_dir}" "${micro_batch}" "${prefetch_mode}" "${prefetch_window}" "${prefetch_optimizer}")"

  echo "[launch] gpu=${physical_gpu} name=${name} micro_batch=${micro_batch} prefetch=${prefetch_mode} optimizer_prefetch=${prefetch_optimizer}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" bash -lc "${PYTHON_BIN} ${cmd} 2>&1 | tee '${log_file}'" &
}

launch_secondary_gpu() {
  local physical_gpu_a="$1"
  local physical_gpu_b="$2"
  local name="$3"
  local micro_batch="$4"
  local prefetch_window="$5"

  local run_dir
  run_dir="$(clone_run "${name}")"
  local log_file="${LOG_ROOT}/${name}.log"
  local cmd
  cmd="$(common_args "${run_dir}" "${micro_batch}" "secondary-gpu" "${prefetch_window}" "yes" "cuda:1")"

  echo "[launch] gpus=${physical_gpu_a},${physical_gpu_b} name=${name} micro_batch=${micro_batch} prefetch=secondary-gpu optimizer_prefetch=yes"
  CUDA_VISIBLE_DEVICES="${physical_gpu_a},${physical_gpu_b}" bash -lc "${PYTHON_BIN} ${cmd} 2>&1 | tee '${log_file}'" &
}

cd "${REPO_ROOT}"

launch_one_gpu 0 gpu0_baseline_mb128 128 off 1 no
launch_one_gpu 1 gpu1_cpu_prefetch_mb128 128 cpu 64 yes
launch_one_gpu 2 gpu2_gpu_param_prefetch_mb128 128 gpu 64 no
launch_one_gpu 3 gpu3_gpu_full_prefetch_mb128 128 gpu 64 yes
launch_one_gpu 4 gpu4_cpu_prefetch_mb256 256 cpu 64 yes
launch_one_gpu 5 gpu5_gpu_param_prefetch_mb256 256 gpu 64 no
launch_secondary_gpu 6 7 gpu6_7_secondary_prefetch_mb256 256 64

echo "[wait] all burn jobs launched; logs are in ${LOG_ROOT}"
wait
echo "[done] H200 burn sweep completed"
