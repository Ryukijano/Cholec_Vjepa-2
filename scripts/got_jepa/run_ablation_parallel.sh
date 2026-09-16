#!/usr/bin/env bash
# =============================================================================
# Parallel ablation runner: launches up to 3 variants simultaneously, one per GPU.
# The 4th variant starts as soon as any GPU frees up.
#
# Usage (from a GPU node with endofm-lv activated):
#   bash scripts/got_jepa/run_ablation_parallel.sh
#
# If ablation-small-detr is already running on GPU 0, this script will
# skip it and launch the remaining 3 on GPUs 1, 2, and 0 (if free).
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# 1. Activate conda env
# ------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module load miniforge || true
fi

SURGI_ENV="${SURGI_ENV:-endofm-lv}"
FALLBACK_ENV="surgi_world_track_cuda"

ACTIVE_ENV="${CONDA_DEFAULT_ENV:-${CONDA_PREFIX##*/}}"
if [ -n "${ACTIVE_ENV}" ] && { [ "${ACTIVE_ENV}" = "${SURGI_ENV}" ] || [ "${ACTIVE_ENV}" = "${FALLBACK_ENV}" ]; }; then
  echo "Using already-active conda env: ${ACTIVE_ENV}"
else
  if ! command -v conda >/dev/null 2>&1; then
    for conda_root in \
      "${HOME}/miniforge3" \
      "${HOME}/miniconda3" \
      "${HOME}/anaconda3" \
      "/opt/miniforge3" \
      "/opt/miniconda3" \
      "/opt/conda"; do
      if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
        source "${conda_root}/etc/profile.d/conda.sh"
        break
      fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "Could not locate conda command."
    exit 1
  fi
  if ! declare -f conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base:-}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      source "${conda_base}/etc/profile.d/conda.sh"
    else
      eval "$("$(command -v conda)" shell.bash hook)"
    fi
  fi
  if ! conda activate "${SURGI_ENV}"; then
    if [ "${SURGI_ENV}" != "${FALLBACK_ENV}" ]; then
      echo "Primary env ${SURGI_ENV} unavailable, falling back to ${FALLBACK_ENV}."
      conda activate "${FALLBACK_ENV}" || { echo "Could not activate any target conda env"; exit 1; }
    else
      echo "Could not activate conda environment: ${SURGI_ENV}"
      exit 1
    fi
  fi
fi

# ------------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------------
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1
export LD_LIBRARY_PATH="/scratch/kcwp264/conda/envs/endofm-lv/lib:${LD_LIBRARY_PATH:-}"
# NCCL not needed for single-GPU runs, but set for safety
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

AVAILABLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "No CUDA devices visible."
  exit 1
fi

echo "Detected ${AVAILABLE_GPUS} GPUs."

# ------------------------------------------------------------------
# 3. Define variants and launch
# ------------------------------------------------------------------
# All 4 variants. Edit this list to skip ones already running.
VARIANTS=(
  "ablation-small-detr"
  "ablation-small-no-dn"
  "ablation-tiny-detr"
  "ablation-micro-detr"
)

# Track PIDs per GPU
declare -A GPU_PIDS
declare -A GPU_VARIANT

run_variant() {
  local gpu=$1
  local variant=$2
  local config="configs/train_mot/dinov2/${variant}.yaml"
  local logfile="outputs/mot/${variant}/console.log"
  mkdir -p "outputs/mot/${variant}"

  echo "[$(date '+%H:%M:%S')] Launching ${variant} on GPU ${gpu} → ${logfile}"
  CUDA_VISIBLE_DEVICES=${gpu} python -m core_app.mot.main \
    --fname "${config}" \
    --devices cuda:0 \
    > "${logfile}" 2>&1 &
  local pid=$!
  GPU_PIDS[${gpu}]=${pid}
  GPU_VARIANT[${gpu}]=${variant}
  echo "  PID=${pid}"
}

# Launch first batch (up to AVAILABLE_GPUS)
gpu=0
for variant in "${VARIANTS[@]}"; do
  if [ ${gpu} -ge ${AVAILABLE_GPUS} ]; then
    break
  fi
  run_variant ${gpu} "${variant}"
  gpu=$((gpu + 1))
done

# remaining variants
REMAINING=()
idx=${AVAILABLE_GPUS}
while [ ${idx} -lt ${#VARIANTS[@]} ]; do
  REMAINING+=("${VARIANTS[${idx}]}")
  idx=$((idx + 1))
done

echo ""
echo "Launched ${AVAILABLE_GPUS} variants. ${#REMAINING[@]} queued."
echo ""

# Wait for GPUs to free up, then launch remaining
while [ ${#REMAINING[@]} -gt 0 ]; do
  for gpu in "${!GPU_PIDS[@]}"; do
    pid="${GPU_PIDS[${gpu}]}"
    variant="${GPU_VARIANT[${gpu}]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null
      exit_code=$?
      echo "[$(date '+%H:%M:%S')] FINISHED ${variant} on GPU ${gpu} (exit=${exit_code})"
      unset GPU_PIDS[${gpu}]
      unset GPU_VARIANT[${gpu}]

      # Launch next queued variant
      next="${REMAINING[0]}"
      REMAINING=("${REMAINING[@]:1}")
      run_variant ${gpu} "${next}"
      break
    fi
  done
  sleep 30
done

# Wait for all remaining
echo ""
echo "All variants launched. Waiting for completion..."
for gpu in "${!GPU_PIDS[@]}"; do
  pid="${GPU_PIDS[${gpu}]}"
  variant="${GPU_VARIANT[${gpu}]}"
  wait "${pid}" 2>/dev/null
  exit_code=$?
  echo "[$(date '+%H:%M:%S')] FINISHED ${variant} on GPU ${gpu} (exit=${exit_code})"
done

echo ""
echo "============================================================"
echo "  All ablation runs complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Check W&B group 'ablation-stage1' for comparison."
echo "  Console logs: outputs/mot/ablation-*/console.log"
echo "============================================================"
