#!/usr/bin/env bash
# =============================================================================
# Run ablation study with RF-DETR pretrained DETR decoder weights.
#
# Usage (from 3-GPU node):
#   bash scripts/got_jepa/run_ablation_rfdetr_init.sh
#
# Launches 3 variants simultaneously (one per GPU), 4th when a GPU frees up.
# Each variant runs 50 epochs (longer than before since we now have good init).
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

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1
export LD_LIBRARY_PATH="/scratch/kcwp264/conda/envs/endofm-lv/lib:${LD_LIBRARY_PATH:-}"

AVAILABLE_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "No CUDA devices visible."
  exit 1
fi
echo "Detected ${AVAILABLE_GPUS} GPUs."

# ------------------------------------------------------------------
# 2. Variants and init checkpoints
# ------------------------------------------------------------------
VARIANTS=(
  "ablation-small-detr"
  "ablation-small-no-dn"
  "ablation-tiny-detr"
  "ablation-micro-detr"
)

declare -A GPU_PIDS
declare -A GPU_VARIANT

run_variant() {
  local gpu=$1
  local variant=$2
  local config="configs/train_mot/dinov2/${variant}.yaml"
  local init_ckpt="outputs/mot/rfdetr_init_${variant}.pth"
  local logfile="outputs/mot/${variant}-rfdetr-init/console.log"
  mkdir -p "outputs/mot/${variant}-rfdetr-init"

  echo "[$(date '+%H:%M:%S')] Launching ${variant} (RF-DETR init) on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES=${gpu} python -m core_app.mot.main \
    --fname "${config}" \
    --devices cuda:0 \
    --detr-init "${init_ckpt}" \
    > "${logfile}" 2>&1 &
  local pid=$!
  GPU_PIDS[${gpu}]=${pid}
  GPU_VARIANT[${gpu}]=${variant}
  echo "  PID=${pid} → ${logfile}"
}

# Launch first batch
gpu=0
for variant in "${VARIANTS[@]}"; do
  if [ ${gpu} -ge ${AVAILABLE_GPUS} ]; then break; fi
  run_variant ${gpu} "${variant}"
  gpu=$((gpu + 1))
done

REMAINING=()
idx=${AVAILABLE_GPUS}
while [ ${idx} -lt ${#VARIANTS[@]} ]; do
  REMAINING+=("${VARIANTS[${idx}]}")
  idx=$((idx + 1))
done

echo ""
echo "Launched ${AVAILABLE_GPUS} variants. ${#REMAINING[@]} queued."
echo ""

# Wait for GPUs to free, launch remaining
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
      next="${REMAINING[0]}"
      REMAINING=("${REMAINING[@]:1}")
      run_variant ${gpu} "${next}"
      break
    fi
  done
  sleep 30
done

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
echo "  All RF-DETR-init ablation runs complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Console logs: outputs/mot/ablation-*-rfdetr-init/console.log"
echo "============================================================"
