#!/usr/bin/env bash
# =============================================================================
# Ablation runner: sequentially runs DETR head size variants for 20 epochs each.
#
# Usage (from a GPU node with endofm-lv activated):
#   bash scripts/got_jepa/run_ablation_stage1.sh
#
# Each variant trains from scratch (no resume) for 20 epochs.
# Results go to outputs/mot/ablation-*/ and W&B group "ablation-stage1".
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# 1. Activate conda env (same logic as train_stage1_ddp_3gpu.sh)
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
# 2. GPU detection
# ------------------------------------------------------------------
REQUESTED_GPUS="${STAGE1_GPUS:-3}"
GPUS="${REQUESTED_GPUS}"
AVAILABLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "No CUDA devices visible."
  exit 1
fi
if [ "${GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
  GPUS="${AVAILABLE_GPUS}"
fi
echo "Using ${GPUS} GPUs for ablation runs."

# ------------------------------------------------------------------
# 3. Environment
# ------------------------------------------------------------------
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1
export LD_LIBRARY_PATH="/scratch/kcwp264/conda/envs/endofm-lv/lib:${LD_LIBRARY_PATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_BLOCKING_WAIT=1

# ------------------------------------------------------------------
# 4. Run ablation variants sequentially
# ------------------------------------------------------------------
CONFIGS=(
  "configs/train_mot/dinov2/ablation-small-detr.yaml"
  "configs/train_mot/dinov2/ablation-small-no-dn.yaml"
  "configs/train_mot/dinov2/ablation-tiny-detr.yaml"
  "configs/train_mot/dinov2/ablation-micro-detr.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  NAME=$(basename "${CONFIG}" .yaml)
  echo ""
  echo "============================================================"
  echo "  Running ablation: ${NAME}"
  echo "  Config: ${CONFIG}"
  echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if [ "${GPUS}" -gt 1 ]; then
    torchrun \
      --standalone \
      --nproc_per_node="${GPUS}" \
      -m core_app.mot.main \
      --fname "${CONFIG}" \
      --devices cuda
  else
    python -m core_app.mot.main \
      --fname "${CONFIG}" \
      --devices cuda:0
  fi

  echo "Finished: ${NAME} at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================================"
echo "  All ablation runs complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Check W&B group 'ablation-stage1' for comparison."
echo "  Output dirs: outputs/mot/ablation-*/"
echo "============================================================"
