#!/usr/bin/env bash
# =============================================================================
# Launch Stage 1 supervised training via torchrun DDP, auto-adjusted to visible GPUs.
#
# Usage:
#   module load miniforge
#   bash scripts/train_stage1_ddp_3gpu.sh
#
# This produces the Stage 1 checkpoint required for the SSL corpus build.
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# 1. Activate conda env (robust across shells and node layouts)
# ------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module load miniforge || true
fi

SURGI_ENV="${SURGI_ENV:-surgi_track}"
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
        # shell: enable conda command in non-login shells
        # shellcheck disable=SC1090
        source "${conda_root}/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Could not locate conda command. Please install Miniforge/Conda or add conda to PATH."
    exit 1
  fi

  if ! declare -f conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base:-}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "${conda_base}/etc/profile.d/conda.sh"
    else
      eval "$("$(command -v conda)" shell.bash hook)"
    fi
  fi

  if ! conda activate "${SURGI_ENV}"; then
    if [ "${SURGI_ENV}" != "${FALLBACK_ENV}" ]; then
      echo "Primary env ${SURGI_ENV} unavailable, falling back to ${FALLBACK_ENV}."
      conda activate "${FALLBACK_ENV}" || { echo "Could not activate any target conda env"; exit 1; }
      ACTIVE_ENV="${FALLBACK_ENV}"
    else
      echo "Could not activate conda environment: ${SURGI_ENV}"
      exit 1
    fi
  else
    ACTIVE_ENV="${SURGI_ENV}"
  fi
fi

if [ "${ACTIVE_ENV:-${CONDA_DEFAULT_ENV:-${CONDA_PREFIX##*/}}}" != "${SURGI_ENV}" ] && \
   [ "${ACTIVE_ENV:-${CONDA_DEFAULT_ENV:-${CONDA_PREFIX##*/}}}" != "${FALLBACK_ENV}" ]; then
  echo "Could not get supported conda env active. Expected ${SURGI_ENV} or ${FALLBACK_ENV}."
  exit 1
fi

CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"

# Optional override for explicit GPU count, otherwise default to 3.
REQUESTED_GPUS="${STAGE1_GPUS:-3}"
if ! [[ "${REQUESTED_GPUS}" =~ ^[0-9]+$ ]] || [ "${REQUESTED_GPUS}" -lt 1 ]; then
  echo "Invalid STAGE1_GPUS value: ${REQUESTED_GPUS}. Must be a positive integer."
  exit 1
fi
GPUS="${REQUESTED_GPUS}"

AVAILABLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "No CUDA devices visible (torch.cuda.device_count() == 0)."
  echo "Either make GPUs visible via CUDA_VISIBLE_DEVICES or run on a machine with CUDA."
  exit 1
fi

if [ "${GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "Requested ${GPUS} processes but only ${AVAILABLE_GPUS} GPUs are visible."
  echo "Overriding nproc_per_node to ${AVAILABLE_GPUS}."
  GPUS="${AVAILABLE_GPUS}"
fi

if [ "${GPUS}" -gt 1 ]; then
  echo "Starting Stage 1 DDP training on ${GPUS} GPUs..."
else
  echo "Starting Stage 1 single-GPU training (no DDP)."
fi
echo "Config: ${CONFIG}"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"

# Optional: log in to W&B first if not already authenticated
# wandb login

cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN

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
