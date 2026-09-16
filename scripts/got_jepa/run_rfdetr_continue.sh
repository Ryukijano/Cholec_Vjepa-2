#!/usr/bin/env bash
# =============================================================================
# RF-DETR continued training: resume from 20-epoch checkpoint for 30 more epochs.
# Uses best EMA checkpoint as pretrain weights for a fresh 30-epoch run.
#
# Usage (from 3-GPU node):
#   bash scripts/got_jepa/run_rfdetr_continue.sh
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

DATASET_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
PRETRAIN_CKPT="outputs/mot/rfdetr-baseline/checkpoint_best_total.pth"
OUTPUT_DIR="outputs/mot/rfdetr-continued"

echo "============================================================"
echo "  RF-DETR Continued Training (30 more epochs)"
echo "  Dataset:   ${DATASET_DIR}"
echo "  Pretrain:  ${PRETRAIN_CKPT}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python - <<'PY'
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

from rfdetr import RFDETRBase

DATASET_DIR = "/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
PRETRAIN_CKPT = "outputs/mot/rfdetr-baseline/checkpoint_best_total.pth"
OUTPUT_DIR = "outputs/mot/rfdetr-continued"

print(f"RF-DETR Base: continuing from best EMA checkpoint")
print(f"  Dataset:   {DATASET_DIR}")
print(f"  Pretrain:  {PRETRAIN_CKPT}")
print(f"  Output:    {OUTPUT_DIR}")

model = RFDETRBase(pretrain_weights=PRETRAIN_CKPT)

model.train(
    dataset_dir=DATASET_DIR,
    epochs=30,
    batch_size=8,
    grad_accum_steps=2,   # effective batch = 16
    lr=5e-5,              # lower LR for continued fine-tuning
    lr_encoder=7.5e-6,    # very low LR for backbone
    output_dir=OUTPUT_DIR,
    use_ema=True,
)

print(f"\nRF-DETR continued training complete. Checkpoints in {OUTPUT_DIR}/")
PY

echo ""
echo "============================================================"
echo "  RF-DETR continued training complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
