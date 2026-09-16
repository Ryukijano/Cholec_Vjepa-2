#!/usr/bin/env bash
# =============================================================================
# RF-DETR baseline: Fine-tune RF-DETR-Base on CholecTrack20 (COCO format).
#
# RF-DETR uses DINOv2 backbone + 3-layer deformable DETR decoder — same
# architecture pattern as our SurgeNetDINO system, but COCO-pretrained
# and with Group DETR training + EMA. This serves as a strong external
# baseline for the paper.
#
# Usage (from 3-GPU node):
#   bash scripts/got_jepa/run_rfdetr_baseline.sh
#
# Dataset: /scratch/kcwp264/data/surgi_world_track/cholec20_coco/
#   train/_annotations.coco.json  (16.9K images, 14.8K annotations)
#   valid/_annotations.coco.json (2.7K images, 2.4K annotations)
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
# 2. Check rfdetr is available
# ------------------------------------------------------------------
if ! python -c "import rfdetr" 2>/dev/null; then
  echo "rfdetr not installed. Installing..."
  pip install rfdetr
fi

# ------------------------------------------------------------------
# 3. Run RF-DETR fine-tuning
# ------------------------------------------------------------------
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

DATASET_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
OUTPUT_DIR="outputs/mot/rfdetr-baseline"

echo "============================================================"
echo "  RF-DETR Baseline Fine-tuning"
echo "  Dataset: ${DATASET_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python - <<'PY'
import os
import sys

os.environ.setdefault("XFORMERS_DISABLED", "1")

from rfdetr import RFDETRBase

DATASET_DIR = "/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
OUTPUT_DIR = "outputs/mot/rfdetr-baseline"

print(f"RF-DETR Base: fine-tuning on CholecTrack20")
print(f"  Dataset: {DATASET_DIR}")
print(f"  Output:  {OUTPUT_DIR}")

model = RFDETRBase()

model.train(
    dataset_dir=DATASET_DIR,
    epochs=20,
    batch_size=8,
    grad_accum_steps=2,   # effective batch = 16
    lr=1e-4,
    lr_encoder=1.5e-5,    # lower LR for DINOv2 backbone (frozen-ish)
    output_dir=OUTPUT_DIR,
    use_ema=True,
)

print(f"\nRF-DETR training complete. Checkpoints in {OUTPUT_DIR}/")
print(f"Best checkpoint: {OUTPUT_DIR}/checkpoint_best_total.pth")
PY

echo ""
echo "============================================================"
echo "  RF-DETR baseline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
