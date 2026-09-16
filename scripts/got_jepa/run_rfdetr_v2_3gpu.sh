#!/usr/bin/env bash
# =============================================================================
# RF-DETR Large v2 — 3-GPU DDP training with copy-paste augmentation
#
# Improvements over v1:
#   - RFDETRLarge (33.9M, 704px) instead of RFDETRBase (29M, 560px)
#   - Copy-paste augmentation for rare classes (clipper, hook, scissors, grasper)
#   - Aggressive Albumentations (blur, CLAHE, noise, rotation, scale)
#   - Lower LR (5e-5) with cosine schedule + 5ep warmup
#   - Early stopping with patience=20
#   - 200 max epochs
#
# Usage (from 3-GPU interactive node):
#   bash scripts/got_jepa/run_rfdetr_v2_3gpu.sh
#
# To skip augmentation if already done:
#   SKIP_AUG=1 bash scripts/got_jepa/run_rfdetr_v2_3gpu.sh
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
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

DATASET_DIR_ORIG="/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
AUG_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco_augmented"
OUTPUT_DIR="/scratch/kcwp264/Cholec_Vjepa-2/outputs/mot/rfdetr-large-v2"

mkdir -p logs

# ------------------------------------------------------------------
# 2. Copy-paste augmentation (skip if already done or SKIP_AUG=1)
# ------------------------------------------------------------------
# Clean up broken augmented dir from previous failed run
if [ -d "${AUG_DIR}/train" ] && [ ! -f "${AUG_DIR}/train/_annotations.coco.json" ]; then
  echo ">>> Cleaning up broken augmented train dir (no COCO json found)"
  rm -rf "${AUG_DIR}/train"
fi

if [ "${SKIP_AUG:-0}" = "1" ]; then
  echo ">>> Skipping augmentation (SKIP_AUG=1)"
elif [ -f "${AUG_DIR}/train/_annotations.coco.json" ]; then
  echo ">>> Augmented dataset already exists at ${AUG_DIR}, skipping"
else
  echo ">>> [$(date '+%H:%M:%S')] Running copy-paste augmentation..."
  python scripts/got_jepa/copy_paste_augment.py \
      --source_dir "${DATASET_DIR_ORIG}/train" \
      --output_dir "/scratch/kcwp264/data/surgi_world_track/cholec20_coco_train_augmented" \
      --target_per_class 3000 \
      --seed 42

  mkdir -p "${AUG_DIR}"
  cp -r "${DATASET_DIR_ORIG}/valid" "${AUG_DIR}/valid"
  mv "/scratch/kcwp264/data/surgi_world_track/cholec20_coco_train_augmented" "${AUG_DIR}/train"
  echo ">>> Augmentation complete. Dataset at ${AUG_DIR}"
fi

# ------------------------------------------------------------------
# 3. Train RF-DETR Large with DDP on 3 GPUs
# ------------------------------------------------------------------
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo ""
echo "============================================================"
echo "  RF-DETR Large v2 — 3-GPU DDP Training"
echo "  Dataset:   ${AUG_DIR}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Use the .py file directly — heredoc breaks Lightning DDP child process spawning
export XFORMERS_DISABLED=1
python scripts/got_jepa/train_rfdetr_stage1.py --ddp

echo ""
echo "============================================================"
echo "  RF-DETR Large v2 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"
