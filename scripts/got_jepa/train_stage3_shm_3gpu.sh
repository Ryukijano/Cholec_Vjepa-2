#!/usr/bin/env bash
# =============================================================================
# Stage 3 Joint Fine-tune — 3-GPU DDP with /dev/shm data caching
#
# Copies CholecTrack20 to /dev/shm (RAM-backed tmpfs) for fast data loading,
# then runs Stage 3 joint fine-tuning loading Stage 1 + Stage 2 checkpoints.
#
# Usage:
#   bash scripts/got_jepa/train_stage3_shm_3gpu.sh
#   bash scripts/got_jepa/train_stage3_shm_3gpu.sh --reset-optimizer
# =============================================================================
set -euo pipefail

# --- Activate conda env by full path ---
ENDOFM_ENV="${ENDOFM_ENV:-/scratch/kcwp264/conda/envs/endofm-lv}"
if [ "${CONDA_DEFAULT_ENV:-}" != "endofm-lv" ]; then
  for conda_root in \
    "/opt/apps/pkg/interpreters/miniforge/24.7.1" \
    "/scratch/kcwp264/.conda_envs" \
    "${HOME}/miniforge3" \
    "/opt/miniforge3"; do
    if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
      source "${conda_root}/etc/profile.d/conda.sh"
      break
    fi
  done
  conda activate "${ENDOFM_ENV}" || { echo "Could not activate ${ENDOFM_ENV}"; exit 1; }
fi
echo "Using conda env: ${CONDA_PREFIX}"

cd /scratch/kcwp264/Cholec_Vjepa-2

# --- NCCL env vars for L40S PCIe ---
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export XFORMERS_DISABLED=1

# --- Copy dataset to /dev/shm for fast loading ---
SRC_DATA="/scratch/kcwp264/datasets_cholec/cholectrack20"
SHM_DATA="/dev/shm/cholectrack20"

if [ -d "${SHM_DATA}" ] && [ -f "${SHM_DATA}/SYNAPSE_METADATA_MANIFEST.tsv" ]; then
  echo ">>> /dev/shm already has cholectrack20, skipping copy"
else
  echo ">>> Copying CholecTrack20 to /dev/shm (RAM-backed tmpfs)..."
  echo "    Source: ${SRC_DATA} ($(du -sh "${SRC_DATA}" 2>/dev/null | cut -f1))"
  echo "    Target: ${SHM_DATA}"
  echo "    /dev/shm free: $(df -h /dev/shm | tail -1 | awk '{print $4}')"
  mkdir -p /dev/shm
  cp -r "${SRC_DATA}" "${SHM_DATA}"
  echo "    Copy complete. /dev/shm free: $(df -h /dev/shm | tail -1 | awk '{print $4}')"
fi

export MOT_DATA_ROOT="${SHM_DATA}"

# --- Verify checkpoints exist ---
STAGE1_CKPT="outputs/mot/cholec20-stage1-surgenet/best.pth.tar"
STAGE2_CKPT="outputs/mot/cholec80-ct20-stage2-jepa-surgenet/latest.pth.tar"
echo ">>> Checking checkpoints..."
ls -lh "${STAGE1_CKPT}" || { echo "ERROR: Stage 1 checkpoint missing"; exit 1; }
ls -lh "${STAGE2_CKPT}" || { echo "ERROR: Stage 2 checkpoint missing"; exit 1; }

# --- Determine GPU count ---
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo ">>> GPUs visible: ${NUM_GPUS}"

CONFIG="configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml"

echo ""
echo "============================================================"
echo "  Stage 3 Joint Fine-tune — 3-GPU DDP"
echo "  Config:   ${CONFIG}"
echo "  Data:     ${SHM_DATA} (RAM-backed tmpfs)"
echo "  Stage 1:  ${STAGE1_CKPT}"
echo "  Stage 2:  ${STAGE2_CKPT}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Start:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# --- Parse extra args ---
EXTRA_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --reset-optimizer) EXTRA_ARGS="${EXTRA_ARGS} --reset-optimizer" ;;
  esac
done

# --- Launch Stage 3 training ---
if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m core_app.mot.main \
    --fname "${CONFIG}" \
    --devices cuda \
    ${EXTRA_ARGS}
else
  python -m core_app.mot.main \
    --fname "${CONFIG}" \
    --devices cuda:0 \
    ${EXTRA_ARGS}
fi

echo ""
echo "============================================================"
echo "  Stage 3 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- Cleanup shm (optional — comment out to keep for re-runs) ---
# echo ">>> Cleaning up /dev/shm..."
# rm -rf "${SHM_DATA}"
