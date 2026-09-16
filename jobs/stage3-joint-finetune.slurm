#!/bin/bash
# =============================================================================
# SLURM: Stage 3 joint fine-tune — detector + per-track predictor + ReID.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# Prerequisites:
#   1. Stage 1 checkpoint: outputs/mot/cholec20-stage1-surgenet/best.pth.tar
#   2. Stage 2 checkpoint: outputs/mot/cholec80-ct20-stage2-jepa-surgenet/best.pth.tar
#
# Stage 3 unfreezes the DETR head + ReID + per-track predictor (student from
# Stage 2) and jointly fine-tunes on CholecTrack20 labeled data.
#
# Submit:
#   sbatch jobs/stage3-joint-finetune.slurm
# =============================================================================
#SBATCH --job-name=stage3-joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail

module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1/bin")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/conda/envs/endofm-lv
set -u
export CONDA_DEFAULT_ENV=endofm-lv

REPO_ROOT="/scratch/kcwp264/Cholec_Vjepa-2"
GPUS=3

cd "${REPO_ROOT}"
mkdir -p logs

export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "=== Stage 3 Joint Fine-tune ==="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "GPUs:   ${GPUS} x L40S"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Check prerequisites
for CKPT in \
  "outputs/mot/cholec20-stage1-surgenet/best.pth.tar" \
  "outputs/mot/cholec80-ct20-stage2-jepa-surgenet/best.pth.tar"; do
  if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
  fi
done

# Use Stage 1 config as base (will be overridden by Stage 3 config if it exists)
STAGE3_CONFIG="configs/train_mot/dinov2/cholec20-mot-stage3-joint-surgenet.yaml"

if [ -f "${STAGE3_CONFIG}" ]; then
  echo "Config:  ${STAGE3_CONFIG}"
else
  echo "[WARNING] Stage 3 config not found: ${STAGE3_CONFIG}"
  echo "          Creating one from Stage 1 config will be needed."
  echo "          For now, using Stage 1 config with stage3_jepa meta override."
  STAGE3_CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-surgenet.yaml"
fi

torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${STAGE3_CONFIG}" \
  --devices cuda

echo ""
echo "=== Stage 3 finished ==="
date
