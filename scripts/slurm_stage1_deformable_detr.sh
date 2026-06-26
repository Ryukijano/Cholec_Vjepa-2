#!/bin/bash
# =============================================================================
# SLURM: Stage 1 supervised MOT training with Deformable DETR head.
# 3x NVIDIA L40S (Ada Lovelace, 48 GB) on a single AIRE GPU node.
# =============================================================================
#SBATCH --job-name=stage1-defdetr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

# Activate the CUDA-capable conda env
source activate surgi_world_track_cuda

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/users/kcwp264/TRACK_JEPA/surgi_world_track"
CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
GPUS=3

cd "${REPO_ROOT}"
mkdir -p logs outputs/mot/cholec20-stage1-supervised-backup

# ---------------------------------------------------------------------------
# 3. Backup old (vanilla DETR) checkpoints — incompatible with Deformable DETR
# ---------------------------------------------------------------------------
CKPT_DIR="outputs/mot/cholec20-stage1-supervised"
if [ -f "${CKPT_DIR}/best.pth.tar" ]; then
  echo "[INFO] Backing up incompatible vanilla-DETR checkpoints..."
  mv "${CKPT_DIR}/best.pth.tar"   "outputs/mot/cholec20-stage1-supervised-backup/best-vanilla-detr.pth.tar"   2>/dev/null || true
  mv "${CKPT_DIR}/latest.pth.tar" "outputs/mot/cholec20-stage1-supervised-backup/latest-vanilla-detr.pth.tar" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------------
echo "=== Stage 1 Deformable DETR training ==="
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-n/a}"
echo "GPUs:       ${GPUS} x L40S"
echo "Config:     ${CONFIG}"
date
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 5. Launch 3-GPU DDP training
# ---------------------------------------------------------------------------
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda

echo ""
echo "=== Training finished ==="
date
