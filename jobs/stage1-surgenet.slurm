#!/bin/bash
# =============================================================================
# SLURM: Stage 1 SurgeNetDINO MOT training — fresh start with label fix.
# 2x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# Label off-by-one bug fix (Jul 23 2026): CholecTrack20 uses 0-indexed
# instrument IDs (0-6) but code had -1 shift, corrupting 47% of labels.
# Also fixed bbox double-normalization in video_dataset.py.
# Target: mAP@50 ~0.45+ after 200 epochs.
#
# Submit:
#   sbatch jobs/stage1-surgenet.slurm
# =============================================================================
#SBATCH --job-name=stage1-surgenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

# Activate endofm-lv conda env — use explicit conda.sh source to avoid
# picking up the wrong environment. The activate scripts have unbound
# variables (NVCC_PREPEND_FLAGS), so temporarily disable -u.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1/bin")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/conda/envs/endofm-lv
set -u
export CONDA_DEFAULT_ENV=endofm-lv

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/scratch/kcwp264/Cholec_Vjepa-2"
CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-surgenet.yaml"
GPUS=2

cd "${REPO_ROOT}"
mkdir -p logs

# ---------------------------------------------------------------------------
# 3. NCCL settings for L40S (PCIe-only, no NVLink)
# ---------------------------------------------------------------------------
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------------
echo "=== Stage 1 SurgeNetDINO training (label-fix fresh start) ==="
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-n/a}"
echo "GPUs:       ${GPUS} x L40S"
echo "Config:     ${CONFIG}"
date
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Back up old checkpoints (trained with label off-by-one bug)
CKPT_DIR="outputs/mot/cholec20-stage1-surgenet"
if [ -f "${CKPT_DIR}/latest.pth.tar" ]; then
  BACKUP_DIR="${CKPT_DIR}_labelbug_backup_$(date +%Y%m%d)"
  echo "[INFO] Backing up old label-bug checkpoints to ${BACKUP_DIR}"
  mkdir -p "${BACKUP_DIR}"
  mv "${CKPT_DIR}"/*.pth.tar "${BACKUP_DIR}/" 2>/dev/null || true
fi
RESUME_ARGS=""

# ---------------------------------------------------------------------------
# 5. Launch 2-GPU DDP training
# ---------------------------------------------------------------------------
torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda \
  ${RESUME_ARGS}

echo ""
echo "=== Training finished ==="
date
