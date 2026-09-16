#!/bin/bash
# =============================================================================
# SLURM: Stage 2 GOT-JEPA SSL pretraining with VISReg regularization.
# 3x NVIDIA L40S (Ada, 48 GB) on a single AIRE GPU node.
#
# Prerequisites:
#   1. Stage 1 SurgeNetDINO checkpoint at:
#      outputs/mot/cholec20-stage1-surgenet/best.pth.tar
#   2. SSL corpus built at:
#      /scratch/kcwp264/data/surgi_world_track/ssl_corpus
#
# Uses VISReg (Variance-Invariance-Sketching) regularization instead of
# VICReg covariance — non-vanishing gradient at collapse, better for
# small surgical datasets (Wu et al., 2026).
#
# Submit:
#   sbatch jobs/stage2-jepa-surgenet.slurm
# =============================================================================
#SBATCH --job-name=stage2-visreg
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

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

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
CONFIG="configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-surgenet.yaml"
GPUS=3

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------
echo "=== Stage 2 GOT-JEPA + VISReg SSL training ==="
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-n/a}"
echo "GPUs:       ${GPUS} x L40S"
echo "Config:     ${CONFIG}"
date
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Check Stage 1 checkpoint
STAGE1_CKPT="outputs/mot/cholec20-stage1-surgenet/best.pth.tar"
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
  echo "        Run Stage 1 first: sbatch jobs/stage1-surgenet.slurm"
  exit 1
fi
python -c "
import torch
ckpt = torch.load('${STAGE1_CKPT}', map_location='cpu', weights_only=False)
print(f'  Stage 1 checkpoint epoch: {ckpt.get(\"epoch\", \"?\")}')
print(f'  Stage: {ckpt.get(\"stage\", \"?\")}')
"

# Check SSL corpus
SSL_ROOT="/scratch/kcwp264/data/surgi_world_track/ssl_corpus"
if [ -d "${SSL_ROOT}/Training" ]; then
  NUM_VIDS=$(find "${SSL_ROOT}/Training" -maxdepth 1 -type d | wc -l)
  echo "[INFO] SSL corpus found: ${SSL_ROOT} (${NUM_VIDS} subdirs)"
else
  echo "[WARNING] SSL corpus not found at ${SSL_ROOT}"
  echo "          Build it first: bash scripts/got_jepa/build_ssl_corpus_3gpu.sh"
  echo "          Proceeding anyway — training will fail if data is missing"
fi

# Check VISReg config
echo "[INFO] VISReg regularization: enabled (jepa_reg_mode=visreg)"

# ---------------------------------------------------------------------------
# 5. Launch 3-GPU DDP Stage 2 training
# ---------------------------------------------------------------------------
torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda

echo ""
echo "=== Stage 2 training finished ==="
date
