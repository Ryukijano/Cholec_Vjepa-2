#!/bin/bash
#SBATCH --job-name=rfdetr-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
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
# 4. Sanity check
# ---------------------------------------------------------------------------
echo "=== RF-DETR Stage 1 Fine-tuning on CholecTrack20 ==="
echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-n/a}"
echo "GPUs:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | wc -l) x L40S"
date
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 5. Launch RF-DETR training (3-GPU DDP)
# ---------------------------------------------------------------------------
GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "${GPUS}" -gt 1 ]; then
    echo "Launching ${GPUS}-GPU DDP training..."
    torchrun --standalone --nproc_per_node="${GPUS}" \
        scripts/got_jepa/train_rfdetr_stage1.py --ddp
else
    echo "Launching single-GPU training..."
    python scripts/got_jepa/train_rfdetr_stage1.py
fi

echo ""
echo "=== RF-DETR training finished ==="
date
