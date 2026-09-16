#!/bin/bash
#SBATCH --job-name=ssl-build-rfdetr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
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
# 3. NCCL + env
# ---------------------------------------------------------------------------
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 4. Sanity
# ---------------------------------------------------------------------------
echo "=== SSL Corpus Build (RF-DETR pseudo-labels) ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
date
python -c "import torch; print('torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 5. Build
# ---------------------------------------------------------------------------
bash scripts/got_jepa/build_ssl_corpus_rfdetr_3gpu.sh

echo ""
echo "=== SSL corpus build finished ==="
date
