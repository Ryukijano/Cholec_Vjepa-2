#!/bin/bash
# =============================================================================
# SLURM: Build SSL corpus from Stage 1 SurgeNetDINO checkpoint.
# 3x NVIDIA L40S — each GPU pseudo-labels ~25 Cholec80 videos in parallel.
#
# Prerequisites:
#   - Stage 1 checkpoint at outputs/mot/cholec20-stage1-surgenet/best.pth.tar
#
# Output:
#   /scratch/kcwp264/data/surgi_world_track/ssl_corpus/Training/
#
# Submit:
#   sbatch jobs/build-ssl-corpus.slurm
# =============================================================================
#SBATCH --job-name=ssl-build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
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
cd "${REPO_ROOT}"
mkdir -p logs/ssl_build

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

echo "=== SSL Corpus Build (3-GPU parallel) ==="
echo "Node:   $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Check Stage 1 checkpoint
STAGE1_CKPT="outputs/mot/cholec20-stage1-surgenet/best.pth.tar"
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

bash scripts/got_jepa/build_ssl_corpus_3gpu.sh

echo ""
echo "=== SSL corpus build finished ==="
date
