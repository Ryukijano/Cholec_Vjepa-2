#!/bin/bash
# =============================================================================
# SLURM: Build SSL corpus using new Deformable DETR Stage-1 checkpoint.
# 3x NVIDIA L40S on a single AIRE GPU node, each GPU pseudo-labels ~25 videos.
# =============================================================================
#SBATCH --job-name=ssl-corpus
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ssl_corpus_%j.out
#SBATCH --error=logs/ssl_corpus_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
module load miniforge/24.7.1 || true
module load cuda/12.6.2 || true

source activate surgi_world_track_cuda

# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/users/kcwp264/TRACK_JEPA/surgi_world_track"
cd "${REPO_ROOT}"

OUT_ROOT="/scratch/kcwp264/data/surgi_world_track/ssl_corpus"
STAGE1_CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
STAGE1_CKPT="outputs/mot/cholec20-stage1-supervised/best.pth.tar"

LOG_DIR="${REPO_ROOT}/logs/ssl_build"
mkdir -p "$LOG_DIR"
mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------------------
# 3. Sanity checks
# ---------------------------------------------------------------------------
echo "=== SSL Corpus Build ==="
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-n/a}"
echo "Checkpoint: ${STAGE1_CKPT}"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-3} x L40S"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f "$STAGE1_CKPT" ]; then
  echo "ERROR: Checkpoint not found: $STAGE1_CKPT"
  exit 1
fi

date

# ---------------------------------------------------------------------------
# 4. Launch 3 parallel builders (one per GPU)
# ---------------------------------------------------------------------------
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo ""
echo "Launching 3 parallel SSL corpus builders..."
for RANK in 0 1 2; do
  python -m scripts.build_ssl_corpus \
    --stage1_config "$STAGE1_CONFIG" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --out_root "$OUT_ROOT" \
    --device cuda \
    --score_threshold 0.5 \
    --rank "$RANK" \
    --world_size 3 \
    > "$LOG_DIR/rank${RANK}_slurm${SLURM_JOB_ID:-0}.log" 2>&1 &
  PIDS[$RANK]=$!
  echo "  Rank ${RANK} → PID ${PIDS[$RANK]}"
done

# ---------------------------------------------------------------------------
# 5. Wait and report
# ---------------------------------------------------------------------------
wait ${PIDS[0]}; EC0=$?
wait ${PIDS[1]}; EC1=$?
wait ${PIDS[2]}; EC2=$?

echo ""
echo "=== Done ==="
echo "Rank 0 exit code: $EC0"
echo "Rank 1 exit code: $EC1"
echo "Rank 2 exit code: $EC2"
date

if [ $EC0 -eq 0 ] && [ $EC1 -eq 0 ] && [ $EC2 -eq 0 ]; then
  echo "All ranks succeeded. SSL corpus at $OUT_ROOT"
else
  echo "ERROR: One or more ranks failed. Check logs in $LOG_DIR/rank*"
  exit 1
fi
