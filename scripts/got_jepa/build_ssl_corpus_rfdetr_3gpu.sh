#!/bin/bash
# Build SSL corpus in parallel across 3 GPUs using RF-DETR pseudo-labels.
# Each GPU gets ~24 Cholec80 videos to pseudo-label.
#
# Usage: bash scripts/got_jepa/build_ssl_corpus_rfdetr_3gpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

module load miniforge 2>/dev/null || true
module load cuda/12.6.2 2>/dev/null || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/apps/pkg/interpreters/miniforge/24.7.1/bin")
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /scratch/kcwp264/conda/envs/endofm-lv
set -u
export CONDA_DEFAULT_ENV=endofm-lv

CKPT="/scratch/kcwp264/Cholec_Vjepa-2/outputs/rfdetr_stage1/checkpoint_best_ema.pth"
OUT_ROOT="/scratch/kcwp264/data/surgi_world_track/ssl_corpus"
LOG_DIR="$REPO_ROOT/logs/ssl_build_rfdetr"
mkdir -p "$LOG_DIR"

# Clear old SSL corpus Training dir (keep Validation symlink)
if [ -d "$OUT_ROOT/Training" ]; then
    echo "Clearing old SSL corpus Training directory..."
    rm -rf "$OUT_ROOT/Training"
fi
mkdir -p "$OUT_ROOT/Training"

echo "=== Launching 3 parallel RF-DETR SSL corpus builders ==="
echo "  Checkpoint: $CKPT"
echo "  Output:     $OUT_ROOT"
echo "  GPU 0 → cuda:0  (log: $LOG_DIR/rank0.log)"
echo "  GPU 1 → cuda:1  (log: $LOG_DIR/rank1.log)"
echo "  GPU 2 → cuda:2  (log: $LOG_DIR/rank2.log)"
echo ""

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1

# Launch 3 background jobs
PIDS=()
for RANK in 0 1 2; do
  python scripts/got_jepa/build_ssl_corpus_rfdetr.py \
    --checkpoint "$CKPT" \
    --out_root "$OUT_ROOT" \
    --device "cuda:${RANK}" \
    --score_threshold 0.25 \
    --batch_size 8 \
    --rank "$RANK" \
    --world_size 3 \
    > "$LOG_DIR/rank${RANK}.log" 2>&1 &
  PIDS[$RANK]=$!
  echo "  Rank $RANK PID: ${PIDS[$RANK]}"
done

echo ""
echo "Monitor progress: tail -f $LOG_DIR/rank*.log"
echo "Wait for completion: wait"

# Wait for all and report exit codes
wait ${PIDS[0]}; EC0=$?
wait ${PIDS[1]}; EC1=$?
wait ${PIDS[2]}; EC2=$?

echo ""
echo "=== Done ==="
echo "Rank 0 exit code: $EC0"
echo "Rank 1 exit code: $EC1"
echo "Rank 2 exit code: $EC2"

if [ $EC0 -eq 0 ] && [ $EC1 -eq 0 ] && [ $EC2 -eq 0 ]; then
  echo "All ranks succeeded. SSL corpus at $OUT_ROOT"
  echo ""
  echo "Video count:"
  ls "$OUT_ROOT/Training/" | wc -l
else
  echo "ERROR: One or more ranks failed. Check logs in $LOG_DIR"
  exit 1
fi
