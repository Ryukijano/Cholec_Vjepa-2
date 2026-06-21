#!/bin/bash
# Build SSL corpus in parallel across 3 GPUs.
# Each GPU gets ~24-25 Cholec80 videos to pseudo-label.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

module load miniforge 2>/dev/null || true
conda activate surgi_world_track_cuda 2>/dev/null || true

OUT_ROOT="/scratch/kcwp264/data/surgi_world_track/ssl_corpus"
STAGE1_CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
STAGE1_CKPT="outputs/mot/cholec20-stage1-supervised/best.pth.tar"

LOG_DIR="$REPO_ROOT/logs/ssl_build"
mkdir -p "$LOG_DIR"

echo "=== Launching 3 parallel SSL corpus builders ==="
echo "  GPU 0 → cuda:0  (log: $LOG_DIR/rank0.log)"
echo "  GPU 1 → cuda:1  (log: $LOG_DIR/rank1.log)"
echo "  GPU 2 → cuda:2  (log: $LOG_DIR/rank2.log)"
echo ""

# Launch 3 background jobs
for RANK in 0 1 2; do
  PYTHONPATH=. python -m scripts.build_ssl_corpus \
    --stage1_config "$STAGE1_CONFIG" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --out_root "$OUT_ROOT" \
    --device cuda \
    --score_threshold 0.5 \
    --rank "$RANK" \
    --world_size 3 \
    > "$LOG_DIR/rank${RANK}.log" 2>&1 &
  PIDS[$RANK]=$!
done

echo "PIDs: ${PIDS[0]} ${PIDS[1]} ${PIDS[2]}"
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
else
  echo "ERROR: One or more ranks failed. Check logs."
  exit 1
fi
