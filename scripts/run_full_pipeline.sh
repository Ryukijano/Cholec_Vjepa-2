#!/usr/bin/env bash
# =============================================================================
# Full pipeline: Stage 1 DDP → Build SSL Corpus → Stage 2 DDP
#
# Usage:
#   module load miniforge
#   bash scripts/run_full_pipeline.sh
# =============================================================================
set -euo pipefail

module load miniforge || true
conda activate surgi_world_track_cuda

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN

GPUS=3
STAGE1_CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
STAGE2_CONFIG="configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain-wandb.yaml"
STAGE1_CKPT="outputs/mot/cholec20-stage1-supervised/best.pth.tar"
SSL_CORPUS="/scratch/kcwp264/data/surgi_world_track/ssl_corpus"

echo "========================================"
echo "STEP 1: Stage 1 Supervised DDP (${GPUS} GPUs)"
echo "========================================"
torchrun \
  --standalone --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${STAGE1_CONFIG}" \
  --devices cuda

echo ""
echo "========================================"
echo "STEP 2: Build SSL Corpus"
echo "========================================"
python -m scripts.build_ssl_corpus \
  --stage1_config "${STAGE1_CONFIG}" \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --out_root "${SSL_CORPUS}"

echo ""
echo "========================================"
echo "STEP 3: Stage 2 JEPA DDP (${GPUS} GPUs)"
echo "========================================"
torchrun \
  --standalone --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${STAGE2_CONFIG}" \
  --devices cuda

echo ""
echo "Pipeline complete!"
