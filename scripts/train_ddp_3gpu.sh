#!/usr/bin/env bash
# =============================================================================
# Launch Stage 2 JEPA training on 3× L40S GPUs via torchrun DDP.
#
# Usage:
#   module load miniforge
#   bash scripts/train_ddp_3gpu.sh
#
# This script uses the wandb-enabled config. Adjust --nproc_per_node if you
# have a different number of GPUs.
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------
# 1. Activate CUDA-capable conda env (surgi_world_track_cuda)
# ------------------------------------------------------------------
module load miniforge || true
conda activate surgi_world_track_cuda

CONFIG="configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain-wandb.yaml"
GPUS=3

echo "Starting DDP training on ${GPUS} GPUs..."
echo "Config: ${CONFIG}"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"

# Optional: log in to W&B first if not already authenticated
# wandb login

cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN

torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda
