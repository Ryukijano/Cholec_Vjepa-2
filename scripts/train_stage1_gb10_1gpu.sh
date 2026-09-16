#!/usr/bin/env bash
# Stage 1 supervised training on CholecTrack20 — single GPU (e.g. NVIDIA GB10 / DGX Spark).
#
# Usage (from repo root):
#   conda activate surgi_track
#   bash scripts/train_stage1_gb10_1gpu.sh
#
# Smoke test (~32 clips, 1 epoch worth of data):
#   STAGE1_SMOKE=1 bash scripts/train_stage1_gb10_1gpu.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
# GB10/Blackwell: disable xFormers for DINOv2 hub (no sm_120 kernels in current xformers).
export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export STAGE1_GPUS=1

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate surgi_track 2>/dev/null || true
fi

CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml"
EXTRA_ARGS=()

if [ "${STAGE1_SMOKE:-0}" = "1" ]; then
  echo "=== Stage 1 smoke test (32 clips, 2 epochs, single GPU) ==="
  CONFIG="configs/train_mot/dinov2/cholec20-mot-stage1-smoke.yaml"
  EXTRA_ARGS+=(--debugmode True)
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

echo "Data root: ${MOT_DATA_ROOT:-data/cholectrack20 (from config)}"
echo "Config: ${CONFIG}"

RESUME_ARGS=()
if [ -n "${STAGE1_RESUME:-}" ]; then
  RESUME_ARGS+=(--resume "${STAGE1_RESUME}")
  if [ "${STAGE1_RESET_OPTIM:-1}" = "1" ]; then
    RESUME_ARGS+=(--reset-optimizer)
  fi
  if [ -n "${STAGE1_START_EPOCH:-}" ]; then
    RESUME_ARGS+=(--start-epoch "${STAGE1_START_EPOCH}")
  fi
  echo "Resume: ${STAGE1_RESUME} ${RESUME_ARGS[*]}"
fi

python -c "import torch; print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda:0 \
  "${EXTRA_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
