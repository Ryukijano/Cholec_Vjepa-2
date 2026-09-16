#!/usr/bin/env bash
# =============================================================================
# RF-DETR Ablation Study on CholecTrack20
#
# Uses RF-DETR directly (DINOv2 + LW-DETR decoder) with architecture variants
# to study the contribution of key components:
#   1. Baseline (RFDETRSmall, default)
#   2. No denoising (group_detr=1)
#   3. Fewer queries (num_queries=50)
#   4. Fewer decoder layers (dec_layers=2)
#   5. No pretraining (from scratch)
#
# Usage (from GPU node):
#   bash scripts/got_jepa/run_rfdetr_ablations.sh [gpu_ids]
#
# Examples:
#   bash scripts/got_jepa/run_rfdetr_ablations.sh          # auto-assign GPUs
#   bash scripts/got_jepa/run_rfdetr_ablations.sh 0,1,2    # use specific GPUs
# =============================================================================
set -eo pipefail

# ------------------------------------------------------------------
# 1. Activate conda env
# ------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module load miniforge || true
fi

SURGI_ENV="endofm-lv"
if [ "${CONDA_DEFAULT_ENV:-}" != "${SURGI_ENV}" ]; then
  for conda_root in \
    "${HOME}/miniforge3" \
    "${HOME}/miniconda3" \
    "/opt/miniforge3" \
    "/opt/conda"; do
    if [ -f "${conda_root}/etc/profile.d/conda.sh" ]; then
      source "${conda_root}/etc/profile.d/conda.sh"
      break
    fi
  done
  conda activate "${SURGI_ENV}" || { echo "Could not activate ${SURGI_ENV}"; exit 1; }
fi

# ------------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------------
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1

DATASET_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

# GPU assignment
GPU_IDS="${1:-}"
if [ -z "${GPU_IDS}" ]; then
  # Auto-detect available GPUs
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
  GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
fi
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "  RF-DETR Ablation Study"
echo "  Dataset: ${DATASET_DIR}"
echo "  GPUs:    ${GPU_IDS} (${NUM_GPUS} available)"
echo "  Epochs:  ${EPOCHS}"
echo "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ------------------------------------------------------------------
# 3. Define ablation variants
#    Each variant: name|python_kwargs_for_constructor
# ------------------------------------------------------------------
VARIANTS=(
  "rfdetr-small-baseline|"
  "rfdetr-small-no-dn|group_detr=1"
  "rfdetr-small-50q|num_queries=50"
  "rfdetr-small-2layer|dec_layers=2"
  "rfdetr-small-no-pretrain|pretrain_weights=None"
)

# ------------------------------------------------------------------
# 4. Run variants — sequentially if 1 GPU, parallel if multiple
# ------------------------------------------------------------------
FAILED=0

if [ "${NUM_GPUS}" -eq 1 ]; then
  echo ""
  echo "  Single GPU mode: running ${#VARIANTS[@]} variants sequentially"
  echo ""

  for variant_spec in "${VARIANTS[@]}"; do
    NAME="${variant_spec%%|*}"
    KWARGS="${variant_spec#*|}"
    OUTPUT_DIR="outputs/mot/${NAME}"
    GPU_ID="${GPU_ARRAY[0]}"

    echo ">>> [$(date '+%H:%M:%S')] Starting ${NAME} on GPU ${GPU_ID}"
    echo "    Output: ${OUTPUT_DIR}"
    echo "    Constructor kwargs: ${KWARGS:-<default>}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<PYEOF
import os
os.environ["XFORMERS_DISABLED"] = "1"

from rfdetr import RFDETRSmall

KWARGS = "${KWARGS}"
OUTPUT_DIR = "${OUTPUT_DIR}"
DATASET_DIR = "${DATASET_DIR}"
EPOCHS = ${EPOCHS}
BATCH_SIZE = ${BATCH_SIZE}
GRAD_ACCUM = ${GRAD_ACCUM}

# Parse kwargs from string like "key1=val1,key2=val2"
kwargs = {}
if KWARGS:
    for pair in KWARGS.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v == "None":
            kwargs[k] = None
        elif v == "True":
            kwargs[k] = True
        elif v == "False":
            kwargs[k] = False
        elif v.isdigit():
            kwargs[k] = int(v)
        else:
            kwargs[k] = v

print(f"[{OUTPUT_DIR}] Constructor kwargs: {kwargs}")
model = RFDETRSmall(**kwargs)

print(f"[{OUTPUT_DIR}] Model config: dec_layers={model.model_config.dec_layers}, "
      f"num_queries={model.model_config.num_queries}, "
      f"group_detr={model.model_config.group_detr}, "
      f"pretrain={model.model_config.pretrain_weights}")

model.train(
    dataset_dir=DATASET_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM,
    lr=1e-4,
    lr_encoder=1.5e-5,
    output_dir=OUTPUT_DIR,
    use_ema=True,
    eval_interval=1,
)

print(f"[{OUTPUT_DIR}] Training complete!")
PYEOF

    if [ $? -eq 0 ]; then
      echo "  [DONE] ${NAME} — $(date '+%H:%M:%S')"
    else
      echo "  [FAILED] ${NAME} — $(date '+%H:%M:%S')"
      FAILED=$((FAILED + 1))
    fi
  done
else
  # Multi-GPU: launch in parallel
  PIDS=()
  GPU_IDX=0

  for variant_spec in "${VARIANTS[@]}"; do
    NAME="${variant_spec%%|*}"
    KWARGS="${variant_spec#*|}"
    OUTPUT_DIR="outputs/mot/${NAME}"
    GPU_ID="${GPU_ARRAY[$((GPU_IDX % NUM_GPUS))]}"
    GPU_IDX=$((GPU_IDX + 1))

    echo ""
    echo ">>> Launching ${NAME} on GPU ${GPU_ID}"
    echo "    Output: ${OUTPUT_DIR}"
    echo "    Constructor kwargs: ${KWARGS:-<default>}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<PYEOF &
import os
os.environ["XFORMERS_DISABLED"] = "1"

from rfdetr import RFDETRSmall

KWARGS = "${KWARGS}"
OUTPUT_DIR = "${OUTPUT_DIR}"
DATASET_DIR = "${DATASET_DIR}"
EPOCHS = ${EPOCHS}
BATCH_SIZE = ${BATCH_SIZE}
GRAD_ACCUM = ${GRAD_ACCUM}

kwargs = {}
if KWARGS:
    for pair in KWARGS.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v == "None":
            kwargs[k] = None
        elif v == "True":
            kwargs[k] = True
        elif v == "False":
            kwargs[k] = False
        elif v.isdigit():
            kwargs[k] = int(v)
        else:
            kwargs[k] = v

print(f"[{OUTPUT_DIR}] Constructor kwargs: {kwargs}")
model = RFDETRSmall(**kwargs)

print(f"[{OUTPUT_DIR}] Model config: dec_layers={model.model_config.dec_layers}, "
      f"num_queries={model.model_config.num_queries}, "
      f"group_detr={model.model_config.group_detr}, "
      f"pretrain={model.model_config.pretrain_weights}")

model.train(
    dataset_dir=DATASET_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM,
    lr=1e-4,
    lr_encoder=1.5e-5,
    output_dir=OUTPUT_DIR,
    use_ema=True,
    eval_interval=1,
)

print(f"[{OUTPUT_DIR}] Training complete!")
PYEOF

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
    sleep 30
  done

  echo ""
  echo "============================================================"
  echo "  All ${#VARIANTS[@]} variants launched. Waiting for completion..."
  echo "  PIDs: ${PIDS[*]}"
  echo "============================================================"

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "  [FAILED] ${VARIANTS[$i]%%|*} (PID ${PIDS[$i]})"
      FAILED=$((FAILED + 1))
    else
      echo "  [DONE] ${VARIANTS[$i]%%|*} (PID ${PIDS[$i]})"
    fi
  done
fi

echo ""
echo "============================================================"
echo "  Ablation study complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ${FAILED}/${#VARIANTS[@]} variants failed"
echo "  Outputs in outputs/mot/rfdetr-small-*"
echo "============================================================"
exit ${FAILED}
