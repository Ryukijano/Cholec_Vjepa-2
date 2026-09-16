#!/usr/bin/env bash
# =============================================================================
# RF-DETR Ablation Study on CholecTrack20 — 3-GPU Parallel Mode
#
# Runs 3 variants in parallel (one per GPU), then the 4th after a slot frees.
# No DDP/NCCL needed — each variant trains on a single GPU.
# This maximises GPU utilisation without NCCL segfaults on PCIe L40S.
#
# Usage (from GPU node):
#   bash scripts/got_jepa/run_rfdetr_ablations_multigpu.sh
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
BATCH_SIZE="${BATCH_SIZE:-4}"        # per-GPU batch size
GRAD_ACCUM="${GRAD_ACCUM:-4}"        # grad accum to reach effective batch ~16

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "1")
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo "============================================================"
echo "  RF-DETR Ablation Study — 3-GPU Parallel"
echo "  Dataset:       ${DATASET_DIR}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Per-GPU batch: ${BATCH_SIZE}"
echo "  Grad accum:    ${GRAD_ACCUM}"
echo "  Effective batch per variant: ${EFFECTIVE_BATCH}"
echo "  Epochs:        ${EPOCHS}"
echo "  Start:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ------------------------------------------------------------------
# 3. Define ablation variants
#    Each variant: name|python_kwargs_for_constructor
# ------------------------------------------------------------------
# Each variant: name|constructor_kwargs|train_kwargs
# Note: no-pretrain runs separately on the 1-GPU node
VARIANTS=(
  "rfdetr-small-no-dn|group_detr=1|"
  "rfdetr-small-50q|num_queries=50|"
  # Extra variants for deeper ablation (run after first 2 finish)
  "rfdetr-small-no-ema||use_ema=False"
  "rfdetr-small-highlr||lr=2e-4,lr_encoder=3e-5"
  "rfdetr-small-560res||resolution=560"
)

# ------------------------------------------------------------------
# 4. Launch variants in parallel — one per GPU
#    6 variants, 3 GPUs: first 3 run in parallel, next 3 after
# ------------------------------------------------------------------

run_variant() {
  local GPU_ID=$1
  local NAME=$2
  local CKWARGS=$3   # constructor kwargs
  local TKWARGS=$4   # train kwargs
  local OUTPUT_DIR="outputs/mot/${NAME}"

  echo ">>> [$(date '+%H:%M:%S')] Starting ${NAME} on GPU ${GPU_ID}"

  # Clean old output dir to avoid CSV logger crash (Lightning bug #19432)
  if [ -d "${OUTPUT_DIR}" ]; then
    echo "    Cleaning old ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  fi

  local TRAIN_SCRIPT=$(mktemp /tmp/rfdetr_train_XXXXXX.py)
  cat > "${TRAIN_SCRIPT}" <<PYEOF
import os
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '${GPU_ID}'

# Monkey-patch Lightning CSV logger to fix "dict contains fields not in fieldnames" bug
# See: https://github.com/Lightning-AI/pytorch-lightning/issues/19432
import csv as _csv
import lightning_fabric.loggers.csv_logs as _csv_logs

def _patched_rewrite_with_new_header(self, fieldnames):
    with self._fs.open(self.metrics_file_path, "r", newline="") as file:
        metrics = list(_csv.DictReader(file))
    with self._fs.open(self.metrics_file_path, "w", newline="") as file:
        writer = _csv.DictWriter(file, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)

_csv_logs._ExperimentWriter._rewrite_with_new_header = _patched_rewrite_with_new_header

from rfdetr import RFDETRSmall

CKWARGS_STR = "${CKWARGS}"
TKWARGS_STR = "${TKWARGS}"
OUTPUT_DIR = "${OUTPUT_DIR}"
DATASET_DIR = "${DATASET_DIR}"
EPOCHS = ${EPOCHS}
BATCH_SIZE = ${BATCH_SIZE}
GRAD_ACCUM = ${GRAD_ACCUM}
NAME = "${NAME}"

def parse_kwargs(s):
    kwargs = {}
    if s.strip():
        for pair in s.split(','):
            pair = pair.strip()
            if not pair: continue
            k, v = pair.split('=', 1)
            k = k.strip(); v = v.strip()
            if v == 'None': kwargs[k] = None
            elif v == 'True': kwargs[k] = True
            elif v == 'False': kwargs[k] = False
            elif v.isdigit(): kwargs[k] = int(v)
            else:
                try: kwargs[k] = float(v)
                except: kwargs[k] = v
    return kwargs

ckwargs = parse_kwargs(CKWARGS_STR)
tkwargs = parse_kwargs(TKWARGS_STR)

print(f'[{NAME}] constructor_kwargs={ckwargs}')
print(f'[{NAME}] train_kwargs={tkwargs}')
model = RFDETRSmall(**ckwargs)
print(f'[{NAME}] dec_layers={model.model_config.dec_layers} '
      f'queries={model.model_config.num_queries} '
      f'group_detr={model.model_config.group_detr} '
      f'pretrain={model.model_config.pretrain_weights}')

# Default train args, overridden by tkwargs
train_args = dict(
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
train_args.update(tkwargs)
print(f'[{NAME}] final train args: {train_args}')

model.train(**train_args)
print(f'[{NAME}] DONE')
PYEOF

  python "${TRAIN_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}.log"
  local rc=$?
  rm -f "${TRAIN_SCRIPT}"

  echo "  [DONE] ${NAME} on GPU ${GPU_ID} — $(date '+%H:%M:%S')"
  return ${rc}
}

# Parse variant spec: name|constructor_kwargs|train_kwargs
parse_variant() {
  local spec=$1
  NAME=$(echo "$spec" | cut -d'|' -f1)
  CKWARGS=$(echo "$spec" | cut -d'|' -f2)
  TKWARGS=$(echo "$spec" | cut -d'|' -f3)
}

# Launch first batch of 3 variants in parallel (one per GPU)
declare -a PIDS=()
BATCH_SIZE_LIMIT=${NUM_GPUS}
if [ ${#VARIANTS[@]} -lt ${BATCH_SIZE_LIMIT} ]; then
  BATCH_SIZE_LIMIT=${#VARIANTS[@]}
fi

for i in $(seq 0 $((BATCH_SIZE_LIMIT - 1))); do
  parse_variant "${VARIANTS[$i]}"
  run_variant $i "${NAME}" "${CKWARGS}" "${TKWARGS}" &
  PIDS+=($!)
done

# Wait for first batch to finish
FAILED=0
for pid in "${PIDS[@]}"; do
  wait $pid || FAILED=$((FAILED + 1))
done

# Launch remaining variants on freed GPUs, 3 at a time
REMAINING_IDX=${BATCH_SIZE_LIMIT}
while [ ${REMAINING_IDX} -lt ${#VARIANTS[@]} ]; do
  PIDS=()
  for i in $(seq 0 $((BATCH_SIZE_LIMIT - 1))); do
    idx=$((REMAINING_IDX + i))
    if [ ${idx} -lt ${#VARIANTS[@]} ]; then
      parse_variant "${VARIANTS[$idx]}"
      run_variant $i "${NAME}" "${CKWARGS}" "${TKWARGS}" &
      PIDS+=($!)
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
  done
  REMAINING_IDX=$((REMAINING_IDX + BATCH_SIZE_LIMIT))
done

echo ""
echo "============================================================"
echo "  Ablation study complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ${FAILED}/${#VARIANTS[@]} variants failed"
echo "  Outputs in outputs/mot/rfdetr-small-*"
echo "============================================================"
exit ${FAILED}
