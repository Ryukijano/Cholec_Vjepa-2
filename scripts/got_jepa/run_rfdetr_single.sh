#!/usr/bin/env bash
# =============================================================================
# RF-DETR Single Variant Training on 1 GPU
# Runs one ablation variant on a single GPU with CSV logger fix.
#
# Usage: bash scripts/got_jepa/run_rfdetr_single.sh <variant_name> <kwargs>
# Example: bash scripts/got_jepa/run_rfdetr_single.sh rfdetr-small-no-pretrain "pretrain_weights=None"
# =============================================================================
set -eo pipefail

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

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export XFORMERS_DISABLED=1

NAME="${1:?Usage: run_rfdetr_single.sh <name> <kwargs>}"
KWARGS="${2:-}"
OUTPUT_DIR="outputs/mot/${NAME}"
DATASET_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

# Clean old output
if [ -d "${OUTPUT_DIR}" ]; then
  echo "Cleaning old ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi

echo ">>> [$(date '+%H:%M:%S')] Starting ${NAME} on 1 GPU"
echo "    Output: ${OUTPUT_DIR}"
echo "    Kwargs: ${KWARGS:-<default>}"
echo "    Epochs: ${EPOCHS}, batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}"

python -c "
import os
os.environ['XFORMERS_DISABLED'] = '1'

# Monkey-patch Lightning CSV logger to fix fieldnames crash
import csv as _csv
import lightning_fabric.loggers.csv_logs as _csv_logs

def _patched_rewrite(self, fieldnames):
    with self._fs.open(self.metrics_file_path, 'r', newline='') as f:
        metrics = list(_csv.DictReader(f))
    with self._fs.open(self.metrics_file_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
        w.writeheader()
        w.writerows(metrics)

_csv_logs._ExperimentWriter._rewrite_with_new_header = _patched_rewrite

from rfdetr import RFDETRSmall

KWARGS_STR = '${KWARGS}'
kwargs = {}
if KWARGS_STR.strip():
    for pair in KWARGS_STR.split(','):
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

print(f'[${NAME}] kwargs={kwargs}')
model = RFDETRSmall(**kwargs)
print(f'[${NAME}] dec_layers={model.model_config.dec_layers} queries={model.model_config.num_queries} group_detr={model.model_config.group_detr} pretrain={model.model_config.pretrain_weights}')

model.train(
    dataset_dir='${DATASET_DIR}',
    epochs=${EPOCHS},
    batch_size=${BATCH_SIZE},
    grad_accum_steps=${GRAD_ACCUM},
    lr=1e-4,
    lr_encoder=1.5e-5,
    output_dir='${OUTPUT_DIR}',
    use_ema=True,
    eval_interval=1,
)
print(f'[${NAME}] DONE')
" 2>&1 | tee "${OUTPUT_DIR}.log"

echo "  [DONE] ${NAME} — $(date '+%H:%M:%S')"
