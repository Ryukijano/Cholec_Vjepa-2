#!/bin/bash
#SBATCH --job-name=stage3-joint-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/kcwp264/logs/stage3_joint_%j.out
#SBATCH --error=/scratch/kcwp264/logs/stage3_joint_%j.err

set -euo pipefail

echo "=== Stage 3 Joint Fine-tune (SHM-cached) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "Start: $(date)"

# --- Use conda env Python directly (conda activate not available in batch) ---
ENDOFM_PY="/scratch/kcwp264/conda/envs/endofm-lv/bin/python"
ENDOFM_TORCHRUN="/scratch/kcwp264/conda/envs/endofm-lv/bin/torchrun"
export PATH="/scratch/kcwp264/conda/envs/endofm-lv/bin:${PATH}"
export LD_LIBRARY_PATH="/scratch/kcwp264/conda/envs/endofm-lv/lib:${LD_LIBRARY_PATH:-}"
echo "Python: ${ENDOFM_PY}"
${ENDOFM_PY} --version

cd /scratch/kcwp264/Cholec_Vjepa-2

# --- NCCL env vars for L40S PCIe ---
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export XFORMERS_DISABLED=1

# --- Copy dataset to /dev/shm (RAM-backed tmpfs, ~94GB) ---
SRC_DATA="/scratch/kcwp264/datasets_cholec/cholectrack20"
SHM_DATA="/dev/shm/cholectrack20"

if [ -d "${SHM_DATA}" ] && [ -f "${SHM_DATA}/SYNAPSE_METADATA_MANIFEST.tsv" ]; then
  echo ">>> /dev/shm already has cholectrack20, skipping copy"
else
  echo ">>> Copying CholecTrack20 to /dev/shm..."
  echo "    /dev/shm free before: $(df -h /dev/shm | tail -1 | awk '{print $4}')"
  cp -r "${SRC_DATA}" "${SHM_DATA}"
  echo "    Copy complete. /dev/shm free after: $(df -h /dev/shm | tail -1 | awk '{print $4}')"
fi

export MOT_DATA_ROOT="${SHM_DATA}"

# --- Verify checkpoints ---
STAGE1_CKPT="outputs/mot/cholec20-stage1-surgenet/best.pth.tar"
STAGE2_CKPT="outputs/mot/cholec80-ct20-stage2-jepa-surgenet/latest.pth.tar"
STAGE3_OUT="outputs/mot/cholec20-mot-stage3-joint"
echo ">>> Checking checkpoints..."
ls -lh "${STAGE1_CKPT}" || { echo "ERROR: Stage 1 checkpoint missing"; exit 1; }
ls -lh "${STAGE2_CKPT}" || { echo "ERROR: Stage 2 checkpoint missing"; exit 1; }

# --- Clean up old broken Stage 3 output ---
if [ -d "${STAGE3_OUT}" ]; then
  echo ">>> Backing up old Stage 3 output to ${STAGE3_OUT}_broken_$(date +%s)"
  mv "${STAGE3_OUT}" "${STAGE3_OUT}_broken_$(date +%s)"
fi

# --- Launch Stage 3 DDP training ---
NUM_GPUS=$(${ENDOFM_PY} -c "import torch; print(torch.cuda.device_count())")
CONFIG="configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml"

echo ""
echo "============================================================"
echo "  Stage 3 Joint Fine-tune — ${NUM_GPUS}-GPU DDP"
echo "  Config:  ${CONFIG}"
echo "  Data:    ${SHM_DATA} (RAM-backed tmpfs)"
echo "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

${ENDOFM_TORCHRUN} --standalone --nproc_per_node="${NUM_GPUS}" \
  -m core_app.mot.main \
  --fname "${CONFIG}" \
  --devices cuda

echo ""
echo "============================================================"
echo "  Stage 3 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- Cleanup shm ---
echo ">>> Cleaning up /dev/shm..."
rm -rf "${SHM_DATA}"

echo "End: $(date)"
