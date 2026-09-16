#!/bin/bash
#SBATCH --job-name=rfdetr-stage1-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rfdetr_stage1_v2_%j.out
#SBATCH --error=logs/rfdetr_stage1_v2_%j.err

set -euo pipefail

echo "=== RF-DETR Stage 1 v2 — Improved Training ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "GPUs:      $SLURM_GPUS_PER_NODE"
echo "Start:     $(date)"
echo ""

# Activate conda env
source /scratch/kcwp264/conda/etc/profile.d/conda.sh
conda activate endofm-lv

export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export XFORMERS_DISABLED=1

cd /scratch/kcwp264/Cholec_Vjepa-2

mkdir -p logs

# Step 1: Run copy-paste augmentation for rare classes
AUG_DIR="/scratch/kcwp264/data/surgi_world_track/cholec20_coco_augmented"

if [ ! -d "$AUG_DIR/train" ]; then
    echo "=== Step 1: Copy-paste augmentation ==="
    python scripts/got_jepa/copy_paste_augment.py \
        --source_dir /scratch/kcwp264/data/surgi_world_track/cholec20_coco/train \
        --output_dir /scratch/kcwp264/data/surgi_world_track/cholec20_coco_train_augmented \
        --target_per_class 3000 \
        --seed 42

    # Build augmented dataset directory
    mkdir -p "$AUG_DIR"
    cp -r /scratch/kcwp264/data/surgi_world_track/cholec20_coco/valid "$AUG_DIR/valid"
    mv /scratch/kcwp264/data/surgi_world_track/cholec20_coco_train_augmented "$AUG_DIR/train"
else
    echo "=== Step 1: Augmented dataset already exists, skipping ==="
fi

echo ""
echo "=== Step 2: Train RF-DETR Large with augmentation ==="

# Train with improved settings using augmented dataset
python scripts/got_jepa/train_rfdetr_stage1.py \
    --epochs 200 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --resolution 704 \
    --early_stopping \
    --early_stopping_patience 20 \
    --ddp

echo ""
echo "=== Training complete ==="
echo "End:       $(date)"
echo "Checkpoints: /scratch/kcwp264/Cholec_Vjepa-2/outputs/rfdetr_stage1_v2"
