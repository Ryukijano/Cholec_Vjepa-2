#!/bin/bash
#SBATCH --job-name=tdv-pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tdv-pretrain_%j.out
#SBATCH --error=logs/tdv-pretrain_%j.err

# ====================================================================
# TDV Pretraining on Cholec80 SSL Corpus (Stage 0)
# ====================================================================
# 3x L40S GPU DDP training, ~50k steps
# Output: outputs/tdv_pretrain/tdv_frame_encoder.pth
# ====================================================================

set -euo pipefail

echo "=== TDV Pretraining ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"

# -- Environment
module load cuda/12.6
export CUDA_VISIBLE_DEVICES=0,1,2

# Use the endofm-lv conda environment (has PyTorch 2.7+ and CUDA 12.6)
source /scratch/kcwp264/conda/etc/profile.d/conda.sh
conda activate endofm-lv
export LD_LIBRARY_PATH=/scratch/kcwp264/conda/envs/endofm-lv/lib:$LD_LIBRARY_PATH

# -- Project root
cd /scratch/kcwp264/Cholec_Vjepa-2

# -- WandB (optional)
export WANDB_MODE=online
# export WANDB_API_KEY=your_key_here

# -- DDP training with torchrun
echo "Launching DDP training with 3 GPUs..."
torchrun --nproc_per_node=3 \
    scripts/pretrain_tdv.py \
    --config configs/train_mot/dinov2/tdv-pretrain.yaml \
    --ddp

echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output: outputs/tdv_pretrain/"
