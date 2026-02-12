"""
Windows Migration Guide for V-JEPA2 Surgical Tool Detection
============================================================

Your Hardware (Excellent for this task):
- RTX 4090 (24GB VRAM) - Perfect, training uses ~16GB
- 64GB RAM - Plenty
- i9-13900K - Fast CPU for data loading

Step 1: Install Prerequisites on Windows
---------------------------------------

1. Install Python 3.11:
   https://www.python.org/downloads/release/python-3117/
   - Check "Add Python to PATH" during install

2. Install Git:
   https://git-scm.com/download/win

3. Install CUDA 12.1:
   https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows
   - Select "Express Installation"

4. Verify CUDA works:
   Open PowerShell:
   ```powershell
   nvidia-smi
   ```
   Should show RTX 4090 and CUDA 12.1

Step 2: Download This Code
--------------------------

```powershell
# Create workspace
cd C:\
mkdir vjepa2-project
cd vjepa2-project

# Download from this cloud instance (see Step 5 for export)
# Or clone from your git repo if you have one
```

Step 3: Install Python Dependencies
-----------------------------------

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements-windows.txt
```

Step 4: Download Checkpoints
----------------------------

From this cloud instance, download:
1. `outputs/detection-hardened-v2/best.pt` (35% recall checkpoint)
2. `outputs/vjepa2-cholec-pretrain/latest.pt` (SSL pretrain)

Place them in your local `outputs/` directory

Step 5: Download CholecTrack20 Dataset
--------------------------------------

Download from: https://github.com/CAMMA-public/cholectrack20

Expected structure:
```
cholec_dataset/
├── Training/
│   ├── VID02/
│   │   ├── Frames/
│   │   │   ├── 100001.png
│   │   │   └── ...
│   │   └── VID02.json
│   └── ... (10 videos)
├── Validation/
│   └── ... (2 videos)
└── Test/
    └── ... (8 videos)
```

Step 6: Run Training
-------------------

Phase 1: Detection (Optional - you already have best.pt)
```powershell
python cholectrack_vjepa2_training/train_detection.py `
    --checkpoint outputs/vjepa2-cholec-pretrain/latest.pt `
    --train_dir cholec_dataset/Training `
    --val_dir cholec_dataset/Validation `
    --out_dir outputs/detection-hardened-v2 `
    --epochs 8 --batch_size 8 --use_wandb
```

Phase 2: Re-ID (Run this now)
------------------------------
```powershell
python cholectrack_vjepa2_training/train_reid.py `
    --detection_checkpoint outputs/detection-hardened-v2/best.pt `
    --out_dir outputs/reid-phase2 `
    --epochs 20 --batch_size 8 --loss_type both --use_wandb
```

Expected Training Time on 4090:
- Phase 2 Re-ID: ~8-10 hours (vs 13 hours on cloud L40S)
- Epoch 1: ~40 minutes

Monitoring:
- TensorBoard: tensorboard --logdir outputs/reid-phase2/tb
- WandB: https://wandb.ai (if you use --use_wandb)

Windows-Specific Notes:
----------------------
1. Use PowerShell or CMD (not WSL)
2. Paths use backslashes (\) or forward slashes (/) - both work with Pathlib
3. If you get "CUDA out of memory", reduce batch_size to 4
4. Disable `persistent_workers` in DataLoader if you get multiprocessing errors
5. num_workers=4 is safer on Windows (not 8)

Troubleshooting:
---------------
- "ModuleNotFoundError: No module named 'vjepa2'": 
  Make sure you're in the root directory (C:\vjepa2-project)
  
- "CUDA not available":
  Reinstall PyTorch with CUDA: 
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  
- "Access denied":
  Run PowerShell as Administrator

Performance Tips:
-----------------
- Your 4090 is ~30% faster than L40S for this workload
- Training will use ~16GB VRAM, leaving 8GB free
- 64GB RAM means you can increase num_workers for faster data loading
- NVMe SSD recommended for dataset (random frame access)
"""
