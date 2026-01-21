# V-JEPA 2 Training Setup - CholecTrack20 Dataset

## Status: Ready for Model Integration ✓

### Current Progress (CholecTrack20 + V-JEPA2)
- **Frozen encoder:** `outputs/vjepa2-cholec-pretrain/latest.pt` (ViT-L/16, tubelet=2, crop=224).
- **Detection probe:** `scripts/train_detection_probe.py` (DETR-style, CE + L1 + GIoU, Hungarian matching). Training in progress -> target output `outputs/detect_probe_roi/best_detect_probe.pt`.
- **Tracker:** `scripts/track_cholec.py` (greedy IOU; drop-in ByteTrack/Bot-SORT possible after detector is ready).
- **Probes/visuals:** t-SNE scripts and attentive/linear probes available; embeddings show moderate separation but need a trained head.

**t-SNE quick read (validation embeddings):** Tool 3 clusters somewhat, but tool 0 vs 2 are mixed → backbone alone isn’t linearly separable; expect a trained head (attentive/detection) to improve. Use linear probe accuracy for a quantitative check.

### How to train the detection probe
```bash
python scripts/train_detection_probe.py \
  --checkpoint outputs/vjepa2-cholec-pretrain/latest.pt \
  --train_dir cholec_dataset/Training \
  --val_dir cholec_dataset/Validation \
  --out_dir outputs/detect_probe_roi \
  --batch_size 128 --epochs 20 --lr 1e-4
```
Notes: Uses AMP + GradScaler. Saves best model by proxy recall@0.5. Loss = CE + 5*L1 + 2*GIoU.

### How to run the tracker (after detector is trained)
```bash
python scripts/track_cholec.py \
  --checkpoint outputs/vjepa2-cholec-pretrain/latest.pt \
  --head_path outputs/detect_probe_roi/best_detect_probe.pt \
  --video_dir cholec_dataset/Validation/VID30
```
Outputs: `outputs/tracking_results/VID30/tracking.json` (track IDs + boxes + classes). Replace `video_dir` as needed.

### Pipeline Architecture (summary)
- **Input:** Single-frame clips from CholecTrack20 (PNG), JSON annotations (instrument id, tlwh bbox) for Training/Validation.
- **Frozen backbone:** V-JEPA2 ViT-L/16 (tubelet=2, crop=224) → token embeddings.
- **Detection head (probe):** Multi-query attentive head (DETR-style) with learnable queries → class logits + cxcywh boxes.
- **Matching + losses:** Hungarian matching on class + L1 + GIoU. Loss = CE + 5×L1 + 2×GIoU. No-object class handled explicitly.
- **Validation proxy:** Recall@0.5 IoU (naive matching) to pick best checkpoint (`best_detect_probe.pt`).
- **Tracking:** `track_cholec.py` loads frozen encoder + trained head, runs per-frame inference, greedy IOU association to maintain track IDs → `tracking.json` per video.
- **Next eval (planned):** Use TrackEval (HOTA/MOTA/IDF1) on Validation/Test once detector is stable.

#### Mermaid Pipeline
```mermaid
flowchart LR
    A[Frames (PNG) + JSON tlwh bboxes] --> B[Frozen V-JEPA2
    ViT-L/16 encoder]
    B --> C[Detection head (DETR-style
    multi-query)]
    C --> D[Hungarian matching
    CE + 5×L1 + 2×GIoU]
    D --> E[Best checkpoint
    best_detect_probe.pt]
    E --> F[track_cholec.py
    per-frame inference]
    F --> G[Greedy IOU association
    tracking.json]
    G -. planned .-> H[TrackEval
    HOTA/MOTA/IDF1]
```

### Dataset Configuration

**Dataset Location:** `/teamspace/studios/this_studio/cholec_dataset`

**Dataset Structure:**
```
cholec_dataset/
├── Training/   (10 videos - VID folders with frames)
├── Testing/    (4 videos)
└── Validation/ (2 videos)
```

**Total Videos:** 16 surgical videos with extracted frames

### Files Created

1. **cholec_dataset_loader_fullssl.py** ✓
   - Full dataset loader for self-supervised learning
   - Uses ALL splits (Training + Testing + Validation) for SSL
   - Loads 16 frames per sequence at 224x224 resolution
   - Ready to use with V-JEPA 2

2. **start_training.py** ✓
   - Dataset verification script  
   - Tests data loading pipeline
   - Confirms all videos are accessible

3. **download_cholectrack20.py** ✓
   - Synapse download script (not needed - dataset already present)
   - Kept for reference

### Dataset Loader Usage

The dataset loader is already configured for your dataset structure:

```python
from cholec_dataset_loader_fullssl import CholecTrack20Dataset

# Create dataset (uses all splits for SSL)
dataset = CholecTrack20Dataset(
    data_root="/teamspace/studios/this_studio/cholec_dataset",
    split="",  # Empty = use all splits for SSL
    num_frames=16,
    frame_size=224
)

# Use with DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=16,
    pin_memory=True
)
```

### Next Steps to Start Training

#### Option 1: Use Existing Training Files

The `cholectrack_vjepa2_training/` directory contains:
- `config_h200_ssl.yaml` - H200-optimized configuration
- `dataloader_h200.py` - High-throughput dataloader
- `train_h200.py` - Complete training script

To use these:
1. Update the config to point to your dataset path
2. Run: `cd cholectrack_vjepa2_training && python train_h200.py`

#### Option 2: Integrate with V-JEPA 2 Repository

If you have the V-JEPA 2 model code:

1. **Clone V-JEPA 2 repo** (if not already present):
   ```bash
   git clone https://github.com/facebookresearch/vjepa2.git
   cd vjepa2
   ```

2. **Update their config** to use CholecTrack20:
   - Point `data.root` to `/teamspace/studios/this_studio/cholec_dataset`
   - Use the `cholec_dataset_loader_fullssl.py` as custom dataset

3. **Start training**:
   ```bash
   python -m torch.distributed.launch \\
       --nproc_per_node=1 \\
       main_vjepa.py \\
       --config configs/cholec_ssl.yaml
   ```

### Hardware: H200 GPU

**Current Hardware:** 1x NVIDIA H200 (141GB HBM3e)

**Recommended Settings:**
- Batch size: 32-64 (depending on model size)
- Workers: 16
- Mixed precision: BF16
- torch.compile: enabled

**Expected Training Time:**
- ~8-10 hours for 50 epochs (ViT-Large)
- ~15-20 hours for 100 epochs (ViT-Huge)

### Troubleshooting

**If dataset loader fails:**
```bash
python start_training.py
```
This will verify:
- Dataset path is correct
- All video folders are accessible
- Frame loading works

**If you need to check a specific video:**
```bash
ls /teamspace/studios/this_studio/cholec_dataset/Training/VID01/
```

### Training Checklist

- [x] Dataset downloaded and extracted
- [x] Dataset structure verified
- [x] Dataset loader created
- [x] H200 optimization configs ready
- [ ] V-JEPA 2 model code integrated
- [ ] Training config updated
- [ ] First training run started

### Important Notes

1. **Dataset Path:** The path `/teamspace/studios/this_studio/cholec_dataset` is hardcoded in several places. Update if needed.

2. **SSL Mode:** The loader uses ALL splits (train+val+test) for self-supervised learning, which is standard for SSL.

3. **Frame Format:** Frames are assumed to be PNG or JPG images in VID## subfolders.

4. **No Labels Needed:** V-JEPA 2 is self-supervised, so no annotation files are required.

---

## Quick Start Commands

```bash
# Verify dataset
python start_training.py

# Check dataset size
du -sh /teamspace/studios/this_studio/cholec_dataset

# Count total frames
find /teamspace/studios/this_studio/cholec_dataset -name '*.png' -o -name '*.jpg' | wc -l

# List all video directories
ls /teamspace/studios/this_studio/cholec_dataset/**/VID*/
```

## Support

For V-JEPA 2 specific questions, refer to:
- V-JEPA 2 paper: https://arxiv.org/abs/2403.xxxxx
- Original V-JEPA: https://github.com/facebookresearch/jepa

