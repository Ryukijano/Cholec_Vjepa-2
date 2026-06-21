# Cholec V-JEPA2 — Dual-Expert SurgiTrack++

V-JEPA2 + RF-DETR pipeline for **CholecTrack20** surgical tool detection, re-identification, and tracking.

- **Detection**: RF-DETR / V-JEPA2 detection head (`code/train_detection.py`, `code/train_rfdetr.py`)
- **Re-ID**: Direction-aware ReIDHeadV2 (`code/train_reid_v2.py`)
- **Tracking**: SurgicalTrackerV2 + HOTA eval (`code/tracker_v2.py`, `code/eval_hota.py`)

Full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quick start (DGX Spark / Linux)

```bash
cd /home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2
git lfs install
git lfs pull   # after checkpoints are pushed to LFS

conda activate surgi_track   # or your PyTorch env
pip install -r requirements.txt

# CholecTrack20 (shared with Gyanateet_tracking on Spark)
export CHOLECTRACK20_ROOT=/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20
ln -sf "$CHOLECTRACK20_ROOT" cholec_dataset
```

### Phase 2 — Re-ID training

```bash
python code/train_reid_v2.py \
  --detection_checkpoint outputs/detection-hardened-v2/best.pt \
  --out_dir outputs/reid-phase2 \
  --epochs 20 --batch_size 8 --loss_type both
```

### HOTA evaluation

```bash
python code/eval_hota.py \
  --checkpoint outputs/reid-phase2/best.pt \
  --data_dir cholec_dataset/Test
```

## Checkpoints (Git LFS)

Weights are **not** in the default clone until uploaded via LFS. See [outputs/README.md](outputs/README.md).

| Checkpoint | Path |
|------------|------|
| SSL pretrain | `outputs/vjepa2-cholec-pretrain/latest.pt` |
| Detection | `outputs/detection-hardened-v2/best.pt` |
| Re-ID | `outputs/reid-phase2/best.pt` |

```bash
bash scripts/setup_lfs.sh
# copy .pt files into outputs/... then:
git add outputs/
git commit -m "Add checkpoints via LFS"
```

## Windows (RTX 4090)

See [readme.md](readme.md) for CUDA 12.1 + PowerShell commands.

## Repo layout

```
code/           Training, tracker, eval scripts
docs/           Architecture + upload notes
outputs/        LFS checkpoints (see outputs/README.md)
scripts/        setup_lfs.sh, Spark helpers
```

## Related

- [CholecTrack20](https://github.com/CAMMA-public/cholectrack20)
- [Gyanateet_tracking](https://github.com/) — GOT-JEPA MOT pipeline (same CT20 data on Spark)

## Branch

Active setup branch: `spark-lfs-setup` — Git LFS + DGX Spark paths.
