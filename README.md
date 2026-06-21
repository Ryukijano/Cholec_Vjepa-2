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
code/              V-JEPA2 / RF-DETR SurgiTrack++ scripts
core_app/          GOT-JEPA MOT pipeline (DINOv2 + Deformable DETR)
configs/train_mot/ Stage 1–4 YAML configs (dinov2/)
dinov2/            Meta DINOv2 vendor (facebookresearch/dinov2)
scripts/got_jepa/  MOT training + eval shell runners
weights/dinov2/    DINOv2 ImageNet pretrain (LFS)
outputs/           LFS checkpoints (V-JEPA2 + MOT)
docs/              Architecture docs
```

### GOT-JEPA MOT training (Spark)

```bash
export XFORMERS_DISABLED=1
export CHOLECTRACK20_ROOT=/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20
ln -sf "$CHOLECTRACK20_ROOT" cholec_dataset

python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage4-lean.yaml \
  --devices cuda:0
```

Pretrained checkpoints are under `outputs/mot/` (see [outputs/README.md](outputs/README.md)).

## Related

- [CholecTrack20](https://github.com/CAMMA-public/cholectrack20)
- [Gyanateet_tracking](https://github.com/) — GOT-JEPA MOT pipeline (same CT20 data on Spark)

## Branch

Active setup branch: `spark-lfs-setup` — Git LFS + DGX Spark paths.
