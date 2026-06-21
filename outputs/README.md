# Checkpoints (Git LFS)

Large weights live here and are tracked with **Git LFS**. Clone with:

```bash
git lfs install
git clone https://github.com/Ryukijano/Cholec_Vjepa-2.git
cd Cholec_Vjepa-2
git lfs pull
```

## Expected layout

| Path | Description |
|------|-------------|
| `outputs/vjepa2-cholec-pretrain/latest.pt` | V-JEPA2 SSL pretrain on Cholec80 |
| `outputs/detection-hardened-v2/best.pt` | Detection checkpoint (~35% recall @0.5) |
| `outputs/reid-phase2/best.pt` | Re-ID head after `train_reid_v2.py` |

## Upload checkpoints (after training)

```bash
git lfs install
git add outputs/detection-hardened-v2/best.pt
git add outputs/vjepa2-cholec-pretrain/latest.pt
git commit -m "Add detection and SSL checkpoints via LFS"
git push origin spark-lfs-setup
```

## Data (not in git)

Point scripts at local CholecTrack20, e.g. on DGX Spark:

```bash
export CHOLECTRACK20_ROOT=/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20
```

Or symlink: `ln -s "$CHOLECTRACK20_ROOT" cholec_dataset`
