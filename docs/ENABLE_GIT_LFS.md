# Enable Git LFS on GitHub (required for checkpoints)

GitHub rejected LFS uploads with: **"Git LFS is disabled for this repository."**

## Fix (one-time, ~30 seconds)

1. Open [Cholec_Vjepa-2 Settings](https://github.com/Ryukijano/Cholec_Vjepa-2/settings)
2. Under **Archives** or search **Git LFS**
3. Check **Allow Git LFS** (wording may vary)
4. Save

## Upload checkpoints from Spark

Checkpoints are already copied locally under `weights/dinov2/` and `outputs/mot/` (~1.7 GB total).

```bash
cd /home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2
git checkout spark-lfs-setup
bash scripts/upload_checkpoints_lfs.sh
```

## Clone with weights (after upload)

```bash
git lfs install
git clone https://github.com/Ryukijano/Cholec_Vjepa-2.git
cd Cholec_Vjepa-2
git checkout spark-lfs-setup
git lfs pull
```
