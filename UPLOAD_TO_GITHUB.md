# Upload scripts to GitHub (Cholec_Vjepa-2)

Use this guide to push the Dual-Expert SurgiTrack++ / Cholec V-JEPA2 scripts to **https://github.com/Ryukijano/Cholec_Vjepa-2**.

---

## 1. Create the repo on GitHub (if needed)

- Go to https://github.com/new
- Repository name: `Cholec_Vjepa-2`
- Description: e.g. *Dual-Expert surgical tool tracking with V-JEPA2 and RF-DETR on CholecTrack20*
- Choose **Public**, add a README if you want, then **Create repository**.

---

## 2. Scripts to include (recommended)

These are the core scripts for the pipeline. The root `.gitignore` already excludes large data, checkpoints, and outputs.

### From `code/` (main scripts)

| Script | Purpose |
|--------|--------|
| `convert_to_coco.py` | CholecTrack20 → COCO format for RF-DETR |
| `train_rfdetr.py` | Fine-tune RF-DETR on CholecTrack20 |
| `eval_rfdetr.py` | Evaluate RF-DETR (Recall@0.5, mAP) |
| `rfdetr_wrapper.py` | RF-DETR wrapper exposing detections + query features |
| `reid_head_v2.py` | 4-branch direction-aware Re-ID head |
| `train_reid_v2.py` | Train Re-ID head (V-JEPA2 + LoRA + ReIDHeadV2) |
| `tracker_v2.py` | Multi-perspective tracker (HBGM-style) |
| `run_tracking.py` | Full pipeline: detect → Re-ID → track |
| `eval_hota.py` | HOTA/MOTA/IDF1 evaluation |
| `dataset.py` | CholecTrack20 dataset + IdentitySampler |
| `reid_head.py` | Re-ID losses, memory bank (used by train_reid_v2) |
| `tracker.py` | Kalman filter, IoU (used by tracker_v2) |
| `viz_rfdetr_detection.py` | Visualize RF-DETR detections |
| `detection_viz.py` | Detection visualization helpers |

### Optional (legacy / extras)

- `train_joint.py`, `train_detection.py`, `train_reid.py`, `models.py`, `losses.py` — older joint training / detection
- `dataset_tracking.py`, `extract_features.py`, `viz_tracks.py` — tracking dataset / viz
- `code/TrackEval/` — submodule or copy if you use it for HOTA

### From project root

- `docs/` — `ARCHITECTURE.md`, `surgitrack++_architecture.svg`
- `.cursor/plans/dual-expert_surgitrack++_13e9f879.plan.md` (optional)
- This file: `UPLOAD_TO_GITHUB.md`

---

## 3. Git commands (from project root)

Replace `Ryukijano/Cholec_Vjepa-2` with your actual repo URL if different.

```powershell
cd H:\vjepa2_complete_windows_20260210_200325

# If this folder is not yet a git repo:
git init
git add .gitignore
git add code/convert_to_coco.py code/train_rfdetr.py code/eval_rfdetr.py
git add code/rfdetr_wrapper.py code/reid_head_v2.py code/train_reid_v2.py
git add code/tracker_v2.py code/tracker.py code/run_tracking.py code/eval_hota.py
git add code/dataset.py code/reid_head.py code/detection_viz.py code/viz_rfdetr_detection.py
git add docs/
git add UPLOAD_TO_GITHUB.md
# Add any README you have
git add README.md 2>$null; git add README.rst 2>$null

git status   # Review before commit
git commit -m "Add Dual-Expert SurgiTrack++ scripts (RF-DETR, V-JEPA2, ReID, tracker, eval)"

git branch -M main
git remote add origin https://github.com/Ryukijano/Cholec_Vjepa-2.git
git push -u origin main
```

If the repo already exists and has a README:

```powershell
git remote add origin https://github.com/Ryukijano/Cholec_Vjepa-2.git
git pull origin main --rebase
git push -u origin main
```

---

## 4. README snippet for the repo

You can add this to the repo README:

```markdown
## Cholec V-JEPA2 / Dual-Expert SurgiTrack++

Surgical tool detection and multi-perspective tracking on CholecTrack20, using:
- **RF-DETR** (DINOv2) for detection
- **V-JEPA2** (LoRA) for temporal Re-ID
- **ReIDHeadV2** (direction-aware 4-branch fusion)
- **SurgicalTrackerV2** (HBGM-style 3-perspective tracker)

### Setup
- Install: PyTorch, rf-detr, motmetrics; clone vjepa2 and point `sys.path` as in the scripts.
- Data: CholecTrack20 (and Cholec80 for V-JEPA2 pretraining).

### Pipeline
1. `convert_to_coco.py` — convert CholecTrack20 to COCO
2. `train_rfdetr.py` — train RF-DETR
3. `train_reid_v2.py` — train Re-ID head (V-JEPA2 + ReIDHeadV2)
4. `run_tracking.py` — run full pipeline
5. `eval_hota.py` — evaluate with `--mot_pred_dir` and `--all_perspectives`

See `docs/ARCHITECTURE.md` for details.
```

---

## 5. Notes

- **Don’t push**: `data/`, `cholec_dataset/`, `outputs/`, `*.pth`, `*.pt`, `wandb/` — they’re in `.gitignore`.
- If the repo is **private**, use HTTPS with a personal access token or SSH:  
  `git remote add origin git@github.com:Ryukijano/Cholec_Vjepa-2.git`
- If you get **404**, confirm the repo exists and the URL is exactly  
  `https://github.com/Ryukijano/Cholec_Vjepa-2` (or create it first).
