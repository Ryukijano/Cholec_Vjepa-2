# Cholec_Vjepa-2 — Unified Surgical Tracking (Two Pipelines)

One GitHub repo, **two complementary** CholecTrack20 research lines:

| Pipeline | Marker | Folder | Branch | Primary machine |
|----------|--------|--------|--------|-----------------|
| **V-JEPA2 SurgiTrack++** | `vjepa2_surgitrack` | `code/` | `main` | any (incl. Windows RTX 4090) |
| **GOT-JEPA MOT** | `got_jepa_mot` | `core_app/` | `spark-lfs-setup` | DGX Spark |
| **AIRE experiments** | `aire_experiments` | TDV + DETR fixes | `tdv-pretraining` | Leeds AIRE |

---

> **REPO_MARKER** `REPO_ID=cholec_vjepa_2`

**Spark GOT-JEPA dev lab (authoritative):** [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) — train/eval `core_app.mot` there first; this repo mirrors code + hosts LFS/docs.

**Repo map (humans + agents):** [docs/REPO_ROLES.md](docs/REPO_ROLES.md) · [docs/BRANCHES_AND_REPOS.md](docs/BRANCHES_AND_REPOS.md) · [AGENTS.md](AGENTS.md)

---

## Which pipeline should I use?

| Goal | Checkout | Entrypoint |
|------|----------|------------|
| RF-DETR detection + V-JEPA2 Re-ID + HOTA | `main` | `code/train_rfdetr.py`, `code/train_reid_v2.py` |
| GOT-JEPA four-stage MOT (DINOv2 + DETR + JEPA) | `spark-lfs-setup` | `python -m core_app.mot.main` (or use sibling repo on Spark) |
| TDV Stage 0 pretrain, SurgeNet encoder, DETR bug fix | `tdv-pretraining` | `pretrain_tdv.py` on AIRE |
| Agent skills, MOT workflows on Spark | — | [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) |

---

## Pipeline A — V-JEPA2 SurgiTrack++ (`code/`, branch `main`)

RF-DETR detection → direction-aware V-JEPA2 Re-ID → `SurgicalTrackerV2` + HOTA eval.

| Component | Script |
|-----------|--------|
| Detection | `code/train_detection.py`, `code/train_rfdetr.py` |
| Re-ID | `code/train_reid_v2.py` |
| Tracking / eval | `code/tracker_v2.py`, `code/eval_hota.py` |

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Quick start — V-JEPA2 (Linux / Spark)

```bash
git checkout main
cd /home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2

conda activate surgi_track
pip install -r requirements.txt

export CHOLECTRACK20_ROOT=/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20
ln -sf "$CHOLECTRACK20_ROOT" cholec_dataset
```

**Phase 2 — Re-ID training**

```bash
python code/train_reid_v2.py \
  --detection_checkpoint outputs/detection-hardened-v2/best.pt \
  --out_dir outputs/reid-phase2 \
  --epochs 20 --batch_size 8 --loss_type both
```

**HOTA evaluation**

```bash
python code/eval_hota.py \
  --checkpoint outputs/reid-phase2/best.pt \
  --data_dir cholec_dataset/Test
```

**Windows (RTX 4090):** see [readme.md](readme.md) for CUDA 12.1 + PowerShell commands.

### V-JEPA2 checkpoints (Git LFS)

| Checkpoint | Path |
|------------|------|
| SSL pretrain | `outputs/vjepa2-cholec-pretrain/latest.pt` |
| Detection | `outputs/detection-hardened-v2/best.pt` |
| Re-ID | `outputs/reid-phase2/best.pt` |

```bash
bash scripts/setup_lfs.sh
# copy .pt files into outputs/... then git add + commit (LFS must be enabled on GitHub)
```

---

## Pipeline B — GOT-JEPA MOT (`core_app/`, branch `spark-lfs-setup`)

Four-stage object-permanence MOT: supervised DETR teacher → JEPA SSL on Cholec80 → joint fine-tune → OccuSolver (Stage 4).

| Stage | Config (example) |
|-------|------------------|
| 1 — supervised scaffold | `cholec20-mot-stage1-supervised.yaml` |
| 2 — JEPA SSL | `cholec80-ct20-stage2-jepa-pretrain.yaml` |
| 3 — joint fine-tune | `cholec20-mot-stage3-joint-finetune.yaml` |
| 4 — OccuSolver lean | `cholec20-mot-stage4-lean.yaml` |

Plan: [docs/plans/gyanateet_mot_understanding.md](docs/plans/gyanateet_mot_understanding.md) · Stages: [docs/TRAINING_STAGES.md](docs/TRAINING_STAGES.md)

**Prefer Spark canonical clone for day-to-day training:**

```bash
cd /home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking   # not this repo
conda activate surgi_track
export XFORMERS_DISABLED=1
bash scripts/train_stage1_ddp_3gpu.sh
```

**Or run from this mirror** (`spark-lfs-setup`):

```bash
git checkout spark-lfs-setup
export XFORMERS_DISABLED=1
export CHOLECTRACK20_ROOT=/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking/data/cholectrack20
ln -sf "$CHOLECTRACK20_ROOT" cholec_dataset

python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage4-lean.yaml \
  --devices cuda:0
```

MOT checkpoints: `outputs/mot/` · DINOv2 weights: `weights/dinov2/` — see [outputs/README.md](outputs/README.md). LFS push requires [docs/ENABLE_GIT_LFS.md](docs/ENABLE_GIT_LFS.md).

---

## Pipeline C — AIRE (`tdv-pretraining`)

TDV Stage 0 encoder prep, SurgeNet init, deformable DETR spatial-shape fix (resolves LoRA Stage 1 val mAP ≈ 0). Typical path: `/scratch/kcwp264/Cholec_Vjepa-2` on Leeds AIRE.

Merge target: bring `tdv-pretraining` fixes into [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) on Spark before retraining Stage 1.

---

## Repo layout

```
Cholec_Vjepa-2/
├── code/              # Pipeline A — V-JEPA2 / RF-DETR SurgiTrack++
├── core_app/          # Pipeline B — GOT-JEPA MOT (mirror of Gyanateet_tracking)
├── configs/train_mot/ # Stage 1–4 YAML (dinov2/)
├── dinov2/            # Meta DINOv2 vendor
├── scripts/
│   ├── got_jepa/      # MOT shell runners
│   └── setup_lfs.sh   # LFS + checkpoint upload
├── weights/dinov2/    # DINOv2 ImageNet pretrain (LFS)
├── outputs/           # V-JEPA2 + MOT checkpoints (LFS)
└── docs/
    ├── REPO_ROLES.md           # Which repo / pipeline / branch (read first)
    ├── EXPERIMENT_TIMELINE.md  # Chronological runs
    ├── BRANCHES_AND_REPOS.md   # Paths and sync status
    └── PLAN.md                 # Master plan index
```

---

## Documentation index

| Doc | Purpose |
|-----|---------|
| [docs/REPO_ROLES.md](docs/REPO_ROLES.md) | **Repo + pipeline markers** — agents read first |
| [AGENTS.md](AGENTS.md) | Agent memory — preferences, blockers, next actions |
| [docs/EXPERIMENT_TIMELINE.md](docs/EXPERIMENT_TIMELINE.md) | Chronological runs and metric evolution |
| [docs/BRANCHES_AND_REPOS.md](docs/BRANCHES_AND_REPOS.md) | Branches, local paths, checkpoint locations |
| [docs/PLAN.md](docs/PLAN.md) | Master plan index |
| [docs/plans/gyanateet_mot_understanding.md](docs/plans/gyanateet_mot_understanding.md) | GOT-JEPA goals, gates, blockers |

## Related

- [CholecTrack20](https://github.com/CAMMA-public/cholectrack20) — benchmark dataset
- [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) — Spark canonical GOT-JEPA lab + agent tooling
- [Temporal_Difference-Vision](https://github.com/Ryukijano/Temporal_Difference-Vision) — TDV upstream reference

## Active branches

| Branch | Role |
|--------|------|
| `main` | V-JEPA2 `code/` (original repo) |
| `spark-lfs-setup` | GOT-JEPA mirror + docs + LFS layout |
| `tdv-pretraining` | AIRE TDV + DETR fixes + SurgeNet Stage 1 |
| `merge-gyanateet-tracking` | Integration attempts |

Target: merge `tdv-pretraining` → Spark `Gyanateet_tracking`, then unify on `main`.
