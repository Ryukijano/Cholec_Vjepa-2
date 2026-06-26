# AGENTS.md — Cholec_Vjepa-2

Agent memory for the unified surgical tracking repo ([Ryukijano/Cholec_Vjepa-2](https://github.com/Ryukijano/Cholec_Vjepa-2)).

> **REPO_MARKER** `REPO_ID=cholec_vjepa_2` · **two pipelines** — pick branch + folder before editing  
>
> | Pipeline | Marker | Path | Branch |
> |----------|--------|------|--------|
> | GOT-JEPA MOT | `got_jepa_mot` | `core_app/` | `spark-lfs-setup` |
> | V-JEPA2 SurgiTrack++ | `vjepa2_surgitrack` | `code/` | `main` |
> | AIRE TDV / DETR fix | `aire_experiments` | `pretrain_tdv.py`, configs | `tdv-pretraining` |
>
> **Spark GOT-JEPA canonical dev:** [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking).  
> **Repo map:** [docs/REPO_ROLES.md](docs/REPO_ROLES.md)

## Project overview

Two research lines on **CholecTrack20**:

1. **GOT-JEPA MOT** (`core_app/`, `configs/train_mot/`) — four-stage pipeline: DETR teacher → JEPA SSL → joint fine-tune → OccuSolver.
2. **V-JEPA2 SurgiTrack++** (`code/`) — RF-DETR + V-JEPA2 Re-ID + SurgicalTrackerV2.

**Repo roles:** [docs/REPO_ROLES.md](docs/REPO_ROLES.md)  
**Experiment history:** [docs/EXPERIMENT_TIMELINE.md](docs/EXPERIMENT_TIMELINE.md)  
**Plan:** [docs/plans/gyanateet_mot_understanding.md](docs/plans/gyanateet_mot_understanding.md)  
**Branches:** [docs/BRANCHES_AND_REPOS.md](docs/BRANCHES_AND_REPOS.md)

## Learned user preferences

- Stage 1 gate: **mAP@50 ≥ 0.35–0.45**; defer HOTA until Stage 3+ with tuned tracker thresholds.
- Stage 1 `detector_only: true` for pseudo-label teacher; full MOT training is Stage 3.
- GOT-JEPA Stage 2: frozen Stage-1 teacher (not EMA); Cholec80 SSL excludes CT20 val/test overlap videos.
- Spark (`surgi_track`, `XFORMERS_DISABLED=1`) for baseline + debug; AIRE for TDV Stage 0 and SurgeNet Stage 1 retrains.
- TDV is Stage 0 encoder prep only — does not replace GOT-JEPA ([TDV paper](https://arxiv.org/abs/2606.15956)).
- Enable Git LFS on GitHub before pushing `outputs/` or `weights/` checkpoints.

## Learned workspace facts

- **Spark clone:** `/home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2` — usually on `spark-lfs-setup`.
- **AIRE clone:** `/scratch/kcwp264/Cholec_Vjepa-2` — usually on `tdv-pretraining`.
- **Sibling repo:** [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) on Spark; same MOT code path before mirror.
- **Completed on Spark (May–Jun 2026):** S1 (~2.8% mAP), S2 (ep 20/30), S3 (~2.4% mAP), S4 lean (~0.6% mAP, `depth_stub: true`).
- **Blocker:** Deformable DETR `(1, total_len)` spatial bug → LoRA Stage 1 val mAP ~0%; fix on `tdv-pretraining`.
- **AIRE TDV:** Resume `outputs/tdv_pretrain_vitb14_surgenet/latest.pth.tar` from step ~29k → 60k after DETR merge.
- **LFS:** Push blocked until enabled in repo settings; use `scripts/upload_checkpoints_lfs.sh`.
- **Eval:** `scripts/got_jepa/eval_checkpoint.py --mot-eval --stratify-smoke`, `eval_mot_hota.py`.

## Quick commands

```bash
conda activate surgi_track
export XFORMERS_DISABLED=1
export CHOLECTRACK20_ROOT=/path/to/cholectrack20
ln -sf "$CHOLECTRACK20_ROOT" cholec_dataset

# GOT-JEPA MOT
python -m core_app.mot.main --fname configs/train_mot/dinov2/cholec20-mot-stage4-lean.yaml --devices cuda:0

# TDV Stage 0 (AIRE, 3 GPU)
torchrun --nproc_per_node=3 scripts/pretrain_tdv.py --config configs/train_mot/dinov2/tdv-pretrain.yaml --ddp
```

## Next actions (priority)

1. Merge `tdv-pretraining` DETR + LoRA config fixes into Spark `Gyanateet_tracking` / `spark-lfs-setup`.
2. Retrain Stage 1; gate on mAP@50.
3. Enable LFS and push checkpoints, or upload to [Hugging Face](https://huggingface.co/Ryukijano).
