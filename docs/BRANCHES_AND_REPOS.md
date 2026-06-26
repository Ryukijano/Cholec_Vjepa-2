# Branches, repos, and where code lives

## GitHub repositories

| Repo | URL | Role |
|------|-----|------|
| **Cholec_Vjepa-2** | [github.com/Ryukijano/Cholec_Vjepa-2](https://github.com/Ryukijano/Cholec_Vjepa-2) | **Unified** repo: V-JEPA2 `code/` + GOT-JEPA `core_app/` + plans + checkpoints (LFS) |
| **Gyanateet_tracking** | [github.com/Ryukijano/Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) | Spark canonical MOT development; agent skills + workflows |
| **Temporal_Difference-Vision** | [github.com/Ryukijano/Temporal_Difference-Vision](https://github.com/Ryukijano/Temporal_Difference-Vision) | Upstream TDV fork (reference) |

## Cholec_Vjepa-2 branches

| Branch | Contents | Use when |
|--------|----------|----------|
| `main` | Original V-JEPA2 SurgiTrack++ scripts (`code/`, `docs/ARCHITECTURE.md`) | RF-DETR + V-JEPA2 Re-ID pipeline |
| `spark-lfs-setup` | GOT-JEPA mirror, `dinov2/`, plans, LFS layout, experiment docs | Spark paths + documentation |
| `tdv-pretraining` | TDV Stage 0, DETR fixes, SurgeNet init, W&B viz, Slurm jobs | AIRE HPC training |
| `merge-gyanateet-tracking` | Integration attempts (if present) | Syncing Spark ↔ GitHub |
| `imgbot` | Image optimization bot | Ignore for training |

**Target state:** merge `tdv-pretraining` fixes into `spark-lfs-setup`, then open PR to `main`.

## Local paths

| Path | Branch (typical) | Machine |
|------|------------------|---------|
| `/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking` | `master` | DGX Spark |
| `/home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2` | `spark-lfs-setup` | DGX Spark |
| `/scratch/kcwp264/Cholec_Vjepa-2` | `tdv-pretraining` | Leeds AIRE |

## Checkpoints (not in git until LFS enabled)

| Location | Size (approx.) |
|----------|----------------|
| `outputs/mot/*/` | ~2 GB MOT stages (Spark copies on `spark-lfs-setup` locally) |
| `weights/dinov2/` | ~415 MB ImageNet DINOv2 ViT-S/B |
| `outputs/vjepa2-*`, `detection-hardened-v2/` | V-JEPA2 track (cloud/Windows) |

See [ENABLE_GIT_LFS.md](ENABLE_GIT_LFS.md) and [outputs/README.md](../outputs/README.md).
