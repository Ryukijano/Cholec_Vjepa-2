# Cholec_Vjepa-2 — Project plan index

> **REPO_MARKER** `REPO_ID=cholec_vjepa_2` — [REPO_ROLES.md](REPO_ROLES.md) explains which pipeline (`got_jepa_mot` vs `vjepa2_surgitrack`) and branch to use.

This repository combines two surgical tracking research lines on **CholecTrack20**:

1. **[GOT-JEPA MOT plan](plans/gyanateet_mot_understanding.md)** — four-stage DINOv2 + Deformable DETR multi-tool tracking (`core_app/`, branch `spark-lfs-setup`)
2. **[Dual-Expert SurgiTrack++](ARCHITECTURE.md)** — RF-DETR + V-JEPA2 Re-ID (`code/`, branch `main`)

Spark canonical GOT-JEPA dev: [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking).

## Quick links

| Doc | Use when |
|-----|----------|
| [REPO_ROLES.md](REPO_ROLES.md) | **Which repo / branch / pipeline** (agents read first) |
| [plans/gyanateet_mot_understanding.md](plans/gyanateet_mot_understanding.md) | Understanding goals, stages, checkpoints, blockers |
| [TRAINING_STAGES.md](TRAINING_STAGES.md) | Running training — frozen params, losses, commands |
| [ARCHITECTURE.md](ARCHITECTURE.md) | V-JEPA2 / RF-DETR dual-expert design |
| [ENABLE_GIT_LFS.md](ENABLE_GIT_LFS.md) | Uploading DINOv2 + MOT checkpoints |
| [agent/](agent/) | Agent handoff and integration notes |

## Training status snapshot (Jun 2026, DGX Spark)

- GOT-JEPA Stages 1–3 + Stage 4 lean: **trained** (checkpoints in `outputs/mot/`)
- LoRA ViT-B detection experiment: **active** (`cholec20-mot-stage1-lora-detect.yaml`)
- Full HOTA eval on CT20 test: **pending** (infra wired, needs tuned tracker + full clips)

Start here: [plans/gyanateet_mot_understanding.md](plans/gyanateet_mot_understanding.md)

**History:** [EXPERIMENT_TIMELINE.md](EXPERIMENT_TIMELINE.md) · **Repos:** [REPO_ROLES.md](REPO_ROLES.md) · [BRANCHES_AND_REPOS.md](BRANCHES_AND_REPOS.md) · **Agents:** [../AGENTS.md](../AGENTS.md)
