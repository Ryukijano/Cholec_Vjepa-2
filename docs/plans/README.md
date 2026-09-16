# Project plans

Canonical planning docs for **Cholec_Vjepa-2** (dual pipeline: V-JEPA2 SurgiTrack++ + GOT-JEPA MOT).

| Document | Description |
|----------|-------------|
| [gyanateet_mot_understanding.md](gyanateet_mot_understanding.md) | Master plan — GOT-JEPA four-stage MOT on CholecTrack20 (problem, architecture, stages, research context) |
| [../TRAINING_STAGES.md](../TRAINING_STAGES.md) | Authoritative stage-by-stage training guide (frozen params, losses, launch commands) |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Dual-Expert SurgiTrack++ (RF-DETR + V-JEPA2 Re-ID) |
| [../agent/](../agent/) | Cursor/agent handoff notes (repo understanding, V-JEPA integration) |

**Repo layout note:** GOT-JEPA shell runners live under `scripts/got_jepa/` (e.g. `bash scripts/got_jepa/train_stage1_gb10_1gpu.sh`). V-JEPA2 scripts remain in `code/`.

**Checkpoints:** `outputs/mot/` and `weights/dinov2/` — see [ENABLE_GIT_LFS.md](../ENABLE_GIT_LFS.md).
