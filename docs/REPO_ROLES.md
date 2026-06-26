# Repo roles — Gyanateet_tracking vs Cholec_Vjepa-2

> **For humans and agents:** read this before choosing a repo, branch, or entrypoint.  
> Markers below are stable identifiers for automation (`REPO_ID`, `PIPELINE`, `MACHINE`).

---

## Quick decision table

| I want to… | Go here |
|------------|---------|
| Train GOT-JEPA Stages 1–4 on **DGX Spark** | **`gyanateet_tracking`** → `python -m core_app.mot.main` |
| Use **Cursor/Windsurf** agent skills & MOT commands | **`gyanateet_tracking`** → `.cursor/`, `workflows/` |
| Run **RF-DETR + V-JEPA2** Re-ID / SurgiTrack++ | **`cholec_vjepa-2`** → `code/` on branch `main` |
| **TDV Stage 0** or **SurgeNet Stage 1** on **AIRE** | **`cholec_vjepa-2`** → branch `tdv-pretraining` |
| Publish docs + mirrored GOT-JEPA + LFS checkpoints | **`cholec_vjepa-2`** → branch `spark-lfs-setup` |
| Latest **DETR spatial-shape fix** (mAP≈0 bug) | **`cholec_vjepa-2`** → branch `tdv-pretraining` (merge pending to Spark) |

---

## Repo A — Gyanateet_tracking

```yaml
REPO_ID: gyanateet_tracking
GITHUB: https://github.com/Ryukijano/Gyanateet_tracking
DEFAULT_BRANCH: master
PRIMARY_MACHINE: dgx_spark
LOCAL_PATH: /home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking
```

| Field | Value |
|-------|--------|
| **ROLE** | Primary **development lab** for GOT-JEPA surgical MOT |
| **PIPELINE** | `got_jepa_mot` only |
| **ENTRYPOINT** | `python -m core_app.mot.main --fname configs/train_mot/dinov2/...` |
| **HAS** | `core_app/`, `configs/`, `scripts/`, `data/`, local `outputs/mot/`, agent tooling |
| **DOES_NOT_HAVE** | V-JEPA2 `code/` pipeline (RF-DETR track) |
| **SIBLING** | [Cholec_Vjepa-2](https://github.com/Ryukijano/Cholec_Vjepa-2) — public mirror + V-JEPA2 + AIRE branches |

**Agent rule:** If the task is Spark MOT training/eval and mentions `core_app.mot`, stay in **this repo**.

---

## Repo B — Cholec_Vjepa-2 (this repo)

```yaml
REPO_ID: cholec_vjepa_2
GITHUB: https://github.com/Ryukijano/Cholec_Vjepa-2
DEFAULT_BRANCH: main
PRIMARY_MACHINE: mixed  # spark mirror + aire hpc
LOCAL_PATH_SPARK: /home/aimsgroupuol/AIMSgeneral/Cholec_Vjepa-2
LOCAL_PATH_AIRE: /scratch/kcwp264/Cholec_Vjepa-2
```

| Field | Value |
|-------|--------|
| **ROLE** | **Unified GitHub umbrella** — two pipelines + experiment archive |
| **PIPELINE_A** | `vjepa2_surgitrack` → `code/` (RF-DETR, V-JEPA2 Re-ID, tracker) |
| **PIPELINE_B** | `got_jepa_mot` → `core_app/` (mirrored from Gyanateet_tracking) |
| **SIBLING** | [Gyanateet_tracking](https://github.com/Ryukijano/Gyanateet_tracking) — Spark canonical GOT-JEPA dev |

### Branch markers (this repo)

| Branch | `BRANCH_ROLE` | Pipeline | Machine |
|--------|---------------|----------|---------|
| `main` | `vjepa2_legacy` | V-JEPA2 `code/` | any |
| `spark-lfs-setup` | `got_jepa_mirror` | GOT-JEPA + docs + LFS layout | Spark |
| `tdv-pretraining` | `aire_experiments` | TDV Stage 0 + DETR fixes + SurgeNet S1 | AIRE |
| `merge-gyanateet-tracking` | `integration` | sync attempts | — |

**Agent rule:** If the task mentions `train_rfdetr`, `train_reid_v2`, or `tracker_v2`, use **`code/`** on `main`. If it mentions TDV or `/scratch/kcwp264`, checkout **`tdv-pretraining`**.

---

## Two pipelines, one benchmark (CholecTrack20)

| Marker | `got_jepa_mot` | `vjepa2_surgitrack` |
|--------|----------------|---------------------|
| **Idea** | Invariant tracking weights ω under corruption (JEPA SSL) | RF-DETR detect + V-JEPA2 temporal Re-ID |
| **Repo home** | Gyanateet_tracking (+ mirror here on `spark-lfs-setup`) | This repo `code/` on `main` |
| **Stages** | 4-stage: S1 DETR → S2 JEPA → S3 joint → S4 OccuSolver | Phase 1 detect → Phase 2 Re-ID → track |
| **Same data?** | Yes — CholecTrack20 / Cholec80 | Yes |

They are **complementary experiments**, not duplicates.

---

## Sync status (Jun 2026)

| Item | Gyanateet_tracking | Cholec_Vjepa-2 |
|------|-------------------|----------------|
| GOT-JEPA `core_app/` | ✅ authoritative on Spark | ✅ mirrored on `spark-lfs-setup` |
| Experiment docs | ✅ | ✅ |
| DETR bug fix | ❌ not merged yet | ✅ on `tdv-pretraining` |
| V-JEPA2 `code/` | ❌ | ✅ on `main` |
| Checkpoints in git | ❌ local only | LFS layout ready; push blocked |

---

## Related docs

- [EXPERIMENT_TIMELINE.md](EXPERIMENT_TIMELINE.md)
- [BRANCHES_AND_REPOS.md](BRANCHES_AND_REPOS.md)
- [../AGENTS.md](../AGENTS.md)
