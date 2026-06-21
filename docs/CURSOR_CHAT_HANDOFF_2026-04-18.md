# Cursor chat handoff — surgi_world_track

**Date:** 2026-04-18  
**Source:** Cursor agent session (plan + implementation roadmap + subagent synthesis).  
**Note:** This folder is **not** a git repository on disk as of this export; there is nothing to `git push` until you `git init`, add a remote, and commit.

---

## What you asked for

1. Research and planning: map the repo, align docs with code, outline next steps (V-JEPA world model, inference, optional LoRA).
2. Export this conversation as a single Markdown file **in the project folder** (this file).
3. Use **multiple agents**: two explore subagents were run in parallel; their outputs are merged below.

---

## Executive summary

- **Training path:** `scripts/*.sh` → `python -m core_app.main` → `WorldModelTrainer` → `VJEPAWorldModel` / `SurgicalTrackingSystem`; data via `CholecDataset` and `collate_fn` in `core_app/data/video_dataset.py`.
- **Frozen backbone:** README describes frozen V-JEPA 2.1 + multi-scale predictor + DETR + ReID. **Trainer note:** `WorldModelTrainer.build_optimizer` sets `predictor.parameters()` to `requires_grad=False` — if the goal is to **train** the temporal predictor / world-model head, that must be revisited.
- **LoRA / PEFT:** No active LoRA in this repo’s Python. `Supervised LoRA Training and Inference.md` is largely a **Cascade chat export** about a **different** project path (`h:\anatomical_classification\`), not executable LoRA code here.
- **Useful context docs:** `V-JEPA 2.1 Integration.md`, `V-JEPA Object Permanence.md`, root `README.md`.
- **External alignment:** V-JEPA 2 emphasizes latent prediction with frozen backbone ([Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/), [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)).

---

## Plan artifact (Cursor plan tool)

A structured roadmap was saved outside the repo by Cursor’s plan feature, e.g.:

`c:\Users\sc23pg\.cursor\plans\surgi-world-track-implementation-roadmap_de6254f7.plan.md`

**Themes from that plan:** docs alignment; new `core_app/inference.py`; optional LoRA behind config; dependencies + smoke tests; policy for `core_app` vs `lightning_upload/core_app` duplication; reproducibility checklist.

---

## Open work (todo-style)

| ID | Item |
|----|------|
| align-docs | One authoritative status doc mapping runtime code vs historical LoRA-only markdown. |
| build-inference-cli | `core_app/inference.py`: checkpoint load, image/video modes, schema for detections + embeddings + future prediction. |
| add-inference-scripts | Shell/Python examples for single-image and video inference. |
| enable-optional-lora | Optional LoRA in encoder wrapper + trainer config; default stays frozen backbone. |
| dependency-and-smoke-tests | Optional `peft` notes; smoke tests for load, forward, checkpoint roundtrip. |
| sync-duplicate-tree | Decide parity policy for `core_app` vs `lightning_upload/core_app`. |
| stability-checklist | Repro checklist: configs, checkpoints, splits, tensor shapes. |

---

## Subagent A — README vs code gaps (with paths)

Parallel explore agent output, condensed:

- **Multi-GPU:** `scripts/run_world_model_3gpu.sh` passes multiple `--devices`, but `core_app/main.py` uses `args.devices[0]` only; no DDP in `core_app/`.
- **Inference:** No dedicated inference CLI; training is `python -m core_app.main`; README shows a Python snippet only.
- **`predict_future` horizon:** In `core_app/models/vjepa_world_model.py`, `predict_future` may not apply the `horizon` argument as documented — verify and fix if README promises `horizon=…`.
- **README layout:** Says `app/`; actual package is `core_app/`.
- **Checkpoint wiring:** YAML may use `meta.read_checkpoint` / HF keys while `world_model_trainer.py` reads `supervised.pretrained_checkpoint` for the encoder — keys may not match.
- **Mixed precision:** Config flags may exist without `autocast` / `GradScaler` in the trainer.
- **ReID loss:** README vs `ReidHead` / YAML `ce_loss_weight` may disagree with what the model constructs.
- **LoRA:** No PEFT hooks in `core_app/` or `configs/`.
- **Curriculum:** YAML `supervised.curriculum` vs hard-coded schedule in `SurgicalTrackingSystem.update_curriculum`.
- **Gradient accumulation:** `optimizer.zero_grad()` every batch vs stepping every `accum_steps` may prevent true accumulation — verify against intended effective batch size.

**Paths:** `README.md`, `scripts/run_world_model_3gpu.sh`, `core_app/main.py`, `core_app/models/vjepa_world_model.py`, `core_app/trainers/world_model_trainer.py`, `configs/train_2_1/vitb16/cholec20-world-model-detr-reid.yaml`.

---

## Subagent B — Markdown inventory

**Cascade / chat exports (4):**

- `Debugging V-JEPA Model Loading.md`
- `Supervised LoRA Training and Inference.md`
- `V-JEPA Object Permanence.md`
- `V-JEPA 2.1 Integration.md`

**Project / tooling (3):**

- `README.md`
- `.windsurf/workflows/make-svg-diagram.md`
- `.windsurf/plans/dino-wm-plan-d1b09c.md`

**Bundled upstream (`tips/`, `dinov2/`) (8):** standard README / CONTRIBUTING / MODEL_CARD / extension docs — not session-specific.

---

## How to “push” this file later

```powershell
cd i:\surgi_world_track
git init
git add docs/CURSOR_CHAT_HANDOFF_2026-04-18.md
git commit -m "Add Cursor chat handoff export for 2026-04-18"
git remote add origin <your-remote-url>
git push -u origin main
```

Use your real default branch name if not `main`.

---

*End of export.*
