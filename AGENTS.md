## Learned User Preferences

- Stage 1 trains a precision-leaning pseudo-label teacher (`detector_only: true`), not a final detector optimized for raw mAP.
- Run training via `conda activate endofm-lv`, then `bash scripts/got_jepa/train_stage1_ddp_3gpu.sh` from the repo root (single-GPU path uses `python -m core_app.mot.main --devices cuda:0`); set `XFORMERS_DISABLED=1` on GB10 for DINOv2.
- Prefer `endofm-lv` conda env (Python 3.11, PyTorch 2.7.0+cu126); fallback env is `surgi_track` if primary is missing.
- Stay GOT-JEPA surgical MOT course; VLA-JEPA and pixel desmearing are complementary only, not primary pivots.
- Stage 2 JEPA should use a frozen Stage-1 teacher checkpoint, not EMA teacher updates (matches GOT-JEPA paper).
- Use Cholec80 as unlabeled SSL corpus for Stage 2; exclude CT20 val/test overlap videos from SSL.
- Add labeled Cholec-80 to Stage 1 only on train split with leak checks; native C80 labels are tool presence, not MOT boxes.
- On NVIDIA GB10 (DGX Spark, single GPU): target `batch_size` 24–32 in stage-1 config; prefer Stage 4 lean over Stage 4 full/VGGT unless ablating geometry.
- After Stage 2 ~epoch 20, Stage 3 then Stage 4 lean is the main path; Stage 2 epochs 21–29 optional unless more SSL gains are needed.
- Use Spark for finishing baseline and memory-heavy Stage 4 full; Leeds AIRE (28 nodes × 3 L40S, max 3 GPUs per node) for HP sweeps, multi-seed, eval arrays, and SSL parallelization.

## Learned Workspace Facts

### Project structure
- Project root: `/scratch/kcwp264/Cholec_Vjepa-2`; four-stage MOT pipeline (Stage 1 supervised teacher → Stage 2 GOT-JEPA SSL → Stage 3 joint finetune → Stage 4 OccuSolver/geometry).
- Model: SurgeNetDINO ViT-B/14 (768-dim) frozen encoder + LoRA adapters + Deformable DETR (48 queries, DN-DETR denoising).
- Stage-1 config: `configs/train_mot/dinov2/cholec20-mot-stage1-surgenet.yaml` (`batch_size` 6, `img_size` 392, `clip_length` 3, `dinov2_vitb14`, deformable DETR).
- Local CholecTrack20: **deleted 2026-09-11** along with `datasets_cholec/cholec80_extracted_frames.tar`, `data/surgi_world_track/cholec20_coco{,_augmented}`, and the `TRACK_JEPA/surgi_world_track` repo copy — re-download via Synapse/scripts before any Stage 1–3 run. Canonical splits in `core_app/data/splits.py`; SSL excludes Cholec80 videos overlapping CT20 val/test: `video01,06,07,12,25,30,39`.
- SSL corpus: `/scratch/kcwp264/data/surgi_world_track/ssl_corpus/Training/` — frame symlinks removed 2026-09-11; only per-video `.json` metadata remains, so the corpus must be rebuilt (`scripts/got_jepa/build_ssl_corpus_3gpu.sh`) after dataset re-download.
- W&B project: `hack-the-thong/surgical-mot`

### Training status (verified Sep 2026, AIRE)
- No JEPA/MOT Slurm jobs are currently active.
- Current Stage 1 `latest.pth.tar` reached epoch 185 with `best_val_map=0.0270`; `best.pth.tar` is epoch 115. This is still far below the mAP@50 ~0.45 / HOTA 35-40 target.
- Stage 2 completed its configured 30 epochs (`latest.pth.tar` epoch 29), but it predates the current Stage 1 checkpoints and therefore belongs to an older checkpoint chain.
- The stronger Stage 3 directory, `outputs/mot/cholec20-mot-stage3-joint`, completed epoch 49 with `best_val_map=0.0326`; the older `cholec20-stage3-joint-finetune` run ended at epoch 9 with `best_val_map=0.000325`.
- No persisted HOTA/MOTA evaluation result was found, so the four-stage pipeline has not demonstrated its tracking targets.
- The separate RF-DETR baseline is the strongest verified detector: `outputs/mot/rfdetr-baseline` completed 20 epochs with validation `mAP_50=0.4014` and has best regular/EMA/total checkpoints. A partial 84-key transfer into the custom DETR left 370 keys missing and did not transfer this performance.
- Resume Stage 1 with `bash scripts/got_jepa/train_stage1_ddp_3gpu.sh` (auto-resumes from `latest.pth.tar`, use `--reset-optimizer` when switching GPU count). Stage 2 configs are `cholec80-ct20-stage2-jepa-surgenet.yaml` (recommended) and `cholec20-mot-stage2-jepa-surgenet.yaml` (fallback); OccuSolver is disabled in Stage 2.

### Bugs fixed (Jun 2026)
1. **Label off-by-one** in `core_app/mot/data.py:70` — CholecTrack20 uses 0-indexed instrument IDs (0-6), code was subtracting 1. Removed `-1` shift.
2. **Label off-by-one** in `core_app/data/video_dataset.py:166` — same fix for consistency.
3. **Validation loss always 0.0** in `core_app/mot/trainer.py:700-703` — DETR head gates loss on `self.training`. Fix: explicitly set `self.model.detr.train()` in `validate()`.
4. **NCCL SIGSEGV on L40S** — set `NCCL_P2P_DISABLE=1`, `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_BLOCKING_WAIT=1` before `dist.init_process_group`.
5. **Optimizer state mismatch** when resuming single-GPU checkpoint on multi-GPU — use `--reset-optimizer` flag.

### Bugs fixed (Jul 2026 — Stage 3 near-zero mAP)
6. **Encoder checkpoint loading broken** in `core_app/models/vjepa_world_model.py:388` — `torch.load(weights_only=True)` loaded full checkpoint dict (5 top-level keys) instead of model state dict. Fix: detect Stage 1/2 format (`dict with 'model'`), extract `encoder.encoder.*` keys, strip prefix, convert `.base.` → `.` for LoRA compat.
7. **Stage 2 checkpoint never loaded for Stage 3** in `core_app/mot/trainer.py` — `meta.load_checkpoint` only read for `stage2_jepa`. Fix: added `_load_stage2_weights_for_joint()` method, called when `stage == 'stage3_joint'`.
8. **Missing `encoder_lora` in Stage 3 config** — Stage 1/2 use LoRA (rank=16, alpha=32) on qkv/proj; Stage 3 had no LoRA config, causing all encoder weights to be skipped. Fix: added matching `encoder_lora` config.
9. **Missing `use_deformable_detr` in Stage 3 config** — Stage 1/2 use Deformable DETR; Stage 3 built standard DETR with different key names. Fix: added `use_deformable_detr: true` and matching denoising params.
10. **Stage 3 epochs too low** — 10 epochs insufficient for DETR convergence. Fix: increased to 50 epochs, warmup 2, added `lora_lr: 1.0e-4`.
- Verification: dry-run confirmed 0 missing keys, 0 unexpected keys when loading Stage 2 checkpoint into fixed Stage 3 model.

### Stage 2 preparation (complete)
- New configs created with fixed mismatches: `dinov2_vits14` → `dinov2_vitb14`, `num_queries: 16` → `48`, `load_checkpoint` → `cholec20-stage1-surgenet`.
- Build script: `scripts/got_jepa/build_ssl_corpus_3gpu.sh` (uses SurgeNetDINO config, `endofm-lv` env, `score_threshold=0.25`).
- JEPA code verified: `core_app/mot/jepa.py` (GOTJEPAWrapper, JEPAProjector, JEPAExpander), `core_app/mot/augment.py` (SurgicalCorruption), `core_app/mot/trainer.py` (`_step_stage2_jepa`, `_setup_jepa_wrapper`, manual DDP gradient sync).

### Next steps
1. Make RF-DETR the Stage-1 detection foundation or use it directly to rebuild high-quality pseudo-labels; do not treat the epoch-185 custom-DETR checkpoint (`best_val_map=0.0270`) as a strong teacher.
2. If retaining the custom Deformable DETR, diagnose it against RF-DETR with a matched split/evaluator and runtime-check the per-level 2D deformable-attention path before more long training.
3. Rebuild SSL corpus: `bash scripts/got_jepa/build_ssl_corpus_3gpu.sh`.
4. Re-run Stage 2 from the new Stage 1 checkpoint: `torchrun --standalone --nproc_per_node=3 -m core_app.mot.main --fname configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-surgenet.yaml --devices cuda`.
5. Re-run Stage 3 joint fine-tune from that Stage 2 checkpoint.
6. Run and persist full HOTA/MOTA/IDF1 evaluation on the CT20 validation/test protocol before considering Stage 4.

### Skills and workflows
- Windsurf skills/workflows synced from `agent-skills-fresh` repo (34 skills, 34 workflows) at `.windsurf/`
- Cursor skills in `.cursor/skills/` and commands in `.cursor/commands/`

### Architecture notes
- Stage 2 `_step_stage2_jepa` only calls `encode_frames` + `GOTJEPAWrapper`—never `model.forward()` or OccuSolver; trains student per-track predictor + ProjNet + Expander (teacher frozen).
- CoTracker is frozen inside OccuSolver when used; OccuSolver heads train only in Stage 4.
- Diagram layers vs stages: ② Gating → Stage 4; ④ JEPA → Stage 2; ⑤ Geometry → Stage 4 full; ⑥ DETR/localization → Stage 1 + 3.
- `scripts/got_jepa/train_stage1_ddp_3gpu.sh`: robust conda init, auto `torch.cuda.device_count()`, single-GPU runs `python -m core_app.mot.main --devices cuda:0`.
- LeVJEPA (arXiv:2608.27395, released 2026-08-27) is a promising compute-efficient video encoder using global/local invariance + SIGReg, 95% random token dropping, and block-causal attention, but it has not been evaluated for dense prediction or tracking. Do not use it to bypass the Stage-1 detector bottleneck or replace Stage-2 VISReg; consider only a matched frozen-backbone ablation after RF-DETR protocol parity and baseline HOTA are established.
