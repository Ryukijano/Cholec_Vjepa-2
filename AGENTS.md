## Learned User Preferences

- Stage 1 trains a precision-leaning pseudo-label teacher (`detector_only: true`), not a final detector optimized for raw mAP.
- Run training via `conda activate surgi_track`, then `bash scripts/train_stage1_ddp_3gpu.sh` from the repo root (single-GPU path uses `python -m core_app.mot.main --devices cuda:0`); set `XFORMERS_DISABLED=1` on GB10 for DINOv2.
- Prefer `surgi_track` conda env; fallback env is `surgi_world_track_cuda` if primary is missing.
- Stay GOT-JEPA surgical MOT course; VLA-JEPA and pixel desmoking are complementary only, not primary pivots.
- Stage 2 JEPA should use a frozen Stage-1 teacher checkpoint, not EMA teacher updates (matches GOT-JEPA paper).
- Use Cholec80 as unlabeled SSL corpus for Stage 2; exclude CT20 val/test overlap videos from SSL.
- Add labeled Cholec-80 to Stage 1 only on train split with leak checks; native C80 labels are tool presence, not MOT boxes.
- On NVIDIA GB10 (DGX Spark, single GPU): target `batch_size` 24–32 in stage-1 config; prefer Stage 4 lean over Stage 4 full/VGGT unless ablating geometry.
- After Stage 2 ~epoch 20, Stage 3 then Stage 4 lean is the main path; Stage 2 epochs 21–29 optional unless more SSL gains are needed.
- Use Spark for finishing baseline and memory-heavy Stage 4 full; Leeds Aire (28 nodes × 3 L40S, 6 GPUs = 2-node DDP) for HP sweeps, multi-seed, eval arrays, and SSL parallelization.

## Learned Workspace Facts

- Project root: `/home/aimsgroupuol/AIMSgeneral/Gyanateet_tracking`; four-stage MOT pipeline (Stage 1 supervised teacher → Stage 2 GOT-JEPA SSL → Stage 3 joint finetune → Stage 4 OccuSolver/geometry).
- Training status (Jun 2026, Spark): S1 complete `outputs/mot/cholec20-stage1-supervised/best.pth.tar` (ep 69, val mAP ~2.8%); S2 paused ep 20/30 `cholec80-ct20-stage2-jepa-pretrain/latest.pth.tar`; S3 complete `cholec20-stage3-joint-finetune-vits/best.pth.tar` (ep 3, val mAP ~2.4%); S4 lean complete `cholec20-stage4-lean-vits/best.pth.tar` (ep 5, latest ep 9, val mAP ~0.6%, `depth_stub: true`); S4 got-edit aborted ep 0; no training running.
- Blockers (priority): (1) no full HOTA/MOTA eval yet—smoke evals got 0 predictions (short clips + `min_hits=3` + `birth_score=0.6`); (2) weak detection (~2–3% val mAP vs CT20 baselines Deformable-DETR ~38%, YOLOv7 ~56%); (3) CT20 association/smoke/occlusion/same-class ReID; (4) Stage 4 `depth_stub: true`, Stage 2 only 20/30 epochs.
- Recommended next: full HOTA eval Stage 3 vs Stage 4 `best` with tuned `birth_score`/`min_hits`; strengthen detection (Stage 1 without `detector_only` or longer DETR tune); set `depth_stub: false` when HF Depth-Anything available; Aire for HP sweeps after baseline eval.
- Eval infra: `scripts/eval_checkpoint.py --mot-eval --stratify-smoke`, `scripts/eval_mot_hota.py`, smoke-stratified HOTA in `core_app/mot/eval.py`; CT20 test MP4 fallback when no `Frames/`; `pytest tests/test_mot_smoke.py` (24 passed).
- Stage 2 `_step_stage2_jepa` only calls `encode_frames` + `GOTJEPAWrapper`—never `model.forward()` or OccuSolver; trains student per-track predictor + ProjNet + Expander (teacher frozen) despite yaml `use_occusolver: true` and `occusolver.freeze: true`.
- CoTracker is frozen inside OccuSolver when used; OccuSolver heads (`light_trans`, `vis_head`, etc.) train only in Stage 4.
- Diagram layers vs stages: ② Gating (CoTracker3 + OccuSolver → visibility mask E + Hadamard gate) → Stage 4 (`_accumulate_per_track_losses`: `L_occu` BCE vs frozen CoTracker teacher + optional `cur_spatial` gate; infer uses OccuSolver for `TrackManager` visibility); ④ JEPA → Stage 2; ⑤ Geometry (VGGT + null-space ΔW) → Stage 4 full (`use_geometry: true`, `L_consist`); ⑥ DETR/localization → Stage 1 + 3.
- Stage-1 config: `configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml` (`batch_size` 6, `img_size` 392, `clip_length` 3, `dinov2_vits14`, deformable DETR).
- Local CholecTrack20: `data/cholectrack20`; canonical splits in `core_app/data/splits.py`; SSL excludes Cholec80 videos overlapping CT20 val/test: `video01,06,07,12,25,30,39`.
- `scripts/train_stage1_ddp_3gpu.sh`: robust conda init, auto `torch.cuda.device_count()`, single-GPU runs `python -m core_app.mot.main --devices cuda:0`.
- GB10 probe (`surgi_track`, ~124.6 GB VRAM): synthetic train-step peaks ~1.3–14.3 GB for B16–B192 (empty per-track targets; real batches use more).
