# GOT-JEPA Surgical Tool Tracking — Multi-Stage MOT with Object Permanence

Frozen DINOv2 encoder + per-track GOT-JEPA predictor SSL + DETR detection + ReID tracking for laparoscopic surgery (CholecTrack20 / Cholec80).

## Architecture Overview

Layered system diagram (canonical Mermaid, init block, classDefs, training boundaries): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Four-stage training pipeline that progressively builds object permanence:

```
Stage 1: Supervised Scaffold          Stage 2: GOT-JEPA SSL
  ┌─────────────────┐                    ┌─────────────────┐
  │ Frozen DINOv2   │                    │ Frozen teacher  │
  │ ViT-B/14        │ ──checkpoint──►    │ predictor (S1)  │
  │ 768-dim tokens  │                    │ + clean frame   │
  └────────┬────────┘                    └────────┬────────┘
           │                                      │
  ┌────────▼────────┐                    ┌────────▼────────┐
  │ DETR + ReID +    │                    │ Student sees      │
  │ per-track predictor│                 │ corrupted frame   │
  │ (all trainable)  │                    │ (smoke/blood/etc) │
  └────────┬────────┘                    └────────┬────────┘
           │            Inv + Cov loss              │
           │     «student ω = teacher ω̂»           │
           │                                      │
  ┌────────▼────────┐                    ┌────────▼────────┐
  │ HOTA ~35 target │                    │ HOTA +10–15 gain  │
  └─────────────────┘                    └─────────────────┘
           │                                      │
           └─────────────► Stage 3 ◄─────────────┘
                        Joint Fine-Tune
                        (unfreeze all, lr=5e-5)
                              │
                              ▼
                  Stage 4 (optional): OccuSolver + VGGT
                  explicit occlusion + 3D geometry
```

Key components at inference:
- **Frozen DINOv2 ViT-B/14** → 768-dim spatial tokens (14×14 patches @ 392px)
- **SimpleFPN neck** → multi-scale features {P3, P4, P5}
- **DETR head** → 16 learned queries detect 7 tool classes
- **Per-track predictor** → filter weights ω_k per active track
- **ReID head** → 256-dim embeddings for re-identification after occlusion
- **TrackManager** → Hungarian association + birth/death/re-entry logic

## Key Features

- **Frozen DINOv2 Encoder**: Strong ImageNet+LVD features, no gradient updates
- **GOT-JEPA Teacher-Student SSL**: Per-track predictor learns invariance to surgical corruptions (smoke, blood, blur, specular glare, cutout)
- **DETR Head**: Transformer-based detection without anchors or NMS
- **ReID Head**: Supervised contrastive embeddings for tool re-identification
- **Stage 4 (optional)**: CoTracker point-tracker visibility (OccuSolver) + VGGT 3D geometry (GOT-Edit)
- **Leak-Free SSL Corpus**: Combines CholecTrack20 (10 train videos) + Cholec80 (73 videos) while excluding CT20 val/test overlap
- **W&B Experiment Tracking**: Per-batch loss curves, clean-vs-corrupted visualisations, DETR prediction overlays, gradient-norm monitoring, and **JEPA collapse-detection alerts**

## Installation

```bash
# Clone repository
cd surgi_world_track

# Install dependencies
pip install -r requirements.txt
```

### Encoder

DINOv2 ViT-B/14 (or ViT-S/14 for L40S GPUs with limited VRAM) is loaded via `torch.hub` and frozen. No additional downloads required beyond standard `torchvision` hub caching.

V-JEPA 2.1 is also supported as an alternative encoder (set `encoder_type: vjepa` in config), but DINOv2 is recommended for this task.

## Datasets

### CholecTrack20 (primary — annotated)
```
cholectrack20/
├── Training/
│   ├── VID02/
│   │   ├── Frames/
│   │   │   ├── 006701.png
│   │   │   └── ...
│   │   └── vid02.json          # {frame_id: [{instrument, tool_bbox, intraoperative_track_id}]}
│   └── ... (VID04, VID11, VID13, VID17, VID23, VID31, VID37, VID96, VID103)
├── Validation/
│   ├── VID30/
│   └── VID110/
└── Testing/
    ├── VID01/
    └── ... (VID06, VID07, VID12, VID25, VID39, VID92, VID111)
```

### Cholec80 (secondary — pseudo-labels for SSL)
```
cholec80/
└── frames/
    ├── video01/
    │   ├── video01_000005.png
    │   └── ...
    └── ... (video01..video80)
```

**Overlap warning**: CholecTrack20 is a subset of Cholec120 (= Cholec80 ∪ 40 extra videos). Videos VID01–VID80 are identical to Cholec80 video01–video80. We exclude any Cholec80 video that overlaps CT20 val/test from SSL pretraining to prevent data leakage. See `core_app/data/splits.py` for the canonical split definitions.

## Training Pipeline

See `docs/TRAINING_STAGES.md` for the full object-permanence rationale. Quick reference:

### Stage 1 — Supervised Scaffold (CholecTrack20 only)
```bash
PYTHONPATH=. python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
  --devices cuda:0
```

### Stage 2 — GOT-JEPA SSL (CholecTrack20 + Cholec80)

**Step 2a**: Build the leak-free SSL corpus with pseudo-labels
```bash
PYTHONPATH=. python -m scripts.build_ssl_corpus \
  --stage1_config configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
  --stage1_checkpoint outputs/mot/cholec20-stage1-supervised/best.pth.tar \
  --out_root /scratch/kcwp264/data/surgi_world_track/ssl_corpus \
  --device cuda:0 \
  --min_track_streak 2 \
  --min_track_total_hits 2 \
  --min_track_streak_by_class '{"0": 2, "1": 2}'
```

`build_ssl_corpus` applies two optional pseudo-label stability knobs:
`--min_track_streak` (default 2) and `--min_track_total_hits` (default 2) to suppress single-frame flicker boxes, with optional per-class overrides via
`--min_track_streak_by_class`.

**Step 2b**: Run SSL pretraining
```bash
PYTHONPATH=. python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain.yaml \
  --devices cuda:0
```

### Stage 3 — Joint Fine-Tune (CholecTrack20 only)
```bash
PYTHONPATH=. python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml \
  --devices cuda:0
```

### Stage 4 — Full Stack (optional)
```bash
PYTHONPATH=. python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage4-full.yaml \
  --devices cuda:0
```

### Interactive/Debug Mode
```bash
PYTHONPATH=. python -m core_app.mot.main \
  --fname configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
  --devices cuda:0 \
  --debugmode True    # truncates to 32 train / 16 val clips
```

## Configuration

All configs live under `configs/train_mot/dinov2/`. Key files:

| Config | Stage | Data | Notes |
|---|---|---|---|
| `cholec20-mot-stage1-supervised.yaml` | 1 | CT20 train (10 vids) | Scaffold — DETR+ReID+predictor |
| `cholec80-ct20-stage2-jepa-pretrain.yaml` | 2 | CT20 train + C80 (73 vids) | SSL — ViT-S (`dinov2_vits14`), teacher-student |
| `cholec20-mot-stage3-joint-finetune.yaml` | 3 | CT20 train (10 vids) | Unfreeze all, lr=5e-5 |
| `cholec20-mot-stage4-full.yaml` | 4 | CT20 train | + OccuSolver + VGGT geometry |

Stage 2 config key parameters:
```yaml
meta:
  stage: stage2_jepa
  load_checkpoint: outputs/mot/cholec20-stage1-supervised/best.pth.tar

losses:
  jepa_inv_weight: 1.0    # invariance: student ω = teacher ω̂
  jepa_cov_weight: 0.5    # covariance: prevent representational collapse

augmentation:
  smoke_p: 0.45   # electrocautery plume
  blood_p: 0.25   # tool-tissue contact
  blur_p:  0.35   # fouled lens / motion blur
```

## Project Structure

```
surgi_world_track/
├── core_app/
│   ├── mot/                      # Main MOT pipeline
│   │   ├── system.py             # SurgicalMOTSystem (encoder + neck + DETR + predictor + ReID)
│   │   ├── trainer.py            # MOTTrainer (4-stage dispatch)
│   │   ├── data.py               # MOTCholecDataset + mot_collate_fn
│   │   ├── jepa.py               # GOTJEPAWrapper (teacher-student inv+cov)
│   │   ├── predictor.py          # PerTrackModelPredictor
│   │   ├── manager.py            # TrackManager (Hungarian + birth/death/re-entry)
│   │   ├── augment.py            # SurgicalCorruption (smoke, blood, blur, ...)
│   │   ├── occusolver.py         # OccuSolver (CoTracker + visibility gating)
│   │   ├── geometry.py           # VGGT + GatedFusion + NullSpaceEditor
│   │   └── main.py               # Entry point --fname <config>
│   ├── models/
│   │   ├── detr_head.py          # DETR transformer decoder
│   │   ├── reid_head.py          # ReID + SupConLoss
│   │   ├── vjepa_world_model.py  # V-JEPA / DINOv2 encoder wrappers
│   │   ├── vision_transformer.py # ViT utilities
│   │   └── fpn.py                # SimpleFPN / EncoderNeck
│   ├── data/
│   │   ├── video_dataset.py      # Base CholecDataset
│   │   └── splits.py             # CT20/CT80 canonical split definitions + leak check
│   ├── trainers/
│   │   └── world_model_trainer.py# V-JEPA 2.1 world model trainer (legacy)
│   └── utils/
│       ├── checkpoint.py         # Save/load
│       └── metrics.py            # MetricLogger
├── configs/
│   ├── train_mot/dinov2/         # Stage 1–4 MOT configs
│   ├── train_2_1/vitb16/         # V-JEPA 2.1 world model configs (legacy)
│   └── splits/
│       └── ct20_c80_ssl_splits.yaml  # Human-readable split rationale
├── scripts/
│   ├── build_ssl_corpus.py       # Pseudo-label Cholec80 → unified SSL corpus
│   ├── run_mot_stage4.sh         # Stage 4 launcher
│   └── run_world_model.sh        # V-JEPA 2.1 launcher (legacy)
├── docs/
│   ├── ARCHITECTURE.md           # Canonical Mermaid system diagram
│   ├── TRAINING_STAGES.md        # Full stage-by-stage guide
│   └── multi_object_tracking_research.md
└── tests/
    └── test_mot_smoke.py         # CPU-only smoke tests
```

## How It Works

### Stage 1: Supervised Scaffold
Trains DETR detector + per-track predictor + ReID head on clean, annotated CholecTrack20 frames. Target HOTA > 35 before proceeding.

### Stage 2: GOT-JEPA Teacher-Student SSL
- **Teacher** = frozen Stage 1 predictor, sees clean current frame
- **Student** = trainable copy, sees corrupted current frame (smoke, blood, blur, specular glare, cutout)
- **Loss**: invariance (student ω = teacher ω̂) + covariance (prevent collapse)
- **Result**: student learns that its tool model is the same whether the view is clear or occluded

### Stage 3: Joint Fine-Tune
Reloads the SSL-trained student predictor and unfreezes the full pipeline (DETR + ReID + predictor + neck). Low LR (5e-5) refines alignment. Target HOTA > 50.

### Stage 4 (Optional): Explicit Occlusion + Geometry
- **OccuSolver**: CoTracker point tracker → per-point visibility map gates features before localization
- **VGGT**: single-view 3D geometry (depth, pose, point maps) fused into filter weights via null-space editing
- Only worth running if Stage 3 HOTA > 50 and you need the hardest occlusion sequences.

## Loss Functions

### Stage 1 / 3: Supervised MOT
```
L = L_det + λ_track · L_track + λ_reid · L_reid

L_det   = focal(class_logits) + L1(boxes) + GIoU(boxes)
L_track = hinge(score_map) + GIoU(regressed_boxes)
L_reid  = SupCon(embeddings, track_ids)   # supervised contrastive
```

### Stage 2: GOT-JEPA SSL
```
L = α · L_inv + β · L_cov

L_inv = MSE(ω_student, ω_teacher)          # invariance to corruption
L_cov = Σ_{i≠j} Cov(ω_exp)_{i,j}²          # off-diagonal covariance penalty (VICReg)
```

### Stage 4 (optional)
```
L = L_stage3 + γ · L_occu + δ · L_consist

L_occu    = BCE(VisHead, GT_visibility)
L_consist = cos_sim(ω_sem, ω_sem+geo)      # prevent geometry overwriting semantics
```

## Training Tips

1. **Do not skip Stage 1**: Stage 2 teacher is a deep-copy of the Stage 1 predictor. An untrained teacher produces garbage pseudo-labels and SSL collapses.
2. **Pseudo-label noise is regularization**: Cholec80 pseudo-boxes are ~5–15% wrong. This is fine — the covariance loss prevents collapse.
3. **Corruption probabilities matter**: If `jepa_cov` explodes, raise `smoke_p`/`blur_p` so the student actually sees corruption.
4. **L40S batch sizes**: Stage 1/3 → batch_size=4 with DINOv2-S/14. Stage 2 → batch_size=4 (two encoder forwards). Stage 4 → batch_size=1 (VGGT + CoTracker are heavy).
5. **Debug mode**: `--debugmode True` truncates to 32 train / 16 val clips for fast iteration.

## References

- **GOT-JEPA** (TCSVT 2026): https://arxiv.org/abs/2602.14771 — teacher-student predictor SSL for tracking
- **GOT-Edit** (ICLR 2026): https://arxiv.org/abs/2602.08550 — null-space geometric editing (Stage 4)
- **DINOv2** (TMLR 2023): https://arxiv.org/abs/2304.07193 — frozen visual backbone
- **CholecTrack20** (CVPR 2025): https://arxiv.org/abs/2312.07352 — annotated MOT dataset
- **CoTracker** (ECCV 2024): https://arxiv.org/abs/2307.07635 — point tracking (Stage 4)
- **VGGT** (CVPR 2025): https://github.com/facebookresearch/vggt — 3D geometry from single view (Stage 4)
- **V-JEPA 2** (2025): https://arxiv.org/abs/2506.09985 — alternative encoder (supported but not default)

## License

MIT License
