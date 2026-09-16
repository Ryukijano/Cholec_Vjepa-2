# Multi-Object Surgical Tracking Research & Design

**Session Date**: April 17–18, 2026  
**Objective**: Design a robust multi-object surgical tool tracking system integrating GOT-JEPA, GOT-Edit, and OccuSolver into the existing V-JEPA/DETR-based pipeline.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [GOT-Edit Architecture Analysis](#got-edit-architecture-analysis)
3. [GOT-JEPA Architecture Analysis](#got-jepa-architecture-analysis)
4. [Multi-Object Tracking Adaptation](#multi-object-tracking-adaptation)
5. [DINOv2 vs V-JEPA Analysis](#dinov2-vs-vjepa-analysis)
6. [CholecTrack20 Dataset Analysis](#cholectrack20-dataset-analysis)
7. [Proposed System Architecture](#proposed-system-architecture)
8. [Training Roadmap](#training-roadmap)
9. [Computational Cost Analysis](#computational-cost-analysis)
10. [References](#references)

---

## Executive Summary

This document outlines a comprehensive design for multi-object surgical tool tracking that adapts single-object tracking paradigms (GOT-JEPA, GOT-Edit) to handle 2–7 concurrent surgical tools. The key innovations are:

- **Shared backbone with per-track filters**: DINOv2 encoder runs once per frame; each track maintains its own lightweight filter (~50k params) for localization
- **Teacher-student predictor training**: GOT-JEPA paradigm where a frozen teacher generates pseudo-tracking models from clean frames and a student learns to predict them from corrupted frames (blood, smoke, blur)
- **Null-space geometric integration**: GOT-Edit's null-space projection fuses VGGT geometric features with DINOv2 semantics without degrading discriminative power
- **Per-track occlusion handling**: OccuSolver uses CoTracker point visibility to freeze predictors under occlusion and maintain long-term memory for tool re-identification
- **Hybrid detection + tracking**: DETR handles birth/death of tracks; per-track filters handle identity continuity and discrimination of visually similar tools

The proposed system is designed for CholecTrack20 dataset with its specific challenges: 23,000 occlusion instances, 8.4× average tool re-entry rate, and severe visual corruption (smoke, specularity, fouled lens).

---

## GOT-Edit Architecture Analysis

### Core Concept

GOT-Edit (ICLR 2026) integrates 3D geometric cues into generic object tracking using an online cross-modality model editing approach. The key insight is that humans track objects using implicit 3D reasoning even from 2D video streams.

### Architecture Components

#### 1. Dual Feature Streams

```
Reference Frame ──┬──► DINOv2 ─► v_ref^s (semantic, C×H×W)
                  └──► VGGT    ─► v_ref^g (geometric, C'×H'×W')

Current Frame   ──┬──► DINOv2 ─► v_cur^s
                  └──► VGGT    ─► v_cur^g
```

- **DINOv2**: Provides semantic features (what the object is)
- **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper): Infers depth, camera pose, point maps from 2D RGB frames

#### 2. Gated Fusion

```
m = sigmoid(Conv([v^s, Align(v^g)]))   # spatial gating mask
z = m ⊙ v^s + (1-m) ⊙ Align(v^g)       # per-pixel blend
```

A convolution predicts a spatial gating mask for each location, choosing between semantic and geometric features.

#### 3. Dual Model Predictors (from ToMP)

- **Semantic predictor**: Generates `W_sem` from DINOv2 features only
- **Geometry predictor**: Generates perturbation `Δ` from fused semantic+geometric features

Both predictors output weight vectors used by the localization head.

#### 4. Null-Space Constrained Online Editing

**The secret sauce**: Project geometric perturbations into the null space of semantic features to preserve semantic knowledge:

```
Δ' = P_null · Δ        where P_null · z_sem = 0
W_final = W_sem + Δ'
```

**Null-space projection via SVD**:
1. Whiten semantic features → `Z`
2. Build regularized correlation matrix `M = Z Z^T + λI`
3. SVD `M = U Σ V^T`
4. `P_null = U_null U_null^T` using low-energy eigenvectors
5. Symmetrize: `P_null = (P̂ + P̂^T) / 2`

This ensures geometric updates only add information in directions orthogonal to semantic subspace.

### Relevance to Surgical Tracking

| Surgical Challenge | GOT-Edit Solution |
|---|---|
| Tool occlusion by tissue | VGGT depth → tool boundary even when partially hidden |
| Distractor tools (2 graspers) | 3D position disambiguates "which grasper is which" |
| Specular reflections | Geometric features more stable than texture-only |
| Re-identification across frames | Null-space editing preserves ReID semantics while adding 3D pose cues |

---

## GOT-JEPA Architecture Analysis

### Core Concept

GOT-JEPA (TCSVT 2026) extends the JEPA (Joint-Embedding Predictive Architecture) paradigm from image-feature prediction to tracking-model prediction. It teaches the model predictor to generate robust tracking models under corruption.

### Architecture Components

#### 1. Teacher-Student Predictor Training

```
                         ┌─► Clean current frame ──► t-Predictor ──► ω̂ (teacher weights, frozen)
   Reference history ────┤                                              │
                         │                                              │  invariance loss
                         │                                              ▼
                         └─► Corrupted current frame ─► s-Predictor ──► ω ──► ProjNet
                                                                        │
                                                                        ▼
                                                                     Expander ──► cov loss
```

**Two losses**:
1. **Invariance loss** (`L_inv`): Student's ω must match teacher's ω̂ — even though student saw a corrupted frame
2. **Covariance loss** (`L_cov`): Off-diagonal of Cov(ω) → 0 — prevents collapse, forces diverse patterns

**Why this works**: The student is forced to produce the *same tracking model* under occlusion/noise/distractors. This teaches the predictor that tracking identity persists through corruption.

#### 2. OccuSolver: Occlusion Specialist

Bolts CoTracker point tracker and makes it object-aware:

```
GOT bounding box ──► Prior Encoder ──► condition CoTracker
                                           │
                                           ▼
CoTracker ──► track 128 points ──► VisHead ──► per-point visibility
                                           │
                                           ▼
           Mapping (Gaussian kernel) ──► E ∈ R^(H×W×C) visibility map
                                           │
                                           ▼
           Ensemble Net fuses E with z_cur ──► z̃_cur (visibility-aware features)
```

The trick: CoTracker knows "is this point visible?" at pixel granularity. GOT provides object priors → CoTracker knows which points belong to the target. The resulting pixel-level visibility map then gates the current-frame features before they hit the localization head.

### Relevance to Surgical Tracking

| Surgical Challenge | GOT-JEPA Solution |
|---|---|
| Corruption (blood, smoke, plume) | Teacher-student training makes features invariant |
| Partial occlusion | OccuSolver provides pixel-level visibility |
| Long occlusion → re-entry | Long-term memory (EMA) preserves identity |
| Unseen tools | Predictor learns tracking as general skill, not memorization |

---

## Multi-Object Tracking Adaptation

### Paradigm Comparison

| Paradigm | How tracks persist | Pros | Cons |
|---|---|---|---|
| **Track queries** (MOTR, MeMOTR) | Learned query embeddings propagate through decoder | End-to-end, no motion model, natural birth/death | Weak semantics → poor discrimination of similar objects |
| **Per-track filters** (DiMP, ToMP, GOT-JEPA) | Each track has its own predicted correlation filter | Discriminative per track, handles long occlusion | Association step needed |
| **Hybrid** (proposed) | DETR for detection + per-track filter for identity | Best of both: DETR handles birth/death, filters handle identity | Slightly more moving parts |

**Hybrid wins for Cholec20**: MOTR struggles with visually-similar objects because track queries don't encode discriminative features. A per-track filter trained discriminatively can distinguish them.

### Concrete Architecture

#### Forward Pass (per frame)

```
Input: frame I_t, active_tracks = [(id_1, ω_1, memory_1), ..., (id_K, ω_K, memory_K)]

1. Shared perception (run ONCE):
   v_sem  = DINOv2(I_t)                  # (B, 576, 768)
   v_geo  = VGGT(I_t)                   # (B, H', W', C')
   z      = GatedFusion(v_sem, v_geo)   # (B, C, H, W)
   pts    = CoTracker(I_t, prev_pts)    # (B, N, 3)

2. Detection branch (DETR, run ONCE):
   det_queries  = Q_det + learned_pos_enc
   det_boxes, det_logits, det_feats = DETR(z, det_queries)

3. Per-track localisation (run K times):
   for k in active_tracks:
       p_k = ω_k ⊛ z                        # score map (H, W)
       box_k_pred = peak(p_k) → RegDec
       vis_k = OccuSolver(box_k_pred, pts)
       Δ_k    = GeomPredictor(RoI(v_geo, box_k_pred))
       Δ_k'   = P_null · Δ_k
       ω_k_new = ω_k + Δ_k'
       if vis_k > 0.3:
           ω_k = JEPAStudentPredictor(RoI(z, box_k_pred), ref_labels)

4. Association:
   cost[i,j] = λ_iou · (1 - IoU(det_i, pred_j))
             + λ_cls · (1 - sim(det_i.class, pred_j.class))
             + λ_reid · (1 - cos(ReID(det_i), memory_j))
             + λ_vis · (1 - vis_j)
   matches, unmatched_dets, unmatched_tracks = hungarian(cost)

5. Track management:
   - Matched → update track memory (EMA of ReID features)
   - Unmatched det → spawn new track
   - Unmatched track → increment age, kill if age > 30 frames
```

#### Key Design Points

**Shared predictor networks, per-track weights**:
```
ω_k = StudentPredictor(z_t, ref_labels_k)
```
Same function, per-track inputs, per-track outputs. Memory footprint stays flat.

**Shared CoTracker, per-track point filtering**:
- Run CoTracker once with ~256 shared points
- Per track, filter points by "are you inside this track's current box?" → subset to ~32–64 points
- VisHead operates on the subset → per-track visibility score

#### Tensor Shapes

```
Shared (once per frame):
  v_sem    : (1, 576, 768)          DINOv2 patch tokens
  v_geo    : (1, 196, 768)          VGGT features
  z        : (1, 256, 28, 28)       fused feature map
  pts      : (1, 128, 3)            CoTracker points

Per-track (K times, K ≤ 8):
  ω_k      : (256,)                 correlation filter weights
  memory_k : (256,)                 EMA of ReID features
  box_k    : (4,)                   current bounding box
  vis_k    : scalar                 visibility ∈ [0, 1]
```

---

## DINOv2 vs V-JEPA Analysis

### Why DINOv2 wins for MOT

1. **Cleaner per-patch localisation**: DINOv2 produces crisp, localised per-patch features. V-JEPA's predictor is holistic — individual patches can be blurrier.
2. **Single-frame inference**: Matches MOT's per-frame paradigm. No temporal collapse, no window management.
3. **GOT-Edit reference design**: The null-space editing paper literally uses DINOv2 semantics.
4. **Better for ReID**: MOT benchmarks overwhelmingly pick DINOv2 for appearance-based re-identification.
5. **Your supervised Cholec20 variant**: Fine-tuned features already aligned to surgical tools.

### What you lose by dropping V-JEPA

1. **Temporal priors in the encoder**: V-JEPA's features encode temporal consistency. But the per-track filter *is* your temporal model.
2. **World model / multi-horizon predictor**: The patch-level future prediction becomes questionable. The per-track filter replaces this role.
3. **Motion robustness**: Mitigated by GOT-JEPA's corruption augmentation during training.

### Updated Stack

| Block | Previous (V-JEPA) | Updated (DINOv2) |
|---|---|---|
| Encoder | V-JEPA 2.1 ViT-B (spatiotemporal) | **DINOv2 ViT-B/14** (spatial) |
| Input | 16-frame clip | Single frame |
| Temporal tokens | 8 (tubelet=2) | 1 |
| Neck | VJEPANeck (temporal collapse) | **SimpleFPN only** |
| Predictor | MultiScaleTemporalPredictor + rollout | **GOT-JEPA per-track filter predictor** |
| World model loss | MSE per horizon + rollout MSE | **Gone** — replaced by per-track hinge loss |
| Phantom for DETR | patch-level future features | **RoI-level predicted features** |
| ReID | SupCon on joint real+phantom | SupCon on real RoI + memory (EMA) |

### Recommendation

Fine-tune DINOv2 on Cholec20 using the same supervised recipe used for V-JEPA. Then:
- DINOv2-Cholec20 as shared encoder
- Drop V-JEPA from pipeline
- Drop multi-horizon world model loss
- Replace temporal predictor with GOT-JEPA per-track filter predictor
- Keep VGGT + null-space + OccuSolver + DETR + ReID

---

## CholecTrack20 Dataset Analysis

### Key Statistics

| Stat | Value | Implication |
|---|---|---|
| Avg tools/frame | ~2 | Per-track overhead is trivial |
| Max concurrent tools | 3–4 (up to 7 classes) | Buffer ~8 track slots |
| Total trajectories | ~2000 across 20 videos | Per-track identities plentiful for SupCon |
| Occlusion instances | 23,000 (~65% of frames) | OccuSolver is not optional |
| Tool re-entries (graspers) | 8.4× avg | Long-term ReID memory is critical |
| Avg on-screen duration | 25s (graspers) | Short tracks → predictor must adapt fast |
| Smoke instances | 2,000 | JEPA corruption augmentation directly motivated |
| Fouled lens frames | 2,196 | Same |

### Design Implications

- **Sparse but severely corrupted**: Not MOT17-style crowd tracking. Skews design away from MOTR's query-propagation toward detector + per-track filter + strong ReID.
- **Frequent object death/rebirth**: Long-term memory (MeMOTR-style) needed for re-identification across re-entries.
- **Occlusion dominates**: OccuSolver visibility gating is critical for freezing predictors under occlusion.

### Track Birth/Death Policy

- **Birth**: Unmatched detection with confidence > 0.6 for 3 consecutive frames
- **Death**: Track unmatched for 30 frames AND vis < 0.1 for 15 of those
- **Long re-entry**: Keep dead-track memory for 300 frames; match new detections against dead-track memory via ReID

---

## Proposed System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FROZEN DINOv2 ViT-B/14                       │
│                    (86M params, 768-d)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleFPN Neck                                │
│                    (256-d, multi-scale)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   DETR Head     │ │ Per-Track       │ │   VGGT          │
│   (newborn      │ │ JEPA Predictor  │ │   (shared,      │
│    detection)   │ │   (K filters)   │ │    fused via    │
└─────────────────┘ └─────────────────┘ │    null-space)  │
                              │           └─────────────────┘
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                              ┌─────────────────┐
                              │ Hungarian       │
                              │ Association     │
                              └─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Track           │
                              │ Management      │
                              │ (birth/death,   │
                              │  memory)        │
                              └─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ RoIAlign → ReID │
                              │ → SupCon        │
                              └─────────────────┘
```

### Component Details

#### Shared Perception (once per frame)
- **DINOv2 ViT-B/14**: 86M params, frozen or LoRA-fine-tuned on Cholec20
- **VGGT**: ~300M params, frozen with light LoRA fine-tune
- **CoTracker**: 60M params, frozen with LadderSide fine-tune
- **Gated Fusion**: 0.3M params, trainable
- **SimpleFPN**: 1M params, trainable

#### Per-Track Components (shared across tracks)
- **JEPA Student Predictor**: 0.5M params, shared hypernetwork
- **Geom Predictor**: 0.3M params, shared hypernetwork
- **Null-Space Editor**: SVD on-the-fly, no params
- **ReID Head**: 0.3M params, shared

#### Detection Components
- **DETR Head**: 5M params, trainable
- **Hungarian Association**: Non-learned

### Diagrams

- **Canonical Mermaid (spec-aligned layers, losses, recursion, Hungarian boundary):** `docs/ARCHITECTURE.md`
- **Legacy SVG stack:** `docs/proposed_system_stack.svg`

---

## Training Roadmap

### Stage 1: Detector Warm-up (1–2 weeks)

Train DETR on CholecTrack20 detection labels (bbox + class). No tracking yet.

```
L_det = FocalLoss + L1 + GIoU
```

### Stage 2: GOT-JEPA Pretraining of Predictor (1 week)

**Input per batch**: clip of 4 frames, list of (track_id, bbox) tuples.

```
for clip in dataloader:
    ref_frames, cur_frame, track_annots = clip
    
    z_ref = DINOv2(ref_frames)
    z_cur_clean = DINOv2(cur_frame)
    z_cur_dirty = DINOv2(augment(cur_frame))  # smoke, specular, blur, cutout
    
    loss = 0
    for tid, bbox_ref, bbox_cur in track_annots:
        p_a, p_b = gaussian_heatmap(bbox_ref_0), gaussian_heatmap(bbox_ref_1)
        
        with no_grad:
            ω_hat = TeacherPredictor(z_ref, p_a, p_b, z_cur_clean)
        
        ω = StudentPredictor(z_ref, p_a, p_b, z_cur_dirty)
        
        L_inv = mse(ω, ω_hat)
        L_cov = covariance_loss(Expander(ω))
        
        loss += α * L_inv + β * L_cov
    
    loss /= len(track_annots)
    loss.backward()
```

Teacher is frozen throughout (initialized from Stage 1 DETR's decoder). Student learns robust model prediction.

### Stage 3: Joint Detection + Per-Track Filter Fine-tuning (1 week)

Train DETR detection + per-track localization end-to-end on short clips (4 frames):

```
L_total = L_det
        + λ_track · L_track  (hinge loss on per-track response maps)
        + λ_reid · L_reid    (SupCon on RoI features with track IDs)
```

### Stage 4: Add VGGT + Null-Space Editor + OccuSolver (2 weeks)

Plug in geometry branch with null-space projection, wire OccuSolver into visibility gating:

```
L_total = L_det + L_track + L_reid
        + λ_occu · L_occu  (BCE on visibility)
        + λ_consist · cos_sim(ω_with_geo, ω_without_geo)
```

### Association Cost Weights

Empirically tuned for surgical MOT:
- IoU: 0.3
- ReID: 0.45 (up-weighted for similar tools)
- Class: 0.15
- Visibility: 0.1

---

## Computational Cost Analysis

### Forward-Pass FLOPs per Frame (approximate)

| Component | Params | FLOPs/frame | Runs |
|---|---:|---:|---|
| DINOv2 encoder | 86M | ~120 GFLOPs | 1× |
| VGGT (if base) | ~300M | ~200 GFLOPs | 1× |
| Gated Fusion | 0.3M | ~0.5 GFLOPs | 1× |
| DETR decoder | 5M | ~3 GFLOPs | 1× |
| JEPA predictor | 0.5M | ~0.1 GFLOPs | K× (up to 8) |
| Geom predictor | 0.3M | ~0.05 GFLOPs | K× |
| Null-space SVD | — | ~0.01 GFLOPs | K× |
| CoTracker | 60M | ~30 GFLOPs | 1× |
| ReID head | 0.3M | ~0.2 GFLOPs | K× |
| **Total (K=4)** | **~450M** | **~355 GFLOPs** | |

**Conclusion**: VGGT and CoTracker dominate cost. Per-track machinery is negligible (~1 GFLOP total for 4 tracks). Per-track scaling is essentially free up to K=20.

**Real-time feasibility**: At 25 FPS on a single A100 (~312 TFLOPs bf16), budget per frame is ~12.5 TFLOPs — way more than needed. 15–20 FPS is realistic even with CoTracker.

### Training Compute

- Stage 2 (JEPA pretrain): ~100 A100-hours
- Stage 3 (joint fine-tune): ~150 A100-hours
- Stage 4 (VGGT+OccuSolver): ~200 A100-hours
- **Total: ~500 A100-hours** (~5 days on 4×A100)

---

## References

### Papers

1. **GOT-Edit**: "GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing", ICLR 2026. [arXiv:2602.08550](https://arxiv.org/abs/2602.08550)
2. **GOT-JEPA**: "GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture", TCSVT 2026. [arXiv:2602.14771](https://arxiv.org/abs/2602.14771)
3. **VGGT**: "VGGT: Visual Geometry Grounded Transformer", CVPR 2025 Best Paper. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
4. **MOTR**: "MOTR: End-to-End Multiple-Object Tracking with Transformer", ECCV 2022. [arXiv:2105.03247](https://arxiv.org/abs/2105.03247)
5. **MeMOTR**: "MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking", ICCV 2023. [arXiv:2307.15700](https://arxiv.org/abs/2307.15700)
6. **CholecTrack20**: "CholecTrack20: A Dataset for Multi-Class Multiple Tool Tracking in Laparoscopic Surgery", CVPR 2025. [arXiv:2312.07352](https://arxiv.org/abs/2312.07352)
7. **ToMP**: "Transforming Model Prediction for Tracking", CVPR 2022. [arXiv:2203.11192](https://arxiv.org/abs/2203.11192)
8. **CoTracker**: "CoTracker: It Takes Two to Track", ECCV 2024. [arXiv:2312.05176](https://arxiv.org/abs/2312.05176)

### GitHub Repositories

1. **GOT (GOT-Edit + GOT-JEPA)**: [https://github.com/chenshihfang/GOT](https://github.com/chenshihfang/GOT)
2. **VGGT**: [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)
3. **MOTR**: [https://github.com/megvii-research/MOTR](https://github.com/megvii-research/MOTR)

### Datasets

- **CholecTrack20**: Multi-class multiple tool tracking dataset for laparoscopic surgery
- **Cholec80**: Binary tool presence dataset (subset of Cholec120)

---

## Appendix: Key Design Decisions

### DINOv2 vs V-JEPA Final Decision

**Use DINOv2** as the primary semantic encoder for MOT surgical tracking. Reasons:
1. Cleaner per-patch localisation for detection + RoIAlign
2. Single-frame inference matches MOT paradigm
3. Better for ReID across long occlusions
4. GOT-Edit reference design uses DINOv2
5. Per-track filter provides temporal modeling, making V-JEPA's temporal encoding redundant

### Per-Track vs Track-Query Approach

**Use per-track filters** (hybrid with DETR for detection). Reasons:
1. Discriminative per-track features handle visually-similar tools better
2. Can freeze individual tracks under occlusion
3. Simpler to debug and interpret
4. Matches GOT-JEPA/ToMP paradigm

### Shared vs Per-Track Predictors

**Use shared predictor networks with per-track weights**. The predictor is a hypernetwork:
```
ω_k = StudentPredictor(z_t, ref_labels_k)
```
Same function, per-track inputs, per-track outputs. Memory footprint stays flat regardless of track count.

### Shared vs Per-Track CoTracker

**Use shared CoTracker with per-track point filtering**. Run CoTracker once with ~256 shared points across the frame. Per track, filter points by box containment → subset to ~32–64 points. Keeps CoTracker at O(1) cost per frame.

---

**Document Version**: 1.0  
**Last Updated**: April 18, 2026
