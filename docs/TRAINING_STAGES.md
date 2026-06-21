# Training Stages — Surgical MOT with Object Permanence

**Scope**: This document is the authoritative guide to the 3 (+1 optional) training stages of the Surgical MOT system. It explains *why* each stage exists, *what* it trains, *which parameters are frozen*, and *how the stages compose to produce object permanence*.

> **Repo note (Cholec_Vjepa-2):** Shell runners referenced below live under `scripts/got_jepa/`. Master plan: [docs/plans/gyanateet_mot_understanding.md](plans/gyanateet_mot_understanding.md).

**Audience**: Researchers and engineers training this pipeline on CholecTrack20 / Cholec80 (or similar).

---

## Table of Contents

1. [The Object Permanence Problem](#1-the-object-permanence-problem)
2. [How the Stages Compose](#2-how-the-stages-compose)
3. [Stage 1 — Supervised MOT Scaffold](#3-stage-1--supervised-mot-scaffold)
4. [Stage 2 — GOT-JEPA SSL Predictor Pretraining](#4-stage-2--got-jepa-ssl-predictor-pretraining)
5. [Stage 3 — Joint Fine-Tune](#5-stage-3--joint-fine-tune)
6. [Stage 4 (optional) — Full Stack with OccuSolver + Geometry](#6-stage-4-optional--full-stack-with-occusolver--geometry)
7. [Data Requirements](#7-data-requirements)
8. [Launch Commands](#8-launch-commands)
9. [Frozen vs Trainable — Full Reference Table](#9-frozen-vs-trainable--full-reference-table)
10. [Expected Metrics per Stage](#10-expected-metrics-per-stage)

---

## 1. The Object Permanence Problem

**Object permanence** (Piaget, 1954) is the understanding that *an object continues to exist even when it is not directly visible*. In surgical video, this manifests as five concrete tracking failures:

| Failure Mode | Example in Laparoscopic Surgery | Why Trackers Fail |
|---|---|---|
| **Smoke occlusion** | Electrocautery produces ~2000 annotated smoke instances in CholecTrack20 | Encoder features degrade; detector loses tool |
| **Blood splatter** | Tool contacts tissue, splatter covers lens | Appearance drifts → wrong ReID match |
| **Specular glare** | Metallic tool reflects endoscope light | Tool blobs into a bright patch, boundary lost |
| **Fouled lens** | ~2196 annotated instances in CholecTrack20 | Entire frame blurs; no detection possible |
| **Out-of-view re-entry** | Graspers re-enter frame **8.4× per average video** | Identity lost; new track ID assigned incorrectly |

A tracker that handles these correctly has learned object permanence. The three training stages are designed to build this capability **progressively**:

> **Stage 1** teaches the system *what* a tool looks like (the "see it clearly" baseline).
>
> **Stage 2** teaches the system *through* corruption — predictions must stay consistent even when the view degrades.
>
> **Stage 3** binds Stage 2's robustness back into supervised detection/tracking so the full pipeline inherits permanence.
>
> **Stage 4** (optional) adds explicit occlusion reasoning via point-tracker visibility (OccuSolver) and 3D geometric priors (VGGT).

---

## 2. How the Stages Compose

```
                       ┌─────────────────────────────────────────┐
                       │       Frozen DINOv2 encoder             │
                       │  (trained once — SSL on ImageNet+LVD)    │
                       └────────────────────┬────────────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
  ┌──────▼──────┐                    ┌──────▼──────┐                    ┌──────▼──────┐
  │   Stage 1    │  checkpoint ──► │   Stage 2    │  checkpoint ──► │   Stage 3    │
  │  supervised  │                  │   GOT-JEPA   │                  │   joint     │
  │   scaffold   │                  │   SSL        │                  │   finetune  │
  └─────────────┘                    └─────────────┘                    └─────────────┘
     20 epochs                          50 epochs                          10 epochs
  DETR+ReID+track                  student predictor only              all heads unfrozen
  HOTA > 35 target                 inv+cov loss only                   HOTA > 50 target
```

Each stage consumes the previous stage's checkpoint. Stage 2 **requires** a Stage 1 checkpoint to initialise the teacher predictor (it is a deep-copy at the start of Stage 2 — see `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:134-137`).

---

## 3. Stage 1 — Supervised MOT Scaffold

### Purpose

Train the **supervised scaffold**: DETR detector + per-track filter predictor + ReID head, using ground-truth bounding boxes and track IDs from CholecTrack20. This establishes the baseline ability to detect, localise, and identify tools *when they are clearly visible*.

### What Trains

| Module | Status | Role |
|---|---|---|
| `Dinov2EncoderWrapper` | **Frozen** | Spatial feature extraction only |
| `EncoderNeck` (SimpleFPN) | Trainable | Project 768/384-dim tokens to 256-dim neck |
| `SurgicalToolDetector` (DETR) | Trainable | Birth-only detection (boxes + classes) |
| `PerTrackModelPredictor` | Trainable | Produce filter ω for each ground-truth track |
| `ClsDec`, `RegDec` | Trainable | Decode ω into score maps and box regressions |
| `ReidHead` | Trainable | Supervised contrastive embeddings |
| `TrackManager` | No params | Inference-only |

### Losses

$$L_\text{stage1} = L_\text{det} + \lambda_\text{track} \cdot L_\text{track} + \lambda_\text{reid} \cdot L_\text{reid}$$

- **`L_det`**: DETR classification (focal) + L1 bbox + GIoU
- **`L_track`**: hinge classification on the per-track score map + GIoU on the regressed box (`TrackLocalizationLoss`)
- **`L_reid`**: supervised contrastive loss (`SupConLoss`) grouping crops from the same track ID

Default weights: `λ_track=1.0`, `λ_reid=0.5`.

### Config

`@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml`

Key fields:
```yaml
meta:
  stage: stage1_supervised
data:
  clip_length: 3            # ref_0, ref_1, current
  img_size: 392             # 28×28 patch grid for DINOv2
  batch_size: 2             # L40S: 4, A100 40GB: 8
```

### Why This Stage Matters for Object Permanence

Stage 1 does **not** yet handle occlusion — it trains on clean, well-annotated frames only. Its role is to produce a **teacher predictor** that is an expert on clean-frame tracking. Stage 2 will then distill that expertise into a corruption-robust student.

Without a strong Stage 1 checkpoint, the Stage 2 teacher provides garbage pseudo-labels and SSL collapses. **Target HOTA > 35 before proceeding to Stage 2.**

---

## 4. Stage 2 — GOT-JEPA SSL Predictor Pretraining

### Purpose

This is **the core object-permanence stage**. The student predictor learns that its output (filter weights ω) must stay **invariant** to visual corruption — i.e., "my model of this tool is the same whether I can see it clearly or not".

Stage 2 can now run on a **leak-free combined corpus** of CholecTrack20 Training (10 videos, real annotations) + Cholec80 (73 videos, pseudo-labels). This increases the SSL data volume by roughly **7×** compared to CholecTrack20 alone, training a more robust permanence prior.

### Architecture (Matches Figure 2(a) of the GOT-JEPA Paper)

```
Few-Shot Examples (ref_0, ref_1 — clean)          Current Frame (clean)
        │                                                │
        └────────────────┬───────────────────────────────┘
                         ▼
                ┌─────────────────┐
                │  t-Predictor    │  ← FROZEN (deep-copied from Stage 1)
                │   (teacher)     │     runs under torch.no_grad()
                └────────┬────────┘
                         │ ω̂  (stop-gradient target)
                         ▼
                   [ Objectives ] ←── Inv Loss = MSE(ω, ω̂)
                         ▲
                         │ ω
                         │
                ┌────────┴────────┐
                │    ProjNet      │  ← trainable (linear 256→256)
                └────────┬────────┘
                         │ ω_raw
                         │
                ┌────────┴────────┐
                │  s-Predictor    │  ← trainable (same architecture as teacher)
                │   (student)     │
                └────────┬────────┘
                         │
        ┌────────────────┴───────────────────────────────┐
        │                                                │
Few-Shot Examples (ref_0, ref_1 — clean)   Corrupted Current Frame
                                           (smoke/blood/blur/spec/cutout)

ProjNet output ω also feeds →  ┌────────┐
                                │Expander│  ← trainable (1×1 conv, 4× channels)
                                └────┬───┘
                                     │ ω_exp
                                     ▼
                              Cov Loss = off-diag(cov(ω_exp))²
```

### What Trains

| Module | Status | File |
|---|---|---|
| `Dinov2EncoderWrapper` | **Frozen** | `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/models/vjepa_world_model.py:381-385` |
| `EncoderNeck` | **Frozen** (loaded from Stage 1) | — |
| `DETR`, `ReID`, `ClsDec`, `RegDec` | **Frozen** (loaded from Stage 1) | — |
| **`teacher` predictor** | **Frozen** (deep-copy of student at start of stage) | `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:148-151` |
| **`student` predictor** | Trainable | `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:180-181` |
| **`JEPAProjector` (ProjNet)** | Trainable | `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:45-55` |
| **`JEPAExpander`** | Trainable | `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:58-79` |

Trainable parameter count: **~1–2 M** (vs ~100M full system). This is why Stage 2 is **fast per step but memory-dense** — we still forward the frozen encoder twice (clean + corrupt).

### Losses

$$L_\text{stage2} = \alpha \cdot L_\text{inv} + \beta \cdot L_\text{cov}$$

where

$$L_\text{inv} = \frac{1}{n} \sum_{i=1}^n \| \omega_i - \hat{\omega}_i \|_2^2$$

$$L_\text{cov} = \sum_{i \ne j} \text{Cov}(\omega_\text{exp})_{i,j}^2$$

- `α = 1.0`, `β = 0.5` (defaults, from `@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec20-mot-stage2-jepa-pretrain.yaml:63-64`)
- `L_inv` drives *invariance* — student ω must equal teacher ω̂
- `L_cov` prevents *representational collapse* (VICReg-style off-diagonal covariance penalty; see `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/jepa.py:92-110`)
- Teacher gradients are explicitly blocked via `.detach()` **and** `torch.no_grad()` (double safety)

### The Corruption Bank (Student-Only)

Applied ONLY to the last temporal slice of the dirty branch:

```@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/trainer.py:241-244
clean_video = current_video
dirty_video = current_video.clone()
last = dirty_video[:, :, -1]
dirty_video[:, :, -1] = self.corruption(last)
```

`SurgicalCorruption` (see `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/augment.py:207-239`) samples from six operations per frame, each with independent Bernoulli probability:

| Corruption | Default Prob | Simulates |
|---|---|---|
| `smoke` | 0.4 | Electrocautery plume |
| `blood_splatter` | 0.2 | Tool-tissue contact |
| `specular_highlights` | 0.3 | Metallic glare |
| `defocus_blur` | 0.3 | Autofocus hunting, motion blur |
| `color_jitter` | 0.5 | Illumination variation |
| `cutout` | 0.2 | Partial occlusion by tissue |

**Crucial design detail**: only the *current* frame is corrupted. The *reference* frames (ref_0, ref_1) are identical on both teacher and student branches. This is what forces the student to learn *"my model of the tool depends on the clean history, not on the degraded current observation"* — the algorithmic definition of object permanence.

### Why This Is Object Permanence

The loss objective literally says: **produce the same tool model whether you see the tool clearly or not**. After 50 epochs of this, the student predictor has internalised that its output should be a function of the track's *historical identity* rather than the current frame's *pixel noise*. This is exactly what enables tracking through occlusion.

### Config

Two configs exist:

**A. CholecTrack20 only** (original, minimal):  
`@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec20-mot-stage2-jepa-pretrain.yaml`

**B. CholecTrack20 + Cholec80 (combined, recommended)**:  
`@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain.yaml`

```yaml
meta:
  stage: stage2_jepa
  load_checkpoint: outputs/mot/cholec20-stage1-supervised/best.pth.tar   # REQUIRED
data:
  datasets:
    - /scratch/kcwp264/data/surgi_world_track/ssl_corpus   # unified corpus
  batch_size: 4     # doubled encoder forward — memory-dense
optimization:
  lr: 5.0e-4        # higher than Stage 1 (only small tail unfrozen)
  epochs: 30         # 75 videos × ~200 frames → ample data
```

The combined config uses higher corruption probabilities (`smoke_p: 0.45`, `blood_p: 0.25`, `blur_p: 0.35`) because pseudo-label noise provides additional regularisation — stronger corruption forces the student to rely on historical identity rather than current pixels.

### Data Requirement (Critical)

Stage 2 as-coded needs **per-track bounding boxes** for the reference frames to build the Gaussian label encodings:

```@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/mot/trainer.py:270-278
heat0 = gaussian_label_encoding(
    sample.ref_bbox_0.to(self.device).view(1, 4), H_pred, W_pred
).view(1, -1, 1)
heat1 = gaussian_label_encoding(
    sample.ref_bbox_1.to(self.device).view(1, 4), H_pred, W_pred
).view(1, -1, 1)
```

Therefore:
- **CholecTrack20** → works out of the box (real annotations).
- **Cholec80** → needs **pseudo-bboxes** from the Stage 1 detector. The `build_ssl_corpus.py` script automates this:
  1. Runs the Stage 1 detector + TrackManager on each Cholec80 frame
  2. Keeps boxes with score ≥ threshold (default 0.5)
  3. Assigns per-video incremental track IDs
  4. Writes CT20-format JSON annotations
  5. Symlinks frames into a unified `ssl_corpus/Training/` layout

**Pseudo-label noise is a feature, not a bug.** Noisy boxes (~5–15% error rate) act as additional regularisation — the student cannot overfit to exact box positions and must instead learn that the tool identity is robust to positional jitter. The covariance loss (`jepa_cov_weight: 0.5`) specifically prevents collapse under this noise.

---

## 5. Stage 3 — Joint Fine-Tune

### Purpose

Take the SSL-pretrained student predictor (with its object-permanence prior) and **fine-tune the whole supervised pipeline jointly**. This is where the corruption robustness learned in Stage 2 propagates back into the detection/ReID heads.

### What Trains

| Module | Status |
|---|---|
| `Dinov2EncoderWrapper` | **Frozen** |
| `EncoderNeck` | Trainable |
| `SurgicalToolDetector` (DETR) | Trainable |
| `PerTrackModelPredictor` | Trainable (initialised from Stage 2 student) |
| `ClsDec`, `RegDec` | Trainable |
| `ReidHead` | Trainable |

Effectively a re-run of Stage 1 with a **much better predictor initialisation**. The encoder stays frozen to protect features learned during DINOv2 pretraining.

### Losses

Same formulation as Stage 1:
$$L_\text{stage3} = L_\text{det} + \lambda_\text{track} \cdot L_\text{track} + \lambda_\text{reid} \cdot L_\text{reid}$$

But with a **lower learning rate** (`5e-5` vs `1e-4`) because we are refining, not scaffolding. Also **fewer epochs** (10) — joint fine-tune converges fast once the predictor is pre-aligned.

### Config

`@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml`

```yaml
meta:
  stage: stage3_joint
  load_checkpoint: outputs/mot/cholec20-stage2-jepa-pretrain/best.pth.tar
optimization:
  lr: 5.0e-5
  epochs: 10
```

### Why Stage 3 Cannot Be Skipped

The Stage 2 student learned to match the Stage 1 teacher's ω outputs. But the downstream `ClsDec` and `RegDec` decoders were trained against the *old* predictor's ω distribution. After Stage 2, the student's ω distribution has shifted slightly (toward the invariant manifold). Stage 3 re-aligns the decoders to this new distribution while simultaneously exposing them to the corruption-robust predictor.

Skipping Stage 3 usually **reduces HOTA by 5–8 points** compared to running all three stages.

---

## 6. Stage 4 (optional) — Full Stack with OccuSolver + Geometry

### Purpose

Add **explicit** occlusion reasoning and 3D geometric priors on top of the Stage 3 model.

### What Is New

| Module | Status | Purpose |
|---|---|---|
| `CoTrackerWrapper` | Frozen | Point-level tracking for visibility estimation |
| `VisHead`, `PriorEncoder`, `lightTrans` | Trainable (LadderSide fine-tune) | Per-point visibility prediction |
| `VGGT` (Visual Geometry Grounded Transformer) | Frozen | Per-frame depth, camera pose, point maps |
| Null-space projector `P_null` | Trainable | Fuse geometric perturbation into filter weights |

### Losses

Adds to Stage 3 loss:
$$L_\text{stage4} = L_\text{stage3} + \gamma \cdot L_\text{occu} + \delta \cdot L_\text{consist}$$

- `L_occu`: BCE on `VisHead` vs ground-truth visibility
- `L_consist`: cosine similarity between `ω_with_geo` and `ω_without_geo` (prevents geometry from catastrophically overwriting semantics)

### Config

`@/users/kcwp264/TRACK_JEPA/surgi_world_track/configs/train_mot/dinov2/cholec20-mot-stage4-full.yaml`

```yaml
stage_flags:
  use_geometry: true
  use_occusolver: true
occusolver:
  stub: false
  cotracker_version: 3
  cotracker_mode: offline
data:
  batch_size: 1     # VGGT ~300M frozen + CoTracker ~60M frozen — memory-heavy
optimization:
  lr: 1.0e-5        # very low — final polish
```

### When to Run Stage 4

Only if:
1. You have Stage 3 HOTA > 50 (stable base).
2. You have VRAM budget for ~360M extra frozen parameters.
3. You are targeting the most challenging subset (heavy occlusion sequences in CholecTrack20).

Otherwise, stop at Stage 3.

### CoTracker3 / OccuSolver evaluation

Treat CoTracker as a **frozen** submodule inside OccuSolver: keep Stage 4 geometry,
loss terms, batch size, optimizer state, and checkpoint protocol fixed.
The only rollout knobs are:

1. `occusolver.cotracker_version: 2 | 3`
2. `occusolver.num_refine_steps: 0 | 1+`

Use `num_refine_steps=0` for a clean v2/v3 comparison of tracker quality only.
Use `num_refine_steps>0` only after v3 is selected, and only if you want to
exercise the iterative refinement path.

**Rollout protocol**

1. **Baseline (v2):**
   - `occusolver.cotracker_version: 2`
   - `occusolver.num_refine_steps: 0`
2. **Candidate (v3 frozen):**
   - Duplicate the exact baseline config (same checkpoint, LR schedule, and seed),
     change only:
     - `occusolver.cotracker_version: 3`
     - `occusolver.num_refine_steps: 0`
3. Optional **candidate-with-refinement**:
   - keep `occusolver.cotracker_version: 3`
   - set `occusolver.num_refine_steps: 2` (or 3 for a stronger ablation)
   - keep all other fields unchanged.
4. Evaluate both on the same CT20 val protocol:
   - `HOTA`, `AssA`, `IDF1`
   - **AJ**: association Jaccard on a fixed occlusion-heavy subset
   - `δavgvis`: mean change in OccuSolver visibility confidence on that subset
   - step-time or FPS and peak GPU memory

**Candidate example**

Example snippet:

```yaml
occusolver:
  stub: false
  cotracker_version: 3
  num_refine_steps: 0     # 0 | 1+; set >0 to run iterative refinement
  cotracker_mode: offline # offline | online
  cotracker_num_query_points: 16
  num_query_points: 16
```

**Adoption thresholds (guidance)**

| Metric | Suggested bar |
|--------|----------------|
| HOTA | ΔHOTA on CT20 val >= **+0.8** vs baseline with `num_refine_steps=0`; if HOTA is flat, accept only if occlusion gains are large |
| AJ | ΔAJ (occlusion subset) >= **+1.5** and does not degrade non-occlusion IDF1 |
| δavgvis | Occlusion-visibility mean should improve by **+0.015** or more |
| Throughput | Candidate step-time within **1.5×** baseline (or matched FPS) at `batch_size: 1` |
| Memory | VRAM increase should be < **+20%** over baseline if using `num_refine_steps > 0` |

Rollback if **any** hard fail occurs:

- HOTA drops by more than **-0.5**
- AJ drops by more than **-1.0**
- δavgvis drops below **-0.01**
- validation run shows instability in visibility curves or repeated track explosions
- throughput or memory exceeds your deployment budget

If rollback is triggered, stay on `cotracker_version: 2` and tune only `num_refine_steps`,
`num_query_points`, and visibility loss weight in isolated follow-up runs.

---

## 7. CoTracker3 Iterative Refinement — v2→v3 Staged Rollout

This section governs adoption of CoTracker3-style iterative local-correlation refinement inside OccuSolver.

### Rollout Steps

| Step | Config | Purpose |
|---|---|---|
| A. Baseline | `cotracker_version: 2`, `num_refine_steps: 0` | Reproducible reference |
| B. Tracker swap | `cotracker_version: 3`, `num_refine_steps: 0` | Verify v3 frozen tracker alone |
| C. Refinement | `cotracker_version: 3`, `num_refine_steps: 2` | Enable iterative loop |
| D. Scale | Keep C, tune `num_query_points` 8→32 | Max local context |

### Adoption Criteria (Hard)

| Metric | Bar |
|---|---|
| HOTA | Δ ≥ +0.8 vs A |
| AJ (occlusion subset) | Δ ≥ +1.5 |
| IDF1 (non-occlusion) | Δ ≥ −0.5 (no regression) |
| δavgvis | +0.015 improvement on occlusion subset |
| FPS | Within 1.5× baseline at batch_size=1 |
| VRAM | Increase < +20% |

### Regression Guards

1. HOTA drop > −0.5 → **reject immediately**
2. AJ drop > −1.0 → **reject immediately**
3. δavgvis < −0.01 → **reject immediately**
4. FPS outside 1.5× → accept only if AJ and δavgvis both ≥ +2.0
5. VRAM > +20% → accept only if HOTA gain ≥ +1.5

### Minimal Config Example

```yaml
stage_flags:
  use_occusolver: true
occusolver:
  stub: false
  cotracker_version: 3
  num_refine_steps: 2
  cotracker_mode: online
  num_query_points: 16
```

Set `num_refine_steps: 0` to disable the iterative loop and test the tracker swap alone.

---

## 7b. Lean Geometry: Depth-Anything-V2 inside OccuSolver

The original Stage 4 uses VGGT (~300M params) for dense 3D reconstruction via `GeometryBranch`. The **Lean Geometry** path replaces this with Depth-Anything-V2-Small (~25M) embedded directly in `OccuSolver`.

### Architecture

```
OccuSolver =
  CoTracker3 (point tracks)
  + visibility gating
  + sparse depth (Depth-Anything-V2-Small)
```

Key properties:
- **Visibility-gated depth**: Only sample depth at visible points (vis > 0.5). Occluded points get sentinel `depth=0`.
- **Per-frame z-score normalization**: Removes monocular depth scale ambiguity.
- **~12× param reduction**: 300M → 25M frozen params.
- **Higher throughput**: Enables `batch_size=2` (was 1).

### Config

```yaml
stage_flags:
  use_geometry: false       # no VGGT
  use_occusolver: true
occusolver:
  cotracker_version: 3
  num_refine_steps: 2
  use_depth: true
  depth_stub: false
```

### Ablation Matrix

| Config | Geometry | Params | batch_size |
|---|---|---|---|
| `stage4-full` | VGGT (dense) | ~300M | 1 |
| `stage4-lean` | Depth-Anything (sparse) | ~25M | 2 |
| `stage3` | None | 0 | 4 |

---

## 8. Data Requirements

| Stage | Required Annotations | Dataset | Notes |
|---|---|---|---|
| **Stage 1** | bboxes + track IDs + tool classes | CholecTrack20 Training (10 vids) | Scaffold only |
| **Stage 2** | bboxes (for ref frames) + track IDs | **Unified SSL corpus** (75 vids) | CT20 Train (10 real) + C80 (73 pseudo) |
| **Stage 3** | bboxes + track IDs + tool classes | CholecTrack20 Training (10 vids) | Validate on CT20 Val (2 vids) |
| **Stage 4** | Stage 3 + GT visibility labels (optional) | CholecTrack20 | Test on CT20 Test (8 vids) |

**Cholec80 standalone** is only usable for Stage 2 after pseudo-labelling, because it ships with binary tool-presence labels only.

### Leak-Free Split (Critical)

CholecTrack20 and Cholec80 overlap: CT20 VID01–VID80 are identical to C80 video01–video80. We exclude any C80 video that overlaps CT20 **val** or **test** splits from the SSL corpus to prevent data leakage.

Canonical splits are defined in `@/users/kcwp264/TRACK_JEPA/surgi_world_track/core_app/data/splits.py`:

| Split | Videos | C80 equivalent | SSL exclude? |
|---|---|---|---|
| CT20 Train | VID02, VID04, VID11, VID13, VID17, VID23, VID31, VID37, VID96, VID103 | video02..video37 (8 overlap) | ❌ Safe — use for SSL |
| CT20 Val | VID30, VID110 | video30 | ✅ **Exclude** |
| CT20 Test | VID01, VID06, VID07, VID12, VID25, VID39, VID92, VID111 | video01,06,07,12,25,39 | ✅ **Exclude** |

Excluded C80 videos: `video01, video06, video07, video12, video25, video30, video39` (7 videos).  
SSL corpus = 73 C80 videos + 10 CT20 Training videos = **75 videos total**.

The `verify_no_leak()` function asserts SSL corpus ∩ CT20 {val, test} = ∅ at runtime.

Directory layout expected by `MOTCholecDataset`:

```
ssl_corpus/Training/
├── VID02/            # symlink → cholectrack20/Training/VID02
│   ├── Frames/
│   └── vid02.json    # real annotations
├── video02/          # Cholec80, pseudo-annotated
│   ├── Frames/       # symlinks → cholec80/frames/video02/
│   └── video02.json  # pseudo annotations from Stage 1 detector
└── ... (75 videos total)
```

---

## 8. Launch Commands

All commands run from the repo root `surgi_world_track/`.

### Stage 1

```bash
PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
    --devices cuda:0 \
    --debugmode False
```

Output: `outputs/mot/cholec20-stage1-supervised/{latest.pth.tar, best.pth.tar}`

### Stage 2a — Build SSL Corpus (run once)

```bash
PYTHONPATH=. python -m scripts.build_ssl_corpus \
    --stage1_config configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
    --stage1_checkpoint outputs/mot/cholec20-stage1-supervised/best.pth.tar \
    --out_root /scratch/kcwp264/data/surgi_world_track/ssl_corpus \
    --device cuda:0 \
    --score_threshold 0.5
```

Expected runtime: ~30–60 min on one L40S (73 videos × ~200 frames each).  
Output: unified corpus at `/scratch/.../ssl_corpus/Training/` (75 videos, CT20-format JSONs).

### Stage 2b — Run SSL Pretraining

```bash
# Verify Stage 1 checkpoint exists
ls -la outputs/mot/cholec20-stage1-supervised/best.pth.tar

# CholecTrack20 only (original)
PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec20-mot-stage2-jepa-pretrain.yaml \
    --devices cuda:0

# Combined Cholec80 + CT20 (recommended)
PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec80-ct20-stage2-jepa-pretrain.yaml \
    --devices cuda:0
```

Output: `outputs/mot/cholec80-ct20-stage2-jepa-pretrain/{latest.pth.tar, best.pth.tar}`

### Stage 3

```bash
# Verify Stage 2 checkpoint exists
ls -la outputs/mot/cholec20-stage2-jepa-pretrain/best.pth.tar

PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec20-mot-stage3-joint-finetune.yaml \
    --devices cuda:0 \
    --debugmode False
```

### Stage 4 (optional)

```bash
PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec20-mot-stage4-full.yaml \
    --devices cuda:0 \
    --debugmode False
```

### Smoke test

```bash
PYTHONPATH=. python -m core_app.mot.main \
    --fname configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
    --devices cuda:0 \
    --debugmode True    # truncates dataset to 32 train / 16 val clips
```

---

## 9. Frozen vs Trainable — Full Reference Table

| Module | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| DINOv2 encoder | 🔒 | 🔒 | 🔒 | 🔒 |
| EncoderNeck (FPN) | 🟢 | 🔒 | 🟢 | 🟢 |
| DETR head | 🟢 | 🔒 | 🟢 | 🟢 |
| **Teacher predictor (t-Predictor)** | — | 🔒 (from Stage 1) | — | — |
| **Student predictor (s-Predictor)** | 🟢 | 🟢 | 🟢 (from Stage 2) | 🟢 |
| ProjNet | — | 🟢 | — | — |
| Expander | — | 🟢 | — | — |
| ClsDec / RegDec | 🟢 | 🔒 | 🟢 | 🟢 |
| ReID head | 🟢 | 🔒 | 🟢 | 🟢 |
| CoTracker (OccuSolver) | — | — | — | 🔒 |
| VisHead / PriorEncoder | — | — | — | 🟢 |
| VGGT | — | — | — | 🔒 |
| Null-space projector | — | — | — | 🟢 |

🟢 trainable  🔒 frozen  — not instantiated

---

## 10. W&B Experiment Tracking & Visualisations

All stages automatically log to **Weights & Biases** when `wandb.enabled: true` is set in the config (or the `WANDB_PROJECT` env variable is set). The integration is implemented in `core_app/utils/wandb_logger.py` and wired into `MOTTrainer`.

### What Gets Logged

| Category | What | Frequency | Which Stage |
|---|---|---|---|
| **Scalars** | `train/{stage}/loss`, `train/{stage}/det`, `train/{stage}/track`, `train/{stage}/reid`, `train/{stage}/inv`, `train/{stage}/cov` | every batch | All |
| **Learning rate** | `train/lr` | every batch | All |
| **Gradient norms** | `train/grad_norm` | every batch | All |
| **Epoch summary** | `epoch/train_*`, `epoch/val_*`, `epoch/best_val_loss` | every epoch | All |
| **Clean vs Corrupted** | Side-by-side images of the student's corrupted current frame vs the teacher's clean frame | every 100 batches | Stage 2 |
| **DETR predictions** | Validation frames overlaid with predicted bounding boxes (coloured by tool class + confidence score) and ground-truth boxes | once per epoch | Stage 1, 3, 4 |
| **BBox distributions** | Histograms of width, height, aspect ratio of GT boxes | once per epoch | Stage 1, 3, 4 |
| **Collapse alerts** | W&B Alert fired if `jepa_inv < 0.001` and `jepa_cov < 0.001` simultaneously | triggered | Stage 2 |
| **Model watching** | Gradient / parameter histograms for the trainable sub-graph | every 100 batches | Stage 1, 3, 4 |

### Configuring W&B

Add a `wandb:` block to any training YAML:

```yaml
wandb:
  enabled: true
  project: surgical-mot
  entity: your-wandb-username      # optional
  name: ct20-stage1-run-01         # optional — null → auto-generated
  group: cholec20-experiments      # optional — groups runs in the UI
  job_type: train
  tags: ["stage1", "cholec20"]
  watch: gradients                 # or "parameters" / "all" / "none"
  watch_freq: 100                  # log histograms every N batches
```

### How to Monitor Stage 2 (SSL) for Collapse

Open the W&B run and watch these three panels:

1. **`jepa/inv_loss`**: Should decrease steadily from ~1.5–2.0 toward 0.2–0.5 over the first 5 epochs. If it stays flat at ~2.0, your **teacher predictor is untrained** (wrong Stage 1 checkpoint).
2. **`jepa/cov_loss`**: Should stay bounded between 0.1 and 1.0. If it explodes to >> 1.0, the student is collapsing to identical ω vectors — raise `β` (`jepa_cov_weight`) or lower corruption probabilities.
3. **`jepa/inv_cov_ratio`**: Healthy training shows this ratio slowly declining but staying > 0.1. A sudden drop to < 0.001 means the student has found a shortcut to satisfy the covariance penalty without matching the teacher — **collapse alert triggered**.

### How to Read Validation Predictions

In the media panel `val/predictions`, you will see:
- **Green boxes**: Ground-truth (from CholecTrack20 annotations).
- **Red/blue boxes**: DETR predictions, labelled with tool class and confidence.

If the model is over-detecting tissue as tools (false positives from pseudo-labels in Stage 3/4), you will see red boxes with low confidence (`< 0.5`) scattered across homogeneous red/pink regions.

---

## 11. Expected Metrics per Stage

On CholecTrack20 test split with DINOv2 ViT-B/14:

| Stage | HOTA | DetA | AssA | IDF1 | mAP@50 | Comment |
|---|---|---|---|---|---|---|
| Stage 1 | 35–40 | 45 | 35 | 55 | 0.45 | Supervised scaffold — detection-dominated |
| Stage 2 | *no eval metric — monitor `jepa_inv` ↓ and `jepa_cov` ≈ 0* | SSL only |
| Stage 3 | 48–55 | 55 | 48 | 66 | 0.55 | Full permanence gains |
| Stage 4 | 52–58 | 56 | 55 | 70 | 0.56 | Only worth it for occlusion-heavy subsets |

Permanence-specific metric to watch: **AssA** (association accuracy). This is the metric most directly tied to "the tool came back after occlusion and I gave it the same ID". Stage 2 → Stage 3 should produce the largest AssA gain (+10–15 points typical).

---

## 11. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Stage 2 `jepa_inv` stuck at ~2.0 | Teacher predictor is untrained (Stage 1 not run / wrong checkpoint) | Verify `meta.load_checkpoint` path |
| Stage 2 `jepa_cov` explodes to >> 1.0 | Representational collapse — all ω's identical | Lower `α`, raise `β`, ensure corruption probabilities > 0 |
| Stage 3 HOTA < Stage 1 HOTA | Student predictor diverged during Stage 2 | Reduce Stage 2 LR to 1e-4 |
| Stage 2 OOM | Double encoder forward on 48GB card | Reduce `batch_size` to 2 or use DINOv2-S/14 |
| Stage 2 hangs with variable batches (DDP only) | Ranks have different N_tracks | Pad tracks to max across ranks (see DDP wiring notes) |

---

## References

- **GOT-JEPA** (TCSVT 2026): https://arxiv.org/abs/2602.14771 — source of teacher-student predictor SSL
- **GOT-Edit** (ICLR 2026): https://arxiv.org/abs/2602.08550 — null-space geometric fusion (Stage 4)
- **ToMP** (CVPR 2022): https://arxiv.org/abs/2203.11192 — base per-track predictor architecture
- **VICReg** (ICLR 2022): https://arxiv.org/abs/2105.04906 — covariance loss formulation
- **CholecTrack20** (CVPR 2025): https://arxiv.org/abs/2312.07352 — dataset with track IDs + corruption annotations
- **DINOv2** (TMLR 2023): https://arxiv.org/abs/2304.07193 — frozen backbone
- **CoTracker** (ECCV 2024): https://arxiv.org/abs/2307.07635 — point tracker for OccuSolver (Stage 4)
- **Object Permanence in Tracking** (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/papers/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.pdf — theoretical motivation

---

**Last updated**: 2026-04-24 (added Cholec80 leak-free SSL corpus, pseudo-label pipeline, combined Stage 2 config, and W&B experiment tracking with collapse-detection alerts)
