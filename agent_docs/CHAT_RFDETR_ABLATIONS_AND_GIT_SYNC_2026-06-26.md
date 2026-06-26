# Chat Transcript: RF-DETR Ablations & Git Sync — 2026-06-26

> **Session date:** 2026-06-26
> **Session type:** Windsurf/Cascade chat
> **Topics:** RF-DETR ablation study results, git repo sync (Cholec_Vjepa-2 ↔ Gyanateet_tracking), experiment documentation, ablation plots
> **Status:** Complete

---

## Summary

This session covered:

1. **RF-DETR Ablation Study** — 7/8 variants completed on CholecTrack20 (baseline, no-dn, 50q, 2layer, no-pretrain, no-ema, highlr). Baseline EMA mAP@50:95 = 0.5410 dominated all ablations. Key finding: pretrained weights are the single most important factor (-74% without).

2. **Git Sync: Cholec_Vjepa-2 ↔ Gyanateet_tracking** — Merged Gyanateet_tracking content into Cholec_Vjepa-2 (branch `merge-gyanateet-tracking`), resolved conflicts, removed large checkpoint files from git history with `git filter-branch`, pushed to both GitHub repos.

3. **Experiment Documentation** — Created comprehensive `EXPERIMENTS.md` covering all experiments: MAE pretraining, DINOv2 fine-tuning, Stage 1 MOT, RF-DETR ablations, TDV pretraining, Stage 2 prep, bug fixes, and pending work.

4. **Ablation Plots** — Generated 9 plots (line charts, bar charts, waterfall, scatter, summary table) from CSV data using `scripts/got_jepa/plot_rfdetr_ablations.py`.

---

## PASTE CHAT TRANSCRIPT BELOW THIS LINE

<!-- Delete this comment and paste the exported chat transcript here -->
