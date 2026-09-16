#!/usr/bin/env bash
# Fast pseudo-label build for DGX Spark (GB10 + Grace CPUs).
# Defaults: infer_fast (DETR+tracker only), bf16 AMP, window_stride=2.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export XFORMERS_DISABLED=1

python -m scripts.build_ssl_corpus \
  --stage1_checkpoint outputs/mot/cholec20-stage1-supervised/best.pth.tar \
  --stage1_config configs/train_mot/dinov2/cholec20-mot-stage1-supervised.yaml \
  --out_root data/ssl_corpus \
  --infer_score_threshold 0.25 \
  --score_threshold 0.25 \
  --window_stride 2 \
  --prefetch_workers 8 \
  --device cuda:0 \
  "$@"
