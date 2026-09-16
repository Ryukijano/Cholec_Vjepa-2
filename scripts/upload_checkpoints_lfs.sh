#!/usr/bin/env bash
# Upload local checkpoints to GitHub after enabling Git LFS on the repo.
# Enable at: https://github.com/Ryukijano/Cholec_Vjepa-2/settings → Git LFS → Allow
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git lfs install
git add weights/dinov2/*.pth outputs/mot/*/*.pth.tar
git status --short | grep -E 'pth|tar' || { echo "No checkpoint files to add"; exit 1; }
git commit -m "Add DINOv2 and MOT checkpoints via Git LFS"
git push origin "$(git branch --show-current)"
git lfs push origin "$(git branch --show-current)" --all
echo "LFS upload complete."
