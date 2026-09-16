#!/usr/bin/env bash
# Initialize Git LFS for checkpoint uploads/pulls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "Install git-lfs first (e.g. apt install git-lfs)" >&2
  exit 1
fi

git lfs install
git lfs pull 2>/dev/null || true

echo "LFS tracked patterns:"
git lfs track 2>/dev/null || cat .gitattributes

echo ""
echo "Checkpoint dirs:"
ls -la outputs/ 2>/dev/null || true
echo ""
echo "To add weights: copy .pt into outputs/<run>/ then git add outputs/ && git commit"
