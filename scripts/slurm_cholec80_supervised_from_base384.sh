#!/bin/bash
#SBATCH --job-name=vjepa2_ch80_sup_from_base384
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=160G

set -euo pipefail

module load miniforge
conda activate dino_wm

REPO_DIR="/users/$USER/Gyanteet/dino_wm/vjepa2_repo"
CFG="$REPO_DIR/configs/train_2_1/vitb16/cholec80-supervised-scaffold-256px-16f-from-base384.yaml"
BASE_DATA="/mnt/scratch/$USER/cholec80/cholec80/frames"
STAGED_DATA="${TMPDIR:-/tmp/$USER}/cholec80/frames"

cd "$REPO_DIR"

echo "[INFO] Staging Cholec80 frames to node-local NVMe..."
bash scripts/stage_to_nvme.sh \
  --parallel xargs \
  --jobs 8 \
  --source "$BASE_DATA" \
  --dest "$STAGED_DATA" \
  --no-progress \
  --no-verify

echo "[INFO] Launching supervised scaffold from pretrained checkpoint..."
python - <<'PY'
from pathlib import Path
import yaml

repo = Path("/users") / Path(__import__("os").environ["USER"]) / "Gyanteet" / "dino_wm" / "vjepa2_repo"
cfg_path = repo / "configs/train_2_1/vitb16/cholec80-supervised-scaffold-256px-16f-from-base384.yaml"
run_cfg_path = repo / "configs/train_2_1/vitb16/.tmp_run_supervised_from_base384.yaml"
staged = Path(__import__("os").environ.get("TMPDIR", f"/tmp/{__import__('os').environ['USER']}")) / "cholec80" / "frames"

with cfg_path.open("r") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["datasets"] = [str(staged)]

with run_cfg_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(run_cfg_path)
PY

TMP_CFG="$REPO_DIR/configs/train_2_1/vitb16/.tmp_run_supervised_from_base384.yaml"
PYTHONPATH=. python -m core_app.main --fname "$TMP_CFG" --devices cuda:0 cuda:1 cuda:2

echo "[INFO] Training complete."
