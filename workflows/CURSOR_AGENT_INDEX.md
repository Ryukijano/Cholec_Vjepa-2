# Cursor agent index

Skills, commands, and workflows for Gyanateet_tracking. Cross-project skills from [Ryukijano/agent-skills](https://github.com/Ryukijano/agent-skills).

## Project skills (`.cursor/skills/`)

| Skill | Trigger |
|-------|---------|
| `mot-browser-research` | @Browser, SOTA, strategic direction |
| `mot-training-workflow` | Train/resume stages 1–4 |
| `mot-repo-orientation` | Repo map and pipeline |
| `aire-slurm-submit` | AIRE Slurm submit/monitor |
| `conda-env-setup` | Conda + CUDA on AIRE |
| `debug-pytorch-gpu` | GPU OOM, DDP, NCCL |
| `git-branch-workflow` | Branches, commits, PRs |
| `lora-finetune` | DINOv2/ViT LoRA |
| `surgical-mot-eval` | CholecTrack20 HOTA/mAP |
| `tdv-pretrain` | TDV surgical video SSL |
| `wandb-experiment` | W&B on HPC |

## Slash commands (`.cursor/commands/`)

**MOT:** `/mot-browser-research`, `/mot-train-eval`, `/mot-hota-eval`

**Cross-project:** `/submit-gpu-job`, `/pretrain-and-evaluate`, `/debug-training`, `/code-review`, `/address-pr-comments`, `/checkpoint-to-deployment`, `/setup-ml-project`

## Devin playbooks (`workflows/devin/`)

- `mot-browser-research.devin.md` → `!mot-browser-research`
- `mot-train-eval.devin.md` → `!mot-train-eval`
- `mot-hota-eval.devin.md` → `!mot-hota-eval`

## Personal skills (`~/.cursor/skills/`)

`iterative-test-loop`, `impact-aware-testing`, `explore-sota`, `digest-paper`, `ship-pr`, `babysit-pr`, `systematic-debug`, and others — see canvas `mot-skills-workflows.canvas.tsx`.

## Quick paths

- Train: `bash scripts/train_stage1_ddp_3gpu.sh` (Spark) or `sbatch jobs/*.slurm` (AIRE)
- Eval: `python scripts/eval_checkpoint.py --mot-eval --stratify-smoke`
- Tests: `pytest tests/test_mot_smoke.py -q`
