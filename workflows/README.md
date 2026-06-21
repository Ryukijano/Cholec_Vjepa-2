# Gyanateet_tracking — Cursor & Devin workflows

GOT-JEPA surgical MOT on CholecTrack20. Agent skills sourced from [Ryukijano/agent-skills](https://github.com/Ryukijano/agent-skills) plus MOT-specific project skills.

## MOT workflows

| Workflow | Devin playbook | Cursor command | Skill |
|----------|----------------|----------------|-------|
| Browser research + SOTA | `devin/mot-browser-research.devin.md` | `/mot-browser-research` | `mot-browser-research` |
| Stage train / resume | `devin/mot-train-eval.devin.md` | `/mot-train-eval` | `mot-training-workflow` |
| HOTA + smoke eval | `devin/mot-hota-eval.devin.md` | `/mot-hota-eval` | `surgical-mot-eval` |
| Repo orientation | — | — | `mot-repo-orientation` |

## Cross-project skills ([agent-skills](https://github.com/Ryukijano/agent-skills))

| Skill | Description |
|-------|-------------|
| `aire-slurm-submit` | Submit/monitor Slurm jobs on Leeds AIRE (L40S) |
| `conda-env-setup` | Conda envs with CUDA on AIRE |
| `debug-pytorch-gpu` | OOM, DDP, NCCL, NaN diagnostics |
| `git-branch-workflow` | Branch naming, commits, PR templates |
| `lora-finetune` | LoRA for DINOv2 / ViT |
| `surgical-mot-eval` | CholecTrack20 MOT metrics and failure modes |
| `tdv-pretrain` | Temporal Difference in Vision pretraining |
| `wandb-experiment` | W&B tracking on HPC |

## Cross-project slash commands

| Command | Description |
|---------|-------------|
| `/submit-gpu-job` | Submit GPU job to AIRE Slurm |
| `/pretrain-and-evaluate` | TDV pretrain → detection → eval |
| `/debug-training` | Debug loss=NaN, OOM, DDP hangs |
| `/code-review` | ML code review checklist |
| `/address-pr-comments` | Systematic PR comment triage |
| `/checkpoint-to-deployment` | Strip checkpoint for inference |
| `/setup-ml-project` | Scaffold new ML project |

## Layout

```
.cursor/skills/          # Cursor agent skills (auto-discovered)
.cursor/commands/        # Cursor slash commands (type / in chat)
.windsurf/skills/        # Windsurf/Cascade mirror (same SKILL.md files)
.windsurf/workflows/     # Full workflow playbooks (detailed steps)
workflows/devin/         # Devin playbooks for MOT
```

**Canonical docs:** `AGENTS.md`, `workflows/CURSOR_AGENT_INDEX.md`, `agent_docs/cursor_explore_mot_training_pipeline.md`

**Cursor:** Type `/` in chat. **Devin:** Upload `.devin.md` playbooks. **@Browser:** `/mot-browser-research` + `AGENTS.md`.
