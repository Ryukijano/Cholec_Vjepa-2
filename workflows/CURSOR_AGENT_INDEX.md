# Cursor agent skills index

Reference for agents working on Gyanateet_tracking. **Project skills** are in-repo under `.cursor/skills/`. **Personal** and **plugin** skills are installed on the machine (`~/.cursor/`).

## Project skills (in this repo)

| Skill | Path | Use when |
|-------|------|----------|
| `mot-browser-research` | `.cursor/skills/mot-browser-research/` | `@Browser`, online SOTA, strategic direction |
| `mot-training-workflow` | `.cursor/skills/mot-training-workflow/` | Train stages 1–4, resume, eval |
| `mot-repo-orientation` | `.cursor/skills/mot-repo-orientation/` | Understand repo layout and pipeline |

## Slash commands (in this repo)

| Command | File |
|---------|------|
| `/mot-browser-research` | `.cursor/commands/mot-browser-research.md` |
| `/mot-train-eval` | `.cursor/commands/mot-train-eval.md` |
| `/mot-hota-eval` | `.cursor/commands/mot-hota-eval.md` |

## Devin playbooks (in this repo)

See [README.md](README.md) — `workflows/devin/*.devin.md` with macros `!mot-browser-research`, `!mot-train-eval`, `!mot-hota-eval`.

## Personal skills (`~/.cursor/skills/` → `~/.agents/skills/`)

| Skill | Use when |
|-------|----------|
| `iterative-test-loop` | Change → test → diagnose until green |
| `impact-aware-testing` | Find tests for changed files |
| `experiment-protocol` | Pre-register ablations / benchmarks |
| `explore-sota` | arXiv / Semantic Scholar triage |
| `digest-paper` | PDF → synthesis + BibTeX |
| `hypothesis-canvas` | Research hypothesis refinement |
| `systematic-debug` | Evidence-driven debugging |
| `tdd-red-green` | Strict red-green TDD |
| `safe-refactor` | Small steps + characterization tests |
| `ship-pr` / `babysit-pr` / `split-to-prs` | PR lifecycle |
| `review-bugbot` / `review-security` | Pre-merge review |
| `ci-watcher` | Watch PR checks |
| `claim-verification` / `adversarial-review` | Draft and paper review |
| `repro-bundle` / `prisma-systematic-review` | Repro bundles, systematic reviews |
| `loop` | Recurring `/loop` execution |

## Plugin skills (installed via Cursor marketplace)

| Plugin | Key skills for this repo |
|--------|--------------------------|
| **parallel** | `parallel-web-search` (default @Browser research), `parallel-deep-research`, `parallel-web-extract` |
| **cursor-team-kit** | `verify-this`, `fix-ci`, `workflow-from-chats`, `run-smoke-tests`, `review-and-ship` |
| **huggingface-skills** | `huggingface-vision-trainer`, `hf-cli`, `huggingface-datasets`, `huggingface-papers` |
| **cursor built-ins** | `canvas`, `create-skill`, `create-rule`, `review-bugbot` |

## Parent AIMSgeneral commands (Cosmos / ESD)

In `/home/aimsgroupuol/AIMSgeneral/.cursor/commands/`: `/cosmos-verify`, `/cosmos-spark-kernels`, `/esd-t2v`, `/esd-forward-dynamics`, `/lap-t2v`. Playbooks: `AIMSgeneral/workflows/devin/`.

## @Browser quick start

```
@Browser @AGENTS.md /mot-browser-research

Search online for [topic]. Map to our four-stage MOT pipeline. Cite sources.
```

Guardrails: stay GOT-JEPA + OccuSolver; VLA-JEPA is robot policies (not MOT); desmoking is optional ablation only.
