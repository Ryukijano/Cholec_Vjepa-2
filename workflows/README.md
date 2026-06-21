# Gyanateet_tracking — Cursor & Devin workflows

GOT-JEPA surgical MOT on CholecTrack20. Reusable agent workflows for research, training, and eval.

| Workflow | Devin playbook | Cursor command | Skill |
|----------|----------------|----------------|-------|
| Browser research + SOTA | `devin/mot-browser-research.devin.md` (`!mot-browser-research`) | `/mot-browser-research` | `mot-browser-research` |
| Stage train / resume | `devin/mot-train-eval.devin.md` (`!mot-train-eval`) | `/mot-train-eval` | `mot-training-workflow` |
| HOTA + smoke eval | `devin/mot-hota-eval.devin.md` (`!mot-hota-eval`) | `/mot-hota-eval` | `mot-training-workflow` |
| Repo orientation | — | — | `mot-repo-orientation` |

**Canonical docs:** `AGENTS.md`, `README.md`, `agent_docs/cursor_explore_mot_training_pipeline.md`

**Cursor:** Type `/` in chat to run `.cursor/commands/`. Project skills live in `.cursor/skills/`.

**Devin:** Upload `.devin.md` playbooks or create playbooks from them in [Devin settings](https://docs.devin.ai/product-guides/creating-playbooks).

**@Browser:** Attach `@Browser` + `AGENTS.md` and run `/mot-browser-research` for online SOTA comparison mapped to the four-stage pipeline.

**Full skills index:** [CURSOR_AGENT_INDEX.md](CURSOR_AGENT_INDEX.md) — project, personal, and plugin skills available on Spark.
