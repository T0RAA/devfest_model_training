# AGENTS.md: UCSC Wildlife Vision v2

A two-stage detect-then-classify system for 20 wild mammal and bird species on
the UC Santa Cruz Campus Natural Reserve. `README.md` describes the stages;
`SPEC.md` and `MILESTONES_v2_detector_classifier.md` are the plan.

## Gate

```bash
pip install -e ".[dev]" || pip install -e .
pytest
```

## Rules

- Honesty first: synthetic checkpoints and metrics stay labelled synthetic.
  Report only metrics you actually measured, with the run that produced them.
- Image licences: only CC0/CC-BY iNaturalist data. No AGPL detector variants
  (the milestone-7 blocker).
- Keep detector and crop version parity. `/v2/predict` refuses mismatched
  versions on purpose.
- Don't commit large data, crops, shards or checkpoints.

## Working alongside Claude (parallel-work protocol)

Claude Code sessions and Codex work in this repo at the same time. Tora reviews
and merges. These rules keep the two from colliding.

1. **Read `CODEX_HANDOFF.md` first.** It records the current state, what Claude
   is working on right now (don't touch it), and a ranked queue of tasks set
   aside for Codex.
2. **Claim before you build.** List open PRs and remote branches before you
   start. If a branch or PR already covers your task or the same files, pick the
   next task. Once you've chosen, push a branch and open a **draft PR** within
   your first few commits, so other agents can see the claim.
3. **Branch names:** Codex uses `codex/<short-topic>`. Claude uses `claude/*`,
   `T0RAA/*` and `phase-*`. Never commit to, rebase or force-push someone
   else's branch.
4. **One task per PR**, kept small (about 300 changed lines or fewer, not
   counting tests and fixtures). Put the task ID from `CODEX_HANDOFF.md` in the PR
   title, e.g. `[codex] C-2: add rights-gate tests`.
5. **Run the gate** listed in this file before you mark a PR ready. Paste the
   command and its result into the PR description. If you couldn't run
   something (no GPU, no credentials, no macOS), say so plainly. Don't claim it
   passed.
6. **Don't merge.** Leave PRs for Tora to review, unless this file says
   otherwise.
7. **Don't edit `CODEX_HANDOFF.md` in a task PR.** Parallel PRs would conflict
   on it. Put progress notes, open questions and follow-ups in the PR
   description; Tora or Claude folds them back into the handoff.
8. **Ask instead of guessing** on product, legal and data-rights questions.
   Write the question in the PR and stop that thread of work.
9. **Never commit secrets** (`.env`, tokens, OAuth client files, personal
   exports). Never send email, post, submit forms or contact anyone on Tora's
   behalf.
