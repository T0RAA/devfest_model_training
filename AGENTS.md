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

<!-- agent-baton:start -->
## Usage limits, model routing and token efficiency (agent-baton)

Claude Code and Codex share this repo and work off each other's usage limits. On Tora's Mac,
`~/.agent-baton/bin/baton` tracks both tools' 5-hour and weekly limits from their status lines.

- **Check before you start.** Codex: run `~/.agent-baton/bin/baton brief --for codex` at the start of
  each task. Claude gets the same lines from a hook. If `baton` isn't installed (a cloud
  session, for example), skip the check and say so once.
- **go:** work normally. When both tools are on go, split the work: Claude takes planning, review and
  hard problems; Codex takes the `C-` queue in `CODEX_HANDOFF.md`.
- **wind-down** (5-hour window ≥ 70%, or weekly use ahead of an even pace): finish the current
  task, then take only small tasks. Leave larger ones for the other tool in the PR description, as a
  `C-`-style task with a done-when check.
- **out** (5-hour ≥ 80% or weekly ≥ 95%): start nothing new. Push the branch, write a resume note in the
  draft PR (what's done, the next step, how to verify it), and stop.
- **Taking over:** pick up another tool's work only when its resume note hands it over, and never
  push to its branch.

**Subagents.** Use the cheapest one that can do the job, and escalate only after a real failure.
On wind-down, drop one tier.

| Work | Agent | Claude | Codex |
|---|---|---|---|
| Find code, files, call sites, log lines | `scout` | haiku / low | gpt-6-luna / low |
| Well-specified edits, tests, small fixes, docs | `builder` | sonnet / medium | gpt-6.2-sol / medium |
| Design, unknown-cause bugs, security, final review | `deep-reviewer` | opus / high | gpt-6.2-astra / high |

Run independent subagents in parallel, with each one owning distinct files.

**Token efficiency.** Search before reading, and read line ranges, not whole files. Don't re-read
what's already in context. Hand searches across many files to `scout` and keep only its answer.
Keep command output short (`| tail`, `-q`). Run only the affected tests until the final gate. Keep
replies to results and decisions.
<!-- agent-baton:end -->
