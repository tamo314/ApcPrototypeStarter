# Phase A.1 Delta Pack

This pack is intended to be copied into the existing APC repository after Phase A. It adds only Phase A.1 documentation and does not replace the historical Phase A plan or result.

## Add these files

- `docs/exec-plans/active/PHASE_A1.md`
- `docs/CODEX_TASKS_PHASE_A1.md`
- `docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1.md`
- `docs/AGENTS_PHASE_A1_ADDENDUM.md`

## Recommended small edits to existing files

### `AGENTS.md`

Append a short pointer:

```md
## Active research phase

Phase A is closed with a negative scientific verdict but a mechanically working closed loop.
The active plan is `docs/exec-plans/active/PHASE_A1.md`.

Before implementing any Phase A.1 task, also read:
- `docs/AGENTS_PHASE_A1_ADDENDUM.md`
- `docs/CODEX_TASKS_PHASE_A1.md`
- `docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1.md`
```

### `README.md`

Add a current-status pointer rather than rewriting Phase A history.

### `docs/DECISIONS.md`

Do not pre-fill decisions from this pack. Append ADRs only when a Phase A.1 task produces measured evidence.

## Recommended Codex prompt

```text
Read AGENTS.md, docs/AGENTS_PHASE_A1_ADDENDUM.md,
docs/exec-plans/active/PHASE_A1.md,
docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md,
docs/EXPERIMENT_PLAN_PHASE_A1.md, and docs/CODEX_TASKS_PHASE_A1.md.

Implement only Task A1-001.
Do not start later tasks.
Preserve all Phase A historical artifacts.
Run every verification command required by the task.
Report changed files, tests, experiment outputs, assumptions, and any failed acceptance criterion.
```
