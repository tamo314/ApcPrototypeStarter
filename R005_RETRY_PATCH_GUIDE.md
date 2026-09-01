# A1-R005 Retry Delta Pack

This pack is inserted after the failed A1-R005 run and before A1-R006.

It does not replace Phase A.1 history. A1-R005 remains a failed STOP GATE. The purpose of this pack is to isolate why parameterized primitive causality failed and define a disciplined retry.

## Add these files

- `docs/exec-plans/active/A1_R005_RETRY.md`
- `docs/CODEX_TASKS_A1_R005_RETRY.md`
- `docs/EXPERIMENT_PLAN_A1_R005_RETRY.md`
- `docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md`
- `docs/AGENTS_A1_R005_RETRY_ADDENDUM.md`

## Minimal edits to existing files

### `docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md`

Add after A1-R005:

```md
> A1-R005 failed at the STOP GATE. Do not begin A1-R006.
> Execute the diagnostic/retry sequence in
> `docs/CODEX_TASKS_A1_R005_RETRY.md`.
```

### `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`

Add:

```md
> A1-R006+ remain blocked until the retry gate defined in
> `docs/exec-plans/active/A1_R005_RETRY.md` passes.
```

### `AGENTS.md`

Add the retry documents to the active read-first list while R005 remains unresolved.
