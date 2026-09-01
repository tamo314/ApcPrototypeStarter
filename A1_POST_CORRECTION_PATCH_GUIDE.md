# Phase A.1 Post-Correction Redesign Pack

This pack replaces the planned A1-007 and later task sequence after successful completion of A1-C001 through A1-C007.

It does not rewrite Phase A history, A1-001 through A1-006, A1-C001 through A1-C007, or existing ADRs/results.

## Why the redesign is needed

The correction phase established that one shared Stable Core can solve all eight explicitly specified known operations with very high unseen-content accuracy; `z_task` carries operation and argument information; `PrimitiveCall(operation, arguments)` is available; and oracle routing can bypass the learned router.

However, the shared Stable Core is already powerful enough to solve the task directly. Therefore high accuracy in:

`Stable Core -> Oracle Router -> Primitive -> output`

does not prove that the primitive is causally responsible.

The redesigned sequence first creates a task-blind content state and then requires primitive execution to supply the operation-specific computation.

## Add these files

- `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`
- `docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md`
- `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md`
- `docs/AGENTS_PHASE_A1_POST_CORRECTION_ADDENDUM.md`

## Minimal edits to existing documents

### `docs/CODEX_TASKS_PHASE_A1.md`

Add:

```md
> Post-correction redesign:
> The original A1-007 and later task definitions are superseded by
> `docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md`.
> Historical task IDs are not deleted; use the new A1-Rxxx task IDs for implementation.
```

### `docs/exec-plans/active/PHASE_A1.md`

Add:

```md
> After A1-C007, continue with
> `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`.
```

### `AGENTS.md`

Add the new documents to the active read-first list.

## Recommended Codex prompt

```text
Read AGENTS.md and all Phase A.1 addenda.
Read:
- docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md
- docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md
- docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md
- docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md

Implement only Task A1-R001.
Do not start later tasks.
Preserve all existing Phase A / Phase A.1 / correction results.
If a STOP GATE fails, stop and report it instead of compensating with a larger model.
```
