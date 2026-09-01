# Phase A.1 Correction Delta Pack

This pack is intended to be copied into the existing APC repository **after A1-006 and before A1-007**.

It does not replace the existing Phase A.1 plan. It adds a corrective gate and updates the assumptions for A1-007+.

## Why this patch exists

A1-006 passed only after the experiment uncovered several identifiability failures in the synthetic environment:

- some known operations required hidden parameters that were not present in model input,
- symbol permutation made value-dependent tasks unidentifiable,
- a mixed-operation pool was unidentifiable when the requested operation was not presented to the model,
- the final gate therefore trained one Stable Core per operation.

The per-operation gate is a valid result, but it does not yet prove that one shared Stable Core can:

1. read an explicit task specification,
2. represent task identity and task arguments,
3. separate task/control information from content state,
4. support later routing over multiple primitives.

This patch adds that missing gate.

## Add these files

- `docs/exec-plans/active/PHASE_A1_CORRECTION.md`
- `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md`
- `docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md`
- `docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`

## Existing files to edit minimally

### `docs/exec-plans/active/PHASE_A1.md`

Add a note before A1-M4:

```md
> Correction after A1-006:
> Before A1-007 / A1-M4, execute the corrective gates defined in
> `docs/exec-plans/active/PHASE_A1_CORRECTION.md`.
> A1-006 is retained as H1a (per-operation systematic generalization).
> A1-006b is H1b (shared-core conditional systematic generalization).
```

### `docs/CODEX_TASKS_PHASE_A1.md`

Do not renumber existing tasks.

Add:

```md
Before Task A1-007, complete every mandatory task in
`docs/CODEX_TASKS_PHASE_A1_CORRECTION.md`.
```

### `docs/DECISIONS.md`

Add ADRs only after experiments are executed. Do not pre-write measured conclusions.

## Codex starting prompt

```text
Read AGENTS.md,
docs/AGENTS_PHASE_A1_ADDENDUM.md,
docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md,
docs/exec-plans/active/PHASE_A1.md,
docs/exec-plans/active/PHASE_A1_CORRECTION.md,
docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md,
docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md,
docs/EXPERIMENT_PLAN_PHASE_A1.md,
docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md,
docs/CODEX_TASKS_PHASE_A1.md,
and docs/CODEX_TASKS_PHASE_A1_CORRECTION.md.

Implement only Task A1-C001.
Do not start later tasks.
Preserve all Phase A and A1-006 historical artifacts and conclusions.
Run all required tests and experiments and report every acceptance criterion.
```
