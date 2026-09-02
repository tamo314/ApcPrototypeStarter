# A1-R005E Representation / Operator Isolation — Patch Guide

This pack closes the failed A1-R005 retry as a negative diagnostic result and inserts a new diagnostic phase before any further Phase A.1 architecture work.

## Purpose

The retry D-001 through D-008 ruled out several simple explanations:
- wrong-argument evaluation was not the main issue,
- SELECT's order-destroying encoder was real but not sufficient,
- counterfactual training alone did not fix COUNT/BIND,
- more steps/rank/arg-dim did not close the causal gap,
- FiLM helped COUNT only partially,
- all four parameterized operations failed under the shared baseline.

The unresolved question is:

> Is frozen task-blind `h_content` sufficient for argument-conditioned computation, with the primitive/operator being the bottleneck, or does the representation itself need redesign?

## Add these files

- `docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md`
- `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`
- `docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md`
- `docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md`
- `docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md`
- `docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md`

## Existing-file edits

### `docs/CODEX_TASKS_A1_R005_RETRY.md`

Add:

```md
> A1-R005D-009 is intentionally skipped/superseded after D-001 through D-008
> produced convergent negative evidence. Do not run the same final four-operation
> retry without a new mechanistic hypothesis.
>
> Close the retry via A1-R005E-001 and continue with
> `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`.
```

### `docs/exec-plans/active/A1_R005_RETRY.md`

Set:

```md
**Status:** closed — negative diagnostic result after D-001 through D-008.
D-009 was not run because preceding diagnostics produced no justified
configuration for a meaningful final retry.
```

### `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`

Add:

```md
> A1-R006 remains blocked.
> Active work is `docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md`.
```

### `AGENTS.md`

While this phase is active, put the new diagnostic queue/plan before the normal A1-R006+ queue.

## Recommended Codex prompt

```text
Read AGENTS.md and all active Phase A.1 post-correction documents.
Then read:
- docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md
- docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md
- docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md
- docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md
- docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md
- docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md

Implement only Task A1-R005E-001.
Do not begin A1-R006.
Preserve all R005/R005D historical artifacts unchanged.
```
