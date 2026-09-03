# A1-R005E Shared Queryable Representation Gate — Patch Guide

Insert this pack **after A1-R005E-006A / ADR-0043** and before any production Branch-B redesign.

## Why

A1-R005E-006A strongly supports representation accessibility, but each operation trained its own task-blind encoder. Operation semantics may therefore be encoded implicitly in encoder weights.

The next gate is:

> Can one **shared task-blind content encoder** simultaneously support SHIFT, SELECT, COUNT, and BIND through compact operation-specific operators?

## Add

- `docs/exec-plans/active/A1_R005E_SHARED_ENCODER_GATE.md`
- `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`
- `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`
- `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md`
- `docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`

## Minimal edits

In `docs/CODEX_TASKS_A1_R005E_E006_PLUS.md` add:

```md
> Before E-006B or a Branch-B commitment, execute
> `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`.
```

In `docs/exec-plans/active/A1_R005E_E006_PLUS.md` add:

```md
> ADR-0043 strongly supports representation accessibility, but
> operation-specific encoder weights remain a confound.
> Continue with `docs/exec-plans/active/A1_R005E_SHARED_ENCODER_GATE.md`.
```

Temporarily add the new shared-gate documents to root `AGENTS.md` read-first pointers.

## Recommended Codex prompt

```text
Read AGENTS.md and all active Phase A.1 diagnostic documents.
Then read:
- docs/exec-plans/active/A1_R005E_SHARED_ENCODER_GATE.md
- docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md
- docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md
- docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md
- docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md

Implement only Task A1-R005E-S001.
Do not begin A1-R006.
Do not give each operation its own encoder.
Do not increase the compact operator architecture relative to E-006A.
Preserve all E-004/E-005/E-006A artifacts unchanged.
```
