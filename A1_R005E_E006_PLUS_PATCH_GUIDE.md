# A1-R005E E006+ Revision — Patch Guide

This pack revises **A1-R005E-006 and later diagnostic tasks** after A1-R005E-005.

It preserves all A1-R005 / Retry / E-001 through E-005 history and run artifacts.

## Why this revision exists

E-004 showed that a large cross-position operator can use frozen `h_content` effectively.
E-005 showed that one uniform primitive-scale single-cross-attention operator cannot reliably recover that upper bound.

The unresolved question is now:

> Is frozen `h_content` merely difficult for a compact operator to use, or is the compact operator architecture itself inadequate?

Use the following 2x2 design:

| Representation | Compact operator | High-capacity operator |
|---|---|---|
| Frozen | E-005 — measured | E-004 — measured |
| Joint task-blind | E-006A — new | E-006B — conditional |

## Add

- `docs/exec-plans/active/A1_R005E_E006_PLUS.md`
- `docs/CODEX_TASKS_A1_R005E_E006_PLUS.md`
- `docs/EXPERIMENT_PLAN_A1_R005E_E006_PLUS.md`
- `docs/design-docs/REPRESENTATION_OPERATOR_FACTORIAL.md`
- `docs/AGENTS_A1_R005E_E006_PLUS_ADDENDUM.md`

## Existing-file edits

In `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` add:

```md
> After A1-R005E-005, use `docs/CODEX_TASKS_A1_R005E_E006_PLUS.md`.
> The older E-006/E-007/E-008 definitions are superseded but remain historical.
```

In `docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md` add:

```md
> E-006 onward was revised after ADR-0042.
> Continue with `docs/exec-plans/active/A1_R005E_E006_PLUS.md`.
```

Update root `AGENTS.md` read-first pointers while this diagnostic is active.

## Recommended Codex prompt

```text
Read AGENTS.md and all active Phase A.1 documents.
Then read:
- docs/exec-plans/active/A1_R005E_E006_PLUS.md
- docs/CODEX_TASKS_A1_R005E_E006_PLUS.md
- docs/EXPERIMENT_PLAN_A1_R005E_E006_PLUS.md
- docs/design-docs/REPRESENTATION_OPERATOR_FACTORIAL.md
- docs/AGENTS_A1_R005E_E006_PLUS_ADDENDUM.md

Implement only Task A1-R005E-006A.
Do not begin A1-R006.
Preserve E-004 and E-005 as fixed comparison cells.
Do not make the compact operator larger in E-006A.
```
