# Phase A.2 — Autonomous Controller & Scaling — Patch Guide

## Purpose

Phase A.1 Branch B Integration established shared task-blind representation, heterogeneous compact primitives, strict sparse execution, composition execution/search, temporary plastic adaptation, safe consolidation, persistent recurrence, and learned primitive routing over the known/consolidated 10-operation universe.

Phase A.2 moves to the unresolved system-level question:

> Can APC autonomously choose reuse, composition, and plastic expansion as the bank grows, while preserving routing stability and real sparse-compute benefits?

## Important interpretation of ADR-0061

Preserve ADR-0061 historically, but add a scope clarification rather than rewriting old artifacts:

1. `99.99% routing` was demonstrated under **explicit model-visible TaskSpec**.
2. `90.11% compute savings` is specifically **primitive-bank active-parameter savings**, not yet total end-to-end FLOPs/latency savings.
3. B008 closed the learned sparse inference loop over the **already-known/consolidated 10-operation universe**.
4. It did not yet demonstrate online K/C/N/R autonomous continual learning with learned novelty decisions.

## Add

- `docs/exec-plans/active/PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/EXPERIMENT_PLAN_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/AGENTS_PHASE_A2_AUTONOMOUS_CONTROLLER_ADDENDUM.md`
- `docs/design-docs/AUTONOMOUS_CONTROLLER_POLICY.md`
- `docs/design-docs/INCREMENTAL_ROUTER_AND_BANK_SCALING.md`
- `docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md`

## Root `AGENTS.md`

Make Phase A.2 the active post-Branch-B plan.

Recommended read order:

1. `docs/exec-plans/active/PHASE_A2_AUTONOMOUS_CONTROLLER.md`
2. `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
3. `docs/EXPERIMENT_PLAN_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
4. `docs/design-docs/AUTONOMOUS_CONTROLLER_POLICY.md`
5. `docs/design-docs/INCREMENTAL_ROUTER_AND_BANK_SCALING.md`
6. `docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md`
7. `docs/AGENTS_PHASE_A2_AUTONOMOUS_CONTROLLER_ADDENDUM.md`
8. existing Branch B / diagnostic decisions and architecture docs

## Recommended first Codex prompt

```text
Read AGENTS.md and all active Phase A.2 documents.
Implement only Task A2-C001.

Do not begin A2-C002 automatically.
Preserve all Phase A.1 / Branch B artifacts unchanged.
Treat ADR-0061 as historical evidence scoped to known/consolidated-task routing.
Do not claim semantic task inference; TaskSpec remains explicit in Phase A.2.
```
