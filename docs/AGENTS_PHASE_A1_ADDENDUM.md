# AGENTS Phase A.1 Addendum

This file supplements the repository root `AGENTS.md`. The root file remains authoritative for general coding rules.

## Purpose

Phase A mechanically demonstrated the closed loop, but did not establish systematic generalization, K/C/N separation, recurrence reuse, strong consolidation, or true top-k compute sparsity. Phase A.1 is therefore a **hypothesis-isolation phase**, not an architecture-expansion phase.

## Oracle-before-learned rule

Never introduce a learned component until the same experiment succeeds with the corresponding oracle.

Progression:

1. oracle task/K-C-N metadata,
2. oracle primitive routing,
3. oracle novelty,
4. learned primitive execution and consolidation,
5. retrieval routing,
6. learned routing,
7. learned novelty,
8. full closed loop.

If an oracle version fails, stop. Do not compensate by adding model scale, RL, or unrelated architecture.

## One mechanism per task

A Phase A.1 task should isolate one mechanism. Avoid simultaneously changing dataset generation, Stable Core architecture, routing, novelty, plastic capacity, and consolidation. If multiple changes are unavoidable, add an oracle control or ablation.

## Gate discipline

Tasks marked **STOP GATE** must pass before dependent tasks begin. On failure:

1. save the run,
2. add an ADR,
3. stop dependent work,
4. investigate only that mechanism.

## Historical integrity

Do not rewrite old Phase A runs or conclusions. Metric/accounting fixes must preserve historical reports and clearly distinguish corrected metrics.

## True sparse compute

For router paths, top-k selection must restrict actual primitive execution. Computing every primitive and multiplying unselected outputs by zero does not count as sparse execution.

Track separately:

- resident total parameters,
- resident primitive parameters,
- active total parameters,
- active primitive parameters,
- temporary peak parameters,
- primitive forward-call count or estimated primitive FLOPs.

## Stable-Core generalization

Fixed-dataset memorization cannot satisfy the Stable Core gate. Scientific runs must use online procedural generation, unseen content, and token/symbol permutation so stable token IDs cannot encode the solution.

## Primitive semantics

Human-readable labels such as `COMPARE`, `COUNT`, or `SORT` are evaluation metadata. Learned components may receive them only in explicitly declared oracle experiments.

## Consolidation semantics

Consolidation is a function-approximation problem. Measure functional agreement on held-out probe states, task-score retention, minimum candidate rank/capacity, and compression ratio.

## Resource target

Remain within the existing single-workstation target: RTX 5060 Ti 16 GB and 64 GB system RAM. Use small dev-tier runs before milestone runs.

## Completion report

Every Codex task must report:

1. changed files,
2. tests run,
3. experiment commands,
4. acceptance criteria pass/fail,
5. run artifact paths,
6. deviations/assumptions,
7. required ADRs.

Do not silently proceed past a failed acceptance criterion.
