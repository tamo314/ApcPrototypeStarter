# AGENTS Addendum — A1-R005E Representation / Operator Isolation

## Current status

A1-R005 and retry D-001 through D-008 produced a negative result for the current parameterized primitive design.

A1-R006 remains blocked. D-009 is not to be run merely to obtain another expected failure.

## Active question

Distinguish:

### Representation-sufficient
`h_content` retains enough task-independent information; operator class is the bottleneck.

### Representation-insufficient
Frozen task-blind `h_content` discards or entangles information so even a strong operator cannot reliably solve the task.

## Diagnostic-only rule

Modules introduced here may intentionally be too large/expensive to be APC primitives:
- high-capacity cross-attention readout,
- oracle latent operator,
- diagnostic reconstruction head.

They are upper-bound probes, not candidate production architecture.

## No premature operator rollout

Do not redesign the Primitive Bank around attention until:
1. frozen `h_content` is shown to retain usable information, and
2. a strong operator on frozen `h_content` succeeds.

## No premature representation retraining

Do not retrain/task-condition the Stable Core before testing frozen-state sufficiency.

## Task blindness remains invariant

Any representation-learning control later in this phase must preserve:

`h_content = f(content)`

Arguments reach the operator only.

## Oracle operator rule

Oracle operators may use ground-truth argument semantics and exact content positions for diagnosis. Clearly label them oracle; do not report them as learned primitive performance.

## Branching discipline

Do not choose the next development phase until `NEXT_PHASE_DECISION_MATRIX.md` is filled with measured results.

## STOP discipline

A1-R006 stays blocked throughout this diagnostic phase.
