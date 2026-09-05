# AGENTS Addendum — Phase A.2 Autonomous Controller & Scaling

## Active scientific question

Phase A.2 is not about improving primitive accuracy.

It tests controller correctness under bank growth, incremental router updates, direct reuse, composition reuse, genuine novelty, recurrence, and sparse execution scaling.

## Architectural invariant

```text
TaskSpec -> Task-side encoding -> z_task -> learned router/controller
Content  -> Shared task-blind content encoder -> h_content
```

The content representation remains:

`h_content = f(content)`

Task/operation/argument information must not enter the content encoder.

## Explicit TaskSpec limitation

Phase A.2 continues to use model-visible explicit TaskSpec.

Therefore routing accuracy is routing under explicit task specification; it is not semantic task inference from language/demonstrations. Task inference is intentionally deferred to a later phase.

## Controller action space

The controller must ultimately choose among:

1. `DIRECT_REUSE`
2. `COMPOSE`
3. `PLASTIC_SEARCH`

Recurrence `R` should normally map to `DIRECT_REUSE` of the previously consolidated primitive.

## Evidence before expansion

Plastic allocation is forbidden until:

1. learned/direct primitive proposal is evaluated,
2. permitted composition search is evaluated,
3. both fail the adequacy threshold.

Do not use novelty as "unknown operation token exists". Novelty must be based on computational inadequacy evidence.

## Compact-first plastic policy

When plastic search is required:

1. attempt compact primitive-scale plastic search first;
2. use overcomplete fallback only if compact search fails a predeclared target/budget;
3. preserve the B007X negative result: large discovery capacity is not assumed beneficial.

## Router growth rule

Adding a new primitive must not require unconstrained full-history retraining in the primary incremental condition.

Always compare:

- full-retrain upper bound,
- naive new-class update,
- bounded incremental update.

Primary success depends on the bounded incremental condition.

## Sparse execution

Never count "compute savings" only from resident vs active parameter counts without labeling it correctly.

Report separately:

- resident parameters,
- active primitive parameters,
- router FLOPs,
- core FLOPs,
- primitive FLOPs,
- total FLOPs estimate,
- measured latency,
- peak memory,
- forward-call counts.

## No hidden dense execution

Non-selected primitives must receive exactly zero forward calls.

## Historical integrity

Do not rewrite negative results:

- B007X discovery-capacity advantage was not demonstrated.
- B008 did not demonstrate full online novelty handling.

## STOP discipline

Do not continue past a STOP gate simply because later tasks are implemented easily.

A negative controller/scaling result is scientifically valid.
