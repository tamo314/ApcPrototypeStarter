# Experiment Plan — A1-B007X Discovery-to-Compact Consolidation

## Prerequisite

A1-B007 must pass first.

Required:
- recurrence reuses consolidated primitive,
- no plastic allocation,
- no bank growth,
- no reconsolidation,
- zero adaptation.

If B007 fails, stop.

## X1 — Measurement/shadow audit

Confirm:
- exact B006 temp params,
- exact B006 candidate params,
- B006 parameter ratio (~1.0),
- per-operation historical before/after metrics,
- representative composition before/after metrics.

Resolve ambiguous aggregate forgetting metrics before compression claims.

## X2 — Novel-task calibration

For each candidate novel task:
1. run bank search;
2. run composition search with declared depth/beam;
3. require best frozen solution below novelty threshold;
4. verify no solution leakage.

Suggested novelty threshold:

`best_existing_EM < 0.20`

Use >=2 tasks.

## X3 — Temporary capacity ladder

Train T0/T1/T2 and optional T3 under matched:
- data stream,
- splits,
- optimizer family,
- max steps,
- stopping rule.

Decision seeds: >=5.

Report:
- final EM,
- success rate,
- steps/examples to 0.90/0.95,
- learning curve,
- wall-clock,
- peak memory,
- params.

## Discovery-advantage criterion

Strong evidence if either:

### Reliability gap
- large tier mean EM >=0.95,
- compact tier mean EM <=0.80,
- robust across >=5 seeds;

or

### Efficiency gap
Both pass, but large tier reaches 0.95 with <=50% of compact median steps/examples and no worse seed reliability.

Otherwise do not claim a discovery-capacity advantage.

## X4 — Compact direct-learning control

Train the <=25k candidate directly from labels using the same discovery support/data budget.

Mandatory before central-hypothesis claims.

## X5 — Functional distillation

Distill the smallest robustly successful overcomplete temporary tier into <=25k candidate.

Targets:
- candidate EM >=0.90,
- retention >=0.95,
- functional agreement >=0.99,
- candidate/temp params <=0.25 for strong compression claim.

## X6 — Shadow validation/promotion

Before install:
- canonical forgetting <=2pp per op,
- composition forgetting <=2pp,
- Core unchanged,
- existing bank unchanged.

On pass:
- install exactly one candidate,
- bank size +1,
- release temp to 0.

On fail:
- abort promotion,
- preserve temp fallback,
- bank unchanged.

## X7 — Persistent-only recurrence

After install:
- save persistent state,
- destroy temp/teacher objects,
- fresh runtime,
- load persistent state only.

Targets:
- recurrence EM >=0.95,
- adaptation steps 0,
- temporary params 0,
- bank unchanged,
- same primitive ID reused,
- no consolidation triggered.

## Final report

Separate:
1. Compressibility
2. Discovery advantage
3. Lifecycle persistence

Do not merge them into one PASS label.
