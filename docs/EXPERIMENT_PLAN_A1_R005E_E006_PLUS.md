# Experiment Plan — A1-R005E E006+ Revision

## 1. Goal

Complete the Representation × Operator diagnosis after E-005.

## 2. Fixed historical cells

Use saved E-004/E-005 results as fixed comparison values.

- C01 = Frozen + High-cap, E-004
- C00 = Frozen + Compact, E-005

## 3. E-006A — Joint task-blind representation + Compact operator

### Architecture

Use E-005 compact operator unchanged in structure and scale.

Change only:
- content encoder becomes trainable.

Keep:
- content-only encoder input,
- argument enters operator only,
- counterfactual groups,
- Correct/Wrong/None controls,
- readout semantics.

### Training

Train content encoder + compact operator + fresh readout jointly.

Development runs may use 1–2 seeds.
Decision evidence requires >=5 seeds.

Training budget may change because the encoder is now trainable, but operator capacity may not.

### Required metrics

Per operation/seed:
- Correct exact
- token accuracy where applicable
- effectful Wrong exact/token
- None exact/token
- causal gap
- final training loss
- encoder/operator gradient summaries
- parameter counts
- task-blind invariance

Compute `R_access` for Correct and causal gap.

### Strong representation-accessibility evidence

- `R_access >= 0.70` on Correct or causal gap,
- material improvement over E-005,
- robust across >=5 seeds,
- task blindness preserved.

### Strong operator-bottleneck evidence

- E-004 materially exceeds E-005,
- E-006A remains close to E-005,
- `R_access < 0.30`,
- no obvious optimization failure.

## 4. Optimization sanity

Because E-004/E-005 showed seed-dependent convergence, branch classification must not rely on a clearly failed optimizer run.

Allowed without architecture change:
- warmup
- cosine LR decay
- gradient clipping
- deterministic initialization policy

Predeclare the stabilization protocol.
Report all seeds; never select best seed as primary.

## 5. E-006B — Joint task-blind representation + High-cap operator

Run only for unresolved operations.

Purpose: fill C11.

Targets reuse E-004:
- Correct exact >=0.90
- SHIFT/SELECT token >=0.98
- effectful Wrong <=0.30
- causal gap >=0.50

Interpretation:
- C11 high, C10 low -> operator bottleneck
- C11 high, C10 high -> representation accessibility mattered
- C11 low -> interface/factorization/optimization concern

## 6. E-006C — Minimal operation-specific structural probes

Run only after factorial classification.

Examples:
- SHIFT: relative/modular position routing
- SELECT: ordered gather / slot-conditioned attention
- BIND: keyed retrieval attention
- COUNT: query-match + explicit aggregation/counting

These remain diagnostic primitive-scale probes, not production architecture.

Positive evidence:
- materially larger causal gap than E-005
- substantial fraction of E-004 upper bound
- robust across seeds
- primitive-scale parameters

## 7. Branch rules

### Branch A
Frozen high-cap strong, Joint+Compact weak, structural compact probe improves.

### Branch B
Joint+Compact strongly outperforms Frozen+Compact and recovers much of E-004.

### Branch C
Joint+High-cap also fails materially.

### Branch D
Operations split.

## 8. A1-R006

A1-R006 remains blocked.

The final report may recommend superseding the old A1-R006 rather than resuming it unchanged.
