# Experiment Plan — Shared Queryable Representation Gate

## 1. Goal

Test whether E-006A's representation-accessibility gain survives when all four operations share one task-blind encoder.

## 2. Main model

One shared `DecoderOnlyTransformer` content encoder.

Four operation-specific compact operators using the unchanged E-006A/E-005 compact architecture.

Per-operation argument encoders and readouts are allowed.

## 3. Training

Mixed-operation online training with balanced default sampling:

- SHIFT: 25%
- SELECT: 25%
- COUNT: 25%
- BIND: 25%

Counterfactual argument groups remain required.

## 4. Budget/accounting

Report:
- total optimizer steps,
- examples per operation,
- total examples,
- encoder updates,
- per-operator updates.

Use dev-tier runs first, then >=5 seeds for decision evidence.

## 5. Evaluation

Per operation:
- Correct exact
- token accuracy where applicable
- effectful Wrong argument
- None
- causal gap

Also compute:

`R_shared_correct = SharedCorrect / E006A_Correct`

`R_shared_gap = SharedGap / E006A_Gap`

## 6. Strong shared-representation support

Per operation:
- `R_shared_correct >= 0.90`
- `R_shared_gap >= 0.90`
- strong argument causality
- no severe seed collapse

Global strong support:
- at least 3/4 operations satisfy those ratios,
- the remaining failure is mechanistically interpretable.

## 7. Branch-B criterion

Branch B is strongly supported if:
- SELECT/COUNT/BIND retain >=90% of E-006A Correct and causal gap,
- one encoder parameter set is truly shared,
- task-blind invariance remains exact,
- >=5-seed result is robust.

SHIFT may remain below final accuracy target if it stays materially above E-005 and its residual is consistent with operator mismatch.

## 8. Failure criterion

Evidence against global Branch B if:
- multiple operations lose >30% of E-006A performance,
- the loss remains after basic optimization sanity,
- imbalance does not explain it.

## 9. Allowed optimization sanity

Without changing architecture:
- warmup
- cosine decay
- gradient clipping
- balanced curriculum scheduling

Not allowed:
- separate encoders
- operation-specific encoder adapters
- PCGrad/gradient surgery inside the primary gate
- task tokens into encoder

## 10. Argument-blind baseline audit

For COUNT and BIND report:
- majority-target baseline
- simple content-only baseline if well-defined
- observed None accuracy across E-004/E-005/E-006A/shared gate

Do not change historical thresholds here.

## 11. SHIFT follow-up trigger

If SELECT/COUNT/BIND strongly support sharing but SHIFT remains weak, authorize a separate compact SHIFT structural probe.

## 12. Outcomes

### A
Shared encoder broadly retains E-006A -> Branch B formally supported.

### B
Broad success except SHIFT -> Branch B + heterogeneous SHIFT operator.

### C
Several operations collapse -> representation specialization/interference unresolved.

### D
Shared encoder fails despite balanced training -> reconsider representation-sharing assumptions.
