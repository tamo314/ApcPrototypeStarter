# A1-R005 Diagnostic and Retry Plan

**Status:** closed — negative diagnostic result after D-001 through D-008.
D-009 was not run because preceding diagnostics produced no justified
configuration for a meaningful final retry.

## Objective

Do not immediately rerun the same benchmark with more compute.

First determine whether failure comes from:
1. evaluation/control ambiguity,
2. argument encoding information loss,
3. training shortcut,
4. insufficient conditioning capacity,
5. wrong primitive computation type.

## Sequence

```text
Existing-run re-analysis
        ↓
Argument-effect audit
        ↓
SELECT encoder audit/fix
        ↓
COUNT counterfactual gate
        ↓
small capacity/training sweep
        ↓
conditioning architecture comparison if needed
        ↓
BIND gate
        ↓
SHIFT gate
        ↓
SELECT gate
        ↓
final 4-operation R005 retry
```

## Milestones

### R005-M1 — Existing-run causal diagnostics
No retraining.

### R005-M2 — Argument encoder integrity
Semantically distinct arguments must be distinguishable.

### R005-M3 — COUNT counterfactual causality
First hard STOP GATE.

### R005-M4 — Minimal sufficient capacity
Only after M3 setup is valid.

### R005-M5 — Conditioning form
Compare additive vs multiplicative/FiLM vs basis modulation if needed.

### R005-M6 — BIND causality

### R005-M7 — SHIFT sequence causality

### R005-M8 — SELECT sequence causality

### R005-M9 — Final H2c retry
All four operations, >=5 seeds.

Only M9 PASS unblocks A1-R006.

## Scientific rule

The retry is not successful merely because Correct rises.

The key outcome is:

`Correct >> effectful Wrong argument`

while Wrong family and None remain low.

## Scope rule

Do not modify R001 task-blind invariant, R002 leakage control, or R003 parameter-free primitive result.
