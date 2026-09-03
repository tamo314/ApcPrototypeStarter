# A1-R005E E006+ Revised Diagnostic Plan

**Starts after:** A1-R005E-005 / ADR-0042.

## Objective

Resolve the mixed E-005 result by isolating representation accessibility from compact operator expressivity.

## Core design

```text
                         Operator
                  Compact        High-cap
Representation
Frozen             E-005          E-004
Joint              E-006A         E-006B*
```

`*` E-006B is conditional.

## Milestones

### E6-M1 — Joint + Compact
Fill C10. Highest-priority experiment.

### E6-M2 — Per-operation factorial diagnosis
Compute representation-accessibility recovery.

### E6-M3 — Joint + High-cap
Fill C11 only where needed.

### E6-M4 — Minimal structural probes
Use operation-specific compact diagnostics only after factorial evidence.

### E6-M5 — Cross-operation diagnosis
Mixed branch is allowed.

### E6-M6 — Final decision report
Recommend the next phase; do not implement it.

## Blocking rule

A1-R006 remains blocked throughout.
