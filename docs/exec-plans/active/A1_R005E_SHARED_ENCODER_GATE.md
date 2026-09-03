# A1-R005E Shared Queryable Representation Gate

**Status:** active after ADR-0043.

## Objective

Determine whether E-006A's gains survive with **one shared task-blind Stable Core** instead of one encoder per operation.

## Sequence

```text
E-006A strong per-operation result
        ↓
Shared encoder wiring
        ↓
Balanced mixed-operation joint training
        ↓
Shared-vs-specialized comparison
        ↓
COUNT/BIND None baseline audit
        ↓
        ├─ broad success -> Branch B
        ├─ success except SHIFT -> Branch B + SHIFT probe
        └─ multi-op collapse -> interference/specialization diagnosis
```

## Milestones

### S-M1
Shared encoder architecture.

### S-M2
Mixed-operation training gate.

### S-M3
Shared vs E-006A retention analysis.

### S-M4
COUNT/BIND argument-blind baseline audit.

### S-M5
Branch decision.

### S-M6
SHIFT structural probe if triggered.

### S-M7
Final diagnostic audit.

## Blocking rule

A1-R006 remains blocked until the shared gate is reviewed.
