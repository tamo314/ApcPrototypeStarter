# Phase A.2 — Autonomous Controller & Scaling

**Status:** active after Phase A.1 Branch B Integration / ADR-0061.

## Goal

Move from learned routing over a fixed known universe to autonomous continual control under bank growth.

## Main sequence

```text
C001 scope/accounting audit [COMPLETED - ADR-0062]
        ↓
C002 controller instrumentation
        ↓
C003 incremental router update
        ↓
C004 bank competition/scaling
        ↓
C005 adequacy evidence interface
        ↓
C006 learned K/C/N controller
        ↓
C007 compact-first plastic policy
        ↓
C008 sequential K/C/N/R closed loop
        ↓
C009 multi-insertion stress
        ↓
C010 FLOPs/latency scaling
        ↓
C011 ablations
        ↓
C012 final audit
```

## STOP gates

### G1 — Incremental routing
C003 must pass before autonomous multi-growth claims.

### G2 — Learned adequacy
C006 must meet novelty/controller thresholds before C008.

### G3 — Sequential loop
C008 must pass before scaling the lifecycle to repeated insertions.

### G4 — Multi-growth
C009 must preserve old routing/performance while bank grows.

## Non-goals

Phase A.2 does not solve:

- natural-language task inference,
- few-shot semantic task induction,
- open-world language instructions.

Those belong to a later phase.

## Final deliverable

`docs/results/PHASE_A2_AUTONOMOUS_CONTROLLER_RESULT.md`
