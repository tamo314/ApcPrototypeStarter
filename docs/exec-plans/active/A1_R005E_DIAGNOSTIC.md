# A1-R005E — Representation / Operator Isolation

**Status:** active after closing A1-R005 Retry D-001 through D-008 as a negative diagnostic result.

## Objective

Determine whether the next architecture phase should focus on:

1. new computational primitive/operator classes,
2. richer task-blind representation learning,
3. or revisiting the causal state/interface factorization.

## Why this phase exists

The retry showed that simple local fixes do not explain R005 failure.

The unresolved confound is:

> A weak primitive can fail because its computation class is inadequate, but it can also fail because the frozen representation does not expose the information the primitive needs.

## Sequence

```text
Close R005 retry
      ↓
Frozen representation probes
      ↓
Oracle latent operators
      ↓
Frozen high-capacity upper-bound operator
      ↓
      ├── succeeds ──> compact cross-position operator probe
      │
      └── fails ─────> joint task-blind representation control
                              ↓
                        phase decision report
```

## Milestones

### E-M0 — Retry closure
Record D-001 through D-008 as final retry evidence. D-009 is skipped/superseded.

### E-M1 — Representation information audit
Measure token, position, sequence reconstruction, and BIND-relevant role information.

### E-M2 — Oracle latent operator
Test hidden-state/decoder compatibility with perfect operation addressing.

### E-M3 — Frozen upper bound
Use a strong learned operator on frozen `h_content`. This is the main branch point.

### E-M4A — Compact operator probe
If frozen upper bound succeeds, test a small cross-position operator.

### E-M4B — Joint representation control
If frozen upper bound fails, jointly learn a task-blind representation with a strong operator.

### E-M5 — Decision audit
Fill `NEXT_PHASE_DECISION_MATRIX.md`.

Do not implement the chosen next architecture phase inside this chain.

## Blocking rule

A1-R006 remains blocked for the entire phase.

The user explicitly selects the next branch after the final diagnostic report.
