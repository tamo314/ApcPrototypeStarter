# Codex Task Queue — A1-R005E E006+ Revision

Execute exactly one task at a time.

A1-R006 remains blocked.

---

## A1-R005E-006A — Joint task-blind representation + unchanged compact operator

### Goal

Test whether E-005's compact operator becomes effective when the task-blind content representation is trained jointly for accessibility.

### Critical control

Reuse the same compact operator class and dimensions as E-005.

Do not make the operator stronger.

### Work

1. Make the task-blind content encoder trainable.
2. Keep task/argument tokens absent from content encoder input.
3. Train content encoder + compact operator + fresh readout jointly.
4. Reuse counterfactual groups.
5. Evaluate SHIFT / SELECT / COUNT / BIND.
6. Run >=5 seeds for decision evidence.
7. Load saved E-004/E-005 summaries and compute C00/C01/C10 comparisons.

### Required validation

For identical content under different tasks/arguments:
- encoder token input is identical,
- `h_content` is identical within declared tolerance at inference.

### Required metrics

- Correct exact
- token accuracy
- effectful Wrong
- None
- causal gap
- train loss
- encoder/operator gradient summaries
- parameter counts
- `R_access` for Correct and causal gap

### Rule

Do not classify a branch until optimization sanity is checked.

---

## A1-R005E-006A2 — Optimization sanity control

### Conditional

Run only if E-006A shows strong seed bimodality or obvious non-convergence.

### Allowed changes

No architecture change.

May use:
- warmup
- cosine LR schedule
- gradient clipping
- deterministic initialization handling

### Acceptance

Report every seed.

If weak performance remains stable, operator-bottleneck evidence strengthens.

---

## A1-R005E-006B — Joint task-blind representation + high-capacity operator

### Conditional

Run only for operations unresolved after E-006A/A2.

### Goal

Fill the C11 factorial cell.

### Work

Use the E-004 high-capacity operator architecture while making the task-blind content encoder jointly trainable.

Arguments remain operator-only.

### Interpretation

- C11 high + C10 low -> compact operator bottleneck
- C11 high + C10 high -> representation accessibility important
- C11 low -> interface/factorization or optimization concern

### Target

Reuse E-004 high-capacity criteria.

---

## A1-R005E-006C — Operation-specific compact structural probes

### Prerequisite

Complete factorial diagnosis first.

### Goal

Test minimal operation-specific inductive biases without launching a production redesign.

### Suggested probes

#### SHIFT
Relative/modular-position routing.

#### SELECT
Ordered slot/index-conditioned gather attention.

#### BIND
Keyed retrieval attention.

#### COUNT
Query-match plus explicit aggregation/counting.

### Constraints

- primitive-scale
- operation-specific
- no 2.44M upper-bound model
- report parameter count and compute

### Positive evidence

- materially larger causal gap than E-005
- substantial fraction of E-004 upper-bound performance
- robust across seeds

---

## A1-R005E-007R — Revised cross-operation diagnosis

### Goal

Classify each operation using the completed factorial evidence.

### Required table

For COUNT/BIND/SHIFT/SELECT include:

- historical pointwise result
- C00 Frozen+Compact
- C01 Frozen+High-cap
- C10 Joint+Compact
- C11 Joint+High-cap if run
- structural probe if run
- Correct recovery
- causal-gap recovery
- optimization notes
- final classification

### Allowed classifications

- Representation accessibility
- Operator architecture
- Interface/factorization
- Mixed/uncertain

Do not force one explanation across all operations.

---

## A1-R005E-008R — Final branch decision report

### Goal

Recommend the next research phase.

### Output

Create:

`docs/results/A1_R005E_DIAGNOSTIC_RESULT_FINAL.md`

### Required sections

1. R005/R005D closure
2. E-002/E-003 findings
3. E-004 high-cap upper bound
4. E-005 compact result
5. E-006A factorial result
6. E-006B if run
7. E-006C if run
8. completed 2x2 table
9. per-operation diagnosis
10. recommended branch
11. whether Stable Core representation learning should change
12. whether heterogeneous primitive classes are justified
13. whether old A1-R006 should be resumed, revised, or superseded
14. unresolved risks
15. what remains unproven

### Rule

Decision/audit only.

Do not implement the selected next architecture phase.

A1-R006 remains blocked until user approval.
