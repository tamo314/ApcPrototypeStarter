# Representation × Operator Factorial Diagnostic

## 1. Factorial design

Two factors:

### Representation
- `FROZEN`: pretrained task-blind content encoder.
- `JOINT`: task-blind content encoder trained jointly with operator.

### Operator
- `COMPACT`: exact E-005 compact operator architecture.
- `HIGH_CAP`: exact E-004 high-capacity operator architecture, or closest faithful refactor.

| Cell | Representation | Operator | Status |
|---|---|---|---|
| C00 | Frozen | Compact | E-005 measured |
| C01 | Frozen | High-cap | E-004 measured |
| C10 | Joint | Compact | E-006A |
| C11 | Joint | High-cap | E-006B conditional |

## 2. Why C10 is decisive

C00 -> C10 changes representation trainability while compact compute stays fixed.

If C10 rises sharply, the frozen latent geometry was difficult for a compact operator to use.

If C10 stays near C00 while C01 is high, compact operator structure is the stronger bottleneck.

## 3. Representation-accessibility recovery score

For a higher-is-better metric `m`:

`R_access = (m_C10 - m_C00) / max(eps, m_C01 - m_C00)`

Interpret only when C01 materially exceeds C00.

Suggested interpretation:
- `R_access >= 0.70`: strong representation-accessibility effect
- `0.30 <= R_access < 0.70`: mixed contribution
- `R_access < 0.30`: operator bottleneck favored

Report this for:
- Correct exact match,
- causal gap.

Do not collapse the decision to one scalar if the two disagree.

## 4. Task-blind invariance

For the same content under different tasks/arguments:

`E_joint(x, t1) == E_joint(x, t2)`

because task/argument is structurally absent from encoder input.

Regression-test this invariant.

## 5. Representation drift diagnostics

Compare Frozen vs Joint representations using:
- token probe,
- position probe,
- optional CKA/cosine or another declared representation-change measure,
- compact-operator downstream performance.

The question is whether accessibility improves without losing basic content information.

## 6. C11 role

Run C11 only for unresolved operations.

Interpretation:
- C11 high + C10 low -> compact operator bottleneck
- C11 high + C10 high -> representation accessibility mattered
- C11 low -> interface/factorization/optimization concern

## 7. Operation-specific expectations

### SHIFT
Likely operator-inductive-bias sensitive: relative/modular position arithmetic.

### SELECT
May benefit from improved representation accessibility; E-005 showed a high-performing seed.

### BIND
Keyed retrieval is compatible with attention; unstable E-005 may indicate poor frozen latent geometry.

### COUNT
May require explicit aggregation/counting rather than generic retrieval attention.

## 8. Final interpretation

### Branch A — Operator/Heterogeneous Primitive
Frozen high-cap strong; Joint+Compact weak; structural compact probe succeeds.

### Branch B — Queryable Representation Learning
Joint+Compact recovers a large fraction of Frozen+High-cap.

### Branch C — Interface/Factorization
Joint+High-cap also fails materially.

### Branch D — Mixed
Operations split across branches.
