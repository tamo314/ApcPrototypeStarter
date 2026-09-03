# Codex Task Queue — A1-R005E Shared Queryable Representation Gate

Execute exactly one task at a time.

A1-R006 remains blocked.

---

## A1-R005E-S001 — Shared encoder architecture wiring

### Goal

Create a diagnostic model with exactly one task-blind content encoder shared by all four operations.

### Work

1. Reuse E-006A compact operator architecture unchanged.
2. Instantiate one shared content encoder.
3. Instantiate four operation-specific compact operators.
4. Use per-operation argument encoders/readouts.
5. Use oracle operation selection.
6. Ensure task/argument never enters the encoder.

### Tests

Prove:
- one encoder object / parameter set,
- all operations call the same encoder,
- identical content produces identical encoder tokens/states across operations,
- no operation-specific modules exist inside the encoder.

### Acceptance

Architecture invariants pass.

No milestone benchmark yet.

---

## A1-R005E-S002 — Balanced mixed-operation training gate

### Goal

Train one shared encoder from all four operation losses.

### Work

1. Balanced mixed-operation sampling.
2. Preserve counterfactual groups.
3. Jointly train:
   - shared encoder,
   - selected compact operator,
   - selected readout.
4. Log per-operation loss/update counts.
5. Run dev seeds, then >=5 decision seeds.

### Required metrics

Per operation:
- Correct exact
- token accuracy
- effectful Wrong
- None
- causal gap
- final loss

Global:
- operation sample counts
- encoder update count
- per-operator update count
- encoder gradient norm by operation if practical

### Acceptance

Complete all measurements; no branch claim yet.

---

## A1-R005E-S003 — Shared-vs-specialized retention analysis

### Goal

Measure how much of E-006A survives sharing.

### Work

Load E-006A summaries.

Compute per operation:
- `R_shared_correct`
- `R_shared_gap`

Also compare to E-005 and E-004.

### Interpretation

- >=0.90 strong shared support
- 0.70–0.90 mixed
- <0.70 strong specialization dependence

### Acceptance

Produce a per-operation evidence table.

---

## A1-R005E-S004 — COUNT/BIND argument-blind baseline audit

### Goal

Test whether historical None ceiling 0.30 is below the natural argument-blind baseline.

### Work

For COUNT and BIND report:
- majority target baseline,
- simple content-only baseline if well-defined,
- observed None from E-004/E-005/E-006A/shared gate.

### Rule

Do not retroactively change thresholds.

### Output

Recommend whether future gates should use a fixed or baseline-relative None criterion.

---

## A1-R005E-S005 — Shared-gate branch decision

### Goal

Decide whether Branch B is sufficiently supported.

### Strong Branch B support

Recommend Branch B if:
- SELECT/COUNT/BIND retain >=90% of E-006A Correct and gap,
- one encoder is truly shared,
- task-blind invariance holds,
- result is robust across >=5 seeds.

### SHIFT-only residual

Recommend:

**Shared Queryable Representation + Heterogeneous SHIFT Operator Probe**

Do not reject Branch B globally.

### Multi-operation collapse

Do not adopt Branch B globally.

Recommend representation-interference diagnosis.

### Output

Create:

`docs/results/A1_R005E_SHARED_ENCODER_GATE_RESULT.md`

---

## A1-R005E-S006 — SHIFT compact structural probe

### Conditional

Run only if S005 finds strong shared-representation support but SHIFT remains materially below target.

### Goal

Test whether SHIFT needs explicit modular/relative positional bias.

### Candidate minimal design

Choose one predeclared:
- relative-position attention bias,
- rotary/phase-like position encoding,
- explicit modular offset routing.

### Controls

Keep successful shared encoder fixed/frozen.

Compare to:
- shared compact baseline,
- E-004 SHIFT,
- E-006A SHIFT.

### Positive evidence

- large Correct/token improvement
- causal gap remains strong
- primitive-scale size
- robust across seeds

---

## A1-R005E-S007 — Final diagnostic audit

### Goal

Integrate the shared gate into the overall decision.

### Output

Update/create:

`docs/results/A1_R005E_DIAGNOSTIC_RESULT_FINAL.md`

### Required conclusions

1. whether E-006A gains generalize to one shared encoder,
2. whether Branch B is formally supported,
3. whether SHIFT needs a heterogeneous operator,
4. whether COUNT/BIND future None criteria should be baseline-relative,
5. whether old A1-R006 should resume, be revised, or be superseded.

### Rule

Decision only; do not implement production architecture.
