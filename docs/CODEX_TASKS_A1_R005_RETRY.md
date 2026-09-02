# Codex Task Queue — A1-R005 Diagnostic and Retry

Execute exactly one task at a time.

A1-R006 remains blocked until A1-R005D-009 passes.

> A1-R005D-009 is intentionally skipped/superseded after D-001 through D-008
> produced convergent negative evidence. Do not run the same final four-operation
> retry without a new mechanistic hypothesis.
>
> Close the retry via A1-R005E-001 and continue with
> `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`.

---

## A1-R005D-001 — Re-analyze existing R005 artifacts

Goal: expand diagnostics without retraining.

Work:
- aggregate Correct / Wrong-argument / Wrong-family / None by operation and seed,
- add token accuracy where recoverable,
- add output length,
- add `argument_effect_rate`,
- compute Wrong-argument metrics on effectful examples only.

Acceptance:
- all 4 operations and 5 seeds represented,
- existing artifacts unchanged,
- raw and effectful wrong-argument scores clearly separated.

---

## A1-R005D-002 — Argument encoder integrity audit

Goal: detect representation collisions before training.

Work:
- audit all argument encoders,
- SELECT: test `[1,3]` vs `[3,1]`, same-length distinct sequences, repeated indices if legal,
- integer encoders: audit modulo/wraparound aliasing against legal domain.

Acceptance:
- structural collisions documented,
- SELECT order-sensitive collisions eliminated,
- illegal/out-of-domain scientific arguments rejected unless aliasing is intentional.

STOP GATE: do not retrain SELECT with known non-injective encoding.

---

## A1-R005D-003 — Order-preserving SELECT argument encoder

Goal: replace mean pooling if D-002 confirms order loss.

Preferred minimal form:
- index embedding,
- argument-position embedding,
- small masked sequence encoder.

Acceptance:
- `[1,3]` and `[3,1]` distinct,
- variable length supported,
- deterministic,
- padding/mask tested.

Skip only if D-002 proves current representation sufficient.

---

## A1-R005D-004 — Counterfactual COUNT gate

Goal: test argument causality when argument use is unavoidable.

Work:
- COUNT only,
- same content paired with multiple targets whose counts differ,
- frozen Stable Core,
- one COUNT family,
- baseline rank/steps first.

Evaluate Correct / effectful Wrong argument / None over >=5 seeds.

Acceptance:
- Correct >=0.90
- effectful Wrong argument <=0.30
- causal gap >=0.50

STOP GATE.

---

## A1-R005D-005 — Minimal capacity/training sweep

Goal: test optimization/capacity only after D-004 is identifiable.

Staged sweep:
1. steps x2
2. steps x4 if needed
3. rank 16
4. arg_dim x2

Log:
- learning curves,
- argument-path gradient norms if easy,
- Correct/effectful-Wrong gap.

Acceptance:
select smallest robust passing configuration.

If none pass, continue to D-006.

---

## A1-R005D-006 — Conditioning architecture comparison

Goal: test whether additive conditioning is the bottleneck.

Variants:
- V0 additive,
- V1 FiLM/gated multiplicative,
- optional V2 basis modulation.

Keep Stable Core and data fixed.

Acceptance:
- choose by causal gap,
- report parameter counts,
- do not choose a model that raises Correct and Wrong together.

STOP if controlled COUNT still cannot pass.

---

## A1-R005D-007 — BIND counterfactual gate

Use same-content/multiple-key groups with deliberately different outputs.

Acceptance:
- Correct >=0.90
- effectful Wrong argument <=0.30
- causal gap >=0.50
- one persistent BIND family across keys.

---

## A1-R005D-008 — SHIFT and SELECT sequence gates

SHIFT:
- exact match,
- token accuracy,
- length-stratified metrics,
- effectful Wrong argument.

SELECT:
- order-preserving argument encoder mandatory,
- same metrics.

Use the selected low-rank conditioned architecture first.

If it fails despite strong COUNT/BIND results, implement a minimal tiny cross-attention primitive as an explicit alternative class.

Acceptance:
- token accuracy >=0.95
- exact match >=0.85
- effectful Wrong-argument performance materially lower.

Record which primitive class is required.

---

## A1-R005D-009 — Final parameterized primitive retry

Goal: rerun H2c across SHIFT / SELECT / COUNT / BIND.

Use the smallest architecture/config justified by D-001 through D-008.

Run >=5 seeds.

Evaluate:
1. Correct
2. effectful Wrong argument
3. Wrong family
4. None

Retain raw Wrong-argument metrics for historical comparison.

Acceptance:
- aggregate Correct >=0.90
- each operation Correct >=0.85
- effectful Wrong argument <=0.30
- Wrong family <=0.30
- None <=0.30
- aggregate causal gap >=0.50
- persistent family count independent of argument values

STOP GATE: only PASS unblocks A1-R006.

---

## A1-R005D-010 — Retry audit

Create/update a result report covering:
- original R005 failure,
- diagnostic findings,
- encoder collisions,
- counterfactual COUNT,
- capacity/training findings,
- conditioning comparison,
- sequence-routing findings,
- final retry verdict,
- whether heterogeneous primitive classes are supported by evidence.

No new architecture in this task.
