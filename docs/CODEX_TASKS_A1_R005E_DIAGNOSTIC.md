# Codex Task Queue — A1-R005E Representation / Operator Isolation

Execute exactly one task at a time.

A1-R006 remains blocked.

---

## A1-R005E-001 — Close the R005 retry

Goal: formally close D-001 through D-008 as a negative diagnostic result.

Work:
1. preserve existing retry artifacts,
2. mark D-009 skipped/superseded,
3. create/update retry audit,
4. explain why repeating the same four-operation baseline adds little information,
5. update active-plan pointers.

Acceptance:
- history intact,
- D-009 not reported as executed,
- A1-R006 remains blocked,
- new diagnostic phase active.

---

## A1-R005E-002 — Frozen `h_content` information audit

Goal: measure information retained by the frozen task-blind representation.

Using the same relevant frozen checkpoints:
1. token identity probe,
2. absolute position probe,
3. full content-sequence reconstruction,
4. BIND key/value role probe,
5. optional adjacency/pair probe.

Do not alter Stable Core.

Report per seed:
- token accuracy,
- position accuracy,
- reconstruction token accuracy,
- reconstruction exact match,
- BIND role/pair metrics.

Acceptance: diagnostic only; explicitly state what information is and is not recoverable.

---

## A1-R005E-003 — Oracle latent operator benchmark

Goal: determine whether perfect addressing on frozen hidden states enables strong decoding.

Implement oracle-only:
- SHIFT hidden-state permutation,
- SELECT ordered gather,
- BIND oracle key-location -> associated value-state readout,
- COUNT oracle match mask + documented aggregation/readout.

Oracle may use raw input symbols to identify positions, never target output.

Acceptance targets:
- SHIFT/SELECT/BIND exact >=0.90 and token >=0.98,
- COUNT exact >=0.90.

Interpret failure as representation/decoder/interface evidence, not learned-routing failure.

---

## A1-R005E-004 — Frozen high-capacity operator upper bound

Goal: test whether expressive learned operator solves tasks from frozen `h_content + argument`.

Work:
1. freeze Stable Core,
2. add intentionally expressive cross-position operator,
3. arguments enter operator only,
4. train per operation first,
5. use counterfactual argument groups,
6. run >=5 seeds for branch evidence.

Suggested: 2–4 attention/Transformer blocks, sufficient width.

Arms:
- Correct,
- effectful Wrong argument,
- None where meaningful.

Targets:
- Correct exact >=0.90,
- SHIFT/SELECT token >=0.98,
- effectful Wrong argument <=0.30,
- causal gap >=0.50.

Branch:
- substantial PASS -> E-005,
- material FAIL -> skip E-005 and run E-006.

---

## A1-R005E-005 — Compact cross-position operator probe

Prerequisite: E-004 substantially passes.

Goal: test whether upper bound can be approximated at primitive scale.

Implement one small operator, preferably:
- single cross-attention block,
- argument-derived query/control,
- frozen content states as K/V,
- small projections.

Compare:
- historical V0/FiLM,
- high-cap upper bound.

Positive evidence:
- materially higher Correct than historical R005,
- materially larger causal gap,
- substantial fraction of upper bound,
- primitive-scale size.

Output whether heterogeneous operator primitives deserve a next phase.

---

## A1-R005E-006 — Joint task-blind representation control

Run if E-004 fails/severely limited.

Goal: test whether representation can become sufficient without task leakage.

Jointly train:
- content encoder,
- strong argument-conditioned operator.

Constraints:
- content encoder sees content only,
- argument reaches operator only,
- counterfactual task-blind invariance maintained.

Evaluate same metrics as E-004 and compare representation probes before/after.

Interpretation:
- joint succeeds, frozen fails -> Queryable Representation Learning branch,
- joint fails -> Interface/Factorization branch.

---

## A1-R005E-007 — Cross-operation diagnosis table

Create table for COUNT/BIND/SHIFT/SELECT with:
- historical primitive result,
- representation probes,
- oracle latent operator,
- frozen high-cap upper bound,
- compact operator if run,
- joint representation control if run.

Classify each as:
- representation bottleneck,
- operator bottleneck,
- interface bottleneck,
- mixed/uncertain.

Every classification must cite measured evidence.

---

## A1-R005E-008 — Final diagnostic decision report

Create:
`docs/results/A1_R005E_DIAGNOSTIC_RESULT.md`

Required sections:
1. retry closure,
2. isolated question,
3. frozen representation audit,
4. oracle latent operators,
5. frozen high-cap upper bound,
6. compact operator if run,
7. joint representation if run,
8. per-operation diagnosis,
9. completed decision matrix,
10. recommended next branch:
   - Operator / Heterogeneous Primitive,
   - Queryable Representation Learning,
   - Interface / Factorization Reconsideration,
   - Mixed,
11. what remains unproven.

Audit/decision only. Do not implement the selected phase.

A1-R006 stays blocked until user approval.
