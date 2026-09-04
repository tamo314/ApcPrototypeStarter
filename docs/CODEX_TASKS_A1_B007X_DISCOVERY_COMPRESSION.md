# Codex Task Queue — A1-B007X Discovery-to-Compact Consolidation

Execute exactly one task at a time.

Prerequisite: A1-B007 PASS.

---

## A1-B007X-001 — Consolidation metric and shadow audit [PASSED - ADR-0053]

### Goal

Repair measurement ambiguity before compression claims.

### Work

1. Report B006 temp/candidate parameter counts. **[DONE: 17,098 temp vs 17,098 candidate]**
2. Record B006 parameter ratio. **[DONE: R_param = 1.0000]**
3. Classify B006 as functional consolidation, not parameter compression. **[DONE: classified as functional consolidation]**
4. Add per-operation canonical before/after metrics. **[DONE: all 8 canonical ops evaluated individually, 0.00% max forgetting]**
5. Add representative composition before/after metrics. **[DONE: all 6 representative recipes evaluated individually, 0.00% max forgetting]**
6. Preserve B006 artifacts unchanged. **[DONE: runs/phase_a1_consolidation_benchmark/ verified byte-for-byte intact]**

### Acceptance

- no opaque aggregate is the sole forgetting metric, **[PASSED: published individual per-task records in runs/phase_a1_consolidation_shadow_audit/]**
- canonical tasks listed individually, **[PASSED: all 8 canonical operations evaluated and listed individually]**
- B006 historical result preserved. **[PASSED: historical B006 run artifacts preserved unchanged]**

---

## A1-B007X-002 — Novel-task and capacity-ladder harness

### Goal

Create controlled discovery-capacity benchmark.

### Work

1. Define >=2 novel operations.
2. Verify bank/composition failure first.
3. Implement temp tiers:
   - ~17k–25k,
   - ~64k,
   - ~128k,
   - optional ~256k.
4. Keep topology comparable where possible.
5. Use matched splits/accounting.

### Acceptance

- novelty verified,
- capacity counts measured,
- no oracle leakage,
- common data protocol established.

---

## A1-B007X-003 — Temporary discovery capacity sweep

### Goal

Measure whether overcomplete capacity improves discovery.

### Work

Across >=5 seeds report per task/tier:

- final EM,
- success rate,
- steps/examples to 0.90,
- steps/examples to 0.95,
- learning-curve AUC,
- wall-clock,
- peak memory.

### Strong evidence

Either:
- large >=0.95 while compact <=0.80,

or:
- large reaches 0.95 with <=50% of compact median steps/examples and no worse reliability.

### Rule

Preserve negative results. Do not blindly scale.

### Status
- **Status:** **COMPLETE** (Negative Result Preserved, ADR-0055)
- **Verdict:** No temporary discovery-capacity advantage demonstrated across 5 seeds and 3 novel tasks.
  - `SWAP_PAIRS`: T0 Compact achieves 99.80% EM (100% success) in 175.0 steps; T2 Overcomplete speedup ratio is 57.14% (> 50% threshold).
  - `INVERT_HALF`: T2 reaches 90.00% EM (< 95% threshold); reliability gap not satisfied.
  - `ROTATE_TRIPLETS`: T0 Compact achieves 94.20% EM (60% success), outperforming T2 Overcomplete (92.90% EM, 40% success).
- **Run Artifacts:** `runs/phase_a1_discovery_capacity_sweep/` (`report.json`, `summary.json`, `system.json`, `config.yaml`).
- **Authorization:** Authorizes Task A1-B007X-004 (Compact direct-learning control).

---

## A1-B007X-004 — Compact direct-learning control

### Goal

Test whether final compact architecture could learn directly just as well.

### Work

Train <=25k candidate from labels under matched discovery budget.

### Acceptance

Report matched:
- params,
- final EM,
- steps/examples thresholds,
- seed success.

Mandatory control.

---

## A1-B007X-005 — Overcomplete-to-compact functional distillation

### Goal

Distill selected overcomplete temp solution into compact persistent primitive.

### Work

1. Freeze temp teacher.
2. Generate separate distillation inputs.
3. Train <=25k candidate on teacher function.
4. Evaluate held-out inputs.
5. Measure functional agreement.

### Acceptance

- candidate EM >=0.90,
- retention >=0.95,
- agreement >=0.99,
- candidate/temp params <=0.25 for strong compression claim.

STOP if retention fails.

---

## A1-B007X-006 — Shadow validation, promotion, release

### Goal

Validate safety before install.

### Acceptance

- canonical forgetting <=2pp per op,
- composition forgetting <=2pp,
- Core unchanged,
- existing bank unchanged.

On pass:
- install exactly one candidate,
- bank +1,
- release temp to 0.

On fail:
- abort install,
- preserve fallback,
- bank unchanged.

---

## A1-B007X-007 — Fresh-runtime recurrence after compression

### Goal

Prove knowledge lives in persistent state.

### Work

1. Serialize Core + Bank.
2. Destroy temp/teacher objects.
3. Start fresh runtime.
4. Load persistent state only.
5. Re-present novel task with oracle primitive selection.

### Acceptance

- recurrence EM >=0.95,
- same primitive ID,
- adaptation steps 0,
- temp params 0,
- bank unchanged,
- no consolidation.

---

## A1-B007X-008 — Capacity-gap final audit

### Output

Create:

`docs/results/A1_B007X_DISCOVERY_COMPRESSION_RESULT.md`

### Required sections

1. B006 reinterpretation
2. B007 prerequisite
3. novelty proof
4. capacity ladder
5. compact direct-learning control
6. discovery-efficiency comparison
7. distillation retention
8. compression ratio
9. shadow safety
10. persistent-only recurrence
11. final verdict

### Allowed verdicts

#### Strong capacity-gap support
- robust large-temp discovery advantage,
- retention >=95%,
- parameter ratio <=0.25,
- recurrence passes.

#### Compressibility only
- compression/recurrence pass,
- compact direct learning equally effective.

#### No compression support
- large temp succeeds,
- compact candidate cannot retain function.

#### Inconclusive
- optimization/task calibration prevents clean inference.

### Rule

Audit only. Do not begin learned retrieval/router/novelty here.
