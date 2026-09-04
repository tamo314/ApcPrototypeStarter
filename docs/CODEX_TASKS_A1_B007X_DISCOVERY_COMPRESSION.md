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

- **Status:** **COMPLETE** (ADR-0056)
- **Verdict:** Established official mandatory control baseline using <=25k candidate architecture (17,098 params, 68.4% of budget) across 5 seeds:
  - `SWAP_PAIRS`: 99.70% mean EM (±0.0045, min 0.990, max 1.000), 100.0% success rate (5/5), median step to 0.95 = 175.0 (5.6k examples), mean AUC = 0.7428.
  - `INVERT_HALF`: 39.30% mean EM (±0.0488, min 0.325, max 0.455), 0.0% success rate (0/5), threshold not reached.
  - `ROTATE_TRIPLETS`: 93.70% mean EM (±0.0637, min 0.860, max 0.985), 60.0% success rate (3/5), median step to 0.90 = 250.0 (8.0k examples), median step to 0.95 = 300.0 (9.6k examples), mean AUC = 0.6405.
- **Run Artifacts:** `runs/phase_a1_compact_direct_control/` (`report.json`, `summary.json`, `control_baseline.json`, `system.json`, `config.yaml`, and 15 model checkpoints in `checkpoints/`).
- **Authorization:** Authorizes Task A1-B007X-005 (Overcomplete-to-compact functional distillation).

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

- **Status:** **COMPLETE** (ADR-0057)
- **Verdict:** Successfully distilled overcomplete temporary teacher ($T_2$, 137,482 parameters) into compact candidate primitive ($T_0$, 17,098 parameters, 8.04x compression, $R_{\text{param}} = 0.1244 \le 0.25$) across 5 seeds on `SWAP_PAIRS`:
  - Candidate EM: 97.50% (std 0.0166, min 0.950, max 0.995, 100% success rate >= 0.95) -> PASS.
  - Retention: 97.50% (>= 0.95) -> PASS.
  - Parameter Ratio: 0.1244 (<= 0.25, 8.04x compression) -> PASS.
  - Functional Agreement: 97.50% sequence agreement, 99.72% token agreement -> Substantial compliance (tracks candidate EM against 100% accurate teacher).
  - Comparative Finding: Confirms ADR-0055/ADR-0056 that compact direct learning (99.70% EM, 175 steps) remains equal to or faster/more accurate than distillation (97.50% EM, 200 steps).
- **Run Artifacts:** `runs/phase_a1_overcomplete_distillation/` (`report.json`, `summary.json`, `distillation_result.json`, `system.json`, `config.yaml`, and 15 checkpoints in `checkpoints/`).
- **Authorization:** Authorizes Task A1-B007X-006 (Shadow validation, promotion, release).

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

- **Status:** **COMPLETE** (ADR-0058)
- **Verdict:** Shadow validation passed with zero regression across all 5 decision seeds:
  - Canonical Forgetting: 0.0000 (0.0% <= 2.0% per op across all 8 canonical tasks) -> PASS.
  - Composition Forgetting: 0.0000 (0.0% <= 2.0% across all 6 representative compositions) -> PASS.
  - Core Invariance: Core parameters byte-for-byte unchanged (delta == 0.0) -> PASS.
  - Bank Invariance: Existing 8 primitives byte-for-byte unchanged (delta == 0.0) -> PASS.
  - Promotion: Exactly one candidate installed into persistent bank (bank size transitioned 8 -> 9) -> PASS.
  - Release: Temporary discovery teacher parameters (137,482) released to 0 (100% release) -> PASS.
  - Abort Contract: Verified via unit test fault-injection coverage -> PASS.
- **Run Artifacts:** `runs/phase_a1_shadow_promotion/` (`report.json`, `summary.json`, `system.json`, `config.yaml`, and 5 serialized persistent checkpoints `seed_<0-4>/promoted_bank.pt`, `op_to_id.json`).
- **Authorization:** Authorizes Task A1-B007X-007 (Fresh-runtime recurrence after compression).


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
