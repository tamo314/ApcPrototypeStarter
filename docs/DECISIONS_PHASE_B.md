# Phase B Decision Log

This file stores architecture/research decisions made during:

**Phase B — Semantic Task Inference & Open-World Extension**

## Numbering

ADR numbers are global across the repository.

Before adding the first Phase B ADR:

1. inspect `docs/DECISIONS.md`;
2. inspect the latest phase-specific decision log;
3. use the next unused `ADR-NNNN`;
4. never renumber historical ADRs.

## Required ADR fields

For each decision record:

- date;
- ADR number and title;
- status;
- task ID;
- decision;
- evidence / reason;
- consequences;
- affected active documents;
- whether downstream tasks are blocked;
- run artifact paths when evidence is empirical.

## Phase B decisions that require an ADR

At minimum, add an ADR when:

- a sealed family is retired because its result changed the design;
- a STOP GATE fails;
- an adequacy threshold changes;
- a search budget changes after gate measurement;
- a Task Inference modality requires an architecture change;
- a pretrained model is proposed (normally out of scope);
- a causal invariant must change;
- the Phase B final verdict is recorded.

Do not preallocate ADR numbers in this scaffold.

---

## ADR-0074: Explicit-TaskSpec Unseen-Family Lifecycle Gate Verification (STOP GATE B1)

**Date:** 2026-09-05  
**Status:** Accepted (Task B-C003 Complete, STOP GATE B1 PASSED)  
**Affects:** `src/apc/evaluation/unseen_family_lifecycle_benchmark.py`, `scripts/run_phase_b_unseen_family_lifecycle.py`, `tests/test_unseen_family_lifecycle.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`  
**Run Artifacts:** `runs/phase_b_unseen_family_lifecycle_gate/` (`report.json`, `summary.json`, `report.md`)

### Context
Phase A.2 verified autonomous controller policies (K/C/N/R decisions, bank scaling to N=128, and class-incremental router updates) over closed operation families with explicit canonical operation IDs.
Phase B investigates whether APC's reuse/composition/plasticity lifecycle generalizes beyond Phase A.2's closed operation universe to genuinely held-out, unseen operation families.
Task B-C003 is STOP GATE B1: evaluating whether the frozen Phase A.2 autonomous controller and compact-first lifecycle policy successfully generalize to sealed held-out families (`SEALED_FAMILIES`: `MAJORITY_THREE`, `NEIGHBOR_MAX`) under explicit model-visible `TaskSpec` across $\ge 5$ seeds without architectural changes or controller retraining.

### Acceptance Criteria & Measured Empirical Results
Across 5 deterministic seeds (0, 1, 2, 3, 4) on CUDA:

1. **Plastic Trigger Rate:** Target $\ge 95.0\%$ -> Measured **100.00%** (10/10 episodes) — **PASS**
2. **Mean Final Novel EM:** Target $\ge 95.0\%$ -> Measured **97.66%** — **PASS**
3. **Worst Seed Novel EM:** Target $\ge 90.0\%$ -> Measured **95.31%** (all 5 seeds: 95.31%, 100.00%, 97.66%, 95.31%, 100.00%) — **PASS**
4. **1:1 Bank Promotion:** Exactly 1:1 per learned novel capability -> Measured **10/10 promotions** (promoted IDs 10, 11) — **PASS**
5. **Workspace Capacity Release:** Parameter leaks $== 0$ -> Measured **0 leaks** — **PASS**
6. **Fresh-Runtime Recurrence:**
   - Recurrence EM $\ge 95.0\%$ -> Measured **97.19%** — **PASS**
   - Recurrence adaptation steps $== 0$ -> Measured **0** — **PASS**
   - Recurrence temporary parameters $== 0$ -> Measured **0** — **PASS**
   - Recurrence bank growth $== 0$ -> Measured **0** — **PASS**
7. **Legacy Regression:**
   - Max old-task EM degradation $\le 1.0\,\text{pp}$ -> Measured **0.00 pp** — **PASS**
   - Max old-routing top-1 drop $\le 1.0\,\text{pp}$ -> Measured **0.00 pp** — **PASS**
   - Legacy false plastic rate $\le 1.0\%$ -> Measured **0.00%** (0/45 episodes) — **PASS**

### Key Findings & Architecture Insights
1. **Zero Oracle Leakage & Autonomous Decision:** The frozen controller (trained on synthetic adequacy profiles in Phase A.2) correctly identified that both sealed operations (`MAJORITY_THREE`, `NEIGHBOR_MAX`) could not be satisfied by direct reuse (direct EM = 0.0) or depth-2 composition (comp EM = 0.0), triggering `PLASTIC_SEARCH` with confidence $> 0.99999$ across 100% of episodes based purely on support-set execution evidence.
2. **Compact Plasticity Generalization:** Without overcomplete fallback, the compact `CrossPositionPrimitive` model (1200 steps, batch size 64) learned the novel 3-element local operations, achieving $>97\%$ sequence EM and passing shadow validation with zero degradation on pre-existing tasks.
3. **Bounded Incremental Router Update Stability:** Updating the router with bounded replay (R2, 250 steps, learning rate 0.005) with deterministic per-stage seed initialization successfully integrated the new keys without causing catastrophic forgetting (0.00 pp drop in top-1 accuracy across all legacy operations).
4. **Fresh-Runtime Knowledge Persistence:** In completely re-instantiated runtimes with zero allocated workspace, previously consolidated novel primitives were autonomously identified and directly reused with $97.19\%$ mean EM and zero adaptation steps or bank growth.

### Consequences
- STOP GATE B1 is PASSED without qualification.
- Validates that APC's core lifecycle (Adequacy -> Plastic Search -> Compact Consolidation -> Shadow Validation -> Promotion -> Incremental Routing -> Direct Recurrence) generalizes to unseen operation families under explicit `TaskSpec`.
- Unblocks Task B-C004 (Hard-negative retrieval competition design and baseline dataset).

---

## ADR-0075: Hard-Negative Routing and Functional Safety (STOP GATE B2)

**Date:** 2026-09-05
**Status:** Accepted — FAILED STOP GATE B2 (Task B-C005 Complete)
**Affects:** `src/apc/evaluation/hard_negative_routing_benchmark.py`, `scripts/run_phase_b_hard_negative_safety_gate.py`, `configs/phase_b_hard_negative_safety_gate.yaml`, `tests/test_hard_negative_routing_benchmark.py`, `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_hard_negative_safety_gate/` (`config.yaml`, `metrics.jsonl`, `protocol.json`, `report.json`, `report.md`, `summary.json`, `system.json`)

### Decision

Record the frozen-router hard-negative result as a failed gate. Do not begin B-C006/B-C007 or any Task Inference task until candidate proposal and the known-episode false-plastic/adequacy path are isolated.

### Evidence / reason

The predeclared five-seed CUDA matrix covered 400 cells: four bank sizes, five hard-negative levels, and four parameterized target operations. At N=128, top-k inclusion was 1.000 at every level, but top-1 was 1.000 (L0), 1.000 (L1), 0.866 (L2), 0.662 (L3), and 0.504 (L4); L2–L4 fail the predeclared B2 thresholds. The functional verifier itself rejected every wrong candidate, including L4's logical same-primitive/wrong-argument candidate, and unselected physical primitive forward calls remained zero. Mean closed-loop EM was 0.959 (>=0.950), but mean false plastic on known episodes was 0.0375 (>0.02), so safety does not meet the full gate even though false functional acceptance was 0.000.

### Consequences

- The immediate bottleneck is candidate proposal/ranking under semantic and argument-near competition, not wrong-computation acceptance.
- The false-plastic outcome must be localized against support adequacy before changing any controller threshold or task-inference mechanism.
- B-C006 onward, especially B-C008–B-C011 Task Inference, are blocked by STOP GATE B2.
- No sealed-family status, controller threshold, router architecture, search budget, or plastic capacity was changed in response to this result.

---

## ADR-0076: Failure Isolation: Retrieval Ranking vs Argument Resolution vs Support Adequacy Variance

**Date:** 2026-09-06  
**Status:** Accepted (Task B-C005D Complete)  
**Affects:** `src/apc/evaluation/hard_negative_failure_isolation.py`, `scripts/run_phase_b_b2_failure_isolation.py`, `configs/phase_b_b2_failure_isolation.yaml`, `tests/test_hard_negative_failure_isolation.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md`  
**Run Artifacts:** `runs/phase_b_b2_failure_isolation/` (`config.yaml`, `metrics.jsonl`, `summary.json`, `system.json`, `failure_breakdown.json`, `margin_summary.json`, `support_variance.json`, `plots/`)

### Context
STOP GATE B2 (Task B-C005) returned FAIL due to two distinct symptoms:
1. Retrieval top-1 degradation under hard-negative levels L2 (0.866), L3 (0.662), and L4 (0.504) at N=128, despite top-5 inclusion remaining 1.000.
2. False plastic rate on known tasks of 3.75% (> 2.0% threshold), despite 0.0% wrong functional acceptance.

Task B-C005D performed mechanism-isolated diagnostic decomposition without modifying any router weights, architecture, thresholds, or policies.

### Core Empirical Findings & Attributions

1. **L4 Confusable Family Degradation is 100.0% Argument Resolution Failure:**
   - Decomposing L4 queries (5,120 queries across 400 cells) revealed:
     - `physical_primitive_top1`: **1.000** (100.0%)
     - `argument_accuracy`: **0.540** (54.0%)
     - `family_ranking_failure_fraction`: **0.0%** (0 / 2,353 failures)
     - `argument_resolution_failure_fraction`: **100.0%** (2,353 / 2,353 failures)
   - The physical primitive family was retrieved correctly in 100% of cases. The router failed strictly to distinguish the correct argument from an engineered same-family competitor (`virtual:wrong_argument`), yielding a mean score margin of $-0.0015$.
   - **Architectural Consequence:** Router keys should NOT be duplicated per argument. Instead, B-C005R1 should introduce factorized primitive-call scoring `score(PrimitiveCall) = score_family(z_task, key) + lambda * score_args(z_task, args)` or an explicit margin ranking objective.

2. **L2 vs L3 Retrieval Degradation Mechanisms:**
   - **L2 Near Neighbor:** Continuous margin collapse. Mean score margin shrank from $7.00$ (L0/L1) to $2.14$, with a $10.0\%$ fraction of margins $\le 0$ and $p05 = -0.39$.
   - **L3 Semantically Related:** Specific semantic collision. Bimodal score margin distribution (p05: $-10.31$, p95: $+10.70$) driven by engineered related-key competitors, with $29.6\%$ fraction of margins $\le 0$.
   - Top-5 inclusion remained $1.000$ across all levels, proving candidate recall is preserved.

3. **False Plastic is 100.0% Attributable to Finite-Support Estimator Variance:**
   - In 100% (15/15) of false plastic cells, the correct candidate was ranked #1 (`correct_rank = 1.0`).
   - The correct candidate achieved query EM of $95.3\%$ or $89.1\%$, but support EM on 32 examples fell just below threshold ($29/32 = 90.6\%$ or $30/32 = 93.8\% < 95.0\%$).
   - **Candidate Ordering:** Comparing Policy A (ranked-first), Policy B (evaluate all top-5), and Policy C (oracle correct candidate) yielded identical false plastic rates ($3.75\%$) and closed-loop EM ($95.92\%$). Candidate ordering contributes $0.0\%$ to false plastic.
   - **Support Size Variance:** Support scaling on disjoint development data verified that increasing support size without changing threshold reduces false plastic from $1.25\%$ ($K=16$) to $0.00\%$ ($K\ge 32$). The theoretical binomial reference curve for $p=0.99$ predicts a $4.07\%$ false-rejection rate at $K=32$, closely matching the observed $3.75\%$.

### Answers to Required B-C005D Questions
1. *Is L2 failure primarily margin/ranking failure?* **YES** (Continuous margin collapse, mean margin 2.14, 10.0% margin $\le 0$).
2. *Is L3 failure primarily margin/ranking failure?* **YES** (Specific semantic collision, 29.6% margin $\le 0$).
3. *For L4, what fraction is family routing vs argument resolution?* **0.0% family routing, 100.0% argument resolution**.
4. *What fraction of false plastic occurs despite correct candidate in top-5?* **100.0%** (15/15 cells, and in fact rank 1.0).
5. *What fraction occurs despite high query EM?* **66.7%** (10/15 cells query EM $\ge 90\%$, remainder $89.1\%$).
6. *Does false plastic decrease as support size increases without changing threshold?* **YES** (Drops to 0.00% at K=128).
7. *Is candidate ordering contributing materially?* **NO** (Identical 3.75% across Policy A, B, and C).

### Consequences
- Unblocks Task **B-C005R1 (Retrieval Ranking Repair)**.
- B-C005R1 targets hard-negative margin ranking and argument-compatibility scoring on development data; adequacy thresholds and statistical verifiers remain frozen during R1.
- Downstream tasks (B-C005R2, B-C005G, B-C006) remain blocked pending R1 and re-gating.

