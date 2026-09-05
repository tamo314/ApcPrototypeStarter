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

