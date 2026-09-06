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

---

## ADR-0077: Retrieval Ranking Repair and Factorized Primitive-Call Scoring

**Date:** 2026-09-06  
**Status:** Accepted (Task B-C005R1 Complete, Acceptance Criteria PASSED)  
**Affects:** `src/apc/primitives/routing_losses.py`, `src/apc/primitives/argument_scoring.py`, `src/apc/evaluation/retrieval_repair_benchmark.py`, `configs/phase_b_b2_retrieval_repair.yaml`, `scripts/run_phase_b_b2_retrieval_repair.py`, `tests/test_routing_ranking_loss.py`, `tests/test_argument_scoring.py`, `tests/test_retrieval_repair_benchmark.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md`  
**Run Artifacts:** `runs/phase_b_b2_retrieval_repair/` (`config.yaml`, `metrics.jsonl`, `summary.json`, `system.json`, `report.md`)

### Context
STOP GATE B2 (Task B-C005) failed retrieval ranking under hard negatives (L2: 0.866, L3: 0.662, L4: 0.504 at $N=128$). Diagnostic task B-C005D isolated L4 failure as 100.0% argument-resolution failure (physical primitive top-1 was 1.000, but router keys could not distinguish same-family wrong-argument competitors). L2/L3 degradation stemmed from continuous margin collapse and specific semantic collisions under frozen Phase A.2 classification keys.

Task B-C005R1 evaluated retrieval ranking repair on disjoint development partitions (`seeds [10, 11, 12, 13, 14]`), strictly isolating the original sealed evaluation partition (`seeds [0, 1, 2, 3, 4]`).

### Architectural Implementation
1. **Routing Margin Ranking Objective:**
   Implemented `CombinedRoutingLoss = L_ce + beta * L_rank` where $L_{\text{rank}} = \max(0, \text{margin} - s_+ + s_-)$, penalizing near-neighbor and semantically related competitors. During router training, `query_proj` is frozen (preserving Phase A.2's invariant score projection geometry) while candidate keys are optimized against balanced replay and synthetic development negatives.
2. **Factorized PrimitiveCall Scoring:**
   Implemented `ArgumentScorer` for parameterized operations (`SHIFT`, `COUNT`, `BIND`, `SELECT`), predicting argument compatibility scores:
   $$\text{score}(\text{PrimitiveCall}) = \text{score}_{\text{family}}(z_{\text{task}}, \text{key}) + \lambda \cdot \text{score}_{\text{args}}(z_{\text{task}}, \text{arguments})$$
   where $\text{score}_{\text{args}} = 2 \cdot (P(\text{args} \mid z_{\text{task}}) - 0.5) \in [-1.0, 1.0]$, giving positive bonuses for matched arguments and negative penalties for conflicting arguments, without duplicating persistent primitive keys per argument value.

### Acceptance Criteria & Measured Empirical Results (Condition R2 at N=128)
Across 5 development seeds (10, 11, 12, 13, 14) on CUDA:

1. **L0-L2 PrimitiveCall Top-1:** Target $\ge 0.98$ -> Measured **1.000** — **PASS**
2. **L3 PrimitiveCall Top-1:** Target $\ge 0.95$ -> Measured **1.000** — **PASS**
3. **L4 PrimitiveCall Top-1:** Target $\ge 0.90$ -> Measured **0.9742** (up from 0.420 in R0) — **PASS**
4. **Top-5 Inclusion:** Target $\ge 0.99$ across all levels -> Measured **1.000** — **PASS**
5. **L4 Physical Primitive Family Top-1:** Target $\ge 0.98$ -> Measured **1.000** — **PASS**
6. **L4 Argument Accuracy:** Target $\ge 0.95$ -> Measured **0.9742** — **PASS**
7. **Legacy Known-Task Regression:** Max drop $\le 1.0\,\text{pp}$ -> Measured **0.00 pp** (0.0% drop) — **PASS**
8. **Unselected Primitive Forward Calls:** Exactly $== 0$ -> Measured **0** — **PASS**
9. **Wrong Functional Acceptance:** Target $\le 1.0\%$ -> Measured **0.0%** — **PASS**

### Condition Comparison at N=128 (PrimitiveCall Top-1)
| Level | R0 (Frozen A.2) | R1 (Standard CE) | R2 (Ranking + ArgScorer) |
|---|:---:|:---:|:---:|
| L0_orthogonal | 1.000 | 1.000 | **1.000** |
| L1_random_score_space | 1.000 | 1.000 | **1.000** |
| L2_near_neighbor | 1.000 | 0.963 | **1.000** |
| L3_semantically_related | 0.988 | 1.000 | **1.000** |
| L4_confusable_family | 0.420 | 0.325 | **0.974** |

### Key Findings & Attribution
1. **Factorized Scoring Resolves L4 without Key Multiplicity:** R0 and R1 both fail on L4 (0.420 and 0.325) because primitive key similarity cannot differentiate argument variants. Condition R2 resolves L4 to 97.42% accuracy purely through argument compatibility scoring while maintaining a single resident primitive key per physical primitive family.
2. **Zero Catastrophic Forgetting:** With frozen `query_proj` and bounded replay, candidate key margin optimization produced 0.00 pp drop on legacy tasks and 0 unselected primitive forward calls.
3. **Partition Cleanliness:** Development repair was executed entirely on seeds [10-14] with zero access to the original sealed evaluation partition [0-4].

### Consequences
- Task **B-C005R1 is PASSED**.
- Unblocks Task **B-C005R2 (Functional Adequacy Estimator Repair)**.
- The repaired retrieval mechanism (CombinedRoutingLoss + ArgumentScorer with frozen query_proj) is frozen for B-C005R2.

---

## ADR-0078: Bounded Sequential Adequacy Verifier for Finite-Support Variance Repair

**Date:** 2026-09-06  
**Status:** Accepted (Task B-C005R2 Complete, Acceptance Criteria PASSED)  
**Affects:** `src/apc/meta/adequacy_verifier.py`, `src/apc/evaluation/adequacy_repair_benchmark.py`, `configs/phase_b_b2_adequacy_repair.yaml`, `scripts/run_phase_b_b2_adequacy_repair.py`, `tests/test_adequacy_verifier.py`, `tests/test_adequacy_repair_benchmark.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md`  
**Run Artifacts:** `runs/phase_b_b2_adequacy_repair/` (`report.md`, `summary.json`, `metrics.jsonl`, `config.yaml`, `system.json`)

### Context
STOP GATE B2 (Task B-C005) returned FAIL partially due to a 3.75% false-plastic rate on known episodes despite 0.0% wrong functional acceptance. Diagnostic task B-C005D isolated this failure as 100.0% attributable to finite-support estimator variance under fixed $K=32$ with a hard $\text{EM} \ge 0.95$ threshold (where stochastic binomial variation on high-performing candidates $p \approx 0.98$ produced occasional $30/32 = 0.938 < 0.95$, triggering false rejection and fallback to plastic search).

Task B-C005R1 resolved the retrieval-ranking bottleneck (ADR-0077). Task B-C005R2 freezes the repaired retrieval mechanism (Condition R2: CombinedRoutingLoss + ArgumentScorer with frozen `query_proj`) and repairs the functional adequacy verifier without altering the authoritative 0.95 adequacy threshold.

### Architectural Implementation
Implemented `SequentialAdequacyVerifier` in `src/apc/meta/adequacy_verifier.py`:
1. **Bounded Sequential Evidence Gathering:** Evaluates candidate execution on support batches of increasing size: initial support $N_{\text{init}}=32$, increment $\Delta N=32$, maximum support budget $N_{\text{max}}=128$.
2. **Statistically Justified Stopping Rules:**
   - **Early Accept (Clearly Adequate):** Empirical accuracy $\hat{p} = k/n \ge 0.95$. Bounded candidates matching or exceeding the target threshold are accepted immediately at $n=32$.
   - **Early Reject (Clearly Inadequate):** Wilson score confidence interval upper bound $U(k, n) < 0.95$ ($1 - \alpha = 0.95$). Inadequate distractors and competitors ($k/n \le 0.15$) are rejected immediately at $n=32$ ($U(k, 32) < 0.30$).
   - **Uncertain (Evidence Gathering):** $\hat{p} < 0.95 \le U(k, n)$ (e.g., $k=30/32$). Instead of premature plastic search, the verifier gathers an additional 32 support examples up to $N_{\text{max}}=128$.
   - **Max Budget Forced Decision:** If $n=N_{\text{max}}$, classify candidate as accepted iff $\hat{p} \ge 0.95$.
3. **Zero Query Leakage:** Query examples and targets are strictly inaccessible to the verifier.

### Acceptance Criteria & Measured Empirical Results (Policy D at N=128 across 5 dev seeds)
Across 5 development seeds (10, 11, 12, 13, 14) on CUDA:

1. **False Plastic Rate:** Target $\le 2.0\%$ -> Measured **0.00%** (0 / 100 episodes @ N=128; down from 5.00% in fixed-32) — **PASS**
2. **Wrong Functional Acceptance:** Target $\le 1.0\%$ -> Measured **0.00%** (0 / 400 competitor evaluations) — **PASS**
3. **Closed-Loop EM:** Target $\ge 95.0\%$ -> Measured **99.61%** (up from 94.92% in fixed-32) — **PASS**
4. **Mean Support Consumed:** Target $< 64.0$ -> Measured **32.29 examples** (96%+ candidates decided at $n=32$) — **PASS**
5. **Leak Audit Passed:** **True** (0 query targets used during verification) — **PASS**
6. **Zero Unselected Forward Calls:** **True** (0 unselected calls across all episodes) — **PASS**

### Comparative Policy Evaluation at N=128
| Policy | Closed-loop EM | False Plastic | False Acceptance | Mean Support | P95 Support | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Policy A (`fixed_32`) | 0.9492 | 0.0500 | 0.0000 | 32.0 | 32.0 | 0.39 |
| Policy B (`fixed_64`) | 0.9961 | 0.0000 | 0.0000 | 64.0 | 64.0 | 0.51 |
| Policy C (`fixed_128`) | 0.9492 | 0.0500 | 0.0000 | 128.0 | 128.0 | 0.74 |
| Policy D (`sequential`) | **0.9961** | **0.0000** | **0.0000** | **32.3** | **33.6** | **0.40** |

### Key Findings & Architecture Insights
1. **Elimination of Estimator Variance without Budget Inflation:** Policy D completely eliminates false plastic ($5.00\% \to 0.00\%$) and raises closed-loop EM to $99.61\%$, while consuming an average of only $32.29$ support examples (virtually identical to fixed-32's $32.0$, and with $0.40\,\text{ms}$ latency vs $0.39\,\text{ms}$).
2. **Preservation of Safety Priority:** Across 1,600 policy-cell evaluations, wrong functional acceptance remained strictly $0.0000$. Early rejection at $U(k, n) < 0.95$ safely discarded wrong candidates without ever expanding support consumption on distractors.
3. **Dual Repair Complete:** The retrieval ranking repair (R1) and adequacy estimator repair (R2) together resolve all identified failure mechanisms of STOP GATE B2 in isolated development partitions.

### Consequences
- Task **B-C005R2 is PASSED**.
- Unblocks Task **B-C005G (New Sealed Hard-Negative B2 Re-Gate)**.
- Freezes the complete repaired architecture (CombinedRoutingLoss + ArgumentScorer + SequentialAdequacyVerifier) for sealed re-gating.

---

## ADR-0079: New Sealed Hard-Negative B2 Re-Gate

**Date:** 2026-09-06
**Status:** Accepted — FAILED STOP GATE B2 (Task B-C005G Complete)
**Affects:** `src/apc/evaluation/hard_negative_repair_gate.py`, `configs/phase_b_b2_regate.yaml`, `scripts/run_phase_b_b2_regate.py`, `tests/test_hard_negative_repair_gate.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_HARD_NEGATIVE_REPAIR.md`, `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
**Run Artifacts:** `runs/phase_b_b2_regate/` (`protocol.json`, `config.yaml`, `metrics.jsonl`, `summary.json`, `system.json`, `report.md`)

### Context

ADR-0077 and ADR-0078 passed their development-only repair criteria using seeds
`[10, 11, 12, 13, 14]`. B-C005G froze the selected mechanism before evaluation:
`CombinedRoutingLoss` (margin `3.0`, beta `1.0`, frozen `query_proj`),
factorized `ArgumentScorer` (lambda `2.0`), and the sequential Wilson adequacy
verifier (threshold `0.95`, confidence `0.95`, support `32/32/128`, top-k `5`).

The runner designated `[20, 21, 22, 23, 24]` as a new sealed partition, disjoint
from both development data and the original B-C005 sealed seeds `[0, 1, 2, 3, 4]`.
It recorded protocol hash `05e703b63f323cefebb909eebdf176b2bb7088c404b240d96db40de34c9bd1af`
before the matrix began; post-seal configuration mutation is rejected.

### Sealed Results

The CUDA matrix completed all 400 required cells: five seeds × four bank sizes
(`16, 32, 64, 128`) × five levels × four parameterized targets.

At `N=128`:

1. L0-L2 full PrimitiveCall top-1 was `1.000` and top-k inclusion was `1.000` — PASS.
2. L3 full PrimitiveCall top-1 was `0.6555`, below `0.95` — **FAIL**. This is the
   first failed mechanism: semantically related candidate ranking remains
   insufficient despite perfect top-k inclusion.
3. L4 physical primitive-family top-1 was `1.000` — PASS, but argument accuracy
   and full PrimitiveCall top-1 were both `0.8883`, below `0.95` and `0.90` — **FAIL**.
4. Wrong functional acceptance was `0.000`, mean closed-loop EM was `0.9588`,
   unselected primitive calls were `0`, router/primitive mutation was absent, and
   evaluation-metadata leakage was `0` — PASS.
5. False plastic reached `1.000` in cells associated with sealed seed `24` and
   target `SHIFT`, exceeding `0.02` — **FAIL**. This is a functional-reuse failure
   after candidate verification, not a wrong-reuse safety failure.

### Consequences

- **B-C005G is FAILED.** ADR-0075 remains the historical original B-C005 failure;
  the re-gate does not reinterpret it as a pass.
- B-C006 and all dependent Task Inference work remain blocked.
- The sealed partition `[20, 21, 22, 23, 24]` must not be used for tuning. Any
  further repair cycle must first designate another disjoint sealed partition.
- The next investigation must isolate the L3 semantic-ranking failure before
  changing downstream controller or Task Inference mechanisms.

---

## ADR-0080: SHIFT Seed-24 Adequacy Audit — Primitive Inadequacy, Asymmetric-Rule Premature Acceptance, and a Benchmark-Generation Nondeterminism Bug

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005D2-005 Complete; diagnostic-only, no repair authorized)
**Affects:** `src/apc/evaluation/adequacy_reference_audit.py`, `scripts/run_phase_b_b2_adequacy_reference_audit.py`, `configs/phase_b_b2_adequacy_reference_audit.yaml`, `tests/test_adequacy_reference_audit.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_SECOND_DIAGNOSTIC.md`
**Run Artifacts:** `runs/phase_b_b2_second_diagnostic/` (`shift_seed24_adequacy_audit.json`, `reference_adequacy_summary.json`, `adequacy_reference_audit_config.yaml`, `adequacy_reference_audit_protocol.json`, `adequacy_reference_audit_system.json`, `adequacy_reference_audit_console.log`)

### Context

ADR-0079 (B-C005G) reported false plastic `1.000` in every sealed seed-24 / SHIFT cell (all five hard-negative levels) at bank sizes 16, 32, and 128, while bank size 64 accepted reuse. ADR-0076 had earlier attributed all false-plastic outcomes in the *original, unrelated* B-C005 sealed gate (seeds 0–4) to "100% finite-support estimator variance" acting on a genuinely adequate candidate. Task B-C005D2-005 tests whether that same attribution explains the new seed-24/SHIFT cells, using a >= 1024-example independent reference batch, the installed `SequentialAdequacyVerifier`'s full step-by-step trace, and a diagnostic-only symmetric counterfactual rule (Wilson lower bound >= 0.95 → accept, upper bound < 0.95 → reject, otherwise uncertain).

### Evidence

1. **Reference adequacy** (`n=1024` held-out examples per bank size, never used for support/query/training, executed through the same installed SHIFT primitive): the primitive at `model_seed=4` (`24 % 5`) measured `reference_EM` in `[0.9033, 0.9248]` across all four bank-size reconstructions, with the Wilson 95% CI upper bound never exceeding `0.940`. Zero of four cells were `reference_adequate`. This falsifies "finite-support variance acting on an adequate candidate" for this seed/operation: **TRUE_PRIMITIVE_INADEQUACY**.
2. **Reclassification** (D2-005.3) of the four sealed cells against this reference: two (bank 32, 64) were `FUNCTIONALLY_JUSTIFIED_PLASTIC` (the verifier correctly rejected a genuinely inadequate candidate); the remaining two (bank 16, 128) were **`UNSAFE_REUSE`** — the verifier *accepted* the same genuinely inadequate candidate after a single `n=32` support draw (`31/32` and `32/32` respectively) whose point estimate happened to clear `0.95`, despite a Wilson lower bound (`0.843`, `0.893`) far below threshold. Zero cells were `TRUE_FALSE_PLASTIC`.
3. In both `UNSAFE_REUSE` cells, the diagnostic-only symmetric rule returns `UNCERTAIN` rather than `ACCEPT` on the identical evidence (D2-005.4), directly implicating the installed rule's asymmetric early-accept condition (empirical EM >= threshold alone, with no confidence-interval confirmation) as the proximate mechanism: **SEQUENTIAL_RULE_BIAS**, specifically in the premature-accept direction. The rule's forced-reject-at-max-support behavior (observed for bank 32/64 at `n=96`–`128`) is a separate, deliberate, conservative tie-break and is not evidence of bias.
4. `verify_installed_rule_matches_spec` confirms `SequentialAdequacyVerifier` implements exactly its documented specification (early accept `EM >= threshold`; early reject Wilson `upper < threshold`; otherwise gather evidence; forced decision at `max_support`) across 10 synthetic `(successes, trials)` cases — the asymmetry is a property of the *specified* rule itself, not an implementation defect in the verifier.
5. **Unrelated to the SHIFT-specific finding, but discovered while reconstructing this cell:** repeated runs of the identical config and seeds produced *different* per-bank-size accept/reject outcomes across separate `python` process invocations, despite `deterministic_algorithms=True`. Root cause isolated to `generate_benchmark_examples` (`src/apc/evaluation/recurrence_benchmark.py`, duplicated in `consolidation_benchmark.py`), whose RNG seed includes `hash(operation) % 10000` — Python randomizes string hashing per process by default, and no run script in this repository pins `PYTHONHASHSEED`. Pinning `PYTHONHASHSEED=0` made two independent full reruns of this audit bit-for-bit identical, confirming the diagnosis. This affects every Phase B benchmark that calls `generate_benchmark_examples`, including the archived B-C005/B-C005D/R1/R2/B-C005G runs, whose exact per-seed examples were therefore never guaranteed reproducible from a fresh process.

### Consequences

- ADR-0076's attribution that false plastic was "100% finite-support estimator variance" is **retrospectively qualified**: it remains the correct attribution for the original B-C005 sealed cells it measured (seeds 0–4; not reinterpreted here), but does **not** generalize to the new sealed seed-24/SHIFT cells, where the installed primitive is genuinely inadequate (`TRUE_PRIMITIVE_INADEQUACY`) and where two of four bank-size draws instead exhibit the opposite failure mode (`SEQUENTIAL_RULE_BIAS`, premature-accept direction). ADR-0076 itself is not rewritten.
- A previously invisible safety gap is newly disclosed: the installed `false_functional_acceptance_rate` metric (`0.000` throughout B-C005G) only tracks acceptance of a *wrong competitor's* arguments and cannot detect acceptance of the *correct* family/arguments backed by a primitive that is itself below the 0.95 functional bar. B-C005G's "wrong functional acceptance = 0.000 — PASS" must not be read as ruling out this failure mode.
- Per AGENTS.md's forbidden-conclusions guidance, this ADR does **not** recommend lowering the adequacy threshold (the primitive is genuinely below it) and does **not** recommend enlarging the router (retrieval/ranking remained perfect top-1 at every level and bank size for this operation; the failure is in primitive execution accuracy and verifier acceptance policy).
- No repair is authorized by this ADR. `B-C005D2-006` must incorporate this evidence into its integrated causal-diagnosis table before any next-repair recommendation.
- The `generate_benchmark_examples` hash-seed nondeterminism (evidence item 5) is a separate, repository-wide reproducibility bug, disclosed for the user's attention. It is outside B-C005D2's diagnostic scope and touches infrastructure shared by many historical results, so it is **not** fixed by this task; recommended as a small, dedicated follow-up (e.g., replacing `hash(operation)` with a deterministic hash, or pinning `PYTHONHASHSEED` in run scripts).

---

## ADR-0081: Integrated Causal Diagnosis and Next-Repair Decision Gate (Second Diagnostic Phase Complete)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005D2-006 Complete; diagnostic-only, no repair authorized — STOP per `docs/CODEX_TASKS_PHASE_B_B2_SECOND_DIAGNOSTIC.md` Section 9)
**Affects:** `src/apc/evaluation/integrated_causal_diagnosis.py`, `scripts/run_phase_b_b2_integrated_causal_diagnosis.py`, `configs/phase_b_b2_integrated_causal_diagnosis.yaml`, `tests/test_integrated_causal_diagnosis.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`, `docs/CODEX_TASKS_PHASE_B_B2_SECOND_DIAGNOSTIC.md`
**Run Artifacts:** `runs/phase_b_b2_second_diagnostic/` (`final_causal_diagnosis.json`, `integrated_causal_diagnosis_config.yaml`, `integrated_causal_diagnosis_protocol.json`, `integrated_causal_diagnosis_system.json`)

### Context

`B-C005D2-001` through `B-C005D2-005` each diagnosed one mechanism of the `B-C005G` sealed re-gate failure in isolation. Task `B-C005D2-006` is the final task of the second diagnostic phase: combine that evidence into one causal findings table and exactly one recommended next research action (or `UNRESOLVED`), without implementing any repair. Every row of the table below is read directly back from the five prior tasks' own JSON artifacts (`summary.json`, `representation_stage_summary.json`, `semantic_relation_summary.json`, `l4_argument_breakdown.json`, `shift_seed24_adequacy_audit.json`) — none is recomputed or asserted from intuition, per the task doc's own D2-006.1 rule. A runtime guard (`_assert_no_forbidden_conclusions`) additionally checks the output never contains the two conclusions Section 2 of the task doc explicitly forbids.

### Findings table

| Mechanism | Evidence | Verdict | Confidence | Next action |
|---|---|---|---|---|
| L2 retrieval | D2-005's SHIFT/seed-24 reconstruction: `primitive_call_top1 = 1.0` at L0/L1/L2 for every bank size; matches ADR-0079's reported `L0-L2 top1 = 1.000` | NO_FAILURE | HIGH | none |
| L3 task representation | D2-002 `z_probe_accuracy = 1.0` at the failing `regate_sealed/R2_frozen_post_repair` cell | NO_FAILURE | HIGH | none |
| L3 query projection | D2-002 `q_probe_accuracy = 1.0` at the same cell; `query_proj` is frozen and identical across R0/R1/R2 | NO_FAILURE | HIGH | none |
| L3 key/scoring | D2-002 per-relation verdicts at the focus cell: `COUNT->BIND` and `BIND->COUNT` = `KEY_SCORING_BOTTLENECK` (`control_a` top1 0.40/0.628 vs `z`/`q` probes both 1.0); `SHIFT->CYCLE_FOUR` = `NO_FAILURE`; `SELECT->BIND` = `UNRESOLVED` (shuffled-control artifact) | KEY_SCORING_BOTTLENECK (scoped to `COUNT<->BIND`) | MEDIUM | scope any future key-scoring repair to `COUNT->BIND`/`BIND->COUNT` only |
| L3 relation split | D2-003: `cross_relation_spread_at_focus_cell = 0.878`; difficulty-matched `matched_gap = 0.366` stays close to `raw_gap = 0.370` | SEMANTIC_RELATION_HOLDOUT_REQUIRED | HIGH | define development/validation/sealed relation sets before any next repair training |
| L4 family routing | D2-004: `family_top1 = 1.0` for SHIFT/SELECT/COUNT/BIND at the focus cell | NO_FAILURE | HIGH | none |
| L4 argument resolution | D2-004 `failure_classification`: SHIFT/COUNT `NO_FAILURE`; BIND `ARGUMENT_SCORER_GENERALIZATION_FAILURE`; SELECT `ARGUMENT_ENCODING_FAILURE` | MIXED | HIGH | scope any future argument-scorer repair to `SELECT`/`BIND` only |
| adequacy estimator | D2-005: `implementation_matches_spec = true` (10/10 synthetic cases), but `SEQUENTIAL_RULE_BIAS` present in `overall_classification.labels` (2/4 `UNSAFE_REUSE` cells) | SEQUENTIAL_RULE_BIAS | HIGH | confirm asymmetric early-accept before deploying any confidence-based verifier change (Option E) |
| installed SHIFT adequacy | D2-005: `reference_EM` in `[0.9033, 0.9248]` across all 4 bank sizes; Wilson upper bound never exceeds `0.940` | TRUE_PRIMITIVE_INADEQUACY | HIGH | primitive functional-generalization repair for this seed/operation, not the controller (Option F) |
| false-plastic metric | D2-005: `n_true_false_plastic = 0`, `n_unsafe_reuse = 2` — the metric tracks only plastic-when-should-reuse and has no signal for reuse-when-should-go-plastic | METRIC_MISCLASSIFICATION | HIGH | adequacy metric/protocol repair (Option E) |

D2-001's own `representativeness_verdict` (`NON_REPRESENTATIVE`, flag `DEVELOPMENT_DIFFICULTY_MISMATCH`) is reproduced in the artifact for completeness but is not itself a table row, since it describes the development/sealed split rather than one runtime mechanism; its consequence is folded into the "L3 relation split" row and the recommendation below.

### Recommendation

Evaluating D2-006.2's six allowed options against the table above: **A** (query-projection repair) and **B** (task-representation repair) are not evidence-supported (both `z_task` and `query_proj` retain full separating information everywhere); **C** (semantic-relation holdout redesign), **D** (argument-scorer repair, scoped to SELECT/BIND), **E** (adequacy metric/protocol repair), and **F** (primitive functional-generalization repair for the installed SHIFT candidate) are all evidence-supported.

**Primary recommended next action: Option C — semantic-relation holdout redesign.** This is recommended as the single gating action, not merely one option among equals, because it precedes every other supported option: D2-003 found genuine model-generalization failure even after matching development and sealed examples on relation and geometric difficulty (`MODEL_GENERALIZATION_FAILURE_AFTER_MATCHING`), meaning any future repair — key-scoring for `COUNT<->BIND`, argument-scorer changes for SELECT/BIND, or SHIFT primitive retraining — trained under the current seed-only development/sealed split risks reproducing the exact develops-fine/fails-sealed pattern this entire D2 phase exists to diagnose. Options D, E, and F remain evidence-supported and are recorded in `final_causal_diagnosis.json` as pending actions, but are not authorized by this ADR.

Per D2-006.3, this diagnosis does not conclude the router needs to be larger (the key-scoring bottleneck is scoped to one specific relation pair, not a capacity claim) and does not conclude the adequacy threshold should be lower (the installed SHIFT primitive is genuinely below it).

### Consequences

- The second diagnostic phase (`B-C005D2-001` through `B-C005D2-006`) is complete. Per the task doc's Section 9 STOP condition, no repair, threshold change, `B-C006`, or Task Inference work is authorized by this ADR or any prior D2 task. `B-C006` and all dependent Task Inference work remain blocked pending an explicit user instruction naming the next repair task.
- All historical ADRs (0075–0080) remain in force; none is reinterpreted or overwritten by this diagnosis.
- Any future repair task must first define development/validation/sealed relation sets (Option C) before training, per this ADR's recommendation; once that protocol change is in place, Options D, E, and F become candidate follow-on repair tasks, each scoped to the specific operations/mechanisms this diagnosis identified rather than applied uniformly.

---

## ADR-0082: Reproducible Benchmark Generation and Artifact Inventory (Task B-C005R3-001)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-001 complete; G0 PASS)
**Affects:** `src/apc/utils/seed_derivation.py` (new), `src/apc/evaluation/consolidation_benchmark.py`, `src/apc/evaluation/recurrence_benchmark.py`, `src/apc/evaluation/post_d2_repair_benchmark.py` (new), `scripts/run_phase_b_b2_post_d2_repair.py` (new), `configs/phase_b_b2_post_d2_reproducibility.yaml` (new), `tests/test_seed_derivation.py` (new), `tests/test_post_d2_repair_benchmark.py` (new), `AGENTS.md`, `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`, `docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_001_reproducibility/` (`generator_audit.json`, `artifact_inventory.json`, `subprocess_comparison.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

ADR-0080 evidence item 5 traced the B-C005D2-005 cross-process nondeterminism to `generate_benchmark_examples`'s RNG seed, which included `hash(operation) % 10000` in two literal-duplicate locations (`src/apc/evaluation/recurrence_benchmark.py` and `src/apc/evaluation/consolidation_benchmark.py`). Python randomizes string hashing per process by default, so identical config/seed produced different examples from a fresh process unless `PYTHONHASHSEED` was pinned. ADR-0081 designated semantic-relation holdout redesign (Option C, `B-C005R3-002`) as the first research-priority repair, contingent on this reproducibility fix landing first (`docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md` S1). This ADR records Task `B-C005R3-001`, the first and (per explicit user instruction) only task executed in this session.

### Evidence

1. **Fix.** Both `hash(operation) % 10000` sites are replaced by `apc.utils.seed_derivation.derive_seed`, which hashes a canonical JSON payload (`generator_version`, `master_seed`, `stream_namespace`, `task_semantics_version`, `task_key`, `sample_index`) with SHA256 and takes the first 63 bits. Each example's seed is now a pure function of its own index rather than of a single continuing RNG stream, which is also what makes the fix satisfy order/resume/worker-count invariance, not only cross-process invariance. `consolidation_benchmark.py` keeps the one canonical implementation; `recurrence_benchmark.py` now imports it instead of defining its own copy, closing the duplication ADR-0080 flagged. All ~25 other call sites (`adequacy_reference_audit.py`, `hard_negative_routing_benchmark.py`, `bank_scaling_benchmark.py`, etc.) pass no `generator_version` argument and therefore receive the new default (`v2_sha256_indexed`) automatically, with zero code changes required at those call sites.
2. **G0 tests (`tests/test_seed_derivation.py`, 22 cases; `tests/test_post_d2_repair_benchmark.py`, 6 cases; all CPU-only, all pass).** Confirmed: identical output across 4 subprocess `PYTHONHASHSEED` variants (unset, `0`, `1`, `123`) for the same config; forward/reverse cell order, single-resume, and worker-count-partition invariance (chunk sizes 1/3/4/12 over `n=12` all reassemble the identical sequence); `recurrence_benchmark.generate_benchmark_examples is consolidation_benchmark.generate_benchmark_examples` (literally the same function object); namespace separation between `train`/`test` splits; a diagnostic-only `v1_legacy_hash` path that reproduces the exact pre-fix formula (for pinned-`PYTHONHASHSEED` reconstruction only) and is never the default; and, via a `builtins.hash` tripwire monkeypatch, that the default (v2) generation path never reaches Python's builtin `hash()` at all, while the legacy path does.
3. **Runtime self-check (`generator_audit.json`, produced by `scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-001`):** confirms `hash(operation) % 10000` is no longer present as a literal in either fixed file, and confirms via the same tripwire technique that the default path does not call `hash()`. A third, unrelated occurrence of the same pattern was found at `src/apc/evaluation/discovery_capacity_harness.py:170` (`generate_novel_examples`, Phase A1 A1-B007X discovery-capacity infrastructure) while tracing the call graph; it is outside R3-001's declared two-location scope, is not imported by any other module, and is left unmodified and disclosed for a possible separately scoped follow-up.
4. **Artifact inventory (`artifact_inventory.json`).** Ledgers existence/SHA256/availability for the shared base bank checkpoint (`runs/phase_a2_bank_scaling_benchmark/seed_<n>/primitive_bank_16.pt`) and for each of B-C005/B-C005D/B-C005R1/B-C005R2/B-C005G/B-C005D2's declared run-directory files, read-only (verified by test: file mtimes under the inspected directories are provably unchanged by building the inventory). Findings: the base checkpoint is `AVAILABLE` with a recorded SHA256 for seeds `{0,1,2,3,4,10,11,12,13,14}`; it was **never independently persisted** for the regate-sealed seeds `{20,21,22,23,24}` used by B-C005G/B-C005D2 (every regate/diagnostic run reconstructs those on the fly), and bank sizes `{32,64,128}` were likewise never persisted (only size 16 is ever written to disk). None of B-C005/D/R1/R2/G/D2's trained router/candidate/verifier state was ever separately serialized; only JSON summary/report/protocol artifacts are retained. No missing artifact was regenerated by this task, and this ADR does not claim any of the unavailable checkpoints as recoverable or as equivalent to a fresh reconstruction (`design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md` S2.3's `EXACT_REPLAY_UNAVAILABLE`/`V2_RECONSTRUCTION` distinction applies to any future attempt to reconstruct these).
5. **Docs pack introduced/wired.** The already-present (untracked) B2 post-D2 repair pack (`docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md`, `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`, `docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md`, `docs/AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md`, `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`, `docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md`) is now referenced from root `AGENTS.md`'s Active-research-phase section, and both `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md` and `docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md` now carry pointer notes to it at their `B-C006`/STOP GATE B2 sections. No prior FAIL text, ADR, or run artifact in either document was rewritten.

### Consequences

- No model weights, router, ArgumentScorer, verifier, controller, primitive, or adequacy threshold was changed by this task; `PYTHONHASHSEED` was not pinned as a substitute fix.
- **G0: INFRASTRUCTURE_OR_PROTOCOL_PASS.** This authorizes `B-C005R3-002` (semantic-relation split, Option C) to proceed next, per ADR-0081's recommendation and `docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md` S3 -- but per explicit user instruction, only `B-C005R3-001` was executed in this session, and `B-C005R3-002` onward, `B-C006`, and Task Inference all remain blocked pending a separate explicit user instruction.
- Historical B-C005/D/R1/R2/G/D2 JSON results are unaffected and unmodified; none is reinterpreted as more or less reproducible than ADR-0080 already established. Any future re-derivation of those results under the v2 generator must be labeled `V2_RECONSTRUCTION`, never treated as replaying the original run.
- The discovery-capacity-harness occurrence of the same bug pattern (evidence item 3) is disclosed but not fixed; it is available as a candidate for a future, separately scoped task.

## ADR-0083: Semantic-Relation Split & Exposure Protocol -- Real Registry Is Insufficient for a Disjoint L3 Transfer Holdout, and the Existing R2 Recipe Structurally Caps L3 Holdout at MINING_HOLDOUT_ONLY (Task B-C005R3-002)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-002 complete; G1 = `PROTOCOL_INSUFFICIENT_RELATIONS` for the `repair_relation_transfer` suite; `targeted_repair_regression` and the seed axis are unaffected)
**Affects:** `src/apc/evaluation/relation_split_protocol.py` (new), `scripts/run_phase_b_b2_post_d2_repair.py`, `configs/phase_b_b2_post_d2_relation_split.yaml` (new), `tests/test_relation_split_protocol.py` (new), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_002_relation_split/` (`relation_catalog.json`, `relation_split.json`, `exposure_manifest.json`, `development_representativeness.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

ADR-0081 designated semantic-relation holdout redesign (Option C) as the first research-priority repair, contingent on ADR-0082/B-C005R3-001's reproducibility fix. This task (`B-C005R3-002`) was required to enumerate the real relation catalogue, define DEV/VALIDATION/SEALED_V2 relation-group and seed membership disjointly, audit training exposure (CE denominator/replay/mining), confirm a development fixture genuinely contains the already-diagnosed failures, and pre-register sealed-partition membership/quotas without measuring any sealed output (`docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`, `docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md` S4/G1). Per that same design document's own explicit escape hatch, an insufficient real relation registry must produce `PROTOCOL_INSUFFICIENT_RELATIONS` rather than a loosened bar.

### Evidence

1. **Relation catalogue is small.** `_RELATED_OPERATION` (`hard_negative_routing_benchmark.py`) defines exactly 4 directed L3 units, which collapse to **3 non-alias unordered-family-pair groups**: `SHIFT-CYCLE_FOUR`, `SELECT-BIND`, `COUNT-BIND` (the last absorbing both `COUNT->BIND` and `BIND->COUNT`, confirmed programmatically, not hand-merged). `ArgumentScorer` (`argument_scoring.py`) gives 4 structurally independent L4 argument-variant groups (one dedicated `nn.Linear` head + optimizer per parameterized operation).
2. **BIND-key coupling merges two of the three L3 groups.** Because `SELECT-BIND` and `COUNT-BIND` share the physical `BIND` primitive's router key, a repair that retrains `COUNT-BIND`'s key (R3-006's declared scope) necessarily perturbs `SELECT-BIND`'s key geometry too, even though `SELECT`'s own data is untouched. This leaves only **2 independent L3 holdout components** system-wide (`{SHIFT,CYCLE_FOUR}` and `{BIND,COUNT,SELECT}` merged), for **6 independent units total** (2 L3 + 4 L4).
3. **Assigning already-diagnosed failures to DEV exhausts the sufficiency margin.** Per design-doc S5 step 5, units with a real diagnosed failure (read from the already-committed `runs/phase_b_b2_second_diagnostic/final_causal_diagnosis.json` and `l4_argument_breakdown.json`, never re-derived or guessed) go to `development`/`targeted_repair_regression`: the merged `{BIND,COUNT,SELECT}` L3 component (COUNT->BIND and BIND->COUNT both `KEY_SCORING_BOTTLENECK`; SELECT->BIND `UNRESOLVED`), plus L4 `SELECT` (`ARGUMENT_ENCODING_FAILURE`) and L4 `BIND` (`ARGUMENT_SCORER_GENERALIZATION_FAILURE`) -- 3 units. That leaves exactly 3 clean (`NO_FAILURE`) units (`SHIFT-CYCLE_FOUR`, L4 `SHIFT`, L4 `COUNT`) to split between `validation` and `sealed_v2`, one short of the design contract's required >=2 each. **G1 result: `PROTOCOL_INSUFFICIENT_RELATIONS` for `repair_relation_transfer`.** `targeted_repair_regression`'s own 3 DEV units satisfy its (non-novelty) requirement independently and are not blocked.
4. **The existing (unmodified) R2 repair-training recipe cannot achieve `STRICT_HOLDOUT` for L3 regardless of group count.** Reading `train_repaired_router_and_scorer` (`retrieval_repair_benchmark.py`): `key_params` spans **every** primitive in `candidate_list` (the full resident bank, not just the ops with training examples) under one `AdamW` optimizer with `requires_grad=True`; every step, `keys = router._stacked_keys(candidate_list)` is rebuilt fresh and the loss backprops through the softmax denominator over **every column** of `logits = query @ keys.T`. A held-out operation's key therefore receives a nonzero gradient every step purely from sitting in the candidate list, whether or not its own operation is ever sampled as a positive target that step. This was **confirmed empirically**, not just read from source: a tiny throwaway (`bank_size=16`, non-persisted, `deepcopy`d) reconstruction excluded `SHIFT` from the positive training examples and measured its router key changing anyway (`exposure_manifest.json:empirical_probe`, `router_key_changed_despite_exclusion: true`). Per design-doc S6, this full-class-CE exposure is exactly `MINING_HOLDOUT_ONLY`, which "must not be used for the primary relation-holdout PASS" -- so even a future repair task correctly excluding a held-out op's own examples can only reach `MINING_HOLDOUT_ONLY` for the router-key axis under the current, unmodified training code; achieving `STRICT_HOLDOUT` would require changing `train_repaired_router_and_scorer`'s `candidate_list` construction itself, which is out of this task's scope. `ArgumentScorer`'s per-op independent heads/optimizers, by contrast, genuinely support `STRICT_HOLDOUT` at L4 as long as a future task does not pass a held-out op's examples into `examples_by_op`.
5. **The base/parent checkpoint's historical exposure is knowable, not `UNKNOWN`.** `_build_frozen_base_system`'s `R0_FULL_RETRAIN` calibration trains all 16 bank operations as positive CE targets every time; this is a code fact (`historical_exposure_classification`), so it is reported as `CONFIRMED_ALL_GROUPS_NO_HOLDOUT` rather than falling back to `UNKNOWN` (a distinct `UNKNOWN` path exists in the same helper and is exercised by a unit test for a genuinely indeterminate-provenance case).
6. **Development fixture profiling confirms real failures are present, not cherry-picked.** A small-scale (`bank_size=16`, disclosed as non-comparable to D2's sealed-gate `bank_size=128` numbers) profile of the development partition (seeds 10-14, pre-repair, L3) reports the **full** per-relation outcome distribution: `COUNT->BIND` and `BIND->COUNT` both show real incorrect examples (2/80 each) alongside the correct ones; `SHIFT->CYCLE_FOUR` and `SELECT->BIND` are at ceiling at this scale, consistent with D2-002/003's own findings.
7. **New disjoint seed axis registered.** `validation` (15-19) and `sealed_v2` (30-34) are new, never-before-used seed ranges (confirmed against R3-001's `artifact_inventory.json`), each using `model_seed = seed` directly (never `seed % 5`) so every registered seed is its own independently initialized core -- unlike `regate_sealed`'s deliberate `seed % 5` reuse of the original 0-4 checkpoints, which must not be miscounted as independent models (design-doc S3). The original `original_sealed` (0-4) and `regate_sealed` (20-24) partitions are recorded as `RETIRED_ALREADY_VIEWED` and are not eligible to be reused as the new sealed partition (both already measured by B-C005/B-C005G/B-C005D2). `sealed_v2`'s membership is registered only; no model output for seeds 30-34 is measured by this task.
8. **Tests.** `tests/test_relation_split_protocol.py`, 19 cases, all pass, CPU-only (the one real reconstruction/training call, `_probe_key_exposure_empirically`, runs only via the milestone script, matching this repo's existing D2-audit precedent of keeping pytest GPU-free). Full repo suite: 1655 passed, 2 failed (24m54s) -- both failures are the same pre-existing `test_adequacy_verifier.py` Wilson/Clopper-Pearson float-equality edge case already documented in the D2-002 through D2-006 sessions, unrelated to this task. `ruff check .` and `mypy src/apc` both clean (122 source files).

### Consequences

- No router, ArgumentScorer, verifier, controller, or primitive weight was changed; the one real training call made by this task trains a throwaway, non-persisted `deepcopy`d router purely to empirically confirm the exposure finding above, and writes back to no checkpoint.
- **`repair_relation_transfer` claims for L3 remain blocked** on two independent grounds: (a) the real registry cannot supply >=2 non-alias, non-coupled clean groups to both `validation` and `sealed_v2` simultaneously; (b) even where group count were sufficient, the existing R2 recipe's full-class CE denominator caps any L3 router-key holdout at `MINING_HOLDOUT_ONLY`, which this project's own design contract disqualifies from a primary PASS. Any future `B-C005R3-0xx` task claiming L3 unseen-relation transfer generalization must either supply a materially richer relation registry, change `train_repaired_router_and_scorer`'s candidate-list scoping (a repair-training-code change, not authorized by this task), or explicitly limit its claim to `targeted_repair_regression`/L4.
- **`targeted_repair_regression` is NOT blocked**: R3-006 (COUNT<->BIND key/scoring), R3-007 (SELECT argument encoding), and R3-008 (BIND argument-scorer) may proceed using the `development_units` membership registered in `relation_split.json`, independent of this ADR's G1 result.
- `sealed_v2` (seeds 30-34) membership and per-relation quotas are registered now; per design-doc S8 and this task's own scope, no model output for these seeds is measured until R3-011 seals this membership and R3-012 runs the sealed gate.
- Historical B-C005/D/R1/R2/G/D2 results are unaffected; this ADR adds a new, independent finding and does not reinterpret any prior FAIL.

## ADR-0084: Functional Metrics v2 & Statistical Contract -- Frozen Adequacy Vocabulary, Offline-Only (Task B-C005R3-003)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-003 complete; G2 = `INFRASTRUCTURE_OR_PROTOCOL_PASS`)
**Affects:** `src/apc/evaluation/functional_metrics_v2.py` (new), `configs/phase_b_b2_post_d2_functional_metrics_v2.yaml` (new), `tests/test_functional_metrics_v2.py` (new), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_003_functional_metrics_v2/` (`metrics_schema_v2.json`, `statistical_contract.json`, `budget_feasibility.json`, `metric_fixture_results.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

ADR-0080 found that a correct-identity SHIFT candidate with independent reference EM of only 90.33-92.48% was accepted by point-estimate premature acceptance, and that the legacy wrong-candidate-acceptance metric could not detect this failure mode at all. `docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md` (Option E) requires that this vocabulary -- reference adequacy, candidate verdict, and action envelope -- be frozen as a type/schema contract *before* any change to the runtime verifier (`B-C005R3-005`), so that later runtime wiring is checked against a spec fixed in advance rather than shaped after seeing results. This task (`B-C005R3-003`) implements that frozen vocabulary and its supporting offline statistics as pure, CPU-only code; it does not touch the runtime verifier.

### Evidence

1. **Schema/envelope (design doc S2, S6).** `ReferenceAdequacyState`, `CandidateVerdict`, `SearchStatus`, `ControllerAction`, `ExecutionStatus`, `LibraryScopeStatus`, `SuiteKind`, and a frozen `VerifiedDecisionEnvelope` dataclass whose `__post_init__` raises (never silently constructs) on an unsafe combination: `EXECUTED` without `candidate_verdict == ACCEPT`; `EXECUTED` with a `controller_action` outside `{DIRECT_REUSE, COMPOSE}`; `NEEDS_MORE_EVIDENCE` carrying a non-`None` `controller_action` (no temporary workspace while still `UNCERTAIN`) or without `search_status == BUDGET_EXHAUSTED`; `NO_VERIFIED_SOLUTION` coexisting with `ACCEPT`. `classify_library_scope` never returns `INADEQUATE_WITHIN_DECLARED_SCOPE` unless every recipe in `H` was actually evaluated (a partially-evaluated all-REJECT scope stays `UNRESOLVED_WITHIN_SCOPE`); any single `ACCEPT` dominates immediately.
2. **Finite-look exact-bound statistical contract (design doc S4), frozen but not connected.** `tau=0.95`, `max_candidates=5`, `looks=(32,64,128,256,512)`, `alpha_accept_episode=alpha_reject_episode=0.01` split evenly per candidate-look (`0.0004` each). `one_sided_exact_bounds` computes the one-sided Clopper-Pearson-family limits via `scipy.stats.beta.ppf` (raises rather than substituting a normal/Wilson approximation under the "exact" name if scipy were absent). `verify_candidate_finite_look` walks the fixed, pre-ordered look schedule and -- unlike the legacy `SequentialAdequacyVerifier.verify_candidate_sequentially` (`apc.meta.adequacy_verifier`), which forces a decision at the final look -- returns `UNCERTAIN` at max support if the bounds never resolved, verified by a test with a constructed cumulative-successes sequence that stays in the `[0.9185, 0.9811]` band at every look through 512. `build_budget_feasibility` reproduces design-doc S4.5's table via this same real Beta-quantile implementation (not the closed form alone): full-success lower bounds `0.783095/0.884926/0.940705/0.969900/0.984835` at looks `32/64/128/256/512`, with acceptance first possible at look 256 (`min_successes_for_accept=254`) -- cross-checked against the design doc's own published numbers by test.
3. **Offline reference adequacy (design doc S5).** `reference_adequacy_state` returns `REF_ADEQUATE`/`REF_INADEQUATE`/`REF_UNRESOLVED` from an independent per-call `alpha_ref_episode` allocation, or `None` ("not estimable") when `trials == 0` -- never a forced binary at `trials == 0` or in the unresolved mid-region (test fixture `125/128` successes, `lower≈0.9169 < 0.95 <= upper≈0.9973`, stays `REF_UNRESOLVED`).
4. **Metric aggregator (design doc S7).** Every metric in the design doc's table is implemented with an explicit numerator/denominator and `None` (never a manufactured `0.0`) when the denominator is empty: `wrong_call_accept_rate` (excludes `functionally_equivalent` aliases from the wrong-call pool, so a same-output alias is never pooled with genuinely wrong calls), `inadequate_call_accept_rate`, `same_identity_unsafe_accept_rate`, `unsafe_reuse_episode_rate`, `avoidable_plastic_rate`, `adequate_solution_nonreuse_rate`, `uncertain_candidate_rate`, `reference_unresolved_rate` (excludes not-estimable `None` records from its own denominator), `unconditional_query_EM`/`selective_query_EM`/`execution_coverage` (abstention counts against `unconditional_query_EM` and is excluded from `selective_query_EM`'s denominator), `legacy_known_task_plastic_rate`, and a separate raw count `accepted_ref_unresolved_count` (never folded into a rate's denominator). All candidate/episode/query aggregates report `overall`/`by_operation`/`by_model_seed` breakdowns, and candidate metrics add `by_relation`.
5. **Nominal vs. degraded-candidate-safety-stress separation (task doc item 6).** `partition_by_suite` is a pure lookup against a pre-declared `suite_by_episode` tag with no branch on `reference_state`/`verdict`/outcome, so a record cannot be silently dropped from `NOMINAL` because its candidate turned out inadequate (verified by a test asserting a `REF_INADEQUATE`-tagged nominal record survives the partition).
6. **Explicitly not runtime-connected.** The module imports none of `apc.meta.adequacy_verifier`'s `SequentialAdequacyVerifier`, no router/scorer/primitive weights, and no training loop; no model or checkpoint is loaded and no GPU is required. `metrics_schema_v2.json`'s `runtime_connection_status` field states `NOT_CONNECTED` explicitly, and `statistical_contract.json` states `connected_to_runtime: false` -- `B-C005R3-005` performs that wiring against this exact frozen contract.
7. **Tests.** `tests/test_functional_metrics_v2.py`, 50 cases, all pass, CPU-only (re-verified in this session). `metric_fixture_results.json` runs and passes all 7 required scenarios from the task doc's "必須テスト" list (empty denominator to null, reference-unresolved stays unresolved, abstention counts as incorrect for unconditional EM, same-identity+`REF_INADEQUATE` acceptance flagged unsafe, functionally-equivalent wrong-ID excluded from the wrong-call pool, an unevaluated recipe blocks the `INADEQUATE_WITHIN_DECLARED_SCOPE` claim, nominal suite not filtered by outcome). `ruff check` and `mypy` both clean on the touched files (re-verified in this session). The CLI dispatcher (`scripts/run_phase_b_b2_post_d2_repair.py --task B-C005R3-003`) was re-run end-to-end in this session into a fresh output directory and reproduced the identical `G2: INFRASTRUCTURE_OR_PROTOCOL_PASS` result.

### Consequences

- No router, ArgumentScorer, verifier, controller, or primitive weight was changed; `apc.meta.adequacy_verifier.py` is untouched and the legacy `SequentialAdequacyVerifier` remains the only runtime-connected verifier until `B-C005R3-005`.
- **G2: INFRASTRUCTURE_OR_PROTOCOL_PASS.** This freezes the schema and statistical contract that `B-C005R3-004` (paired frozen-baseline comparison, offline only) and `B-C005R3-005` (actual runtime wiring) must use verbatim; neither task may redefine these enums, the `tau`/`looks`/`alpha` values, or the metric numerator/denominator definitions without a new ADR.
- This is a spec/offline-statistics task only: it does not itself demonstrate any change in runtime verifier safety or acceptance behavior. That claim is reserved for `B-C005R3-005`'s own gate (G3).
- Historical B-C005/D/R1/R2/G/D2 results and the legacy asymmetric-rule verifier are unaffected and unmodified.

## ADR-0085: Paired Frozen Baseline & Comparison Repair -- Known D2 Routing/Argument Failures Do Not Reproduce on `development` Seeds; SHIFT Reference-Adequacy Is Unverifiable There for a Structural Reason (Task B-C005R3-004)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-004 complete; no Gate -- `COMPARISON_ESTABLISHED`, performance PASS not required)
**Affects:** `src/apc/evaluation/paired_baseline_repair.py` (new), `configs/phase_b_b2_post_d2_paired_baseline.yaml` (new), `tests/test_paired_baseline_repair.py` (new), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_004_paired_baseline/` (`paired_baseline.json`, `comparison_manifest.json`, `failure_reproduction_matrix.json`, `select_bind_control_status.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

`B-C005R3-002` (ADR-0083) registered `development_units` (the merged L3 `{BIND,COUNT,SELECT}` group, L4 `BIND`, L4 `SELECT`) as the scope for the next repair tasks (`R3-006`/`007`/`008`) and `development` seeds 10-14 as their only authorized seed range. `B-C005R3-003` (ADR-0084) froze the offline adequacy vocabulary and statistical contract. This task establishes the repair *starting point* on the reproducible v2 fixture (ADR-0082) for those specific mechanisms -- without repeating the full second-diagnostic phase -- and reports, per the task doc's own design, whether each previously-diagnosed failure actually reproduces under (v2 generator) x (development seeds) x (the existing, unmodified R1/R2 training/eval code), rather than assuming it does.

### Evidence

1. **Retrieval comparison (R0/R1/R2), real, unmodified B-C005R1 machinery.** `train_repaired_router_and_scorer` and `evaluate_repair_cell` (`retrieval_repair_benchmark.py`, not touched) were run as-is for conditions R0 (frozen parent) / R1 (cross-entropy reconstruction) / R2 (current margin-ranking + `ArgumentScorer` repair recipe) on `development` seeds 10-14, `bank_size=128`, restricted to `development_units`' operations (SELECT/COUNT/BIND) at levels L3 (semantically-related) and L4 (confusable-family); SHIFT is excluded (its L3/L4 relation-split units are `validation`/`sealed_v2`, out of scope, confirmed via `assert_sealed_access_permitted`). Per-cell `build_hard_negative_candidates` is called with the same `seed`/`target_id` across R0/R1/R2, so support/query examples and the candidate identity graph are byte-identical across conditions; only the router's key values (and, for R2, the trained `ArgumentScorer`) differ -- the deliberate per-condition change being measured.
2. **None of the four previously-diagnosed routing/argument failures reproduce at R2 on `development` seeds (all `NOT_REPRODUCED_ON_V2`).** Measured mean `primitive_call_top1` at R2 (5 seeds): `COUNT->BIND` 1.0 (original regate_sealed/bank_size=128 finding: 0.4, `KEY_SCORING_BOTTLENECK`), `BIND->COUNT` 0.984 (original: 0.628, `KEY_SCORING_BOTTLENECK`), `SELECT:argument_variant` 0.994 (original: `ARGUMENT_ENCODING_FAILURE`), `BIND:argument_variant` 0.963 (original: `ARGUMENT_SCORER_GENERALIZATION_FAILURE`). For the two L4 cells, R0/R1 (no trained `ArgumentScorer`) sit near chance (0.40-0.49) as expected, and R2's trained `ArgumentScorer` lifts both to near-ceiling on this partition's own held-out query examples -- the opposite of D2's finding that R2's `ArgumentScorer` failed to generalize. This does not overwrite or reinterpret the original D2 findings (`representation_stage_summary.json`, `l4_argument_breakdown.json`, both unmodified); per the task doc's explicit escape hatch, a non-reproducing failure is reported as `NOT_REPRODUCED_ON_V2` and its downstream task is flagged, not silently treated as resolved.
3. **SELECT->BIND shuffled-control anomaly re-checked once: no wiring/leakage bug found; the anomaly itself still appears on 2/5 seeds (`UNRESOLVED`).** D2-002's exact `_fit_binary_probe`/`_probe_accuracy` methodology (`representation_stage_probe.py`, reused unchanged) was re-run scoped to SELECT-vs-BIND at the R2 condition, plus an explicit train/eval content-overlap audit the original code never asserted directly (zero overlap on all 5 seeds -- no probe leakage). The shuffled-label control accuracy was outside the chance tolerance (`|acc-0.5|<=0.15`) on seeds 10/11 (`q_probe_shuffled_accuracy` 0.516/0.711) and within it on seeds 12/13/14, matching the original's own qualitative characterization ("shuffled-label probe control was not near chance") rather than a clean pass or a code bug. `select_bind_control_status.json` records `UNRESOLVED`; router weights were not touched based on this outcome, per the task doc's explicit constraint.
4. **SHIFT reference-adequacy: a genuine structural scope limit, reported honestly rather than as a false pass/fail.** `development` seeds 10-14 have no entry under `runs/phase_a1_shift_compact_structural_probe/` (only sealed seeds 0-4 do, confirmed by directory listing), so `_build_frozen_base_system` falls back to a fresh random Core init for every seed this task is authorized to use (`assert_sealed_access_permitted` blocks the sealed seeds that do have a pretrained checkpoint). A real, frozen, per-example correctness stream for the identity-match SHIFT candidate (512 fresh v2 examples/seed) measured **0% closed-loop EM on all 5 seeds** -- consistent with encoder noise, not a reproduction of D2-005's ~92% finding. All three legacy/fixed-N/new-contract evaluators correctly and unanimously `REJECT` this stream (no interesting divergence, since 0% is nowhere near the 0.95 decision boundary the D2-005 premature-acceptance bug requires). This is reported as `UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE`, not `NOT_REPRODUCED_ON_V2` -- the mechanical four-evaluator comparison code is validated as working, but this specific stream cannot exercise or refute D2-005's finding. Resolving this for real requires either sealed access via the `R3-011`/`R3-012` pathway or a separately-authorized change of validation strategy; this task does not attempt either.
5. **Comparison validity.** `comparison_manifest.json` records what was fixed (generator version, seed formulas, candidate identity graph, shared hyperparameters) vs. deliberately varied (router/scorer weights) across conditions, discloses the smaller-than-primary-design support/query counts (32/64 vs. the primary design's proposed 256), and discloses the untrained-development-core fact so routing-level findings are not misread as full reproductions. `comparison_validity: VALID`.
6. **Tests.** `tests/test_paired_baseline_repair.py`, 33 cases, all pass, CPU-only -- covers config validation, `_pretrained_core_available` against the real checkoint layout, the relation-label/downstream-task mappings, the four-evaluator replay over synthetic streams (all-correct/all-wrong/~92%-boundary), and every classification branch of the three aggregation/status builders. Matching this repo's D2/R3-002 precedent, the GPU-backed orchestration (`run_paired_baseline_repair` itself) is exercised only via the milestone script, not pytest. `ruff check` and `mypy` both clean on the touched files.

### Consequences

- **No performance Gate is claimed or failed here** -- per the task doc, `B-C005R3-004` requires only that the comparison itself be valid, which it is.
- **`B-C005R3-006` (COUNT<->BIND), `B-C005R3-007` (SELECT encoding), and `B-C005R3-008` (BIND scorer) are each flagged `NEEDS_SCOPE_REVIEW`**: their originally-diagnosed target failure does not reproduce on `development` seeds under the current, unmodified R1/R2 pipeline. Per the task doc, this does not auto-authorize or auto-block their training; it means whoever starts one of those tasks next must first decide, with this evidence in hand, whether its declared local-repair scope still has a real target to repair on this partition, needs a different seed/scale condition, or needs its own further scoping work -- this task does not make that call itself.
- **`B-C005R3-009` (SHIFT)** inherits an unresolved evidentiary gap, not a contradicted finding: this task could not validly test SHIFT reference adequacy on `development` seeds at all (structural, not a bug), so D2-005's ~92% finding stands unchallenged and unconfirmed on v2; R3-009 (or a future scope-review task) must decide how to obtain a valid comparison (sealed access via R3-011/012, or an explicitly authorized alternative) before claiming to have repaired or measured this mechanism on the v2 fixture.
- **The SELECT->BIND `UNRESOLVED` classification from D2-002 stands, now with an explicit negative wiring/leakage audit attached**: it is not a measurement bug (train/eval overlap = 0 on every seed checked), and the anomaly itself is seed-dependent rather than universal on `development` seeds too.
- Historical B-C005/D/R1/R2/G/D2 results, the legacy asymmetric verifier, and `representation_stage_probe.py`'s own methodology are unmodified; every number here is a fresh `V2_RECONSTRUCTION`, never claimed as an `EXACT_REPLAY` of the original regate_sealed run.

## ADR-0086: Safe Bounded Verification -- New Finite-Look Exact-Bound Verifier Empirically Blocks the ADR-0080 Premature-Acceptance Failure Mode; Untrained-Core Limitation Extends to SELECT/COUNT/BIND Raw Execution (Task B-C005R3-005)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-005 complete; **G3: `G3_PASS`**)
**Affects:** `src/apc/meta/adequacy_verifier.py` (extended: new `BoundedExactLookVerifier`/`BoundedExactLookVerifierConfig`/`classify_search_status`/`enforce_verified_execution`; `SequentialAdequacyVerifier` unmodified), `src/apc/evaluation/safe_bounded_verification.py` (new), `configs/phase_b_b2_post_d2_safe_bounded_verification.yaml` (new), `tests/test_safe_bounded_verification.py` (new, 37 cases), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_005_safe_bounded_verification/` (`verifier_operating_characteristics.json`, `same_identity_safety.json`, `neural_candidate_stress_metrics.json`, `verification_trace.jsonl`, `cost_breakdown.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

ADR-0080/D2-005 found that the legacy `SequentialAdequacyVerifier`'s Wilson-sequential rule prematurely accepted a genuinely inadequate SHIFT candidate (independent reference EM 90.33-92.48%, below `tau=0.95`) by point estimate at `n=32`. `B-C005R3-003` (ADR-0084) froze the finite-look exact-bound statistical contract (`tau=0.95`, `looks=(32,64,128,256,512)`, `alpha_accept_episode=alpha_reject_episode=0.01`) as pure offline/CPU code, explicitly *not* connected to any runtime path. This task performs that runtime wiring and validates it two ways per the task doc: a CPU-only synthetic Bernoulli contract sweep, and a real GPU-backed neural candidate safety stress -- kept strictly separate.

### Evidence

1. **Runtime wiring, legacy verifier untouched.** `BoundedExactLookVerifier` (`apc/meta/adequacy_verifier.py`) delegates to the exact frozen `apc.evaluation.functional_metrics_v2.verify_candidate_finite_look`/`FiniteLookVerifierContract` (byte-for-byte identical trace, verified by test) rather than re-implementing the Beta-quantile formula under the same "exact" name -- a deliberate, documented reversal of the usual `meta`->`evaluation` import direction, per `B2_FUNCTIONAL_ADEQUACY_V2.md` S4.1's frozen-contract-reuse rule. `enforce_verified_execution` is a thin wrapper: `ACCEPT` + `DIRECT_REUSE`/`COMPOSE` -> `EXECUTED`; any `UNCERTAIN` (including at budget exhaustion) -> `NEEDS_MORE_EVIDENCE` with `controller_action=None` (no temporary workspace committed); `REJECT`, or a reuse attempt against a non-`ACCEPT` verdict -> `NO_VERIFIED_SOLUTION`. The existing 3-action controller's weights are never read or retrained. `SequentialAdequacyVerifier` is byte-for-byte unmodified.
2. **CPU-only Bernoulli contract sweep (`verifier_operating_characteristics.json`, 2000 episodes/`p`, 19.4s, `contract_empirically_consistent_all_p: true` across all 12 grid points).** At `p=0.949` (just below `tau`, the ADR-0080 danger zone) on the *identical* per-episode trial stream: **legacy asymmetric accepts 75.1% of episodes** (reproducing the premature-acceptance failure mode as a general phenomenon, not a one-off) while **the new verifier accepts 0.0%** (99.95% stay `UNCERTAIN` through look 512; 99%-CI upper bound on the false-accept rate is `0.0026`, comfortably inside the declared `alpha_accept_episode=0.01` budget). Across the whole `p<0.95` region the new verifier's accept rate is `0.0` at every grid point (`p=0.10` through `0.951`). Availability (`p in {0.99, 0.995, 1.0}`, required `P(ACCEPT)>=0.97`): new-verifier accept rates `0.987/1.0/1.0` -- **availability_check: pass**. `uncertain_rate` is reported only for the new verifier (legacy/fixed-N always force a binary decision, by construction).
3. **Historical old-insufficient-SHIFT replay (`same_identity_safety.json`), never a live sealed-seed access, SHIFT not retrained.** The 4 bank-size cells (16/32/64/128) from the already-committed `B-C005D2-005` audit (`runs/phase_b_b2_second_diagnostic/shift_seed24_adequacy_audit.json`, model_seed=4 / regate_sealed seed 24, true rate 90.3-92.5% via `reference_correct`/`n_reference_examples=1024`) are replayed by (a) copying the audit's own already-computed legacy sequential decision verbatim (no recomputation) and (b) reconstructing one deterministic, disclosed per-trial order (`derive_seed`-seeded shuffle, never Python's process-randomized `hash()`) consistent with the stored aggregate count, to exercise the new verifier's sequential looks. **Result: the stored historical legacy decision is `ACCEPT` on 2 of 4 cells (bank_size 16 and 128 -- an exact reproduction of the original `PREMATURE_ACCEPT` finding), while the new verifier's reconstruction never accepts on any of the 4 cells** (2 `REJECT`, 2 `UNCERTAIN`). Combined `same_identity_unsafe_accept_rate = 0/19` and `inadequate_call_accept_rate = 0/49` (denominators include both this historical source and the live neural stress below); per `EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md` S6.1, the small historical denominator (4 cells, one model) is an explicitly disclosed limitation, not treated as `NOT_ESTIMABLE`-blocking or padded to look larger.
4. **Real neural candidate stress on `development` seeds only (`neural_candidate_stress_metrics.json`, 45 episodes, 140.2s; `assert_sealed_access_permitted` enforced).** SELECT/COUNT/BIND x {wrong-family execution (e.g. BIND's real primitive executed on SELECT's examples, correct-arg fallback via `.get(default)`, never a crash), wrong-argument execution (same primitive, a genuinely different valid argument value), correct-identity/correct-argument execution} on seeds 10-14. **`wrong_call_accept_rate = 0/30`** (zero false accepts among wrong-family+wrong-argument streams). Zero unauthorized parameter updates, zero unselected-primitive forward calls, and a bounded real composition-search check (`search_composition_recipe`, depth 2, beam 8, on 32 examples) executed for every non-`ACCEPT` episode before it is logged as unresolved (never skipped straight to a "candidate failed" record without checking).
5. **Unexpected but structural finding: raw closed-loop execution correctness, not just routing/argument-scoring, is also near-zero for SELECT/COUNT/BIND on `development` seeds.** All 45 neural records -- including the 15 "adequate" (correct-identity, correct-argument) ones -- resolved `REF_INADEQUATE` (e.g. SELECT: 0-1/1024 correct across all 5 seeds; COUNT: 0-88/1024; BIND: 85-118/1024). Root cause (confirmed by reading `_build_frozen_base_system`, not guessed): `development` seeds 10-14 have no entry under `runs/phase_a1_shift_compact_structural_probe/` (only sealed seeds 0-4 do, per ADR-0085), so the shared Content Encoder (`core.model`) is a fresh random init there; every primitive's output is computed from `h_content = core.model.encode(...)`, so a garbled random encoding degrades every primitive's *raw* output equally, regardless of family. This generalizes ADR-0085's SHIFT-specific "`UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE`" finding to all four parameterized operations **when measured by raw execution EM** -- it does **not** reopen `NOT_REPRODUCED_ON_V2` for COUNT<->BIND/SELECT/BIND (ADR-0085's own metrics there, `primitive_call_top1`/`argument_accuracy`, are routing/scoring comparisons that ADR-0085 already established stay valid under a random-but-fixed core; this task measured a different, stricter signal). **Consequence for this task's own safety claim:** the live neural stress could not supply an empirical positive/`ACCEPT` control (no genuinely `REF_ADEQUATE` candidate exists on `development` seeds under this direct-execution measurement); that role is filled only by the Bernoulli sweep's availability check (finding 2 above). Recorded explicitly as `neural_candidate_stress_metrics.json:untrained_core_disclosure`.
6. **Tests.** `tests/test_safe_bounded_verification.py`, 37 cases, all pass, CPU-only: the runtime verifier matches the frozen offline contract bit-for-bit; `enforce_verified_execution`'s every branch including its two documented out-of-contract `ValueError`s (a `PLASTIC_SEARCH` request, and an `ACCEPT` verdict without a reuse/compose request); `_generate_trial_stream`/`_reconstruct_ordered_stream` determinism (via `derive_seed`, never `hash()`); the historical-replay path against both a missing-file fixture (`HISTORICAL_ARTIFACT_UNAVAILABLE`) and a small synthetic committed-artifact fixture. `ruff check` and `mypy src/apc` both clean on every touched file (re-verified in this session); `python -m pytest -q tests/test_safe_bounded_verification.py` re-run in this session, 37/37 pass.

### Consequences

- **G3: `G3_PASS`.** All three false-acceptance metrics are `0.0` (`wrong_call_accept_rate=0/30`, `inadequate_call_accept_rate=0/49`, `same_identity_unsafe_accept_rate=0/19`), the Bernoulli contract sweep is empirically consistent with its declared alpha budgets at all 12 tested `p`, and the availability requirement passes. No router, `ArgumentScorer`, primitive, or controller weight was changed; SHIFT was not retrained.
- **This does not certify SHIFT's functional adequacy is fixed.** The historical replay demonstrates only that the *new verifier* correctly rejects/stays uncertain about the old-insufficient SHIFT identity; `B-C005R3-009`'s own repair-and-measure scope (still blocked on sealed access per ADR-0085) is unaffected.
- **The legacy `SequentialAdequacyVerifier` is kept, named `legacy_asymmetric`, as a historical-comparison baseline** -- not deleted, per this branch's historical-integrity rule; its premature-acceptance behavior at `p` near `tau` is now empirically quantified (75.1% false-accept at `p=0.949`) rather than only qualitatively described.
- **`B-C005R3-006`/`007`/`008` are unaffected by finding 5**: their own `NEEDS_SCOPE_REVIEW` flag and reproduction status (ADR-0085) rest on routing/argument-scorer metrics, not raw execution EM; this ADR's finding is a new, additional disclosure about a *different* metric on the same partition, not a reopening of ADR-0085's own conclusion.
- Historical B-C005/D/R1/R2/G/D2 results and `shift_seed24_adequacy_audit.json` are unmodified; the historical legacy verifier decision is read and reported verbatim, never recomputed or edited.
- Per the task doc's STOP rule, only `B-C005R3-005` was executed; `B-C005R3-006` onward and `B-C006`/Task Inference remain blocked pending an explicit next user instruction.

## ADR-0087: COUNT<->BIND Key/Scoring Repair -- A Genuinely Scoped Mechanism Is Implemented and Passes, but There Is No Confusion Left to Repair on `development` Seeds (Task B-C005R3-006)

**Date:** 2026-09-06
**Status:** Accepted (Task B-C005R3-006 complete; **local repair Gate: `VALIDATION_PASS`**, chosen mechanism `scoped_pairwise_margin`)
**Affects:** `src/apc/evaluation/count_bind_key_scoring_repair.py` (new), `configs/phase_b_b2_post_d2_count_bind_key_scoring_repair.yaml` (new), `tests/test_count_bind_key_scoring_repair.py` (new, 32 cases), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_006_count_bind_key_scoring_repair/` (`count_bind_scoring_repair.json`, `pair_margin_report.json`, `checkpoint_hashes.json`, `frozen_state_audit.json`, `loss_exposure_log.json`, `variant_selection.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

`B-C005R3-004` (ADR-0085) flagged this task `NEEDS_SCOPE_REVIEW`: the original ADR-0081 `KEY_SCORING_BOTTLENECK` finding for `COUNT->BIND`/`BIND->COUNT` (regate_sealed seed 24 / model_seed=4, bank_size=128: `primitive_call_top1` 0.40 / 0.628) did not reproduce on `development` seeds 10-14 under the existing, unmodified R1/R2 recipe (R2 there already measured 1.0 / 0.984). Per the task doc's mandatory "修正前の必須確認", this task first re-confirmed that finding was not itself an artifact of a key/ID bug, a normalization inconsistency, an artificial score offset, or a training-exposure defect, before deciding whether/how to train anything -- then implemented and evaluated the scoped repair mechanism the task doc requires regardless, since the original sealed-partition failure remains inaccessible for re-diagnosis under current sealed-access rules.

### Evidence

1. **Pre-repair diagnostics are clean on all 5 development seeds.** `audit_key_id_correspondence`: all 16 bank operations bijectively map to a registered router key, no duplicates. `audit_key_norms`: `score_fn='dot'` (no cosine normalization anywhere in the scoring path), COUNT/BIND key norms are close in magnitude (e.g. seed 10: 1.1998 / 1.0464, ratio 1.147) -- no gross norm imbalance that could bias raw dot-product scores. `audit_no_artificial_score_offset`: manually recomputing `dot(query_proj(z), key)` element-by-element against the router's batched formula over 8 examples x 128 candidates matches to `max_abs_difference=1.9e-6` (float32 numerical noise) -- `NO_ARTIFICIAL_OFFSET_FOUND`. All seeds: `all_pre_repair_checks_clean=true`.
2. **A genuinely narrower repair mechanism than the existing recipe, with an empirically stronger holdout guarantee than R3-002/ADR-0083 found achievable.** `train_count_bind_scoped_repair` (new function; `train_repaired_router_and_scorer` is not modified and remains available as the "normal objective control") sets `requires_grad=True` on only the COUNT and BIND router keys; every other key (14 ops, including SELECT despite its physical-BIND coupling from R3-002's `SELECT-BIND`/`COUNT-BIND` merged component) and `query_proj` stay `requires_grad=False` for the whole run. `frozen_state_audit.json` confirms this bit-for-bit in all 10 runs (2 variants x 5 seeds): every non-target key and `query_proj` byte-identical before/after, while COUNT's and BIND's own keys are confirmed changed in every run -- training had a real, but strictly scoped, effect. This is a stronger, measured guarantee for every non-target op (`FROZEN_NEGATIVE_EXPOSURE_ZERO_PARAMETER_UPDATE`, `loss_exposure_log.json`) than R3-002's own `MINING_HOLDOUT_ONLY` finding for the existing full-candidate-list recipe -- achieved by adding a new, narrower function, not by changing the one R3-002 found capped (out of authorized scope through R3-010).
3. **Two candidate mechanisms scored an exact tie on the internal selection split; the disclosed tie-break rule chose `scoped_pairwise_margin`.** `scoped_ce` (plain cross-entropy) and `scoped_pairwise_margin` (explicit margin loss against the real `_RELATED_OPERATION` competitor key, not R2's synthetic random near-neighbor) were trained and scored on a disjoint split (`seed*70_000`/`*80_000` offsets, never reused for the Gate numbers below): both scored identically per seed (`0.875, 1.0, 0.921875, 1.0, 1.0`, mean `0.959375`). This exact tie is explainable rather than suspicious: with only one real hard competitor per class in this 2-op scoped setting, CE's gradient naturally concentrates on that same closest competing logit, so the two objectives coincide in effect here.
4. **Central finding: R0 (untouched frozen parent) already meets both directions' Gate thresholds on `development` seeds, independently re-deriving R3-004/ADR-0085's exact numbers this session** (`COUNT->BIND` top-1 `0.965625`, `BIND->COUNT` top-1 `0.984375`, both top-5 `1.0`). The chosen scoped repair's measured top-1/top-5 for both directions are **identical** to R0's (`delta_over_r0=0.0` both directions) even though training measurably increased score-margin separation (`COUNT->BIND` mean margin `5.427->8.459`; `BIND->COUNT` `5.384->8.538`) -- the small residual ~1.6-3.4% of examples R0 already got wrong are not simply marginal near-boundary cases that more separation resolves; some other, undiagnosed effect keeps them wrong, which this task's declared scope does not investigate further.
5. **R2 (existing generic full-bank-retrain recipe, reused as the "normal objective control") does resolve `COUNT->BIND`'s residual (top-1 `1.0`) but not `BIND->COUNT`'s (`0.984375`, same as R0/scoped)** -- an observation, not a claim: fixing that specific residual apparently needs a broader intervention than touching only COUNT/BIND's own keys, which is out of R3-006's authorized scope.
6. **`SELECT->BIND` (disclosure-only; explicitly excluded from this task's Gate per the task doc's own scope boundary) mildly improved as a side effect** (top-1 `0.996875 -> 1.0`) of BIND's key moving -- reported for completeness, not claimed as a repair, per the explicit prohibition on broadening scope to SELECT->BIND.
7. **Legacy routing regression (all 16 bank operations, easy dev-split examples): R0 is a perfect `1.0` on every seed; the chosen scoped repair drops `0.898pp`**, inside the `<=1.0pp` budget but close to it -- consistent with COUNT/BIND's own easy-task accuracy shifting slightly as their keys moved apart from each other, not a regression in any other operation's key (which `frozen_state_audit.json` confirms did not move at all).
8. **Result: local repair Gate `VALIDATION_PASS`** for both directions (top-1 >= 0.95, top-5 >= 0.99, legacy regression <= 1.0pp, zero unselected-primitive forward calls, leak audit passed) -- reported together with an explicit `scientific_caveat` (finding 4): this PASS demonstrates the mechanism is implemented correctly, does not regress an already-passing partition, and satisfies every stated freeze/leak/exposure constraint; it does **not** demonstrate repair of the original ADR-0081 sealed-partition `KEY_SCORING_BOTTLENECK`, which remains inaccessible under current sealed-access rules and is not re-diagnosable here.
9. **Tests.** `tests/test_count_bind_key_scoring_repair.py`, 32 cases, all pass, CPU-only -- config validation; the three pure audits against a real (small) `Router`; `train_count_bind_scoped_repair` exercised with real CPU optimizer steps for **both** variants (needs no Core/bank/generator at all), confirming exactly the two target keys change and every other key plus `query_proj` stay byte-identical; `build_freeze_audit`/`build_checkpoint_hashes` positive cases (after real training) and negative cases (a manually mutated non-target key or `query_proj` is correctly flagged); `build_loss_exposure_log`'s classification; `_build_variant_selection`'s tie-break rule. Combined with the existing R3-002/004/005 suites (121 cases total across the four files), all pass. `ruff check` and `mypy` both clean on every touched file. Real GPU milestone run: 261.4s.

### Consequences

- **`B-C005R3-006`'s `NEEDS_SCOPE_REVIEW` flag from ADR-0085 is closed**, with the answer being "implemented, tested, and correctly scoped -- there was nothing left to repair on `development` seeds," not "repair confirmed" or "nothing to do here." Any future claim that this task repaired the original sealed-partition `KEY_SCORING_BOTTLENECK` would be unsupported; that remains open pending the `R3-011`/`R3-012` sealed-access pathway.
- **`train_repaired_router_and_scorer` is unmodified**; `train_count_bind_scoped_repair` is a new, additional, narrower-scoped function. This does not reopen or relax R3-002/ADR-0083's `MINING_HOLDOUT_ONLY` finding for the *existing* recipe -- it demonstrates a stronger guarantee is achievable via a new, narrowly-scoped function instead.
- **`B-C005R3-007` (SELECT encoding) and `B-C005R3-008` (BIND scorer) are unaffected**: their own `NEEDS_SCOPE_REVIEW` flags from ADR-0085 concern argument encoding/scoring, a different mechanism than this task's routing/key-scoring scope.
- **An open, undiagnosed question is flagged, not resolved**: the small residual fraction of `COUNT<->BIND`/`BIND<->COUNT` misroutes that persist under R0 and the scoped repair (but that R2's broader full-bank retrain resolves for `COUNT->BIND` only) is not investigated further here, per this task's declared scope limits.
- Historical B-C005/D/R1/R2/G/D2 results and R3-001..005 artifacts are unmodified; every number in this task is a fresh measurement on `development` seeds under the v2 generator, not a replay.
- Per the task doc's STOP rule, only `B-C005R3-006` was executed; `B-C005R3-007` onward and `B-C006`/Task Inference remain blocked pending an explicit next user instruction.

## ADR-0088: SELECT Argument-Encoding Repair -- A Genuine Train/Inference Formula Mismatch Confirmed and Fixed, Requiring No Retraining (Task B-C005R3-007)

**Date:** 2026-09-07
**Status:** Accepted (Task B-C005R3-007 complete; **local repair Gate: `VALIDATION_PASS`**)
**Affects:** `src/apc/primitives/argument_scoring.py` (fixed), `src/apc/evaluation/select_argument_encoding_repair.py` (new), `configs/phase_b_b2_post_d2_select_argument_encoding_repair.yaml` (new), `tests/test_argument_scoring.py` (extended, +4 cases), `tests/test_select_argument_encoding_repair.py` (new, 25 cases), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_007_select_argument_encoding_repair/` (`select_encoding_contract.json`, `select_argument_repair.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

`B-C005R3-004` (ADR-0085) flagged this task `NEEDS_SCOPE_REVIEW`: SELECT's `argument_variant` did not reproduce ADR-0081's sealed-partition `ARGUMENT_ENCODING_FAILURE` on `development` seeds (R2 `argument_accuracy_mean=0.99375`, vs. the sealed partition's `sealed_post_repair_argument_accuracy=0.8625`, `sealed_rare_value_argument_accuracy=0.333`). Per the task doc's own escape hatch, this task first had to determine from code and tests -- not presumed -- whether SELECT's actual encoding schema has a real contract violation, or whether the original failure was purely a training-insufficiency artifact, in which case the task doc requires stopping at scope review rather than making an unrelated encoding change.

### Evidence

1. **The schema audit confirms no defect in the `PrimitiveCall`/generator encoding itself.** `audit_select_argument_schema`/`audit_round_trip_and_canonicalization` (all 5 development seeds): SELECT's `indices` is a variable-length, order-preserving `list[int]`, size `= SelectOp.output_length(len) = max(1, len // 2)` (a deterministic function of input length, never an independent random draw); the generator always emits ascending-sorted, distinct, in-bounds indices; `PrimitiveCall`/`TaskStepSpec` round-trips exact-order with no defect; there is no padding/mask/size-field contract to violate; the L4 wrong-argument competitor (`hard_negative_routing_benchmark._wrong_arguments`) also re-sorts ascending, preserving the invariant. Every check is `true` on every seed.
2. **A genuine, still-present `ArgumentScorer` train/inference formula mismatch was found and confirmed as the actual, evidence-supported root cause.** `ArgumentScorer.train_on_examples` fits SELECT with independent multi-hot `BCEWithLogitsLoss` targets (each index an independent Bernoulli), while `forward` read a single shared `softmax` distribution for every operation, gathering and averaging each SELECT index's share of that ONE unit of total probability mass -- exactly ADR-0081's D2-004 `structural_note`, confirmed still present by reading the live source (`audit_argument_scorer_train_inference_contract`). Per the task doc's own rule, finding a real contract violation (not just insufficient training) authorizes a minimal fix rather than a scope-review stop.
3. **The fix is a single conditional, requiring no retraining.** `ArgumentScorer.forward` now reads `torch.sigmoid(logits)` for `operation == "SELECT"` and the unchanged `F.softmax(logits, dim=-1)` for every other operation. Because the head's weights were already fit under `BCEWithLogitsLoss` (which directly targets a sigmoid readout), the SAME trained weights are simply scored correctly for the first time (`select_argument_repair.json:no_retraining_required=true`).
4. **A hand-built characterization isolates and quantifies the mechanism dramatically, independent of any trained checkpoint.** `softmax_dilution_characterization` (maximally-confident logits, `arg_vocab_size=32`): at cardinality 1 the legacy softmax score is `0.9996` (comparable to the repaired sigmoid's `0.9951` -- no dilution partner yet); the instant a SECOND simultaneously-correct index appears, the legacy score **collapses to `-0.00009`** (each of the two must now split softmax's one unit of probability mass ~50/50, which this `[-1,1]`-centered formula reads as "completely uncertain"); it continues falling to `-0.50` at cardinality 4, `-0.75` at cardinality 8, and `-0.875` at cardinality 16 -- strongly *negative*, i.e. worse than uninformative, despite the network being maximally confident about every index. The repaired sigmoid score stays flat at `0.9951` regardless of cardinality.
5. **Live measurement on real `development`-seed checkpoints confirms the mechanism matters even within this benchmark's narrow cardinality range (3-5), and confirms SHIFT/COUNT/BIND are unaffected.** Using the SAME trained R2 `ArgumentScorer` weights (one training run per seed), scored once through the real (repaired) `forward` and once through a read-only `_LegacyPreFixArgumentScorer` adapter reproducing the exact pre-fix formula on the identical weights: standard L4 SELECT cell (cardinality 3-5 mixed) repaired `argument_accuracy_mean=1.0` vs. legacy `0.99375` (+0.625pp, independently re-deriving R3-004/ADR-0085's exact `0.99375` legacy number this session); cardinality-stress at cardinality 4 (`sequence_length=8`) repaired `1.0` vs. legacy `0.9875` (+1.25pp); SHIFT/COUNT/BIND L4 cells repaired and legacy are **numerically identical to full floating-point precision on every seed** (`other_op_regression_pp: {"SHIFT": 0.0, "COUNT": 0.0, "BIND": 0.0}` -- freeze proof by direct measurement, not just code inspection); rare-value stratification (bottom-20%-frequency index values, 62-74 "rare-containing" examples per seed) is mixed and small-sample at the per-seed level (e.g. seed 12: legacy `0.9855` vs. repaired `1.0`; seed 10: legacy `1.0` vs. repaired `0.9726`), reported honestly rather than cherry-picked; the legacy 16-op routing regression (`R0=1.0`, `R2=0.9375` on every seed) is a property of the shared, unmodified `train_repaired_router_and_scorer` recipe, not of this task's fix, and independently reproduces R3-006's own `R2` measurement (`0.9375` on every seed) on this same partition/config.
6. **Why the standard benchmark's aggregate delta is small despite a confirmed severe defect.** `_wrong_arguments` builds SELECT's L4 competitor by shifting every correct index by +1 (mod length) and re-sorting, so the wrong-argument competitor always has the SAME cardinality as the correct answer. Under a shared softmax this same-cardinality pairing cancels most of the systematic per-cardinality dilution bias (both sides are diluted by roughly the same amount), which is why R2's pre-fix aggregate accuracy was already high despite the real formula defect. The defect's practical impact should matter most for combining this score with a family score at a different fixed scale (`arg_lambda`-weighted) or comparing candidates of *different* SELECT cardinalities -- neither of which this benchmark's same-cardinality-competitor design can detect, but which finding 4's characterization confirms directly.
7. **Result: local repair Gate `VALIDATION_PASS`** (full argument accuracy `1.0 >= 0.95`, full-call top-1 `1.0 >= 0.90`, max other-op regression `0.0pp <= 1.0pp`, zero unselected-primitive forward calls, leak audit passed) -- reported with an explicit `scientific_caveat` (finding 6): this PASS demonstrates the fix corrects a real, confirmed architectural defect with a minimal change requiring no retraining, and does not regress SHIFT/COUNT/BIND or the already-passing `development` benchmark; it does **not** demonstrate repair of the original ADR-0081 sealed-partition finding (`sealed_rare_value_argument_accuracy=0.333`), which remains inaccessible under current sealed-access rules (R3-011/R3-012 pathway required).
8. **Tests.** `tests/test_argument_scoring.py` gains 4 cases (SELECT-specific sigmoid-vs-softmax formula check, non-SELECT freeze check, multi-index BCE-trained discrimination, hand-built dilution characterization) -- 7 total, all pass. `tests/test_select_argument_encoding_repair.py` (new, 25 cases, CPU-only, no Core/bank/GPU): config validation; the real (not mocked) schema/round-trip audits against the actual lightweight example generator; the dilution characterization's monotonicity and cardinality-1-to-2-collapse properties; the train/inference contract audit; the `_LegacyPreFixArgumentScorer` adapter's weight-sharing and formula-reproduction on both SELECT and non-SELECT operations. Full repository suite: 1835 passed, 0 failed. `ruff check` and `mypy` both clean (127 files). Real GPU milestone run: 347.2s.

### Consequences

- **`B-C005R3-007`'s `NEEDS_SCOPE_REVIEW` flag from ADR-0085 is closed**, with the answer being "a genuine defect was found and fixed with a minimal, well-evidenced change" -- not a scope-review stop, and not an unrelated encoding change either (the `PrimitiveCall`/generator encoding itself has no defect).
- **`ArgumentScorer.forward`'s only change is the SELECT-specific probability-normalization branch**; `train_on_examples`, the router, every primitive, and the verifier are unmodified and confirmed (not just assumed) unaffected.
- **`B-C005R3-008` (BIND argument-scorer generalization repair) is unaffected**: it concerns BIND's own head/compatibility mechanism, a different failure (`ARGUMENT_SCORER_GENERALIZATION_FAILURE`), not SELECT's readout formula.
- **The rare-value stratification's mixed, small-sample per-seed signal is disclosed, not resolved further** -- out of this task's declared scope; a future task with sealed access (R3-011/R3-012 pathway) would be needed to test whether the fix helps on the original sealed-partition's more severe rare-value degradation (`0.333`).
- Historical B-C005/D/R1/R2/G/D2/R3-001..006 results and artifacts are unmodified; every number in this task is a fresh measurement on `development` seeds under the v2 generator, not a replay.
- Per the task doc's STOP rule, only `B-C005R3-007` was executed; `B-C005R3-008` onward and `B-C006`/Task Inference remain blocked pending an explicit next user instruction.

## ADR-0089: BIND Argument-Scorer Generalization Repair -- The Sealed-Partition Rare-Value Failure Mode Reproduces on `development` (Masked by Aggregate Metrics) and Is Fixed by Value-Coverage-Stratified Training (Task B-C005R3-008)

**Date:** 2026-09-07
**Status:** Accepted (Task B-C005R3-008 complete; **local repair Gate: `VALIDATION_PASS`**)
**Affects:** `src/apc/evaluation/bind_argument_scorer_repair.py` (new), `configs/phase_b_b2_post_d2_bind_argument_scorer_repair.yaml` (new), `tests/test_bind_argument_scorer_repair.py` (new, 35 cases), `scripts/run_phase_b_b2_post_d2_repair.py` (extended), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B.md`
**Run Artifacts:** `runs/phase_b_b2_post_d2/r3_008_bind_argument_scorer_repair/` (`bind_argument_repair.json`, `checkpoint_hashes.json`, `frozen_state_audit.json`, `variant_selection.json`, `config.yaml`, `system.json`, `protocol.json`)

### Context

ADR-0081 (D2-004) classified BIND's argument scorer as `ARGUMENT_SCORER_GENERALIZATION_FAILURE`: sealed-partition accuracy `0.89375` overall, `0.9316` on `query_key` values seen during training vs. `0.0` on unseen ("rare") values (`sealed_rare_value_share=0.0406`), and a development-only probe trained on more examples measurably closed part of the gap (`data_scale_mean_improvement=0.075`). `B-C005R3-004` (ADR-0085) found the AGGREGATE number did not clearly reproduce on `development` seeds 10-14 under the existing, unmodified R2 recipe (`argument_accuracy_mean=0.9625`, just above the local repair Gate's `0.95` threshold) and flagged this task `NEEDS_SCOPE_REVIEW`. Per the task doc, this task first had to confirm BIND's value/structure, wrong-key-distance, frequency, and content-length-conditioned failures on `development` -- reusing, not re-deriving, `apc.evaluation.argument_generalization_audit`'s existing D2-004 descriptor functions restricted to `development` seeds only (that module's own config requires sealed-seed access, so this task builds its own orchestration loop around the same row-building/breakdown functions instead of calling it directly) -- before deciding whether or how to repair anything.

### Evidence

1. **The aggregate `0.9625` hid the exact same rare/seen split ADR-0081 found on the sealed partition, just at lower incidence.** Per-example decomposition of the standard R2 recipe's 320 `development` query examples (`seen_vs_rare` in `argument_structure_breakdown.json`, reused verbatim from `argument_generalization_audit._rows_for_operation`): 308 examples whose correct `query_key` value was present in that seed's 32-example training draw scored `argument_accuracy=1.0`; the other 12 (`3.75%`, comparable to ADR-0081's sealed `4.06%`) whose value was ABSENT from training scored `argument_accuracy=0.0` -- total failure, not partial degradation. `(308*1.0 + 12*0.0)/320 = 0.9625` independently re-derives R3-004/ADR-0085's exact aggregate number via a completely different (per-example rare/seen) computation path, cross-validating both measurements.
2. **The training-value-coverage audit confirms the concrete mechanism, per seed.** `audit_training_value_coverage` on the standard 32-example i.i.d. BIND draw (seed 10, vocab size 10): value `1` has **zero** training examples, value `5` has exactly one; `min_count_over_seen_values=1`. With only 32 draws over a 10-value domain (env `vocab_size=10`, confirmed via `core.tokens.env_vocab_size`, not the `ArgumentScorer`'s 32-slot head), i.i.d. sampling leaves real, measurable coverage gaps -- exactly the precondition the ADR-0081 classification requires.
3. **Content-length and wrong-key-distance decompositions (reused D2-004 descriptors) show no separate, distance- or length-driven failure mode.** `argument_distance` is a constant `1.0` for every BIND row in both conditions (`_wrong_arguments` always perturbs by exactly one, modulo the key domain -- a structural property, not a repair-relevant signal). `sequence_length` shows a mild monotonic baseline trend (`0.9596` at length 6 to `0.9655` at length 10) fully explained by the same rare-value mechanism spread roughly evenly across lengths, not an independent length effect.
4. **Repair scope and mechanism.** Per the task doc, only BIND's `ArgumentScorer` head and its training loss/*sampling* may change; the Router (`query_proj`/keys), SHIFT/SELECT/COUNT heads (SELECT's R3-007 sigmoid fix included), the verifier, and the primitive bank are frozen. Two dev-only *sampling*-only mechanisms (no example is synthesized; every training example is drawn unmodified from a larger, `development`-only pool) were trained per seed and compared on an internal, disjoint selection split (`variant_selection.json`, seed offsets `*770_000`/`*780_000`, never reused in final Gate numbers): `larger_iid_sample` (the standard i.i.d. draw, just larger -- 128 instead of 32 examples, doubling as this task's own reproduction of ADR-0081's `data_scale_control` finding on `development`) and `stratified_value_coverage` (the same larger pool, rebalanced so every observed `query_key` value gets a roughly equal share, capped at `target_total // n_distinct_values` per value with leftover top-up). Both scored `1.0` mean selection accuracy (tied); the disclosed tie-break rule (prefer the value-coverage-targeted mechanism) selected `stratified_value_coverage`.
5. **The stratified repair concretely eliminates the measured coverage gap.** Seed 10's chosen-variant training set (`chosen_variant_training_value_coverage_by_seed`): every one of the 10 legal values now has 12-20 training examples (`min_count_over_seen_values=12`, `n_values_never_seen=0`), versus the standard draw's `0`-`5` range with one value entirely unseen. Post-repair, the same 320-example `development` query set shows **zero** remaining "rare" examples and `argument_accuracy=1.0` on all of them (`seen_vs_rare`: `{"seen": {"n": 320, "argument_accuracy": 1.0}, "rare": {"n": 0, "argument_accuracy": None}}`).
6. **`data_scale_control` reproduces ADR-0081's own question on `development`, with an appropriately smaller effect size.** Training on 128 dev-only examples (`larger_iid_sample`) raised mean argument accuracy from R2's `0.9625` to `1.0` (`mean_improvement=0.0375`) -- smaller than ADR-0081's sealed-partition `0.075` because `development`'s R2 baseline already has less headroom (`0.9625` vs. the sealed partition's `0.89375`), not because the mechanism differs.
7. **Freeze audit and checkpoint hashes confirm the repair is scoped exactly as claimed, on all 10 (variant x seed) combinations.** `frozen_state_audit.json:all_freeze_audits_passed=true`: SHIFT/SELECT/COUNT heads are bit-for-bit identical before/after (`torch.equal` on `.weight`/`.bias`) in every case; BIND's own head is confirmed changed in every case (not a vacuous freeze audit). `checkpoint_hashes.json` records SHA256 hashes of the full scorer state and each individual head before/after. Router, verifier, and primitive bank are never touched by this task's code at all (no train/eval call in this module accepts or mutates them); candidate-bank size is measured identical before/after for every seed (`candidate_per_argument_persistent_duplication_check: no_persistent_duplication=true`), confirming no per-argument-value primitive duplication.
8. **Calibration readout is explicitly disclosed as uncalibrated softmax probability, not a guarantee.** `calibration_summary` (`mean_p_correct_argument`: baseline `0.950` -> repaired `0.999`; `mean_p_correct_when_argument_incorrect`: baseline `~3e-6` -> repaired `None`, no incorrect examples remain) reuses `argument_generalization_audit`'s own readout verbatim -- these are `F.softmax` probabilities read directly from `ArgumentScorer.heads["BIND"]`, with no temperature/Platt scaling applied anywhere in this repair; `bind_argument_repair.json:calibration_readout_note` states this explicitly.
9. **Result: local repair Gate `VALIDATION_PASS`.** Chosen-variant final cells (5 development seeds, disjoint from both the repair-training pool and the selection split): `argument_accuracy=1.0 >= 0.95`, `full_call_top1=1.0 >= 0.90`, `family_top1=1.0 >= 0.98`, `top5=1.0 >= 0.99`, `max_other_op_regression_pp=0.0 <= 1.0` (SHIFT/SELECT/COUNT numerically identical to the R2 baseline, freeze-proof by measurement), zero unselected-primitive forward calls, leak audit passed. Reported with an explicit `scientific_caveat`: this PASS demonstrates the mechanism is implemented correctly, scoped exactly to BIND's own head, clears every declared threshold, and does not regress the other three parameterized operations -- it does **not** demonstrate repair of the original ADR-0081 sealed-partition finding (`sealed_rare_value_argument_accuracy=0.0`), which remains inaccessible under current sealed-access rules (R3-011/R3-012 pathway required); the improvement measured here is over `development`'s own already-higher R2 baseline, not over the sealed partition's.
10. **Tests.** `tests/test_bind_argument_scorer_repair.py` (new, 35 cases, CPU-only, no Core/bank/GPU): config validation; the real (not mocked) schema audit against the actual lightweight example generator; the training-value-coverage audit on both empty and real generated inputs; the stratified-sampling function's coverage/determinism/no-synthesis properties on real generated `Example` pools; the freeze-audit and checkpoint-hash helpers exercised on real (lightweight, Core-free) `ArgumentScorer` instances with deliberate mutations; the pure aggregation/gate-decision helpers. Full repository suite: 1870 passed, 0 failed. `ruff check` and `mypy` both clean (129 files). Real GPU milestone run: 30.7s.

### Consequences

- **`B-C005R3-008`'s `NEEDS_SCOPE_REVIEW` flag from ADR-0085 is closed**, with the answer being "the sealed-partition failure mode does reproduce on `development`, masked by low incidence in the aggregate metric, and is fixed by a dev-only, sampling-only repair" -- not a scope-review stop, and not evidence that nothing needed fixing (unlike R3-006's COUNT<->BIND finding).
- **`ArgumentScorer`'s class code (`argument_scoring.py`) is completely unmodified by this task** (unlike R3-007, which changed `forward` itself); the repair lives entirely in a new, additional training function scoped to one head, mirroring R3-006's "new narrower function, existing recipe untouched" pattern rather than R3-007's "minimal in-place fix" pattern.
- **This does not demonstrate repair of the original sealed-partition finding.** The sealed partition's `sealed_rare_value_argument_accuracy=0.0` remains unmeasured and unrepaired under current sealed-access rules; whether the same stratified-coverage mechanism would close that larger gap (rare share `4.06%` there vs. `3.75%` measured here, but on a different, inaccessible model/data pairing) is open pending the R3-011/R3-012 sealed-access pathway.
- **`B-C005R3-006` (COUNT<->BIND routing) and `B-C005R3-007` (SELECT encoding) are unaffected**: this task never touches the Router or SELECT's head; freeze audits confirm this by measurement.
- Historical B-C005/D/R1/R2/G/D2/R3-001..007 results and artifacts are unmodified; every number in this task is a fresh measurement on `development` seeds under the v2 generator, not a replay.
- Per the task doc's STOP rule, only `B-C005R3-008` was executed; `B-C005R3-009` onward and `B-C006`/Task Inference remain blocked pending an explicit next user instruction.

