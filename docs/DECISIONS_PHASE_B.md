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

