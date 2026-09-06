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

