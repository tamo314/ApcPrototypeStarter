# Phase B — B2 Hard-Negative Routing Repair Tasks

## Status

This document defines the corrective task sequence after:

**B-C005 — STOP GATE B2: hard-negative routing and functional safety**

returned **FAIL**.

The original B-C005 result remains valid historical evidence and must not be rewritten, reclassified, or retroactively made to pass.

This repair sequence does **not** authorize B-C006 or Task Inference work.

The Phase B dependency graph becomes:

```text
B-C005 FAIL
   ↓
B-C005D   failure isolation
   ↓
B-C005R1  retrieval-ranking repair
   ↓
B-C005R2  adequacy-estimator repair
   ↓
B-C005G   new sealed B2 re-gate
   ↓ PASS
B-C006 may resume
```

Implement **only the explicitly requested task**.

Do not automatically continue to the next repair task.

---

# 1. Scientific interpretation of the failed gate

The B-C005 failure must be treated as two potentially independent mechanisms.

## 1.1 Retrieval-ranking failure

At `N=128`, the original B-C005 run reported:

```text
L0 top-1 = 1.000
L1 top-1 = 1.000
L2 top-1 = 0.866
L3 top-1 = 0.662
L4 top-1 = 0.504
```

while:

```text
top-5 inclusion = 1.000 at every level
```

Therefore the first retrieval hypothesis is:

> the frozen Phase A.2 router retains high candidate recall but lacks fine-grained score margin among near-neighbor, semantically related, and argument-confusable candidates.

Do not describe this as loss of semantic retrieval until candidate-set recall itself fails.

---

## 1.2 Functional adequacy / false-plastic failure

The original B-C005 run also reported:

```text
wrong functional acceptance = 0.0%
closed-loop EM             = 95.9%
false plastic              = 3.75%
```

with:

```text
support examples = 32
adequacy threshold = 0.95
```

At 32 support examples, `EM >= 0.95` effectively requires at least `31/32` correct.

The corrective investigation must test the hypothesis that part or all of the 3.75% false-plastic rate is caused by finite-support estimator variance rather than true functional inadequacy.

Do not lower the adequacy threshold in the diagnostic task.

---

# 2. Load-bearing constraints

The following rules apply to every repair task.

## 2.1 Preserve the failed result

Do not modify or overwrite:

```text
runs/phase_b_hard_negative_safety_gate/
ADR-0075
```

except for adding cross-references from later documentation.

The original B-C005 measurement remains:

**FAIL**

even if the repaired system later passes a new gate.

---

## 2.2 Do not tune on the original sealed evaluation set

The hard-negative examples / competitor configurations measured in the original B-C005 sealed gate may be used for:

- post-hoc diagnosis;
- failure localization;
- plotting;
- descriptive statistics.

They may **not** be used for:

- training;
- margin-loss examples;
- model selection;
- threshold selection;
- search-budget selection;
- hyperparameter tuning.

Create separate **development hard-negative partitions** for repair work.

The re-gate must use a newly designated sealed evaluation partition.

If the repository currently lacks hard-negative partition status, add:

```text
DEVELOPMENT
SEALED_EVALUATION
RETIRED_FROM_SEALED
```

or the closest equivalent protocol metadata.

---

## 2.3 Do not repair two mechanisms at once

The order is mandatory:

```text
diagnose
→ retrieval repair
→ adequacy repair
→ re-gate
```

B-C005R1 must not alter adequacy thresholds or statistical decision rules.

B-C005R2 must freeze the repaired retrieval mechanism selected in B-C005R1.

This is required for causal attribution.

---

## 2.4 Preserve APC invariants

Do not weaken:

```text
h_content = f(content)
```

Do not introduce:

- task metadata into the content path;
- oracle operation/family labels into router inference;
- hard-negative difficulty labels into router inference;
- query targets into functional verification;
- per-argument persistent primitive duplication;
- execution of all primitives followed by masking.

Only selected/proposed candidates may execute according to the explicit verification budget.

---

# B-C005D — Failure Isolation: Ranking vs Argument Resolution vs Adequacy Variance

## Goal

Determine precisely why B-C005 failed before modifying the architecture.

This task is **diagnostic only**.

No router weights, router architecture, adequacy threshold, controller threshold, or plastic policy may change.

---

## D1. Decompose retrieval metrics

The current B-C005 metric `logical candidate top-1` is insufficient for L4.

Add separate metrics:

```text
physical_primitive_top1
physical_primitive_topk
argument_accuracy
primitive_call_top1
primitive_call_topk
```

where:

```text
PrimitiveCall = (physical primitive ID, arguments)
```

For parameterized primitives such as:

```text
SHIFT(amount)
SELECT(indices)
COUNT(target)
BIND(query_key)
```

measure:

1. whether the correct primitive family was retrieved;
2. whether the correct argument was resolved;
3. whether the full `PrimitiveCall` was ranked correctly.

### Required L4 interpretation

Classify each L4 failure as one of:

```text
FAMILY_RANKING_FAILURE
ARGUMENT_RESOLUTION_FAILURE
BOTH
NEITHER
```

Do not use the old single top-1 number alone to characterize L4.

---

## D2. Log score-margin distributions

For every matrix cell, log:

```text
score_correct
score_best_wrong
score_margin = score_correct - score_best_wrong
correct_rank
best_wrong_provenance
```

Aggregate by:

```text
N
hard-negative level
target operation
seed
```

Produce:

- mean margin;
- median margin;
- p05 / p95 margin;
- fraction margin <= 0;
- target-rank histogram.

The purpose is to determine whether L2/L3 degradation is:

```text
continuous margin collapse
```

or:

```text
specific semantic collision
```

---

## D3. Diagnose correct-candidate support adequacy

For every episode where the final decision becomes false plastic, record for the **correct candidate**:

```text
support_correct_count
support_size
support_EM
query_EM
operation
seed
N
hard_negative_level
correct_candidate_rank
```

Also record the same statistics for accepted known episodes.

Do not use query targets in the decision itself.

Query metrics are diagnostic only.

---

## D4. Support-size variance experiment

Freeze the complete B-C005 system.

Run the same known-task verification logic with:

```text
support_size ∈ {16, 32, 64, 128}
```

Keep:

```text
adequacy_threshold = 0.95
```

unchanged.

Use development diagnostic examples only.

Measure:

```text
false_plastic_rate
correct_candidate_false_reject_rate
wrong_candidate_false_accept_rate
mean support EM
mean query EM
```

Also compute the binomial reference curve for representative underlying candidate accuracies:

```text
p ∈ {0.97, 0.98, 0.99, 0.995}
```

The binomial calculation is a diagnostic reference only; do not assume independent Bernoulli errors when interpreting the neural system.

---

## D5. Candidate-order sensitivity

Because B-C005 accepts the first proposed candidate reaching adequacy, verify whether outcome depends on ranking order.

For diagnostics only, compare:

```text
A. ranked-first acceptance
B. evaluate all top-5, choose highest support adequacy
C. oracle correct-candidate adequacy only
```

Do not use B or C as the production repair in this task.

Measure whether false plastic remains when the correct top-5 candidate is explicitly evaluated.

This distinguishes:

```text
ranking/order problem
```

from:

```text
correct-candidate adequacy estimation problem
```

---

## D6. Required outputs

Suggested run:

```text
runs/phase_b_b2_failure_isolation/
```

Required artifacts:

```text
config.yaml
metrics.jsonl
summary.json
system.json
failure_breakdown.json
margin_summary.json
support_variance.json
plots/
```

Recommended plots:

```text
top1_vs_level
topk_vs_level
margin_distribution_by_level
rank_histogram_by_level
l4_family_vs_argument_failures
false_plastic_vs_support_size
support_count_histogram_false_plastic
```

---

## D7. Acceptance criteria

This is not a scientific PASS/FAIL gate.

Task completion requires that every original B-C005 failure can be assigned to measurable categories.

The summary must answer:

1. Is L2 failure primarily margin/ranking failure?
2. Is L3 failure primarily margin/ranking failure?
3. For L4, what fraction is family routing vs argument resolution?
4. What fraction of false plastic occurs despite the correct candidate being in top-5?
5. What fraction occurs despite the correct candidate having high query EM?
6. Does false plastic decrease as support size increases without changing threshold?
7. Is candidate ordering contributing materially?

If any answer cannot be determined, mark it:

```text
UNRESOLVED
```

Do not infer it from aggregate EM.

---

## D8. ADR

Append an ADR after the diagnostic run.

Record the measured decomposition.

Do not yet select the repair architecture unless the diagnostic evidence supports it.

---

# B-C005R1 — Retrieval Ranking Repair

## Goal

Improve fine-grained hard-negative ranking while preserving:

```text
top-k candidate recall
functional verifier
controller thresholds
plastic policy
primitive execution semantics
```

This task targets retrieval only.

---

## R1.1 Preconditions

B-C005D must be complete.

Proceed only if the diagnostic shows a meaningful retrieval-ranking component.

If L4 is predominantly argument-resolution failure while physical primitive family ranking remains strong, do not force the router to solve an argument problem through primitive-key classification.

In that case use the separate argument-aware scoring path described below.

---

## R1.2 Development data only

Construct hard negatives from development partitions.

Training may include:

```text
near-neighbor negatives
semantically related primitive negatives
same-family argument-confusable calls
```

but never the original B-C005 sealed evaluation cells.

Record the provenance of every negative.

---

## R1.3 Primary repair: ranking objective before capacity increase

Do not increase router size first.

Prefer adding an explicit ranking objective to the existing task/key scoring mechanism.

A recommended formulation is a margin or contrastive loss:

```text
L_rank = max(0, margin - s_positive + s_negative)
```

or an equivalent multi-negative ranking loss.

The implementation must remain compatible with current routing scores.

The exact loss may differ if the repository already has an established ranking-loss utility.

---

## R1.4 Family and argument separation

For parameterized primitives, preserve:

```text
primitive family != primitive argument
```

If B-C005D shows L4 family ranking is already correct, implement L4 scoring as:

```text
family score
+
argument compatibility score
```

or an equivalent factorized `PrimitiveCall` score.

Do not create one learned persistent primitive key for every argument value.

Recommended conceptual scoring:

```text
score(PrimitiveCall)
    = score_family(z_task, primitive_key)
    + λ * score_args(z_task, call.arguments)
```

`λ` must be a declared config value.

If no additional argument scorer is necessary based on B-C005D evidence, do not add one.

---

## R1.5 Training controls

Compare at minimum:

```text
R0 frozen Phase A.2 router
R1 same architecture + standard routing objective
R2 same architecture + hard-negative ranking objective
```

Optional only if R2 clearly fails:

```text
R3 modest capacity increase + same ranking objective
```

Do not jump directly to R3.

---

## R1.6 Metrics

On a **development hard-negative validation set**:

```text
physical primitive top-1/top-k
argument accuracy
PrimitiveCall top-1/top-k
margin
closed-loop EM
false functional acceptance
false plastic
unselected primitive calls
old semantic routing degradation
```

Also evaluate the original easy routing suite to detect regression.

---

## R1.7 Acceptance criteria

This task selects a repair candidate; it does not reopen the sealed B2 gate.

Across >= 5 development seeds:

### Retrieval

At `N=128`:

```text
L0-L2 primitive-call top-1 >= 0.98
L3    primitive-call top-1 >= 0.95
L4    primitive-call top-1 >= 0.90
top-5 inclusion >= 0.99 at all levels
```

If L4 is factorized, also require:

```text
physical primitive family top-1 >= 0.98
argument accuracy >= 0.95
```

### Regression

```text
easy known-task routing drop <= 1.0 pp
old semantic routing top-1 drop <= 1.0 pp
unselected primitive calls == 0
```

### Safety

```text
wrong functional acceptance <= 1%
```

Failure to meet these criteria requires an ADR and a mechanism-specific investigation.

Do not modify adequacy estimation in R1.

---

# B-C005R2 — Functional Adequacy Estimator Repair

## Goal

Reduce false plastic caused by uncertain finite-support adequacy estimates without weakening protection against wrong functional reuse.

Freeze the retrieval mechanism selected by B-C005R1.

Do not retrain or resize it in this task.

---

## R2.1 Preserve the semantic meaning of adequacy

The target concept remains:

> Is the candidate functionally adequate?

Do not redefine adequacy as:

- router confidence;
- task-key distance;
- candidate identity;
- known registry membership.

Functional execution remains authoritative.

---

## R2.2 Do not simply lower the threshold

The repair must not be:

```text
0.95 → lower number
```

selected because B-C005 failed.

Instead test a statistical or sequential verification policy.

---

## R2.3 Recommended primary policy: sequential support verification

Implement a bounded sequential verifier.

Conceptually:

```text
initial support batch
    ↓
clearly adequate?
    ├─ yes → ACCEPT
    ↓ no
clearly inadequate?
    ├─ yes → REJECT
    ↓ uncertain
evaluate additional support examples
    ↓
ACCEPT / REJECT at max budget
```

Recommended development budgets:

```text
initial_support = 32
support_increment = 32
max_support = 128
```

These are development defaults, not a retrospective reinterpretation of B-C005.

All budgets must be config fields.

---

## R2.4 Confidence rule

Prefer a confidence-bound or hypothesis-test formulation.

For example, estimate a confidence interval for candidate success probability and classify:

```text
lower_bound >= adequacy_threshold
    → ACCEPT

upper_bound < adequacy_threshold
    → REJECT

otherwise
    → UNCERTAIN / gather more evidence
```

Use a statistically justified interval implementation.

Do not hand-code an arbitrary margin around 0.95.

The exact confidence level must be predeclared in config.

Recommended development starting point:

```text
confidence = 0.95
```

---

## R2.5 Required baselines

Compare:

```text
A. original fixed-32 hard threshold
B. fixed-64 hard threshold
C. fixed-128 hard threshold
D. sequential bounded verifier
```

All use the same:

```text
adequacy target = 0.95
```

The target is unchanged.

---

## R2.6 Metrics

Measure:

```text
false plastic rate
wrong functional acceptance
closed-loop EM
mean support examples consumed
p95 support examples consumed
decision latency
verification FLOPs
candidate execution count
```

Break down by:

```text
operation
hard-negative level
N
seed
```

---

## R2.7 Acceptance criteria

Across >= 5 development seeds:

```text
false plastic <= 2.0%
wrong functional acceptance <= 1.0%
closed-loop EM >= 0.95
```

and:

```text
mean support examples consumed < 64
```

for the sequential policy unless the diagnostic evidence demonstrates that a larger evidence budget is intrinsically required.

Also require:

```text
no query-target leakage
unselected primitive calls == 0
```

If reducing false plastic materially increases wrong acceptance, the repair FAILS.

Safety has priority over avoiding unnecessary plastic expansion.

---

# B-C005G — New Sealed Hard-Negative B2 Re-Gate

## Goal

Re-evaluate STOP GATE B2 on a new sealed evaluation partition after the retrieval and adequacy repairs have been selected using development data only.

This is the only task that may unblock B-C006.

---

## G1. Seal protocol

Before running:

1. freeze router architecture and weights/training recipe;
2. freeze ranking objective;
3. freeze argument-scoring design;
4. freeze adequacy verifier;
5. freeze support budgets;
6. freeze controller thresholds;
7. freeze top-k;
8. freeze bank-size matrix;
9. freeze hard-negative construction procedure.

Then designate the new sealed evaluation partition.

Record a protocol hash or serialized config snapshot.

No tuning after the first sealed result.

---

## G2. Required matrix

Use the original Phase B B2 matrix shape unless an ADR explicitly records a scientifically necessary change:

```text
seeds = 5 or more
N = {16, 32, 64, 128}
L = {L0, L1, L2, L3, L4}
```

Include the same parameterized target families where applicable.

If the repaired L4 metric is factorized, report both:

```text
physical primitive family metrics
full PrimitiveCall metrics
```

Do not omit the original-style logical-call metric.

---

## G3. Re-gate thresholds

The original predeclared B2 functional thresholds remain authoritative unless an ADR written **before the new sealed run** formally replaces an invalid metric definition.

Required:

### Retrieval

At `N=128`:

```text
L0-L2 top-1 >= 0.98
L3    top-1 >= 0.95
L4    top-1 >= 0.90
top-k inclusion >= 0.99 at every level
```

If L4 is formally decomposed through an ADR, additionally require:

```text
physical primitive family top-1 >= 0.98
argument accuracy >= 0.95
full PrimitiveCall top-1 >= 0.90
```

### Functional safety

```text
false wrong acceptance <= 1%
closed-loop EM >= 0.95
false plastic <= 2%
unselected primitive calls == 0
```

### Integrity

```text
router/primitives do not mutate during matrix evaluation
evaluation metadata leakage == 0
```

---

## G4. PASS behavior

If B-C005G passes:

1. record a new ADR stating that the repaired B2 gate passed;
2. keep ADR-0075 as the historical original failure;
3. update the active Phase B task dependency so B-C006 is unblocked;
4. do not automatically start B-C006.

Use wording such as:

> Original B-C005 failed under the frozen Phase A.2 retrieval/adequacy design. After mechanism-isolated development repair, B-C005G passed on a newly sealed hard-negative evaluation partition.

Do not write:

> B-C005 was actually a pass.

---

## G5. FAIL behavior

If the new sealed gate fails:

1. save all artifacts;
2. append an ADR;
3. keep B-C006 blocked;
4. classify the first failing mechanism;
5. do not tune on the failed sealed partition.

A further repair cycle requires another new sealed partition.

---

# 3. Recommended repository changes

Prefer existing neighboring abstractions.

Do not create a new top-level package.

Suggested additions only if equivalent modules do not already exist:

```text
src/apc/
├── primitives/
│   ├── routing_losses.py
│   └── argument_scoring.py
├── meta/
│   └── adequacy_verifier.py
└── evaluation/
    ├── hard_negative_failure_isolation.py
    └── hard_negative_repair_gate.py

scripts/
├── run_phase_b_b2_failure_isolation.py
├── run_phase_b_b2_retrieval_repair.py
├── run_phase_b_b2_adequacy_repair.py
└── run_phase_b_b2_regate.py

configs/
├── phase_b_b2_failure_isolation.yaml
├── phase_b_b2_retrieval_repair.yaml
├── phase_b_b2_adequacy_repair.yaml
└── phase_b_b2_regate.yaml

tests/
├── test_hard_negative_failure_isolation.py
├── test_routing_ranking_loss.py
├── test_argument_scoring.py
└── test_adequacy_verifier.py
```

Do not duplicate an existing router/loss/verifier abstraction merely to match these suggested names.

---

# 4. Required tests

At minimum add coverage for:

## Retrieval

- hard-negative development/sealed partitions are disjoint;
- original B-C005 sealed examples are rejected as training data;
- positive candidate score receives the intended ranking gradient;
- unselected primitives remain unexecuted;
- family and argument metrics are separately correct;
- argument-confusable L4 candidates do not create new persistent primitive families.

## Adequacy

- fixed-32 behavior reproduces legacy B-C005 logic;
- sequential verifier stops early for clearly adequate candidates;
- sequential verifier stops early for clearly inadequate candidates;
- uncertain cases request more evidence up to max budget;
- query examples are inaccessible to the verifier;
- confidence-rule implementation matches a trusted reference calculation;
- wrong candidates are not accepted merely because extra support is available.

## Protocol

- development data may train the repair;
- sealed data cannot be consumed by training;
- first sealed re-gate locks configuration;
- a post-seal config mutation is rejected or visibly invalidates the seal.

---

# 5. Verification commands

Use repository-standard verification:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

During iteration, run focused tests first.

Suggested experiment interfaces:

```bash
python scripts/run_phase_b_b2_failure_isolation.py \
  --config configs/phase_b_b2_failure_isolation.yaml \
  --run-dir runs/phase_b_b2_failure_isolation

python scripts/run_phase_b_b2_retrieval_repair.py \
  --config configs/phase_b_b2_retrieval_repair.yaml \
  --run-dir runs/phase_b_b2_retrieval_repair

python scripts/run_phase_b_b2_adequacy_repair.py \
  --config configs/phase_b_b2_adequacy_repair.yaml \
  --run-dir runs/phase_b_b2_adequacy_repair

python scripts/run_phase_b_b2_regate.py \
  --config configs/phase_b_b2_regate.yaml \
  --run-dir runs/phase_b_b2_regate
```

If the repository has a different canonical CLI pattern, follow the repository pattern instead of creating redundant wrappers.

---

# 6. Completion-report format

After each repair task report:

```text
Task ID:
Files changed:
Tests run:
Experiment commands:
Run artifacts:
Seeds:
Development/sealed partition:
Router configuration:
Adequacy configuration:
Support budget:
Bank sizes:
Hard-negative levels:
Acceptance criteria:
  - criterion → measured → PASS/FAIL
Failure localization:
Assumptions/deviations:
ADR added/required:
Downstream blocked?:
```

For B-C005D also include:

```text
L2 primary failure:
L3 primary failure:
L4 family-ranking failure fraction:
L4 argument-resolution failure fraction:
False-plastic with correct candidate in top-5:
False-plastic with high correct-candidate query EM:
Support-size dependence:
Candidate-order dependence:
```

---

# 7. Explicit non-goals

During this repair sequence do **not** add:

- Task Inference;
- natural-language inputs;
- pretrained LMs;
- RL meta-controller;
- distributed training;
- vector database;
- architecture search;
- new Stable Core architecture;
- decoder redesign;
- arbitrary threshold relaxation.

The B2 repair is complete only when hard-negative retrieval and adequacy are repaired and independently validated on a new sealed gate.

---

# 8. Definition of done

The repair sequence is complete only when:

```text
B-C005D complete
AND
B-C005R1 development criteria pass
AND
B-C005R2 development criteria pass
AND
B-C005G new sealed STOP GATE passes
```

Only then may B-C006 become unblocked.

A repaired development result without a new sealed re-gate is not sufficient.
