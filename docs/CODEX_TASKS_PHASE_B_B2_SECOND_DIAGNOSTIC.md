> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Phase B — B2 Second Diagnostic Phase

## Status

This document defines the **second diagnostic phase** after:

```text
B-C005   original STOP GATE B2          → FAIL
B-C005D  first failure isolation        → COMPLETE
B-C005R1 retrieval repair (development) → PASS
B-C005R2 adequacy repair (development)  → PASS
B-C005G  new sealed re-gate             → FAIL
```

The second diagnostic phase exists because the first repair cycle succeeded on development partitions but did **not** generalize to the new sealed partition.

This phase is **diagnostic only**. It does not authorize another repair, threshold changes, B-C006, or Task Inference.

The dependency graph is:

```text
B-C005G FAIL
    ↓
B-C005D2-001  sealed/development discrepancy audit
    ↓
B-C005D2-002  L3 representation-stage localization
    ↓
B-C005D2-003  L3 semantic-relation generalization audit
    ↓
B-C005D2-004  L4 argument-generalization decomposition
    ↓
B-C005D2-005  SHIFT seed-24 adequacy / false-plastic audit
    ↓
B-C005D2-006  integrated causal diagnosis + next-repair decision
    ↓
STOP
```

Implement **only the explicitly requested task**. Do not automatically continue to the next task.

---

# 1. Why a second diagnostic is required

The first repair cycle produced strong development results:

```text
Retrieval repair:
L0-L2 PrimitiveCall top-1 = 1.000
L3 PrimitiveCall top-1    = 1.000
L4 PrimitiveCall top-1    = 0.9742

Adequacy repair:
false plastic             = 0.00%
wrong functional accept   = 0.00%
closed-loop EM            = 0.9961
```

but the new sealed re-gate produced:

```text
L0-L2 PrimitiveCall top-1 = 1.000
L3 PrimitiveCall top-1    = 0.6555
L4 family top-1           = 1.000
L4 argument accuracy      = 0.8883
L4 PrimitiveCall top-1    = 0.8883
wrong functional accept   = 0.0000
closed-loop EM            = 0.9588
false plastic             = 1.000 in seed-24 / SHIFT cells
```

The second diagnostic must distinguish at least four possibilities:

1. **Representation bottleneck** — L3-relevant semantic information is absent from or lost between `z_task`, `query_proj`, and primitive-key scoring.
2. **Hard-negative split failure** — development L3 negatives do not represent the semantic collision structures appearing in sealed evaluation.
3. **Argument generalization failure** — factorized family/argument scoring is conceptually correct but `ArgumentScorer` does not generalize across operations/argument structures/values.
4. **Adequacy-definition / primitive-performance conflation** — some "false plastic" cases may actually be justified because the installed primitive itself fails the declared adequacy threshold on that episode distribution.

No repair may be selected until these possibilities are separated.

---

# 2. Immutable research boundaries

## 2.1 Preserve all prior evidence

Do not overwrite or reinterpret away:

```text
ADR-0075  original B-C005 FAIL
ADR-0076  first diagnostic
ADR-0077  retrieval repair
ADR-0078  adequacy repair
ADR-0079  new sealed B-C005G FAIL
```

The new diagnostic may add retrospective interpretation, including evidence that a prior attribution was too strong. Historical conclusions remain preserved.

## 2.2 The B-C005G sealed partition remains sealed

The B-C005G sealed partition:

```text
seeds [20, 21, 22, 23, 24]
```

may be used for diagnostic readout, probes, failure localization, descriptive statistics, evaluation-only oracle labels, and plots.

It may **not** be used for training, gradient updates, hyperparameter selection, threshold selection, model selection, scoring-coefficient tuning, support-budget tuning, or architecture selection.

Any later repair must use a separate development partition. Any later re-gate must designate another new sealed partition.

## 2.3 No architecture change in D2

During `B-C005D2-001` through `B-C005D2-006`, freeze:

```text
Task Encoder
query_proj
primitive keys
CombinedRoutingLoss-trained router state
ArgumentScorer
SequentialAdequacyVerifier
controller thresholds
top-k
support budgets
primitive bank
Stable Core
```

Evaluation-only probes may be trained only where explicitly permitted and may never feed back into runtime.

## 2.4 Preserve APC causal invariants

Always preserve:

```text
h_content = f(content)
```

Do not expose to runtime:

- sealed/development status;
- hard-negative level;
- oracle operation/family labels;
- failure-category labels;
- query targets;
- reference-adequacy labels.

Only evaluation code may access these fields.

---

# B-C005D2-001 — Development vs Sealed Discrepancy Audit

## Goal

Establish whether the development hard-negative benchmark reproduced the difficulty later observed in sealed evaluation.

This task is descriptive. Do not change models or thresholds.

## D2-001.1 Reconstruct comparable conditions

For each condition:

```text
original sealed B-C005       seeds [0..4]
R1 development               seeds [10..14]
new sealed B-C005G           seeds [20..24]
```

report comparable pre-repair and post-repair metrics whenever the corresponding model state is available.

At minimum for L3 and L4:

```text
family_top1
argument_accuracy
PrimitiveCall_top1
top5
score_margin
correct_rank
```

Do not compare incompatible checkpoints without labeling them.

## D2-001.2 Difficulty descriptors

For every hard-negative pair, log evaluation-only descriptors:

```text
target primitive family
competitor primitive family
target arguments
competitor arguments
relation_type
target_key_norm
competitor_key_norm
key_cosine_similarity
query_target_score
query_competitor_score
score_margin
```

For L3 additionally record:

```text
semantic_relation_id
target_family_group
competitor_family_group
```

`semantic_relation_id` must describe a structural/semantic relation, not a seed or outcome label. Use the repository's real operation taxonomy where available.

## D2-001.3 Distribution comparison

Compare development vs both sealed partitions for:

```text
key similarity
score margin
target rank
relation-type frequencies
target/competitor pair frequencies
argument-value frequencies
```

Report mean/std, median, p05/p95, empirical distributions, and per-relation sample counts.

The key question is:

> Was R1 development L3 intrinsically easier before repair?

## D2-001.4 Required verdict

Classify development representativeness as one of:

```text
REPRESENTATIVE
PARTIALLY_REPRESENTATIVE
NON_REPRESENTATIVE
UNRESOLVED
```

If pre-repair development L3 is already near-perfect while both sealed sets are poor, explicitly flag:

```text
DEVELOPMENT_DIFFICULTY_MISMATCH
```

---

# B-C005D2-002 — L3 Representation-Stage Localization

## Goal

Determine where L3 semantic discrimination is lost.

Inspect:

```text
TaskSpec
   ↓
Task Encoder
   ↓
z_task
   ↓
query_proj
   ↓
q_task
   ↓
primitive-key scoring
   ↓
candidate ranking
```

The diagnostic must distinguish:

```text
TASK_REPRESENTATION_BOTTLENECK
QUERY_PROJECTION_BOTTLENECK
KEY_SCORING_BOTTLENECK
RELATION_GENERALIZATION_FAILURE
MIXED
UNRESOLVED
```

## D2-002.1 Capture representations

For matched L3 examples from development, original sealed, and new sealed, record:

```text
z_task
q_task = query_proj(z_task)
target primitive key
competitor primitive key
target score
competitor score
```

Do not modify the runtime path.

## D2-002.2 Direct representation diagnostics

First use geometry-only diagnostics:

```text
cosine / Euclidean distance
target-key vs competitor-key margin
nearest-key rank
pairwise separability
```

If geometry alone is insufficient, train small evaluation-only linear probes on **development/probe-only data**, then evaluate on held-out semantic relations and sealed diagnostic data.

Possible questions:

```text
Can z_task predict target primitive family?
Can z_task distinguish the L3 target from the specific competitor?
Can q_task do the same?
Can q_task predict the target key neighborhood?
```

Probe parameters must never be used by runtime. Include a label-shuffled control where practical.

## D2-002.3 Counterfactual diagnostic controls

For diagnosis only, compare:

```text
A. current q_task → current keys
B. z_task → simple evaluation-only similarity/readout
C. q_task → evaluation-only target/competitor classifier
D. oracle target family → current key lookup
```

These controls are not repair candidates.

## D2-002.4 Interpretation logic

If `z_task` separates target/competitor but `q_task` does not:

```text
QUERY_PROJECTION_BOTTLENECK
```

If both `z_task` and `q_task` fail:

```text
TASK_REPRESENTATION_BOTTLENECK
```

or, if only unseen relations fail:

```text
RELATION_GENERALIZATION_FAILURE
```

If both representations separate but current key ranking fails:

```text
KEY_SCORING_BOTTLENECK
```

If results differ by relation family, report `MIXED` plus per-relation conclusions.

---

# B-C005D2-003 — L3 Semantic-Relation Generalization Audit

## Goal

Determine whether seed-based development/sealed separation was insufficient because semantic-relation distributions were uncontrolled.

The core question is whether future evaluation requires:

```text
semantic-relation holdout
```

rather than seed-only holdout.

## D2-003.1 Define relation units

Create an evaluation taxonomy for L3 competitor relations using meaningful units such as:

```text
(target family, competitor family)
dependency-structure relation
operator-category relation
learned-key neighborhood relation
```

Use the smallest taxonomy supported by the existing registry and benchmark design. Do not create labels from success/failure outcomes.

## D2-003.2 Relation-group evaluation

Without changing the APC model, compute performance grouped by relation.

If an evaluation-only probe is used, train it on some development relation groups and test on held-out relation groups. Runtime remains frozen.

## D2-003.3 Difficulty-matched analysis

Construct **diagnostic-only** matched subsets from development and sealed data using pre-model or frozen-model descriptors such as:

```text
key cosine similarity bins
pre-repair score-margin bins
relation type
target family
```

Ask:

> When development and sealed examples are matched for relation and difficulty, does the performance gap remain?

This distinguishes distribution shift from model generalization failure.

Do not use query outcome labels for matching.

## D2-003.4 Required verdict

Choose one or more:

```text
SEED_SPLIT_SUFFICIENT
SEMANTIC_RELATION_HOLDOUT_REQUIRED
DIFFICULTY_MATCHING_REQUIRED
MODEL_GENERALIZATION_FAILURE_AFTER_MATCHING
UNRESOLVED
```

If `SEMANTIC_RELATION_HOLDOUT_REQUIRED`, future repair/re-gate protocols must be changed by ADR before any new training.

---

# B-C005D2-004 — L4 Argument-Generalization Decomposition

## Goal

Explain why factorized scoring reached approximately 97% development L4 accuracy but only approximately 88.8% on new sealed evaluation while physical family routing remained perfect.

Do not change `ArgumentScorer`.

## D2-004.1 Break down by operation

Report separately for:

```text
SHIFT
COUNT
BIND
SELECT
```

For each:

```text
family_top1
argument_accuracy
PrimitiveCall_top1
top5
argument_score_margin
```

## D2-004.2 Break down by argument structure

Where applicable, stratify by:

```text
argument value frequency
seen vs rare values
distance between correct and wrong argument
single-valued vs set-valued argument
sequence length
content-dependent ambiguity
```

For SELECT, preserve its structured/set-valued argument semantics rather than forcing a scalar metric.

## D2-004.3 Calibration

Log:

```text
P(correct argument | z_task)
P(best wrong argument | z_task)
argument_score_margin
```

Assess calibration separately from top-1 accuracy. Do not alter `lambda` or logits.

## D2-004.4 Failure classification

Classify each operation:

```text
VALUE_COVERAGE_FAILURE
ARGUMENT_ENCODING_FAILURE
ARGUMENT_SCORER_GENERALIZATION_FAILURE
TASK_REPRESENTATION_FAILURE
NO_FAILURE
UNRESOLVED
```

If one operation dominates sealed error, do not generalize its diagnosis to all parameterized primitives.

---

# B-C005D2-005 — SHIFT Seed-24 Adequacy and False-Plastic Audit

## Goal

Determine whether seed-24 / SHIFT plastic decisions reflect:

```text
FINITE_SUPPORT_VARIANCE
TRUE_PRIMITIVE_INADEQUACY
SEQUENTIAL_RULE_BIAS
DATA_DISTRIBUTION_SHIFT
IMPLEMENTATION_BUG
METRIC_MISCLASSIFICATION
UNRESOLVED
```

This task is critical. Do not change the sequential verifier.

## D2-005.1 Reconstruct every affected episode

For every sealed seed-24 / SHIFT cell that selected plasticity, log:

```text
candidate rank
family rank
argument correctness
support cumulative correct counts at n=32,64,96,128
support cumulative EM
Wilson lower/upper bounds
verifier state at each step
final verifier decision
independent query/reference EM
```

Use independent evaluation-only data. Never feed it into runtime.

## D2-005.2 Large-sample reference adequacy

For diagnosis only, estimate the installed candidate's true functional performance on a much larger independent sample.

Recommended:

```text
>= 1024 examples per affected task/cell
```

if computationally practical. Otherwise use the largest practical deterministic batch and report its size.

Compute:

```text
reference_EM
reference confidence interval
reference_adequate = reference_EM >= 0.95
```

`reference_adequate` is evaluation-only.

## D2-005.3 Reclassify plastic decisions for diagnosis

### True false plastic

```text
reference_adequate == True
AND
runtime selects PLASTIC
```

### Functionally justified plastic

```text
reference_adequate == False
AND
runtime selects PLASTIC
```

### Unsafe reuse

```text
reference_adequate == False
AND
runtime accepts reuse
```

Report all three. Do not use this retrospective classification to alter the original B-C005G verdict.

## D2-005.4 Verify sequential-rule behavior

The current verifier is documented as:

```text
early accept if empirical accuracy >= 0.95
early reject if Wilson upper bound < 0.95
otherwise gather more evidence
```

Verify implementation exactly matches this rule.

Also compute the counterfactual **evaluation-only** classification:

```text
Wilson lower bound >= 0.95 → accept
Wilson upper bound < 0.95  → reject
otherwise                  → uncertain
```

This symmetric rule is a diagnostic control only. Do not deploy it in D2.

## D2-005.5 ADR consequence

If `TRUE_PRIMITIVE_INADEQUACY`, `SEQUENTIAL_RULE_BIAS`, or `METRIC_MISCLASSIFICATION` is found, add a retrospective qualification to ADR-0076's attribution that false plastic was "100% finite-support estimator variance".

Do not rewrite ADR-0076; append a new ADR that cites the new evidence.

---

# B-C005D2-006 — Integrated Causal Diagnosis and Repair Decision Gate

## Goal

Combine D2-001 through D2-005 into a single causal diagnosis.

This task does **not** implement the next repair.

## D2-006.1 Required findings table

Produce:

| Mechanism | Evidence | Verdict | Confidence | Next action |
|---|---|---|---|---|
| L2 retrieval | ... | ... | ... | ... |
| L3 task representation | ... | ... | ... | ... |
| L3 query projection | ... | ... | ... | ... |
| L3 key/scoring | ... | ... | ... | ... |
| L3 relation split | ... | ... | ... | ... |
| L4 family routing | ... | ... | ... | ... |
| L4 argument resolution | ... | ... | ... | ... |
| adequacy estimator | ... | ... | ... | ... |
| installed SHIFT adequacy | ... | ... | ... | ... |
| false-plastic metric | ... | ... | ... | ... |

No row may be filled from intuition alone.

## D2-006.2 Allowed next-repair recommendations

Choose only evidence-supported options.

### Option A — Query projection repair

Allowed only if:

```text
z_task retains discriminative information
AND
query_proj loses it
```

Possible later repair: unfreeze/retrain query projection or add relation-aware contrastive training with bounded replay.

### Option B — Task representation repair

Allowed only if `z_task` itself lacks the required information.

Do not modify the task-blind content path.

### Option C — Semantic-relation holdout redesign

Allowed if development/sealed mismatch is primarily relation-distribution mismatch.

Before any next repair training, define:

```text
development relation set
validation relation set
sealed relation set
```

Seed-only separation is then no longer sufficient.

### Option D — Argument scorer repair

Allowed only for the operations/argument structures empirically shown to fail.

Do not create one persistent primitive key per argument value.

### Option E — Adequacy metric/protocol repair

Allowed if false-plastic labels misclassify truly inadequate primitives or sequential confidence logic is biased.

Possible later changes may include reference-based evaluation definitions or a symmetric sequential confidence policy, but do not alter historical gate results.

### Option F — Primitive functional-generalization repair

Allowed if installed SHIFT is genuinely below 0.95 on the sealed episode distribution.

In that case, the problem is not primarily the novelty controller. Investigate primitive/generalization before changing adequacy thresholds.

## D2-006.3 Forbidden conclusions

Do not conclude:

```text
router needs to be larger
```

unless diagnostics show the required information exists and the current scorer cannot express the separation.

Do not conclude:

```text
adequacy threshold should be lower
```

because a known primitive fails 0.95.

Primitive quality and adequacy criterion are separate variables.

---

# 3. Recommended repository additions

Prefer existing modules where possible.

Suggested files only if equivalent infrastructure does not already exist:

```text
src/apc/evaluation/
├── hard_negative_second_diagnostic.py
├── semantic_relation_audit.py
├── representation_stage_probe.py
├── argument_generalization_audit.py
└── adequacy_reference_audit.py

scripts/
├── run_phase_b_b2_second_diagnostic.py
├── run_phase_b_b2_representation_probe.py
└── run_phase_b_b2_adequacy_reference_audit.py

configs/
├── phase_b_b2_second_diagnostic.yaml
├── phase_b_b2_representation_probe.yaml
└── phase_b_b2_adequacy_reference_audit.yaml

tests/
├── test_semantic_relation_audit.py
├── test_representation_stage_probe.py
├── test_argument_generalization_audit.py
└── test_adequacy_reference_audit.py
```

Do not add duplicate abstractions only to match these names.

---

# 4. Required artifacts

Suggested root:

```text
runs/phase_b_b2_second_diagnostic/
```

Required outputs:

```text
config.yaml
protocol.json
system.json
summary.json
metrics.jsonl

development_sealed_comparison.json
semantic_relation_summary.json
representation_stage_summary.json
l3_failure_breakdown.json
l4_argument_breakdown.json
shift_seed24_adequacy_audit.json
reference_adequacy_summary.json
final_causal_diagnosis.json

plots/
```

Recommended plots:

```text
l3_margin_dev_vs_sealed
l3_relation_frequency_dev_vs_sealed
l3_per_relation_top1
z_task_separability_by_relation
q_task_separability_by_relation
target_rank_by_representation_stage
l4_argument_accuracy_by_operation
l4_argument_margin_by_value
shift_seed24_cumulative_support_em
shift_seed24_reference_em
plastic_decision_vs_reference_adequacy
```

---

# 5. Minimum scientific controls

## Probe isolation

Any learned evaluation probe must train only on development/probe data, never modify APC parameters, never feed predictions into runtime, report its own train/test split, and include a shuffled-label control where practical.

## Reference adequacy isolation

Large reference/query sets are evaluation-only. They may classify whether an existing primitive was truly adequate after the fact. They may not influence runtime decisions in D2.

## No sealed tuning

Training APIs should reject sealed examples. In particular, prevent accidental use of:

```text
seeds [20..24]
```

for gradient updates or model selection.

## Historical reproducibility

Where practical, reproduce summary statistics from B-C005, B-C005D, B-C005R1, B-C005R2, and B-C005G before adding new interpretations. If reproduction differs, report the discrepancy.

---

# 6. Completion criteria

The second diagnostic phase is complete only when all questions below have evidence-backed answers.

## L3

1. Was R1 development L3 easier before repair than sealed L3?
2. Are development and sealed L3 relation distributions matched?
3. Is target/competitor information present in `z_task`?
4. Is it preserved by `query_proj`?
5. Is the remaining failure in key/scoring geometry?
6. Does failure persist after difficulty/relation matching?

## L4

7. Which operations contribute most to sealed argument failure?
8. Is the failure caused by unseen/rare argument values?
9. Is the problem argument encoding, task representation, or scorer generalization?

## Adequacy

10. Is seed-24 SHIFT actually >= 0.95 on a large independent reference set?
11. Which plastic decisions are truly false versus functionally justified?
12. Does the sequential verifier behave exactly as specified?
13. Does its asymmetric early-accept rule materially affect classification?
14. Must ADR-0076's "100% finite-support variance" attribution be qualified?

## Next action

15. Which single mechanism should the next repair target first?
16. Does the hard-negative protocol need semantic-relation holdout before any new training?
17. Must the false-plastic metric definition change for future gates?

If any answer remains unsupported, mark it:

```text
UNRESOLVED
```

Do not begin the next repair merely because most questions have answers.

---

# 7. Per-task completion report

After each `B-C005D2-xxx` task report:

```text
Task ID:
Files changed:
Tests run:
Experiment commands:
Run artifacts:
Model/checkpoint state used:
Development/sealed/probe partitions:
Metrics produced:
Required questions answered:
Unresolved questions:
Historical ADRs affected:
New ADR added/required:
Downstream blocked?: YES
```

For D2-006 additionally include the full causal findings table and exactly one recommended next research action, or `UNRESOLVED` if evidence does not justify one.

---

# 8. Verification commands

Use repository-standard verification:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

Run focused tests first while iterating.

Do not claim scientific conclusions from a diagnostic run whose relevant test/protocol assertions failed.

---

# 9. STOP condition

After `B-C005D2-006`:

**STOP.**

Do not implement the recommended repair automatically.

Report findings to the user and wait for an explicit instruction defining the next repair task.

B-C006 and all Task Inference work remain blocked.
