> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Codex Task Queue — Phase B Semantic Task Inference & Open-World Extension

> **現在地・実行順の正本:** [Phase B 再開計画](exec-plans/active/PHASE_B_RESTART.md)（2026-09-12、ADR-0127）。
> 本文は既存の仕様・作成当時の状態を保持する。現在の進捗と今回の継続指示は再開計画を参照。

## Operating rule

Implement **only the task explicitly requested by the user**.

Do not automatically continue to the next `B-Cxxx` task.

For every task:

- inspect adjacent code and tests before adding a new abstraction;
- preserve the task-blind causal content path;
- add focused CPU-friendly tests;
- run the smallest relevant tests while iterating;
- run repository verification before completion;
- save milestone run artifacts;
- report acceptance criteria as explicit PASS / FAIL;
- stop on a failed STOP GATE.

The scientific thresholds referenced below are defined in:

`docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

---

# B-C001 — Activate Phase B documentation and freeze the baseline protocol

## Goal

Make Phase B the repository's active research phase without changing model behavior.

## Required changes

1. Add the Phase B documents from this pack.
2. Update root `AGENTS.md` active-phase navigation.
3. Update `README.md` project status.
4. Add `docs/DECISIONS_PHASE_B.md` and index it from `docs/DECISIONS.md`.
5. Define a serializable `PhaseBProtocol` or equivalent config structure containing:
   - family split identifier;
   - explicit TaskSpec visibility;
   - operation-ID visibility;
   - task-inference modality;
   - support/inference/verification/query counts;
   - hard-negative level;
   - bank size;
   - candidate-search budget.
6. Register the frozen Phase A.2 controller/plastic-policy configuration used as Phase B upper bound.

## Forbidden shortcuts

- no controller retraining;
- no threshold retuning;
- no new task family results used yet;
- no root package-layout change.

## Tests

- protocol serialization round-trip;
- config rejects impossible combinations;
- default Phase B explicit upper-bound mode reproduces the intended A2-style task-visible path.

## Acceptance

PASS only if documentation navigation and protocol/config tests pass.

No scientific claim is made.

---

# B-C002 — Holdout-family registry, identifiability checks, and leak audit

## Goal

Create development and sealed evaluation family partitions before any Phase B model tuning.

## Implementation

Add deterministic generators for:

1. development-only structurally new operations;
2. at least one sealed evaluation family;
3. recurrence instances for consolidated holdout operations.

Prefer same-length deterministic transforms to avoid changing the output interface.

Add preflight checks:

### A. Identifiability

For each task specification/modality, verify that the provided evidence can identify the target mapping under the declared support/example count.

For demonstration-based tasks, detect collisions where two candidate tasks are consistent with the same demonstration set.

### B. Novelty validity

Evaluate the frozen existing primitive bank and allowed composition depth on the candidate holdout tasks.

A task that already reaches the adequacy threshold is not a valid novel holdout.

### C. Leak audit

Assert that sealed family identifiers, oracle operation IDs, target family labels, and registry membership are absent from:

- controller inference features;
- task-inference model input when forbidden;
- router inference input when forbidden;
- plastic learner input except legitimate examples;
- saved examples used for development tuning.

## Suggested files

Prefer:

```text
src/apc/environments/holdout_families.py
src/apc/evaluation/holdout_protocol.py
tests/test_phase_b_holdout_protocol.py
```

or the closest existing equivalents.

## Acceptance

PASS if:

- generators are deterministic under seed;
- development/sealed partitions are disjoint;
- leak audit is zero;
- at least one sealed operation passes novelty-validity precheck;
- ambiguous demonstration sets are detected rather than silently treated as model errors.

No STOP GATE yet.

---

# B-C003 — STOP GATE B1: explicit-TaskSpec unseen-family lifecycle [PASSED - ADR-0074]

## Goal

Test whether the frozen APC controller/lifecycle generalizes to a genuinely held-out operation family before adding Task Inference.

## Fixed conditions

- explicit TaskSpec remains model-visible;
- Phase A.2 controller architecture and thresholds remain frozen;
- Stable Core and persistent bank obey their existing freeze rules;
- compact-first plastic policy remains unchanged;
- overcomplete capacity remains fallback only.

## Episode sequence

For each sealed novel operation:

```text
first presentation
  → direct verification
  → composition verification
  → controller action
  → plastic learning if inadequate
  → consolidation
  → shadow validation
  → promotion
  → workspace release

later presentation in fresh runtime
  → installed primitive/recipe
  → no adaptation
```

Also interleave legacy K/C/R episodes to measure regressions.

## Required metrics

- best direct EM/loss before plasticity;
- best composition EM/loss before plasticity;
- controller PLASTIC trigger;
- compact/fallback path used;
- plastic final EM;
- promotion count;
- workspace leak count;
- fresh-runtime recurrence EM;
- recurrence adaptation steps;
- recurrence temporary parameters;
- recurrence bank growth;
- old-task EM/routing degradation;
- false plastic on legacy K/C/R.

## STOP GATE

Use Experiment Plan Gate B1.

**Verification Status: PASSED (ADR-0074)**
- Seeds evaluated: 5 seeds (`(0, 1, 2, 3, 4)`) on CUDA
- Plastic Trigger Rate: **100.00%** ($\ge 95.0\%$) — PASS
- Mean Final Novel EM: **97.66%** ($\ge 95.0\%$) — PASS
- Min Seed Novel EM: **95.31%** ($\ge 90.0\%$) — PASS
- 1:1 Promotion: **10/10** — PASS
- Workspace Leaks: **0** ($== 0$) — PASS
- Fresh Recurrence EM: **97.19%** ($\ge 95.0\%$) — PASS
- Recurrence Adaptation Steps: **0** ($== 0$) — PASS
- Recurrence Temporary Parameters: **0** ($== 0$) — PASS
- Recurrence Bank Growth: **0** ($== 0$) — PASS
- Max Old-Task EM Drop: **0.00 pp** ($\le 1.0\,\text{pp}$) — PASS
- Max Old-Routing Top-1 Drop: **0.00 pp** ($\le 1.0\,\text{pp}$) — PASS
- Legacy False Plastic: **0.00%** ($\le 1.0\%$) — PASS
- Artifacts: `runs/phase_b_unseen_family_lifecycle_gate/` (`report.json`, `summary.json`, `report.md`)

---

# B-C004 — Hard-negative bank construction and diagnostic matrix

## Goal

Implement the hard-negative difficulty ladder without yet changing the learned router.

## Levels

```text
L0 orthogonal
L1 random score-space
L2 near-neighbor
L3 semantically related learned primitive
L4 same-family / argument-confusable competitor
```

## Requirements

- difficulty generation must be deterministic;
- competitor construction must never alter primitive function;
- L3/L4 must be based on semantic/query proximity, not merely a renamed random vector;
- parameterized primitive families must not be duplicated per argument solely to create L4;
- bank sizes: 16, 32, 64, 128;
- all unselected primitives must retain zero forward calls.

## Diagnostics

Produce a matrix over:

- bank size;
- difficulty level;
- top-1;
- top-k;
- positive/negative margin;
- candidate rank;
- false reuse;
- false plastic;
- closed-loop EM;
- primitive forward-call count.

## Acceptance

Infrastructure PASS if level ordering is measurable, deterministic, and leak-free.

Scientific gate is B-C005.

**Verification Status: Infrastructure PASS (B-C004 complete)**
- Matrix: 80 cells across one deterministic GPU seed: `N ∈ {16, 32, 64, 128}` × `L0`–`L4` × four parameterized target operations.
- Deterministic, non-mutating construction: PASS; the frozen router and every resident primitive parameter were unchanged after every cell.
- Leak audit: PASS; evaluator-only difficulty/provenance metadata never enters the router input.
- Sparse execution: PASS; unselected primitive forward calls were zero in every cell.
- Artifacts: `runs/phase_b_hard_negative_diagnostic/` (`config.yaml`, `metrics.jsonl`, `protocol.json`, `report.json`, `report.md`, `summary.json`, `system.json`).
- This is infrastructure evidence only. Gate B2's retrieval and functional-safety thresholds remain unmeasured until B-C005.

---

# B-C005 — STOP GATE B2: hard-negative routing and functional safety

## Goal

Test retrieval under realistic semantic competition while preserving functional verification as the safety layer.

## Baselines

1. easy orthogonal distractors;
2. frozen learned router with hard negatives;
3. router + functional verification;
4. optional oracle candidate ranking upper bound.

Do not retrain a larger router before measuring the frozen router.

## Required separation

Report both:

- retrieval failure: correct candidate not ranked sufficiently high;
- safety failure: wrong candidate is accepted functionally and executed as if adequate.

Do not collapse them into one "accuracy" number.

## STOP GATE

Use Experiment Plan Gate B2.

If the frozen router fails retrieval but functional verification remains safe, the next investigation may improve candidate proposal while keeping controller thresholds fixed.

If functional verification accepts wrong computations, stop and repair adequacy evidence before any Task Inference work.

## Verification Status: STOP GATE B2 FAIL (B-C005 complete)

- Matrix: 400 seeded CUDA cells across five seeds: `N ∈ {16, 32, 64, 128}` × `L0`–`L4` × four parameterized target operations.
- Retrieval at `N=128`: top-k inclusion was `1.000` at every level, but top-1 was `1.000` (L0), `1.000` (L1), `0.866` (L2), `0.662` (L3), and `0.504` (L4). L2–L4 miss their predeclared B2 thresholds.
- Functional safety: PASS in isolation. Wrong-candidate functional acceptance was `0.000`; the logical L4 wrong-argument candidate was verified with its wrong argument rather than its shared primitive ID alone; unselected physical primitive calls were zero in every cell; the frozen router and primitives were unchanged; and the evaluator-only hard-negative metadata did not enter routing.
- Closed-loop safety: FAIL. Mean closed-loop EM was `0.959 >= 0.950`, but mean false plastic on known episodes was `0.0375 > 0.02`.
- Artifacts: `runs/phase_b_hard_negative_safety_gate/` (`config.yaml`, `metrics.jsonl`, `protocol.json`, `report.json`, `report.md`, `summary.json`, `system.json`).
- Consequence: STOP GATE B2 blocks B-C006 onward, including all Task Inference work. The next investigation must isolate frozen candidate proposal and the false-plastic/adequacy path without changing thresholds from this failed measurement.

> **Repair branch update (2026-09-06):** the B-C005G sealed re-gate failure above
> (ADR-0079) was followed by a second diagnostic phase, `B-C005D2-001`..`B-C005D2-006`
> (ADR-0080: benchmark-generation nondeterminism + SHIFT seed-24 primitive
> inadequacy; ADR-0081: integrated causal diagnosis, recommending semantic-relation
> holdout redesign as the first repair step). That diagnostic phase is now complete
> and no repair was authorized by it. The active repair task queue is
> `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md` (tasks `B-C005R3-001`..`B-C005R3-012`);
> see the pointer in root `AGENTS.md`. `B-C006`'s BLOCKED status below is
> superseded in detail by that document but not lifted: `B-C006` stays blocked
> until a new sealed `B2_PROTOCOL_V2` Gate (`B-C005R3-012`) passes.

---

# B-C006 — Measure decision/search cost with the current algorithm

## Status: BLOCKED by B-C005G (ADR-0079); see B-C005R3 series (ADR-0080/ADR-0081) for the active repair branch

The B-C005G sealed re-gate failed. Do not begin B-C006 until a subsequent repair
cycle passes on a newly designated sealed partition.

## Goal

Measure the cost that Phase A.2 did not establish: the cost of deciding what computation to use.

## Conditions

Bank sizes:

`16, 32, 64, 128`

Composition depth:

- depth 1/direct;
- depth 2 primary;
- depth 3 diagnostic where supported.

Measure separately:

- task encoding;
- candidate retrieval;
- direct candidate verification;
- composition search;
- controller MLP;
- primitive execution used only for final action;
- total decision latency;
- total end-to-end latency.

## Required metrics

- median and p95 CUDA latency where CUDA is used;
- CPU timing for lightweight test coverage;
- analytical/estimated FLOPs;
- number of primitive forward calls during decision;
- number of candidate primitives verified;
- number of candidate recipes executed;
- support examples consumed;
- peak VRAM;
- resident vs active parameters.

## Acceptance

This is a measurement task.

PASS if the profiler is correct and produces the full scaling report.

Do not optimize in this task.

---

# B-C007 — STOP GATE B3: bounded adequacy/composition search

## Goal

If B-C006 shows bank-wide or combinatorial decision growth, replace it with the smallest bounded search that preserves functional decisions.

## Preferred mechanism

Use existing retrieval scores to propose a bounded candidate set.

Recommended defaults to test:

```text
direct candidate budget K_direct <= 8
composition beam width <= 8
max tested depth <= 3
total recipe-evaluation budget <= 64 per episode
```

Budgets are configuration, not hidden constants.

A cached recipe may be reused only after its functional identity and task conditions are validated.

## Baseline

The existing exhaustive decision procedure is the oracle-like search reference.

## Required metrics

- best-candidate coverage versus exhaustive search;
- best-recipe coverage versus exhaustive search;
- controller action agreement;
- closed-loop EM gap;
- false reuse / false plastic;
- decision latency scaling;
- candidate execution count;
- unselected primitive calls.

## STOP GATE

Use Experiment Plan Gate B3.

If FAIL:

- do not proceed to Task Inference;
- do not hide search cost behind asynchronous/offline preprocessing;
- do not increase model capacity first;
- isolate proposal recall versus functional verification versus composition beam failure.

---

# B-C008 — Task Inference interface, split discipline, and leakage controls

## Goal

Introduce a task-side inference abstraction without changing controller semantics.

## Interface

Add an abstraction equivalent to:

```text
TaskObservation
  ├── structured_descriptor?
  ├── inference_examples?
  ├── language_instruction?
  └── metadata allowed only for evaluation

TaskInference
  ↓
z_task / inferred task state
  ↓
existing router/controller
```

Do not require exactly these class names if equivalent abstractions already exist.

## Data separation

Demonstration-based episodes must distinguish:

1. `inference_examples`
   - used to infer task semantics;

2. `verification_examples`
   - used by functional adequacy checks;

3. `query_examples`
   - used for final task score.

Do not silently use query targets to infer the task.

## Controls

Add assertions/tests that:

- canonical operation enum/ID is absent when a no-ID mode is active;
- oracle K/C/N/R labels are absent;
- `h_content` remains invariant to task observation;
- task inference output can change while content encoding does not;
- ground-truth identity exists only in supervision/evaluation paths where allowed.

## Acceptance

PASS if interface and leakage tests pass.

No scientific Task Inference claim yet.

---

# B-C009 — STOP GATE B4-A: structured descriptor without operation ID

## Goal

Infer task semantics from a structured semantic description with no explicit operation ID.

## Primary design

Use a deterministic descriptor grammar.

The descriptor may express the operation semantics and arguments but must not contain the canonical operation enum/token used by the existing router registry.

Include held-out descriptor templates so the model cannot pass by memorizing one fixed string per operation.

## Baselines

1. explicit TaskSpec upper bound;
2. no-operation-ID structured descriptor;
3. shuffled/mismatched descriptor negative control;
4. descriptor removed.

## Metrics

- inferred operation/family accuracy for evaluation only;
- argument accuracy where applicable;
- router top-1/top-k;
- controller action agreement with explicit upper bound;
- closed-loop EM;
- task/content invariance;
- upper-bound gap.

## STOP GATE

Use Experiment Plan B4-A.

Do not proceed to few-shot if this easier representation cannot be learned.

---

# B-C010 — STOP GATE B4-B: few-shot demonstration task inference

## Goal

Build task representation only from input-output demonstrations.

## Primary setting

Use 8 inference demonstrations per episode as the primary gate.

Also report 2-shot and 4-shot curves as diagnostics if computationally cheap.

## Identifiability

Before model evaluation, compute whether the demonstration set uniquely identifies the target among the declared candidate task universe.

Ambiguous episodes must be:

- regenerated under the deterministic protocol, or
- marked ambiguous and excluded from the primary identifiable subset while being reported separately.

Do not score an unidentifiable episode as an ordinary model failure.

## Baselines

1. explicit TaskSpec upper bound;
2. few-shot inference;
3. shuffled output demonstrations;
4. demonstrations from a wrong task;
5. insufficient-shot diagnostic.

## Metrics

- identifiable episode rate;
- task/family inference accuracy for evaluation;
- argument accuracy;
- router top-k;
- action agreement;
- closed-loop EM;
- support/query separation;
- upper-bound gap.

## STOP GATE

Use Experiment Plan B4-B.

---

# B-C011 — STOP GATE B4-C: controlled natural-language instruction

## Goal

Infer task semantics from natural-language-like instructions without adding a pretrained LM.

## Scope

Use a controlled instruction generator with:

- multiple paraphrase templates per semantic operation;
- held-out template families;
- held-out lexical aliases where practical;
- argument realization variation;
- no canonical operation enum token.

Examples may be English-like, but the claim is **controlled natural-language task inference**, not general language understanding.

## Baselines

1. explicit TaskSpec upper bound;
2. training-template language;
3. held-out-template language;
4. mismatched instruction negative control;
5. instruction removed.

## Metrics

Same downstream separation as B-C009, plus:

- seen-template vs held-out-template gap;
- seen-lexeme vs held-out-lexeme gap where implemented.

## STOP GATE

Use Experiment Plan B4-C.

Failure here does not erase earlier open-world/controller results. It changes the Phase B verdict category.

---

# B-C012 — Integrated no-operation-ID K/C/N/R sequential loop

## Goal

Run the autonomous lifecycle with explicit operation IDs absent from model input.

## Primary modality

Use the best **already-passed** task-inference modality without changing its architecture for this task.

Prefer few-shot demonstrations as the primary integrated modality because they specify a function without requiring a language model.

Structured descriptors may be a secondary upper bound.

## Sequence requirements

Include:

- familiar K;
- known composition C;
- novel operation N;
- later recurrence R.

No oracle K/C/N/R action label is visible at runtime.

## Metrics

For each category and globally:

- task EM;
- controller action;
- false plastic;
- composition selection;
- promotion count;
- recurrence direct reuse;
- old-task degradation;
- workspace leaks;
- task-inference correctness;
- retrieval correctness;
- functional adequacy correctness.

## Acceptance

Use the integrated-known-family criteria in the Experiment Plan.

This is not yet the final crossed holdout gate.

---

# B-C013 — FINAL STOP GATE: inferred-semantics × unseen-family open-world integration

## Goal

Cross the two previously isolated axes:

1. task semantics are inferred without operation ID;
2. the novel operation belongs to a sealed holdout family.

## Protocol

The controller/plastic/search policy is frozen from earlier gates.

Do not tune on the sealed family after observing the result.

Primary modality:

- few-shot demonstrations if B4-B passed.

Secondary modalities:

- structured descriptor;
- controlled natural language if B4-C passed.

## Required localization

For every failed episode, classify the first failing stage:

```text
TASK_INFERENCE
RETRIEVAL
DIRECT_ADEQUACY
COMPOSITION_SEARCH
CONTROLLER
PLASTIC_LEARNING
CONSOLIDATION
RECURRENCE
```

Do not report only end-to-end failure.

## FINAL STOP GATE

Use Experiment Plan Final Gate.

This gate determines whether the strongest Phase B open-world claim is supported.

---

# B-C014 — Mechanistic ablations, final verdict, and archival transition

## Goal

Close Phase B scientifically and operationally.

## Required ablations

At minimum:

1. remove/disable functional support verification;
2. disable composition evidence;
3. replace hard negatives with easy orthogonal distractors;
4. remove bounded replay during semantic bank updates used in Phase B;
5. use unbounded exhaustive search versus bounded search;
6. collapse `inference_examples` and `verification_examples` as a contamination control;
7. for Task Inference, mismatch task observation to content examples;
8. compact-only versus compact-first + fallback on holdout novel tasks.

Only run an ablation when the corresponding mechanism exists in the final path.

## Required result document

Write a Phase B result summary containing:

- question;
- protocol;
- all gate outcomes;
- per-seed metrics;
- negative results;
- ablations;
- compute/search scaling;
- exact scope of supported claims;
- unresolved issues for the next phase.

## Verdict vocabulary

Use only:

- Strong Phase B support;
- Core Phase B support;
- Partial Phase B support;
- Negative / blocked result.

## Completion

After the result document and ADRs are complete:

- move/archive the Phase B execution plan according to repository convention;
- update root `AGENTS.md` to the next phase only when the user explicitly starts that phase.
