# Experiment Plan — Phase B Semantic Task Inference & Open-World Extension

## 1. Purpose

This document predeclares Phase B hypotheses, controls, metrics, and STOP GATE thresholds.

The thresholds below are new Phase B design criteria. They are intentionally declared before milestone measurements so they are not retrofitted to the observed results.

Use at least **5 seeds** for every scientific STOP GATE unless a task explicitly states that it is infrastructure-only.

Report every seed.

---

# 2. Core hypotheses

## H-B1 — Unseen-family controller generalization

With explicit TaskSpec retained, the frozen APC controller/lifecycle can identify functional inadequacy for a previously held-out operation family, trigger plasticity, consolidate the learned computation, and later reuse it without adaptation.

## H-B2 — Hard-negative retrieval remains functionally safe

As semantic competitors become more similar and bank size grows, learned retrieval may degrade gradually, but functional verification prevents most wrong-reuse decisions and preserves closed-loop performance.

## H-B3 — Decision/search cost can be bounded

Candidate proposal plus bounded functional verification/composition search can preserve the decisions of the exhaustive procedure while making decision cost scale sublinearly in actual executed candidate computations over the tested bank sizes.

## H-B4A — Structured no-ID Task Inference

A task-side encoder can infer sufficient task state from a structured semantic descriptor that does not expose the canonical operation ID.

## H-B4B — Demonstration Task Inference

A task-side encoder can infer sufficient task state from identifiable few-shot input-output demonstrations.

## H-B4C — Controlled natural-language Task Inference

A task-side encoder can infer sufficient task state from held-out controlled natural-language instruction templates without using a pretrained LM.

## H-B5 — Crossed open-world integration

APC can combine inferred task semantics with an unseen operation family while retaining the K/C/N/R lifecycle and fresh-runtime recurrence.

---

# 3. Global invariants

Every experiment must preserve or explicitly ablate:

- task-blind content state;
- oracle/control separation;
- zero execution of unselected primitives;
- Stable Core freeze requirements;
- compact-first plastic policy;
- composition-before-plastic ordering;
- separate temporary and persistent parameters;
- shadow validation before release;
- fresh-runtime recurrence;
- bounded replay for incremental router updates;
- separate resident/active/FLOPs/latency metrics.

Operation name or registry membership may not be used as a novelty feature.

---

# 4. Data partitions

## 4.1 Family partitions

Create:

- `DEV_FAMILIES`
- `SEALED_FAMILIES`

A sealed family becomes invalid for final generalization evidence if any controller threshold, architecture choice, search budget, or plastic budget is changed because of its measured outcome.

If invalidated, designate a new sealed family before further claims.

## 4.2 Episode data partitions

For Task Inference experiments use disjoint roles:

```text
inference_examples
verification_examples
query_examples
```

The primary query metric must never reuse targets that were consumed by task inference or adequacy verification.

## 4.3 Identifiability

Before model scoring, verify that the declared task evidence is sufficient to distinguish the target under the benchmark's candidate universe.

Report:

`identifiable_episode_rate`

If an episode is ambiguous by construction, classify it separately.

Do not compensate for unidentifiability with a larger model.

---

# 5. Gate B1 — Unseen-family lifecycle

## Conditions

- >= 5 seeds;
- explicit TaskSpec visible;
- frozen Phase A.2 controller policy and thresholds;
- sealed operation family;
- legacy K/C/R controls interleaved.

## Novelty-validity precondition

A sealed candidate is valid as a novel operation only if neither direct reuse nor allowed composition already reaches the existing adequacy threshold on its verification set.

If it is adequate, reclassify it as K/C and choose another sealed novel operation. This is a benchmark-validity issue, not a controller failure.

## PASS criteria

Across the sealed novel episodes:

1. `plastic_trigger_rate >= 0.95`
2. `mean_final_novel_EM >= 0.95`
3. every seed's novel EM `>= 0.90`
4. exactly one promotion per successfully learned novel capability
5. `workspace_leaks == 0`
6. fresh-runtime recurrence:
   - `mean_recurrence_EM >= 0.95`
   - `adaptation_steps == 0`
   - `temporary_params == 0`
   - `bank_growth == 0`
7. legacy regression:
   - old-task mean EM drop `<= 1.0 percentage point`
   - old-routing top-1 drop `<= 1.0 percentage point`
   - false plastic on legacy K/C/R `<= 1%`

## FAIL interpretation

Localize:

- direct/composition validity;
- adequacy evidence;
- controller;
- compact plastic learning;
- fallback;
- consolidation;
- recurrence.

Do not modify Task Inference.

---

# 6. Gate B2 — Hard-negative routing

## Matrix

Bank size:

`N ∈ {16, 32, 64, 128}`

Difficulty:

`L ∈ {0, 1, 2, 3, 4}`

Primary retrieval top-k:

`k <= 5`

## PASS criteria

At `N=128`:

### Retrieval

- L0-L2 top-1 `>= 0.98`
- L3 top-1 `>= 0.95`
- L4 top-1 `>= 0.90`
- top-k inclusion `>= 0.99` for every level

### Functional safety

Across all levels:

- false functional acceptance of a wrong primitive/recipe `<= 1%`
- closed-loop EM `>= 0.95`
- false plastic on known/recurrence episodes `<= 2%`
- unselected primitive forward calls `== 0`

### Scaling report

Also report top-1/top-k/margin curves for N=16/32/64/128.

A PASS does not require constant top-1 across levels; the difficulty ladder should actually be harder.

## Diagnostic distinction

- if top-1 fails but top-k passes and functional verification remains safe: retrieval proposal is the bottleneck;
- if wrong candidates are functionally accepted: adequacy verification is the bottleneck;
- if correct candidate is found but final task fails: execution is the bottleneck.

---

# 7. Gate B3 — Decision/search-cost scaling

## 7.1 Measurement baseline

First measure the current/exhaustive procedure.

No optimization claim is allowed without this baseline.

## 7.2 Bounded-search primary configuration

Predeclared initial budget:

```text
K_direct <= 8
beam_width <= 8
max_depth <= 3
total_recipe_evaluations <= 64
```

If the repository already uses a smaller equivalent budget, keep the smaller value.

Changing these bounds after seeing sealed-family results invalidates the sealed-family claim unless a new sealed family is designated.

## 7.3 PASS criteria

Compared with exhaustive decision search:

1. best direct candidate coverage `>= 0.99`
2. best functional recipe coverage `>= 0.98`
3. controller action agreement `>= 0.97`
4. closed-loop EM drop `<= 1.0 percentage point`
5. false reuse increase `<= 1.0 percentage point`
6. false plastic increase `<= 1.0 percentage point`
7. actual candidate/recipe executions respect configured budgets
8. unselected primitive calls remain zero

Scaling criterion with fixed support-set size:

- median decision latency at N=128 `<= 2.0x` median at N=16
- p95 decision latency at N=128 `<= 3.0x` p95 at N=16

Measure and report FLOPs separately. Do not infer latency from active-parameter counts.

## Why this is a gate

If the decision procedure still executes a bank-proportional or combinatorial number of candidates at N=128, semantic Task Inference would be added on top of an unresolved scalability bottleneck.

---

# 8. Gate B4-A — Structured descriptor without operation ID

## Input restrictions

Model-visible task input must not contain:

- canonical operation enum;
- registry ID;
- oracle K/C/N/R label;
- target primitive ID.

The descriptor may contain semantic words/fields and operation arguments.

## Splits

Use descriptor-template train/dev/test splits.

At least one primary test split must use templates not seen during training.

## PASS criteria

On held-out descriptor templates:

1. operation/family evaluation accuracy `>= 0.98`
2. every applicable argument accuracy `>= 0.95`
3. router top-1 `>= 0.97`
4. controller action agreement with explicit TaskSpec upper bound `>= 0.97`
5. closed-loop EM `>= 0.97`
6. closed-loop EM gap to explicit TaskSpec upper bound `<= 2.0 percentage points`
7. mismatched descriptor control materially degrades performance:
   - at least `20 percentage points` lower task EM or action agreement
8. task-blind content invariant passes at the existing repository tolerance.

If operation labels are not directly decoded by the implementation, metric 1 may be computed from an evaluation-only linear/readout head or nearest registered task key. The evaluation mechanism must not feed back into runtime.

---

# 9. Gate B4-B — Few-shot demonstrations

## Primary setting

`8-shot` inference examples.

Diagnostics:

`2-shot`, `4-shot`

## PASS precondition

Primary identifiable subset:

`identifiable_episode_rate >= 0.95`

If lower, improve the demonstration sampling protocol rather than the model.

## PASS criteria on identifiable 8-shot episodes

1. task/family inference accuracy `>= 0.95`
2. applicable argument accuracy `>= 0.90`
3. router top-k inclusion `>= 0.97`
4. controller action agreement `>= 0.95`
5. closed-loop EM `>= 0.95`
6. gap to explicit TaskSpec upper bound `<= 3.0 percentage points`
7. shuffled-output demonstration control drops task/action performance by at least `20 percentage points`
8. wrong-task demonstration control drops task/action performance by at least `20 percentage points`
9. query targets are never consumed by task inference or functional verification.

Report ambiguous episodes separately.

---

# 10. Gate B4-C — Controlled natural-language instruction

## Scope

This is not a pretrained-LM benchmark.

Use an explicit instruction grammar with semantic template splits.

Primary test:

- unseen template family;
- no canonical operation enum token.

Secondary diagnostic:

- held-out lexical alias where the operation remains identifiable.

## PASS criteria

On held-out instruction templates:

1. task/family inference accuracy `>= 0.90`
2. applicable argument accuracy `>= 0.85`
3. router top-k inclusion `>= 0.95`
4. controller action agreement `>= 0.90`
5. closed-loop EM `>= 0.90`
6. gap to explicit TaskSpec upper bound `<= 5.0 percentage points`
7. mismatched instruction control degrades performance by at least `20 percentage points`
8. instruction-removed control is materially below the primary model.

If this gate fails after B4-A/B4-B pass, use the verdict category **Core Phase B support** rather than claiming natural-language success.

---

# 11. B-C012 integrated known-family no-ID loop

## Primary modality

Use the strongest passed non-language modality, normally 8-shot demonstrations.

## Episode mix

Use at least 40 episodes per seed with K/C/N/R all represented and recurrence appearing only after successful consolidation.

Recommended minimum proportions:

- K: >= 25%
- C: >= 20%
- N: >= 15%
- R: >= 15%

The remaining episodes may be allocated to balance families.

## PASS criteria

Across >=5 seeds:

- K EM `>= 0.97`
- C EM `>= 0.95`
- N final EM `>= 0.93`
- R EM `>= 0.93`
- N plastic trigger `>= 0.95`
- R direct reuse `>= 0.95`
- K/C false plastic `<= 2%`
- reconsolidation on valid R `<= 2%`
- old-task mean degradation `<= 2 percentage points`
- workspace leaks `== 0`
- unselected primitive calls `== 0`

Also report task-inference failure rate independently.

---

# 12. Final Gate — inferred semantics × sealed unseen family

## Conditions

- >=5 seeds;
- controller/search/plastic policy frozen before observing final sealed results;
- no operation ID visible;
- primary Task Inference modality already passed its own gate;
- family is sealed and passes novelty-validity precheck.

## PASS criteria

1. identifiable episode rate `>= 0.95`
2. task-inference success `>= 0.90`
3. novel-family plastic trigger `>= 0.90`
4. novel-family final EM `>= 0.90`
5. successful capabilities are promoted once
6. workspace leaks `== 0`
7. fresh-runtime recurrence:
   - recurrence EM `>= 0.90`
   - adaptation steps `== 0`
   - temporary params `== 0`
   - bank growth `== 0`
8. legacy K/C mean EM drop `<= 2 percentage points`
9. false functional reuse `<= 2%`
10. every failed episode is assigned a first-failure stage.

This threshold is intentionally lower than Gate B1 because task inference and unseen-family generalization are being crossed, but it still requires reliable lifecycle behavior.

---

# 13. Baselines and controls

Use applicable subsets of:

## Task representation baselines

- explicit TaskSpec upper bound;
- descriptor removed;
- mismatched descriptor;
- demonstrations shuffled;
- demonstrations from wrong task;
- instruction removed;
- mismatched instruction.

## Retrieval/search baselines

- orthogonal distractors;
- hard-negative ladder;
- exhaustive candidate verification;
- bounded top-k verification;
- exhaustive composition;
- bounded beam composition.

## Controller/novelty ablations

- support functional score removed;
- composition evidence removed;
- router-confidence-only novelty;
- recurrence-similarity removed when relevant.

## Plasticity baselines

- compact-only;
- compact-first + fallback;
- always-overcomplete only as a resource/behavior comparison.

## Continual-routing baseline

- bounded replay;
- new-class-only update as the destructive control if still meaningful.

---

# 14. Reporting rules

For every gate report:

- mean;
- standard deviation;
- per-seed values;
- minimum/worst seed where relevant;
- exact number of episodes/examples;
- task family split;
- bank size;
- hard-negative level;
- support/inference/verification/query counts;
- search budget;
- model-visible task fields;
- git commit;
- wall-clock;
- device;
- peak VRAM;
- active/resident/temporary parameter counts;
- measured latency;
- FLOPs estimate;
- primitive forward-call counts.

Do not report only aggregate accuracy.

---

# 15. Interpretation rules

## A. Task Inference failure is not controller failure

If the explicit TaskSpec upper bound passes but inferred-task mode fails before routing, classify as Task Inference failure.

## B. Retrieval failure is not adequacy failure

If the correct candidate is absent from proposal top-k, classify retrieval first.

## C. Wrong functional acceptance is more severe than top-1 loss

If a wrong candidate is proposed but rejected by functional verification, the system may still be safe.

## D. Holdout validity precedes novelty claims

If the existing composition library already solves the holdout task, do not call it novel.

## E. Negative results remain first-class outputs

Do not raise thresholds after observing a failure.

Do not increase model size without a specific mechanism-level hypothesis.
