# Codex Tasks — Phase A.2 Autonomous Controller & Scaling

Execute exactly one task at a time.

---

## A2-C001 — ADR-0061 scope and metric audit [PASSED - ADR-0062]

### Goal

Clarify what B008 established without rewriting history.

### Work

Add an interpretive ADR/addendum stating:

- routing is under explicit TaskSpec,
- B008 covers the known/consolidated 10-operation universe,
- 90.11% is primitive-bank active-parameter savings,
- total FLOPs/latency remain unmeasured.

### Acceptance

Historical artifacts unchanged.

No rerun required.

---

## A2-C002 — Controller and compute instrumentation [PASSED - ADR-0063]

### Goal

Create common instrumentation before new scientific runs.

### Add

Per episode/log:

- controller action,
- proposed primitive ID,
- composition recipe,
- support-set direct score,
- support-set composition score,
- plastic trigger,
- bank size before/after,
- router version,
- adaptation steps,
- temporary params,
- selected/unselected forward calls.

Compute accounting:

- resident/active params,
- router/core/primitive FLOPs estimate,
- latency hooks,
- peak memory.

### Acceptance

Tiny deterministic tests can rederive all accounting.

No controller-learning claim yet.

---

## A2-C003 — Incremental router update gate [PASSED - ADR-0064]

### Goal

Test class-incremental routing after new primitive installation.

### Conditions

Compare:

- R0 full retrain upper bound,
- R1 naive new-class-only update,
- R2 bounded replay/prototype update — primary.

### Work

Grow semantic bank from 10 to at least 12, preferably 16, using successfully consolidated new executable operations.

After every insertion measure:

- old-class top-1,
- new-class top-1,
- worst-class drop,
- recurrence routing,
- forward-call sparsity.

### Primary acceptance R2

- new-class top-1 >=0.95
- old-class mean drop <=0.02
- worst old-class drop <=0.05
- overall top-1 >=0.95
- unselected forward calls ==0
- >=5 seeds

**STOP GATE.**

---

## A2-C004 — Bank competition and routing scaling [PASSED - ADR-0065]

### Goal

Measure routing robustness as resident bank grows.

### Sizes

`10, 16, 32, 64, 128`

Use real semantic entries where available.

Larger routing-only sizes may use clearly labeled frozen matched-scale distractor entries.

### Metrics

- top-1/top-k
- distractor false-selection rate
- recurrence accuracy
- resident/active params
- router cost
- selected/unselected calls

### Acceptance at N=128 routing-only scale

- known-task top-1 >=0.95
- distractor false selection <=0.05
- top-1 selected primitive only executes
- unselected calls ==0

Do not call this 128-semantic-task continual learning.

---

## A2-C005 — Functional adequacy evidence interface

### Goal

Build the runtime evidence used to choose DIRECT / COMPOSE / PLASTIC.

### Work

For each support set compute:

- best direct primitive EM/loss
- direct margin
- best composition EM/loss/depth
- direct-vs-composition improvement
- router confidence/margin
- optional recurrence similarity

### Leakage tests

The interface must not expose:

- oracle operation class,
- hidden program,
- oracle metadata,
- held-out target.

### Acceptance

K/C/N/R fixtures produce deterministic evidence.

No learned controller yet.

---

## A2-C006 — Learned adequacy / novelty controller

### Goal

Learn action selection from functional evidence.

### Outputs

- DIRECT_REUSE
- COMPOSE
- PLASTIC_SEARCH

### Runtime inputs

Evidence from C005 only.

Do not feed oracle K/C/N/R labels or registry-membership truth.

### Decision evaluation

Use >=5 seeds and held-out task instances/families where practical.

### Acceptance

- K/C vs N AUROC >=0.90
- K false plastic <=0.10
- C false plastic <=0.10
- N plastic trigger >=0.90
- R direct reuse >=0.90
- composition action accuracy >=0.85

Freeze thresholds/model before final run.

**STOP GATE.**

---

## A2-C007 — Compact-first plastic lifecycle policy

### Goal

Integrate the B007X lesson into runtime plasticity.

### Policy

```text
direct insufficient
 -> composition insufficient
 -> compact plastic search
 -> if success: consolidate
 -> if fixed-budget failure: optional overcomplete fallback
 -> consolidate if successful
```

### Controls

Report separately:

- compact success cases,
- compact failure cases,
- fallback invoked,
- fallback success/failure.

Do not assume overcomplete fallback is superior.

### Acceptance

Mechanical lifecycle is correct:

- no plastic before direct/composition inadequacy
- exactly one promotion per successful N
- workspace returns to zero
- no K/C/R promotion

---

## A2-C008 — Full sequential K/C/N/R autonomous stream

### Goal

Run one online stream where the learned controller autonomously decides each episode.

### Minimum stream per seed

>=40 episodes:

- >=10 K
- >=10 C
- >=6 N
- >=6 R

### Runtime oracle prohibition

No oracle action labels.

Oracle K/C/N/R categories may be logged only for evaluation.

### Primary acceptance

K:
- EM >=0.95
- false plastic <=0.10

C:
- EM >=0.90
- composition action >=0.85
- expansion <=0.10

N:
- final EM >=0.90
- plastic trigger >=0.90

R:
- EM >=0.95
- direct reuse >=0.90
- no new consolidation >=0.90

Global:
- old-task degradation <=0.02
- no temporary leak
- >=5 seeds

**STOP GATE.**

---

## A2-C009 — Repeated bank-growth stress

### Goal

Test repeated novel insertions and incremental routing interference.

### Work

Starting from the validated bank, cause multiple successful N -> consolidation cycles.

Preferred semantic bank target:

`10 -> 12 -> 14 -> 16`

After each insertion:

- incremental router update only,
- old routing evaluation,
- old task evaluation,
- recurrence evaluation.

### Acceptance

At final semantic bank:

- overall routing >=0.95
- old routing mean drop <=0.02
- canonical performance drop <=0.02
- consolidated-task drop <=0.02
- recurrence reuse >=0.90

**STOP GATE.**

---

## A2-C010 — End-to-end compute and latency scaling

### Goal

Replace active-parameter proxy with actual compute measurements.

### Sizes

`10, 16, 32, 64, 128`

### Compare

- sparse top-1 APC path,
- executable dense-all-primitives baseline.

### Report

- primitive active-param savings
- total estimated FLOPs
- measured median/p95 latency
- throughput
- peak GPU memory
- router overhead
- forward-call counts

### Hard acceptance

- unselected calls ==0
- accuracy >=0.95 on the routing benchmark

### Practical target, not scientific STOP

At N=128:

`sparse median latency <= 0.30 * dense median latency`

If missed, report the actual ratio.

---

## A2-C011 — Controller ablations and failure analysis

### Required ablations

- no composition evidence
- no support-set functional score
- router confidence only
- no bounded replay on router growth
- no recurrence similarity
- compact-only plastic policy
- always-overcomplete plastic policy

### Goal

Identify which mechanisms actually prevent:

- false expansion,
- missed novelty,
- routing forgetting.

No new architecture features.

---

## A2-C012 — Phase A.2 final audit

### Output

Create:

`docs/results/PHASE_A2_AUTONOMOUS_CONTROLLER_RESULT.md`

### Required sections

1. ADR-0061 scope clarification
2. incremental routing
3. bank-size routing scaling
4. learned adequacy/novelty controller
5. compact-first plastic policy
6. K/C/N/R sequential stream
7. repeated bank-growth stress
8. FLOPs/latency scaling
9. ablations
10. remaining limitations
11. next-phase recommendation

### Allowed conclusion levels

#### Strong autonomous-controller support

All STOP gates pass.

#### Partial support

Routing works but novelty/controller or growth fails.

#### Sparse-inference only

Fixed-bank routing works but continual control does not.

#### Inconclusive

Experimental confounds prevent a clean conclusion.

### Rule

Audit only.

Do not start task-inference-from-language work in this task.
