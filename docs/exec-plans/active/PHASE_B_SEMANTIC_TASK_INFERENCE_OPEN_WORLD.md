> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](../../results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Phase B — Semantic Task Inference & Open-World Extension

> **現在地・実行順の正本:** [Phase B 再開計画](PHASE_B_RESTART.md)（2026-09-12、ADR-0127）。
> 本文は既存の仕様・作成当時の状態を保持する。現在の進捗と今回の継続指示は再開計画を参照。

## Status

**Active proposed phase**

Phase A.2 is treated as closed historical evidence.

Task IDs for this phase:

`B-C001` through `B-C014`

Implementation details and per-task acceptance criteria live in:

`docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

Scientific hypotheses, baselines, metrics, and STOP GATE thresholds live in:

`docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

Agent-specific rules live in:

`docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md`

---

## 1. Phase question

> Can APC generalize its reuse / composition / plasticity lifecycle beyond the operation families used to shape Phase A.2, remain safe under hard semantic retrieval competition, keep decision/search cost bounded as the bank grows, and infer task semantics without an explicit operation ID?

The phase deliberately decomposes this question.

It must not change task inference and controller behavior at the same time.

---

## 2. Preserved Phase A.2 mechanisms

Unless a task explicitly performs an ablation, preserve:

1. `h_content = f(content)` task-blind causal content representation.
2. Task information enters routing/controller through a separate task-side representation.
3. Existing direct primitive adequacy is tested before composition.
4. Existing composition adequacy is tested before plastic expansion.
5. Novelty means functional inadequacy, not merely latent anomaly.
6. Plastic search is compact-first.
7. Overcomplete plastic capacity is fallback, not the default.
8. Consolidation produces a separate persistent candidate.
9. Temporary capacity is released only after shadow validation.
10. Recurrence is verified in a fresh runtime with no temporary workspace.
11. Only selected primitives execute.
12. Incremental routing uses bounded replay unless a declared ablation disables it.
13. Resident parameters, active parameters, FLOPs, latency, and decision/search cost are separate metrics.

No Phase B task may silently weaken these invariants to make a new gate easier.

---

## 3. Phase decomposition

### Milestone B0 — Freeze and protocol registration

Tasks: `B-C001`, `B-C002`

Goals:

- activate Phase B documentation;
- freeze the Phase A.2 controller/plastic policy used as the explicit-TaskSpec upper bound;
- register development families and sealed evaluation families;
- implement identifiability checks before learning;
- add leak auditing for holdout data and task metadata.

No scientific credit is assigned yet.

### Milestone B1 — Unseen task-family holdout

Task: `B-C003`

This is the first and most important STOP GATE.

A task family not used to tune the controller, plastic policy, or adequacy thresholds is presented with explicit TaskSpec.

Required lifecycle:

```text
unseen family
    ↓
direct reuse inadequate
    ↓
composition inadequate
    ↓
controller selects PLASTIC_SEARCH
    ↓
compact plastic search
    ↓ if required
overcomplete fallback
    ↓
shadow validation
    ↓
promotion
    ↓
fresh-runtime recurrence
    ↓
DIRECT_REUSE
```

This gate tests controller/lifecycle generalization before Task Inference is introduced.

### Milestone B2 — Hard-negative retrieval

Tasks: `B-C004`, `B-C005`

Replace easy orthogonal distractors with a difficulty ladder:

```text
L0  orthogonal distractor
L1  random score-space competitor
L2  near-neighbor key
L3  semantically related learned primitive
L4  confusable same-family / argument-near variant
```

Measure both retrieval quality and closed-loop safety.

A router miss is not automatically a controller failure. Functional verification must still prevent confident wrong reuse when possible.

### Milestone B3 — Decision/search-cost scaling

Tasks: `B-C006`, `B-C007`

Measure:

```text
C_decision
  = C_task_inference_or_task_encoding
  + C_candidate_proposal
  + C_direct_verification
  + C_composition_search
  + C_controller
```

Do not hide support-set execution cost.

First measure the existing exhaustive behavior. If it violates the predeclared bound, add the smallest bounded proposal/search mechanism that preserves functional adequacy:

- top-k candidate proposal;
- bounded direct verification;
- bounded beam composition search;
- cached recipes only when causally justified.

Do not jump to RL or a large learned search model.

### Milestone B4 — Task Inference

Tasks: `B-C008` through `B-C011`

Task Inference is introduced only after B1-B3 pass.

Stages:

```text
B4-A  structured semantic descriptor, no operation ID
B4-B  few-shot demonstrations
B4-C  controlled natural-language instruction
```

All stages produce a task-side representation and may never inject task information into the causal content state.

Task Inference and APC controller metrics are logged separately.

### Milestone B5 — Integrated semantic/open-world loop

Tasks: `B-C012`, `B-C013`

Run K/C/N/R sequences with operation IDs absent from model input.

Then cross the two difficult axes:

- inferred task semantics;
- held-out operation family.

The final open-world gate must demonstrate that a failure can be localized to task inference, retrieval, adequacy, plastic learning, consolidation, or recurrence rather than reported as one undifferentiated end-to-end number.

### Milestone B6 — Ablations and verdict

Task: `B-C014`

Run the declared mechanistic ablations, produce the Phase B verdict, update ADRs, and move the plan to completed status only after all required artifacts exist.

---

## 4. Holdout-family policy

Phase B requires at least two levels of family isolation.

### Development families

May be used to implement and debug the benchmark infrastructure.

They may not be used as evidence for unseen-family generalization.

### Sealed evaluation family/families

May be implemented in source code, but:

- no controller architecture choice may be motivated by their measured result;
- no adequacy threshold may be tuned on them;
- no plastic budget may be increased because they fail;
- no router hyperparameter may be selected from their result.

If a sealed family is later used for tuning, it becomes a development family and a new sealed family must be designated before the next claim.

A holdout operation must also pass a **novelty validity precheck**: the existing bank/composition library must not already solve it to the declared adequacy threshold. If it does, that operation is a composition/reuse case, not a valid novel-family case.

---

## 5. Suggested structurally new family shape

The exact operation names are an implementation choice, but the primary sealed family should not be merely another permutation, tokenwise arithmetic transform, gather, count, or keyed lookup.

Prefer a same-length deterministic family with a different computational dependency structure so that decoder/output format changes are not required.

A recommended example class is a **local-neighborhood conditional transform** in which each output position depends on a relation between adjacent or nearby content positions.

This recommendation is a Phase B design choice, not a prior empirical result.

Before adopting any concrete family, run the identifiability and novelty-validity prechecks in `B-C002`.

---

## 6. Phase-level STOP GATES

### STOP GATE B1 — Unseen-family lifecycle

Task `B-C003`.

Failure blocks hard-negative/search/task-inference work that would otherwise obscure a controller/lifecycle problem.

### STOP GATE B2 — Hard-negative retrieval safety

Task `B-C005`.

Failure blocks Task Inference. First isolate candidate retrieval and functional verification.

`B-C005` failed (ADR-0075), its repair/re-gate cycle `B-C005D`/`R1`/`R2`/`G` also
failed at the sealed re-gate (ADR-0079), and the follow-on second diagnostic
phase `B-C005D2-001`..`B-C005D2-006` (ADR-0080/ADR-0081) is complete but
authorized no repair by itself. The active repair branch pursuing a passing
sealed re-gate is `docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md` /
`docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md` (tasks `B-C005R3-001`..
`B-C005R3-012`; see root `AGENTS.md`). `B-C006` onward stays BLOCKED until
that branch's sealed `B2_PROTOCOL_V2` Gate (`B-C005R3-012`) passes.

### STOP GATE B3 — Decision/search scalability

Task `B-C007`.

Failure blocks Task Inference if the current decision procedure requires unbounded bank-wide execution at the tested scale.

### STOP GATE B4-A/B/C — Task Inference ladder

Tasks `B-C009`, `B-C010`, `B-C011`.

Each stage is independently gated. Do not skip a failed easier stage and compensate with a larger model.

### FINAL STOP GATE — Crossed open-world integration

Task `B-C013`.

This is the strongest Phase B claim.

---

## 7. Phase success vocabulary

Use one of the following verdicts.

### Strong Phase B support

All required gates pass, including controlled natural-language task inference and the crossed open-world integration gate.

### Core Phase B support

Unseen-family, hard-negative, decision/search-scaling, structured-task, few-shot, and integrated open-world gates pass, but controlled natural-language does not meet its declared threshold.

This supports semantic task inference from structured descriptions/demonstrations, not natural-language generalization.

### Partial Phase B support

The controller/open-world gates pass but Task Inference stops at the structured stage or before integrated no-ID evaluation.

### Negative / blocked result

A load-bearing earlier mechanism fails and downstream work is stopped.

Never relabel a failed natural-language gate as full semantic-language success.

---

## 8. Out of scope unless promoted by a later ADR

- pretrained language models;
- RL meta-control;
- distributed training;
- custom CUDA/Triton kernels;
- arbitrary web-scale natural language;
- stochastic/partially observable environments;
- automatic architecture search;
- large semantic/vector databases;
- claims about 128 learned semantic skills unless that experiment is separately added.

Controlled natural-language instructions generated from an explicit grammar are in scope because they isolate Task Inference without introducing a pretrained model.

---

## 9. Required outputs

Every milestone run must use the repository run format and add Phase B-specific fields where relevant.

At minimum:

```text
runs/<run_name>/
├── config.yaml
├── metrics.jsonl
├── summary.json
├── system.json
├── checkpoint/
└── protocol.json
```

`protocol.json` must record:

- phase task ID;
- data split/family designation;
- whether explicit TaskSpec was model-visible;
- whether operation ID was model-visible;
- task-inference modality;
- support/inference/verification/query example counts;
- candidate-search budget;
- bank size and hard-negative level;
- holdout leak-audit result;
- git commit.

---

## 10. End-of-phase deliverables

Before Phase B is closed:

1. all task acceptance criteria have explicit PASS/FAIL;
2. negative results remain preserved;
3. all STOP GATE failures have an ADR;
4. `docs/DECISIONS_PHASE_B.md` and `docs/DECISIONS.md` are updated;
5. a Phase B result document is written under the repository's completed/result convention;
6. root `AGENTS.md` no longer points to Phase B as active after the phase is formally closed.
