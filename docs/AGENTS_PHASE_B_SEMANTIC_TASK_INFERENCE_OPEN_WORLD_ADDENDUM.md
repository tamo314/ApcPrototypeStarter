> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AGENTS Phase B Addendum — Semantic Task Inference & Open-World Extension

## Authority

This file adds rules for Phase B research implementation and execution. Routine
documentation/tooling changes use the reading and verification scope in
[root AGENTS.md](../AGENTS.md).

If there is a conflict:

1. user instruction for the current task;
2. root `AGENTS.md`;
3. [restart plan](exec-plans/active/PHASE_B_RESTART.md) for current status and
   continuation scope, with the applicable Phase B task/experiment contracts;
4. this addendum;
5. historical documents.

Do not let historical Phase A/A.1/A.2 plans override the active Phase B plan.

---

## 1. Stay within authorized research scope

Implement the requested `B-Cxxx` task or explicitly authorized continuation scope.
Proceed through already authorized tasks when their prerequisites pass without
requesting approval again at each task boundary.

Completion alone does not authorize additional research interventions.

If a STOP GATE fails, stop dependent work.

---

## 2. Do not change Task Inference and controller policy together

This is the most important Phase B isolation rule.

Before `B-C008`, Task Inference must remain explicit TaskSpec.

During `B-C009` through `B-C011`, the controller/search/plastic policy is an upper-bound-fixed downstream consumer unless the task explicitly declares a separate ablation.

If a Task Inference mode fails:

- verify explicit TaskSpec upper bound first;
- inspect task inference;
- do not retune controller thresholds to compensate.

---

## 3. Preserve the causal content invariant

Always preserve:

`h_content = f(content)`

A descriptor, demonstration set, or natural-language instruction must not alter the content encoder's primitive input state.

Task observations may affect only the task-side representation, routing, controller decisions, and explicit primitive arguments through the authorized path.

Add invariance tests for every new task-inference modality.

---

## 4. Sealed-family rule

A sealed family is not secret source code; it is a **no-tuning evaluation partition**.

After measuring a sealed family:

- do not change thresholds because of that result;
- do not enlarge plastic capacity because of that result;
- do not change search budgets because of that result;
- do not change router architecture because of that result.

If a change is scientifically necessary, record the result, reclassify that family as development, add an ADR, designate a new sealed family, and rerun only the new final gate later.

---

## 5. Identifiability before capacity

For structured descriptors, demonstrations, and controlled language:

1. verify the task evidence is sufficient to identify the target;
2. detect collisions/ambiguity;
3. report ambiguous cases;
4. only then evaluate model capacity.

Do not respond to an unidentifiable setup by:

- increasing hidden size;
- adding layers;
- adding a pretrained LM;
- adding RL;
- leaking task labels.

---

## 6. Demonstration split rule

For few-shot inference use three semantic roles:

```text
inference_examples
verification_examples
query_examples
```

Do not use query targets for task inference.

Do not quietly use the same examples for all three roles in the primary gate.

A contamination ablation may intentionally collapse roles, but it must be labeled as an ablation and never as the primary result.

---

## 7. Functional novelty remains authoritative

Phase B does not redefine novelty as:

- unfamiliar descriptor;
- low task-encoder confidence;
- large embedding distance;
- unknown operation word.

Novelty remains primarily:

> the existing primitive/composition library is functionally inadequate on the allowed verification evidence.

Router confidence and task distance may propose candidates or serve as secondary evidence.

They must not replace functional verification without a declared experiment.

---

## 8. Hard-negative safety rule

When hard negatives are present, distinguish:

- candidate proposal quality;
- candidate ranking;
- functional acceptance;
- final controller action.

A wrong top-1 retrieval that is rejected by adequacy verification is not the same failure as executing the wrong computation.

Log both.

---

## 9. Search-budget rule

Decision/search cost is now a first-class resource.

Track:

- candidate primitives proposed;
- candidate primitives executed for verification;
- candidate recipes generated;
- candidate recipes executed;
- composition depth;
- support examples used;
- decision latency;
- final execution latency.

A "sparse" final primitive call does not justify a sparse-compute claim if the controller evaluated the entire bank first.

Do not hide search work in preprocessing that is still required per novel task.

---

## 10. Natural-language scope

Do not add a pretrained language model in Phase B unless the user explicitly changes scope.

The Phase B language gate uses controlled generated instructions to test semantic task inference while preserving causal isolation.

If controlled language fails, report that result.

Do not bypass the gate by calling an external LLM.

---

## 11. Parameterized primitives

Do not create one persistent primitive per argument value solely to improve task-inference or hard-negative metrics.

Keep primitive identity and arguments separate where the established architecture already does so.

L4 hard-negative tests should challenge argument inference/routing without destroying this design invariant.

---

## 12. Compute safety

Target remains a single 16 GB GPU-class development machine.

Milestone sweeps should fit this target.

Before increasing model size:

- reduce batch size;
- use mixed precision where valid;
- use bounded candidate evaluation;
- profile where memory is actually spent.

Do not add distributed training.

---

## 13. Required completion report for each task

For a research task, report the applicable evidence below; routine documentation
and tooling changes use the root guide's completion requirements:

1. task ID;
2. files changed;
3. tests run;
4. experiment commands run;
5. run artifact paths;
6. acceptance criteria with PASS/FAIL;
7. seed count;
8. holdout family status: development or sealed;
9. task-inference modality;
10. explicit operation-ID visibility;
11. bank size and hard-negative level if applicable;
12. search budget if applicable;
13. assumptions/deviations;
14. ADRs added/required;
15. whether downstream tasks are blocked.

Do not claim a gate passed without saved results.

---

## 14. Phase B definition of done

A `B-Cxxx` task is done only when:

- the exact requested task is implemented;
- focused tests pass;
- repository verification passes or documented pre-existing failures are isolated;
- run artifacts exist for experimental tasks;
- task/content and oracle leakage invariants are checked;
- all acceptance criteria are explicitly evaluated;
- no unrelated refactor is bundled;
- failed STOP GATEs halt downstream work.

The entire Phase B is done only after `B-C014`.
