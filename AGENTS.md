# AGENTS.md

## Purpose

This repository prototypes **APC (Adaptive Primitive Consolidation)**: a continual-learning architecture intended to separate:

- a sparse, stable execution system,
- reusable primitive computation,
- temporary high-plasticity learning capacity,
- consolidation of newly learned computation into persistent primitives,
- and later reuse without repeated full adaptation.

Treat this file as the **current navigation and execution authority for coding agents**. It is a map, not an encyclopedia. Detailed architecture, experiment, and acceptance criteria live in the active documents listed below.

Historical Phase A and earlier Phase A.1 documents remain valuable evidence, but they are **not the active implementation plan** unless an active document explicitly refers back to them.

---

## Active research phase

The active phase is:

**Phase B — Semantic Task Inference & Open-World Extension**

The current scientific question is:

> Can APC generalize its reuse / composition / plasticity lifecycle beyond the operation families used to shape Phase A.2, remain safe under hard semantic retrieval competition, keep decision/search cost bounded as the bank grows, and infer task semantics without an explicit operation ID?

The active task sequence uses IDs:

`B-C001` through `B-C014`

from:

`docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

> **Phase A.2 Completion & Phase B Scope Note (ADR-0073):**
> 1. Phase A.2 established strong autonomous controller support across all 4 STOP GATES,
>    demonstrating autonomous K/C/N/R decisions, bank scaling to N=128 with real 20.19x latency speedup,
>    and class-incremental router updates under explicit model-visible TaskSpec.
> 2. Phase B investigates open-world extension beyond Phase A.2's closed operation families,
>    hard-negative retrieval competition, bounded decision/search cost scaling, and semantic task
>    inference without explicit canonical operation IDs.
>
> Follow the active execution plan:
> 1. `docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
> 2. `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
> 3. `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
> 4. `docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md`

> **B2 Post-D2 Repair branch (ADR-0080/ADR-0081, active as of 2026-09-06):**
> `B-C005G`'s sealed re-gate failed (ADR-0079), and the second diagnostic
> phase (`B-C005D2-001`..`B-C005D2-006`, ADR-0080/ADR-0081) is now complete.
> `B-C006` and all Task Inference work remain **blocked** until a new sealed
> `B2_PROTOCOL_V2` Gate passes. The current repair branch uses task IDs
> `B-C005R3-001`..`B-C005R3-012` and its own document set, read in this order:
> 1. `docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md`
> 2. `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`
> 3. `docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md`
> 4. `docs/AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md`
> 5. `docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`
> 6. `docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md`
>
> This is a repair sub-phase of Phase B, not a new phase; it does not replace
> the Phase B documents listed above, and it does not authorize starting
> `B-C006`. Only one `B-C005R3-0NN` task runs per explicit user instruction;
> completing one does not authorize starting the next.

Implement **only the currently requested B-Cxxx task** unless the user explicitly asks to change scope.

Do not continue automatically to the next task after completing one.

If a STOP GATE fails, stop dependent work and report the failure.

---

## Read first

Before changing architecture-level behavior, read these documents in order:

1. `docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
   - current milestone order and scientific gates.

2. `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
   - current implementation tasks and acceptance criteria.

3. `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
   - current hypotheses, controls, thresholds, baselines, and stop conditions.

4. `docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md`
   - current phase-specific agent rules.

5. `docs/design-docs/OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md`
   - open-world holdout protocol and family partitions.

6. `docs/design-docs/HARD_NEGATIVE_ROUTING_PHASE_B.md`
   - retrieval competition difficulty ladder and verification design.

7. `docs/design-docs/DECISION_SEARCH_SCALING_PHASE_B.md`
   - decision and composition search cost decomposition and bounded search.

8. `docs/design-docs/TASK_INFERENCE_PHASE_B.md`
   - semantic task inference modalities and identifiability constraints.

9. `docs/design-docs/AUTONOMOUS_CONTROLLER_POLICY.md`
   - runtime controller decision policy (direct reuse vs compose vs plastic search).

10. `docs/design-docs/INCREMENTAL_ROUTER_AND_BANK_SCALING.md`
    - incremental router updates, replay/prototype bounds, and bank growth.

11. `docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md`
    - comprehensive parameter, FLOPs, and latency accounting requirements.

12. `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`
    - source of truth for task-blind content encoding and causal primitive execution.

13. `docs/HARDWARE_ENVIRONMENT.md`
    - target workstation and compute constraints.

14. `docs/DECISIONS.md`
    - index of every ADR (measured findings and architecture decisions), split by
      research phase into `docs/DECISIONS_PHASE_A.md`, `docs/DECISIONS_PHASE_A1.md`,
      `docs/DECISIONS_PHASE_A1_CORRECTION.md`, `docs/DECISIONS_PHASE_A1_POST_CORRECTION.md`,
      `docs/DECISIONS_A1_R005E_DIAGNOSTIC.md`, `docs/DECISIONS_PHASE_A2.md`,
      and `docs/DECISIONS_PHASE_B.md` (active).
      ADR numbers are a single global sequence and are never renumbered; append a new ADR
      to the active phase's file and add one row to the index.

15. `README.md`
    - repository setup, commands, and general project context.

### Historical reference only

The following documents remain part of the research record but must not override the active documents above:

- `docs/exec-plans/active/PHASE_A.md`
- `docs/EXPERIMENT_PLAN.md`
- `docs/CODEX_TASKS_PHASE_A1.md`
- `docs/exec-plans/active/PHASE_A1.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1.md`
- `docs/AGENTS_PHASE_A1_ADDENDUM.md`
- `docs/exec-plans/active/PHASE_A1_CORRECTION.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md`
- `docs/AGENTS_PHASE_A1_CORRECTION_ADDENDUM.md`
- `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md`
- `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`
- `docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1_POST_CORRECTION.md`
- `docs/AGENTS_PHASE_A1_POST_CORRECTION_ADDENDUM.md`
- `docs/exec-plans/active/PHASE_A1_BRANCH_B_INTEGRATION.md`
- `docs/CODEX_TASKS_PHASE_A1_BRANCH_B_INTEGRATION.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1_BRANCH_B_INTEGRATION.md`
- `docs/CODEX_TASKS_A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/EXPERIMENT_PLAN_A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/exec-plans/active/A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/results/A1_R005E_DIAGNOSTIC_RESULT_FINAL.md`
- `docs/exec-plans/active/PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/EXPERIMENT_PLAN_PHASE_A2_AUTONOMOUS_CONTROLLER.md`
- `docs/AGENTS_PHASE_A2_AUTONOMOUS_CONTROLLER_ADDENDUM.md`
- `docs/results/PHASE_A2_AUTONOMOUS_CONTROLLER_RESULT.md`

These documents explain how the project reached the current design. Do not delete or rewrite their historical conclusions.

---

## Current architectural invariant

The causal primitive path must conceptually follow:

```text
task specification
        |
        v
   Task Encoder
        |
      z_task
        |
 router / oracle
        |
 PrimitiveCall
        |
        +----------------------+
                               |
content                        v
   |                     selected primitive
   v                           |
Content Encoder                |
   |                           |
h_content ---------------------+
              |
              v
       transformed state
              |
              v
           decoder
```

The load-bearing invariant is:

`h_content = f(content)`

not:

`h_content = f(task, content)`.

The task identity and primitive arguments must not leak into the primitive input state through the causal content path.

---

## Scientific priority

The project is currently testing **causal modularity**, not general language capability and not benchmark scale.

The immediate sequence is:

1. task-blind content representation,
2. decoder leakage control,
3. parameter-free primitive causality,
4. parameterized primitive causality,
5. primitive composition,
6. oracle novelty,
7. residual plastic learning,
8. functional consolidation,
9. recurrence reuse,
10. retrieval routing,
11. learned routing,
12. learned novelty,
13. sparse-compute scaling,
14. full closed-loop evaluation.

Do not skip a failed earlier mechanism by adding complexity downstream.

---

## Oracle-before-learned rule

A learned mechanism may not be credited until the corresponding oracle or deterministic control succeeds.

Examples:

- oracle primitive execution before learned routing,
- oracle composition before learned composition search,
- oracle novelty before learned novelty,
- oracle recurrence before retrieval or learned recurrence routing.

If an oracle version fails, investigate that mechanism.

Do not compensate by increasing model size, adding RL, or broadening architecture.

---

## Causal primitive evidence rule

High accuracy with a primitive present is not sufficient evidence that the primitive performs the computation.

Every causal primitive benchmark must compare at least:

1. **Correct**
   - correct primitive family and correct arguments.

2. **Wrong**
   - incorrect primitive family.

3. **None**
   - identity / no primitive transformation.

For parameterized primitives also test:

4. **Wrong argument**
   - correct primitive family with an incorrect argument.

A primitive is considered causally supported only when performance strongly depends on the Correct condition according to the active experiment thresholds.

If Correct, Wrong, and None are all high, suspect Stable Core or decoder leakage.

If all are low, suspect content representation, decoder, or primitive capacity.

---

## Stable Core role

In the current causal path, the Stable Core may provide:

- task encoding,
- task-independent content encoding,
- representation transport,
- decoding infrastructure.

It must not be allowed to perform the operation-specific transformation before primitive execution.

During primitive-causality tests, freeze Stable Core parameters unless the active task explicitly requires otherwise.

Preserve the earlier high-performing shared-core solver from A1-C004 as a **baseline**, not as the causal primitive path.

---

## Task/content separation

For identical content under different task specifications, the primitive input representation must remain invariant within the tolerance declared by the active experiment.

Prefer a structurally task-blind content path over merely testing leakage with probes.

`z_task` may contain:

- operation identity,
- operation arguments,
- routing information.

`h_content` should represent the content/state being transformed.

---

## Parameterized primitives

Primitive identity and primitive arguments are separate concepts.

Prefer:

- `SHIFT(amount)`
- `SELECT(indices)`
- `COUNT(target)`
- `BIND(query_key)`

over creating a new persistent primitive for every argument value.

Neural primitive execution must actually consume `PrimitiveCall.arguments`.

Arguments that are stored or logged but ignored by the neural transform do not count as parameterized execution.

Do not introduce hypernetworks or unnecessarily general function signatures unless a later task explicitly requires them.

---

## Composition

Composition must operate through ordered primitive execution:

```text
h0 = ContentEncoder(content)
h1 = P_a(h0, args_a)
h2 = P_b(h1, args_b)
...
```

Do not re-run a task-conditioned Stable Core between primitive steps.

A new composition of known primitives is not, by itself, a new primitive.

Keep:

- `PrimitiveBank`
- `CompositionLibrary`

conceptually and operationally distinct.

---

## Plastic Workspace

Plastic capacity is temporary.

For novel operations, prefer residual learning:

```text
best existing primitive/recipe
        +
temporary residual computation
```

rather than relearning the entire task from scratch.

During controlled plastic-learning tests:

- Stable Core stays frozen,
- stable primitives stay frozen unless an explicit ablation says otherwise,
- temporary parameters are clearly separable from persistent parameters.

---

## Consolidation

Consolidation is a **functional compression** problem.

Do not define success merely as deleting parameters.

Measure:

- held-out functional agreement,
- task-score retention,
- minimum candidate rank/capacity satisfying thresholds,
- candidate / temporary compression ratio.

For compressibility-controlled experiments, ground-truth compact structure may be used for evaluation but must not leak into learner/controller inputs.

Consolidation must produce a candidate primitive separate from the temporary solution.

Temporary capacity may be released only after shadow validation passes.

---

## Recurrence and reuse

When a learned operation reappears:

1. try the installed primitive / recipe first,
2. measure whether new adaptation is required,
3. only expand if the existing bank is demonstrably insufficient.

If oracle recurrence fails, do not attribute the failure to routing.

If oracle recurrence passes but retrieval fails, isolate retrieval.

If retrieval passes but learned routing fails, isolate learned routing.

---

## Learned routing

The learned router consumes `z_task`.

Oracle metadata may be used only for supervision or evaluation, never as inference input.

Track:

- top-k inclusion of oracle-required primitive,
- recurrence reuse,
- behavior as the bank grows,
- selected primitive IDs,
- routing confidence / entropy where relevant.

Only selected primitives may actually execute.

---

## Novelty

Novelty should approximate:

> "Does the existing computational library fail to explain the required computation?"

not simply:

> "Is this input unusual?"

Preferred signals include:

- residual loss after the best existing recipe,
- retrieval confidence,
- task-key distance,
- optional gradient/subspace residual when justified by evidence.

Do not add expensive gradient novelty if simpler residual signals already pass the declared gate.

---

## Sparse execution

Top-k selection must restrict **actual primitive computation**.

This does not count as sparse execution:

```text
compute every primitive
-> zero out unselected outputs
```

This does count:

```text
select IDs
-> gather selected primitives
-> execute only selected primitives
```

Track separately:

- resident total parameters,
- resident primitive parameters,
- active total parameters,
- active primitive parameters,
- temporary peak parameters,
- primitive forward-call count,
- estimated or measured FLOPs when relevant.

Do not conflate resident capacity with active compute.

---

## Experimental discipline

Do not describe a result as supporting APC unless it is backed by the controls and baselines defined in:

`docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

For every meaningful run, record at least:

- git commit,
- config,
- seed,
- wall-clock time,
- GPU / system information,
- peak VRAM if CUDA is used,
- resident total parameters,
- resident primitive parameters,
- active total parameters,
- active primitive parameters,
- temporary peak parameters,
- task accuracy / loss,
- relevant causal-control scores,
- retention / forgetting when sequential,
- compression ratio when consolidation is involved,
- training examples / steps or other compute-budget proxy.

Never compare runs with materially different data or compute budgets without stating the difference.

Milestone claims must use the seed count required by the active experiment plan.

---

## STOP GATE discipline

Tasks marked STOP GATE are hard scientific boundaries.

On failure:

1. save the run artifacts,
2. report the failed acceptance criterion,
3. add or update an ADR,
4. stop dependent work,
5. investigate only the failing mechanism.

Do not silently continue.

Do not hide negative results by:

- increasing model size prematurely,
- expanding data without a specific hypothesis,
- adding a larger router,
- adding RL,
- adding unrelated memory systems,
- moving to an LLM.

Negative results are valid project outputs.

---

## Historical integrity

Never rewrite old Phase A or Phase A.1 runs to make current conclusions cleaner.

When newer experiments reveal that an older result was confounded:

- preserve the old measurement,
- add a retrospective interpretation,
- record the new evidence in `docs/DECISIONS.md`,
- clearly distinguish historical result from current interpretation.

Do not delete superseded task documents merely because they are no longer active.

---

## Engineering principles

- Prefer the smallest implementation that can falsify the current hypothesis.
- Keep components independently testable.
- One task should isolate one mechanism whenever practical.
- Do not silently broaden scope.
- Do not optimize before measurements show a bottleneck.
- Use deterministic seeds wherever practical.
- Every architectural change requires a test or experiment capable of showing whether it helped.
- Avoid unrelated refactors during experimental tasks.
- Preserve CPU-testable logic even when milestone runs use CUDA.
- Keep research code readable over clever.
- Avoid hidden global state.
- Configuration must be explicit and serializable.

---

## Python and framework

- Target Python 3.12.
- Use the PyTorch version constraints declared in `pyproject.toml`.
- PyTorch is the only required deep-learning framework unless an active task explicitly changes that.
- Prefer typed Python and dataclasses where useful.
- Public functions and classes need concise docstrings.
- Use `ruff` and `mypy` according to repository configuration.

Do not change Python or PyTorch compatibility constraints casually. If compatibility must change, record why.

---

## Repository conventions

Expected package layout:

```text
src/apc/
  core/
  primitives/
  plastic/
  consolidation/
  meta/
  environments/
  evaluation/
  utils/

tests/
configs/
scripts/
runs/            # gitignored outputs
docs/
```

Do not create an alternate top-level package layout without an architecture decision and corresponding documentation update.

---

## Tests

For every implementation task:

1. add or update focused unit tests,
2. run the smallest relevant tests while iterating,
3. run the repository verification commands before declaring completion,
4. provide CPU fallback coverage for logic that does not intrinsically require CUDA.

Tests should emphasize invariants.

Important current invariants include:

- task-blind content state does not change with task specification,
- task information does not bypass primitive execution in causal mode,
- non-selected primitives receive zero forward calls,
- top-k never executes more than selected primitives,
- parameterized primitive families do not multiply with argument values,
- disabled/frozen primitives receive no updates,
- Stable Core is frozen in experiments that require it,
- temporary and persistent parameters remain distinguishable,
- consolidation does not release temporary capacity before shadow validation,
- resident/active accounting matches actual execution,
- deterministic generators reproduce under identical seeds.

Before completion, run the repository's standard verification commands, typically:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

If the repository documentation declares a newer canonical command, use that instead.

---

## Codex workflow

Work in issue-sized changes.

Before editing:

1. identify the exact current task in:
   `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

2. read its corresponding scientific gate in:
   `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`

3. inspect the architecture requirements in:
   `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md`

4. inspect adjacent code and tests.

5. check `docs/DECISIONS.md` for relevant prior findings.

Do not implement the next task in the queue unless explicitly requested.

After editing, report:

1. files changed,
2. tests run,
3. experiment commands run,
4. acceptance criteria with explicit PASS / FAIL,
5. run artifact paths,
6. assumptions or deviations,
7. known limitations,
8. ADRs added or required.

Do not claim benchmark or hypothesis success without recorded results.

---

## Decision log

If implementation or experiments reveal that an architecture assumption is wrong, underspecified, or impractical, add a concise ADR to the current active phase's file (see `docs/DECISIONS.md`'s index for which file that is, and the next unused `ADR-NNNN` number), then add one row to the index in `docs/DECISIONS.md` itself.

Include:

- date,
- decision,
- evidence / reason,
- consequences,
- which active document is affected,
- whether downstream tasks are blocked.

Measured evidence should drive architecture changes.

---

## Hardware and compute safety rails

Target workstation:

- Ryzen 7 9800X3D,
- GeForce RTX 5060 Ti 16 GB,
- 64 GB system RAM,
- single GPU.

Default experiments must fit within the target machine.

- Do not add default configurations expected to OOM on 16 GB VRAM.
- Prefer mixed precision, gradient accumulation, activation checkpointing, and smaller synthetic batches before CPU offload.
- Multi-hour sweeps must be explicit milestone experiments, not default tests.
- Unit tests should remain lightweight and CPU-friendly unless CUDA behavior itself is under test.
- Do not introduce distributed training in Phase B.

---

## Out of scope for the active phase

Unless the user explicitly changes scope, do not add:

- pretrained language-model integration,
- RL meta-controller,
- semantic/vector database,
- distributed training,
- custom CUDA/Triton kernels,
- automatic architecture search,
- dense-teacher mechanistic circuit extraction,
- neuromorphic hardware work.

These belong to later phases.

---

## Definition of done

A task is done only when:

- the exact requested B-Cxxx task is implemented,
- its acceptance criteria are explicitly evaluated,
- required tests pass,
- relevant run artifacts are saved,
- metrics use the current accounting definitions,
- relevant docs/configs are updated,
- no unrelated refactor is bundled in,
- failed STOP GATEs halt dependent work,
- measured claims are backed by actual results.

Completion of one task does not authorize beginning the next task.
