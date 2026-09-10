# APC research execution rules

Detailed scientific rules referenced by the root AGENTS.md. Read the relevant sections before changing architecture or running experiments. Active task contracts supply task-specific thresholds and exceptions; historical sequences below are not authorization to start work.

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
