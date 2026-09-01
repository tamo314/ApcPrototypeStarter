# Phase A.1 Correction — Shared-Core and Parameterized Primitive Gate

**Insert point:** after A1-006, before A1-007.

## 1. Reason for correction

A1-006 passed after several non-identifiabilities in the synthetic task design were discovered and corrected.

The final A1-006 gate trained separate models per operation.

That proves per-operation systematic generalization, but not the capability needed for the later APC router:

> a single Stable Core must represent which task is requested and expose a useful routing state.

Therefore A1-006 is retained as H1a and this correction introduces H1b/H1c.

## 2. Corrected conceptual task

The model must observe:

```text
task specification + arguments + content
```

not merely content.

The intended factorization is:

```text
task spec -------> z_task ---------> router / novelty / arguments
content ---------> h_content ------> primitive execution
```

## 3. New milestones

### A1-CM1 — Explicit task specification

Add model-visible operation identity and operation parameters.

Do not change target semantics.

### A1-CM2 — Shared Stable Core

Train one model over the mixed known-operation universe.

No per-operation model separation.

### A1-CM3 — Shared-core generalization gate

Evaluate unseen procedural content across >=5 seeds.

Gate:

- overall mean >=0.95,
- per-operation mean >=0.90.

### A1-CM4 — Representation probes

Probe `z_task` for operation identity and parameters.

Gate:

- operation identity >=0.95,
- applicable argument decoding >=0.90.

### A1-CM5 — Parameterized PrimitiveCall

Add `PrimitiveCall(primitive_id, arguments)` abstraction.

Verify one primitive family handles multiple arguments without allocating per-argument primitives.

### A1-CM6 — Oracle routing readiness

Update A1-007's oracle interface to provide complete PrimitiveCalls.

Only after this milestone should A1-007 begin.

## 4. Interpretation rule

Passing this correction does not prove routing.

It proves only that:

- one Stable Core can solve multiple explicitly specified tasks,
- `z_task` contains task-relevant information,
- the later router receives an identifiable problem.

## 5. Phase A reinterpretation

Do not rewrite Phase A results.

At final audit, note that some Phase A generalization failures were confounded by hidden operation choice / hidden parameters.

This is a retrospective interpretation, not retroactive alteration of historical measurements.
