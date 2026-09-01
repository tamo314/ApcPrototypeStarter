# Phase A.1 Correction Experiment Plan

## 1. Purpose

A1-006 established H1a:

> a per-operation Stable Core can learn identifiable known operations and generalize to unseen procedurally generated content.

This correction tests H1b:

> one shared Stable Core can condition on an explicit task specification and generalize across multiple operations and arguments.

## 2. H1b — Shared-core conditional systematic generalization

### Required environment

Use one model.

Each example contains:

- explicit operation identity,
- explicit operation arguments when required,
- content input,
- target output.

No output-changing control variable may exist only in hidden oracle metadata.

### Operations

Prefer all eight known operations if they can all be made identifiable:

- COPY
- NEGATE
- COMPARE
- ACCUMULATE
- SELECT(index)
- COUNT(target)
- SHIFT(amount)
- BIND(key/spec)

If an operation remains semantically ambiguous after explicit task specification, document and exclude it with an ADR.

### Anti-memorization

Maintain:

- online procedural generation,
- fresh content,
- deterministic seeded generation.

Permutation is not required for the primary experiment.

A separate permutation-compatible diagnostic may remain for COPY or other equivariant operations.

### Gate

Minimum 5 seeds.

Targets:

- mean overall unseen exact match >= 0.95,
- each operation mean >= 0.90,
- no operation below 0.85 in any seed without explicit investigation,
- training uses a single shared model.

If this fails, stop before A1-007.

## 3. H1c — Task/content representation usefulness

This is a representation diagnostic, not a demand for perfect disentanglement.

Train lightweight frozen-state probes after H1b training.

### Probe T1 — operation identity from `z_task`

Target:

- >= 0.95 accuracy.

### Probe T2 — operation arguments from `z_task`

For applicable tasks, target:

- >= 0.90 accuracy or exact decoding accuracy appropriate to the argument domain.

### Probe C1 — content variables from `h_content`

Target:

- high enough to support task execution; predeclare metric per content format.

### Optional leakage probes

Measure operation prediction from `h_content` and content prediction from `z_task`.

Do not require these to be low unless later routing experiments show interference.

## 4. Routing-readiness criterion

A1-007 is unblocked only if:

1. H1b passes,
2. `z_task` predicts operation identity reliably,
3. required task arguments are model-visible,
4. oracle routing can be represented as a `PrimitiveCall`.

## 5. Comparison runs

Run at least:

- per-operation A1-006 reference,
- shared-core explicit-task model,
- shared-core model with task specification removed (negative control).

Expected negative control:

- mixed operations become ambiguous or materially worse.

This confirms that explicit task specification, not incidental architecture changes, resolves the problem.

## 6. Parameterized primitive diagnostic

Before full A1-007, add a small execution test showing that:

- one primitive can execute multiple argument values,
- argument values change outputs as expected,
- no new primitive object is allocated per argument value.

## 7. Reporting

Add a correction report or ADR containing:

- exact input format,
- task-token format,
- operations included,
- seed table,
- per-operation scores,
- probe scores,
- negative-control results,
- parameterized-primitive test results,
- whether A1-007 is unblocked.

## 8. Stop conditions

Stop before A1-007 if:

- shared-core generalization < threshold,
- explicit task tokens are ignored,
- task-state probe fails,
- required arguments cannot be represented without leaking oracle-only metadata,
- parameterized primitive execution is not functionally well-defined.
