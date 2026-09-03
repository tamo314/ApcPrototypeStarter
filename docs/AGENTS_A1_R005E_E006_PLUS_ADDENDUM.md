# AGENTS Addendum — A1-R005E E006+ Factorial Diagnosis

## Current evidence

Measured:
- Frozen representation + High-capacity operator = E-004
- Frozen representation + Compact operator = E-005

Do not silently alter these cells.

## Active question

Separate:

1. **representation accessibility bottleneck**
2. **compact operator architecture bottleneck**

## Primary causal control

E-006A changes one major factor relative to E-005:

`Frozen task-blind encoder -> Trainable task-blind encoder`

Keep the compact operator architecture and scale unchanged.

## Task blindness

The content encoder must still satisfy:

`h_content = f(content)`

Arguments/task identity must never enter content encoding.

## No operator rescue in E-006A

Do not add layers, width, relative bias, rotary embeddings, recurrence, or operation-specific logic to the compact operator.

## Conditional E-006B

Run joint representation + high-capacity operator only for operations that remain unresolved after E-006A.

## Per-operation diagnosis

COUNT/BIND/SHIFT/SELECT may land in different branches.

## No production redesign yet

Do not implement heterogeneous primitives or representation-learning objectives as production architecture until the final report and user approval.

## Blocking rule

A1-R006 remains blocked.
