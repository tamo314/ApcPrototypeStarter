# AGENTS Phase A.1 Correction Addendum

This file supplements `AGENTS.md` and the existing Phase A.1 addendum.

## Historical interpretation

Do not rewrite A1-006 as a failure.

Record it as:

- **H1a PASS:** per-operation systematic generalization on identifiable known operations,
- **H1b UNTESTED until this correction:** shared-core conditional systematic generalization.

The correction exists because A1-006 ultimately trained one model per operation.

## Main scientific rule

Before learned primitive routing is tested, the repository must demonstrate that **one shared Stable Core** can execute multiple operations when the requested operation and its arguments are explicitly observable.

## No hidden control variables

Every model behavior that changes the correct output must be inferable from model-visible input/state.

This includes:

- operation identity,
- operation parameters,
- primitive arguments,
- composition instructions when the experiment requires them.

Oracle-only metadata may remain hidden only in experiments explicitly labeled oracle.

## Parameterized primitive rule

Do not create separate primitives merely because an operation uses a different argument.

Prefer:

- `SHIFT(amount)`
- `SELECT(index)`
- `COUNT(target)`
- `BIND(key)`

over:

- `SHIFT_1`, `SHIFT_2`, ...
- `SELECT_0`, `SELECT_1`, ...

unless an experiment explicitly tests discrete specialization.

## Shared-core rule

The corrective shared-core gate must use one model for all included operations.

Do not satisfy it by training separate models.

## Task/content probe rule

Task/content factorization is not considered validated merely because the API returns two tensors.

Probe whether:

- `z_task` predicts operation identity and arguments,
- `h_content` predicts content variables,
- routing-relevant information exists in `z_task`.

The goal is useful factorization, not perfect disentanglement.

## Permutation rule

Symbol permutation is optional and task-dependent.

Use it only when the operation remains identifiable under the permutation.

Do not require semantic/value-dependent operations to solve an unobservable remapping.

Online procedural generation remains the primary anti-memorization mechanism.

## STOP discipline

If the shared-core gate fails:

- stop before A1-007,
- investigate task encoding / training objective / representation,
- do not compensate by adding a larger router or Plastic Workspace.

If probes show `z_task` contains no usable task information:

- stop learned routing work,
- fix the representation path first.
