# AGENTS Addendum — A1-R005 Diagnostic and Retry

## Current status

A1-R005 failed:

- Correct exact match: 0.308
- Wrong argument: 0.255
- Wrong family: 0.028
- None: 0.143
- causal gap: 0.053

A1-R006 and all dependent tasks remain blocked.

## Interpretation discipline

Do not summarize the failure as "insufficient training" unless diagnostics support that conclusion.

The current evidence supports at least two distinct facts:

1. primitive family identity is causally important;
2. primitive argument identity is weakly causal under the current architecture/training setup.

## Required diagnostic order

Before changing capacity:

1. re-analyze the existing run,
2. audit wrong-argument semantic effect,
3. audit argument encoders for information loss,
4. test counterfactual argument training on a simple single-token task,
5. only then sweep training/capacity,
6. only then compare primitive architectures if needed.

## No blind scaling

Do not begin with more rank, more steps, larger Stable Core, or larger arg embedding.

## SELECT rule

`SELECT.indices` must preserve order and multiplicity if task semantics depend on them.

A commutative pooling representation such as mean/sum is invalid for an ordered index sequence unless an explicit experiment proves otherwise.

## Wrong-argument control rule

A wrong argument is scientifically useful only if it changes the ground-truth output.

Track:

`argument_effect = [target(correct_arg) != target(wrong_arg)]`

Primary wrong-argument causal metrics must be reported on the subset where `argument_effect == True`.

## Counterfactual training rule

At least one retry experiment must present the same content with multiple different arguments in the same batch or tightly coupled training group.

The purpose is to make argument identity the only information that can explain target differences.

## Exact-match rule

For multi-token operations, always report sequence exact match and token accuracy.

## Primitive-class rule

Do not assume all primitives must use the same low-rank residual implementation.

The common contract is:

`P_i(h, arguments) -> h'`

while internal implementation may differ.

## STOP discipline

A1-R006 remains blocked until the final retry gate passes.
