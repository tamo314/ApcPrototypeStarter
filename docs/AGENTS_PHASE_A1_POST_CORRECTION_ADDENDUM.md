# AGENTS Phase A.1 Post-Correction Addendum

This file supplements the existing repository and Phase A.1 agent instructions.

## Scientific objective

The central question is now:

> Can operation-specific computation be causally moved out of the Stable Core and into sparse reusable primitives?

## Causal evidence rule

Every primitive-execution gate must compare at least:

1. correct primitive,
2. wrong primitive,
3. no primitive.

For a successful causal primitive test:

- correct must be high,
- wrong and no-primitive must be materially lower.

## Task-blind content rule

The content representation used by primitives must not depend on requested task identity or arguments.

For the same content under different task specifications, the primitive input state must be equal or nearly equal according to the declared invariant.

Do not route task tokens through the content encoder in the causal primitive path.

## Stable Core role

The Stable Core may provide token/content encoding, task encoding, state transport, and decoding infrastructure.

It must not compute the operation-specific transformation before the primitive is applied.

## Parameterized primitive rule

Parameterized neural primitives must consume `PrimitiveCall.arguments`.

Arguments that are merely logged but ignored by the neural transform do not count.

## Required progression

1. task-blind content path,
2. parameter-free primitive causality,
3. parameterized primitive causality,
4. compositions,
5. plastic learning,
6. consolidation,
7. recurrence,
8. learned routing,
9. learned novelty.

Do not collapse these steps.

## Freeze discipline

When testing primitive causality, freeze the Stable Core unless a task explicitly requires otherwise.

## Failure interpretation

If Correct/Wrong/None are all high:
- suspect Stable Core or decoder leakage.

If all are low:
- suspect content representation or primitive capacity.

If Correct is high but recurrence later fails:
- suspect routing/retrieval.

If temporary learning succeeds but consolidation fails:
- consolidation is isolated as the failing mechanism.

## No architecture escalation

Do not add a larger core, deeper router, RL controller, complex memory, or LLM integration to rescue a failed causal gate.
