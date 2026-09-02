# Next Phase Decision Matrix

Fill only with measured results.

## Core diagnostics

| Diagnostic | Pass meaning |
|---|---|
| Token reconstruction | raw content identity retained |
| Position reconstruction | positional information retained |
| Structured-role probe | BIND-relevant structure accessible |
| Oracle latent operator | latent states/decoder support oracle-selected computation |
| Frozen high-cap upper bound | frozen representation computationally sufficient |
| Compact cross-position probe | small operator class may be sufficient |
| Joint task-blind representation control | representation can be learned without task leakage |

## Branch A — Operator / Heterogeneous Primitive

Choose when:
- oracle latent operator passes,
- frozen high-capacity upper bound passes strongly,
- historical low-rank conditioned primitive fails,
- compact cross-position operator materially closes the gap.

Next phase: heterogeneous computational primitives such as:
- local low-rank transforms,
- query-conditioned aggregation,
- keyed retrieval,
- position routing/gather,
- small cross-attention operators.

Keep task-blind Stable Core.

## Branch B — Queryable Representation Learning

Choose when:
- frozen upper bound fails/is clearly limited,
- but jointly trained task-blind representation + strong operator succeeds.

Next phase: train richer task-independent representation using candidate objectives such as:
- token reconstruction,
- position reconstruction,
- local relation prediction,
- key/value role prediction,
- masked retrieval,
- content autoencoding.

Do not feed task identity into content encoder.

## Branch C — Interface / Factorization Reconsideration

Choose when:
- oracle latent operator fails badly, or
- joint task-blind representation + strong operator also fails.

Next phase: revisit decoder interface, state layout, causal split, or latent workspace representation.

Do not proceed to learned routing/consolidation.

## Branch D — Mixed result

If operations split across branches, preserve the mixed diagnosis.

Example:
- COUNT may fit aggregation,
- BIND may require keyed attention,
- SHIFT/SELECT may require position-routing operators.

Do not force all operations into one primitive class for uniformity.

## Unblocking rule

A1-R006 remains blocked until the user explicitly selects the next branch after the final diagnostic report.
