# Phase A.1 Post-Correction Experiment Plan

## 1. Main question

Can APC make operation-specific computation depend causally on sparse reusable primitives rather than on a dense Stable Core that already solves the task?

## 2. H2a — Task-blind content representation

For identical content under different task specifications, primitive input state is invariant.

Target:
- exact or near-exact invariance under a predeclared tolerance,
- verified over >=5 seeds / generated batches.

## 3. H2b — Parameter-free primitive causality

For COPY / NEGATE / COMPARE / ACCUMULATE:

- Correct >= 0.95
- Wrong <= 0.30
- None <= 0.30
- Correct minus max(Wrong, None) >= 0.50

## 4. H2c — Parameterized primitive causality

For SHIFT / SELECT / COUNT / BIND:

- Correct family + correct argument >= 0.90
- correct family + wrong argument materially lower
- wrong family materially lower
- no primitive materially lower
- one persistent family supports multiple argument values

## 5. H3 — Composition without new capacity

- oracle composition >=0.90
- search/retrieval composition >=0.85
- no Plastic Workspace allocation for successful C tasks

## 6. H4 — Selective residual plasticity

Under oracle novelty:
- K/C expansion <=0.10
- N expansion >=0.90
- N post-adaptation >=0.95 on controlled tasks

## 7. H5 — Functional consolidation

For controlled low-rank N:
- functional agreement >=0.99
- task retention >=0.95
- candidate/temporary parameter ratio <=0.50
- stretch: recovered rank <=2x ground-truth rank

## 8. H6 — Recurrence reuse

Oracle recurrence:
- >=0.95
- near-zero adaptation
- >=90% R events avoid new consolidation

Retrieval recurrence:
- >=0.90
- oracle item retrieved >=95%

## 9. H7 — Learned routing

- oracle-required primitive in selected top-k >=0.95
- recurrence reuse >=0.90
- no severe collapse as bank grows through declared range

## 10. H8 — Learned computational novelty

- K/C expansion <=0.10
- N expansion >=0.90
- AUROC >=0.90 or predeclared equivalent

## 11. H9 — Sparse-compute scaling

As bank size grows:
- active primitive params track top-k, not total bank
- non-selected forward calls remain zero
- active primitive FLOPs remain approximately fixed for fixed top-k

## 12. H10 — Full closed loop

Only after H2-H9 pass.

Use repeated K/C/N/R sequence and >=5 seeds.

Report:
- score
- transitions
- expansions
- recurrence reuse
- forgetting
- resident/active parameters
- temporary peak
- compression
- lifetime train examples/steps
- wall-clock

## 13. Required causal controls

For every oracle primitive experiment:
- Correct
- Wrong
- None

For parameterized primitives:
- Correct family + wrong argument

Never report only Correct.

## 14. Baselines

At minimum:
- dense shared-core solver from A1-C004
- causal primitive system
- no-primitive decoder
- wrong-primitive control
- grow-only residual
- full-task plastic
- residual plastic
- full APC

At this tiny task scale, the dense baseline may remain smaller or simpler. Phase A.1's goal is causal modularity and active-compute separation, not yet scaling superiority.

## 15. Deferred scaling experiment

Before claiming scaling benefits, later test fixed Stable Core capacity while increasing primitive families, e.g. 8 / 16 / 32 / 64 / 128 as practical.

Compare dense baseline against APC resident and active capacity.

## 16. Stop conditions

Stop if:
- task-blind invariance fails,
- Correct/Wrong/None are all high,
- Correct/Wrong/None are all low,
- parameterized primitive ignores arguments,
- oracle composition fails,
- residual learner cannot solve controlled N,
- controlled compressible target cannot be consolidated.
