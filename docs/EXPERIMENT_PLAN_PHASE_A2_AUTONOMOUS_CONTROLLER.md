# Experiment Plan — Phase A.2 Autonomous Controller & Scaling

## Phase objective

Demonstrate an autonomous sequential APC lifecycle under bank growth.

## Evidence tiers

### Tier 1 — Routing stability
Can new persistent primitives be added without destroying old routing?

### Tier 2 — Adequacy controller
Can the system distinguish direct reuse, composition, and genuine novelty from functional evidence?

### Tier 3 — Sequential lifecycle
Can K/C/N/R tasks be interleaved in one online stream?

### Tier 4 — Scaling
Does sparse execution remain correct and useful as resident bank size grows?

## Task universe

Use:

- 8 canonical operations,
- already consolidated `SWAP_PAIRS`, `INVERT_HALF`,
- compositions from the existing composition suite,
- additional novel operations introduced only after bank/composition failure is verified.

Do not use an operation as "N" merely because its name is new.

## Sequential episode protocol

Each episode provides:

- explicit TaskSpec,
- a small support/adaptation set,
- held-out evaluation set.

The controller may use support targets for adequacy/search. Held-out targets are evaluation-only.

## Recommended final stream

At least 40 episodes per seed, including:

- >=10 K
- >=10 C
- >=6 first-occurrence N
- >=6 R
- additional mixed episodes

Randomize order subject to recurrence occurring after consolidation.

Decision seeds: >=5.

## Category acceptance

### K

- EM >=0.95
- false plastic <=10%

### C

- EM >=0.90
- composition action >=85%
- bank expansion <=10%

### N

- post-adaptation/consolidation EM >=0.90
- plastic trigger >=90%
- exactly one bank growth per successful novel task

### R

- EM >=0.95
- direct reuse >=90%
- adaptation steps approximately 0
- no bank growth/reconsolidation >=90%

## Historical retention

After each bank insertion:

- canonical-task drop <=2pp
- previously consolidated-task drop <=2pp
- routing old-class mean drop <=2pp

## Resource lifecycle

After completed consolidation or normal inference:

- temporary params return to 0
- no workspace leak

## Router update

Primary full-loop condition uses bounded incremental router update.

Full historical retraining is not allowed in the primary result.

## Learned novelty / controller

Primary target:

- K/C vs N AUROC >=0.90
- K/C false-plastic <=10%
- N trigger >=90%

## Composition

Composition search must be attempted before plastic expansion when direct reuse is inadequate.

No persistent bank growth for composition-only tasks.

## Plastic policy

Compact-first:

1. compact plastic search
2. if target not reached within fixed budget, optional overcomplete fallback
3. consolidate successful solution
4. release temporary capacity

Overcomplete fallback is not assumed superior.

## Compute scaling

Measure bank sizes:

`10, 16, 32, 64, 128`

Semantic end-to-end growth is required at least through 16.

Sizes above available semantic classes may use explicitly labeled routing/compute distractors.

## Final scientific claims

Phase A.2 may claim full autonomous K/C/N/R continual learning only if the final sequential stream passes without oracle runtime action labels.

Explicit TaskSpec remains a known scaffold limitation.
