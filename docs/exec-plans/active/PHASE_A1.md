# Phase A.1 — Hypothesis Isolation and Oracle Ladder

**Status:** planned

## Objective

Phase A.1 corrects the scientific-test design after Phase A. Phase A showed the controller loop can run mechanically, but did not establish systematic generalization, K/C/N separability, recurrence reuse, strong consolidation, or true sparse primitive execution.

Guiding principle:

> First prove the behavior is possible with oracle information. Then replace one oracle at a time with a learned mechanism.

## Non-goals

Do not add an RL meta-controller, language-model integration, semantic memory, custom CUDA kernels, distributed training, or dense-teacher circuit extraction.

## Milestones

### A1-M0 — Measurement/execution repair

- fix resident parameter accounting,
- make top-k restrict actual primitive execution,
- make environment/PyTorch setup reproducible.

Gate: active/resident metrics must be trustworthy.

### A1-M1 — Anti-memorization environment

- online procedural generation,
- seeded determinism,
- symbol permutation,
- K/C/N/R metadata for evaluation/oracle paths.

### A1-M2 — Task/content representation split

Expose `z_task` and `h_content`; routing/novelty uses the former and primitive execution the latter.

### A1-M3 — Stable systematic-generalization gate

Train only known operations with online randomized data.

Gate: unseen-content K exact match >=0.95 across >=5 seeds. **STOP if failed.**

### A1-M4 — Oracle primitive execution and composition

Use environment-supplied primitive IDs/recipes. Add a Composition Library and small-bank composition search.

Gate: K >=0.95; C >=0.90; successful C requires no temporary allocation.

### A1-M5 — Oracle novelty + residual plasticity

K/C/N labels control expansion. Plastic Workspace learns only residual behavior after the best existing recipe.

Gate: K/C expansion <=10%; N expansion >=90%; adapted N >=0.95 on designated tasks.

### A1-M6 — Compressibility-controlled N tasks

Introduce hidden low-rank ground-truth novel transforms and intentionally overcomplete temporary capacity.

### A1-M7 — Functional consolidation

Fit the smallest candidate primitive that reproduces the temporary residual function.

Gate: functional agreement >=0.99; task retention >=0.95; candidate/temporary params <=0.50 on controlled N.

### A1-M8 — Oracle recurrence

Force the installed primitive/recipe on R events.

Gate: R >=0.95 with no new cycle on >=90% of recurrence events.

### A1-M9 — Retrieval routing

Store task-key prototypes and retrieve primitive/recipe by similarity.

Gate: R >=0.90; reuse without new cycle >=90%.

### A1-M10 — Learned routing

Train a router from `z_task`.

Gate: oracle-required primitive in selected top-k >=0.95; recurrence reuse >=0.90.

### A1-M11 — Learned computational novelty

Replace oracle novelty with residual-computation novelty. Start with residual loss/retrieval confidence; test gradient-subspace novelty only if needed.

Gate: K/C expansion <=10%; N expansion >=90%; discrimination >=0.90 AUROC or predeclared equivalent.

### A1-M12 — Full learned loop

Run K/C/N/R streams with all learned mechanisms, corrected accounting, and actual sparse primitive execution across >=5 seeds.

## Phase-wide invariants

1. Phase A artifacts remain historical and untouched.
2. Every learned mechanism has an oracle predecessor.
3. Fixed-dataset memorization does not count as H1 evidence.
4. Non-selected primitives do not execute.
5. Stable Core is included in resident/active totals.
6. Compression means function preservation at materially smaller capacity.
7. Recipe creation and primitive creation are distinct outcomes.
8. A failed gate stops dependent tasks.

## Final result

At completion create `docs/results/PHASE_A1_RESULT.md` with the same audit-oriented style as the Phase A result: scope, verdict, implementation summary, reproducibility, accounting, stop conditions, hypothesis verdicts, baselines, seed robustness, failure cases, and recommendations.
