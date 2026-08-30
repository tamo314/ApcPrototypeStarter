# Phase A Execution Plan — Synthetic Closed-Loop Proof of Concept

## Goal

Demonstrate, on controlled compositional tasks, a complete and repeatable cycle:

`reuse -> search -> allocate -> learn -> consolidate -> shadow-validate -> release -> reuse`

The phase succeeds only if the system performs this cycle more than once in a sequential task stream.

## Non-goals

- LLM benchmark performance;
- human-language semantics;
- million-expert retrieval;
- training a meta-controller with RL;
- discovering circuits in an arbitrary pretrained dense model;
- hardware-specific kernel optimization.

## Milestone A0 — Repository bootstrap

Deliverables:
- Python package under `src/apc`;
- `pyproject.toml` with runtime/dev dependencies;
- pytest, ruff, mypy configuration;
- config loader;
- seed utility;
- system/VRAM metadata logger;
- CI-friendly CPU smoke test.

Acceptance:
- `pytest -q` passes;
- lint/type-check commands are documented;
- CUDA availability is detected but not required for unit tests.

## Milestone A1 — Synthetic environment

Implement an executable symbolic task generator.

Required operation set for initial curriculum:
- COPY;
- SELECT;
- COMPARE;
- COUNT;
- SHIFT;
- BIND;
- NEGATE;
- ACCUMULATE.

Provide:
- deterministic generation by seed;
- operation graph metadata;
- train/validation/test split generation;
- novel-composition split;
- novel-operation extension interface.

Acceptance:
- reference interpreter produces exact targets;
- same seed reproduces examples;
- novel composition can be identified by metadata;
- no model code is needed to validate the environment.

## Milestone A2 — Fixed dense baseline

Implement a small Transformer baseline that learns the initial task curriculum.

Start small:
- 4 layers;
- d_model 192 or 256;
- context <= 128;
- sequence-to-sequence or autoregressive output chosen for simplest exact evaluation.

Acceptance:
- learns the initial known-operation tasks to a configured target accuracy;
- logs training curves, parameter count, peak VRAM;
- checkpoint can be reloaded exactly.

## Milestone A3 — Primitive bank and router

Add low-rank residual primitives and top-k routing.

Start with a fixed primitive count and train jointly on the initial curriculum.

Acceptance:
- routing respects top-k invariant;
- primitive usage is logged;
- total/persistent/active parameter accounting is correct;
- disabling selected primitives measurably changes output on at least some examples after training.

## Milestone A4 — Composition benchmark

Train on a subset of operation compositions and test on held-out combinations of known operations.

Acceptance:
- system can solve at least some held-out compositions without creating new persistent or temporary parameters;
- expansion counter remains zero for these cases under the heuristic controller;
- compare against fixed dense and fixed-MoE-like baselines.

## Milestone A5 — Plastic workspace

Implement temporary low-rank transforms and an explicit allocator.

During PLASTIC:
- stable core frozen;
- persistent primitives frozen;
- temporary transforms trainable;
- temporary router parameters trainable if necessary.

Acceptance:
- a task configured as a novel operation can allocate temporary capacity;
- trainable parameter count increases only while PLASTIC is active;
- prior stable parameters are bitwise unchanged in the strict-freeze experiment.

## Milestone A6 — Novelty estimator and finite-state controller

Implement logged novelty signals and a rule-based controller.

First controller may use only error + router entropy; gradient novelty can be added after the loop works.

Required transition logging:
- timestamp/step;
- old state;
- new state;
- trigger metrics;
- allocated/released parameter count.

Acceptance:
- known tasks stay STABLE;
- held-out composition enters SEARCH before PLASTIC;
- novel operation reaches PLASTIC after bounded SEARCH failure;
- hysteresis prevents rapid oscillation in a regression test.

## Milestone A7 — Consolidation

Implement temporary-to-persistent compression.

Minimal first version:
- distill all active temporary transforms into one or a few lower-rank candidate transforms;
- use replay from previous tasks;
- keep candidates separate from stable bank until validation.

Acceptance:
- candidate parameter count < temporary parameter count;
- candidate reproduces >= configured fraction of temporary performance;
- consolidation artifacts and metrics are saved.

## Milestone A8 — Shadow validation and release

Run temporary and candidate solutions in parallel on held-out and replay batches.

Acceptance default targets:
- candidate current-task score >= 95% of temporary score;
- retention degradation <= 2 percentage points;
- temporary capacity is released only after pass;
- after release, memory accounting returns to persistent-only plus the new compressed primitive.

## Milestone A9 — Sequential benchmark

Use a stream such as:

1. known tasks;
2. novel composition A;
3. novel operation X;
4. known tasks again;
5. novel composition B using X + old primitives;
6. novel operation Y;
7. operation X again.

Run at least 5 seeds for the final Phase A result if compute permits; use 3 during development.

Acceptance:
- at least two successful expand/consolidate/release cycles;
- X is reused on recurrence without relearning from scratch;
- resident persistent size grows much less than cumulative temporary peak allocations;
- final retention remains within defined bounds.

## Milestone A10 — Baselines and ablations

Baselines:
1. fixed dense Transformer;
2. fixed sparse/MoE-like primitive model with no expansion;
3. expansion without consolidation;
4. expansion + replay;
5. APC full loop.

Ablations:
- no SEARCH stage;
- no replay during consolidation;
- no shadow period;
- no hysteresis;
- no functional merge/compression.

Phase A should not be declared complete without baseline comparisons.

## Phase A headline metrics

- task accuracy / exact-match;
- backward transfer / forgetting;
- persistent parameter growth;
- temporary peak parameter count;
- active parameter count;
- expansion frequency;
- primitive reuse rate;
- consolidation compression ratio;
- shadow disagreement rate;
- training/inference wall-clock;
- peak VRAM;
- lifetime FLOP proxy.

## Stop conditions

Pause architecture expansion and investigate if any occurs:
- no distinction between novel composition and novel operation is measurable;
- consolidation repeatedly needs almost all temporary parameters;
- new primitives are not reused on recurrence;
- controller expands on most familiar tasks;
- retention requires replay buffers so large that they dominate the system;
- dynamic model is consistently worse than all static baselines at equal compute.

These are useful negative results; do not hide them by increasing model size prematurely.
