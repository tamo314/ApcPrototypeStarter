# AGENTS.md

## Purpose
This repository prototypes **APC (Adaptive Primitive Consolidation)**: a continual-learning architecture that alternates between a sparse stable execution system and a temporary high-plasticity learning system, then consolidates reusable computation into a persistent primitive bank.

Treat this file as a map, not an encyclopedia. Read the linked documents before changing architecture-level behavior.

## Read first
1. `README.md` — project goals, non-goals, commands, milestone order.
2. `docs/design-docs/ARCHITECTURE.md` — source of truth for modules and state transitions.
3. `docs/exec-plans/active/PHASE_A.md` — current implementation plan and acceptance criteria.
4. `docs/EXPERIMENT_PLAN.md` — baselines, metrics, ablations, reproducibility rules.
5. `docs/HARDWARE_ENVIRONMENT.md` — single-GPU constraints and environment assumptions.
6. `docs/research/REFERENCES.md` — research context; do not treat speculative claims as established facts.

## Current scope
Implement **Phase A only** unless the user explicitly asks to advance the milestone.

Phase A tests whether the following loop can work in a synthetic compositional environment:

`STABLE -> SEARCH -> PLASTIC -> CONSOLIDATE -> SHADOW -> STABLE`

The initial prototype must NOT include:
- pretrained language models;
- RL-based meta-control;
- distributed training;
- custom CUDA/Triton kernels;
- semantic or vector databases;
- mechanistic-interpretability tooling;
- internet-dependent training pipelines;
- automatic architecture search.

## Engineering principles
- Prefer the smallest implementation that can falsify the current hypothesis.
- Keep components independently testable.
- Do not silently broaden scope.
- Do not optimize before measurements show a bottleneck.
- Use deterministic seeds wherever practical.
- Every architectural change requires an experiment or test that can show whether it helped.
- Keep temporary plastic capacity separate from persistent primitive capacity in code and checkpoints.
- Never mutate stable primitives during Phase A unless a task explicitly says to test that ablation.
- Consolidation must produce a new candidate primitive; it must not overwrite the temporary solution in place.
- Resource release is allowed only after shadow validation passes.

## Python and style
- Target Python 3.12.
- Use PyTorch as the only required deep-learning framework.
- Prefer typed Python (`typing`, dataclasses where useful).
- Public functions/classes need concise docstrings.
- Keep research code readable over clever.
- Avoid global mutable state.
- Configuration must be explicit and serializable.
- All reported metrics must include the seed and experiment configuration.

## Repository conventions
Expected package layout:

```text
src/apc/
  core/
  primitives/
  plastic/
  consolidation/
  meta/
  environments/
  evaluation/
  utils/
tests/
configs/
scripts/
runs/            # gitignored outputs
```

Do not create alternate top-level package layouts without updating `docs/design-docs/ARCHITECTURE.md` first.

## Tests
For every implementation task:
1. Add or update unit tests.
2. Run the smallest relevant test set while iterating.
3. Before declaring completion, run the repository verification command documented in `README.md`.
4. If GPU-specific code is added, provide a CPU fallback test for logic that does not intrinsically require CUDA.

Tests should cover invariants, not only happy paths. Important invariants include:
- top-k routing selects no more than k primitives;
- disabled/frozen primitives receive no parameter updates;
- temporary parameters are distinguishable from persistent parameters;
- consolidation cannot delete temporary capacity before shadow validation;
- capacity accounting matches actual trainable/persistent parameters;
- deterministic tasks reproduce with the same seed.

## Experiment discipline
Do not describe a result as supporting APC unless it beats or clearly differs from the relevant baseline defined in `docs/EXPERIMENT_PLAN.md`.

For each run, record at least:
- git commit;
- config;
- seed;
- wall-clock time;
- peak VRAM if CUDA is used;
- persistent parameter count;
- temporary peak parameter count;
- active parameter count per step or its measured proxy;
- task accuracy/loss;
- retention on prior tasks;
- consolidation compression ratio.

Never compare runs with materially different data budgets without saying so.

## workflow
Work in issue-sized changes. Prefer one milestone task or a few hundred lines of focused code per change.

Before editing:
- identify the relevant acceptance criteria in `docs/exec-plans/active/PHASE_A.md`;
- inspect adjacent code and tests;
- state any assumption that changes architecture behavior in the implementation notes or commit message.

After editing:
- summarize files changed;
- report exact tests/commands run;
- report known limitations;
- do not claim benchmark success without actual recorded results.

## Decision log
If implementation reveals that an architecture assumption is wrong or impractical, do not hide the deviation. Add a short entry to `docs/DECISIONS.md` with:
- date;
- decision;
- evidence/reason;
- consequences;
- whether the architecture document must change.

## Safety rails for compute
This project is designed for one RTX 5060 Ti 16GB and 64GB system RAM.
- Default experiments must fit in 16GB VRAM.
- Do not add a default config expected to OOM on the target machine.
- Prefer gradient accumulation, mixed precision, activation checkpointing, and small synthetic batches over CPU offload during Phase A.
- Treat multi-hour sweeps as explicit experiments, not default tests.
- Unit tests must stay lightweight and should run on CPU unless CUDA behavior is specifically under test.

## Definition of done
A task is done only when:
- acceptance criteria are satisfied;
- tests pass;
- relevant docs/configs are updated;
- no unrelated refactor is bundled in;
- measured claims are backed by saved run artifacts.

## Active research phase

Phase A is closed with a negative scientific verdict but a mechanically working closed loop.
The active plan is `docs/exec-plans/active/PHASE_A1.md`.

Before implementing any Phase A.1 task, also read:
- `docs/AGENTS_PHASE_A1_ADDENDUM.md`
- `docs/CODEX_TASKS_PHASE_A1.md`
- `docs/design-docs/PHASE_A1_ARCHITECTURE_DELTA.md`
- `docs/EXPERIMENT_PLAN_PHASE_A1.md`
