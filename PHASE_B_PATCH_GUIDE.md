# APC Phase B Documentation Pack

## Purpose

This pack is a proposed repository-aligned instruction set for:

**Phase B — Semantic Task Inference & Open-World Extension**

It assumes Phase A.2 is closed and preserved as historical evidence. It does not ask the coding agent to rerun or reinterpret Phase A.2 unless a Phase B task explicitly needs a frozen baseline artifact.

The pack follows the repository's established documentation hierarchy:

1. `docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
2. `docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
3. `docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md`
4. `docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md`
5. supporting design documents under `docs/design-docs/`

The root `AGENTS.md` remains the navigation/execution authority. Update it to point to this Phase B document set before beginning `B-C001`.

## Scientific scope

Phase B asks whether APC can move beyond the Phase A.2 scope of explicit model-visible TaskSpec plus a controlled operation universe while preserving the mechanisms that already have evidence:

- task-blind shared content representation;
- heterogeneous compact primitives;
- strict sparse primitive execution;
- composition before plastic expansion;
- functional inadequacy as the main novelty signal;
- compact-first plasticity with overcomplete fallback;
- safe consolidation;
- fresh-runtime recurrence;
- bounded incremental routing;
- separate accounting for resident parameters, active parameters, FLOPs, and latency.

Phase B is deliberately staged. It does **not** begin by adding a pretrained language model.

## Proposed file additions

```text
docs/
├── exec-plans/
│   └── active/
│       └── PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md
├── CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md
├── EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md
├── AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md
├── DECISIONS_PHASE_B.md
└── design-docs/
    ├── OPEN_WORLD_HOLDOUT_PROTOCOL_PHASE_B.md
    ├── HARD_NEGATIVE_ROUTING_PHASE_B.md
    ├── DECISION_SEARCH_SCALING_PHASE_B.md
    └── TASK_INFERENCE_PHASE_B.md
```

## Required root-document updates

Before implementation, make a small documentation-only change:

### `AGENTS.md`

Change the active phase to:

`Phase B — Semantic Task Inference & Open-World Extension`

Change the active task sequence to:

`B-C001` through `B-C014`

and make the Phase B documents above the active "Read first" authority.

Retain Phase A.2 documents under historical reference. Do not delete or rewrite their conclusions.

Also replace stale workflow/definition-of-done references to Phase A.1/A.2 task documents with the Phase B task and experiment documents.

### `README.md`

Update only the project-status section and canonical Phase B commands after those scripts exist. Do not rewrite historical Phase A / A.1 descriptions that are still useful.

### `docs/DECISIONS.md`

Add `docs/DECISIONS_PHASE_B.md` to the decision-log index. Determine the next ADR number from the repository at implementation time; do not guess or renumber prior ADRs.

## Recommended code locations

Do not create a new top-level package. Extend the existing layout.

Prefer existing neighboring modules when they already provide the required abstraction. If no suitable module exists, the following names are recommended:

```text
src/apc/
├── environments/
│   ├── holdout_families.py
│   ├── task_descriptions.py
│   └── task_demonstrations.py
├── meta/
│   ├── task_inference.py
│   └── decision_budget.py
├── primitives/
│   └── hard_negative_keys.py
└── evaluation/
    ├── unseen_family_holdout.py
    ├── hard_negative_routing.py
    ├── decision_cost_scaling.py
    ├── task_inference.py
    └── phase_b_sequential.py
```

Recommended scripts:

```text
scripts/
├── unseen_family_holdout_gate.py
├── hard_negative_routing_gate.py
├── decision_cost_scaling_gate.py
├── task_inference_gate.py
└── phase_b_sequential_benchmark.py
```

Recommended configs:

```text
configs/
├── phase_b_unseen_family_holdout.yaml
├── phase_b_hard_negative_routing.yaml
├── phase_b_decision_cost_scaling.yaml
├── phase_b_task_inference_structured.yaml
├── phase_b_task_inference_fewshot.yaml
├── phase_b_task_inference_natural_language.yaml
└── phase_b_integrated_open_world.yaml
```

These are suggested locations, not permission to duplicate an abstraction that already exists.

## Primary ordering rule

Do not implement Task Inference before the explicit-TaskSpec upper-bound gates pass.

The intended order is:

```text
freeze Phase A.2 mechanisms
        ↓
unseen-family holdout with explicit TaskSpec
        ↓
hard-negative retrieval
        ↓
decision/search-cost scaling
        ↓
Task Inference interface + identifiability controls
        ↓
structured descriptor
        ↓
few-shot demonstrations
        ↓
controlled natural-language instruction
        ↓
integrated semantic/open-world loop
```

This ordering is load-bearing because it prevents a Task Inference failure from being misdiagnosed as a controller/lifecycle failure.

## What Phase B does not claim

Passing Phase B would still not establish:

- competitive general-purpose language understanding;
- large-scale LLM behavior;
- arbitrary real-world open-world learning;
- robust partially observable or stochastic learning;
- 128 genuinely learned semantic skills;
- universal discovery advantage for overcomplete plastic capacity.

Those require later phases unless explicitly promoted by measured evidence.
