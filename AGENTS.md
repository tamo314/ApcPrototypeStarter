# AGENTS.md

## Purpose and scope

APC (Adaptive Primitive Consolidation) is a continual-learning research prototype
separating stable sparse execution, reusable primitives, temporary plastic capacity,
and functional consolidation. This file is the agent entry point, not a run log.

- Complete the user's requested work without silently broadening scope.
- Execute only the explicitly requested research task. Completing a task does not
  authorize the next task, training, candidate selection, or a sealed evaluation.
- On a failed STOP GATE, preserve artifacts, report the failed criterion, record
  an ADR, and stop dependent work. Negative results are valid outputs.
- Preserve existing uncommitted work and historical measurements.

## Find the applicable contract

Find current progress, active work, and blocked prerequisites in the execution
plans and [decision index](docs/DECISIONS.md), not in this file. A diagnostic
success or proposed follow-up document does not authorize further work or satisfy
a recovery/research gate. Verify the applicable prerequisites before execution.

For research work, identify the exact requested task and read its task-specific
contract, applicable execution plan, experiment criteria, and agent addendum
before editing or running experiments. Inserted REC-004 lettered tasks have their
own contracts; locate them through the decision index. Read relevant prior ADRs
and adjacent code/tests. Routine documentation or tooling work does not start a
research task.

| Scope | Entry point and reading order |
| --- | --- |
| Model Bundle Recovery | [Execution plan](docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md), [tasks](docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md), [experiment plan](docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md), [bundle contract](docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md), [agent addendum](docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md), [source notes](docs/research/B2_MODEL_BUNDLE_RECOVERY_SOURCE_NOTES.md) |
| Parent Post-D2 Repair | [Execution plan](docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md), [tasks](docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md), [experiment plan](docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md), [agent addendum](docs/AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md) |
| Parent Phase B | [Execution plan](docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md), [tasks](docs/CODEX_TASKS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md), [experiment plan](docs/EXPERIMENT_PLAN_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md), [agent addendum](docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md) |

Parent requirements still apply; recovery does not relax R3/G4 thresholds.
Phase A/A.1/A.2 documents are historical evidence unless an active contract
explicitly incorporates them. Preserve their conclusions and artifacts.

Read these references when the change touches their subject:

- Architecture or experiments: [research execution rules](docs/RESEARCH_EXECUTION_RULES.md)
  and [causal primitive execution](docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md).
- B2 evaluation: [relation splits](docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md)
  and [functional adequacy](docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md).
- Compute reporting: [accounting definitions](docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md).
- Hardware/setup: [hardware environment](docs/HARDWARE_ENVIRONMENT.md) and [README](README.md).
- Other Phase B designs: follow the applicable execution plan's references.

## Non-negotiable invariants

- `h_content = f(content)`: task identity and primitive arguments must not leak
  through the content path or bypass primitive execution through the decoder.
- Stable Core must not perform the operation-specific transform before the
  primitive. Freeze Core and stable primitives where the experiment requires it.
- Establish oracle/deterministic controls before crediting learned mechanisms.
  Causal benchmarks require Correct/Wrong/None and, when parameterized, Wrong
  argument controls under the active thresholds.
- Neural execution must consume `PrimitiveCall.arguments`. Compose by ordered
  primitive execution without rerunning a task-conditioned Core between steps.
  Keep `PrimitiveBank` and `CompositionLibrary` distinct.
- Execute only selected primitives; masking outputs after dense execution is not
  sparse computation. Account for resident, active, and temporary capacity separately.
- Keep temporary and persistent parameters separable. Consolidation produces a
  separate candidate; release temporary capacity only after shadow validation.
- Try installed primitives/recipes before new adaptation on recurrence. Oracle
  metadata is supervision/evaluation data, not learned-router inference input.
- Protect sealed data and enforce the applicable data-disjointness checks before
  training. Build and evaluate separately; no hidden training/cache fallback.
- Preserve existing run files and shared caches. Use new run/bundle namespaces;
  do not overwrite evidence or fabricate missing artifacts and metrics.

## Implementation and verification

Use Python 3.12 and the dependency constraints in `pyproject.toml`. From an
activated development environment, the documented install command is:

```bash
python -m pip install -e ".[dev,plots]"
```

Keep code in `src/apc/`, tests in `tests/`, configurations in `configs/`, runners
in `scripts/`, and generated artifacts in gitignored `runs/`. Prefer typed,
readable Python, explicit serializable configuration, deterministic seeds, and
concise public API docstrings. Avoid hidden global state, unrelated refactors,
and optimization without measurements. Preserve CPU-testable logic.

For implementation changes, add meaningful invariant/regression coverage and run
focused tests while iterating. Before completion, run the standard checks and any
additional checks required by the active task:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

For documentation-only changes, check links, consistency, and the diff; no model
training or full Python suite is needed unless a task explicitly requires it.
Do not add tests that merely mirror reversible, low-impact edits. Report checks
not run and their reasons; never present an unexecuted check as passing.

Default experiments must fit the single RTX 5060 Ti (16 GB VRAM), 64 GB RAM
workstation. Keep unit tests lightweight; multi-hour sweeps belong only to explicit
milestone experiments. Do not stop other processes in the shared environment.
Unless the user changes scope, do not introduce pretrained LMs, RL controllers,
vector databases, distributed training, custom CUDA/Triton kernels, architecture
search, dense-teacher circuit extraction, or neuromorphic work.

## Evidence and completion

- Evaluate the requested acceptance criteria explicitly. Distinguish task
  completion, recovery-gate status, and research-gate status.
- Record commit/config/seeds, data and compute budgets, hardware/time/memory,
  parameter accounting, relevant controls, and metrics for meaningful runs as
  specified in the research rules and active experiment contract.
- Report changed files, checks/experiments run, PASS/FAIL or not-executed status,
  artifact paths, deviations, and known limitations. Claims require recorded evidence.
- Record architectural decisions and changed interpretations in the active ADR
  file identified by `docs/DECISIONS.md`, then add an index entry. Use the next
  unused global ADR number; never renumber or rewrite historical measurements.

## Maintain this guide

Keep this file stable and concise. Change it only when repository-wide working
rules, architecture invariants, canonical commands, or document organization
materially change. Routine task completion does not require an AGENTS.md update.

Do not add progress reports, latest-task summaries, experimental results, current
blocker lists, or chronological histories here. Maintain those in execution plans,
task documents, ADRs, and run artifacts. Keep detailed scientific rules in
`docs/RESEARCH_EXECUTION_RULES.md` and link to specialized contracts rather than
duplicating them here.
