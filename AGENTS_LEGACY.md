# AGENTS.md

## Purpose and scope

APC (Adaptive Primitive Consolidation) is a continual-learning research prototype
separating stable sparse execution, reusable primitives, temporary plastic capacity,
and functional consolidation. This file is the agent entry point, not a run log.

- Complete the authorized work, including necessary fixes, verification, and local
  commit, without asking again for steps already covered by the user's instruction.
  Explicit user instructions take precedence over repository workflow conventions
  and skill guidelines; do not infer new research authorization from task completion.
- Research execution must stay within the task or continuation scope explicitly
  authorized by the user. Existing continuation permission covers its named work
  when prerequisites pass; it does not implicitly authorize additional training,
  candidate selection, or sealed evaluation.
- On a failed STOP GATE, preserve artifacts, report the failed criterion, record
  an ADR, and stop dependent work. Negative results are valid outputs.
- Preserve existing uncommitted work and historical measurements.
- If an instruction blocks requested work, cite its file and exact clause, explain
  the unmet prerequisite, and continue independent authorized work. Distinguish
  a scientific STOP GATE from missing permission or an optional recommendation.

## Find the applicable contract

For routine documentation, tooling, or a local fix that does not change research
behavior, read the affected files and relevant references. No full research-plan
review or experiment execution is required.

For research implementation, execution, or a status review, start with the relevant
sections of [the restart plan](docs/exec-plans/active/PHASE_B_RESTART.md) for current
progress, authorization, and prerequisites. Follow its links to the exact task
contract, acceptance criteria, and applicable addendum before changing research
behavior or running experiments. Use the [decision index](docs/DECISIONS.md) for
relevant prior decisions and inserted REC-004 lettered contracts; do not read the
entire history by default. A diagnostic success or proposed follow-up alone does
not authorize execution or satisfy a recovery/research gate.

Read only the applicable branch and the parent requirements it incorporates:

| Scope | Detailed contract entry point |
| --- | --- |
| Model Bundle Recovery | [Execution plan](docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md), [agent addendum](docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md) |
| Parent Post-D2 Repair | [Execution plan](docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md), [agent addendum](docs/AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md) |
| Parent Phase B | [Execution plan](docs/exec-plans/active/PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md), [agent addendum](docs/AGENTS_PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD_ADDENDUM.md) |

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

For implementation changes, use focused tests while iterating and add regression
coverage for changed behavior or invariants where needed. Before completion, run
the standard checks below and any checks required by the active task. Running
these checks and fixing failures caused by the requested change are authorized
parts of implementation work; they do not authorize research experiments.

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

For documentation-only changes, check links, consistency, and the diff; no model
training or full Python suite is needed unless a task explicitly requires it.
Do not add tests that merely mirror reversible, low-impact edits. Once required
checks pass, repeat or broaden testing only for new changes, failures, or unresolved
concerns. Report checks not run and their reasons; never label them as passing.

Default experiments must fit the single RTX 5060 Ti (16 GB VRAM), 64 GB RAM
workstation. Keep unit tests lightweight; multi-hour sweeps belong only to explicit
milestone experiments. Do not stop other processes in the shared environment.
Unless the user changes scope, do not introduce pretrained LMs, RL controllers,
vector databases, distributed training, custom CUDA/Triton kernels, architecture
search, dense-teacher circuit extraction, or neuromorphic work.

## Evidence and completion

- Evaluate the requested acceptance criteria explicitly. Distinguish task
  completion, recovery-gate status, and research-gate status.
- Once all required verifications and task criteria pass, commit the completed
  task artifacts to git with a concise and descriptive commit message. Check
  `git status` and diffs before staging to avoid including transient files,
  caches, or unrelated changes. Keep commits local; do not push to remote
  unless explicitly instructed.
- A failed scientific gate still requires an evidence handoff and a local commit
  of the task's tracked code/documentation when verification is complete. Preserve
  gitignored run artifacts in place and report the gate as FAIL, not PASS.
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

Keep project skills limited to reusable APC workflows that need non-obvious
guidance. Use a short, precise description and link conditional detail from
`SKILL.md`; do not duplicate this guide or turn historical task recipes into
automatic instructions. Shared installed skills are not project-owned files.
