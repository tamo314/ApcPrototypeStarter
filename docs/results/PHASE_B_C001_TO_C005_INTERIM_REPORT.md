> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Phase B B-C001–B-C005 Interim Implementation and Result Report

**Date:** 2026-09-06

## Scope and conclusion

This report records the Phase B work completed through B-C005. It is an
interim engineering and experimental record, not a Phase B completion claim.

| Task | Purpose | Status | Evidence |
|---|---|---|---|
| B-C001 | Activate Phase B and freeze the explicit-TaskSpec upper bound | PASS | Protocol/config tests |
| B-C002 | Register holdout families and enforce preflight checks | PASS | Focused protocol tests |
| B-C003 | Unseen-family lifecycle, explicit TaskSpec (STOP GATE B1) | PASS | Five-seed CUDA lifecycle run; ADR-0074 |
| B-C004 | Construct hard-negative diagnostic infrastructure | Infrastructure PASS | One-seed, 80-cell diagnostic matrix |
| B-C005 | Hard-negative routing and functional safety (STOP GATE B2) | **FAIL** | Five-seed, 400-cell CUDA matrix; ADR-0075 |

The Phase B controller/lifecycle generalizes to the sealed unseen families
when an explicit TaskSpec is visible (B-C003). However, the frozen learned
router does not satisfy the predeclared top-1 requirements under L2–L4
semantic/argument competition (B-C005). The functional verifier did reject
wrong candidates, but false-plastic decisions on known episodes exceeded the
gate bound. By the Phase B STOP-GATE rule, B-C006 onward, including all Task
Inference tasks, are blocked.

## Preserved experimental boundary

All B-C001–B-C005 runtime experiments retained explicit TaskSpec and canonical
operation-ID visibility. No Task Inference mechanism was introduced. The
causal content path remains task-blind: benchmark primitive calls obtain their
state through the content-only encoder path, while task information is used
only for router/candidate proposal. The B-C005 hard-negative level and
provenance remain evaluation-only fields and are not router inputs.

The persistent bank and router were frozen for every hard-negative matrix
cell. L4 is represented as a logical candidate that reuses an existing
parameterized primitive with an incorrect argument; it does not add a
per-argument persistent primitive.

## B-C001 — Phase activation and frozen baseline protocol

### Implementation

B-C001 activated Phase B in repository navigation and introduced a serializable
`PhaseBProtocol` contract. It records the family split, TaskSpec and operation
ID visibility, task-inference modality, support/inference/verification/query
counts, hard-negative level, bank size, candidate budget, and frozen Phase A.2
controller/plastic/incremental-router configurations.

Primary implementation files:

- `src/apc/meta/phase_b_protocol.py`
- `configs/phase_b_baseline_upper_bound.yaml`
- `tests/test_phase_b_protocol.py`
- Phase B plan, task, experiment, and design documents under `docs/`

### Result

Protocol serialization/validation and explicit-upper-bound tests passed. This
task made no scientific-performance claim and did not retrain the controller
or retune a threshold.

### Commit

`8618a89 Task B-C001: Activate Phase B documentation and freeze baseline protocol`

## B-C002 — Holdout registry, identifiability, and leak audit

### Implementation

B-C002 added a family registry with mutually exclusive `DEV_FAMILIES`,
`SEALED_FAMILIES`, and `RETIRED_FROM_SEALED` states. It provides deterministic
same-length holdout generators, family metadata, recurrence instances,
identifiability/collision detection, novelty-validity prechecks, and a leak
audit that rejects evaluation-only family/operation metadata in forbidden
runtime inputs.

Primary implementation files:

- `src/apc/environments/holdout_families.py`
- `src/apc/evaluation/holdout_protocol.py`
- `src/apc/environments/task_spec.py`
- `tests/test_phase_b_holdout_protocol.py`

### Result

The focused checks passed: seeded generators reproduce, development and sealed
partitions are disjoint, ambiguity is surfaced rather than classified as a
model error, sealed-family metadata is excluded from model-visible inputs, and
the selected sealed operations pass the novelty-validity precheck before the
B-C003 lifecycle measurement.

### Commit

`ea85e8e Task B-C002: Holdout-family registry, identifiability checks, and leak audit`

## B-C003 — STOP GATE B1: unseen-family lifecycle

### Design and implementation

B-C003 evaluates the complete explicit-TaskSpec lifecycle for sealed
`MAJORITY_THREE` and `NEIGHBOR_MAX` families:

```text
direct verification -> composition verification -> PLASTIC_SEARCH
-> compact learning -> consolidation -> shadow validation -> promotion
-> temporary-capacity release -> fresh-runtime recurrence
```

The controller, thresholds, Stable Core, stable bank, compact-first policy,
and fallback role were frozen. Legacy K/C/R controls are interleaved with the
sealed novel episodes.

Primary implementation files:

- `src/apc/evaluation/unseen_family_lifecycle_benchmark.py`
- `scripts/run_phase_b_unseen_family_lifecycle.py`
- `tests/test_unseen_family_lifecycle.py`

### Five-seed CUDA result — PASS

| Criterion | Threshold | Measured |
|---|---:|---:|
| Plastic trigger rate | >= 95.0% | 100.00% |
| Mean final novel EM | >= 95.0% | 97.66% |
| Worst-seed novel EM | >= 90.0% | 95.31% |
| Promotion | exactly one/capability | 10/10 |
| Workspace leaks | 0 | 0 |
| Fresh-runtime recurrence EM | >= 95.0% | 97.19% |
| Recurrence adaptation / temporary params / bank growth | 0 / 0 / 0 | 0 / 0 / 0 |
| Maximum old-task EM / route top-1 drop | <= 1.0 pp | 0.00 pp / 0.00 pp |
| Legacy false plastic | <= 1.0% | 0.00% |

This supports lifecycle generalization under the explicit-TaskSpec upper bound.
It does not support semantic Task Inference, because operation IDs remained
visible by design.

Artifacts: `runs/phase_b_unseen_family_lifecycle_gate/`.

### Commit and ADR

- `8c244ab Task B-C003: STOP GATE B1 explicit-TaskSpec unseen-family lifecycle (ADR-0074)`
- `ADR-0074` in `docs/DECISIONS_PHASE_B.md`

## B-C004 — Hard-negative diagnostic infrastructure

### Design and implementation

B-C004 constructed an evaluation-only difficulty ladder around a frozen router
and bank:

| Level | Competitor construction |
|---|---|
| L0 | Orthogonal score-space distractor |
| L1 | Seeded random score-space competitor |
| L2 | Controlled near-neighbor key |
| L3 | Real semantically related learned primitive key |
| L4 | Same parameterized family with a wrong argument |

The builder is non-mutating. It snapshots router and primitive parameters,
records competitor provenance/similarity, and checks actual sparse execution.
Only selected primitives execute. L4 reuses the existing physical primitive ID,
with an evaluator-only logical candidate carrying the wrong argument.

Primary implementation files:

- `src/apc/evaluation/hard_negative_routing_benchmark.py`
- `scripts/hard_negative_routing_benchmark.py`
- `configs/phase_b_hard_negative_diagnostic.yaml`
- `tests/test_hard_negative_routing_benchmark.py`

### One-seed diagnostic result — infrastructure PASS

The run covered 80 cells: `N={16,32,64,128}` × L0–L4 × four parameterized
targets (`SHIFT`, `SELECT`, `COUNT`, `BIND`). It passed matrix completeness,
measurable difficulty ordering, non-mutation of router/primitive parameters,
metadata leak audit, and zero unselected primitive calls.

This result deliberately did **not** measure functional candidate acceptance,
false plastic, or the B2 STOP-GATE thresholds.

Artifacts: `runs/phase_b_hard_negative_diagnostic/`.

### Commit

`a507526 Implement Phase B hard-negative diagnostics`

## B-C005 — STOP GATE B2: hard-negative routing and functional safety

### Design and implementation

B-C005 extends B-C004 with a frozen-router safety layer. For each cell it:

1. ranks logical candidates from task-side state without passing evaluation
   metadata to the router;
2. gathers the declared top-5 candidate proposal set;
3. executes each proposed logical candidate only on an independent support set;
4. accepts a candidate only when support exact match reaches the predeclared
   `0.95` adequacy threshold;
5. executes the first accepted candidate on query examples, whose targets were
   unavailable during acceptance; and
6. logs raw retrieval, acceptance, controller/final-action, compute, sparse
   execution, and mutation-audit evidence separately.

For L4, candidate identity includes both physical primitive and arguments, so a
wrong argument cannot be evaluated as if it were the correct call merely
because both calls share a primitive ID.

New files:

- `scripts/run_phase_b_hard_negative_safety_gate.py`
- `configs/phase_b_hard_negative_safety_gate.yaml`

Extended files:

- `src/apc/evaluation/hard_negative_routing_benchmark.py`
- `tests/test_hard_negative_routing_benchmark.py`

### Final five-seed CUDA matrix

Configuration: five seeds `(0,1,2,3,4)`, `N={16,32,64,128}`, L0–L4, four
parameterized target operations, 32 independent support examples, 64 query
examples, top-k `=5`, and support EM adequacy threshold `=0.95`. This is 400
matrix cells. The final artifact records deterministic-algorithm mode.

At N=128:

| Level | Top-1 | Top-k inclusion | B2 top-1 threshold | Result |
|---|---:|---:|---:|---|
| L0 | 1.000 | 1.000 | >= 0.98 | PASS |
| L1 | 1.000 | 1.000 | >= 0.98 | PASS |
| L2 | 0.866 | 1.000 | >= 0.98 | FAIL |
| L3 | 0.662 | 1.000 | >= 0.95 | FAIL |
| L4 | 0.504 | 1.000 | >= 0.90 | FAIL |

Functional-safety and final-action metrics:

| Metric | B2 threshold | Measured | Result |
|---|---:|---:|---|
| False functional acceptance of wrong candidate | <= 1% | 0.0% | PASS |
| Mean closed-loop EM | >= 0.95 | 0.959 | PASS |
| Mean false plastic on known episodes | <= 2% | 3.75% | FAIL |
| Unselected primitive forward calls | 0 | 0 in every cell | PASS |
| Router/primitives mutate during matrix | forbidden | no mutation | PASS |
| Evaluation metadata leaks to routing | forbidden | none detected | PASS |

### Interpretation and failure localization

The observed first failure is retrieval ranking, not unsafe acceptance:

- The correct candidate is always present in top-5, so candidate **coverage**
  is not the limiting measurement.
- L2–L4 top-1 declines sharply as the negative approaches task-key or
  same-family/argument semantics. The frozen router does not sufficiently
  distinguish these candidates.
- Wrong proposed candidates were rejected by functional verification. Thus the
  verifier is conservative in this matrix; the result is not evidence that
  APC executes a verified-wrong computation.
- False plastic occurs when no proposed candidate reaches support adequacy. On
  this all-known matrix it is a controller/adequacy efficiency failure. It
  should be investigated without using query targets and without retuning a
  threshold based on this result.

These are failure-localization conclusions, not a prescription to increase the
router size, alter thresholds, or continue Task Inference. The next work must
isolate candidate proposal and support adequacy while preserving this failed
measurement as historical evidence.

Artifacts: `runs/phase_b_hard_negative_safety_gate/`.

### ADR

`ADR-0075` records the failed STOP GATE B2 and blocks B-C006 onward.

## Verification and reproducibility notes

The B-C005 implementation passed:

- `python -m pytest -q tests/test_hard_negative_routing_benchmark.py` — 5 passed
- `python -m pytest -q tests/test_phase_b_protocol.py --basetemp /tmp/pytest_phase_b_protocol_codex` — 4 passed
- `python -m ruff check .` — passed
- `python -m mypy src/apc` — passed for 106 source files

The full `python -m pytest -q` command was started but stopped after detecting
two pre-existing full-suite processes in the shared WSL environment. Only the
newly started duplicate process was interrupted; the pre-existing processes
were not modified. This report therefore does not claim a fresh full-suite
PASS for the B-C005 commit.

## Current phase state

Phase B remains active. B-C001–B-C004 provide protocol, holdout, lifecycle,
and hard-negative infrastructure evidence. B-C005 is a valid negative STOP
GATE result. No B-C006 or Task Inference implementation was started after the
failure.
