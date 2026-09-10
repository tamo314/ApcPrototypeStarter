# AI Coding Task — FFN-Anchored Downstream Interaction Audit

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004K`/ADR-0106 and the environment-repair sub-task
`B-C005REC-004L-ENV1`/ADR-0107) as `B-C005REC-004L`. This document is
written by the implementing agent from that instruction, not supplied by
the user as a file; it exists so the task has the same durable, re-readable
record every other `B-C005REC-00N` task has.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004K` (ADR-0106) found I03@17500's position-4 oracle-attention
residual is `DISTRIBUTED_DOWNSTREAM_COADAPTATION`: rolling back any ONE of
five forward-graph-derived component groups (`VALUE_OUTPROJ`,
`QUERY_RESIDUAL_PATH`, `POST_ATTN_NORM`, `FFN_BLOCK`, `READOUT`) from
step=17500 to step=6000 does not clear the pre-registered
`oracle EM >= 0.95 AND position-4 accuracy >= 0.95` floor, though
`FFN_BLOCK` alone was by far the strongest single lever (oracle EM
0.75 -> 0.89-0.90) — a residual ~5-6 point shortfall remained unexplained.
This task asks exactly one question: **which downstream component's
INTERACTION with `FFN_BLOCK` explains that remaining shortfall?**

**Stage 0 — environment preflight (mandatory, gates everything after it).**
Before any experiment code runs: confirm `scipy` is importable, confirm the
current dependency manifest, and confirm the state of the (previously
scipy-blocked) baseline tests. If `scipy` is still missing, this task stops
immediately with the label `ENVIRONMENT_DEPENDENCY_BLOCKED` — dependency
repair is out of scope here and must be its own separate task. (This gate
was satisfied in-session: `B-C005REC-004L-ENV1`/ADR-0107 had already
declared `scipy` as a core dependency in `pyproject.toml`, and this
session's own re-verification found the shared host `.venv` had not yet
been reinstalled against that manifest; with the user's explicit
permission, `pip install -e .` was run to sync it, `scipy==1.18.1` now
imports, and a full-repo `pytest -q` reconfirmed `2324 passed, 0 failed`.)

**Stage A fixes the target to an FFN anchor.** Base = I03@17500, under O1
(oracle-attention) exclusively. Exactly these fixed conditions are compared
— no 3-component combination, no full subset/powerset search, and no
condition added after seeing a result:

```
R0      = I03@17500 as-is
F       = FFN_BLOCK rolled back to step=6000
F_V     = FFN_BLOCK + VALUE_OUTPROJ rolled back to step=6000
F_Q     = FFN_BLOCK + QUERY_RESIDUAL_PATH rolled back to step=6000
F_N     = FFN_BLOCK + POST_ATTN_NORM rolled back to step=6000
F_R     = FFN_BLOCK + READOUT rolled back to step=6000
R_ALL   = all 5 component groups rolled back to step=6000 (positive control)
EARLY   = I03@6000's own real, unmodified checkpoint
```

Since `FFN_BLOCK` was REC-004K's strongest single lever, this is a
pre-registered "FFN plus exactly one partner" interaction test, not an
exploratory or exhaustive search.

**Stage B adds one new confirmation dataset.** REC-004K's own two datasets
(`clean_v2_length10`, n=230; `length10_mechanism_probe_v1`, n=512) are
reused byte-identically for paired continuity. One new dataset,
`length10_downstream_interaction_probe_v1` (512 examples), is generated —
disjoint from the training stream (steps 1-18000), old validation sets,
`clean_selection_validation_v2`'s full set, `length10_mechanism_probe_v1`
itself, and every other REC-004A-D reference/sealed split. Its digest is
frozen to disk BEFORE any Stage A/C condition is evaluated on it. It is
explicitly NOT an RG3 query dataset — it is registered as
development-exposed diagnostic data.

**Stage C decides, centered on output position 4.** For each of the 7
merge-conditions above, across ALL THREE datasets, record: sequence EM,
position-4 accuracy, position-5 accuracy, position-4 target-token logit
margin, a positions-{0-3,6-9} regression check (descriptive only), the
max-abs hidden-state difference immediately after the O1 attention output
(vs `EARLY`), and the max-abs final-logit difference (vs `EARLY`).

Decision rule — `oracle_sequence_exact_match >= 0.95 AND
position_4_oracle_accuracy >= 0.95`, required on **all three** datasets:

- Exactly one of `{F_V, F_Q, F_N, F_R}` passes ->
  `FFN_<PARTNER>_INTERACTION_SUFFICIENT`.
- More than one passes -> `MULTIPLE_FFN_INTERACTION_PATHS_SUFFICIENT`
  (never resolved by picking the highest EM).
- None passes but `R_ALL` does ->
  `HIGHER_ORDER_OR_MULTI_COMPONENT_COADAPTATION` — this task does NOT
  automatically escalate to a 3-component search on this outcome.
- Neither -> `DOWNSTREAM_DECOMPOSITION_INCOMPLETE`.

**Zero new optimizer updates.** This task never trains, never adds a
checkpoint under `rec004d/`, `rec004g/`, or `rec004h/`, and never re-enters
`frozen_evaluation()`'s guarded surface for anything but read-only runtime
reconstruction. `selected_init`, `selected_step`, and `selected_component`
stay `null`; `child_bundle` stays `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`; `rec005_eligible` stays `false` — unconditionally,
regardless of which label the result matches. This task does not select,
adopt, retrain, or start `B-C005REC-005`, `B-C005R3-011`, `B-C006`, or Task
Inference, even if an FFN-anchored pair fully recovers I03@17500. Rollback
here is a causal-diagnosis intervention on an evaluation-only copy of the
primitive, never a candidate weight change.

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004L-ENV1 history paragraph
   (ADR-0096..0107).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0106
   (REC-004K, the 5-group decomposition and `DISTRIBUTED_DOWNSTREAM_
   COADAPTATION` finding this task anchors on) and ADR-0107 (REC-004L-ENV1,
   the scipy dependency repair that unblocked this task's own Stage 0).
3. `src/apc/evaluation/mirror_temporal_mechanism_rollback_audit.py`
   (REC-004K) — `REC004K_COMPONENT_PREFIXES`, `REC004K_EXCLUDED_FROM_O1_
   PREFIXES`, `partition_state_dict_keys`, `verify_o1_invariance_to_
   excluded_params`, `_build_rollback_primitive`, `build_stage_a_datasets`,
   `run_component_decomposition_parity_check`, `_position_margins_batch`:
   every one of these is reused UNMODIFIED by this task.
4. `src/apc/evaluation/mirror_late_stage_attention_bottleneck_revalidation.py`
   (REC-004J) — `regenerate_clean_selection_validation_v2`,
   `build_length10_mechanism_probe_v1`, `_load_and_verify_primitive`: also
   reused unmodified; `build_length10_mechanism_probe_v1`'s exact
   collision-substitution recipe is the template this task's new dataset
   function follows (different split label, different — larger — protected
   set).
5. `src/apc/evaluation/mirror_oracle_attention_substitution_probe.py`
   (REC-004F) — `_oracle_attention`/`run_oracle_forward`: this task composes
   `_oracle_attention` with `resid_audit._prepare_query_kv`/`_post_
   attention` directly (rather than calling `run_oracle_forward`) solely to
   also expose the intermediate hidden state right after the attention
   output, which `run_oracle_forward` does not return.
6. `src/apc/evaluation/mirror_contamination_free_checkpoint_trajectory_audit.py`
   (REC-004I) — `build_protected_digest_registry`, `_digest`,
   `_digest_examples`: reused unmodified for the new dataset's
   disjointness proof.

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those trees, or in
  `rec004i/`/`rec004j/`/`rec004k/`.
- Read every checkpoint exactly as saved; `strict=True` state-dict load,
  hash-verified against its own source task's recorded
  `checkpoint_state_hash` before trusting its forward pass.
- `clean_v2_length10` and `length10_mechanism_probe_v1` are regenerated via
  REC-004K's own `build_stage_a_datasets`, unmodified — byte-identical to
  REC-004K's own datasets. `length10_downstream_interaction_probe_v1` is a
  pure function of `(seed, protected_digests)` — never conditioned on any
  checkpoint prediction or rollback result — and its frozen digest is
  written to disk before any Stage A/C point is computed on it.
- Stage A/C never searches component combinations beyond the fixed
  `{R0, F, F_V, F_Q, F_N, F_R, R_ALL, EARLY}` set. No additional condition
  is added after seeing intermediate results.
- Every merge-condition (`R0`/`F`/`F_V`/`F_Q`/`F_N`/`F_R`/`R_ALL`) is a
  fresh, freestanding `CrossPositionLengthBiasPrimitive` built via
  REC-004K's own `_build_rollback_primitive`; it never mutates the
  originally loaded step=6000/step=17500 primitive in place, and never
  writes a `.pt` file. `EARLY` is the real, loaded step=6000 primitive
  itself, not a merge.
- Every threshold used by the decision rule is fixed in this document
  before any point is computed, and is never widened after seeing a
  result.

## 3. Stage A/C conditions

| Condition | Definition |
|---|---|
| `R0` | I03@17500 unmodified (baseline) |
| `F` | R0 with `FFN_BLOCK` replaced by I03@6000's values |
| `F_V` | R0 with `FFN_BLOCK` + `VALUE_OUTPROJ` replaced by I03@6000's values |
| `F_Q` | R0 with `FFN_BLOCK` + `QUERY_RESIDUAL_PATH` replaced by I03@6000's values |
| `F_N` | R0 with `FFN_BLOCK` + `POST_ATTN_NORM` replaced by I03@6000's values |
| `F_R` | R0 with `FFN_BLOCK` + `READOUT` replaced by I03@6000's values |
| `R_ALL` | R0 with ALL FIVE groups replaced by I03@6000's values (positive control) |
| `EARLY` | I03@6000's own real, unmodified checkpoint |

**Positive-control parity gate (reused from REC-004K):** `R_ALL`'s O1
forward MUST reproduce `EARLY`'s own real O1 forward within `5e-3`
max-abs-logit tolerance AND exact discrete-prediction agreement, on every
dataset. If this fails, the task stops and reports
`COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP` — no `F`/`F_X`/`R_ALL` result
is interpreted as a causal finding.

## 4. Stage B — new dataset

`length10_downstream_interaction_probe_v1`, 512 examples, generated by
`build_length10_downstream_interaction_probe_v1` — the same
collision-substitution rule as `rec004j.build_length10_mechanism_probe_v1`,
disjoint from `protected_v1` (training stream steps 1-18000, old
validation, every REC-004A-D reference/sealed split) `| digests(clean_
selection_validation_v2's full 1024-example set) | digests(length10_
mechanism_probe_v1)`. Development-exposed diagnostic data, not sealed/RG3.

## 5. Stage C — metrics and decision, centered on position 4

For every `(condition, dataset)` pair: sequence EM (O1), per-output-position
(0-9) accuracy and mean target-token logit margin (O1), max-abs
hidden-state difference immediately after the O1 attention output vs
`EARLY`, and max-abs final-logit difference vs `EARLY`. Positions 0-3 and
6-9 are checked for regression against `R0` (descriptive only — never gates
the decision). Position 4 is the primary residual (REC-004J/ADR-0105);
position 5 is the symmetric control.

**Pre-registered rule (fixed here):** for a given `X in {F_V, F_Q, F_N,
F_R}`, on **all three** datasets, `oracle_sequence_exact_match >= 0.95 AND
position_4_oracle_accuracy >= 0.95`:

- Exactly one `X` satisfies this -> `FFN_<PARTNER>_INTERACTION_SUFFICIENT`.
- More than one -> `MULTIPLE_FFN_INTERACTION_PATHS_SUFFICIENT` (never
  resolved by highest EM).
- None, but `R_ALL` does -> `HIGHER_ORDER_OR_MULTI_COMPONENT_COADAPTATION`
  (no automatic escalation to a 3-component search).
- Neither -> `DOWNSTREAM_DECOMPOSITION_INCOMPLETE`.

## 6. Fixed non-adoption fields

Regardless of outcome: `new_optimizer_updates: 0`, `selected_init: null`,
`selected_step: null`, `selected_component: null`, `child_bundle: null`,
`rg3_recheck: "NOT_EXECUTED"`, `rec005_eligible: false`. No I03 weight is
ever actually rolled back as a saved candidate — every condition in this
task is an in-memory, evaluation-only causal probe.

## 7. Deliverables

- `src/apc/evaluation/mirror_ffn_anchored_downstream_interaction_audit.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004l.yaml`
- `tests/test_mirror_ffn_anchored_downstream_interaction_audit.py`
  (CPU-only, synthetic fixtures in the established pattern)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004l/run_001/`:
  dataset manifests (all three), `checkpoint_load_info.json`,
  `component_key_groups.json`, `forward_graph_invariance_check.json`,
  `condition_points.jsonl`, `cross_check_against_rec004k.json`,
  `component_decomposition_parity_check.json`,
  `ffn_interaction_decision.json`, `regression_check.json`,
  `freeze_audit.json`, `side_effect_audit.json`, `cost_accounting.json`,
  `summary.json`, `report.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
  plus its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history
  update.

## 8. Completion report format

```
Task: B-C005REC-004L
Stage 0: scipy/environment status, resolution taken
Stage A/C: R0/F/F_V/F_Q/F_N/F_R/R_ALL/EARLY results (EM, position-4/5
  accuracy, hidden/logit diff vs EARLY), all three datasets
Component decomposition parity check (R_ALL vs EARLY): status
FFN-anchored interaction decision label, with numbers
new_optimizer_updates=0 / selected_init=null / selected_step=null /
  selected_component=null / child_bundle=null / rg3_recheck=NOT_EXECUTED /
  rec005_eligible=false
freeze/side-effect audit result
real tests / commands / hashes
```
