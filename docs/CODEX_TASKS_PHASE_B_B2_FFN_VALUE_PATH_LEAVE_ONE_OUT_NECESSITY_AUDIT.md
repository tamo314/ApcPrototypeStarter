> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Task — I03 FFN–Value-Path Leave-One-Out Necessity Audit & Freezeability Gate

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004M`/ADR-0109) as `B-C005REC-004N`. This document is written by
the implementing agent from that instruction, not supplied by the user as a
file; it exists so the task has the same durable, re-readable record every
other `B-C005REC-00N` task has. The user's own instruction used the
shorthand `C`/`V`/`O`/`F` for `CONTENT_PREP`/`V_PROJECTION`/`ATTN_OUT_PROJ`/
`FFN_BLOCK` — this document keeps that shorthand alongside REC-004M's own
full names.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004M` (ADR-0109) found `value_path_subcomponent_decision:
DISTRIBUTED_WITHIN_VALUE_PATH`: `F_ALLV` (FFN_BLOCK + all four VALUE_OUTPROJ
subcomponents — CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ,
SCORE_PROJECTION_CONTROL) reproduces REC-004L's own perfect `F_V` recovery,
but no SINGLE subcomponent suffices alone; `CONTENT_PREP`/`ATTN_OUT_PROJ`
rolled back alone each actively hurt relative to `FFN_BLOCK` alone, and
`SCORE_PROJECTION_CONTROL` (Q/K) was independently confirmed exactly
O1-invariant two ways. Since `SCORE_PROJECTION_CONTROL` never enters O1's
forward graph at all, `F_ALLV`'s O1 result is identical to a hypothetical
`FFN_BLOCK + CONTENT_PREP + V_PROJECTION + ATTN_OUT_PROJ` condition that
never touches Q/K — i.e. REC-004M's own `F_ALLV` number already **is** this
task's `F_CVO` comparator under O1, and this task's Stage A source replay
treats reproducing it as a strong parity check rather than assuming it.

This task asks exactly one question, from the OTHER direction of REC-004M's
one-at-a-time search: starting from the already-established sufficient set
`FFN_BLOCK + CONTENT_PREP + V_PROJECTION + ATTN_OUT_PROJ` (`F_CVO`), remove
exactly one of the three VALUE_OUTPROJ subcomponents at a time — does a
**pairwise** (leave-one-out) subset stay sufficient, and in particular, is
`FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ` (`F_VO` — i.e. `CONTENT_PREP`
NEVER rolled back) sufficient? If so, this is the first candidate freeze set
that never touches `CONTENT_PREP` — the one subcomponent REC-004M's own
contract explicitly flagged as unsafe to freeze because it feeds the SAME
`kv` tensor the real, learnable K path (`SCORE_PROJECTION_CONTROL`) also
reads from.

**Zero new optimizer updates.** Reads the SAME already-saved I03 checkpoints
REC-004K/L/M used (step=6000 from `B-C005REC-004D`'s tree, step=17500 from
`B-C005REC-004H`'s), plus — success-model safety check only, and only if
`F_VO` passes — I04@18000's and I05@17500's own step=6000/late checkpoints
(never another init's weights, never transplanted into I03). Every
condition is a fresh, in-memory, evaluation-only `CrossPositionLengthBiasPrimitive`
merged from two real state dicts via REC-004M's own row-slice-aware merge
function, reused UNMODIFIED — never a new checkpoint file, never a mutation
of either loaded primitive in place. `selected_init`, `selected_step`,
`selected_value_subcomponent`, and `child_bundle` stay `null`; `rg3_recheck`
stays `"NOT_EXECUTED"`; `rec005_eligible` stays `false`, unconditionally.
This task never trains, never freezes anything in a real training run,
never feeds `pi_n` into a forward input, never extends position bias, and
never changes Core.

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004M history paragraph (ADR-0096..0109).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0106
   (REC-004K), ADR-0108 (REC-004L, `FFN_VALUE_OUTPROJ_INTERACTION_SUFFICIENT`),
   and ADR-0109 (REC-004M, `DISTRIBUTED_WITHIN_VALUE_PATH`, the anchor this
   task refines from the opposite direction).
3. `src/apc/evaluation/mirror_ffn_value_path_subcomponent_attribution.py`
   (REC-004M) — `build_value_path_subcomponent_key_groups`,
   `_build_subcomponent_rollback_primitive` (the general n-subcomponent
   row-slice-aware merge — reused UNMODIFIED for every condition in this
   task, including the new leave-one-out combinations, since it already
   accepts an arbitrary `subcomponent_ids` tuple), `verify_qk_rollback_is_o1_invariant`,
   `build_length10_value_path_subcomponent_probe_v1` (the collision-
   substitution recipe this task's own new dataset follows verbatim),
   `build_stage_c_datasets` (REC-004M's own 4-dataset byte-identical
   assembly, reused as this task's starting point before adding a 5th).
4. `src/apc/evaluation/mirror_ffn_anchored_downstream_interaction_audit.py`
   (REC-004L) — `compute_interaction_diagnostic_point` (O1-only metrics,
   reused unmodified for every Stage B condition) and `_dataset_digest`.
5. `src/apc/evaluation/mirror_temporal_mechanism_rollback_audit.py`
   (REC-004K) — `compute_rollback_diagnostic_point` (J0 AND O1 both,
   reused unmodified for the success-model safety check).
6. `src/apc/evaluation/mirror_position_score_residual_audit.py` (REC-004E) —
   `run_intervention_forward` (the real `primitive.cross_attn(...,
   need_weights=True, average_attn_weights=False)` path for J0, used here
   for the first time to compare two DIFFERENT primitives' attention output
   rather than one primitive across J-conditions), `_prepare_query_kv`,
   `_manual_attention` (its `s_other` return is pure Q/K^T score, computed
   before any bias/mask is added — reused here to extract the pre-softmax
   raw score for the same comparison).
7. `src/apc/primitives/primitive.py` — `CrossPositionLengthBiasPrimitive`:
   confirms `content_in_proj`/`content_position_embedding` (CONTENT_PREP)
   and `cross_attn`'s fused `in_proj_weight`/`in_proj_bias` Q/K rows feed
   `_prepare_query_kv`'s `query`/`kv` and the attention score, while
   `V_PROJECTION` (the V row-slice of the SAME fused tensor),
   `cross_attn.out_proj` (ATTN_OUT_PROJ), and `ffn` (FFN_BLOCK) feed only
   the post-score value/output path — the architectural basis for this
   task's Stage C claim.

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those trees, or in
  `rec004i/`/`rec004j/`/`rec004k/`/`rec004l/`/`rec004m/`.
- Read every checkpoint exactly as saved; `strict=True` state-dict load,
  hash-verified against its own source task's recorded
  `checkpoint_state_hash` before trusting its forward pass.
- `clean_v2_length10`, `length10_mechanism_probe_v1`,
  `length10_downstream_interaction_probe_v1`, and
  `length10_value_path_subcomponent_probe_v1` are regenerated via REC-004K's/
  REC-004L's/REC-004M's own functions, unmodified — byte-identical to their
  existing saved sets. `length10_value_path_necessity_probe_v1` (new) is a
  pure function of `(seed, protected_digests)` — never conditioned on any
  checkpoint prediction or rollback result — and its frozen digest is
  written to disk before any Stage B/C/decision point is computed on it.
- `C`/`V`/`O`/`F` (the user's own shorthand) map exactly to REC-004M's real,
  self-verified key groups `CONTENT_PREP`/`V_PROJECTION`/`ATTN_OUT_PROJ`/
  `FFN_BLOCK` — never redefined here. `SCORE_PROJECTION_CONTROL` (Q/K) is
  never rolled back by any Stage B condition in this task; it is used only
  for the Stage A parity replay (reproducing REC-004M's own zero-diff
  finding) and is otherwise held at I03@17500's real, current value in
  every condition.
- Stage B never searches component combinations beyond the fixed
  `{R0, F, F_CV, F_CO, F_VO, F_CVO, EARLY}` set. No additional condition is
  added after seeing intermediate results. No 1-subcomponent condition
  (REC-004M's own `F_C`/`F_V`/`F_O`) is recomputed here — this task starts
  from the 3-subcomponent sufficient set and removes one at a time, the
  opposite direction from REC-004M's own search.
- Every threshold used by a decision rule is fixed in this document before
  any point is computed, and is never widened after seeing a result.
- The J0 attention-equivalence check (Stage C) never substitutes oracle
  attention (`pi_n`) for anything — it compares two primitives' REAL,
  learned attention (via the production `cross_attn` module with
  `need_weights=True`) under ordinary forward computation.

## 3. Stage A — source/parity reconfirmation (STOP gate)

Before any new leave-one-out condition is trusted, reproduce, on the four
datasets REC-004K/L/M already established
(`clean_v2_length10`, `length10_mechanism_probe_v1`,
`length10_downstream_interaction_probe_v1`,
`length10_value_path_subcomponent_probe_v1`):

```
F      (FFN_BLOCK alone, @6000)        oracle EM in [0.85, 0.92]  (REC-004L/M's own 0.87-0.90)
F_CVO  (FFN_BLOCK+C+V+O, @6000)        oracle EM == 1.0000 on every dataset
EARLY  (I03@6000's own real forward)   oracle EM == 1.0000 on every dataset
Q/K rollback (F_QK vs F, real weights) max_abs_logit_diff == 0.0
```

`F_CVO` is cross-checked against REC-004M's own saved
`condition_points_summary.json` `F_ALLV` entries (informationally identical
under O1, since `SCORE_PROJECTION_CONTROL` is inert there); `F`/`EARLY`/`R0`
are cross-checked against REC-004M's own `F`/`EARLY`/`R0` entries the same
way. `abs(recorded_em - recomputed_em) < 1e-9` on every shared dataset is
required. The Q/K check reuses REC-004M's own `verify_qk_rollback_is_o1_invariant`
unmodified.

**If any of the four checks above fails, the task stops with
`SOURCE_REPLAY_MISMATCH`** — no `F_CV`/`F_CO`/`F_VO` condition is computed
or interpreted, and no dataset construction proceeds past what Stage A
itself needed.

## 4. Stage B — three fixed leave-one-out conditions, plus comparators

From the already-established sufficient set `F_CVO = FFN_BLOCK + CONTENT_PREP
+ V_PROJECTION + ATTN_OUT_PROJ`, remove exactly one VALUE_OUTPROJ
subcomponent at a time. Exactly these fixed conditions — no combination
beyond this set, no condition added after seeing a result:

```
R0     = I03@17500 as-is
F      = FFN_BLOCK rolled back to step=6000
F_CV   = FFN_BLOCK + CONTENT_PREP + V_PROJECTION rolled back to step=6000   (O excluded)
F_CO   = FFN_BLOCK + CONTENT_PREP + ATTN_OUT_PROJ rolled back to step=6000  (V excluded)
F_VO   = FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ rolled back to step=6000  (C excluded)
F_CVO  = FFN_BLOCK + CONTENT_PREP + V_PROJECTION + ATTN_OUT_PROJ rolled back to step=6000
EARLY  = I03@6000's own real, unmodified checkpoint
```

This is NOT a 3-subcomponent search; it is three pre-registered leave-one-
out necessity tests against the already-confirmed minimal sufficient
candidate `C+V+O` (plus the two comparators already computed in Stage A and
`R0`/`EARLY`). Every condition reuses REC-004M's own
`_build_subcomponent_rollback_primitive` unmodified — it already accepts an
arbitrary subcomponent tuple and composes row-slices correctly, so no new
merge logic is written for this task.

All seven conditions run on **all five** datasets (the four from Stage A
plus Stage D's new set below), under O1 exclusively (oracle-attention
substitution, per REC-004K/L/M's own established decoupling from the
attention-score-learning problem).

## 5. Stage C — `F_VO` score-safety: J0 attention equivalence

`F_VO` never rolls back `CONTENT_PREP` or `SCORE_PROJECTION_CONTROL` (Q/K)
— both stay at I03@17500's real, current value, as does position bias and
`QUERY_RESIDUAL_PATH`. Since `V_PROJECTION`, `ATTN_OUT_PROJ`, and
`FFN_BLOCK` feed only the post-attention-score path (never the score
itself), `F_VO`'s J0 (normal, non-oracle) attention computation is expected
to be **mathematically identical** to `R0`'s: same `query`/`kv` tensors (same
weights, same content), same position-bias additive term, hence same
pre-softmax score and same post-softmax attention distribution.

This is checked empirically, not assumed: on a fixed length-10 batch,
compute (via `run_intervention_forward(..., "J0")` through the REAL
`cross_attn` module, `need_weights=True`) `F_VO`'s and `R0`'s post-softmax
`attn_probs`, plus (via `_manual_attention`'s `s_other` return) the raw
pre-softmax Q/K score, for both. Report every intermediate diff (`query`,
`kv`, position bias, raw score, `attn_probs`). A single pre-fixed absolute
tolerance (`1e-6`, matched to float32 forward-pass noise on this backend —
literal `0.0` is the expected, not merely tolerated, result given identical
weights) gates `attention_score_path_invariant`. This check runs
unconditionally (regardless of whether `F_VO` clears the Stage D floor) —
it is a mechanistic property check, not a decision-rule input by itself.

## 6. Stage D — new confirmation dataset

The prior four datasets are already consumed by REC-004K/L/M diagnostics.
Build one new dataset, `length10_value_path_necessity_probe_v1` (512
examples), via REC-004M's own collision-substitution recipe verbatim (new
split label, hence a new RNG stream via `_derive_local_seed`), with a
protected set that is the union of: the training stream (steps 1-18000),
`rec004a_budget_validation` (old validation), every REC-004A-D
reference/sealed split (all via `traj_audit.build_protected_digest_registry`),
`clean_selection_validation_v2`'s full 1024-example set,
`length10_mechanism_probe_v1`, `length10_downstream_interaction_probe_v1`,
and `length10_value_path_subcomponent_probe_v1`. Its digest is frozen to
disk BEFORE any Stage B/C/decision point is computed on it.
Development-exposed diagnostic data, not sealed/RG3 — an `RG3` query set is
never generated or consumed by this task.

## 7. Decision rule

For each of `{F_CV, F_CO, F_VO}`, on **all five** datasets:
`oracle_sequence_exact_match >= 0.95 AND position_4_oracle_accuracy >= 0.95`.
Priority order (fixed here, before any point is computed):

1. **`F_VO` passes** → `FFN_V_OUTPROJ_SUBPATH_SUFFICIENT` +
   `CONTENT_PREP_NOT_REQUIRED_FOR_O1_RECOVERY`. If Stage C's J0 attention
   equivalence check also passes (`attention_score_path_invariant: true`),
   additionally append `SCORE_PATH_PRESERVED`. This is the most useful
   result: it would mean, for the first time, that a freeze candidate
   (`FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ`) exists that never touches
   `CONTENT_PREP` and is empirically shown not to move the real attention
   score — the necessary precondition (not yet an authorization) for a
   future `B-C005REC-004O` training-repair design that freezes
   `FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ` after step=6000 while leaving
   `CONTENT_PREP`/Q/K/position bias free to keep learning past it.
2. **Else, `F_CV` or `F_CO` passes (but not `F_VO`)** →
   `SHARED_CONTENT_PREP_REQUIRED` + `FREEZE_REPAIR_NOT_SCORE_SAFE`. Since
   `CONTENT_PREP` feeds the same `kv` tensor the real, learnable K path also
   reads from (REC-004M's own disclosed caveat), this does NOT propose a
   freeze-based training repair; the next open lead would be an
   architecture-level question (separating K's and V's input
   representations) — not authorized or designed by this task.
3. **Else, `F_CVO` passes but none of `F_CV`/`F_CO`/`F_VO` do** →
   `ALL_THREE_VALUE_SUBPATHS_JOINTLY_NECESSARY_WITH_FFN`. This does not
   authorize a further subset search (there is no smaller subset left to
   test without re-deriving REC-004M's own single-subcomponent results),
   and — since `CONTENT_PREP` is one of the three and is shared with the
   real K path — does NOT authorize proceeding to a whole-value-path
   freeze design.
4. **Else** (should not occur given Stage A's own `F_CVO` replay check, but
   handled for completeness) → `VALUE_PATH_NECESSITY_INCOMPLETE`.

"More than one of `F_CV`/`F_CO`/`F_VO` passes" is not treated as a separate
branch: per rule 1's priority, if `F_VO` is among the passing set its label
governs regardless of what else also passes (which is recorded in
`per_condition_passes` either way, never hidden).

## 8. Success-model safety check (conditional on `F_VO` passing)

Runs ONLY if the Stage 7 label is `FFN_V_OUTPROJ_SUBPATH_SUFFICIENT`.
Applies the SAME `FFN_BLOCK + V_PROJECTION + ATTN_OUT_PROJ` rollback to:

```
I04 @18000   # currently-stable success case (REC-004I/J)
I05 @17500   # success case just before its own step=18000 collapse
```

each using **that init's own** step=6000 state (never I03's, never any
cross-init weight transplant; no weight is ever moved from a successful
init into I03). For each, measures on all five Stage D datasets:

- the SAME J0 attention-equivalence check as Stage C (rollback vs that
  init's own real late-step baseline) — is the attention distribution also
  preserved for I04/I05, not just I03?
- J0 (normal-forward) sequence EM before/after the rollback, via REC-004K's
  own `compute_rollback_diagnostic_point` reused unmodified — the decisive
  safety metric, since a real repair must work without oracle substitution.
- O1 (oracle) sequence EM before/after, for completeness.

A J0 EM drop `>= 0.05` on any dataset (pre-registered here, descriptive,
never gates the Stage 7 label) is recorded as `DEGRADATION_OBSERVED` for
that init; otherwise `NO_DEGRADATION_OBSERVED`. No candidate is adopted
here.

## 9. Deliverable — `next_step_repair_contract.md`

Written regardless of outcome, mirroring REC-004M's own
`next_training_repair_contract.md` pattern. Only if the Stage 7 label is
`FFN_V_OUTPROJ_SUBPATH_SUFFICIENT` AND Stage C's `attention_score_path_invariant`
is true does it propose (status `PROPOSED_NOT_AUTHORIZED`, never implemented
by this task) the two-stage training design already sketched in Section 7,
tentatively `B-C005REC-004O`. If Stage 8's safety check shows
`DEGRADATION_OBSERVED` for I04 or I05, that is recorded as a disclosed risk
factor, not a blocker of the proposal text itself (per the same
non-adoption charter every REC-00N diagnostic task uses). If the label is
anything else, the file records `status: NOT_PROPOSED` and why, per Section
7's own rules.

## 10. Forbidden in this task

- Any new optimizer update.
- Any condition beyond the fixed `{R0, F, F_CV, F_CO, F_VO, F_CVO, EARLY}`
  set (no `F_C`, `F_V`, `F_O`, `F_QK` recomputation as new Stage B
  conditions — those are REC-004M's own single-subcomponent results, read
  only via the Stage A cross-check).
- Feeding `pi_n` into a model input.
- FFN retraining, an actual VALUE-path freeze training run, or any change
  to Core.
- Extending position bias.
- Adopting a best init, best checkpoint, or best condition as a candidate.
- Any child bundle.
- Any RG3 recheck.
- Starting `B-C005REC-005`.
- Any cross-init weight transplant (I04/I05's own step=6000 state is used
  only for I04/I05's own safety check, never copied into I03 or vice versa).

## 11. Fixed non-adoption fields

Regardless of outcome:

```
new_optimizer_updates = 0
selected_init = null
selected_step = null
selected_value_subcomponent = null
child_bundle = null
rg3_recheck = NOT_EXECUTED
rec005_eligible = false
```

## 12. Deliverables

- `src/apc/evaluation/mirror_ffn_value_path_leave_one_out_necessity_audit.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004n.yaml`
- `tests/test_mirror_ffn_value_path_leave_one_out_necessity_audit.py`
  (CPU-only, synthetic fixtures in the established pattern)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004n/run_001/`:
  dataset manifests (all five), `checkpoint_load_info.json`,
  `value_path_subcomponent_key_groups.json`, `source_replay_check.json`,
  `qk_rollback_invariance_check.json`, `condition_points.jsonl`,
  `j0_attention_equivalence_check.json`,
  `value_path_necessity_decision.json`, `safety_check.json` (or a
  `NOT_EXECUTED` stub), `freeze_audit.json`, `side_effect_audit.json`,
  `cost_accounting.json`, `summary.json`, `report.md`,
  `next_step_repair_contract.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
  plus its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history
  update.

## 13. Completion report format

```
Task: B-C005REC-004N
Stage A: source replay status (F range, F_CVO/EARLY perfect, Q/K diff=0),
  cross-check vs REC-004M
Stage B: R0/F/F_CV/F_CO/F_VO/F_CVO/EARLY results (EM, position-4/5
  accuracy), all five datasets
Stage C: J0 attention equivalence check for F_VO vs R0 (query/kv/bias/
  raw-score/attn_probs diffs, tolerance, verdict)
Decision label + tags, with numbers
Stage 8 safety check: executed or not, per-init J0/O1 deltas + J0 attention
  equivalence if executed
next_step_repair_contract.md: proposed or not, and why
new_optimizer_updates=0 / selected_init=null / selected_step=null /
  selected_value_subcomponent=null / child_bundle=null /
  rg3_recheck=NOT_EXECUTED / rec005_eligible=false
freeze/side-effect audit result
real tests / commands / hashes
```
