# AI Coding Task — I03 Matched-Data Temporal Mechanism Recheck & Same-Init Downstream Rollback Audit

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004J`/ADR-0105) as `B-C005REC-004K`. This document is written by
the implementing agent from that instruction, not supplied by the user as a
file; it exists so the task has the same durable, re-readable record every
other `B-C005REC-00N` task has.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004J` (ADR-0105) found that at step=17500, oracle `pi_n`
attention substitution for I03 produces a large paired recovery (delta
+0.54 to +0.56) but does NOT clear the 0.95 oracle-EM floor on either of two
disjoint length-10 datasets (`clean_v2_length10`, `length10_mechanism_probe_v1`)
— `LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM`, residual concentrated at
output positions 4 and 5. This contrasts with `B-C005REC-004F` (ADR-0101),
which found the SAME oracle substitution recovered I03's length-10 EM from
0.086 to 1.000 at step=6000 — but on a DIFFERENT, older diagnostic set (the
length-balanced diagnostic suite, not digest-verified disjoint from training
to REC-004I's standard). Two candidate explanations are confounded in that
comparison: (a) the mechanism genuinely shifted downstream between step=6000
and step=17500 ("temporal shift"), or (b) the two measurements simply used
different data and the step=6000 number would already have been ~0.75 on the
newer, stricter datasets ("dataset effect"). This task localizes, on
matched data, which learned component(s) — changed between step=6000 and
step=17500 within I03's own trajectory — cause the position-4 residual that
survives oracle attention substitution at step=17500.

**Stage A must run first and gates everything after it.** If I03@6000's own
oracle EM on BOTH of REC-004J's own datasets does not clear 0.95, the
temporal-shift premise is not established from matched data, and Stage
B/C/D (component rollback) must NOT run — proceeding would misattribute a
dataset effect to a temporal mechanism shift.

**Zero new optimizer updates.** This task never trains, never adds a
checkpoint under `rec004d/`, `rec004g/`, or `rec004h/`, and never re-enters
`frozen_evaluation()`'s guarded surface for anything but read-only runtime
reconstruction. `selected_init`, `selected_step`, and `selected_component`
stay `null`; `child_bundle` stays `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`; `rec005_eligible` stays `false` — unconditionally,
regardless of which label(s) below the result matches. This task does not
select, adopt, retrain, or start `B-C005REC-005`, `B-C005R3-011`, `B-C006`,
or Task Inference, even if a component rollback fully recovers I03@17500.
**Rollback here is a causal-diagnosis intervention on an evaluation-only
copy of the primitive, never a candidate weight change** — I03's real saved
checkpoint files are never modified, and no "I03 rolled back to step=6000"
artifact is produced as a deliverable of this task.

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004J history paragraph (ADR-0096..0105).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0101
   (REC-004F, step=6000 oracle-attention full recovery for I03 on the
   length-balanced diagnostic suite) and ADR-0105 (REC-004J, step=17500
   mixed/incomplete oracle recovery for I03 on the two matched datasets this
   task reuses, plus the position-4/5 residual localization).
3. `src/apc/evaluation/mirror_late_stage_attention_bottleneck_revalidation.py`
   (REC-004J) — `regenerate_clean_selection_validation_v2`,
   `build_length10_mechanism_probe_v1`, `_load_and_verify_primitive`,
   `_run_j0_decomposition`, `_run_o1`, `compute_diagnostic_point`: every one
   of these is reused UNMODIFIED by this task for Stage A, so Stage A's
   step=17500 row must reproduce REC-004J's own recorded numbers exactly.
4. `src/apc/evaluation/mirror_oracle_attention_substitution_probe.py`
   (REC-004F) — `_oracle_attention`/`run_oracle_forward`: read line by line
   to enumerate every real parameter the O1 forward pass actually touches
   after the attention distribution is substituted (Stage B must be built
   from this reading, not from guessed names).
5. `src/apc/evaluation/mirror_position_score_residual_audit.py` (REC-004E) —
   `_prepare_query_kv`, `_pad_mask`, `_position_bias_raw`, `_manual_attention`,
   `_post_attention`, `_predict_from_logits`, `_assert_eval_only`.
6. `src/apc/primitives/primitive.py`, `CrossPositionLengthBiasPrimitive.
   __init__`/`forward` — the actual submodule graph: `content_in_proj`,
   `content_position_embedding`, `answer_query_embedding`,
   `position_bias_hidden`/`position_bias_out`, `cross_attn`
   (`nn.MultiheadAttention`, fused `in_proj_weight`/`in_proj_bias` +
   `out_proj`), `attn_norm`, `ffn` (`nn.Sequential` of two `Linear`s),
   `ffn_norm`, `readout`.
7. `src/apc/evaluation/mirror_contamination_free_checkpoint_trajectory_audit.py`
   (REC-004I) — `_source_for_step`/`_checkpoint_path` (step=6000 resolves to
   `B-C005REC-004D`'s own `run_001` tree; step=17500/18000 resolve to
   `B-C005REC-004H`'s), `_load_learning_curve_rows`.

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those trees, or in
  `rec004i/`/`rec004j/`.
- Read every checkpoint exactly as saved; `strict=True` state-dict load into
  a freshly constructed `CrossPositionLengthBiasPrimitive`, never a partial
  or coerced load. Hash-verify each against its own source task's recorded
  `checkpoint_state_hash` before trusting its forward pass.
- Reuse the frozen Core / 16-primitive bank reconstruction exactly as every
  prior REC task does (`ibc._reconstruct_parent_runtime`).
- Stage A's two datasets (`clean_v2_length10`, `length10_mechanism_probe_v1`)
  MUST be regenerated via REC-004J's own functions, unmodified — not
  redrawn, not resampled, not newly seeded. This task does not define a
  third dataset.
- Stage B's component groups are derived by reading the real forward code
  (`_oracle_attention`/`run_oracle_forward`) and confirmed by an empirical
  invariance check (Section 5) — never assumed from names alone.
- Stage C never searches component combinations. Exactly the fixed set
  `{R0, R1, R2, R3, R4, R5, R_ALL}` (Section 6) is run — no additional
  rollback variant is added after seeing intermediate results.
- Rollback constructs a fresh, freestanding `CrossPositionLengthBiasPrimitive`
  per variant, loaded from a merged state dict; it never mutates the
  originally loaded step=6000 or step=17500 primitive object in place, and
  never writes any `.pt` file.
- Attention argmax/rank/margin remain descriptive only if reported; no
  pass/fail reading in this task is gated on them (same rule as REC-004E/J).
- Every threshold used by a decision rule below is fixed in this document
  before any point is computed, and is never widened after seeing a result.

## 3. Stage A — matched-data temporal recheck (I03 only)

Run I03 @ {6000, 17500, 18000} — J0 and O1 — against BOTH of REC-004J's own
datasets (`clean_v2_length10`, n=230; `length10_mechanism_probe_v1`, n=512,
regenerated byte-identically). Reuses `compute_diagnostic_point` from
REC-004J unmodified; the step=17500 row is cross-checked against REC-004J's
own saved `diagnostic_points_summary.json` (must match within `1e-9`, or
Stage A itself reports `SOURCE_REPLAY_MISMATCH` and the task stops before
Stage B).

**Pre-registered gate (fixed here, before any point is computed):**

- If I03@6000's oracle sequence EM `>= 0.95` on **both** datasets, AND
  I03@17500's oracle sequence EM `< 0.95` on **at least one** dataset (already
  known from REC-004J to be true on both) ->
  `TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED`. Stage B/C/D proceed.
- Otherwise (I03@6000 itself fails to clear 0.95 on the new matched data) ->
  `TEMPORAL_SHIFT_NOT_ESTABLISHED`. Stage B/C/D do NOT run; the task ends
  here with this label as its final result, on the reasoning that the
  REC-004F-vs-REC-004J gap already includes an unresolved dataset effect and
  attributing any further finding to a temporal mechanism shift would not be
  licensed by evidence.

I03@18000 is included in Stage A for descriptive trajectory context (does
the loss, if any, continue past step=17500) but is not part of the gate.

## 4. Stage B — forward-graph parameter grouping (read from real code)

From `_oracle_attention`/`run_oracle_forward` (REC-004F), the O1 forward
graph after the attention distribution is substituted is:

```
oracle attn_probs (one-hot pi_n, not a learned parameter)
   -> V = in_proj_weight[V-slice] @ kv + in_proj_bias[V-slice]
        (kv = content_in_proj(content_features) + content_position_embedding(pos))
   -> attn_out = out_proj(attn_probs @ V)
   -> hidden = attn_norm(query + attn_out)
        (query = answer_query_embedding(query_ids); the residual add, no
        learned params of its own beyond the embedding that produced query)
   -> hidden = ffn_norm(hidden + ffn(hidden))
   -> logits = readout(hidden)
```

`position_bias_hidden`/`position_bias_out` (the learned position-bias MLP)
and the Q/K slices of `cross_attn.in_proj_weight`/`in_proj_bias` feed only
the attention SCORE (`S_other`, `b`) — which O1 discards entirely in favor
of the oracle one-hot map. They are read by J0 but **provably do not affect
O1's output**. Section 5 verifies this empirically (perturb-and-compare)
before it is relied on.

Five parameter groups, assigned by real `state_dict()` key (exhaustive,
non-overlapping — self-verified by `partition_state_dict_keys`):

| Component ID | `state_dict` keys |
|---|---|
| `VALUE_OUTPROJ` | `content_in_proj.*`, `content_position_embedding.*`, `cross_attn.in_proj_weight`, `cross_attn.in_proj_bias`, `cross_attn.out_proj.*` |
| `QUERY_RESIDUAL_PATH` | `answer_query_embedding.*` |
| `POST_ATTN_NORM` | `attn_norm.*` |
| `FFN_BLOCK` | `ffn.*`, `ffn_norm.*` |
| `READOUT` | `readout.*` |

`VALUE_OUTPROJ` deliberately rolls back the WHOLE fused `in_proj_weight`/
`in_proj_bias` tensor (not just the V-slice): since the Q/K slices are
provably inert under O1 (Section 5), rolling back the whole tensor and
rolling back only the V-slice produce IDENTICAL O1 output — the whole-tensor
form is used because `nn.MultiheadAttention` does not expose Q/K/V as
separate parameters to slice-assign into a fresh module's `state_dict`.

Excluded from every rollback group (never touched by any `Rx`, including
`R_ALL`): `position_bias_hidden.*`, `position_bias_out.*`.

## 5. Empirical forward-graph invariance check

Before Stage C runs, on the real loaded I03@17500 primitive and a batch of
real examples: compute O1 baseline logits, perturb (add large random noise
to) `position_bias_hidden`/`position_bias_out` and the Q/K slices of
`cross_attn.in_proj_weight`/`in_proj_bias` in place, recompute O1 logits,
restore the original values exactly, recompute once more. Requires:
`max_abs_logit_diff` between baseline and perturbed O1 output is exactly
`0.0`, and the post-restore recomputation exactly matches the baseline. If
this fails, Stage B's grouping premise is falsified — the task stops and
reports `FORWARD_GRAPH_INVARIANCE_CHECK_FAILED` rather than trusting any
rollback result built on it.

## 6. Stage C — same-init component rollback (I03@17500 base, O1 only intervention target)

Base: `I03@17500`'s real, hash-verified `state_dict()`. Early: `I03@6000`'s
real, hash-verified `state_dict()`. Both loaded once each; every rollback
variant below is a fresh in-memory merge, never a new file.

| Variant | Definition |
|---|---|
| `R0` | I03@17500 unmodified (baseline) |
| `R1` | R0 with `VALUE_OUTPROJ` keys replaced by I03@6000's values |
| `R2` | R0 with `QUERY_RESIDUAL_PATH` keys replaced by I03@6000's values |
| `R3` | R0 with `POST_ATTN_NORM` keys replaced by I03@6000's values |
| `R4` | R0 with `FFN_BLOCK` keys replaced by I03@6000's values |
| `R5` | R0 with `READOUT` keys replaced by I03@6000's values |
| `R_ALL` | R0 with ALL FIVE groups replaced by I03@6000's values (positive control) |

No combination search beyond this fixed set. Each variant is run under BOTH
J0 and O1, on BOTH datasets (`clean_v2_length10`, `length10_mechanism_probe_v1`).

**Positive-control parity gate (fixed here):** since `R_ALL` replaces every
O1-relevant parameter with I03@6000's own values, and Section 5 proves the
excluded parameters (which stay at step=17500) do not affect O1's output,
`R_ALL`'s O1 forward on a given dataset MUST reproduce I03@6000's own real
O1 forward on that same dataset to within `5e-3` max-abs-logit tolerance AND
exact discrete-prediction agreement (same tolerance convention as
REC-004E's `REC004E_SCORE_LOGIT_ABS_TOL`/`_MANUAL_RECONSTRUCTION_ABS_TOL`).
**If this fails on either dataset, the component decomposition is
incomplete or incorrect: the task stops and reports
`COMPONENT_DECOMPOSITION_PARITY_FAILED_STOP`, and R1-R5/`R_ALL`'s results
are recorded but NOT interpreted as a causal finding.**

## 7. Stage D — decision rule, centered on output position 4

For every `(rollback variant, dataset)` pair, record: sequence EM (J0, O1),
token accuracy, per-output-position (0-9) accuracy (J0, O1), per-output-
position mean target-token logit margin (J0, O1) — the margin between the
readout logit for the correct token and the best competing logit, the
quantity that changes right before a discrete prediction flips — and the
paired delta `O1 - J0` (to separate a rollback's downstream effect from any
residual attention-distribution effect it might incidentally also change:
`VALUE_OUTPROJ`'s rollback includes the Q/K slices, so it CAN shift J0's own
attention pattern even though it cannot shift O1's). Position 4 (source
index 0 under `pi_10`) is the primary residual per REC-004J/ADR-0105;
position 5 (source index 9) is the symmetric control on the other side of
MIRROR_HALVES's length-10 pivot; positions 0-3 and 6-9 are reported to
confirm a rollback does not silently break already-correct positions.

**Pre-registered per-component sufficiency rule (fixed here):** for a given
`Rx in {R1, R2, R3, R4, R5}`, on **both** datasets:

```
oracle_sequence_exact_match >= 0.95   AND   position_4_oracle_accuracy >= 0.95
```

- Exactly one `Rx` satisfies this -> label `<COMPONENT_ID>_ROLLBACK_SUFFICIENT`
  (e.g. `READOUT_ROLLBACK_SUFFICIENT`) — that component is a strong single
  repair-lead candidate (not authorized here).
- More than one `Rx` satisfies this -> label
  `MULTIPLE_COMPONENTS_ROLLBACK_SUFFICIENT`, all qualifying components listed
  (this task does not adjudicate which one is "truly" causal among
  co-sufficient candidates — that requires a follow-up ablation this task
  does not run).
- No single `Rx` satisfies this, but `R_ALL` does (which, given the parity
  gate in Section 6, is expected whenever Stage A confirms
  `TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED`) -> label
  `DISTRIBUTED_DOWNSTREAM_COADAPTATION` (no single component explains the
  residual; several components jointly changed between step=6000 and
  step=17500 in a way that matters only in combination).
- `R_ALL` itself does not satisfy the rule (only possible if Section 6's
  parity gate already failed) -> label `DOWNSTREAM_DECOMPOSITION_INCOMPLETE`;
  no repair direction is proposed.

This rule applies ONLY if Stage A returned
`TEMPORAL_ORACLE_SUFFICIENCY_LOSS_CONFIRMED` (Section 3).

## 8. Fixed non-adoption fields

Regardless of outcome: `new_optimizer_updates: 0`, `selected_init: null`,
`selected_step: null`, `selected_component: null`, `child_bundle: null`,
`rg3_recheck: "NOT_EXECUTED"`, `rec005_eligible: false`. This task cannot
select, adopt, retrain, or start any further repair. No I03 weight is ever
actually rolled back to step=6000 as a saved candidate — every rollback in
this task is an in-memory, evaluation-only causal probe.

## 9. Deliverables

- `src/apc/evaluation/mirror_temporal_mechanism_rollback_audit.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004k.yaml`
- `tests/test_mirror_temporal_mechanism_rollback_audit.py` (CPU-only,
  synthetic fixtures in the established pattern)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004k/run_001/`:
  `stage_a_temporal_points.jsonl`, `stage_a_temporal_decision.json`,
  `component_key_groups.json`, `forward_graph_invariance_check.json`,
  `stage_c_rollback_points.jsonl`, `component_decomposition_parity_check.json`,
  `i03_component_decision.json`, `freeze_audit.json`, `side_effect_audit.json`,
  `cost_accounting.json`, `summary.json`, `report.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` plus
  its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history update.

## 10. Completion report format

```
Task: B-C005REC-004K
Stage A: I03@{6000,17500,18000} oracle/J0 EM on both datasets; temporal decision label
Stage B: component groups (real state_dict keys), forward-graph invariance check result
Stage C: R0-R5/R_ALL results (EM, position-4/5 accuracy, logit margin), both datasets
Component decomposition parity check (R_ALL vs I03@6000 real O1): status
I03 component decision label, with numbers
new_optimizer_updates=0 / selected_init=null / selected_step=null / selected_component=null / child_bundle=null / rg3_recheck=NOT_EXECUTED / rec005_eligible=false
freeze/side-effect audit result
real tests / commands / hashes
```
