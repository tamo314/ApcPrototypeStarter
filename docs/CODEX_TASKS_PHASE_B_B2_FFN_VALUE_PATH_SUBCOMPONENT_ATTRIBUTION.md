# AI Coding Task — I03 FFN–Value-Path Subcomponent Attribution & Freeze-Repair Contract

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004L`/ADR-0108) as `B-C005REC-004M`. This document is written by
the implementing agent from that instruction, not supplied by the user as a
file; it exists so the task has the same durable, re-readable record every
other `B-C005REC-00N` task has.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004L` (ADR-0108) found `ffn_interaction_decision:
FFN_VALUE_OUTPROJ_INTERACTION_SUFFICIENT` — jointly rolling back `FFN_BLOCK`
+ `VALUE_OUTPROJ` (REC-004K's own group: `content_in_proj`,
`content_position_embedding`, the fused `cross_attn.in_proj_weight`/
`in_proj_bias`, and `cross_attn.out_proj`) from I03@17500 to I03@6000
perfectly recovers O1 (oracle-attention) sequence EM and position-4/5
accuracy on all three datasets, mechanistically explained by O1's
content-independent oracle attention depending only on `VALUE_OUTPROJ`'s
parameters. `VALUE_OUTPROJ` is a coarse group spanning four functionally
distinct roles (content preparation, the V projection, the attention output
projection, and — via the same fused tensor as V — the Q/K score
projection, which REC-004K's own note already flagged as a disclosed
confound when rolled back as a whole). This task asks exactly one question:
**does the FFN+VALUE_OUTPROJ interaction localize to ONE of these four
subcomponents, and if so, can a two-stage training repair be designed that
freezes only that subcomponent (plus FFN) after step=6000 while leaving
Q/K and position-bias free to keep learning the attention score?**

**Zero new optimizer updates.** Reads the SAME already-saved I03 checkpoints
REC-004K/L used (step=6000 from `B-C005REC-004D`'s tree, step=17500 from
`B-C005REC-004H`'s), plus — Stage E only, and only if Stage C localizes to
one subcomponent — I04@18000's and I05@17500's own step=6000/late
checkpoints (never another init's weights). Every condition is a fresh,
in-memory, evaluation-only `CrossPositionLengthBiasPrimitive` merged from
two real state dicts, or (new in this task) a merge with one fused tensor
split by **row-slice** (Q/K rows vs. V rows of `cross_attn.in_proj_weight`/
`in_proj_bias`) rather than swapped whole — never a new checkpoint file,
never a mutation of either loaded primitive in place. `selected_init`,
`selected_step`, `selected_value_subcomponent`, and `child_bundle` stay
`null`; `rg3_recheck` stays `"NOT_EXECUTED"`; `rec005_eligible` stays
`false`, unconditionally. This task never trains, never freezes anything in
a real training run, never feeds `pi_n` into a forward input, never extends
position bias, and never changes Core.

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004L history paragraph (ADR-0096..0108).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0106
   (REC-004K, the original 5-group decomposition and the disclosed
   fused-Q/K-slice confound noted for `VALUE_OUTPROJ`) and ADR-0108
   (REC-004L, `FFN_VALUE_OUTPROJ_INTERACTION_SUFFICIENT`, the anchor this
   task refines).
3. `src/apc/evaluation/mirror_temporal_mechanism_rollback_audit.py`
   (REC-004K) — `REC004K_COMPONENT_PREFIXES`, `partition_state_dict_keys`,
   `verify_o1_invariance_to_excluded_params`, `_build_rollback_primitive`,
   `build_stage_a_datasets`, `run_component_decomposition_parity_check`,
   `compute_rollback_diagnostic_point` (J0+O1 both — reused unmodified for
   this task's Stage E safety check): every one of these is reused
   UNMODIFIED.
4. `src/apc/evaluation/mirror_ffn_anchored_downstream_interaction_audit.py`
   (REC-004L) — `build_length10_downstream_interaction_probe_v1`,
   `compute_interaction_diagnostic_point`, `_run_o1_with_hidden`: reused
   UNMODIFIED for Stage C's per-condition metrics (O1 EM, position
   accuracy, hidden/logit diff vs `EARLY`) and for regenerating REC-004L's
   own new dataset byte-identically.
5. `src/apc/primitives/primitive.py` — `CrossPositionLengthBiasPrimitive`:
   confirms the real state-dict keys this task partitions
   (`content_in_proj.{weight,bias}`, `content_position_embedding.weight`,
   `cross_attn.in_proj_weight`, `cross_attn.in_proj_bias`,
   `cross_attn.out_proj.{weight,bias}`) and that `cross_attn` is a
   `torch.nn.MultiheadAttention` whose fused `in_proj_weight`/`in_proj_bias`
   stacks `[Q; K; V]` along dim 0 in three equal chunks — the same
   convention REC-004F's `_oracle_attention` and REC-004K's own invariance
   check already rely on (`mha.in_proj_weight.chunk(3, dim=0)`).

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those trees, or in
  `rec004i/`/`rec004j/`/`rec004k/`/`rec004l/`.
- Read every checkpoint exactly as saved; `strict=True` state-dict load,
  hash-verified against its own source task's recorded
  `checkpoint_state_hash` before trusting its forward pass.
- `clean_v2_length10`, `length10_mechanism_probe_v1`, and
  `length10_downstream_interaction_probe_v1` are regenerated via REC-004K's/
  REC-004L's own functions, unmodified — byte-identical to their existing
  saved sets. `length10_value_path_subcomponent_probe_v1` (Stage D, new) is
  a pure function of `(seed, protected_digests)` — never conditioned on any
  checkpoint prediction or rollback result — and its frozen digest is
  written to disk before any Stage B/C point is computed on it.
- Names in the user's instruction (`C`/`V`/`O`/`QK`) are conceptual; the
  actual state-dict keys backing each are determined from the real, loaded
  checkpoint's own `state_dict()` and self-verified to exactly account for
  every key REC-004K's `VALUE_OUTPROJ` group ever contained (no gap, no
  double-count) — never assumed from a static list alone.
- The fused `cross_attn.in_proj_weight`/`in_proj_bias` tensors are never
  rolled back whole for `V_PROJECTION` or `SCORE_PROJECTION_CONTROL`; only
  the relevant row-slice (`chunk(3, dim=0)` order Q, K, V) is replaced,
  building on top of whatever the tensor's current merged state already is
  (so requesting both slices for the same condition composes to the full
  tensor, rather than one call clobbering the other).
- Stage B never searches component combinations beyond the fixed
  `{R0, F, F_C, F_V, F_O, F_QK, F_ALLV, EARLY}` set. No additional condition
  is added after seeing intermediate results. `F_C`+`F_V`, `F_V`+`F_O`,
  `F_C`+`F_O`, or any 3-subcomponent combination is never constructed.
- Every threshold used by a decision rule is fixed in this document before
  any point is computed, and is never widened after seeing a result.

## 3. Stage A — split `VALUE_OUTPROJ` by real state and negative-control it

From I03@17500's and I03@6000's real, hash-verified `state_dict()`, further
partition REC-004K's own `VALUE_OUTPROJ` group into:

| Group | Real keys |
|---|---|
| `C` = `CONTENT_PREP` | `content_in_proj.weight`, `content_in_proj.bias`, `content_position_embedding.weight` |
| `V` = `V_PROJECTION` | the V row-slice (rows `2d:3d`) of `cross_attn.in_proj_weight`/`in_proj_bias` |
| `O` = `ATTN_OUT_PROJ` | `cross_attn.out_proj.weight`, `cross_attn.out_proj.bias` |
| `QK` = `SCORE_PROJECTION_CONTROL` | the Q/K row-slices (rows `0:2d`) of `cross_attn.in_proj_weight`/`in_proj_bias` — **negative control** |

Self-verified exhaustive/non-overlapping against the real key list (raises
if any `VALUE_OUTPROJ` key is unaccounted for). `QK` is provably inert under
O1 (the oracle one-hot substitution never reads Q/K — only `wv`/`bv` feed
`_oracle_attention`'s value computation): this is verified empirically, not
just theoretically, two ways — (a) REC-004K's own
`verify_o1_invariance_to_excluded_params`-style perturbation, reused
unmodified with the two Q/K chunks as the perturbed parameters, and (b) a
new rollback-based check comparing `F_QK`'s real O1 forward against `F`'s
real O1 forward on the actual I03@6000/@17500 weights — both must show
`max_abs_logit_diff == 0.0` before any `F_QK` result is trusted as a
negative control.

## 4. Stage B — 8 fixed conditions, FFN-anchored, one subcomponent at a time

Base = I03@17500 under O1 exclusively. Exactly these fixed conditions —
no combination beyond this set, no condition added after seeing a result:

```
R0     = I03@17500 as-is
F      = FFN_BLOCK rolled back to step=6000
F_C    = FFN_BLOCK + CONTENT_PREP rolled back to step=6000
F_V    = FFN_BLOCK + V_PROJECTION (V row-slice only) rolled back to step=6000
F_O    = FFN_BLOCK + ATTN_OUT_PROJ rolled back to step=6000
F_QK   = FFN_BLOCK + SCORE_PROJECTION_CONTROL (Q/K row-slices only) rolled back to step=6000  # negative control
F_ALLV = FFN_BLOCK + all four subcomponents (C+V+O+QK) rolled back to step=6000
EARLY  = I03@6000's own real, unmodified checkpoint
```

**Decomposition parity gate (new for this task).** `F_ALLV`'s row-slice
composition must reproduce, exactly, an independently-built reference that
rolls back REC-004K's whole (un-sliced) `VALUE_OUTPROJ` group jointly with
`FFN_BLOCK` via `rec004k._build_rollback_primitive` (the same call REC-004L
made for its own `F_V`) — `max_abs_logit_diff` within `5e-3` and identical
discrete predictions, on every dataset. This also means `F_ALLV` must
reproduce REC-004L's own saved `F_V` numbers on the three shared datasets —
checked as an informational cross-check. If either check fails, the task
stops and reports `VALUE_PATH_DECOMPOSITION_PARITY_FAILED_STOP` — no
`F_X`/`F_ALLV` result is interpreted as a causal finding.

## 5. Stage C — one new, fully disjoint diagnostic set

`length10_value_path_subcomponent_probe_v1`, 512 examples, generated by
`build_length10_value_path_subcomponent_probe_v1` — REC-004L's own
collision-substitution recipe, verbatim, with a new split label (hence a
new RNG stream) and a protected set that is the union of: the training
stream (steps 1-18000), `rec004a_budget_validation`, every REC-004A-D
reference/sealed split, `clean_selection_validation_v2`'s full 1024-example
set, `length10_mechanism_probe_v1`, AND `length10_downstream_interaction_
probe_v1` (REC-004L's own new set — not yet excluded by any earlier task's
protected registry). Its digest is frozen to disk BEFORE any Stage B/C
point is computed. Development-exposed diagnostic data, not sealed/RG3.

All four Stage D decision points below run on **all four** datasets:
`clean_v2_length10`, `length10_mechanism_probe_v1`,
`length10_downstream_interaction_probe_v1`, and
`length10_value_path_subcomponent_probe_v1`.

## 6. Stage D — decision, centered on output position 4

For each of `{F_C, F_V, F_O, F_QK}`, on **all four** datasets:
`oracle_sequence_exact_match >= 0.95 AND position_4_oracle_accuracy >= 0.95`.

- Exactly one satisfies this -> `FFN_<SUBCOMPONENT>_INTERACTION_SUFFICIENT`.
- More than one -> `MULTIPLE_VALUE_SUBPATHS_SUFFICIENT` (never resolved by
  picking the highest EM).
- None, but `F_ALLV` does -> `DISTRIBUTED_WITHIN_VALUE_PATH`. This task does
  NOT auto-escalate to a further subset search here, and does NOT propose
  freezing the entire value path as the next repair even in this case:
  `CONTENT_PREP` feeds `kv`, which both `V_PROJECTION` and
  `SCORE_PROJECTION_CONTROL` (the real, learnable K path) read from the SAME
  tensor — freezing `CONTENT_PREP` would also freeze part of what the
  attention SCORE (not just its oracle-substituted output) depends on,
  undermining the exact attention-score learning a repair is meant to
  preserve. This caveat is recorded in the report/contract regardless of
  which label is reached.
- Neither -> `VALUE_PATH_DECOMPOSITION_INCOMPLETE`.

## 7. Stage E — success-model safety check (conditional)

Runs ONLY if Stage D reaches exactly one `FFN_<SUBCOMPONENT>_INTERACTION_
SUFFICIENT` label. Applies the SAME localized rollback (`FFN_BLOCK` +
the one sufficient subcomponent, rolled back to step=6000) to:

```
I04 @18000   # currently-stable success case (REC-004J/004I)
I05 @17500   # success case just before its own step=18000 collapse
```

each using **that init's own** step=6000 state (never I03's, never any
cross-init weight transplant). Measures BOTH **J0 (normal forward — the
decisive metric, since a real repair must work without oracle substitution)
and O1**, via REC-004K's own `compute_rollback_diagnostic_point` reused
unmodified, on all four Stage C datasets, comparing the rolled-back
condition against that init's own real, unmodified late-step baseline. No
candidate is adopted here; this only checks for signs that reverting the
localized subcomponent (plus FFN) to step=6000 would break an
already-succeeding trajectory. A J0 sequence-EM drop `>= 0.05` on any
dataset (pre-registered here, descriptive, never gates Stage D's own label)
is recorded as `DEGRADATION_OBSERVED` for that init; otherwise
`NO_DEGRADATION_OBSERVED`.

## 8. Deliverable — `next_training_repair_contract.md`

Written regardless of outcome. Only if Stage D localizes to exactly one
subcomponent does it propose (status `PROPOSED_NOT_AUTHORIZED`, never
implemented by this task) a two-phase training design:

```
Phase 1: step 0 -> 6000, current REC-004D/G recipe unchanged.
Phase 2: step 6001 -> a fixed, finite target step
  FFN_BLOCK + the one localized value subcomponent held at their
  step=6000 values; Q/K, position bias, and everything else in
  QUERY_RESIDUAL_PATH/POST_ATTN_NORM/READOUT continue training normally.
```

step=6000 is treated as a **mechanistic transition boundary** (independent
diagnostic evidence: I03@6000 was oracle-sufficient — REC-004K Stage A —
and oracle sufficiency was lost by step=17500), never as a result-selected
"best checkpoint." If Stage D does not localize to one subcomponent, or
Stage E shows degradation for I04/I05, the contract records that as
evidence against proposing this design and does not propose an
alternative, per the task's own diagnosis-only charter. Actually running
any such training is a separate task requiring its own explicit user
instruction and its own fixed, pre-registered protocol.

## 9. Forbidden in this task

- Any new optimizer update.
- `F_C+F_V`, `F_V+F_O`, `F_C+F_O`, or any combination beyond the fixed
  Stage B set.
- FFN retraining.
- An actual VALUE-path freeze training run.
- Using oracle attention (`pi_n`) in any loss or training forward.
- Feeding `pi_n` into a model input.
- Extending position bias.
- Any Core change.
- Adopting a best init or best checkpoint.
- Any child bundle.
- Any RG3 recheck.
- Starting `B-C005REC-005`.

## 10. Fixed non-adoption fields

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

## 11. Deliverables

- `src/apc/evaluation/mirror_ffn_value_path_subcomponent_attribution.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004m.yaml`
- `tests/test_mirror_ffn_value_path_subcomponent_attribution.py` (CPU-only,
  synthetic fixtures in the established pattern)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004m/run_001/`:
  dataset manifests (all four), `checkpoint_load_info.json`,
  `value_path_subcomponent_key_groups.json`,
  `forward_graph_invariance_check.json`, `qk_rollback_invariance_check.json`,
  `condition_points.jsonl`, `decomposition_parity_check.json`,
  `cross_check_against_rec004l.json`,
  `value_path_subcomponent_decision.json`, `regression_check.json`,
  `safety_check.json` (or a `NOT_EXECUTED` stub), `freeze_audit.json`,
  `side_effect_audit.json`, `cost_accounting.json`, `summary.json`,
  `report.md`, `next_training_repair_contract.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
  plus its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history
  update.

## 12. Completion report format

```
Task: B-C005REC-004M
Stage A: real key groups for C/V/O/QK, QK invariance check (theoretical +
  rollback-based, both 0.0 diff)
Stage B/D: R0/F/F_C/F_V/F_O/F_QK/F_ALLV/EARLY results (EM, position-4/5
  accuracy, hidden/logit diff vs EARLY), all four datasets
Decomposition parity check (F_ALLV vs FFN+whole-VALUE_OUTPROJ, and vs
  REC-004L's own saved F_V): status
Value-path subcomponent decision label, with numbers
Stage E safety check: executed or not, per-init J0/O1 deltas if executed
next_training_repair_contract.md: proposed or not, and why
new_optimizer_updates=0 / selected_init=null / selected_step=null /
  selected_value_subcomponent=null / child_bundle=null /
  rg3_recheck=NOT_EXECUTED / rec005_eligible=false
freeze/side-effect audit result
real tests / commands / hashes
```
