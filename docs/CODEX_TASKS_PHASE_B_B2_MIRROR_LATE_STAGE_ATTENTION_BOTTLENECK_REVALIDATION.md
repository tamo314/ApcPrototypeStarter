# AI Coding Task — Late-Stage Attention Bottleneck Revalidation (I03) & Collapse Contrast (I05)

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004I`/ADR-0104) as `B-C005REC-004J`. This document is written by
the implementing agent from that instruction, not supplied by the user as a
file; it exists so the task has the same durable, re-readable record every
other `B-C005REC-00N` task has.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004F` (ADR-0101) found that at **step=6000**, substituting the
oracle `pi_n` one-hot attention distribution for I03's own combined attention
recovered length-10 EM from 0.086 to 1.000 — strong evidence that, at that
early checkpoint, the attention distribution itself was I03's bottleneck.
`B-C005REC-004I` (ADR-0104) later found I03 **never** clears the 0.95
length-10 floor on a fully disjoint set at *any* checkpoint from step=6000
through step=18000, with its own best point at **step=17500** (clean-v2
length-10: 44/230). REC-004F's oracle-attention finding was never re-checked
at this much later step. Extrapolating a step=6000 mechanism finding onto a
step=17500 checkpoint 11500 updates later is not licensed by anything already
measured — the value/residual/readout weights I03's attention feeds into have
also changed substantially over that window (I03's own overall clean-v2 EM
rose from ~0.01 at step 6000 to 0.82 at step 17500), so the earlier
attention-is-the-bottleneck reading may or may not still hold.

This task asks exactly one question: **at step=17500 (I03's own clean-v2
peak), is I03's length-10 failure still explained by the attention
distribution, or has the mechanism shifted downstream since step=6000?**
`B-C005REC-004I`'s own `next_step_recommendation.md` proposed a
mechanism-level fix for I03/length-10 as a *future* step (`status:
PROPOSED_NOT_AUTHORIZED`); this task is a **diagnostic prerequisite** to that
future repair, not the repair itself, and does not implement or authorize any
fix.

**Zero new optimizer updates.** This task never trains, never adds a
checkpoint under `rec004d/`, `rec004g/`, or `rec004h/`, and never re-enters
`frozen_evaluation()`'s guarded surface for anything but read-only runtime
reconstruction (which itself happens before the frozen block, exactly as in
REC-004E/F/H/I). `selected_init`, `selected_step`, and `child_bundle` stay
`null`; `rg3_recheck` stays `"NOT_EXECUTED"`, unconditionally, regardless of
which of the two labels below the result matches. This task does not select,
adopt, retrain, or start `B-C005REC-005`, `B-C005R3-011`, `B-C006`, or Task
Inference, even if the result is `PERSISTENT_ATTENTION_DISTRIBUTION_
BOTTLENECK_SUPPORTED`.

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004I history paragraph (ADR-0096..0104).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0101
   (REC-004F, the oracle-attention substitution mechanism and its
   step=6000 `MIXED_ACROSS_INITS` result) and ADR-0104 (REC-004I, I03's
   never-clears-clean-v2 trajectory finding, I05's final-checkpoint
   collapse finding, and the real on-disk checkpoint layout for
   step=12500..18000 under `rec004h/run_001/`).
3. `src/apc/evaluation/mirror_oracle_attention_substitution_probe.py`
   (REC-004F) — `_oracle_attention`/`run_oracle_forward` (the ONE
   substitution this task reuses unmodified: value projection, `out_proj`,
   `attn_norm`, `ffn`, `readout` are the real checkpoint's weights; only the
   attention distribution is replaced with a one-hot `pi_n` map).
4. `src/apc/evaluation/mirror_position_score_residual_audit.py` (REC-004E) —
   `_prepare_query_kv`, `_pad_mask`, `_position_bias_raw`, `_manual_attention`
   (gives real per-head attention probabilities, `S_other`, and the bias term
   `b` in one reconstruction, verified to reproduce the real forward within
   5e-3 logit tolerance and exact discrete-prediction agreement), and
   `_post_attention`.
5. `src/apc/evaluation/mirror_contamination_free_checkpoint_trajectory_audit.py`
   (REC-004I) — `build_protected_digest_registry`,
   `build_clean_selection_validation_v2` (the collision-substitution
   procedure this task extends), `_source_for_step`/`_checkpoint_path` (the
   real on-disk checkpoint layout — for every step this task reads,
   17500 and 18000, `_source_for_step` resolves to `B-C005REC-004H`'s own
   `run_001` tree), and `_old_dev_metric_row` (source-replay ground truth
   from REC-004H's own `learning_curve.jsonl`).
6. `src/apc/evaluation/mirror_late_progress_conditional_extension.py`
   (REC-004H) — confirms the real checkpoint files
   `runs/.../rec004h/run_001/{I03,I04,I05}/P_LENGTH_POSITION_BIAS/checkpoints/
   step{17500,18000}.pt` exist on disk before this doc was written.

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those three trees, or
  in `rec004i/`.
- Read every checkpoint exactly as saved; `strict=True` state-dict load into
  a freshly constructed `CrossPositionLengthBiasPrimitive` (arm
  `P_LENGTH_POSITION_BIAS`), never a partial or coerced load. Hash-verify
  each loaded checkpoint against REC-004H's own recorded
  `checkpoint_state_hash` before trusting its forward pass (`SOURCE_REPLAY_
  MISMATCH` on any unexplained difference, exactly as REC-004E/F/I do).
- Reuse the frozen Core / 16-primitive bank reconstruction exactly as
  REC-004A onward do (`ibc._reconstruct_parent_runtime`), never a fresh
  build.
- The oracle substitution is confined to exactly `_oracle_attention`/
  `run_oracle_forward` (REC-004F's own functions, imported not re-derived);
  `mirror_halves_position_map` is never fed into any OTHER forward input in
  this module.
- `length10_mechanism_probe_v1`'s generation (including collision
  substitution) must be a pure function of its own seed and the declared
  protected-digest set — it must never depend on, or be adjusted after
  seeing, any checkpoint's prediction.
- Attention argmax/rank/margin are DESCRIPTIVE metrics only. Neither this
  task's `PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED` /
  `LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM` label, nor any pass/fail
  reading, may be gated on them. The label is gated ONLY on oracle sequence
  EM and the oracle-minus-J0 paired delta (Section 6).

## 3. Target checkpoints

Fixed, minimum set (all real, on-disk, `P_LENGTH_POSITION_BIAS` arm,
`B-C005REC-004H`'s `run_001` tree per `_source_for_step`):

| Checkpoint | Role |
|---|---|
| I03 @ 17500 | I03's own clean-v2 peak (ADR-0104); primary diagnostic target |
| I03 @ 18000 | terminal comparison |
| I04 @ 18000 | same-architecture success control |
| I05 @ 17500 | pre-collapse success control |
| I05 @ 18000 | post-collapse control |

I01/I02 stable-pass checkpoints are NOT added — the task's own charter
(Section 0) says they are not required for the primary diagnosis, and adding
them would broaden scope without changing the I03 gate in Section 6.

## 4. Data — two sets, paired

### 4a. `clean_selection_validation_v2`, length-10 subset

Regenerate REC-004I's own `clean_selection_validation_v2` EXACTLY (same
seed, same `build_protected_digest_registry` inputs, same
`build_clean_selection_validation_v2` collision-substitution procedure — a
pure function, so this reproduces the identical 1024 examples, byte for
byte) and filter to the 230 examples with `len(input_tokens) == 10`. This is
the SAME length-10 subset REC-004I already evaluated all 5 inits against at
every checkpoint from 6000-18000 (its own by-length row records I03's
step=17500 result as 44/230) — reusing it here gives an exactly paired
comparison against REC-004I's own trajectory numbers, not a new sample.

### 4b. `length10_mechanism_probe_v1` (new, diagnostic-only, 512 examples)

A NEW set, generated the same way as `clean_selection_validation_v2`
(`ibc._generate_step_training_examples`-compatible RNG stream,
`MIRROR_HALVES`, `vocab_size=10`) but with sequence length fixed at exactly
10 for every example (not drawn from the 6-10 range), checked by
`(input_tokens, target_tokens)` sha256 digest against the union of:

1. Every digest `clean_selection_validation_v2` was already protected
   against (`build_protected_digest_registry`'s own set: the full training
   stream steps 1-18000, `rec004a_budget_validation`,
   `rec004a_recheck_query`/`rec004b_recheck_query`/`rec004d_recheck_query`,
   and all four REC-004C reference/diagnostic splits).
2. **`clean_selection_validation_v2` itself** (all 1024 examples, not just
   the length-10 subset) — since it is now itself a reference split this
   lineage has generated, and this new set must not silently duplicate the
   material Section 4a already covers.

Same collision-substitution rule as REC-004I (Section 3 of
`CODEX_TASKS_PHASE_B_B2_MIRROR_CONTAMINATION_FREE_CHECKPOINT_TRAJECTORY_AUDIT.md`):
on a collision, keep drawing the NEXT candidate from the same continuing RNG
stream — never a fresh reseed, never conditioned on any checkpoint's
prediction. Log every substitution in
`length10_mechanism_probe_v1_manifest.json`, written and hashed **before**
any checkpoint is loaded for a forward pass.

**Disclosure, not a sealed set.** `length10_mechanism_probe_v1` is a
development-exposed diagnostic set, not a sealed or final RG3 query set. Once
consumed by this task it must be labeled `development_exposed: true` in its
own manifest and in every future task that might reuse it; it is not eligible
to serve as an independent RG3 recheck query without a fresh, disjoint
regeneration.

## 5. Intervention — exactly two conditions

- **J0**: normal forward. Implemented via the real per-head reconstruction
  (`_prepare_query_kv` + `_position_bias_raw` + `_manual_attention` with
  `s_other_scale=1.0` + `_post_attention`) — REC-004E's own manual
  reconstruction path, already verified to reproduce the real
  `primitive.forward()` within 5e-3 logit tolerance and exact discrete-
  prediction agreement. Using this path (rather than the hooked-MHA path) in
  one pass yields per-head attention probabilities, `S_other`, and the bias
  term `b` together, without a second forward.
- **O1**: REC-004F's oracle `pi_n` one-hot attention substitution
  (`_oracle_attention`/`run_oracle_forward`, imported unmodified). Value
  projection, `out_proj`, `attn_norm`, `ffn`, and `readout` are the SAME real
  checkpoint weights `J0` uses; only the attention distribution is replaced.

No other J-condition (J1-J6) is run. This is a deliberate narrowing from
REC-004E's full matrix — the question here is exactly "does substituting the
oracle attention recover this specific late-stage checkpoint," nothing else.

## 6. Metrics and decision rule

For every (checkpoint, dataset) pair — 5 checkpoints x 2 datasets = 10
points — record:

- sequence EM (J0, O1), token accuracy (J0, O1),
- paired delta `O1_EM - J0_EM`, `J0`-only-correct / `O1`-only-correct /
  both-correct / both-wrong counts,
- per-output-position (0..9) accuracy under J0 and under O1,
- correct-key rank and margin in the head-averaged J0 attention distribution
  against `pi_n` (descriptive only — see Section 2's invariant),
- J0 attention entropy (head-averaged and per-head, mean/std over the
  dataset and per output position); O1's entropy is reported too, purely as
  a construction sanity check (it is exactly 0 by definition of a one-hot
  substitution, not a new finding),
- inter-head agreement (fraction of (example, output position) where all
  `n_head` heads' argmax key agree) and inter-head divergence (mean L1
  distance from each head's distribution to the heads' mean distribution),
- `S_other` and bias-term `b` row-centered std/range (same formula as
  REC-004E's `_row_centered_stats`), and their row-wise correlation,
- the count and per-output-position breakdown of examples still wrong under
  O1 (the residual that oracle attention substitution does not explain).

**Pre-registered decision rule (fixed before any point above is computed):**

For **I03 @ 17500 only**, on **both** datasets (4a and 4b):

- If `O1_EM >= 0.95` **and** `O1_EM - J0_EM >= 0.30` on both datasets ->
  `PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED`. (0.30 is set
  well above REC-004F's own 0.05 recovery-delta threshold, because I03's own
  J0 length-10 baseline at step=17500 is already known to be low (ADR-0104:
  44/230 =~ 0.19); reaching `>=0.95` from that baseline necessarily implies a
  much larger delta than 0.30, so this bar rules out a technically-passing
  EM produced by a barely-there improvement, without being reachable only by
  coincidence.)
- Otherwise -> `LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM`.

I03 @ 18000, I04 @ 18000, and I05 @ 17500/18000 are NOT part of this gate —
they are reported with the identical metric set for context and for the I05
contrast (Section 7), but the label in this section is I03 @ 17500 only, per
Section 0's single stated purpose.

**Consequence field (descriptive, not authorization):** if
`PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED`, this task records
that a follow-up repair scoped to attention-score learning stabilization
would be the narrower next candidate; if `LATE_STAGE_BOTTLENECK_MIXED_OR_
DOWNSTREAM`, this task records that value/residual/readout-side diagnosis
should precede any attention-score change. Neither statement authorizes or
starts that follow-up work.

## 7. I05 collapse contrast

Using the SAME per-(checkpoint, dataset) points already computed for I05 @
17500 and I05 @ 18000 (Section 6), compare: does J0 EM collapse (matching
REC-004I's finding, ADR-0104 Evidence 5) accompanied by (a) O1 EM also
collapsing (attention substitution does NOT rescue the terminal checkpoint,
implying the attention distribution's own change is not the whole story —
consistent with REC-004F's step=6000 finding that I05's residual sits
downstream), or (b) O1 EM staying high while J0 falls further (attention
distribution itself degraded and is now more load-bearing at this
checkpoint than it was)? Also compare entropy/head-agreement/`S_other`/bias
statistics between the two steps to see whether the attention distribution's
own shape visibly shifts at step=18000 or stays stable while only the
readout/value path breaks. This is a **contrast, not this task's primary
purpose** (Section 0) — it exists so a future I03 fix can be checked for
whether it would also perturb I05, not to explain I05's collapse.

## 8. Fixed non-adoption fields

Regardless of outcome: `selected_init: null`, `selected_step: null`,
`child_bundle: null`, `rg3_recheck: "NOT_EXECUTED"`, `new_optimizer_updates:
0`. This task cannot select, adopt, retrain, or start any further repair.

## 9. Deliverables

- `src/apc/evaluation/mirror_late_stage_attention_bottleneck_revalidation.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004j.yaml`
- `tests/test_mirror_late_stage_attention_bottleneck_revalidation.py`
  (CPU-only, synthetic fixtures in the established pattern — no dependency on
  the real multi-gigabyte `rec004d`/`rec004g`/`rec004h`/`rec004i` trees)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004j/run_001/`:
  `length10_mechanism_probe_v1_manifest.json`, `clean_v2_source_replay.json`,
  `diagnostic_points.jsonl`, `diagnostic_points_summary.json`,
  `i03_17500_decision.json`, `i05_collapse_contrast.json`, `freeze_audit.json`,
  `side_effect_audit.json`, `cost_accounting.json`, `summary.json`,
  `report.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` plus
  its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history update.

## 10. Completion report format

```
Task: B-C005REC-004J
length10_mechanism_probe_v1: n=512, protected-set size (incl. clean-v2), substitutions, manifest hash
clean_selection_validation_v2 length-10 subset: reproduced n, source-replay status
5 checkpoints x 2 datasets x {J0,O1}: EM / token-acc / paired delta / per-position table
rank/margin, entropy, head-agreement, S_other/bias stats (descriptive)
I03@17500 decision: PERSISTENT_ATTENTION_DISTRIBUTION_BOTTLENECK_SUPPORTED | LATE_STAGE_BOTTLENECK_MIXED_OR_DOWNSTREAM, with numbers
I05@17500 -> I05@18000 collapse contrast
selected_init=null / selected_step=null / child_bundle=null / rg3_recheck=NOT_EXECUTED / new_optimizer_updates=0
freeze/side-effect audit result
real tests / commands / hashes
```
