# AI Coding Task — Contamination-Free Checkpoint Trajectory Audit & Early-Stopping Feasibility

Formalizes the user's own chat instruction (2026-09-10, immediately after
`B-C005REC-004H`/ADR-0103) as `B-C005REC-004I`. This document is written by
the implementing agent from that instruction, not supplied by the user as a
file; it exists so the task has the same durable, re-readable record every
other `B-C005REC-00N` task has.

## 0. Purpose, authority, and stop boundary

`B-C005REC-004H` (ADR-0103) extended all 5 `P/I01-I05` `MIRROR_HALVES`
initializations to a common step=18000 against `rec004a_budget_validation`
(the "existing validation" set): only I01/I02/I04 individually cleared the
0.95 floor there, and that set has been project-documented since REC-004A as
already-adaptively-used development data, not a sealed independent holdout.
REC-004H's own `KNOWN_LIMITATIONS.md` additionally disclosed a small
(7/192000, none length-10) train/validation digest overlap for the
12001-18000 extension stream, bounded but not zero.

This task asks a narrower, prior question before trusting any of those
step-18000 numbers as a stopping point: **on saved checkpoints only, with a
newly-built validation set proven disjoint from every training example and
every reference/sealed set ever generated for this lineage, does the
trajectory across steps 6000-18000 tell the same story as the old validation
set did?** In particular: is I03's persistent length-10 weakness structural
(never reaches 0.95, at any saved checkpoint, on a clean set), or is it an
artifact of evaluating everyone at the same fixed step? And does I02's
borderline old-validation pass at step=18000 reproduce on a set the model
could not have touched during training even indirectly through selection?

**Zero new optimizer updates.** This task never trains, never adds a
checkpoint, and never re-enters `frozen_evaluation()`'s guarded surface for
anything but read-only runtime reconstruction (which itself must happen
before the frozen block, exactly as in REC-004E/F/H). `selected_init`,
`selected_step`, and `child_bundle` stay `null`; `rg3_recheck` stays
`"NOT_EXECUTED"`, unconditionally, regardless of how many inits clear 0.95
on the clean set. Candidate adoption, an independent RG3 query, and
`B-C005REC-005` remain separate, future, explicit-instruction-only work —
this task does not authorize or start any of them, even if the answer is
"all 5 pass."

## 1. Basis

Read first, in order:

1. `AGENTS.md` — the full REC-004A..004H history paragraph (ADR-0096..0103).
2. `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` — ADR-0096 (REC-004A,
   `rec004a_budget_validation`'s origin and its explicit
   already-adaptively-used status), ADR-0099/ADR-0102/ADR-0103 (REC-004D/G/H,
   the actual saved-checkpoint trajectory this task reads).
3. `src/apc/evaluation/incremental_budget_calibration.py` — canonical
   `_generate_step_training_examples`, `_derive_local_seed` reuse, and the
   `rec004a_budget_validation`/`rec004a_recheck_query` split definitions.
4. `src/apc/evaluation/mirror_position_bias_repair.py` (REC-004D),
   `mirror_budget_extension.py` (REC-004G), and
   `mirror_late_progress_conditional_extension.py` (REC-004H) — the exact,
   unbroken per-step training-data formula
   (`ibc._generate_step_training_examples(seed=10, step, "MIRROR_HALVES",
   vocab_size=10, sequence_length_range=(6,10))`) that produced every
   optimizer update from step 1 through step 18000 for **every** init (the
   docstring on REC-004D's own repair function records this as "same Core,
   same per-step training-data stream" — one shared deterministic stream,
   not five separate ones), and the real on-disk checkpoint layout:
   `runs/.../rec004d/run_001/{init}/P_LENGTH_POSITION_BIAS/checkpoints/step6000.pt`,
   `runs/.../rec004g/run_001/{init}/.../step{6500..12000}.pt`,
   `runs/.../rec004h/run_001/{init}/.../step{12500..18000}.pt` (confirmed
   present on disk for all 5 inits before writing this doc).
5. `src/apc/evaluation/mirror_position_initialization_diagnostic.py`
   (REC-004C) — the other MIRROR_HALVES-specific generated splits this task
   must also treat as reference/sealed:
   `rec004c_length_balanced_diagnostic`, `rec004c_position_identifiable_diagnostic`,
   `rec004c_content_counterfactual_diagnostic` (base + variant examples), and
   `rec004c_padding_batch_sample`.
6. `src/apc/evaluation/mirror_schedule_comparison.py` (REC-004B) —
   `rec004b_recheck_query`.

## 2. Invariants

- No training. No new checkpoint files under `rec004d/`, `rec004g/`, or
  `rec004h/`. No modification of any existing file in those three trees.
- Read every checkpoint exactly as saved; `strict=True` state-dict load into
  a freshly constructed `CrossPositionLengthBiasPrimitive` (arm
  `P_LENGTH_POSITION_BIAS`), never a partial or coerced load.
- Reuse the frozen Core / 16-primitive bank reconstruction exactly as REC-004A
  onward do (`ibc._reconstruct_parent_runtime`), never a fresh build.
- The new validation set's generation (including collision substitution) must
  be a pure function of its own seed and the declared protected-digest set —
  it must never depend on, or be adjusted after seeing, any checkpoint's
  prediction.

## 3. `clean_selection_validation_v2`

A new 1024-example `MIRROR_HALVES` set (same size as
`rec004a_budget_validation`, for a like-for-like by-length comparison),
namespace `clean_selection_validation_v2`, generated by the same
`_generate_parameter_free_examples`-style procedure (seed=10,
`vocab_size=10`, `sequence_length_range=(6,10)`) but checked, per candidate,
by `(input_tokens, target_tokens)` sha256 digest against the union of:

1. **The full training stream, steps 1-18000** — every example every init
   was ever updated on, regenerated (not assumed) via
   `ibc._generate_step_training_examples(seed=10, step, "MIRROR_HALVES", ...)`
   for `step` in `1..18000` inclusive (576000 examples; pure RNG, no model,
   no gradient — cheap to regenerate exactly).
2. **`rec004a_budget_validation`** (existing validation, 1024 examples).
3. **Every other reference/sealed MIRROR_HALVES split this lineage has ever
   generated**: `rec004a_recheck_query`, `rec004b_recheck_query`,
   `rec004d_recheck_query` (1024 each), and REC-004C's
   `rec004c_length_balanced_diagnostic` /
   `rec004c_position_identifiable_diagnostic` /
   `rec004c_content_counterfactual_diagnostic` (bases and variants) /
   `rec004c_padding_batch_sample`.

If a candidate's digest collides with this protected set, do not discard and
resample from a fresh RNG draw (that would be an ad hoc, unregistered
substitution). Instead keep drawing the **next** example from the *same*
continuing RNG stream used to build the set, in order, until a
non-colliding candidate is found, and use that one in the colliding slot.
This is fully deterministic given the protected set and touches no model
output. Log every substitution (original vs. replacement digest, slot index)
in `clean_selection_validation_v2_manifest.json`, written and hashed
**before** any checkpoint is loaded for a forward pass.

## 4. Checkpoint trajectory audit (forward-only)

For each of `I01`..`I05`, evaluate every saved checkpoint at step
`6000, 6500, ..., 18000` (25 points; step 6000 from REC-004D's tree,
6500-12000 from REC-004G's, 12500-18000 from REC-004H's) against
`clean_selection_validation_v2`. Per (init, step), record:

- overall clean-v2 sequence exact match,
- by-length (6/7/8/9/10) sequence exact match and counts,
- length-10 sequence-correct count specifically,
- mean valid-token cross-entropy, overall and per length,
- the source checkpoint's raw file sha256 (freeze/no-mutation evidence).

## 5. Trajectory decomposition (derived, no new forward passes)

Per init, from the 25-point clean-v2 trajectory:

- first step (if any) at which clean-v2 overall EM `>= 0.95`,
- the number of consecutive recorded steps (from that first crossing) for
  which EM stays `>= 0.95` (0 if it never crosses),
- the peak clean-v2 EM reached anywhere in the trajectory and the step it
  occurs at,
- the regression amount (`peak - final_at_18000`) and, if the peak is not at
  step 18000, whether EM ever returns to within `1e-6` of the peak after
  falling below it.

Compute the same four derived quantities a second time from each init's
**old** development metric (`existing_validation.correct_exact_match`, read
verbatim from REC-004D's/REC-004G's/REC-004H's own `learning_curve.jsonl`
rows at the matching step — never recomputed, never overwritten) so the two
trajectories can be compared side by side per init.

## 6. Old vs. clean-v2 agreement

For every (init, step) pair, report old-development EM and clean-v2 EM
together. Specifically confirm or refute, with the real numbers: does I02's
old-validation step=18000 pass (>=0.95) reproduce on clean-v2? Does I03 ever
reach 0.95 on clean-v2 at *any* of the 25 checkpoints, even ones later than
its old-validation floor check? Do not describe overall-clean-v2-EM
improvement as length-10 improvement — report the length-10-specific figure
separately, as REC-004H did.

## 7. Fixed non-adoption fields

Regardless of outcome: `selected_init: null`, `selected_step: null`,
`child_bundle: null`, `rg3_recheck: "NOT_EXECUTED"`. This task cannot select,
adopt, retrain, or extend a budget; it can only describe the trajectory that
already exists on disk.

## 8. Three-pattern interpretation (report only; do not act on it)

Classify the result into exactly one of:

- **`ALL_FIVE_CLEAR_CLEAN_V2`** — every init reaches `>=0.95` clean-v2 EM at
  some checkpoint in `6000..18000`. Suggested (not authorized) next step: a
  pre-registered early-stopping policy, applied identically to all five, then
  validated by independent fresh reinitializations under that policy — not a
  best-of-five pick from checkpoints already on disk.
- **`ONLY_I03_NEVER_CLEARS`** (or names whichever specific init(s), if not
  exactly I03, never reach 0.95 across the whole trajectory) — suggested next
  step: a mechanism-level fix scoped to that init's/length-10's failure, not
  a further budget extension for everyone.
- **`OLD_PASSES_LARGELY_INVALIDATED`** — clean-v2 removes most or all of the
  old-validation passes that existed anywhere in the trajectory. Suggested
  next step: redesign the development-evaluation/checkpoint-selection
  protocol itself before any `B-C005REC-005` candidate work.

State the classification with the literal numbers behind it. The "suggested
next step" is a proposal field only (`status: PROPOSED_NOT_AUTHORIZED`);
this task does not implement it.

## 9. Deliverables

- `src/apc/evaluation/mirror_contamination_free_checkpoint_trajectory_audit.py`
- `configs/phase_b_b2_model_bundle_recovery_rec004i.yaml`
- `tests/test_mirror_contamination_free_checkpoint_trajectory_audit.py`
  (CPU-only, synthetic fixtures in the established pattern — no dependency on
  the real multi-gigabyte `rec004d`/`rec004g`/`rec004h` trees)
- Real run under `runs/phase_b_b2_model_bundle_recovery/rec004i/run_001/`:
  `clean_selection_validation_v2_manifest.json`, `checkpoint_trajectory.jsonl`,
  `trajectory_decomposition.json`, `old_vs_clean_v2_comparison.json`,
  `pattern_classification.json`, `freeze_audit.json`, `side_effect_audit.json`,
  `cost_accounting.json`, `summary.json`, `report.md`.
- An ADR appended to `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` plus
  its index row in `docs/DECISIONS.md`, and an `AGENTS.md` history update.

## 10. Completion report format

```
Task: B-C005REC-004I
clean_selection_validation_v2: n, protected-set sizes, substitutions made, manifest hash
Checkpoint coverage: 5 inits x 25 steps, source dirs, hash-verified count
Init x step trajectory (clean-v2 EM, by-length, length-10 count, token loss): table
first>=0.95 step / duration / peak / regression, clean-v2 vs old-dev: table
Pattern classification + evidence
selected_init=null / selected_step=null / child_bundle=null / rg3_recheck=NOT_EXECUTED
freeze/side-effect audit result
real tests / commands / hashes
proposed (not executed) next step
```
