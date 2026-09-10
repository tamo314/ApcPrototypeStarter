# AI Coding Task — I03 Oracle-Compatible Downstream Freeze / Score-Only Continuation Pilot

Task ID: `B-C005REC-004O`.

This task is the explicitly authorized, single-init mechanism-repair pilot
following the diagnostic REC-004K–N sequence.  It does not select a candidate,
build a bundle, re-run RG3, start REC-005, or access a sealed/RG3 query set.
The earlier REC-004N finding that a `CONTENT_PREP`-preserving *rollback* was
not sufficient remains historical evidence; this task tests the deliberately
different hypothesis that fixed content representations can still support
learning in the attention score path.

## Question and source

Starting from the complete I03 `P_LENGTH_POSITION_BIAS` training state at
step 6000, can normal attention (`J0`) reach length-10 performance when only
Q/K rows and the learned position-bias MLP continue to learn, while every O1
downstream tensor is protected?

The source is REC-004D I03 step 6000, including primitive, AdamW,
CosineAnnealingLR, CPU RNG, and CUDA RNG state.  Training examples are exactly
the original deterministic steps 6001–12000; all recipe fields other than the
update-permitted parameter set are inherited from REC-004D/G.

## Stage A — real forward-graph partition

Before training, derive the parameter inventory from the loaded primitive's
actual `named_parameters`/`state_dict` and run a real forward/backward trace.
The trace must be exhaustive and non-overlapping.  It must identify:

- frozen O1-compatible downstream: `CONTENT_PREP`, V rows of fused QKV,
  attention output projection, query residual, post-attention norms, FFN,
  readout;
- score-only trainable path: Q/K rows of fused QKV and
  `position_bias_hidden` / `position_bias_out`.

Any unexpected, missing, duplicate, non-forward, or unclassifiable parameter
is `FUSED_QKV_SELECTIVE_FREEZE_UNSAFE` and stops before an optimizer update.
`CONTENT_PREP` is deliberately frozen; it is not a convenience choice.

## Stage B — fail-closed fused QKV handling

The fused `cross_attn.in_proj_weight` and (when present) bias are not split
into modules.  The restored historical AdamW optimizer remains structurally
intact to preserve its recipe/state, but frozen whole tensors have no gradient
and cannot step.  For fused tensors, before every optimizer step the V rows of
the parameter and every row-shaped AdamW state tensor are snapshotted; V
gradients are zeroed, then V parameter/state rows are restored after the step.
The only shared scalar counter is AdamW's tensor-level `step`, which advances
because Q/K rows in that same parameter legitimately update; it is recorded as
shared metadata rather than a V-row mutation.

The run must prove after every update and at the terminal state that V weights
and V row-state are bit-identical to step 6000, and that every other frozen
tensor and its optimizer state are unchanged.  A failure stops with
`FUSED_QKV_SELECTIVE_FREEZE_UNSAFE`.

## Stage C — paired historical joint replay

Before score-only training, restore an independent copy of I03@6000 and replay
the unchanged joint recipe for 500 updates.  The resulting complete state must
match REC-004G I03@6500 (primitive, optimizer row/state, scheduler, RNG, and
metadata) exactly.  Otherwise stop with `JOINT_CONTROL_REPLAY_MISMATCH`.

REC-004G I03@12000 is the historical joint endpoint control.  It is evaluated
on this task's new data; it is not retrained for 6000 updates.

## Pre-output data lock

Before any model output is computed, create and write manifests for:

- `score_only_repair_validation_v1`: 1024 normal `MIRROR_HALVES` examples;
- `score_only_length10_confirmation_v1`: 512 examples, length 10.

Each digest must be disjoint from training steps 1–12000 and all prior
validation/query/reference/diagnostic/sealed sets known to the REC-004 lineage;
the two new sets are also mutually disjoint.  The datasets are development
exposed, not sealed and not RG3 query data.

## Training and endpoint decision

Run exactly 6000 score-only updates (terminal cumulative step 12000).  Optional
records may be made every 500 updates, but the decision uses only step 12000;
no best-intermediate selection is allowed.

Primary J0 acceptance:

- validation overall sequence EM >= 0.95 and validation length-10 EM >= 0.95;
- length-10 confirmation sequence EM >= 0.95;
- every frozen downstream tensor is exactly unchanged;
- O1 compatibility is >= 0.95 on both new datasets.

Decision labels are fixed: all acceptance conditions yields
`SCORE_ONLY_CONTINUATION_SUPPORTED_ON_I03_PILOT`; a J0 improvement over the
historical joint endpoint but a missed floor yields
`SCORE_ONLY_IMPROVEMENT_INSUFFICIENT`; no J0 improvement yields
`SCORE_ONLY_CONTINUATION_NOT_SUPPORTED`.  A completed run with score gradients
and score updates but no J0 improvement may additionally carry
`FROZEN_CONTENT_PREP_LIMITS_SCORE_LEARNING` as a hypothesis tag, never as a
proof of cause.  Selective-freeze and control-replay stop labels take priority.
