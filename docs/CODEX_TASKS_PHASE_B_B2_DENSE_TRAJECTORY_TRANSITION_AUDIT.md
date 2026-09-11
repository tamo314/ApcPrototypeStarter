# I03 Dense Trajectory Transition Replay & Oracle-Compatibility Onset Audit

Task ID: `B-C005REC-004T`. Positioned after REC-004S and before REC-005.
This is a trajectory-level diagnostic only. It performs zero new candidate training updates
and does not authorize a repair, candidate selection, child bundle, RG3, REC-005, or sealed evaluation.

## Purpose

Prior diagnostics (REC-004Q, REC-004R, REC-004S) refuted isolated instantaneous hypotheses:
local gradient misdirection, AdamW state sign reversal, score gradient weakness, cross-position
or cross-length interference, single-step overshoot, and strong parameter-to-score non-linearity.
However, in the historical JOINT trajectory of I03, downstream oracle compatibility ($O_1 \ge 0.95$)
was present at step 6000 ($O_1 \text{ EM} = 1.000$) and severely degraded by step 17500 ($O_1 \sim 0.75$).

Task B-C005REC-004T performs:
1. **Stage A (Coarse Localization):** Scan 25 historical 500-step checkpoints (steps 6000 to 18000)
   with $O_1$ oracle attention to find the last checkpoint where $O_1 \ge 0.95$ and the first where $O_1 < 0.95$.
   Mechanically locks a window $\le 1000$ updates.
2. **Stage B (Bit-Exact Dense Replay):** Starting from the last passing checkpoint, replay the exact
   historical training trajectory step by step up to the first failing checkpoint using the saved
   optimizer, scheduler, and RNG state. Verifies bit-exact parity at the endpoint against the saved checkpoint.
   Probes dynamics every 25 updates with full probe and every update with sentinel probe.
3. **Stage C (Temporal Ordering Analysis & I04 Control):** Identify the temporal ordering of:
   - $T_{\text{score\_peak}}$ (attention score margin peak)
   - $T_{\text{oracle\_loss}}$ (first step where $O_1 < 0.95$)
   - $T_{\text{o1\_margin\_peak}}$ (downstream logit margin peak under $O_1$)
   - $T_{\text{j0\_output\_peak}}$ (full model output logit margin peak)
   Evaluate whether score degradation precedes downstream compatibility loss or vice versa.
   Run identical window scan on I04 as a successful control.
4. **Stage D (Next Repair Contract Proposal):** Generate a single non-authorized next repair contract proposal.

## Checkpoints and Probes

Source Checkpoints:
- `I03 JOINT @6000..18000` (25 checkpoints at 500-step intervals)
- `I04 P @6000..18000` (successful control checkpoints)

Probe Datasets:
1. `score_function_dynamics_probe_v1` (1024 examples, byte-identical continuity set from REC-004S)
2. `dense_transition_probe_v1` (1024 examples, fresh confirmation set disjoint from training stream 1--18000)

## Parity Gate & Bounds

- Parity Gate: `replayed_canonical_state_hash == saved_canonical_state_hash`, `max_state_abs_diff == 0.0`, `max_opt_abs_diff < 1e-6`.
- Window size: $\le 1000$ optimizer updates.
- Invariants: `new_candidate_training_updates = 0`, `rg3_recheck = NOT_EXECUTED`, `rec005_eligible = false`.

Artifacts are written under `runs/phase_b_b2_model_bundle_recovery/rec004t/run_001/`.
