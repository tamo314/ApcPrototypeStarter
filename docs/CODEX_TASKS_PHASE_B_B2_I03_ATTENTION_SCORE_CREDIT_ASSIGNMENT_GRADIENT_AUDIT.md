# I03 Attention-Score Credit-Assignment Gradient Audit & Optimizer-State Gate

Task ID: `B-C005REC-004Q`. This is a diagnostic-only task after REC-004P and
before REC-005. It performs zero optimizer updates and authorizes no repair,
candidate selection, child bundle, RG3 query, sealed evaluation, or REC-005.

## Scope

First audit REC-004P's O1 metric consistency. A sequence EM of 1.0 requires
every valid output position to be correct. Preserve old artifacts. If its O1
position values were selected from J0 fields but terminal checkpoints allow a
correct read-only re-evaluation, record `REC004P_EVALUATION_METRIC_BUG`, correct
only the evaluator/reaggregation, and proceed only if the main J0 label
`CONTENT_PREP_RELEASE_NOT_SUPPORTED` remains valid. A changed main J0 result is
`REC004P_RESULT_REQUIRES_REINTERPRETATION` and stops this task.

Lock `score_credit_assignment_probe_v1` before loading a model or observing an
output. It has 128 examples at each length 6--9 and 512 at length 10. Its
`(input_tokens, target_tokens)` digests must be disjoint from steps 1--18000,
the old validation/query/reference/diagnostic/sealed registries, the
mechanism/downstream/value-path probe series, and REC-004O/P sets. Collision
replacement is deterministic and independent of model output. The dataset is
development-exposed and not a sealed or RG3 query set.

Read only these source checkpoints, verifying canonical state hashes against
the original learning curve or terminal manifest when one exists:

- I03 P at 6000 (`REC-004D`);
- I03 SCORE_ONLY and CP_SCORE at 12000 (`REC-004O/P`);
- I03 historical JOINT at 12000 (`REC-004G`);
- I04 P at 6000 and at 7000 (`REC-004D/G`).

I04 is descriptive successful-trajectory context, not oracle gradient ground
truth. If its requested 7000 checkpoint were missing, the nearest earlier
saved checkpoint would be disclosed as `CONTROL_CHECKPOINT_SUBSTITUTED`; no
training may create one.

## Diagnostic calculations

Use normal J0 token cross entropy only. Observe the real pre-softmax attention
score `S = scaled(QK^T) + B + M`. `mirror_halves_position_map` is diagnostic
only. Per example, head, and output position record `dL_token/dS`,
`dL_align/dS`, cosine, dot, norms, correct-key and strongest-wrong-key task
gradients, and the analytical correct-key margin derivative under `-dL_token`.
`L_align` is a mean valid-key cross entropy to the mirror position mapping and
is never used for training.

For Q rows, K rows, `position_bias_hidden`, `position_bias_out`, and (only for
CP_SCORE) `CONTENT_PREP`, record task/alignment gradient norms, cosine,
finite-state status, and an update-to-weight scale proxy. Do not include fused
V rows in score-path quantities.

Reconstruct the next AdamW delta analytically from each checkpoint's saved
optimizer state and actual hyperparameters. Do not construct an update by
calling `optimizer.step()`. Record raw gradient and effective-update directions,
weight-decay, first-moment, second-moment, and a zero-moment counterfactual.
Use a forward-mode JVP to measure the first-order real score-margin change under
the reconstructed delta; no parameter is mutated.

## Fixed labels and stop

`TASK_LOSS_SCORE_CREDIT_MISALIGNMENT_SUPPORTED` requires both terminal I03
repair arms to decrease the length-10 position-4 oracle margin under task-loss
descent for at least 60% of valid rows, while I04@7000 does not. An AdamW
conflict requires both terminal arms' raw direction to improve at least 60% and
their effective AdamW delta to decrease at least 60%. Weak signal uses a
pre-registered one-order-of-magnitude (`<=0.10`) norm-ratio criterion against
I04@7000 and the corresponding length-6--9 aggregate. Otherwise the label is
`SCORE_CREDIT_ASSIGNMENT_UNRESOLVED`.

Write the task artifacts listed in the user-supplied contract under
`runs/phase_b_b2_model_bundle_recovery/rec004q/run_001/`. The task ends with:

```text
new_optimizer_updates = 0
selected_init = selected_step = selected_objective = selected_optimizer_change = null
child_bundle = null
rg3_recheck = NOT_EXECUTED
rec005_eligible = false
```
