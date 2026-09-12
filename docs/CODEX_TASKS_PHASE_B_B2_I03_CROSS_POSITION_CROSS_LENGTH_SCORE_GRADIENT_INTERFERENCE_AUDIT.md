> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# I03 Cross-Position / Cross-Length Score-Gradient Interference Audit

Task ID: `B-C005REC-004R`. This is the read-only diagnostic immediately after
REC-004Q. It performs zero optimizer updates and does not authorize a repair,
candidate selection, child bundle, RG3, REC-005, or sealed evaluation.

The task tests only whether the *actual training-loss aggregation* over shared
score parameters suppresses the length-10 output-position-4 token gradient.
It uses the six immutable checkpoints and the byte-identical
`score_credit_assignment_probe_v1` from REC-004Q. Before any model output it
fixes 128 unique training steps `6001 + floor(k * 6000 / 128)` for
`k=0..127`, regenerates their batches from the original deterministic stream,
and verifies their digest against the REC-004O/P stream contract.

For Q rows, K rows, `position_bias_hidden`, and `position_bias_out`, the live
masked-token CE implementation is decomposed by `(input length, output
position)`. The direct score-parameter gradient of each batch loss must agree
with the sum of its exactly weighted strata gradients (absolute tolerance
`1e-6`, relative tolerance `1e-4`); otherwise the task stops with
`LOSS_GRADIENT_DECOMPOSITION_INVALID`. Fused QKV row slices are separated
without treating V rows as score-path parameters. `CONTENT_PREP` may be
reported separately for CP_SCORE but is not part of any decision gate.

The primary target is length 10, position 4; length 10, position 5 is a
secondary target. For every score group and their concatenation, report dot,
cosine, and retention against (a) length-10 other positions and (b) all
positions of lengths 6--9. The same exact token-mean weighting is retained.
The oracle `pi_n` alignment gradient is diagnostic-only. A fixed 128-example
subset of the REC-004Q probe supplies the per-example stability check, with a
single fixed cross-length reference mini-batch for every example.

The operational labels and thresholds are those supplied with this task:
60% negative same-position/cross-length drive in both I03 terminal arms and
below 60% in I04@7000; or median retention <=0.25 in both I03 terminal arms
and >0.50 in I04@7000. A mixed label needs both components. No optimizer state
search or repair training is performed.

Artifacts are written under
`runs/phase_b_b2_model_bundle_recovery/rec004r/run_001/`. The task always
records `new_optimizer_updates = 0`, null selection fields,
`rg3_recheck = NOT_EXECUTED`, and `rec005_eligible = false`.
