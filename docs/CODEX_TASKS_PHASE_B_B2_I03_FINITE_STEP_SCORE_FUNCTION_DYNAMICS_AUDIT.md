> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# I03 Finite-Step Score-Function Dynamics vs Local Linear Prediction Audit

Task ID: `B-C005REC-004S`. This is the read-only diagnostic immediately after
REC-004R and before REC-005. It performs zero optimizer updates and does not
authorize a repair, candidate selection, child bundle, RG3, REC-005, or sealed
evaluation.

The task tests whether the local linear prediction of the next parameter update
(derived from the analytical AdamW delta on the actual next training batch)
faithfully matches the exact finite-difference score-function and attention-distribution
changes under step sizes $\alpha=0.1$ and $\alpha=1.0$, or whether optimizer-sized
overshoot or strong parameter-to-score non-linearities explain I03's failure.

It uses the six immutable checkpoints:
- `I03 P @6000`
- `I03 SCORE_ONLY @12000` (REC-004O run_003)
- `I03 CP_SCORE @12000` (REC-004P run_001)
- `I03 JOINT @12000` (REC-004G)
- `I04 P @7000` (successful control)
- `I04 P @6000` (baseline control)

The evaluation uses two disjoint probe datasets (1024 examples each):
1. `score_credit_assignment_probe_v1` (byte-identical continuity set from REC-004Q)
2. `score_function_dynamics_probe_v1` (fresh confirmation set, verified disjoint from
   the training stream 1--18000, prior evaluation sets, and protected registry)

For each checkpoint, the next training batch is generated deterministically:
- `@6000` -> step 6001
- `@7000` -> step 7001
- `@12000` -> step 12001

The analytical AdamW delta $\Delta\theta_{\text{adam}}$ is reconstructed without
calling `optimizer.step()`, targeting only the score parameters:
- `Q projection rows`
- `K projection rows`
- `position_bias_hidden`
- `position_bias_out`
- `CONTENT_PREP` (for CP_SCORE, JOINT, and P; excluded for SCORE_ONLY)

Local linear prediction $\Delta S_{\text{linear}} = J_S(\theta) \cdot \Delta\theta_{\text{adam}}$
is evaluated via autograd forward-mode JVP (`torch.func.jvp`).
Exact finite virtual steps $\theta_{\text{virtual}} = \theta + \alpha \Delta\theta_{\text{adam}}$
are evaluated for $\alpha \in \{0.1, 1.0\}$.

Implementation/locality parity requires:
`median cosine(ΔS_linear, ΔS_exact_0.1) >= 0.95` across all checkpoints on both probes.
Primary operational labels:
- `OPTIMIZER_SIZED_SCORE_FUNCTION_OVERSHOOT_SUPPORTED`: $\ge 60\%$ of $\Delta m_{\text{linear}} > 0$ cases invert to $\Delta m_{\text{exact}, 1.0} < 0$ while $\ge 80\%$ improve at $\alpha=0.1$ in both SCORE_ONLY and CP_SCORE, and I04 does not.
- `STRONG_PARAMETER_TO_SCORE_NONLINEARITY_SUPPORTED`: median $r_{1.0} \ge 1.0$ and median cosine $\le 0.5$ in both SCORE_ONLY and CP_SCORE, and I04 does not.
- `ONE_STEP_SCORE_DYNAMICS_LOCALLY_FAITHFUL`: median cosine at $\alpha=1.0 \ge 0.8$, median $r_{1.0} < 0.5$, and margin sign agreement $\ge 0.8$ in both terminal arms.
- `FINITE_STEP_SCORE_DYNAMICS_MIXED_OR_UNRESOLVED`: otherwise.

Artifacts are written under `runs/phase_b_b2_model_bundle_recovery/rec004s/run_001/`.
The task always records `new_optimizer_updates = 0`, null selection fields,
`rg3_recheck = NOT_EXECUTED`, and `rec005_eligible = false`.
