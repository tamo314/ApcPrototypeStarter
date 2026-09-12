# Architecture Decision Log -- Phase B / B2 Model Bundle Recovery (Active: Task B-C005REC-004T onward)

Part of the split docs/DECISIONS.md architecture decision log (ADR-0116 onward). See docs/DECISIONS.md for the full index across all phases.

Earlier ADRs for this recovery phase are archived in:
- [docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART1.md](DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART1.md): ADR-0092 through ADR-0105 (Tasks B-C005REC-001 through B-C005REC-004J: Core/Bundle Restoration, Build DAG, and MIRROR_HALVES Position Bias Repair).
- [docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART2.md](DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART2.md): ADR-0106 through ADR-0115 (Tasks B-C005REC-004K through B-C005REC-004S: I03 Attention/Downstream Mechanism Attribution, Leave-One-Out Necessity, and Dynamics).

**Active phase.** A recovery sub-branch of docs/DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md (ADR-0082-0091), not a new phase -- it does not relax any R3/G4 threshold and does not authorize B-C005R3-011. See docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md, docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md, docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md, and docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md. Append new ADRs from this recovery sequence here.

Use this file for short decisions discovered during implementation. Do not rewrite history; append entries.

---

## ADR-0116: I03 Dense Trajectory Transition Replay & Oracle-Compatibility Onset Audit: Transition Window Localized to [7500, 8000], Bit-Exact Parity Verified, and Early Score Margin Peak Precedes Downstream Compatibility Loss (Task B-C005REC-004T, `trajectory_diagnosis: SCORE_DEGRADATION_PRECEDES_DOWNSTREAM_COMPATIBILITY_LOSS`)

**Date:** 2026-09-11

**Status:** Accepted (Task B-C005REC-004T complete; historical dense trajectory transition replay and mechanism audit. No new candidate training; zero non-historical optimizer updates. Stage A mechanically localized the first transition window to steps [7500, 8000] from saved 500-step checkpoints. Stage B executed 500 updates of historical dense replay from step 7500 with bit-exact parity verified at step 8000 against the saved historical state. Stage C established temporal ordering across 25-update full probe probes and per-step dynamics, demonstrating that attention score margin degradation precedes downstream oracle compatibility loss. Stage D produced next repair contract proposal `PROPOSED_NOT_AUTHORIZED`.)

**Affects:** `src/apc/evaluation/mirror_dense_trajectory_transition_audit.py` (new), `configs/phase_b_b2_model_bundle_recovery_rec004t.yaml` (new), `tests/test_mirror_dense_trajectory_transition_audit.py` (new), `scripts/run_phase_b_b2_model_bundle_recovery.py` (`--task B-C005REC-004T` dispatch added), `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`, `docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`, `AGENTS.md`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`. No file under `src/apc/primitives/` or `src/apc/core/` was modified.

**Run artifacts:** `runs/phase_b_b2_model_bundle_recovery/rec004t/run_001/`.
The directory records source trajectory manifest, probe dataset manifest, coarse checkpoint scan, transition window decision, dense replay protocol, dense replay parity audit, per-step training trace, per-step sentinel dynamics, 25-step full probe dynamics, internal stage dynamics NPZ and index, parameter group drift, position-4 transition summary, position-5 control summary, I04 success control, freeze audit, side-effect audit, cost accounting, trajectory transition decision, next repair contract proposal, summary, and report.

### Context

Prior diagnostics REC-004Q/R/S refuted isolated instantaneous hypotheses (gradient misdirection, AdamW state sign reversal, gradient interference, single-step overshoot, and strong score nonlinearity). Meanwhile, historical JOINT training exhibited a non-monotonic trajectory: early score margin improvement between steps 6000 and 7000 followed by severe oscillations, degradation toward step 12000, and an ultimate breakdown of downstream oracle compatibility ($O_1 \ge 0.95$ at step 6000 vs $\sim 0.75$ at step 17500).
Task B-C005REC-004T was commissioned to reconstruct the historical dense trajectory, pinpoint the exact transition window where downstream oracle compatibility is first lost, and establish the temporal sequence connecting attention score dynamics, attention distributions, downstream outputs, and 10-parameter-group drift.

### Evidence

1. **Stage A: Coarse localization isolates transition to [7500, 8000].** Scanning all 25 historical checkpoints (steps 6000 to 18000 at 500-step intervals) on both continuity (128 examples) and fresh (128 examples) probe datasets identified step 7500 as the last common checkpoint with $O_1 \text{ EM} \ge 0.95$ ($0.998$ on both splits). At step 8000, $O_1 \text{ EM}$ plummeted to $0.576$ (continuity) and $0.592$ (fresh). The window was mechanically locked to $[7500, 8000]$ (500 updates, satisfying $\le 1000$ bound).
2. **Stage B: Bit-exact historical dense replay achieved.** Replaying 500 updates from step 7500 using exact historical state (RNG states, batch sequence, AdamW optimizer, and cosine scheduler) achieved perfect parity at step 8000 against the historical saved checkpoint:
   - `max_state_abs_diff`: $0.0$
   - `replayed_canonical_state_hash`: `4836cea9d83d2b1af38e6c19b7e5135713b7ec5bfca5f3ed446cf4a0d4e31bc6` (byte-identical)
   - `max_optimizer_moment_abs_diff`: $0.0$ ($< 10^{-6}$)
   - `scheduler_epoch_matches`: `true`
   - `status`: `BIT_EXACT`
3. **Stage C: Score degradation precedes downstream oracle compatibility loss.**
   - $T_{\text{score\_peak}} = 7500$ (best position-4 score margin: $-30.345$)
   - $T_{\text{o1\_margin\_peak}} = 7500$
   - $T_{\text{j0\_output\_peak}} = 7500$
   - $T_{\text{oracle\_loss}} = 7525$ (first full probe step where $O_1 \text{ EM} < 0.95$)
   The attention score margin began degrading immediately after step 7500, preceding the loss of downstream compatibility at step 7525. Diagnosis: `SCORE_DEGRADATION_PRECEDES_DOWNSTREAM_COMPATIBILITY_LOSS`.
4. **Observer non-interference verified.** Manual decomposition of MultiheadAttention for internal stage extraction showed zero discrete prediction divergence and maximal absolute error of $\approx 2.5 \times 10^{-5}$ (within float32 precision limits, $\ll 5 \times 10^{-3}$).
5. **Auxiliary control on I04 (successful trajectory).** Over the identical step range (6000 to 18000), I04 maintained $O_1 \text{ EM} \ge 0.95$ throughout without experiencing downstream compatibility breakdown or score margin collapse.
6. **Strict isolation and zero new updates.** `side_effect_audit.json` verifies `diagnostic_replay_optimizer_updates = 500`, `new_candidate_training_updates = 0`, `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, and `rec005_eligible = false`. `freeze_audit.json` confirms Core and all 12 protected operations remain bitwise unchanged.

### Consequences

- **Diagnosis established:** I03's failure is not an instantaneous single-step breakdown nor an initial downstream degradation, but an early loss of attention score stability (peaking at step 7500) that subsequently drags downstream components out of the oracle-compatible basin around step 7525.
- **Next repair contract proposed (not authorized):** `short-horizon score-stability repair` focusing on stabilizing attention score margins around step 7500 without premature downstream component freezes.
- **Scope bounded:** No candidate training or candidate selection occurred. RG3/REC-005 remain blocked.

## ADR-0117: I03 Pre-Transition Attention-Clamp Causal Replay & Score-Stability Repair Gate: Fixing Step-7500 Attention Trajectory Fails to Prevent Downstream Oracle-Compatibility Collapse, Establishing Downstream Drift Is Autonomous and Refuting Score-Degradation as the Primary Causal Driver (Task B-C005REC-004U, `causal_diagnosis: DOWNSTREAM_COMPATIBILITY_LOSS_PERSISTS_UNDER_FIXED_ATTENTION`)

**Date:** 2026-09-11

**Status:** Accepted (Task B-C005REC-004U complete; causal diagnosis counterfactual training pilot. Counterfactual optimizer updates = 500; new candidate training updates = 0. Model bundle recovery RG3/REC-005 remains blocked.)

**Affects:** `src/apc/evaluation/mirror_attention_clamp_causal_replay.py` (new), `configs/phase_b_b2_model_bundle_recovery_rec004u.yaml` (new), `tests/test_mirror_attention_clamp_causal_replay.py` (new), `scripts/run_phase_b_b2_model_bundle_recovery.py` (`--task B-C005REC-004U` dispatch added), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`. No file under `src/apc/primitives/` or `src/apc/core/` was modified.

**Run artifacts:** `runs/phase_b_b2_model_bundle_recovery/rec004u/run_001/`.
The directory records source and REC-004T control manifest, attention clamp protocol, attention reference manifest, initial parity audit, fused QKV freeze audit, training batch digests, counterfactual training trace, 25-step multi-dataset metrics, attention reference drift trace, O1 compatibility trajectory, fresh probe manifest, causal decision, next repair contract, freeze audit, side-effect audit, cost accounting, summary, and report.

### Context

Task B-C005REC-004T established that historical JOINT attention score degradation peaked at step 7500 and temporally preceded downstream oracle compatibility breakdown at step 7525. However, temporal precedence is not causation. To determine whether attention trajectory drift was the causal driver of downstream oracle compatibility collapse, Task B-C005REC-004U implemented a counterfactual training intervention (`ATTENTION_CLAMP_7500`): clamping the training-time attention distribution to the immutable normal J0 attention $A_{\text{ref}}$ produced by step-7500 I03, while training the downstream parameters (`CONTENT_PREP`, `V`, `ATTN_OUT_PROJ`, `QUERY_RESIDUAL`, `POST_ATTN_NORM`, `FFN`, `READOUT`) across the identical 500 historical batches (steps 7501–8000). Score-only parameters (Q, K rows and position bias) were frozen fail-closed, with exact weight and AdamW moment invariance verified at every step.

### Evidence

1. **Information boundary and initial parity verified:**
   - Training APIs and clamped forward functions enforce zero access to oracle $\pi_n$, target tokens, or teacher positions.
   - At step 7500, initial parity between normal J0 forward and clamped forward passed pre-declared tolerances (`attn_diff = 1.04e-6 < 1e-5`, `logits_diff = 2.03e-5 < 1e-4`, discrete predictions exactly matched).
2. **Fail-closed Q/K and position-bias freeze maintained:**
   - At every step of the 500 updates, Q and K rows of `cross_attn.in_proj_weight` and `in_proj_bias`, as well as `position_bias_hidden` and `position_bias_out`, remained strictly invariant to step-7500 parameters and AdamW moments (`max_state_diff = 0.0`).
3. **Primary causal finding: downstream compatibility collapses identically under fixed attention:**
   - $T_{\text{oracle\_loss\_clamp}} = 7525$ (identical to $T_{\text{oracle\_loss\_historical}} = 7525$).
   - At step 7525, $O_1\text{ EM}$ plummeted below 0.95 across all three datasets (Continuity 1: $0.881$, Continuity 2: $0.904$, Fresh: $0.891$).
   - At step 8000, $O_1\text{ EM}$ reached the identical collapsed level as historical replay:
     - Continuity 1 (`length10_mechanism_probe_v1`): $0.5762$ vs historical $0.5762$ ($\Delta = +0.0000$).
     - Continuity 2 (`dense_trajectory_transition_probe_v1`): $0.5918$ vs historical $0.5918$ ($\Delta = +0.0000$).
     - Fresh Confirmation (`attention_clamp_causal_probe_v1`): $0.5938$.
   - Position-4 accuracy also dropped identically to $0.5762$ / $0.5918$ / $0.5938$.
4. **Classification:**
   - Causal diagnosis: `DOWNSTREAM_COMPATIBILITY_LOSS_PERSISTS_UNDER_FIXED_ATTENTION`.
   - The hypothesis that attention degradation causally drove downstream compatibility collapse is refuted. Downstream component drift occurs autonomously during joint batch exposure.
5. **Auxiliary controls and strict boundaries:**
   - Successful trajectory I04 displayed healthy compatibility ($O_1\text{ EM} \ge 0.70$–$0.75$, maintaining robust margin and J0 accuracy $\sim 0.88$–$0.96$).
   - `side_effect_audit.json` confirms `counterfactual_optimizer_updates = 500`, `new_candidate_training_updates = 0`, `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, and `rec005_eligible = false`.

### Consequences

- **Refutation of attention-score repair priority:** Attention-stability or score-regularization repairs around step 7500 cannot prevent downstream compatibility collapse, because downstream collapse occurs even when attention distribution drift is artificially eliminated.
- **Next repair direction:** Focus shifts back to downstream trajectory drift (C / V / O / FFN) under joint batch updates, as specified in `next_repair_contract.md` (`DOWNSTREAM_COMPATIBILITY_LOSS_PERSISTS_UNDER_FIXED_ATTENTION`).
- **Scope bounded:** No candidate training or candidate selection occurred. RG3/REC-005 remain blocked.

## ADR-0118: I03 Attention-Clamped Downstream Freeze Necessity Replay: Single-Group Downstream Freezes Fail to Prevent Compatibility Collapse, but Joint Freezing of CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ, and FFN_BLOCK Fully Preserves Step-8000 O1 Compatibility, Establishing Joint Downstream Drift as the Necessary Driver of Collapse (Task B-C005REC-004V, `causal_decision: CVOF_JOINT_DRIFT_NECESSARY_FOR_COLLAPSE`)

**Date:** 2026-09-11

**Status:** Accepted (Task B-C005REC-004V complete; causal diagnosis counterfactual training replay. Counterfactual optimizer updates = 2500 (replay) + 25 (Stage A parity) = 2525; new candidate training updates = 0. Model bundle recovery RG3/REC-005 remains blocked.)

**Affects:** `src/apc/evaluation/mirror_downstream_freeze_causal_replay.py` (new), `configs/phase_b_b2_model_bundle_recovery_rec004v.yaml` (new), `tests/test_mirror_downstream_freeze_causal_replay.py` (new), `scripts/run_phase_b_b2_model_bundle_recovery.py` (`--task B-C005REC-004V` dispatch added), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`. No file under `src/apc/primitives/` or `src/apc/core/` was modified.

**Run artifacts:** `runs/phase_b_b2_model_bundle_recovery/rec004v/run_001/`.
The directory records source REC-004U manifest, Stage A baseline parity audit, downstream freeze protocol, fresh probe manifest (4 locked datasets), per-arm subdirectories (`D_C/`, `D_V/`, `D_O/`, `D_F/`, `D_CVOF/`) with respective trace/metrics/freeze-audit files, collated per-step freeze audit, collated training trace, collated 25-step functional metrics, oracle loss onset summary, clamped vs oracle tradeoff, single-group protection summary, CVOF joint protection summary, causal decision, next repair contract, freeze audit, side-effect audit, cost accounting, summary, and report.

### Context

Task B-C005REC-004U demonstrated that freezing the normal J0 attention distribution to step-7500 state ($A_{\text{ref}}$) failed to prevent downstream oracle-compatibility collapse ($T_{\text{oracle\_loss}} = 7525$), proving that attention score degradation was not the causal driver of downstream degradation. In historical backward-rollback diagnostics (REC-004L/M/N), rolling back the joint combination of `CONTENT_PREP`, `V_PROJECTION`, `ATTN_OUT_PROJ`, and `FFN_BLOCK` from step 17500 to step 6000 was necessary and sufficient for full recovery.
Task B-C005REC-004V was commissioned to test forward counterfactual training during steps 7501–8000 under fixed attention clamp: evaluating which downstream group freeze ($D_C, D_V, D_O, D_F$) or joint freeze ($D_{CVOF}$) prevents oracle-compatibility collapse.

### Evidence

1. **Stage A Baseline Parity Verified:**
   - 25 updates of `BASELINE_CLAMP` from step 7500 to step 7525 reproduced REC-004U baseline metrics within float tolerance: `loss_diff = 0.0 < 1e-4`, `cont1_o1_diff = 0.0 < 1e-4`, `cont1_p4_diff = 0.0 < 1e-4`, confirming exact code path parity.
2. **Selective Freeze Contract Enforced:**
   - At every step across all 5 arms (2500 total replay updates), frozen parameter values and AdamW 1st/2nd moments were preserved bitwise against step-7500 state (`max_frozen_diff = 0.0`). Fused QKV slicing correctly froze target rows (Q/K rows always; V rows in $D_V$ and $D_{CVOF}$) while allowing non-frozen rows and downstream components to train.
3. **Single-Group Freezes All Fail ($T_{\text{oracle\_loss}} = 7525$):**
   - In all four single-freeze arms ($D_C, D_V, D_O, D_F$), downstream oracle compatibility collapsed at the very first evaluation step ($T_{\text{oracle\_loss}} = 7525$), identical to unconstrained clamp baseline $D_0$.
   - At step 8000, $O_1\text{ EM}$ remained degraded ($D_C: 0.682$, $D_V: 0.684$, $D_O: 0.576$, $D_F: 0.684$), and position-4 accuracy remained low ($\sim 0.58$–$0.69$). None satisfied Strong or Partial protection criteria.
4. **Joint $D_{CVOF}$ Freeze Achieves Complete Strong Protection:**
   - Freezing the joint 4-group set (`CONTENT_PREP` + `V_PROJECTION` + `ATTN_OUT_PROJ` + `FFN_BLOCK`) completely prevented compatibility collapse throughout all 500 updates:
     - $T_{\text{oracle\_loss}}(D_{CVOF}) = \text{None}$ (never dropped below 0.95 across continuity splits).
     - Step 8000 $O_1\text{ EM}$: Continuity 1 = **$0.9980$**, Continuity 2 = **$0.9980$**, Continuity 3 = **$1.0000$**, Fresh Confirmation (`downstream_freeze_causal_probe_v1`) = **$1.0000$**.
     - Step 8000 Position-4 Accuracy: **$1.0000$** across all 4 datasets.
5. **Causal Classification (Case B):**
   - Diagnosis: `CVOF_JOINT_DRIFT_NECESSARY_FOR_COLLAPSE`.
   - The breakdown of oracle compatibility cannot be attributed to any single downstream group alone, but is causally driven by the multi-component joint co-adaptation of C, V, O, and FFN under training batch exposure.
6. **Auxiliary Tradeoff Label:**
   - In single arms $D_C, D_V, D_O, D_F$, $A_{\text{ref}}$-clamped task performance improved slightly while $O_1$ compatibility collapsed, exhibiting `FIXED_ATTENTION_ADAPTATION_ORACLE_COMPATIBILITY_TRADEOFF`.
7. **Audit and Resource Bounds:**
   - Total updates: 2500 counterfactual + 25 Stage A parity = 2525 updates. `new_candidate_training_updates = 0`.
   - Core and 12 protected primitives remained bitwise invariant. `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, `rec005_eligible = false`.

### Consequences

- **Causal Mechanism Established:** Downstream oracle-compatibility collapse during transition is causally driven by joint drift across the four value-path and content components (CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ, FFN_BLOCK). Stopping any single group update does not prevent collapse, whereas jointly stabilizing the four groups completely protects compatibility.
- **Next Repair Direction:** Rather than an unprincipled permanent freeze of all 4 groups, the next task should design a consolidation / plasticity trade-off rule for CVOF joint parameters to enable learning new attention tasks while preserving oracle compatibility, as specified in `next_repair_contract.md` (`AUTHORIZED_JOINT_LEAD`).
- **Scope Bounded:** No candidate training or candidate selection occurred. RG3/REC-005 remain blocked.

## ADR-0119: I03 Normal-Attention CVOF Protection Continuation Pilot: Freezing CVOF Joint Groups Under Unconstrained Normal Forward Perfectly Preserves Downstream Oracle Compatibility (O1 EM = 0.998–1.000) Across All Length-10 Splits, but Gains Over Historical J0 Task Trajectory Are Sub-Threshold (+0.038 on Normal Validation, +0.035 on Length-10 Confirmation vs +0.10 Floor), Establishing Hard CVOF Protection Blocks Sufficient Task Learning (Task B-C005REC-004W, `result_label: CVOF_PROTECTION_PRESERVES_COMPATIBILITY_BUT_BLOCKS_LEARNING`)

**Date:** 2026-09-11

**Status:** Accepted (Task B-C005REC-004W complete; mechanism-repair continuation pilot. Counterfactual intervention updates = 500, control parity updates = 25; new candidate training updates = 0. Model bundle recovery RG3/REC-005 remains blocked.)

**Affects:** `src/apc/evaluation/mirror_normal_cvof_protection_pilot.py` (new), `configs/phase_b_b2_model_bundle_recovery_rec004w.yaml` (new), `tests/test_mirror_normal_cvof_protection_pilot.py` (new), `scripts/run_phase_b_b2_model_bundle_recovery.py` (`--task B-C005REC-004W` dispatch added), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`. No file under `src/apc/primitives/` or `src/apc/core/` was modified.

**Run artifacts:** `runs/phase_b_b2_model_bundle_recovery/rec004w/run_001/`.
The directory records source manifest (@7500 verified), historical control manifest (REC-004T verified), protocol manifest, initial parity audit (PASS), Stage A parity audit (PASS), fresh validation manifest (2 locked fresh datasets + 4 continuity datasets), selective freeze audit, training batch manifest, per-step training trace, per-25step dynamics, endpoint metrics, historical endpoint comparison, O1 compatibility audit, J0 learning audit, result decision, next repair contract, freeze audit, side-effect audit, cost accounting, summary, and report.

### Context

Task B-C005REC-004V proved that joint drift across CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ, and FFN_BLOCK (CVOF) is the necessary causal driver of downstream compatibility collapse under an attention-clamped condition ($A_{\text{ref}}$ from step 7500).
Task B-C005REC-004W was commissioned as an I03-specific mechanism-repair pilot to test the primary research question: **whether completely removing the attention clamp and returning to unconstrained production J0 forward while strictly freezing CVOF at step 7500 enables continued learning of score pathways (Q/K rows, position bias) while maintaining downstream $O_1$ compatibility and improving normal J0 performance over the historical trajectory**.

### Evidence

1. **Source State and Baseline Parities Verified:**
   - Step 7500 source state verified bit-exact against REC-004T/U/V canonical hash (`7c71a7a43ec2ba766623a70cf4b62e18a8ad685bb1073657ef3fd54d676cb3cb`).
   - Initial parity at step 7500 passed (`score_diff = 7.63e-6`, `prob_diff = 1.04e-6`, `final_logits_diff = 2.67e-5`, `pred_mismatches = 0` within $10^{-4}$ tolerance).
   - Stage A historical control parity reproduced REC-004T step-7525 loss exactly (`loss_diff = 0.0 < 1e-4`).
2. **Selective Freeze Contract Enforced:**
   - Across all 500 optimizer updates (steps 7501–8000), CONTENT_PREP, V_PROJECTION (rows 64:96 of `cross_attn.in_proj`), ATTN_OUT_PROJ, and FFN_BLOCK parameter values and AdamW 1st/2nd moments remained bitwise identical to step 7500 (`max_freeze_diff = 0.0`).
   - Trainable parameters (Q rows, K rows, position bias, query residual, post-attention norm, readout) actively updated throughout training (e.g. Q/K update norm $\sim 6\times 10^{-3}$ to $1.1\times 10^{-2}$, position-bias update norm $\sim 1.3\times 10^{-3}$ to $3.1\times 10^{-3}$).
3. **Question A — Complete Downstream Oracle Compatibility Preservation (PASS):**
   - At step 8000, $O_1\text{ EM}$ and position-4 accuracy remained at ceiling across all 5 length-10 splits:
     - Continuity 1 (`length10_mechanism_probe_v1`): $O_1\text{ EM} = \mathbf{0.9980}$, pos4 acc = $\mathbf{1.0000}$
     - Continuity 2 (`dense_trajectory_transition_probe_v1`): $O_1\text{ EM} = \mathbf{0.9980}$, pos4 acc = $\mathbf{1.0000}$
     - Continuity 3 (`attention_clamp_causal_probe_v1`): $O_1\text{ EM} = \mathbf{1.0000}$, pos4 acc = $\mathbf{1.0000}$
     - Continuity 4 (`downstream_freeze_causal_probe_v1`): $O_1\text{ EM} = \mathbf{1.0000}$, pos4 acc = $\mathbf{1.0000}$
     - Fresh Length-10 Confirmation (`normal_cvof_protection_length10_v1`): $O_1\text{ EM} = \mathbf{1.0000}$, pos4 acc = $\mathbf{1.0000}$
   - This decisively confirms that downstream compatibility collapse does **not** occur when score pathways train under unconstrained forward, provided CVOF is protected.
4. **Questions B & C — Sub-Threshold Normal Task Learning Gain (FAIL):**
   - On the fresh normal validation set (`normal_cvof_protection_validation_v1`, 1024 examples, lengths 2..10):
     - Protected arm J0 EM = **$0.7861$** vs Historical @8000 = **$0.7480$** ($\Delta = \mathbf{+0.0381}$, below the $+0.10$ effect-size floor).
     - Length-10 subset J0 EM = **$0.0853$** vs Historical @8000 = **$0.0569$** ($\Delta = \mathbf{+0.0284}$).
   - On the fresh length-10 confirmation set (`normal_cvof_protection_length10_v1`, 512 examples):
     - Protected arm J0 EM = **$0.0859$** vs Historical @8000 = **$0.0508$** ($\Delta = \mathbf{+0.0352}$, below the $+0.10$ floor).
5. **Decision Classification (Section 15):**
   - `result_label: CVOF_PROTECTION_PRESERVES_COMPATIBILITY_BUT_BLOCKS_LEARNING`.
   - Hard permanent freezing of CVOF prevents oracle collapse, but severely restricts representation plasticity, blocking the model from learning the normal length-10 position correspondence task to the required $+0.10$ gain floor.
6. **Cost and Scope Accounting:**
   - `counterfactual_intervention_updates = 500`, `parity_updates = 25`, `new_candidate_training_updates = 0`.
   - Core and 15 other primitives invariant. `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, `rec005_eligible = false`.

### Consequences

- **Mechanistic Finding:** Hard CVOF protection successfully decouples downstream stability from attention training, entirely preventing downstream collapse under unconstrained normal forward. However, hard zero-gradient clamping of CVOF simultaneously impairs the value-projection and content representation adaptation necessary for full task learning.
- **Next Repair Direction:** Hard freezing is too restrictive. As authorized by Section 20 of the task contract, the next task should explore a **soft-stability / proximal-plasticity mechanism** (e.g. bounded trust region or proximal anchor regularization on CVOF updates) rather than a rigid freeze.
- **Scope Bounded:** Diagnostic continuation pilot only. No candidate was selected, and RG3 / REC-005 remain blocked.

## ADR-0120: I03 CVOF Pre-Transition Trust-Region Plasticity Pilot: Constraining CVOF Joint Drift to Pre-Transition (7000->7500) Radial Bounds (1.0x) Fails to Preserve Downstream Oracle Compatibility (O1 EM = 0.76–0.81 Across Length-10 Splits vs 0.95 Floor) and Produces No Substantial J0 Task Learning Gain Over Hard Freeze, Disconfirming the Pre-Transition Radius Trust-Region Hypothesis (Task B-C005REC-004X, `result_label: PRETRANSITION_RADIUS_TRUST_REGION_NOT_SUPPORTED`)

**Date:** 2026-09-11

**Status:** Accepted (Task B-C005REC-004X complete; mechanism-repair continuation pilot. Counterfactual intervention updates = 500, control parity updates = 25; new candidate training updates = 0. Model bundle recovery RG3/REC-005 remains blocked.)

**Affects:** `src/apc/evaluation/mirror_cvof_trust_region_pilot.py` (new), `configs/phase_b_b2_model_bundle_recovery_rec004x.yaml` (new), `tests/test_mirror_cvof_trust_region_pilot.py` (new), `scripts/run_phase_b_b2_model_bundle_recovery.py` (`--task B-C005REC-004X` dispatch added), `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`. No file under `src/apc/primitives/` or `src/apc/core/` was modified.

**Run artifacts:** `runs/phase_b_b2_model_bundle_recovery/rec004x/run_001/`.
The directory records source manifest (@7500 verified), pretransition radius calibration (VERIFIED), protocol manifest, initial parity audit (PASS), Stage A parity audit (PASS), fresh validation manifest (2 locked fresh datasets + 4 continuity datasets), CVOF boundary summary, per-step training trace, per-step trust region trace, per-25step dynamics, endpoint metrics, historical comparison, hard freeze comparison, O1 compatibility audit, plasticity audit, result decision, next repair contract, freeze audit, side-effect audit, cost accounting, summary, and report.

### Context

Task B-C005REC-004W proved that completely freezing the four CVOF groups (CONTENT_PREP, V_PROJECTION, ATTN_OUT_PROJ, FFN_BLOCK) under unconstrained normal forward execution perfectly preserves downstream $O_1$ compatibility ($O_1\text{ EM} \ge 0.998$ across all length-10 splits), but severely curtails task learning, producing sub-threshold J0 EM gains over the historical trajectory ($+0.038$ on normal validation, $+0.035$ on length-10 confirmation vs $+0.10$ floor).
Task B-C005REC-004X was commissioned as an I03-specific mechanism-repair pilot testing a single intermediate approach between hard freeze and unconstrained training: **allowing CVOF to remain trainable while constraining cumulative drift from step 7500 within the pre-transition (steps 7000–7500) observed drift radius $R_g$ (multiplier 1.0x) via radial projection after each AdamW update**.
The pilot investigated whether this pre-transition trust-region constraint preserves downstream $O_1$ compatibility ($\ge 0.95$ across all length-10 splits) while unlocking greater J0 task learning gain than hard freeze.

### Evidence

1. **Pre-Transition Radius Calibration and Baseline Parity Verified:**
   - Evaluated step-7000 and step-7500 checkpoints across 4 length-10 continuity splits; all confirmed $O_1\text{ EM} \ge 0.9980$ and position-4 accuracy $= 1.0000$, validating calibration suitability.
   - Calibrated pre-transition drift radii (Euclidean norm of parameter differences between step 7000 and step 7500):
     - $R_C = 0.14780$ (`CONTENT_PREP`)
     - $R_V = 0.07506$ (`V_PROJECTION`, rows 64:96 of `cross_attn.in_proj`)
     - $R_O = 0.10736$ (`ATTN_OUT_PROJ`)
     - $R_F = 0.21301$ (`FFN_BLOCK`)
   - All radii strictly positive; multiplier locked at $1.0\times$.
   - Initial parity at step 7500: PASS (`score_diff = 7.63e-6`, `prob_diff = 1.04e-6`, `final_logits_diff = 2.67e-5`, `pred_mismatches = 0` within $10^{-4}$ tolerance).
   - Stage A parity: PASS (`loss_diff = 0.0 < 1e-4` against REC-004T at step 7525).
2. **Trust-Region Dynamics and Boundary Activity:**
   - Across 500 intervention updates (steps 7501–8000), all four parameter groups actively engaged the trust-region boundary:
     - Group C: 75.80% boundary hit fraction (379/500 steps, max consecutive = 89)
     - Group V: 74.20% boundary hit fraction (371/500 steps, max consecutive = 178)
     - Group O: 77.20% boundary hit fraction (386/500 steps, max consecutive = 386)
     - Group F: 69.20% boundary hit fraction (346/500 steps, max consecutive = 346)
   - Unconstrained groups (Q/K rows, position bias, norms, readout) updated freely, while Core and 15 other primitives remained bitwise invariant.
3. **Compatibility Gate: FAIL:**
   - At step 8000, downstream $O_1$ compatibility collapsed across all 5 length-10 splits, falling well below the $0.95$ threshold:
     - Continuity 1 (`length10_mechanism_probe_v1`): $O_1\text{ EM} = \mathbf{0.7891}$, pos4 acc = $\mathbf{0.7910}$
     - Continuity 2 (`dense_trajectory_transition_probe_v1`): $O_1\text{ EM} = \mathbf{0.7617}$, pos4 acc = $\mathbf{0.7617}$
     - Continuity 3 (`attention_clamp_causal_probe_v1`): $O_1\text{ EM} = \mathbf{0.8105}$, pos4 acc = $\mathbf{0.8105}$
     - Continuity 4 (`downstream_freeze_causal_probe_v1`): $O_1\text{ EM} = \mathbf{0.8047}$, pos4 acc = $\mathbf{0.8047}$
     - Fresh Length-10 Confirmation (`cvof_trust_region_length10_v1`): $O_1\text{ EM} = \mathbf{0.8027}$, pos4 acc = $\mathbf{0.8047}$
4. **Plasticity Gate: FAIL:**
   - Fresh normal validation (overall J0 EM, 1024 examples, lengths 2..10):
     - Trust-R1 = **$0.7539$** vs Historical @8000 = **$0.7500$** ($\Delta = \mathbf{+0.0039}$), vs Hard Freeze @8000 = **$0.7637$** ($\Delta = \mathbf{-0.0098}$).
     - Length-10 subset: Trust-R1 = **$0.0796$** vs Historical = **$0.0752$** ($\Delta = \mathbf{+0.0044}$), vs Hard Freeze = **$0.0885$** ($\Delta = \mathbf{-0.0088}$).
   - Fresh length-10 confirmation (J0 EM, 512 examples):
     - Trust-R1 = **$0.0488$** vs Historical @8000 = **$0.0547$** ($\Delta = \mathbf{-0.0059}$), vs Hard Freeze = **$0.0742$** ($\Delta = \mathbf{-0.0254}$).
   - Constraining CVOF to radius $R_g$ not only failed to match hard freeze J0 performance, but actually degraded confirmation performance below the historical unconstrained baseline.
5. **Decision Classification:**
   - `result_label: PRETRANSITION_RADIUS_TRUST_REGION_NOT_SUPPORTED`.
   - The pre-transition displacement radius $R_g$ observed during normal oracle-compatible trajectory is still far too large an envelope when applied as an isotropic radial constraint; allowing CVOF drift within this boundary is sufficient to destroy downstream $O_1$ compatibility while providing no plasticity gain for J0 task learning.
6. **Cost and Scope Accounting:**
   - `counterfactual_intervention_updates = 500`, `parity_updates = 25`, `new_candidate_training_updates = 0`.
   - Core and 15 other primitives invariant. `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, `rec005_eligible = false`.

### Consequences

- **Hypothesis Disconfirmed:** The hypothesis that "I03's failure is merely excessive drift beyond the safe local region observed pre-transition, and that confining CVOF updates within $R_g$ preserves compatibility while enabling learning" is refuted. The geometry of safe downstream representations cannot be captured by an isotropic parameter-norm ball calibrated from pre-transition drift.
- **Trust-Region Pilot Concluded:** Per contract constraints, radius sweeps (e.g. 0.5x, 2.0x), anchor-free trust regions, or auxiliary penalty formulations are not authorized and are not pursued.
- **Repair Implications:** Downstream compatibility protection requires either strict freezing of shared value/FFN components or architecture-level isolation (e.g. task-specific adapters, subspace decoupling, or downstream primitive recalibration) rather than parameter-space ball constraints on shared weights.
- **Scope Bounded:** Diagnostic continuation pilot only. Candidate selection and RG3 / REC-005 recheck remain blocked.


## ADR-0121: I03 CVOF Functional-Stability Gate Identifiability Audit — Function-Space Drift Metrics (M1–M5) Fail to Pre-Identify Compatibility Collapse (R_onset < 1.0x vs 2.0x Floor) Due to Sub-Calibration Displacement at Collapse Onset (Step 7525), Directing Focus Toward Architecture-Level Functional-Role Isolation (Task B-C005REC-004Y, `primary_identifiability_decision: FUNCTION_SPACE_GATE_NOT_IDENTIFIABLE`)

- **Context:** Following ADR-0119 (hard CVOF freeze preserves $O_1$ compatibility but blocks learning) and ADR-0120 (parameter-space pre-transition trust-region radius fails to preserve compatibility), Task B-C005REC-004Y was executed as an audit-only diagnostic task with 0 new optimizer updates. Its goal was to test whether safe pre-transition CVOF updates (steps 7000->7500) and unsafe post-transition CVOF updates leading to oracle compatibility collapse (steps 7525, 8000) can be identifiably separated using non-oracle function-space metrics (M1: Final-logit RMS, M2: Output KL divergence, M3: Post-FFN normalized L2 displacement, M4: Attention output normalized L2 displacement, M5: Replay CE under fixed non-oracle reference attention). The audit operated on a newly generated and locked probe dataset `cvof_functional_gate_probe_v1` (1024 examples: 512 normal + 512 length-10, proven disjoint from all previous registries), using reference attention $A_{\text{ref7500}}(x)$ cached exclusively from I03@7500 normal forward execution, and evaluating a diagnostic graph fixing non-CVOF components (Query residual, Post-attn norm, Readout) to step 7500. Step 7525 state was reproduced via bit-exact historical replay.
- **Decision:** Classify the audit outcome as `primary_identifiability_decision: FUNCTION_SPACE_GATE_NOT_IDENTIFIABLE`. Set `selected_metric: null`.
- **Key Findings:**
  1. **Primary Identifiability Gate: FAIL (`FUNCTION_SPACE_GATE_NOT_IDENTIFIABLE`):**
     - Across all five candidate metrics (M1–M5), none achieved the required $R \ge 2.0\times$ separation ratio between the unsafe onset displacement ($d_{7525}$) and the safe calibration baseline ($d_{\text{safe}}$):
       - **M1 (Final-logit RMS):** $d_{\text{safe}} = 0.4117$, $d_{7525} = 0.3020 \implies R_{\text{onset}} = \mathbf{0.733\times}$ ($< 2.0\times$, FAIL); $d_{8000} = 0.8569 \implies R_{\text{late}} = 2.081\times$ (PASS).
       - **M2 (Output KL):** $d_{\text{safe}} = 0.0082$, $d_{7525} = 0.0045 \implies R_{\text{onset}} = \mathbf{0.543\times}$ ($< 2.0\times$, FAIL); $d_{8000} = 0.0284 \implies R_{\text{late}} = 3.457\times$ (PASS).
       - **M3 (Post-FFN L2):** $d_{\text{safe}} = 0.0834$, $d_{7525} = 0.0579 \implies R_{\text{onset}} = \mathbf{0.695\times}$ ($< 2.0\times$, FAIL); $d_{8000} = 0.1705 \implies R_{\text{late}} = 2.046\times$ (PASS).
       - **M4 (Attn-Out L2):** $d_{\text{safe}} = 0.0600$, $d_{7525} = 0.0391 \implies R_{\text{onset}} = \mathbf{0.651\times}$ ($< 2.0\times$, FAIL); $d_{8000} = 0.1395 \implies R_{\text{late}} = 2.326\times$ (PASS).
       - **M5 (Replay CE):** $d_{\text{safe}} = 0.0052$, $d_{7525} = 0.0038 \implies R_{\text{onset}} = \mathbf{0.739\times}$ ($< 2.0\times$, FAIL); $d_{8000} = 0.0023 \implies R_{\text{late}} = 0.455\times$ (FAIL).
     - Full probe passing list: `passing_metrics_full: []`.
  2. **Onset Invariance and Sub-Calibration Drift:**
     - At step 7525 (where ADR-0116 proved compatibility collapse has already irreversibly begun), the function-space displacement under non-oracle reference attention has *not yet magnified*; in fact, $R_{\text{onset}} < 1.0\times$ across every metric. The model undergoes genuine task-destructive representational warping without producing a function-space anomaly larger than normal pre-transition trajectory drift between steps 7000 and 7500.
     - While late-stage displacement at step 8000 does cross $2.0\times$ for M1–M4, any threshold capable of rejecting step 7525 updates would inevitably reject safe pre-transition updates (false positive rate $\ge 100\%$).
  3. **Split Consistency:**
     - The failure of all metrics to separate at step 7525 holds symmetrically across both split halves:
       - Normal half (lengths 2..10): $R_{\text{onset}}$ ranges from $0.505\times$ (M5) to $0.789\times$ (M1), all $< 1.0\times$.
       - Length-10 half: $R_{\text{onset}}$ ranges from $0.503\times$ (M2) to $0.935\times$ (M5), all $< 1.0\times$.
     - `verified_passing_metrics: []`.
  4. **Virtual Rollback Attribution (Diffuse Drift):**
     - Single-group virtual rollbacks from step 7525 (RB_C, RB_V, RB_O, RB_F) reveal that no single parameter group is solely responsible for onset displacement:
       - Baseline $d_{7525}$ (M1 RMS): $0.3020$.
       - RB_C: $0.2116$ (30.0% reduction).
       - RB_V: $0.2253$ (25.4% reduction).
       - RB_O: $0.2684$ (11.1% reduction).
       - RB_F: $0.2457$ (18.6% reduction).
       - RB_CVOF: $0.0000$ (exact positive control verification).
     - This confirms that onset drift is coupled and distributed across all four CVOF components simultaneously.
  5. **Auxiliary Full-Downstream Check:**
     - Auxiliary evaluation with unconstrained downstream components (readout, post-attn norm, query residual) similarly yielded $R_{\text{onset}} < 1.0\times$ ($M1: 0.725\times, M2: 0.540\times, M3: 0.695\times, M4: 0.651\times$).
  6. **Cost and Invariant Verification:**
     - `new_optimizer_updates = 0`, `parity_updates = 25` (from historical bit-exact replay to 7525), `wall_clock_seconds = 261.0s`.
     - Core and 15 other primitives invariant. `candidate_selected = null`, `child_bundle = null`, `rg3_recheck = NOT_EXECUTED`, `rec005_eligible = false`.
- **Consequences:**
  - **Function-Space Gate Disconfirmed:** Non-oracle function-space monitoring (replay CE, logit RMS, KL divergence, or intermediate representation shift) cannot serve as a reliable gate to detect or filter destructive CVOF updates at their onset.
  - **Rejection/Replay Gates Ruled Out:** Methods relying on rejection sampling, replay loss thresholds, or functional trust regions during adaptation are structurally unviable because destructive updates are indistinguishable from safe progress in function space.
  - **Directs Focus Toward Architecture-Level Functional Separation:** Resolving the tension between plasticity and stability requires decoupling the functional roles within the primitive architecture (e.g. separating coordinate/key routing from value/content transformation, or ensuring score-adaptation cannot backpropagate into shared representation channels).
  - **Strict STOP Boundary Enforced:** B-C005REC-004Y completed. No candidate training, child bundle creation, or evaluation on sealed splits.


## ADR-0122: I03 Key/Value Content-Prep Functional-Role Isolation Exact-Parity & Training Pilot — Separating Key/Value Content Preparation Verifies Exact Step-7500 Parity and Strict Gradient Isolation, Fully Preserving O1 Compatibility Across All Splits Under Unconstrained J0 Continuation (O1 EM = 0.996–1.000), but Key-Branch-Only Plasticity Remains Sub-Threshold for J0 Learning (Delta = +0.015 vs Historical, Delta = -0.026 vs Hard Freeze) (Task B-C005REC-004Z, `primary_decision: KEY_VALUE_ROLE_SPLIT_PRESERVES_STABILITY_BUT_KEY_PLASTICITY_INSUFFICIENT`)

- **Context:** Following ADR-0121 (which established that non-oracle function-space monitoring M1–M5 cannot identify CVOF compatibility collapse at onset, de-activating the proposed replay acceptance gate and directing focus to architecture-level functional-role isolation), Task B-C005REC-004Z evaluated the minimal architectural separation of the shared `CONTENT_PREP` component in primitive I03 into two distinct role-specific subcomponents: `KEY_CONTENT_PREP` ($f_{\text{key}}(\text{content}, \text{pos})$) and `VALUE_CONTENT_PREP` ($f_{\text{val}}(\text{content}, \text{pos})$). Both were cloned bit-exactly from legacy `CONTENT_PREP@7500` (weights and AdamW first/second moments). The task executed in three stages:
  1. **Stage A (Exact Parity Gate):** Verifying that at step 7500, forward predictions and probabilities match legacy I03 bit-identically ($\Delta_{\text{prob}} \le 10^{-5}$, $\Delta_{\text{logits}} \le 10^{-4}$, 0 discrete prediction mismatches).
  2. **Stage B (Gradient-Path Isolation Gate):** Verifying that normal J0 backpropagation produces non-zero gradients on `KEY_CONTENT_PREP`, while downstream oracle substitution ($O_1$) backpropagation produces strictly zero gradient on `KEY_CONTENT_PREP` and key in-projection ($\le 10^{-12}$) while flowing normally through `VALUE_CONTENT_PREP`, $V$, $O$, and $F$.
  3. **Stage C (Training Continuation Pilot):** Under the `KV_ROLE_SPLIT_PROTECTED_VALUE` regime, freezing `VALUE_CONTENT_PREP` along with $V$, $O$, and $FFN$ at their exact step-7500 values, while granting full plasticity to `KEY_CONTENT_PREP`, $Q/K$, position bias, and readout over 500 unconstrained normal forward J0 optimizer updates (steps 7501..8000).
- **Decision:** Classify the outcome as `primary_decision: KEY_VALUE_ROLE_SPLIT_PRESERVES_STABILITY_BUT_KEY_PLASTICITY_INSUFFICIENT`. Set `strong_functional_floor: false`.
- **Key Findings:**
  1. **Stage A Parity Gate: PASS:**
     - Maximum probability difference: $1.52 \times 10^{-6}$ (threshold $\le 1.0 \times 10^{-5}$).
     - Maximum logits difference: $3.10 \times 10^{-5}$ (threshold $\le 1.0 \times 10^{-4}$).
     - Discrete prediction mismatches across 512 probe examples: $0$.
  2. **Stage B Gradient-Path Isolation Gate: PASS:**
     - Normal J0 forward: `KEY_CONTENT_PREP` gradient norm $= 1.19 > 0$, `K_proj` gradient norm $= 0.45 > 0$.
     - Downstream oracle ($O_1$) forward: `KEY_CONTENT_PREP` gradient norm $= 0.00 \le 10^{-12}$, `K_proj` gradient norm $= 0.00 \le 10^{-12}$.
     - Downstream oracle ($O_1$) value path: `VALUE_CONTENT_PREP` gradient norm $= 4.11 > 0$, `V_proj` gradient norm $= 1.85 > 0$, `ATTN_OUT` gradient norm $= 1.42 > 0$, `FFN` gradient norm $= 3.55 > 0$.
     - Functional role separation is strictly isolated: key-routing adaptations cannot directly backpropagate into the value/content transmission channel.
  3. **Stage C Compatibility Gate: PASS:**
     - All 4 continuity probes and fresh length-10 confirmation exhibited near-perfect $O_1$ downstream compatibility at step 8000:
       - `length10_mechanism_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `dense_trajectory_transition_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `attention_clamp_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `downstream_freeze_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `fresh_length10_confirmation`: $O_1$ EM $= 0.996$, position-4 acc $= 1.000$.
     - Freeze audit verified $0.0$ drift across all frozen subcomponents (`value_content_in_proj`, `v_projection`, `attn_out_proj`, `ffn`).
  4. **Stage C Plasticity Gate: FAIL:**
     - Key-branch plasticity was confirmed active: `KEY_CONTENT_PREP` in-projection weight displacement $= 0.5316$, divergence from value in-projection $= 0.5316$, cumulative gradient norm $= 646.25$ (`KEY_BRANCH_PLASTICITY_ACTIVE`).
     - However, task learning gains remained severely sub-threshold:
       - Fresh normal validation (overall J0 EM, 1024 examples): Role-split $= 0.7734$ vs Historical unconstrained @8000 $= 0.7588$ ($\Delta = +0.0146$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.7998$ ($\Delta = -0.0264$ vs $+0.05$ floor).
       - Fresh length-10 confirmation (J0 EM, 512 examples): Role-split $= 0.0703$ vs Historical unconstrained @8000 $= 0.0547$ ($\Delta = +0.0156$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.0898$ ($\Delta = -0.0195$ vs $+0.05$ floor).
     - At length 10, the model remained unable to route attention reliably to the correct key (position 4 accuracy $= 0.0938$, score margin median $= -30.82$, correct key rank mean $= 8.25$).
  5. **Cost and Resource Accounting:**
     - Added resident parameters: $7,200$ (cloned key content prep). Active trainable parameters: $24,490$.
     - Optimization updates: $500$ intervention updates, $0$ new candidate training updates. Wall-clock training duration: $12.3$ seconds. Peak VRAM: $44.3$ MB.
     - Stable Core and 15 other primitives invariant. `candidate_selected: null`, `child_bundle: null`, `rg3_recheck: NOT_EXECUTED`, `rec005_eligible: false`.
- **Consequences:**
  - **Structural Result:** Separating `CONTENT_PREP` into independent Key and Value channels provides complete architectural protection against downstream $O_1$ compatibility collapse when value-side parameters are frozen.
  - **Plasticity Limitation:** Granting plasticity exclusively to the Key-content preparation and attention-routing mechanisms does not suffice to enable J0 task adaptation on challenging length-10 sequences; the model fails to overcome the attention-routing barrier without plastic downstream representation or plastic value-residual capacity.
  - **Directs Subsequent Repair:** Architecture exploration must move beyond purely routing-side (Key) plasticity. Next investigations must consider compact task-blind plastic residual capacity (e.g. residual adapter over frozen base, or downstream plastic bypass) rather than relying on coordinate/key transformations alone.
  - **Strict STOP Boundary Enforced:** B-C005REC-004Z completed. No candidate training, child bundle creation, candidate selection, or evaluation on sealed splits.


## ADR-0123: I03 Frozen-Base Compact Value-Residual Plasticity Pilot — Adding an Independent Low-Rank Task-Blind Value Residual over Frozen Step-7500 CVOF Base Preserves Downstream O1 Compatibility (EM = 0.998–1.000) but Fails to Recover Normal J0 Plasticity (Delta = -0.023 vs Historical, Delta = -0.045 vs Hard Freeze) (Task B-C005REC-004AA, `primary_decision: COMPACT_VALUE_RESIDUAL_PRESERVES_STABILITY_BUT_CAPACITY_INSUFFICIENT`)

- **Context:** Following ADR-0122 (which established that KEY/VALUE role-split with frozen CVOF base preserves downstream $O_1$ compatibility while KEY-branch plasticity alone is insufficient for J0 learning), Task B-C005REC-004AA evaluated whether introducing an independent, compact, task-blind low-rank value residual ($W_{\text{up}}(\text{GELU}(W_{\text{down}}(h)))$, $d=32, r=4$, 256 parameters, exact-zero initialized) immediately after `VALUE_CONTENT_PREP` and prior to $V$ projection could restore normal J0 learning without perturbing the step-7500 frozen base parameters (`VALUE_CONTENT_PREP`, $V$, $O$, and $FFN$).
  The task executed in three stages:
  1. **Stage A (Exact-Parity Gate):** Verifying bit-exact parity at step 7500 against the legacy I03 checkpoint ($\Delta_{\text{prob}} \le 10^{-5}$, $\Delta_{\text{logits}} \le 10^{-4}$, 0 discrete prediction mismatches).
  2. **Stage B (Gradient-Path Isolation Gate):** Verifying that normal J0 backpropagation produces active gradients on $W_{\text{up}}$, $W_{\text{down}}$, and `KEY_CONTENT_PREP`, while downstream CVOF base parameters remain strictly zero-gradient frozen.
  3. **Stage C (Training Continuation Pilot):** Training for 500 unconstrained normal forward J0 optimizer updates (steps 7501..8000) with active KEY branch, position bias, readout, and the compact value residual, while CVOF base parameters remain strictly frozen.
- **Decision:** Classify the outcome as `primary_decision: COMPACT_VALUE_RESIDUAL_PRESERVES_STABILITY_BUT_CAPACITY_INSUFFICIENT`. Set `strong_functional_floor: false`.
- **Key Findings:**
  1. **Stage A Parity Gate: PASS:**
     - Maximum probability difference: $1.79 \times 10^{-6}$ (threshold $\le 1.0 \times 10^{-5}$).
     - Maximum logits difference: $2.77 \times 10^{-5}$ (threshold $\le 1.0 \times 10^{-4}$).
     - Discrete prediction mismatches across 512 probe examples: $0$.
  2. **Stage B Gradient-Path Isolation Gate: PASS:**
     - Active gradients verified: $W_{\text{up}}$ gradient norm $= 1.61 \times 10^{-2} > 0$, `KEY_CONTENT_PREP` gradient norm $= 8.65 \times 10^{-1} > 0$.
     - Base CVOF parameters remained frozen with zero gradient.
  3. **Stage C Compatibility Gate: PASS:**
     - All 4 continuity splits and fresh length-10 confirmation exhibited near-perfect $O_1$ downstream compatibility at step 8000:
       - `length10_mechanism_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `dense_trajectory_transition_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `attention_clamp_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `downstream_freeze_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `fresh_length10_confirmation`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
     - Freeze audit verified $0.0$ drift across all frozen base CVOF subcomponents.
  4. **Stage C Plasticity Gate: FAIL:**
     - Fresh normal validation (overall J0 EM, 1024 examples): Residual pilot $= 0.7607$ vs Historical unconstrained @8000 $= 0.7832$ ($\Delta = -0.0225$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.8057$ ($\Delta = -0.0449$ vs $+0.05$ floor); vs Role-Split (REC-004Z) @8000 $= 0.7842$ ($\Delta = -0.0234$).
     - Fresh length-10 confirmation (J0 EM, 512 examples): Residual pilot $= 0.0781$ vs Historical unconstrained @8000 $= 0.0469$ ($\Delta = +0.0312$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.0840$ ($\Delta = -0.0059$ vs $+0.05$ floor); vs Role-Split (REC-004Z) @8000 $= 0.0801$ ($\Delta = -0.0020$).
     - Attention routing at length 10 remained impaired (position 4 accuracy $= 0.1074$, score margin median $= -33.28$, correct key rank mean $= 8.51$).
  5. **Residual Utilization Audit: ACTIVE:**
     - $W_{\text{down}}$ L2 displacement $= 0.4098$, $W_{\text{up}}$ L2 displacement $= 0.3985$.
     - Cumulative gradient norm $= 34.85$ ($W_{\text{up}}: 20.00, W_{\text{down}}: 14.85$).
     - Effective output rank $= 4.0 / 4$ (singular values: $[14.80, 5.39, 1.14, 0.55]$).
     - Mean residual-to-base ratio $= 0.0134$ (`VALUE_RESIDUAL_PLASTICITY_ACTIVE`).
  6. **Cost and Resource Accounting:**
     - Added residual parameters: $256$. Total added resident parameters vs legacy single primitive: $7,456$ ($7,200$ key split + $256$ residual). Active trainable parameters: $24,746$.
     - Optimization updates: $500$ intervention updates, $0$ candidate training updates. Wall-clock training duration: $14.7$ seconds. Peak VRAM: $48.0$ MB.
     - Stable Core and 15 other primitives invariant. `candidate_selected: null`, `child_bundle: null`, `rg3_recheck: NOT_EXECUTED`, `rec005_eligible: false`.
- **Consequences:**
  - **Stability Guaranteed:** Freezing the CVOF base at step 7500 completely isolates downstream oracle compatibility from degradation, even in the presence of an active plastic value-side residual.
  - **Plasticity Insufficient:** A compact rank-4 value residual ($r=4, d=32$) prior to $V$ projection is actively trained but structurally insufficient to restore normal J0 learning or overcome the length-10 attention routing bottleneck.
  - **Formal Outcome Classification:** Lands squarely in Section 20.B (`COMPACT_VALUE_RESIDUAL_PRESERVES_STABILITY_BUT_CAPACITY_INSUFFICIENT`).
  - **Directs Subsequent Repair:** Further exploration of plastic capacity cannot rely on pre-$V$ low-rank value residuals alone. Next repairs must consider downstream residual bypasses (e.g. post-attention / output-projection bypass), higher rank or multi-stage adaptation, while preserving the task-blind invariant.
  - **Strict STOP Boundary Enforced:** B-C005REC-004AA completed. No candidate training, child bundle creation, candidate selection, or evaluation on sealed splits.


## ADR-0124: I03 Post-Attention Compact Residual Bypass Location Pilot — Moving Rank-4 Plastic Residual to Post-Attention Output Projection Preserves Downstream O1 Compatibility (EM = 0.998–1.000) with Verified Score-Path Isolation, but Confirms Plasticity Insufficiency and Lacks Location Advantage Over Pre-V Residual (Task B-C005REC-004AB, `primary_decision: POST_ATTN_RESIDUAL_PRESERVES_STABILITY_BUT_PLASTICITY_INSUFFICIENT`, `location_advantage: false`)

- **Context:** Following ADR-0123 (which found that adding a compact rank-4 pre-V residual preserved downstream $O_1$ compatibility but failed to restore J0 plasticity, leaving length-10 attention routing uncorrected), Task B-C005REC-004AB investigated the residual-location hypothesis: holding capacity identical ($r=4, d=32$, 256 parameters, GELU, exact-zero functional initialization), the plastic residual bypass was relocated from pre-V to immediately after the frozen attention output projection ($h_{\text{attn}} = h_{\text{attn\_base}} + \text{POST\_ATTN\_RESIDUAL}(h_{\text{attn\_base}})$), prior to the existing residual addition and norm.
  The task tested whether downstream plastic capacity could functionally compensate for imperfect attention mixtures to recover task output without altering attention routing weights.
  Stages:
  1. **Stage A (Exact-Parity Gate):** Verifying bit-exact parity at step 7500 against legacy I03 and REC-004Z role-split.
  2. **Stage B (Functional-Path & Attention-Score Isolation Gate):** Verifying active forward gradient flow to $W_{\text{up}}$, K projection, and Key content prep, alongside live-graph verification that $d(\text{scores})/d(\theta_{\text{post\_attn}}) = 0$ (confirming the post-attention residual does not leak into attention score logits).
  3. **Stage C (Training Continuation Pilot):** 500 unconstrained normal forward J0 optimizer updates (7501..8000) with frozen CVOF base and active Key/Q/K/post-attention residual.
- **Decision:** Classify outcome as `primary_decision: POST_ATTN_RESIDUAL_PRESERVES_STABILITY_BUT_PLASTICITY_INSUFFICIENT`. Set `location_advantage: false` and `strong_functional_floor: false`.
- **Key Findings:**
  1. **Stage A Parity Gate: PASS:**
     - Maximum attention probability difference: $1.55 \times 10^{-6} \le 1.0 \times 10^{-5}$.
     - Maximum final logits difference: $2.88 \times 10^{-5} \le 1.0 \times 10^{-4}$.
     - Discrete prediction mismatches across 512 probe examples: $0$.
  2. **Stage B Functional-Path & Score Isolation Gate: PASS:**
     - Active gradients verified: $W_{\text{up}}$ gradient norm $= 1.49 \times 10^{-2} > 0$, K projection gradient norm $= 1.03 > 0$, `KEY_CONTENT_PREP` gradient norm $= 8.80 \times 10^{-1} > 0$.
     - Live-graph attention score gradient $d(\text{scores})/d(\text{params}) = 0.0$ and perturbation score logit difference $= 0.0 \le 1.0 \times 10^{-7}$, confirming zero score-path leakage.
  3. **Stage C Compatibility Gate: PASS:**
     - All continuity splits and fresh length-10 confirmation maintained ceiling $O_1$ downstream compatibility at step 8000:
       - `length10_mechanism_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `dense_trajectory_transition_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `attention_clamp_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `downstream_freeze_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `fresh_length10_confirmation`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
     - Freeze audit verified $0.0$ drift across frozen base CVOF subcomponents.
  4. **Stage C Plasticity Gate: FAIL:**
     - Fresh normal validation (overall J0 EM, 1024 examples): Post-attn pilot $= 0.6914$ vs Historical unconstrained @8000 $= 0.7422$ ($\Delta = -0.0508$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.7686$ ($\Delta = -0.0771$ vs $+0.05$ floor); vs Pre-V (REC-004AA) @8000 $= 0.7246$ ($\Delta = -0.0332$).
     - Fresh length-10 confirmation (J0 EM, 512 examples): Post-attn pilot $= 0.0703$ vs Historical unconstrained @8000 $= 0.0508$ ($\Delta = +0.0195$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.0879$ ($\Delta = -0.0176$ vs $+0.05$ floor); vs Pre-V (REC-004AA) @8000 $= 0.0723$ ($\Delta = -0.0020$).
  5. **Location-Attribution Metric: NOT SUPPORTED:**
     - $\Delta_{\text{location\_normal}} = 0.6914 - 0.7246 = -0.0332 < +0.05$.
     - $\Delta_{\text{location\_length10}} = 0.0703 - 0.0723 = -0.0020 < +0.05$.
     - Post-attention residual provides no location advantage over pre-V residual (`location_advantage: false`).
  6. **Routing-vs-Compensation Audit:**
     - Attention routing remained severely impaired (median correct-key score margin $= -32.65$, mean correct-key rank $= 8.51$, median correct-key probability $= 7.89 \times 10^{-8}$).
     - End-task sequence EM did not substantially improve (+0.0195 vs historical), giving label `NONE` (no compensation without routing).
  7. **Residual Utilization: ACTIVE:**
     - $W_{\text{down}}$ L2 displacement $= 0.6694$, $W_{\text{up}}$ L2 displacement $= 0.5170$.
     - Total gradient accumulation $= 17.61$ ($W_{\text{up}}: 10.56, W_{\text{down}}: 7.05$).
     - Effective rank $= 4.0 / 4$ (singular values: $[27.15, 2.30, 1.46, 0.53]$).
     - Mean residual-to-base ratio $= 0.0184$ (`POST_ATTN_RESIDUAL_PLASTICITY_ACTIVE`).
  8. **Cost and Resource Accounting:**
     - Added residual parameters: $256$. Total resident parameters added vs single primitive: $7,456$. Active trainable parameters: $24,746$.
     - 500 intervention updates, 0 candidate training updates. Training duration: $29.08$ seconds. Peak VRAM: $48.29$ MB.
     - Stable Core and 15 persistent primitives invariant. `candidate_selected: null`, `child_bundle: null`, `rg3_recheck: NOT_EXECUTED`, `rec005_eligible: false`.
- **Consequences:**
  - **Decisive Attribution on Residual Location:** Neither pre-V nor post-attention compact rank-4 residual bypass architectures are sufficient to recover normal J0 plasticity or compensate for broken attention routing while the CVOF base is frozen.
  - **Location Advantage Disconfirmed:** Relocating the residual to post-attention does not outperform pre-V residual ($\Delta = -0.033$ on normal, $-0.002$ on length-10).
  - **Cease Simple Residual-Location Search:** Per Section 26.3, simple residual-location exploration is terminated. Further investigations must directly address the score/routing capacity bottleneck or architecture re-design rather than downstream compensation.
  - **Strict STOP Boundary Enforced:** Task B-C005REC-004AB completed. No candidate training, child bundle creation, candidate selection, RG3, or REC-005.

## ADR-0125: I03 Parallel Low-Rank Score-Residual Routing Pilot — Adding Independent Low-Rank Score Residual (r=4, d=32) Preserves O1 Compatibility (EM = 0.998–1.000) and Structural Score Isolation, but Confirms Plasticity Insufficiency Due to Dominant Base Score Attractor (Task B-C005REC-004AC, `primary_decision: PARALLEL_SCORE_RESIDUAL_PRESERVES_STABILITY_BUT_ROUTING_INSUFFICIENT`, `score_capacity_advantage: false`)

- **Context:** Following ADR-0124 (which disconfirmed residual location advantage, demonstrating that neither pre-V nor post-attention compact residuals can compensate for broken attention routing), Task B-C005REC-004AC directly evaluated the score/routing-side architecture hypothesis: adding an independent, task-blind, low-rank score residual channel ($\Delta S = q_r k_r^T / \sqrt{r}$, $r=4, d=32$, 256 parameters, exact-zero initialized on $W_{kr}$) directly to pre-softmax cross-attention scores ($S_{\text{total}} = S_{\text{base}} + \Delta S$), while keeping the base CVOF parameters strictly frozen at step 7500.
  The task tested whether directly augmenting the pre-softmax score matrix with compact plastic capacity could overcome the length-10 attention routing bottleneck without destabilizing downstream stability or modifying downstream value/FFN pathways.
  Stages:
  1. **Stage A (Exact-Parity Gate):** Verifying bit-exact parity at step 7500 ($\Delta S = 0$).
  2. **Stage B1 (Score-Path Functional Gradient Flow Isolation Gate):** Verifying active gradient flow through $\Delta S$ to $W_{kr}$ and Key content prep.
  3. **Stage B2 ($O_1$ Structural Score Isolation Gate):** Verifying live-graph zero gradient and perturbation invariance under oracle $O_1$ execution.
  4. **Stage C (Training Continuation Pilot):** 500 unconstrained normal forward J0 optimizer updates (7501..8000) with frozen CVOF base and active Key/Q/K/score-residual.
- **Decision:** Classify outcome as `primary_decision: PARALLEL_SCORE_RESIDUAL_PRESERVES_STABILITY_BUT_ROUTING_INSUFFICIENT`. Set `score_capacity_advantage: false` and `strong_functional_floor: false`.
- **Key Findings:**
  1. **Stage A Exact-Parity Gate: PASS:**
     - Maximum $\Delta S = 0.00$.
     - Maximum attention probability difference: $1.43 \times 10^{-6} \le 1.0 \times 10^{-5}$.
     - Maximum final logits difference: $2.67 \times 10^{-5} \le 1.0 \times 10^{-4}$.
     - Prediction mismatches: $0$.
  2. **Stage B1 Score-Path Isolation Gate: PASS:**
     - Gradient norms: `KEY_CONTENT_PREP` $= 1.45$, K projection $= 1.30$, $W_{kr} = 0.179$, $W_{qr} = 0.0$ (exact zero init on $W_{kr}$).
  3. **Stage B2 $O_1$ Structural Score Isolation Gate: PASS:**
     - Live-graph $O_1$ gradient $\max |d(\text{logits}_{O_1})/d(\theta_{\Delta S})| = 0.0 \le 1.0 \times 10^{-7}$.
     - Perturbed logit difference: $0.0 \le 1.0 \times 10^{-7}$. Prediction mismatches: $0$.
  4. **Stage C Compatibility Gate: PASS:**
     - Ceiling $O_1$ downstream compatibility maintained across all evaluation splits:
       - `length10_mechanism_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `dense_trajectory_transition_probe_v1`: $O_1$ EM $= 0.998$, position-4 acc $= 1.000$.
       - `attention_clamp_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `downstream_freeze_causal_probe_v1`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
       - `fresh_length10_confirmation`: $O_1$ EM $= 1.000$, position-4 acc $= 1.000$.
     - Freeze audit verified $0.0$ drift across frozen base CVOF subcomponents.
  5. **Stage C Plasticity Gate: FAIL:**
     - Fresh normal validation (overall J0 EM, 1024 examples): Score residual $= 0.7646$ vs Historical unconstrained @8000 $= 0.7500$ ($\Delta = +0.0146$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.7861$ ($\Delta = -0.0215$ vs $+0.05$ floor).
     - Fresh length-10 confirmation (J0 EM, 512 examples): Score residual $= 0.0801$ vs Historical unconstrained @8000 $= 0.0605$ ($\Delta = +0.0195$ vs $+0.10$ floor); vs Hard Freeze (REC-004W) @8000 $= 0.0879$ ($\Delta = -0.0078$ vs $+0.05$ floor).
  6. **Matched-Capacity Location Comparison: NOT SUPPORTED:**
     - Best Downstream 256 (max of pre-V REC-004AA and post-attn REC-004AB): Normal J0 EM $= 0.7334$, Length-10 J0 EM $= 0.0742$.
     - $\Delta_{\text{location\_normal}} = 0.7646 - 0.7334 = +0.0312 < +0.05$.
     - $\Delta_{\text{location\_length10}} = 0.0801 - 0.0742 = +0.0059 < +0.05$.
     - Score residual architecture fails to achieve the $+0.05$ capacity advantage floor (`score_capacity_advantage: false`).
  7. **Score Decomposition & Residual Utilization Audit:**
     - Plasticity active: $W_{kr}$ displacement $= 0.0278$, $W_{qr}$ displacement $= 0.1340$, cumulative gradient accumulation $= 37.85$, effective rank $= 3.67 / 4.0$ (`SCORE_RESIDUAL_PLASTICITY_ACTIVE`).
     - Base score margin median: $-18.36$; Total score margin median: $-18.36$; Residual contribution median: $+0.0021$.
     - $\Delta S$ RMS mean: $0.0034$; Max abs mean: $0.0194$; Norm ratio $\|\Delta S\| / \|S_{\text{base}}\| = 0.00011$ ($0.01\%$).
     - Correct key rank at pos 4 remained at mean $5.01$ (median $5.0$), correct key probability remained at median $0.0005$, top-1 recall $= 0.0$, top-3 recall $= 0.0$, top-5 recall $= 1.0$.
     - Root cause: The frozen base attention score attractor ($-18.4$ margin) dominates the unscaled low-rank linear perturbation by orders of magnitude, preventing meaningful routing re-direction within standard optimization dynamics.
  8. **Cost and Resource Accounting:**
     - Added resident parameters: $256$. Total model parameters: $7,456$. Active trainable parameters: $24,746$.
     - 500 intervention updates, 0 candidate updates. Training duration: $32.53$ seconds. Peak VRAM: $50.7$ MB.
     - Core and 15 primitives strictly invariant. `candidate_selected: null`, `child_bundle: null`, `rg3_recheck: NOT_EXECUTED`, `rec005_eligible: false`.
- **Consequences:**
  - **Tripartite Architecture Pilot Closure (AA, AB, AC):**
    1. Pre-V residual (AA): Stability PASS, Plasticity FAIL ($-0.045$ vs freeze).
    2. Post-Attn residual (AB): Stability PASS, Plasticity FAIL ($-0.077$ vs freeze, no location advantage).
    3. Parallel Score residual (AC): Stability PASS, Plasticity FAIL ($-0.021$ vs freeze, $+0.031$ vs downstream, $< +0.05$ floor).
  - **Scientific Implication:** Compact rank-4 ($256$-parameter) linear residual adaptations—whether applied pre-V, post-attention, or directly to attention scores—are structurally incapable of overcoming the strong negative attractor formed during early pre-7500 training while base CVOF weights are held frozen.
  - **Directs Next Research Stage:** Pure low-rank linear bypasses are exhausted. Next repair design must consider temperature/scale modulation of base scores, non-linear score routing gates, or structured multi-stage unfreezing schedules under task-blind constraints.
  - **Strict STOP Boundary Enforced:** Task B-C005REC-004AC completed. No candidate training, child bundle creation, candidate selection, RG3 recheck, or REC-005 transition.

## ADR-0126: Phase B Blocker Audit Finds REC-004AC Attention Metric Coordinate Error

**Date:** 2026-09-12

**Status:** Accepted (user-requested inspection of Phase B progress and unblock strategy;
documentation and read-only evidence audit, not execution of REC-004AD or another research task).

**Decision:** Prioritize correction and artifact-only reanalysis of REC-004AC's attention
metrics before selecting another repair intervention. Preserve its EM-based negative
result and all recovery/research blocks. Narrow ADR-0125's causal and representational
claims to the conditions actually measured; do not rewrite its historical entry.

**Evidence:** At source commit `6f54fdfe1639d44a466f05eab7061d813034749e`,
`mirror_parallel_score_residual_pilot.py:114-115` declares correct keys at output
positions 4/5 as 4/5. The actual MIRROR_HALVES operation on length 10 maps those
positions to input keys 0/9. The independent operation call on unique input tokens
and comparison with Z/AA/AB's correct constants confirm the mismatch. AC's
`evaluate_length10_metrics` uses the erroneous P4 key for score margin, probability,
rank, recall and base/total/residual margin decomposition. A synthetic ideal score
row with true margin +10 is reported as -10 under AC's coordinate. P5's erroneous
constant is currently unused outside its declaration.

J0/O1 sequence EM and token accuracy use target tokens, and O1 constructs attention
with the correct `mirror_halves_position_map`. Compatibility/plasticity/strong-floor
and location-advantage predicates use these EM/accuracy fields. Thus the coordinate
bug does not itself invalidate their measured failure: AC normal EM=0.7646484375,
length-10 J0 EM=0.080078125, O1 EM=1.0, plasticity gate=false. Corrected checkpoint
attention metrics are NOT_RECOMPUTED in this inspection.

Two further reporting issues were verified: thresholding the mean of head ranks
at 1.5/3.5/5.5 is not ordinary top-k recall; and counting `requires_grad` includes
parameters zeroed/restored by the freeze procedure. The saved AC primitive has
24,746 resident scalars, of which 13,568 correspond to frozen/restored CVOF masks,
leaving 11,178 update-eligible scalars. These are primitive-only counts, excluding
Core and the other 15 primitives. Legacy-relative added capacity 7,456 is not the
model total. The raw score norm ratio also compares a headless residual with a
four-head base; functional comparison should use matched shapes/masks and row
centering. Q/K, Key prep and position bias train, so `S_base` itself is not frozen.

**Consequences:** The wrong-key score margin cannot establish a base-score attractor
or a repair priority. A single I03 rank-4, 500-update negative pilot does not prove
low-rank adaptations structurally incapable in general. Earlier evidence for
downstream compatibility preservation and J0 insufficiency remains useful. Proposed
follow-up: correct the observer and accounting, reproduce stored inputs/checkpoints
without optimizer updates, then preregister one supported repair hypothesis. Any
new recipe must explicitly connect pilot validation to all-init validation,
candidate adoption, full-bundle RG3 and the separate REC-005 cohort; diagnostic
completion alone is not admission to those steps. No threshold or selection rule
is relaxed, no passing initialization is cherry-picked, and G1 relation sufficiency
and G4 integration remain independent prerequisites for the eventual sealed G5.

**Artifacts and validation:**
`runs/phase_b_blocker_audit/20260912/run_001/{audit.py,audit.json}` records 28 historical
summary files plus selected source/decision/checkpoint hashes, the operation and
recall counterexamples, and tensor-count audit. Executed with Python 3.12.13;
assertions pass and hashed historical files remain unchanged. Zero model forward
calls, optimizer updates, candidate selections, bundle writes or sealed-data reads.
Documentation links and diff checked; full pytest/ruff/mypy not run because no
product implementation changed.

**Affected documents:**
[audit and proposed unblock sequence](research/PHASE_B_BLOCKER_AUDIT_2026_09_12.md),
[active recovery plan](exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md), and
[decision index](DECISIONS.md). RG3/REC-005 onward, R3-011/012, B-C006 and Task
Inference remain blocked. This ADR authorizes no follow-on experiment.


## ADR-0127: Consolidate Phase B Planning and Resume Under Explicit User Instruction

**Date:** 2026-09-12

**Status:** Accepted; planning reorganization and REC-004AC metric_v2 correction authorized by the user's instruction to reorganize Phase B and then resume it.

**Decision:** Use [PHASE_B_RESTART.md](exec-plans/active/PHASE_B_RESTART.md) as the single current status/dependency ledger. Existing Phase B, R3, and REC plans remain technical and historical contracts. Completed REC-004 lettered diagnostics are evidence, not an automatic execution queue. The user's continuation instruction supersedes historical per-task reauthorization wording for work explicitly specified in the restart plan, while all scientific STOP GATEs, sealed boundaries, thresholds, and source-preservation rules remain in force.

**Immediate work:** Correct REC-004AC attention-coordinate/recall/norm/accounting metrics and reanalyze existing endpoints on manifest-identical data without optimization or candidate selection. Only after that evidence is available may a bounded intervention be preregistered in the same plan. Its scientific failure stops dependent work; this decision does not promise a passing recovery or permit unlimited search.

**Consequences:** Separate MIRROR/bundle recovery, integration quality, and relation-holdout sufficiency. Preserve previous outcomes and uncommitted audit work. Add navigation pointers to parent plans/task packs and update AGENTS only for the canonical document entry point. No historical task IDs are renumbered, no old runs are overwritten, and no research gate is relaxed.

## ADR-0128: Correct AC Measurements and Pin Parent Vocabulary During Runtime Reconstruction

**Date:** 2026-09-12

**Status:** Accepted. Measurement reanalysis and focused regressions PASS; final repository
verification is recorded in the [restart plan](exec-plans/active/PHASE_B_RESTART.md).

**Decision:** Correct REC-004AC's evaluation-only coordinates, head-level recall,
matched/centered score norm ratio, and parameter accounting. Preserve the model forward,
stored weights, original development splits and EM-based negative result. Reconstruct
the content runtime with the vocabulary state already verified by the strict bundle
loader, rather than sizing its vocabulary from the current global operation registry.

**Measurement evidence:**
`runs/phase_b_restart/rec004ac_metric_v2/run_002/` reproduces the complete original AC
data manifest and evaluates AC@8000, historical/W/Z/AA/AB comparators and the specified
AC@7500 migration. All comparator EMs match their historical JSONs. Old/new AC
predictions are identical. The reanalysis took 379.19 seconds with Python 3.12.13,
PyTorch 2.13.0+cu130 and the RTX 5060 Ti. No optimizer was constructed; source hashes,
Core/bank weights and target weights remain unchanged. The failed `run_001` (a
comparator evaluator method-name error) remains preserved.

On the 512-example length-10 confirmation set, corrected P4 key 0 gives margin
median −30.8355989456, correct-key probability median 1.0562095554e−7, mean head
rank 8.4541015625, and top-1/top-3/top-5 head recall all zero. The median residual
margin contribution is −0.0002880096; the centered, matched-head norm ratio is
0.0003876730. Normal EM remains 0.7646484375, length-10 J0 EM 0.080078125, and O1 EM
1.0. The target has 24,746 resident/execution parameters; the original CVOF recipe
freezes/restores 13,568 and leaves 11,178 update-eligible. Added capacity is 256
relative to Z, or 7,456 relative to the legacy primitive. These are target-only
counts; the read-only reanalysis updates no parameters.

The explicit initial forward-parity supplement has zero discrete mismatches,
zero residual, maximum probability difference 1.55e−6 and maximum logit difference
3.10e−5, within the original tolerances. It was executed after the main run and is
identified separately by `initial_parity.json` and `supplemental_validation.json`.
`report.md` is a later rendering of saved results, not a rerun or replacement of
the main run's source snapshot.

**Runtime/verification defects discovered:** Full test collection imports
`holdout_families`, registering five additional operations. The old constructor
therefore built a 49-token Core for a verified 44-token parent (18 recorded
operations versus 23 currently registered). This affected 17 artifact-dependent
checks while isolated execution passed. The strict loader now exposes its already
verified vocabulary state, and REC-004 reconstruction validates its integer fields,
offsets and Core row counts before explicitly injecting that schema. It does not
resize checkpoint tensors, relax loading, modify the registry or infer missing data.
A CPU regression demonstrates unchanged content states under registry growth and
rejects missing/inconsistent schemas. This is a content-runtime fix; it does not
certify the future learned router's operation-token mapping or complete REC-006.

The complete reanalysis was repeated with `holdout_families` imported after this
fix: `runs/phase_b_restart/rec004ac_metric_v2/run_003/` PASS in 359.38 seconds.
Its runtime check records 23 globally registered operations, 18 parent operations
and a 44-token Core. The complete original data manifest, every comparator EM,
initial migration and source/weight preservation checks pass again; the corrected
metrics above are unchanged. This is the current reanalysis result. An earlier
supplement initialization attempt passed a checkpoint container instead of its
`primitive_state_dict`; it performed no metric forward and remains recorded as
`runtime_schema_supplement_001/summary.json` with `FAIL_INITIALIZATION`.

Older Phase A recurrence/shadow fault-injection tests also silently retrained when
a 39-token historical Core was loaded under a newer vocabulary. They now serialize
small coherent local fixtures, exercise the actual invariant/promotion paths and
forbid training fallback. The historical experiments are unchanged. A directory
enumeration assertion now sorts both sides instead of assuming Windows ordering.
Pre-existing Z lint violations were formatted without changing its AST, including
string contents (`runs/phase_b_restart/format_equivalence.json`). The WSL SciPy
dependency was restored; Windows-manifest tests use qualified native Python 3.12.

Final verification covers all 2,514 collected cases: 977 native passes before an
interpreter access violation, 1,536 native passes in a resumed process, and the
interrupted pure-data test passing in WSL. The case-ID union was checked exactly;
no case was omitted. The monolithic native run itself did not pass, and its crash
cause remains unresolved. Full collection was retained on resume to preserve
registry-import effects. Ruff, mypy (162 files), local links and diff checks pass.
Logs and the qualified result are recorded under
`runs/phase_b_restart/verification_coverage_result.json` and restart plan §8.

**Interpretation and boundary:** The corrected residual has a negative contribution
at the failing key. Small norm alone is not proof that amplification repairs it,
nor that low-rank learning is impossible. The restart plan preregisters four fixed
global scale conditions as a no-training precheck. Candidate selection, child
bundles, all-init expansion, RG3 and REC-005 remain unexecuted. G1 relation
sufficiency and G4 integration are independent unresolved requirements.

## ADR-0129: Fixed Global Score Rescaling Fails MIRROR Endpoint Recovery

**Date:** 2026-09-12

**Status:** Completed negative precheck; `execution_status: PASS`, `research_gate: FAIL_STOP`.

**Contract:** The [restart plan §6](exec-plans/active/PHASE_B_RESTART.md#6-次の介入-mirror-score-scale-precheck事前登録)
fixed four conditions before execution: `S = alpha * S_base + beta * DeltaS`
on saved AC I03@8000, with all parameters frozen and no optimizer. It uses only
the seven already reproduced development datasets, including normal validation
1024 and length-10 confirmation 512. No new split, sealed evaluation, coefficient
search or training was permitted. The execution-order note was changed before
observing these results to allow this read-only precheck alongside the remaining
full repository suite, after the complete metric_v2 rerun and schema regressions
passed. Scientific thresholds and conditions were unchanged; final repository
verification remains required for closing the restart work.

**Results:**

| Condition | alpha | beta | Normal J0 EM | Length-10 J0 EM | Minimum O1 EM |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 1 | 0.7646484375 | 0.080078125 | 0.998046875 |
| Base temperature | 0.125 | 1 | 0.0 | 0.0 | 0.998046875 |
| Residual amplification | 1 | 1000 | 0.0419921875 | 0.0 | 0.998046875 |
| Combined | 0.125 | 1000 | 0.0 | 0.0 | 0.998046875 |

No condition meets the simultaneous 0.95 normal/length-10 execution floors.
Baseline discrete predictions and historical EMs agree; maximum baseline logit
difference is 3.7431717e−5, within 1e−4. O1 predictions are identical across
conditions. Core, the parent bank and the separate AC primitive retain their
state hashes; all input/source hashes are unchanged. The registered study used
zero optimizer updates, zero new parameters and zero selections, taking 2.3864
seconds with peak allocated VRAM 60,725,760 bytes (about 57.9 MiB). The target has
24,746 resident/execution scalars; the Core has 1,797,504 resident scalars and the
parent bank 280,080 resident scalars, with no parent-bank execution.

**Interpretation:** These three fixed rescalings degrade this endpoint. The
measured small residual norm does not justify treating its amplification as a
repair. This finite inference precheck does not rule out other coefficients,
training with a different scale, or a different score representation, and it
does not establish that rank-4 learning is impossible. Those possibilities are
unexecuted hypotheses, not successful recovery evidence.

**Stop and remaining work:** Close this fixed-global-scale repair attempt.
Do not append more coefficients or auto-launch 500-step training. A subsequent
repair contract would need to explain how it changes the failing key ordering
while preserving the working value path, and connect a bounded pilot to all-init
validation, the fixed I01 adoption rule and full-bundle RG3. No candidate,
child bundle, RG3 recheck or REC-005 was produced. G1 and G4 remain independent
unresolved requirements in the single restart plan.

**Artifacts:** `runs/phase_b_restart/mirror_score_scale_precheck/run_001/`
contains the preregistration/source hashes, data digests and counts, source
snapshot, system record, all condition/split metrics, summary, side-effect audit
and report. Historical AC artifacts and ADR-0125 were preserved.
The separately labelled `post_run_source/manifest.json` records the final source
snapshot and working-tree patch after report/guard additions. It does not replace
the original preregistration hashes or claim those later additions were executed
in the recorded precheck.
