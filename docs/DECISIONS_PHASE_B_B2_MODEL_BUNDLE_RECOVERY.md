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

## ADR-0130: REC-004AE Localizes Length-10 Errors to Routing but Cannot Isolate a Single QK or Position-Bias Repair Target

**Date:** 2026-09-12

**Status:** Completed diagnostic; `execution_status: PASS`,
`decision: INSUFFICIENT_EVIDENCE_STOP`.

**Contract and boundary:** REC-004AE used only the immutable AC I03@8000 checkpoint and
the seven manifest-identical development datasets reproduced by metric_v2 `run_003`.
It froze the Core, parent bank and target primitive, fail-closed optimizer construction,
and performed zero updates, parameter additions, selections, candidate/bundle writes,
RG3 checks or sealed-data reads. Oracle attention was evaluation-only; it was not
provided to the J0 runtime or proposed recipe. The two preregistered score-component
removals (remove all position bias; remove all QK) are ablation-only diagnostics, not
coefficient search or repair candidates.

**Evidence:** `runs/phase_b_restart/rec004ae/run_004/` reproduces baseline sequence EM
exactly and records unchanged Core/bank/primitive hashes. On the length-10 confirmation,
baseline J0 EM is 0.080078125, oracle EM is 1.0, oracle-persistent token error is 0.0,
and 100% of direct token errors recover under oracle attention. The normal validation's
length-10 stratum likewise has oracle-persistent token error 0.0 and direct-error
oracle recovery 1.0. Across the two strata, output-position QK and position-bias
correct-key margin-contribution signs agree. Thus the retained error is localized to
the score-routing/value-selection boundary, rather than the fixed downstream value,
FFN or readout path.

The remaining required distinction did not hold. Removing either position bias or QK
produced sequence EM 0.0 on both normal validation and the length-10 confirmation;
the position-bias-minus-QK ablation delta is exactly 0.0 in both. Both components are
therefore necessary to the saved behavior under this destructive removal test, but the
test cannot identify either as the unique repair target. The original `run_001` remains
preserved as an initialization failure caused by a rank calculation on a sliced score
row; it constructed no optimizer and mutated no source. The corrected `run_004` is the
qualified result.

**Decision:** Do not select a repair intervention. This rules out treating the routing
localization itself as evidence for another global scale trial, score-only continuation,
gradient repair, compact residual location change, hard CVOF freeze, trust-region retry,
or a duplicate additive position-bias trial; these have already been rejected or stopped
by ADR-0111--0129. The earlier potential role-separated position-only routing recipe is
not selected because this result cannot show it is the unique causal remedy rather than
another QK/position interaction.

**Next-task start condition:** A new explicit contract may begin only with a
non-degenerate, fixed-endpoint causal comparison that separates QK-content competition
from position-routing insufficiency while holding the saved value/readout path fixed and
keeping the correct position map out of runtime inputs. It must state non-overlap with
the rejected hypotheses, specify a single subsequent recipe rather than an architecture
search, and preregister how any pilot result would connect to all-init validation, the
fixed I01 adoption rule and full-bundle RG3. Until then, no training, coefficient
expansion, candidate selection, RG3, REC-005 or sealed evaluation may start. G1 and G4
remain separate blocks.

**Verification:** Python 3.12.13 focused REC-004AC stage tests pass (4 tests), as do
`ruff check .`, `mypy src/apc` (162 source files), and `git diff --check`. A full-suite
attempt under the shell default Python 3.10 stopped at collection because `tomllib` is
unavailable; the project Python 3.12.13 attempt reported test errors before
completion and was interrupted after progress ceased. It is not claimed as passing.

## ADR-0131: REC-004AF Matched-Endpoint Counterfactuals Show QK-Sensitive Token Recovery but Do Not Support a Unique QK or Position Repair Target

**Date:** 2026-09-12

**Status:** Completed diagnostic; `execution_status: PASS`,
`decision: INSUFFICIENT_EVIDENCE_STOP`.

**Contract and boundary:** REC-004AF used only the immutable AC I03@8000 checkpoint,
metric_v2 `run_003` development inputs, and the qualified REC-004AE records. It froze
Core, parent bank, target primitive, value path, FFN and readout; fail-closed optimizer
construction verified zero updates. There were no parameter additions, candidate or
bundle writes, architecture/coefficient search, RG3/REC-005 activity, or sealed-data
reads. The correct position map was used only by post-forward metrics. Oracle attention
was evaluated only to reconfirm downstream sufficiency, never supplied to J0 or matching.

**Preregistered causal comparison:** Before any model forward, each saved endpoint was
matched within its split by relation `MIRROR_HALVES`, sequence length, output position,
and raw input-token-sum modulo 4. The pool was every saved endpoint in that stratum--not
only successes--sorted by SHA-256 of split, example index and input tuple. The next four
distinct endpoints in the circular order were the fixed controls. This rule does not take
targets, correct keys, oracle routing, baseline prediction, or intervention result as an
input. QK counterfactuals replaced only saved `S_QK`; position counterfactuals replaced
only saved `S_position_bias`; `S_residual` and the target endpoint's fixed value/readout
path stayed unchanged. Thus the measured score partition was
`S = S_QK + S_position_bias + S_residual`.

**Evidence:** Qualified artifacts are in `runs/phase_b_restart/rec004af/run_005/`.
Baseline sequence EM reproduced exactly: normal validation 0.7646484375 and length-10
confirmation 0.080078125. State hashes before/after exactly agree with REC-004AE's Core,
bank and primitive hashes; source hashes and all source datasets remained unchanged.
Normal-validation length-10 and confirmation oracle controls each have direct-error
recovery 1.0 and persistent error 0.0.

Across all four controls, QK-only replacement recovered direct-error tokens consistently:
0.135965--0.149123 on normal-validation length-10 and 0.128295--0.138840 on confirmation.
It raised sequence EM from 0.081340 to 0.105263--0.124402 and from 0.080078 to
0.095703--0.101562, respectively. However, correct-key top-1 routing changes span only
-0.001555 to +0.000488 and correct-key margin changes only -0.002519 to +0.003447.
They fail the preregistered +0.05 and +0.25 all-control/all-stratum minima. Replacing the
position bias produces exactly baseline output and routing measurements because that saved
position component is invariant across matched examples at fixed length and output position.

**Decision:** The experiment causally separates a limited QK-sensitive value-selection
effect: changing QK endpoints can recover some direct-error tokens while preserving the
entire downstream path. It does **not** establish QK-content competition as the dominant
routing repair target, because correct-key routing/margin lack a non-degenerate,
control-robust gain. Nor does it refute or support position-routing insufficiency: the
matched position controls have no example-level variation, so their zero effect is not an
informative position intervention. No component satisfies the fixed selection rule;
`QK_CONTENT_TARGET_SUPPORTED` and `POSITION_ROUTING_TARGET_SUPPORTED` are both rejected.
No repair recipe is selected or documented, and no learning pilot may follow from this run.

The non-qualified `run_001`--`run_004` artifacts preserve fail-closed baseline-parity
stops caused by diagnostic aggregation precision checks; their side-effect audits show
unchanged sources and zero optimizer construction. They are not scientific comparison
results. G1 and G4 remain uncleared and independent of this diagnosis.

## ADR-0132: REC-004AG Non-Degenerate Position-Routing Transport Disconfirms Position Bias as Unique Score-Routing Repair Target

**Date:** 2026-09-12

**Status:** Completed diagnostic; `execution_status: PASS`,
`decision: POSITION_ROUTING_TARGET_NOT_SUPPORTED`.

**Contract and boundary:** REC-004AG used only the immutable AC I03@8000 checkpoint,
metric_v2 `run_003` development inputs, and qualified REC-004AE/AF artifacts. It froze
Core, parent bank, target primitive, value path, FFN, and readout; optimizer construction
was patched to fail-closed (`RuntimeError`). Zero parameter updates, parameter additions,
candidate or bundle writes, architecture/coefficient sweeps, RG3 checks, REC-005 actions,
or sealed-data reads occurred. Correct position map, oracle attention, and baseline
correctness were withheld from intervention construction and runtime inputs, being used
strictly post-forward for metric evaluation.

**Preregistered transport causal intervention:** Prior to model forward, the source
profile was deterministically preregistered to relation `MIRROR_HALVES`, sequence length 9
(minimal difference $|10 - 9| = 1$ from target length 10), and mapped via
architecture-native normalized coordinates $u(k, L) = k / (L - 1)$ (defined by
$\phi(i, j, n) = [i/d, j/d, (j-i)/d, n/\text{length\_ref}]$). Ties were broken to the
smaller index, producing deterministic mappings `[0, 1, 2, 3, 4, 4, 5, 6, 7, 8]` for both
output and input coordinates. `S_position_bias` was replaced with the transported profile
while holding saved $S_{QK}$, $S_{residual}$, value vectors, FFN, and readout bitwise
identical.

**Pre-intervention non-degeneracy audit:** Profile analysis verified that within-stratum
example-level variance at length 10 is strictly 0.0 (confirming REC-004AF's position
degeneration mechanism), while cross-length variance (0.0147) and cross-position variance
(16.6426) are non-zero. The transported position bias differed substantially from baseline
length-10 bias (Frobenius norm difference 9.500572, max absolute difference 2.258081),
establishing a non-degenerate intervention.

**Evidence:** Qualified artifacts are in `runs/phase_b_restart/rec004ag/run_001/`. Baseline
sequence EM reproduced exactly: normal validation length-10 0.081340 and length-10
confirmation 0.080078. All state and source hashes matched.
Under `POSITION_TRANSPORT_CF`:
- On `length10_confirmation`: Sequence EM dropped from 0.080078 to 0.000000; token
  accuracy dropped from 0.888867 to 0.588867. While a small fraction of baseline direct
  errors recovered (recovery rate 0.033392, 19 tokens), 1555 previously correct tokens
  worsened (worsening rate 0.341683). Top-1 correct-key routing rate degraded from
  0.447754 to 0.297119 ($\Delta = -0.150635$ vs $+0.05$ threshold), and correct-key margin
  deteriorated from -7.154024 to -7.538437 ($\Delta = -0.384413$ vs $+0.25$ threshold).
- On `normal_validation_length10`: Sequence EM dropped from 0.081340 to 0.000000; token
  accuracy dropped from 0.890909 to 0.586603; recovery rate was 0.039474 (9 tokens) while
  645 tokens worsened (worsening rate 0.346402). Top-1 correct-key routing degraded from
  0.448565 to 0.295574 ($\Delta = -0.152990$), and correct-key margin deteriorated from
  -7.158134 to -7.544068 ($\Delta = -0.385934$).

**Decision:** The non-degenerate position transport intervention decisively fails all
selection thresholds across both primary strata, severely worsening sequence EM, top-1
correct-key routing, and correct-key margin. Therefore, `POSITION_ROUTING_TARGET_SUPPORTED`
is rejected, and the result is `POSITION_ROUTING_TARGET_NOT_SUPPORTED`.
Per execution rules, QK repair is not resumed because REC-004AF already failed the QK
routing target selection thresholds.
Conclusion: Neither QK-content competition nor position-routing bias can be isolated as a
unique repair target under the current score-component decomposition.
No learning pilot, recipe selection, coefficient sweep, or architecture search may follow.
The next phase must be a structural/identifiability review of the score decomposition
itself. G1 and G4 remain uncleared and independent blocks.

## ADR-0133: REC-004AH Score-Decomposition Structural Identifiability Review Stops Single-Component Repair

**Date:** 2026-09-12

**Status:** Completed review; `execution_status: PASS`,
`decision: SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`.

**Contract and boundary:** REC-004AH is an analytical and artifact-auditing review of the
immutable AC I03@8000 scorer implementation and qualified REC-004AE/AF/AG artifacts.
Zero optimizer construction, zero parameter updates, zero parameter additions, zero candidate
or bundle writes, zero architecture or coefficient sweeps, zero RG3 checks, zero REC-005
cohort executions, and zero sealed-data accesses occurred. The review evaluated the scorer
computation graph and tested pre-registered candidate interventions against a binary decision
rule under four necessary identifiability conditions.

**Scorer computation graph analysis:** The runtime score decomposition
$S = S_{QK} + S_{\text{position\_bias}} + S_{\text{residual}}$ was mapped across all upstream
and downstream dependencies:
1. $S_{QK} = (q k^T) / \sqrt{d_k}$: Driven by $k_{in}$ which couples directly to content token
   representations from Core. Because the target relation `MIRROR_HALVES` requires a pure index
   permutation $\pi(i) = (L-1) - i$, content token variance is task-orthogonal and acts as
   routing interference.
2. $S_{\text{position\_bias}} = W_{out}\text{ReLU}(W_h \phi(i, j, n))$: Computed from continuous
   normalized coordinates $\phi(i, j, n) = [i/d, j/d, (j-i)/d, n/\text{length\_ref}]$.
   Within-stratum example-level variance is identically 0.0 at any fixed sequence length.
3. $S_{\text{residual}} = (q_r k_r^T) / \sqrt{r}$: Low-rank ($r=4$) parallel score residual
   sharing upstream $k_{in}$ and `query`, with a tiny trained norm ratio ($0.00039$) and negative
   correct-key margin contribution ($-0.00029$).
4. Downstream losslessness: REC-004AE proved that downstream pathways (attention output projection,
   LayerNorm, FFN, and readout) achieve 100% sequence EM and 0.0% downstream persistent error
   under oracle attention. The failure resides entirely within the score decomposition and
   competitive Softmax normalization.

**Component identifiability matrix evaluation:** Each component was evaluated against four
pre-registered conditions: (a) architecture-native, (b) target/oracle-independent, (c) preserves
other score components and downstream paths, and (d) tests a repair-relevant local perturbation
rather than an out-of-distribution destructive transport.
- $S_{QK}$: Satisfies (a), (b), (c). Fails (d) because matched-endpoint donor substitution
  produces near-zero routing improvement ($\Delta \text{top-1} \in [-0.001555, +0.000488]$,
  $\Delta \text{margin} \in [-0.002519, +0.003447]$, REC-004AF). Content variance cannot provide
  a directional routing signal for an index permutation task.
- $S_{\text{position\_bias}}$: Within-stratum substitution is degenerate ($\Delta = 0.0$).
  Cross-length architecture coordinate transport (length 9 to 10) satisfies (a), (b), (c), but
  fails (d) due to catastrophic out-of-distribution collapse (EM dropped to 0.00, top-1 routing
  dropped by $-0.150635$, margin dropped by $-0.384413$, REC-004AG). Discrete coordinate grid
  mismatch (target positions 4 and 5 colliding onto source position 4) destroys permutation routing.
- $S_{\text{residual}}$: Satisfies (a), (b), (c). Fails (d) because global scaling fails all execution
  floors (ADR-0129) and donor substitution yields sub-millivolt perturbations.
- Joint Softmax Coupling: Additive composition followed by competitive all-to-all Softmax
  normalization couples all key logits. No individual component can be isolated as a unique repair
  target under permitted target-independent interventions.

**Decision:** Exactly zero components satisfy all four conditions. Under the pre-registered
binary decision rule, `SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP` is declared.
No single-component repair intervention, learning pilot, coefficient sweep, or architecture search
may proceed under the current score decomposition. G1 and G4 remain uncleared and independent blocks.

## ADR-0134: REC-004AI MIRROR Routing Representational Contract Review Identifies Minimal Single Architecture Contract (CD-DPCA)

**Date:** 2026-09-12

**Status:** Completed review; `execution_status: PASS`,
`decision: MINIMAL_ROUTING_CONTRACT_IDENTIFIED`.

**Contract and boundary:** REC-004AI is an analytical, semantics-driven representational review
of the routing requirements for `MIRROR_HALVES` and qualified REC-004AE/AF/AG/AH artifacts.
It enforces zero optimizer construction, zero parameter updates, zero parameter additions,
zero checkpoint mutation, zero candidate creation, zero bundle write, zero architecture candidate
sweep, zero RG3 check, zero REC-005 actions, and zero sealed-data access. G1 and G4 remain
uncleared independent blocks.

**MIRROR_HALVES routing semantics reconstruction:**
From `MirrorHalvesOp` (`src/apc/environments/operations.py`) and `mirror_halves_position_map`,
routing semantics was verified across all legal lengths $L \in [2, 16]$:
$$\pi_L(i) = \begin{cases} \lfloor L/2 \rfloor - 1 - i & \text{if } 0 \le i < \lfloor L/2 \rfloor \\ L + \lfloor L/2 \rfloor - 1 - i & \text{if } \lfloor L/2 \rfloor \le i < L \end{cases}$$
Routing is strictly content-invariant, strictly bijective involution on $\{0, \dots, L-1\}$,
length-dependent with an integer floor step discontinuity at $\lfloor L/2 \rfloor$, and query-position
dependent.

**Representational property audit of current architecture:**
The current scorer $S = S_{QK} + S_{\text{position\_bias}} + S_{\text{residual}}$ fails:
1. Content Invariance: $S_{QK}$ couples to $k_{\text{in}} = W_k(h_{\text{content}} + p)$, injecting
   task-orthogonal content noise.
2. Discrete Positional Distinguishability: $S_{\text{position\_bias}}$ uses continuous normalized
   coordinates $\phi = [i/(L-1), j/(L-1), (j-i)/(L-1), L/L_{\text{ref}}]$ in a smooth MLP, unable to
   guarantee sharp discrete margins.
3. Length Awareness: Continuous coordinate scaling causes coordinate grid aliasing across lengths
   (e.g. REC-004AG length 9->10 collapse).
4. Low-rank residual and additive softmax coupling cannot compensate for these structural defects.

**Deductive identification of minimal architecture contract:**
Without architectural candidate exploration or hyperparameter sweeps, accumulating the necessary
conditions yields a single minimal architecture contract:
`ContentDecoupledDiscretePositionalCrossAttention` (`CD-DPCA`):
- Runtime inputs: $(h_{\text{content}}, \text{content\_lengths}, \text{output\_lengths}, \text{argument\_values}=\text{None})$.
  No runtime oracle, target tokens, or teacher maps.
- Integer positional representations: Discrete query position embeddings $E_{\text{query\_pos}}(i) \in \mathbb{R}^{d_{\text{op}}}$
  and key position embeddings $E_{\text{key\_pos}}(j) \in \mathbb{R}^{d_{\text{op}}}$.
- Length representation: Discrete length embeddings $E_{\text{length}}(L) \in \mathbb{R}^{d_{\text{op}}}$.
- Query representation: $q(i, L) = E_{\text{query\_pos}}(i) + E_{\text{length}}(L)$.
- Key representation: $k(j) = E_{\text{key\_pos}}(j)$.
- Score generation: Standard multihead dot-product $S_h(i, j; L) = (q_h(i, L) \cdot k_h(j)^T) / \sqrt{d_{\text{head}}}$
  with padding mask ($-\infty$ for $j \ge L$).
- Content separation: $h_{\text{content}}$ is excluded from the score path and fed solely to the
  value projection $V(j) = W_v h_{\text{content}}(j) + E_{\text{val\_pos}}(j)$.
- Downstream connection: Unchanged connection to existing LayerNorm, FFN, and Readout (confirmed
  100% loss-free under oracle attention in REC-004AE).

**Hardcoding boundary:**
Target-specific formulas (e.g. `mid - 1 - i`) and lookup tables are strictly prohibited. The contract
specifies a general-purpose permutation hypothesis class initialized with standard random weights,
taking only generic integer metadata $(i, j, L)$ as input.

**Cross-relation static compatibility:**
Static analysis confirms full compatibility with all 15 non-SHIFT operations: tensor shapes, shared
frozen Core, downstream value/readout, and bundle serialization contracts remain preserved.

**Decision and authorized next steps:**
Decision: `MINIMAL_ROUTING_CONTRACT_IDENTIFIED`.
Next task authorization is strictly limited to:
1. REC-004AJ: Implementation and untrained structural/unit validation (verifying discrete distinguishability,
   length awareness, content invariance, zero oracle leakage, and gradient reachability).
Zero training, parameter updates, candidate creation, or bundle modification occurred in REC-004AI.
RG3, REC-005, G1, and G4 remain blocked.

## ADR-0135: REC-004AJ Minimal Routing Contract Implementation & Untrained Structural Validation Verifies All Seven Architectural Invariants (CD-DPCA)

**Date:** 2026-09-12

**Status:** Completed implementation and untrained validation; `execution_status: PASS`,
`decision: MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED`.

**Contract and boundary:** REC-004AJ implements the single minimal CD-DPCA routing path defined in
ADR-0134 and executes untrained structural/unit validation. It enforces zero optimizer construction,
zero parameter updates, zero checkpoint mutation, zero candidate creation or adoption, zero bundle writes,
zero architecture/coefficient sweeps, zero RG3 checks, zero REC-005 actions, and zero sealed-data access.
G1 and G4 remain uncleared independent blocks.

**Implementation details:**
- Added `ContentDecoupledDiscretePositionalCrossAttentionPrimitive` (`CD-DPCA`) and
  `ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig` (with short aliases `CDDPCAPrimitive`
  and `CDDPCAPrimitiveConfig`) in `src/apc/primitives/primitive.py`.
- Registered `new_content_decoupled_primitive` (and `new_cd_dpca_primitive`) in `PrimitiveBank`
  (`src/apc/primitives/bank.py`).
- Score path decoupling:
  - Query representation: $q(i, L) = E_{\text{query\_pos}}(i) + E_{\text{length}}(L) + \text{arg\_token}$,
    where $E_{\text{query\_pos}} \in \mathbb{R}^{L_{\text{max}} \times d_{\text{op}}}$ and
    $E_{\text{length}} \in \mathbb{R}^{(L_{\text{max}}+1) \times d_{\text{op}}}$.
  - Key representation: $k(j) = E_{\text{key\_pos}}(j)$, where
    $E_{\text{key\_pos}} \in \mathbb{R}^{L_{\text{max}} \times d_{\text{op}}}$.
  - Scoring: Multihead dot-product $S_h(i, j; L) = (q_h(i, L) \cdot k_h(j)^T) / \sqrt{d_{\text{head}}}$,
    with padded positions ($j \ge L$) masked to $-\infty$.
  - Value path: Content features $h_{\text{content}}$ flow solely to the value projection
    $v(j) = W_v h_{\text{content}}(j) + E_{\text{val\_pos}}(j)$, preserving existing downstream connections.
- Generic relation-conditioning boundary:
  - Parameterized operations generically project `argument_values` via `arg_encoder` and `arg_proj`
    into `arg_token`. Parameter-free operations (`MIRROR_HALVES`) pass `argument_values=None`.
  - Zero relation-specific branches, closed-form permutation formulas, or teacher maps exist.

**Seven pre-registered structural validations:**
1. Scorer Content Invariance: Evaluated across vastly different content tensors ($h^{(1)}$ vs $h^{(2)}$).
   Routing scores and attention weights are bitwise/numerically identical ($\max |\Delta S| = 0.0$,
   $\max |\Delta A| = 0.0$), while downstream logits differ ($\max |\Delta y| = 3.577 > 0.1$).
2. Discrete Addressability: All legal Phase B sequence lengths $L \in [2, 16]$ are addressable;
   pairwise distances between distinct embedding rows are $> 0.1$; out-of-bounds lengths ($L > 32$)
   strictly raise `ValueError`.
3. Permutation Representability: Constructive demonstration confirms that across all legal MIRROR_HALVES
   lengths $L \in [2, 16]$, the model can represent full permutation score matrices with correct key
   probability $> 0.99$ and margin over runner-up key $> 6.0$.
4. Information Boundary: Verified through AST/reflection audit that zero target tokens, oracle attention,
   or label maps reach runtime routing.
5. Padding Masking: Padded positions ($j \ge L$) have routing scores $-\infty$ and attention weights
   identically $0.0$; attention weights over valid positions ($j < L$) sum to $1.0$.
6. Gradient Reachability: Untrained forward pass with differentiable scalar loss yields non-null
   gradients for all routing parameters ($E_{\text{query\_pos}}$, $E_{\text{key\_pos}}$, $E_{\text{length}}$,
   and $W_q, W_k, W_v, W_o$). No parameter updates occurred.
7. Downstream Compatibility: Verified tensor shapes across multiple batch sizes and lengths,
   `PrimitiveBank` registration and bookkeeping, and standard `state_dict` serializability.

**Finite representability boundary:**
- Configured domain: $i, j \in [0, 31]$, $L \in [1, 32]$.
- Full-rank permutation capacity: $L \le \min(L_{\text{max}}, d_{\text{operator}}) = 32 \ge 16$.
- Unbounded extrapolation is explicitly REJECTED: discrete distinguishability trades continuous
  interpolation (the cause of REC-004AG coordinate aliasing) for finite, alias-free, orthogonal precision.

**Decision and authorized next steps:**
Decision: `MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED`.
All 7 criteria PASS. Next task authorization is strictly limited to:
1. REC-004AK: Serialization and Fresh-Load Validation under the immutable bundle loader contract.
Zero training, parameter updates, candidate creation, or bundle modification occurred in REC-004AJ.
RG3, REC-005, G1, and G4 remain blocked.

## ADR-0136: REC-004AK CD-DPCA Serialization & Fresh-Load Validation Confirms Exact Structural and Behavioral Equivalence under Strict Loader Contract

**Date:** 2026-09-12

**Status:** Completed serialization and strict fresh-load validation; `execution_status: PASS`,
`decision: CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED`.

**Contract and boundary:** REC-004AK validates the persistence and fresh-load reconstruction of the
`ContentDecoupledDiscretePositionalCrossAttentionPrimitive` (`CD-DPCA`) and `PrimitiveBank` under
the immutable model bundle loader contract (`strict=True`). It enforces zero optimizer construction,
zero parameter updates, zero training, zero checkpoint mutation, zero candidate adoption, zero ModelBundle
publish/writes (`bundle_write = False`), zero architecture/coefficient sweeps, zero RG3 checks,
zero REC-005 actions, and zero sealed-data access. G1 and G4 remain uncleared independent research blocks.

**Implementation additions:**
- Added `CD_DPCA_ARCHITECTURE_SIGNATURE = "content_decoupled_discrete_positional_cross_attention_v1"`
  in `src/apc/primitives/primitive.py`.
- Implemented `to_dict()` and strict-validating `from_dict()` for `PrimitiveConfig`,
  `CrossPositionPrimitiveConfig`, `CrossPositionLengthBiasPrimitiveConfig`, `CDDPCAPrimitiveConfig`,
  `ShiftRelativePrimitiveConfig`, and `ReverseRelativePrimitiveConfig`.
- Added `to_config_dict()` on `PrimitiveBase` and `ARCHITECTURE_SIGNATURE` + `from_config_dict()`
  on all primitive classes.
- Added `PRIMITIVE_TYPE_REGISTRY` and `build_primitive_from_config_dict()` factory in `src/apc/primitives/primitive.py`.
- Added manifest-based persistence on `PrimitiveBank` (`to_manifest`, `from_manifest`, `save_manifest`,
  `load_manifest`, `save_artifacts`, `from_artifacts(strict=True)`).

**Validation results (`runs/phase_b_restart/rec004ak/run_001/`, `tests/test_rec004ak_cd_dpca_serialization.py`):**
1. Exact Parameter Equivalence:
   - 20 parameter tensors matching bit-exact across name, shape, and `torch.float32` dtype.
   - Total parameter count: 19,178 params.
   - Canonical state hash: exact match (`canonical_state_hashes_match: True`).
   - State ABI hash: exact match under `content_decoupled_discrete_positional_cross_attention_v1`.
2. Behavioral Equivalence on Fixed Inputs:
   - Routing scores: bitwise identical ($\max |\Delta S| = 0.0$).
   - Attention weights: $\max |\Delta A| = 0.0$.
   - Padding masking: invalid positions ($j \ge L$) have score $-\infty$ and weight $0.0$; valid positions sum to $1.0$.
   - Forward logits: $\max |\Delta y| = 0.0$ across all batch/length dimensions.
   - Call instrumentation: forward call counter, usage increments, and reset operation verified identical.
3. Ten Fail-Closed Negative Checks:
   - Missing state_dict parameter -> `RuntimeError("Missing key(s)")`
   - Unexpected state_dict parameter -> `RuntimeError("Unexpected key(s)")`
   - Incompatible tensor shape -> `RuntimeError("size mismatch")`
   - Missing required config field -> `KeyError`
   - Unexpected config field -> `ValueError`
   - Incompatible operator dimension divisibility (`d_operator % n_head != 0`) -> `ValueError`
   - Incompatible max sequence length (`<= 0`) -> `ValueError`
   - Unknown primitive type in manifest -> `ValueError`
   - Duplicate primitive ID in manifest -> `ValueError`
   - Architecture signature mismatch -> `ValueError`
4. Runtime Information Boundary Audit:
   - Verified via AST and signature reflection that routing representation and score computations
     receive exclusively integer sequence lengths and discrete coordinates.
   - Confirmed zero target tokens, oracle attention maps, or labels leak into restored routing.

**Exact compatibility boundaries recorded:**
- Input schema: `d_model=192, d_operator=32, n_head=4, d_head=8, d_operator_ff=64, vocab_size=10, max_sequence_length=32, arg_dim=16`.
- Parameter accounting: 19,178 total parameters.
- Length domain: $[1, 32]$ supported; Phase B evaluation domain $[2, 16]$; hard `ValueError` outside domain.

**Decision and authorized next steps:**
Decision: `CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED`.
All equivalence, negative, and boundary criteria PASS. Next task authorization is strictly limited to:
1. REC-004AL: Single-Init Training Pilot under CD-DPCA (evaluating whether unconditioned CD-DPCA can
   learn the MIRROR_HALVES permutation without oracle guidance).
Zero training, parameter updates, candidate creation, or bundle modification occurred in REC-004AK.
RG3, REC-005, G1, and G4 remain blocked.

## ADR-0137: REC-004AL CD-DPCA Single-Init Learning Pilot Reaches 79.4% Sequence EM, Failing Terminal Viability Floor (>=95%) and Triggering Fail-Closed Stop

**Date:** 2026-09-12

**Status:** Completed single-init learning pilot; `execution_status: PASS`,
`decision: PILOT_TERMINAL_VIABILITY_NOT_MET`. Multi-init validation (REC-004AM), candidate adoption,
and bundle promotion are BLOCKED. G1 and G4 remain uncleared independent research blocks.

**Contract and preregistered boundary:**
- Pilot objective: Evaluate whether a single fresh initialization (I01) of the Content-Decoupled
  Discrete Positional Cross-Attention primitive (`CD-DPCA`, physical id 12) can learn the `MIRROR_HALVES`
  permutation from an ordinary token-output loss under the inherited optimizer recipe (AdamW, lr=1e-3,
  cosine decay, 6,000 updates maximum, evaluation cadence every 500 steps) without oracle guidance.
- Strict isolation & freeze:
  - Parent bundle Core (`canonical_state_hash: b3a0d5c0774a36f56281bfeadffce83ce61a6818816c7cf69dcae4a5d3fec585`)
    and router are strictly frozen (`requires_grad = False`).
  - 15 non-MIRROR primitives in the parent bank are strictly frozen.
  - Source REC-004AK artifacts (`bank_manifest.json`, `primitive_config.json`, `bank_state.pt`,
    `primitive_state.pt`) verified against raw and canonical SHA-256 hashes prior to training.
  - Runtime routing computation is audited via AST and reflection: zero target tokens, teacher maps,
    labels, oracle attention, or relation-specific permutation tables reach the forward pass.
- Bounded scope & zero side effects:
  - Exactly one initialization (I01) evaluated; zero additional inits run.
  - Decision strictly fixed to null: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`.
  - Zero hyperparameter search, zero learning rate sweeps, zero budget extensions beyond 6,000 updates.
  - Zero sealed evaluation partition accessed.
  - RG3 recheck `NOT_EXECUTED`, `rec005_eligible: false`. Independent research blocks G1 and G4 strictly preserved (`NOT_CLEARED`).
- Terminal viability criterion:
  - Terminal criterion: sequence EM $\ge 0.95$ at decisive step 6,000 on the 1,024-example existing development validation set.
  - Fail-closed protocol: if decisive sequence EM $< 0.95$ or any audit fails, record a specific STOP status (`PILOT_TERMINAL_VIABILITY_NOT_MET`) and do NOT proceed to all-init validation (REC-004AM).

**Empirical results (`runs/phase_b_restart/rec004al/run_001/`):**
1. Validation Metrics at Decisive Step 6,000:
   - Overall Sequence Exact Match: `0.793945` (813 / 1,024 examples).
   - Terminal Viability Threshold: `0.950000`.
   - Terminal Viability Met: `False` (`PILOT_TERMINAL_VIABILITY_NOT_MET` triggered).
   - Overall Token Accuracy: `0.973254` (8,029 / 8,250 tokens).
2. Per-Length Breakdown at Decisive Step 6,000:
   - Length 6: Sequence EM = `0.9755` (199 / 204), Token Acc = `0.9959` (1,195 / 1,200) — passes length floor ($\ge 0.95$).
   - Length 7: Sequence EM = `0.7944` (170 / 214), Token Acc = `0.9693` (1,660 / 1,712).
   - Length 8: Sequence EM = `0.9175` (178 / 194), Token Acc = `0.9897` (1,536 / 1,552).
   - Length 9: Sequence EM = `0.9515` (196 / 206), Token Acc = `0.9946` (1,844 / 1,854) — passes length floor ($\ge 0.95$).
   - Length 10: Sequence EM = `0.3398` (70 / 206), Token Acc = `0.9311` (1,794 / 1,927) — primary failure locus.
3. Causal Controls on Decisive Step 6,000 Checkpoint:
   - Correct Control EM: `0.7939` (813 / 1,024).
   - Wrong-Family Control EM (`REVERSE` donor): `0.0000` (0 / 1,024).
   - None Control EM (unrouted baseline): `0.0000` (0 / 1,024).
   - Causal Gap: `0.7939` (statistically non-spurious operator dependence).
4. Attention & Masking Diagnostics:
   - Padding Masking: strictly verified (`padding_mask_verified: True`); padded key positions ($j \ge L$) have score $-\infty$ and weight $0.0$; valid attention weights sum to $1.0$ ($\text{max diff} < 10^{-5}$).
   - Top-1 Routing Accuracy to Correct Key $\pi_L(i)$: `0.8964` (1,826 / 2,037 positions).
   - Mean Correct vs Runner-up Score Margin: `6.4475`.
5. Integrity & Boundary Audits:
   - Source Manifest SHA-256: verified bit-identical against REC-004AK.
   - Freeze Audit: `core_frozen_verified: True`, `parent_primitives_frozen: True`, optimizer updated zero Core or non-MIRROR parameters (`status: PASS`).
   - Side Effect Audit: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, `rg3: NOT_EXECUTED`, `rec005_eligible: false`, `g1: NOT_CLEARED`, `g4: NOT_CLEARED` (`status: PASS`).

**Scientific analysis:**
- Representation vs. Optimization Gap:
  ADR-0135 constructively proved that CD-DPCA possesses the exact representational capacity to achieve
  $100\%$ sequence EM across all lengths $L \in [2, 16]$ with high confidence and large margins ($>6.0$).
  However, empirical training from random initialization with the standard cross-entropy loss under
  the inherited schedule fails to converge to this representational attractor on the full distribution,
  achieving $79.4\%$ overall EM with a severe collapse specifically on Length 10 ($34.0\%$ EM),
  mirroring the known length-dependent difficulty observed in prior architectures (REC-004A, REC-004D).
- Fail-Closed Consequence:
  Because the terminal viability floor ($\ge 0.95$) was not reached, this pilot conclusively stops the
  CD-DPCA promotion pipeline without wasting resources on multi-initialization sweeps (REC-004AM)
  or corrupting the model bundle.

**Decision and authorized next steps:**
Decision: `PILOT_TERMINAL_VIABILITY_NOT_MET`.
1. All-init validation (REC-004AM) is BLOCKED.
2. Candidate adoption and model bundle creation remain strictly BLOCKED (`candidate_selected: null`, `child_bundle: null`).
3. RG3, REC-005, G1, and G4 remain uncleared and BLOCKED.
4. Artifacts from `runs/phase_b_restart/rec004al/run_001/` (all 13 checkpoints 0..6000, training states, logs, audits, diagnostics) are preserved immutably for post-mortem analysis.

## ADR-0138: REC-004AN CD-DPCA Length-10 Optimization Failure Localization Confirms Never-Learned Pattern and Isolates Failure Exclusively to Output Position 4 Lock-In to False Attractor (`LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`)

**Date:** 2026-09-13

**Status:** Completed diagnostic failure localization; `execution_status: PASS`,
`decision: LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`. Multi-init validation (REC-004AM), candidate adoption,
and bundle promotion remain BLOCKED. G1 and G4 remain uncleared independent research blocks.

**Contract and preregistered boundary:**
- Diagnostic objective: Localize the mechanism of the single-init (I01) CD-DPCA failure on `MIRROR_HALVES`
  observed in REC-004AL, distinguishing among representation deficiency, architecture deficiency,
  mere training budget deficit, length-specific parameter failure, shared-routing parameter interference,
  late-training regression / forgetting, and position-level localization.
- Strict evaluation-only execution boundary:
  - Zero optimizer updates (`optimizer.step()` forbidden, updates = 0).
  - Parent bundle Core (`canonical_state_hash: b3a0d5c0774a36f56281bfeadffce83ce61a6818816c7cf69dcae4a5d3fec585`)
    and 15 non-MIRROR primitives are strictly immutable.
  - Source REC-004AL artifacts (all 13 checkpoints 0..6000) verified bit-identical via raw SHA-256.
  - Zero candidate adoption, zero bundle write (`candidate_selected: null`, `child_bundle: null`).
  - Zero sealed evaluation partition accessed. RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`.

**Empirical results (`runs/phase_b_restart/rec004an/run_001/`):**
1. Full 13-Checkpoint Trajectory Re-Evaluation:
   - Trajectory Pattern: `NEVER_LEARNED_PATTERN`.
   - Length-10 sequence EM trajectory across steps 0..6000:
     - step 0: `0.0000`
     - step 500: `0.0631`
     - step 1000: `0.0825`
     - step 1500: `0.0728`
     - step 2000: `0.1019`
     - step 2500: `0.1068`
     - step 3000: `0.1408`
     - step 3500: `0.1214`
     - step 4000: `0.1845`
     - step 4500: `0.2136`
     - step 5000: `0.2524`
     - step 5500: `0.2961`
     - step 6000: `0.3398` (peak value across trajectory).
   - At no point during training did length 10 approach an acceptable performance regime (>=0.50, floor 0.95).
     Late-training regression (`LEARNED_THEN_REGRESSED_PATTERN`) is conclusively refuted.
2. Position-Level Localization:
   - Terminal Step 6000 per-position token accuracy for Length 10:
     - pos 0: `1.0000` (206 / 206)
     - pos 1: `1.0000` (206 / 206)
     - pos 2: `1.0000` (206 / 206)
     - pos 3: `0.9320` (192 / 206, 14 errors)
     - pos 4: `0.3835` (79 / 206, 127 errors) — primary failure locus
     - pos 5: `1.0000` (206 / 206)
     - pos 6: `1.0000` (206 / 206)
     - pos 7: `1.0000` (206 / 206)
     - pos 8: `1.0000` (206 / 206)
     - pos 9: `0.9951` (205 / 206, 1 error)
   - Extreme localization: Position 4 alone accounts for **89.44%** (127 / 142) of all terminal token errors.
     Positions 3 and 4 (the boundary positions of the first half) account for **99.30%** (141 / 142) of all errors.
   - Classification: Position 4 is the sole `consistently_failing_position`.
     Positions 0, 1, 5, 6, 7, 8, 9 are `consistently_correct_positions`.
3. Routing Lock-In to False Attractor:
   - For Length 10, the oracle mapping is $\pi_{10} = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5)$.
   - At step 6000, 9 of 10 positions select their correct oracle key as top-1.
   - Position 4 (oracle key 0) routes to key 7 with probability `0.4906` vs correct key 0 probability `0.0001`
     (score margin `-14.2144`).
   - Across the entire training trajectory (steps 500..6000), position 4 routed to key 7 in every single
     saved checkpoint without exception, proving that position 4 was trapped in a persistent false attractor
     from the initial 500 steps onward.
4. Causal State Transplantation Diagnostics:
   - Base model: step 6000 endpoint.
   - Intervention A ($E_{\text{length}}[10]$ transplant from step $t$): max L10 EM = `0.3495` (no recovery).
   - Intervention B (Shared routing state transplant from step $t$): max L10 EM = `0.3689` (no recovery).
   - Intervention C (Full routing module transplant sanity reference): max L10 EM = `0.3641` (no recovery).
   - Transplant interpretation: `NO_TRAJECTORY_LOCALIZATION`. Because length 10 was locked into the false
     attractor from step 500, no past checkpoint state contains a correct routing configuration for length 10.
5. Gradient Conflict Diagnosis:
   - Evaluated routing parameter loss gradients across lengths 6..10 without optimizer step:
     - Step 0 cosine similarity with aggregate lengths 6..9: `+0.4190` (positive alignment).
     - Step 6000 cosine similarity with aggregate lengths 6..9: `-0.0090` (essentially orthogonal, $|\cos| < 0.05$).
   - `SHARED_ROUTING_GRADIENT_INTERFERENCE_IDENTIFIED` is refuted: no severe antiparallel gradient conflict exists.
6. Training Data Exposure Audit:
   - Reconstructed exact 6,000 updates (192,000 examples):
     - Length 6: 38,497 (20.05%), 230,982 tokens, 99.88% step exposure.
     - Length 7: 38,416 (20.01%), 268,912 tokens, 99.92% step exposure.
     - Length 8: 38,645 (20.13%), 309,160 tokens, 99.93% step exposure.
     - Length 9: 38,188 (19.89%), 343,692 tokens, 99.92% step exposure.
     - Length 10: 38,254 (19.92%), 382,540 tokens, 99.93% step exposure.
   - Data scarcity is conclusively refuted.
7. Integrity Audits:
   - Optimizer updates: 0. Checkpoint hashes bit-identical to REC-004AL. Core and parent bank unmodified.
   - Candidate selected: null, child bundle: null, bundle write: false. RG3: NOT_EXECUTED. Sealed access: 0.

**Scientific conclusion:**
The CD-DPCA single-init optimization failure is conclusively identified as:
`LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`.
The failure is neither a representational ceiling (ADR-0135 proved $100\%$ representability), nor late regression,
nor data scarcity, nor shared-routing gradient conflict. Rather, it is a single-position local optimization
failure: under standard cross-entropy loss from random initialization, output position 4 fell into a persistent
false attractor (key 7 instead of 0) at step 500 and became numerically locked in (margin $-14.2$), accounting
for $89.4\%$ of all terminal errors while the remaining $90\%$ of output positions converged correctly.

**Decision and authorized next steps:**
Decision: `LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`.
1. Multi-init validation (REC-004AM) remains BLOCKED.
2. Candidate adoption and bundle promotion remain BLOCKED (`candidate_selected: null`, `child_bundle: null`).
3. RG3, REC-005, G1, and G4 remain uncleared and BLOCKED.
4. Next learning pilot authorization: NOT AUTHORIZED within REC-004AN.
   Any subsequent optimization repair must specifically target this single localized mechanism (e.g. boundary-routing
   optimization / local attractor escape) without multi-recipe exploration, curriculum search, or teacher-loss shortcuts.

## ADR-0139: REC-004AO CD-DPCA Position-4 False-Attractor Gradient Accessibility Diagnostic Identifies Loss-Gradient Misalignment and Key-0 Softmax Starvation (`LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`)

**Date:** 2026-09-13

**Status:** Completed evaluate-only diagnostic; `execution_status: PASS`,
`decision: LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`. Multi-init validation (REC-004AM), candidate adoption,
and bundle promotion remain BLOCKED. G1 and G4 remain uncleared independent research blocks.

**Contract and preregistered boundary:**
- Diagnostic objective: Determine whether standard token-output cross-entropy loss provides a corrective gradient signal
  to routing parameters to escape the length-10 position-4 false attractor (key 7 instead of 0) in CD-DPCA.
- Strict evaluation-only execution boundary:
  - Zero optimizer updates (`optimizer.step()` forbidden, updates = 0).
  - Parent bundle Core (`canonical_state_hash: b3a0d5c0774a36f56281bfeadffce83ce61a6818816c7cf69dcae4a5d3fec585`)
    and 15 non-MIRROR primitives are strictly immutable.
  - Source REC-004AL artifacts (all 13 checkpoints 0..6000) verified bit-identical via raw SHA-256.
  - Post-forward oracle position map used exclusively for diagnostic metrics/Jacobians, never in training loss.
  - Zero candidate adoption, zero bundle write (`candidate_selected: null`, `child_bundle: null`).
  - Zero sealed evaluation partition accessed. RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`.

**Empirical results (`runs/phase_b_restart/rec004ao/run_001/`):**
1. Position-4 Routing Trajectory Across All 13 Checkpoints:
   - Step 0: top-1 key 3, $p(0) = 0.0500$, $p(7) = 0.0951$, margin $-0.5964$, entropy $2.2740$, accuracy $0.0777$.
   - Step 500: rapid fall into false attractor key 7 ($p(7) = 0.4048$, $p(0) = 0.00184$, margin $-6.4724$, entropy $1.4966$).
   - Steps 1000..6000: persistent lock-in to key 7 across every checkpoint. Margin deepens monotonically to $-14.2144$,
     entropy falls to $1.1221$, terminal $p(0) = 1.15 \times 10^{-4}$, terminal accuracy $0.3835$.
2. Local Gradient Accessibility and Directional Derivative:
   - Gradient norm on CD-DPCA routing parameters is substantial throughout training ($\|g_{\text{loss, routing}}\| = 0.2707$
     at step 6000, peaking at $1.314$ at step 3500; $\|g_{Wk}\| = 0.2031$, $\|g_{Wq}\| = 0.1299$, $\|g_{k7}\| = 0.0463$).
   - Gradients do reach routing parameters from standard token-output cross-entropy loss.
   - However, first-order predicted margin change $\Delta M \propto - g_{\text{margin}} \cdot g_{\text{loss}}$ is
     non-positive across 11 of 13 checkpoints (84.6%), and strictly negative at terminal step 6000 ($-0.6976$,
     cosine alignment $-0.1116$).
3. Softmax Gradient Starvation on Key 0:
   - While shared routing parameters receive large loss gradients, the gradient specifically reaching key 0 embedding
     $E_{\text{key\_pos}}[0]$ drops from $1.37 \times 10^{-3}$ (step 0) to $7.87 \times 10^{-5}$ (step 3000) and
     $7.24 \times 10^{-4}$ (step 6000), starved by a factor of 64x to 1500x relative to key 7.
   - This is directly caused by softmax saturation: backpropagation scales $\nabla_{S(4, 0)} \mathcal{L} \propto p(0) \approx 10^{-4}$.
4. Per-Example Consistency:
   - Across all 206 length-10 validation examples at step 6000: 57.8% show negative margin change, median is $-1.2380$,
     mean is $-0.6976$.
   - Across 127 position-4 error examples at step 6000: median is $+0.0346$ (near zero), 49.6% negative, 50.4% positive.
     Starvation ratio $\|g_{k0}\| / \|g_{k7}\| = 0.0063$.
5. Control Comparisons:
   - Length 10, Position 3: accuracy 0.932, top-1 key 1 (correct), predicted margin change $+2.4685$ ($\cos = +0.0875$).
   - Length 10, Position 0: accuracy 1.000, top-1 key 4 (correct), predicted margin change $+0.6618$ ($\cos = +0.2949$).
   - Length 8, Position 4: accuracy 1.000, top-1 key 7 (correct), predicted margin change $+0.4894$ ($\cos = +0.4550$).
   - Length 9, Position 4: accuracy 1.000, top-1 key 8 (correct), accuracy 1.000.
   - Position 4 uniquely exhibits persistent false-attractor routing, extreme negative margin ($-14.21$), and negative
     gradient alignment ($-0.6976$).
6. Integrity Audits:
   - Optimizer updates: 0. Checkpoint hashes bit-identical to REC-004AL. Core and parent bank unmodified.
   - Candidate selected: null, child bundle: null, bundle write: false. RG3: NOT_EXECUTED. Sealed access: 0.

**Scientific conclusion:**
The failure mechanism is conclusively identified as:
`LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`.
The failure is governed by coupled mechanisms: early rapid onset of the false attractor at step 500 produces severe softmax
saturation on key 0 ($p(0) \to 10^{-4}$), which scales down backpropagation to $E_{\text{key\_pos}}[0]$ by orders of magnitude.
Meanwhile, substantial loss gradients flow to the active wrong key ($k_7$) and shared projections ($W_k, W_q$), but because
key 7 generates erroneous tokens, gradient descent on standard token loss produces a negative margin change
($-g_{\text{margin}} \cdot g_{\text{loss}} \le 0$ across 84.6% of checkpoints), reinforcing the false attractor.

**Decision and authorized next steps:**
Decision: `LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`.
1. Multi-init validation (REC-004AM) remains BLOCKED.
2. Candidate adoption and bundle promotion remain BLOCKED (`candidate_selected: null`, `child_bundle: null`).
3. RG3, REC-005, G1, and G4 remain uncleared and BLOCKED.
4. Next learning pilot authorization: NOT AUTHORIZED within REC-004AO.
   Because loss-gradient misalignment was identified rather than simple uncoupled starvation, simple anti-saturation
   heuristics alone are insufficient. An optimization formulation review within standard token-output cross-entropy loss
   boundaries must precede any candidate pilot. Oracle/teacher supervision remains strictly forbidden.

## ADR-0140: REC-004AP CD-DPCA Position-4 Token-Identifiability-Stratified Gradient Alignment Diagnostic Resolves Misalignment Paradox into Mid-Training Token-Aliasing Credit Dilution and Terminal Softmax Saturation (`TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`)

**Date:** 2026-09-13

**Status:** Completed evaluate-only diagnostic; `execution_status: PASS`,
`decision: TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`. Multi-init validation (REC-004AM), candidate adoption,
and bundle promotion remain BLOCKED. G1 and G4 remain uncleared independent research blocks.

**Contract and preregistered boundary:**
- Diagnostic objective: Evaluate token-identifiability-stratified gradient alignment under standard token-output
  cross-entropy loss without optimizer updates (zero parameter mutation) across all 13 checkpoints (0..6000) of REC-004AL
  single-init on development validation, to determine whether the apparent loss-gradient misalignment identified in
  ADR-0139 reflects an inherent standard-loss routing geometry failure, parameterization inversion, or token-aliasing
  credit dilution.
- Fixed mutually exclusive strata for Length 10 / Position 4 (correct key 0, false attractor key 7):
  - (A) UNIQUE_TARGET: target token occurs ONLY at correct key 0 (`target_token not in input_tokens[1:]`).
  - (B) ALIASED_OTHER: target token occurs at other keys, but NOT at false attractor key 7 (`target_token != input_tokens[7]`, matches > 1).
  - (C) ALIASED_KEY7: target token occurs at key 7 (`target_token == input_tokens[7]`).
- Strict evaluation-only execution boundary:
  - Zero optimizer updates (`optimizer.step()` forbidden, updates = 0).
  - Parent bundle Core (`canonical_state_hash: b3a0d5c0774a36f56281bfeadffce83ce61a6818816c7cf69dcae4a5d3fec585`)
    and 15 non-MIRROR primitives are strictly immutable.
  - Source REC-004AL artifacts (all 13 checkpoints 0..6000) verified bit-identical via raw SHA-256.
  - Post-forward oracle position map used exclusively for diagnostic metrics, never in training loss.
  - Zero candidate adoption, zero bundle write (`candidate_selected: null`, `child_bundle: null`).
  - Zero sealed evaluation partition accessed. RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`.

**Empirical results (`runs/phase_b_restart/rec004ap/run_001/`):**
1. Strata Partition and Distribution on Length 10 (206 total validation examples):
   - Stratum A (unique target): 77 examples (37.38%).
   - Stratum B (aliased other): 102 examples (49.51%).
   - Stratum C (aliased key 7): 27 examples (13.11%).
   - Mutual exclusivity and exhaustiveness verified: $77 + 102 + 27 = 206$. Non-aliased/other-aliased examples constitute 86.89%.
2. Mid-Training Dynamics (Steps 500..5000):
   - For 86.89% of examples (Strata A and B), standard token CE loss produces predominantly **CORRECTIVE** gradient alignment
     in parameter space ($-g_{\text{margin}} \cdot g_{\text{loss}} > 0$ in 7/9 checkpoints for Stratum A, and 8/9 checkpoints for Stratum B).
   - At onset step 500:
     - Stratum A: $\Delta M_{\text{param}} = +9.1296$ (54.5% positive examples, median $+3.9381$).
     - Stratum B: $\Delta M_{\text{param}} = +11.5596$ (61.8% positive examples, median $+6.5714$).
     - Stratum C: $\Delta M_{\text{param}} = -31.0566$ (96.3% negative examples, median $-27.0940$).
   - Across steps 500..5000, Stratum C produces an enormous destructive gradient (mean magnitude $28.52$ vs $2.30$ for Stratum A,
     $12.4\times$ dominance ratio), diluting and reversing the pooled margin gradient and driving position 4 into key 7.
   - In score space, Stratum C exhibits $\partial \mathcal{L} / \partial S(4, 7) \ll 0$ ($-4.7 \times 10^{-3}$ at step 500), emitting
     a strong false success signal because routing to key 7 happens to retrieve the correct token.
3. Terminal Step 6000 Lock-In:
   - Prolonged lock-in causes key 0 probability to collapse to $p(0) \approx 1.15 \times 10^{-4}$, starving key 0 parameter gradients
     by $>1000\times$ relative to key 7.
   - Under this extreme saturation at step 6000:
     - Stratum A: $\Delta M_{\text{param}} = -0.7266$ (median $-0.9921$, 55.8% negative). Corrective alignment attenuates and inverts.
     - Stratum B: $\Delta M_{\text{param}} = +0.8263$ (median $+0.0800$, 52.0% positive).
     - Stratum C: $\Delta M_{\text{param}} = -6.3716$ (median $-6.2296$, 100.0% negative).
     - Pooled: $\Delta M_{\text{param}} = -0.6976$ (57.8% negative).
4. Control Comparisons across Identical Stratification:
   - L10 Pos 3 Control (correct key 1, competitor 7): 93.2% token accuracy, top-1 key 1, positive margin change throughout training.
   - L10 Pos 0 Control (correct key 4, competitor 3): 100% token accuracy, top-1 key 4, consistently positive alignment.
   - L8 Pos 4 & L9 Pos 4 Controls: 100% token accuracy, consistently positive alignment.
5. Integrity Audits:
   - Optimizer updates: 0. Checkpoint hashes bit-identical to REC-004AL. Core and parent bank unmodified.
   - Candidate selected: null, child bundle: null, bundle write: false. RG3: NOT_EXECUTED. Sealed access: 0.

**Scientific conclusion:**
The failure mechanism is conclusively resolved as:
`TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`.
The ADR-0139 misalignment paradox is refined into a coupled two-phase mechanism:
1. **Primary Onset Driver (Steps 500-5000): Token-Aliasing Credit Dilution.**
   Standard token-output cross-entropy loss cannot distinguish between routing to the correct coordinate and accidentally routing
   to an identical token elsewhere in the sequence. For 86.9% of examples (Strata A & B), the loss produces corrective parameter-space
   margin alignment. However, in 13.1% of examples (Stratum C), key 7 happens to contain the target token, producing an overpowering
   destructive gradient ($12.4\times$ larger magnitude) that reinforces key 7 and inverts the pooled gradient.
2. **Secondary Terminal Lock-In (Steps 5500-6000): Softmax Gradient Starvation.**
   Once key 7 is reinforced over early updates, key 0 probability drops to $\sim 10^{-4}$, starving its backpropagation so severely
   that even non-aliased examples (Stratum A) lose corrective traction by terminal step 6000.
Thus, pooled loss-gradient misalignment in ADR-0139 was not an inherent representation geometry failure, but credit assignment corruption
from token aliasing exacerbated by softmax saturation.

**Decision and authorized next steps:**
Decision: `TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`.
1. Multi-init validation (REC-004AM) remains BLOCKED.
2. Candidate adoption and bundle promotion remain BLOCKED (`candidate_selected: null`, `child_bundle: null`).
3. RG3, REC-005, G1, and G4 remain uncleared and BLOCKED.
4. Next learning pilot authorization: NOT AUTHORIZED within REC-004AP.
   Any future optimization repair must specifically mitigate token-aliasing credit dilution (e.g. sequence-level distinctiveness,
   coordinate regularization, or anti-saturation) strictly within standard unoracle/unsupervised loss boundaries.
   Oracle/teacher supervision remains strictly forbidden.

## ADR-0141: REC-004AQ CD-DPCA Sequence-Distinctness Warm-Start Single-Recipe Causal Pilot Resolves Position-4 False Attractor and Clears Terminal Viability Floor (98.5% Sequence EM, 100% Length-10 EM) (`WARM_START_PILOT_VIABILITY_MET`)

**Date:** 2026-09-13

**Status:** Completed single-recipe causal pilot; `execution_status: PASS`,
`decision: WARM_START_PILOT_VIABILITY_MET`. Multi-init validation (REC-004AM) is authorized for consideration.
Candidate adoption, child bundle creation, RG3 recheck, REC-005, G1, and G4 remain strictly BLOCKED and uncleared.

**Contract and preregistered boundary:**
- Pilot objective: Test the causal hypothesis derived from ADR-0138 through ADR-0140 that early token-aliasing
  credit dilution drives position-4 lock-in to false attractor key 7 in CD-DPCA (`ContentDecoupledDiscretePositionalCrossAttentionPrimitive`).
  Evaluate whether a sequence-distinctness warm-start (steps 1–500) under standard token-output CE loss enables
  CD-DPCA from initialization I01 to clear the terminal viability floor (sequence EM $\ge 0.95$ at step 6000)
  and resolve the position-4 false attractor without regression.
- Baseline match against REC-004AL (I01):
  - Step-0 model parameters & canonical hash (`045d85cae86d54ce1caca1947a805f2f424df55c34fba4cde11cafa6bbec49dc`) bit-for-bit identical.
  - Optimizer: AdamW (`lr = 0.0008, weight_decay = 0.0001, grad_clip = 1.0`).
  - Scheduler: CosineAnnealingLR (`T_max = 1000, eta_min = 1e-5`, mechanical extension).
  - Batch size: 32 examples per update; total updates: 6,000; checkpoint cadence: every 500 steps (steps 0..6000, 13 checkpoints).
  - Evaluation set: fixed development validation split (1,024 examples).
  - Freeze isolation: Parent bundle Core (`canonical_state_hash: b3a0d5c0774a36f56281bfeadffce83ce61a6818816c7cf69dcae4a5d3fec585`)
    and 15 non-MIRROR primitives strictly frozen (`requires_grad = False`). Only the MIRROR temporary primitive (id 12) updated.
- Single causal intervention:
  - Steps 1–500: Tokens in each sequence sampled without replacement from `range(vocab_size=10)`, ensuring pairwise-distinct
    tokens within every sequence.
  - Steps 501–6000: Exact return to REC-004AL baseline sampler (with replacement) and per-step seed formula (`ibc._generate_step_training_examples`).
  - Boundary: 500-step boundary fixed a priori from false-attractor onset evidence; zero sweep over boundary steps.
  - Uniformity: Rule applies uniformly across all lengths (6..10) and positions; no filtering by length 10, position 4, or correct-key map.
  - Loss: Standard token-output cross-entropy loss only. Zero oracle/teacher routing loss, zero auxiliary loss,
    zero entropy/temperature changes, zero learning rate modifications.
- Terminal viability criteria:
  - Sequence EM $\ge 0.95$ at decisive step 6,000 on the 1,024-example existing development validation set.
  - Length-10 position-4 false attractor fully resolved (top-1 key = 0, token accuracy $\ge 0.95$).
  - No regression across other positions (token accuracy $\ge 0.90$).

**Empirical results (`runs/phase_b_restart/rec004aq/run_001/`):**
1. Validation Metrics at Decisive Step 6,000:
   - Overall Sequence Exact Match: `0.985352` (1,009 / 1,024 examples) vs baseline `0.793945` (+0.1914).
   - Terminal Viability Threshold: `0.950000`.
   - Terminal Viability Met: `True` (`terminal_floor_met: true, position4_attractor_cleared: true, other_positions_regressed: false`).
   - Overall Token Accuracy: `0.998168` (8,235 / 8,250 tokens) vs baseline `0.973254`.
2. Per-Length Breakdown at Decisive Step 6,000:
   - Length 6: Sequence EM = `0.9608` (196 / 204), Token Acc = `0.9935` (1,192 / 1,200).
   - Length 7: Sequence EM = `0.9813` (210 / 214), Token Acc = `0.9973` (1,707 / 1,712).
   - Length 8: Sequence EM = `1.0000` (194 / 194), Token Acc = `1.0000` (1,552 / 1,552).
   - Length 9: Sequence EM = `0.9854` (203 / 206), Token Acc = `0.9984` (1,851 / 1,854).
   - Length 10: Sequence EM = **`1.0000`** (206 / 206), Token Acc = **`1.0000`** (1,927 / 1,927) — total recovery from baseline 0.3398.
   - All 5 lengths clear the 0.95 floor.
3. Position-4 Routing Trajectory & Attractor Resolution:
   - Step 0: top-1 key 3, $p(0) = 0.0500$, $p(7) = 0.0951$, margin $-0.60$, token accuracy $0.0777$.
   - Step 500 (end of warm-start): top-1 key **0** (correct), $p(0) = 0.4083$, $p(7) = 0.1627$, margin **$+0.55$**, token accuracy $0.3495$.
     (vs baseline REC-004AL step 500: top-1 key 7, margin $-6.47$, locked in).
   - Step 1000 (reverted to standard data stream for 500 steps): top-1 key 0, $p(0) = 0.7127$, $p(7) = 0.0219$, margin **$+2.97$**, token accuracy $0.8252$.
   - Steps 1500..6000: position-4 margin expands monotonically from $+3.59$ to **$+5.38$**; token accuracy reaches **$1.0000$** (206/206) at step 2000 and stays perfect through step 6000.
   - At terminal step 6000: 206 of 206 examples (100.0%) select correct key 0 as top-1; competitor key 7 count is **0** ($0.0\%$).
4. Strata Gradient Alignment Across Checkpoints:
   - During warm-start (steps 1..500), pairwise-distinct sampling eliminated token aliasing, preventing Stratum C destructive signals.
   - At step 500: Stratum A produced $\Delta M_{\text{param}} = +12.52$, driving position 4 into key 0.
   - Across steps 1000..6000 (post-reversion): pooled margin gradient remained positive ($+6.19$ at step 1000, $+0.34$ at step 5500, $+0.015$ at step 6000), while Stratum C destructive magnitude attenuated to near zero ($-0.0050$), refuting post-reversion relock.
5. Causal Controls on Step 6000:
   - Correct Control EM: `0.9854` (1,009 / 1,024).
   - Wrong-Family Control EM (REVERSE): `0.0000` (0 / 1,024).
   - None Control EM (unrouted baseline): `0.0000` (0 / 1,024).
   - Causal Gap: `0.9854` (statistically verified causal dependency on operator).
6. Attention & Boundary Diagnostics:
   - Padding mask: strictly verified (`padding_mask_verified: True`).
   - Freeze audit: `core_frozen_verified: True`, optimizer updated zero Core or non-MIRROR parameters.
   - Side effect audit: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, `rg3: NOT_EXECUTED`, `rec005_eligible: false`, `g1: NOT_CLEARED`, `g4: NOT_CLEARED`.

**Scientific conclusion:**
The causal intervention conclusively confirms the hypothesis established across ADR-0138 through ADR-0140:
1. The CD-DPCA single-init optimization failure in REC-004AL was entirely driven by early token-aliasing credit dilution (Stratum C)
   locking output position 4 into false attractor key 7 during the initial 500 steps.
2. A sequence-distinctness warm-start for the initial 500 updates—applied uniformly across all lengths and positions without
   architectural changes, auxiliary losses, or teacher supervision—cleanly guides the primitive into the basin of the true permutation attractor.
3. Upon returning to the standard data stream at step 501, the true permutation attractor is globally stable against subsequent
   token-aliasing noise, achieving **$98.54\%$** decisive sequence EM and **$100.0\%$** length-10 sequence EM at step 6000.

**Decision and authorized next steps:**
Decision: `WARM_START_PILOT_VIABILITY_MET`.
1. Next authorized step: REC-004AM multi-initialization validation (5 inits under this fixed warm-start recipe) may now be planned and considered.
2. Candidate adoption and bundle creation remain strictly BLOCKED (`candidate_selected: null`, `child_bundle: null`).
3. RG3, REC-005, G1, and G4 remain uncleared and BLOCKED.

## ADR-0142: Scope agent guidance to the requested work and preserve research gates

**Date:** 2026-09-13

**Status:** Accepted — documentation and agent-guidance maintenance only.

**Context:** The user requested an audit of project skills and `AGENTS.md` against
OpenAI's [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).
The article recommends precise skill triggers, conditional reference reading,
and explicit completion and authorization boundaries. The root guide's opening
research-reading requirement was not clearly scoped away from routine edits.
Its single-task wording and the three Phase B addenda could also conflict with
the restart plan's existing explicit continuation permission. Recovery's
all-task test requirement obscured the root's documentation-only exception.

**Decision:**

- Scope research-contract reading to research implementation, execution, and
  status review; use affected files and relevant references for routine edits.
  Route through the applicable execution plan instead of listing every document
  as an apparent mandatory reading sequence.
- Honor explicit user scope and existing continuation permission without repeated
  task-boundary approval. Completion or a diagnostic success alone grants no new
  research authority. Explain an actual blocker with its file and exact clause.
- Keep all causal invariants, STOP GATEs, sealed-data boundaries, historical
  evidence, implementation verification commands, and task-specific checks.
  A negative research result still receives an evidence handoff; it is never
  relabeled PASS to satisfy a completion instruction.
- Centralize verification scope in the root guide. Documentation-only edits need
  link/consistency/diff checks; new behavior needs appropriate regression coverage.
  Passing checks need repeating only for new changes or unresolved concerns.
- No project-owned `SKILL.md` was found in the active checkout, including hidden
  instruction locations (excluding other worktrees, environments, caches, and
  generated outputs). Do not create a redundant skill merely to package these
  repository rules. Shared installed plugin/system skills remain outside this
  project's ownership. Future project skills should have narrow descriptions and
  conditional references without duplicating `AGENTS.md`.

**Validation:** Relative file links and anchors in the root and addendum guides
were checked (23 references across 13 files). The root's entire scientific
invariants section is unchanged from the preceding commit, and all three standard
implementation-check commands remain present. Manual scenario review covered a
documentation edit, an implementation fix, an already authorized continuation,
a failed scientific gate, and a request with no applicable project skill.
`git diff --check` passed for the guidance changes. These are static checks and
instruction review, not a behavioral model evaluation. Pytest, Ruff, mypy,
training, and evaluations were not run because this change only edits Markdown.

**Consequences:** This maintenance task does not start a research task, change
any acceptance threshold, clear any recovery/research gate, or expand the user's
research authorization. Existing changes to `config.json` and `prompts/planner.md`
are excluded from this commit. Historical contracts and measurements are preserved.
