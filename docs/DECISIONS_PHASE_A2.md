# Architecture Decisions -- Phase A.2 Autonomous Controller & Scaling

## ADR-0062: Phase A.2 Scope and Metric Audit of ADR-0061: Clarification of Explicit TaskSpec, 10-Operation Closed Universe, Primitive-Bank Active Parameter Savings, and Unmeasured FLOPs/Latency

**Date:** 2026-09-05  
**Status:** Accepted (Task A2-C001 Complete)  
**Affects:** `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_A2.md`, `docs/exec-plans/active/PHASE_A2_AUTONOMOUS_CONTROLLER.md`, `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`, `docs/EXPERIMENT_PLAN_PHASE_A2_AUTONOMOUS_CONTROLLER.md`, `docs/design-docs/COMPUTE_ACCOUNTING_PHASE_A2.md`, `AGENTS.md`

### Context
Phase A.1 Branch B Integration culminated in Task A1-B008 (Milestone B-M8, documented in ADR-0061), demonstrating that a learned top-k linear router conditioned on $z_{\text{task}}$ achieved $99.99\%$ top-1 routing accuracy and $99.79\%$ closed-loop exact match over the 10-operation universe with strict sparse execution (0 unselected calls) and $90.11\%$ active parameter reduction.

As the project transitions to Phase A.2 (Autonomous Controller & Scaling), it is scientifically critical to audit and circumscribe the claims of ADR-0061 without rewriting historical artifacts or re-running past experiments. This ensures downstream scaling and controller decisions build on precise foundational claims.

### Decision
1. **Preserve Historical Artifacts:**  
   Preserve ADR-0061, the B008 milestone report, and all experimental artifacts in `runs/phase_a1_learned_routing_benchmark/` exactly as recorded. No historical evidence is retroactively altered or deleted.

2. **Explicit TaskSpec Clarification:**  
   The $99.99\%$ routing fidelity demonstrated in B008 was achieved using explicit, model-visible TaskSpec token sequences (`[TASK_START] <op> ... [TASK_END]`) pooled into $z_{\text{task}}$ at the `[TASK_END]` token position. This mechanism is task routing under explicit specification; it is **not** semantic task inference from natural language, few-shot demonstrations, or implicit context. Semantic task inference is out of scope for Phase A.2 and deferred to subsequent research phases.

3. **Known/Consolidated Universe Scope:**  
   B008 validated learned routing exclusively over a fixed, pre-consolidated 10-operation universe (8 canonical operations + `SWAP_PAIRS` + `INVERT_HALF`). It did **not** demonstrate autonomous continual learning in an online streaming environment with dynamic K/C/N/R decisions, incremental router class addition without catastrophic interference, or open-ended bank expansion.

4. **Primitive-Bank Active Parameter Savings Scope:**  
   The $90.11\%$ "compute savings" reported in B008 strictly measures **primitive-bank active-parameter savings** (17,559 active parameters executed in the selected primitive vs. 177,492 total parameters resident across all 10 bank primitives). It does **not** represent a $90.11\%$ reduction in total end-to-end FLOPs, memory footprint, or wall-clock latency, because the Stable Core (content encoder), Task Encoder, linear router projection, and decoder readout were not included in that parameter reduction ratio.

5. **Unmeasured End-to-End FLOPs and Real Latency:**  
   Total end-to-end FLOPs, median/p95 wall-clock latency, runtime throughput, and the routing scoring overhead as bank size $N$ scales ($O(N)$ router computation) remain unmeasured in Phase A.1. Phase A.2 explicitly introduces comprehensive compute accounting, FLOPs profiling, and latency benchmarking (Tasks A2-C002 and A2-C010).

6. **Phase Activation:**  
   Formally activate **Phase A.2 — Autonomous Controller & Scaling** as the governing research phase.

### Consequences
- Establishes a rigorous baseline for Phase A.2 without invalidating Phase A.1 achievements.
- Prevents overclaiming semantic task inference or end-to-end FLOPs reduction.
- Formally satisfies all acceptance criteria for Task A2-C001 in `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`.
- Unblocks Task A2-C002 (Controller and compute instrumentation).

---

## ADR-0063: Controller Instrumentation and Comprehensive Compute Accounting Architecture (Task A2-C002)

**Date:** 2026-09-05  
**Status:** Accepted (Task A2-C002 Complete)  
**Affects:** `src/apc/meta/episode_log.py`, `src/apc/evaluation/compute_accounting.py`, `tests/test_compute_accounting.py`, `tests/test_episode_log.py`, `docs/DECISIONS.md`, `docs/DECISIONS_PHASE_A2.md`, `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`

### Context
In Phase A.1, compute savings were measured exclusively as active vs. resident primitive parameters (90.11% reduction in B008). As audited in ADR-0062, total FLOPs, wall-clock latency, throughput, peak memory, and per-episode controller action logging were unmeasured and unstandardized.

To prepare for incremental router updates (A2-C003), bank competition scaling (A2-C004), adequacy/novelty control (A2-C005/C006), compact-first plastic policy (A2-C007), and sequential closed-loop stream evaluation (A2-C008), common instrumentation is required prior to running new scientific experiments.

### Decision
1. **Standardized Per-Episode Controller Instrumentation (`src/apc/meta/episode_log.py`):**
   - Implemented `ControllerAction` (`DIRECT_REUSE`, `COMPOSE`, `PLASTIC_SEARCH`) matching Phase A.2 runtime action space.
   - Built `EpisodeRecord` capturing all 11 required per-episode runtime signals:
     1. controller action
     2. proposed primitive ID
     3. composition recipe
     4. support-set direct score
     5. support-set composition score
     6. plastic trigger
     7. bank size before/after
     8. router version
     9. adaptation steps
     10. temporary params
     11. selected/unselected forward calls
   - Built `EpisodeLogger` for sequential episode accumulation, sparse invariant validation, summary metrics, and JSON persistence.

2. **Comprehensive Compute Accounting (`src/apc/evaluation/compute_accounting.py`):**
   - **Parameter Accounting:** Implemented `ParameterBreakdown` and `count_system_parameters` inspecting Stable Core, router, resident primitive, active primitive, and temporary workspace parameters, computing primitive and total parameter savings ratios.
   - **Analytical FLOPs Models:** Implemented $2 \times \text{MACs}$ formulas for task encoder, linear top-k router (capturing $O(N)$ candidate scoring overhead), content Stable Core, selected primitive execution, decoder readout, total sparse path, and dense-all-primitives baseline, reporting `FLOPsBreakdown`.
   - **Runtime Latency Profiling:** Implemented `profile_execution` with warmup iterations, repeated trials, CUDA events or high-resolution clock, reporting median, p95, p99, mean, std, throughput, and peak GPU memory (`torch.cuda.max_memory_allocated`).
   - **True Executable Dense Baseline:** Implemented `execute_dense_primitive_baseline` executing all enabled STABLE primitives in `PrimitiveBank` (strictly avoiding multiplication approximations per `COMPUTE_ACCOUNTING_PHASE_A2.md` section 3).
   - **Strict Sparse Execution Verification:** Implemented `verify_sparse_execution` confirming zero unselected primitive forward calls.

3. **Deterministic Acceptance:**
   - Validated via 15 unit tests across `tests/test_compute_accounting.py` and `tests/test_episode_log.py`.
   - All FLOPs models, parameter formulas, and sparse invariants rederive deterministically from first principles without learned weights.
   - Preserved constraint: no controller-learning claim is made in Task A2-C002.

### Consequences
- Standardizes episode logging and compute accounting across all subsequent Phase A.2 experiments.
- Formally satisfies all acceptance criteria for Task A2-C002 in `docs/CODEX_TASKS_PHASE_A2_AUTONOMOUS_CONTROLLER.md`.
- Unblocks Task A2-C003 (Incremental router update gate).

