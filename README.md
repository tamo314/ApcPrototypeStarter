# APC Prototype

**Adaptive Primitive Consolidation (APC)** is a research prototype for testing whether a neural system can:

1. solve familiar tasks with a small sparse stable system;
2. detect when existing primitives/compositions are insufficient;
3. temporarily allocate extra plastic capacity;
4. learn a genuinely new operation;
5. compress the learned computation into a smaller reusable primitive;
6. validate the compressed primitive against the temporary solution;
7. release temporary capacity without catastrophic forgetting;
8. reuse the learned primitive when the capability is needed again.

The first goal is **not** to build a competitive LLM. The first goal is to establish or falsify the closed learning loop on controlled synthetic tasks.

## Target development machine

- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- System RAM: 64GB
- Recommended development environment: Linux or WSL2 Linux; native Windows is acceptable if the PyTorch/CUDA stack is verified.
- Python: 3.12
- PyTorch: current stable release with a Blackwell-compatible CUDA wheel (see `docs/HARDWARE_ENVIRONMENT.md`).

## Project status

- **Phase A**: Complete with a negative scientific verdict; preserved as historical evidence in `docs/exec-plans/completed/PHASE_A_RESULT.md`.
- **Phase A.1**: Complete with Branch B integration and 10-operation closed-universe learned routing (`docs/exec-plans/active/PHASE_A1_BRANCH_B_INTEGRATION.md`, `ADR-0061`).
- **Phase A.2**: Complete with strong autonomous controller support across all 4 STOP GATES and real sparse latency scaling up to N=128 (`docs/results/PHASE_A2_AUTONOMOUS_CONTROLLER_RESULT.md`, `ADR-0073`).
- **Phase B (Closed / archived)**: `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`. The [final evidence ledger](docs/results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md) preserves the supported results, falsified claims and unproved boundaries; the [archived restart plan](docs/exec-plans/active/PHASE_B_RESTART.md) retains historical contracts. No Phase-B experiment is queued.
- **Phase C (Closed / archived — `TERMINATED_CURRENT_CHARTER`)**: [Independent routing-identifiability charter](docs/research/PHASE_C_RESEARCH_CHARTER.md), never approved and never executed experimentally; falsified/retracted entirely during pre-execution mathematical review. The [termination evidence ledger](docs/results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md) is authoritative. No Phase-C task is queued; any future research question requires a new, independently authorized charter. Task C-D001 stopped at `RELATION_INVENTORY_FEASIBILITY_STOP` ([ADR-0150](docs/DECISIONS_PHASE_C.md#adr-0150-c-d001-oracle-free-routing-identifiability-contract-derivation--relation-inventory-feasibility-audit-stops-on-relation-count-sufficiency-relation_inventory_feasibility_stop)); Task C-D001R corrected stop to `ROUTING_IDENTIFIABILITY_STOP` ([ADR-0151](docs/DECISIONS_PHASE_C.md#adr-0151-c-d001r-adversarial-cross-examplecross-relation-identifiability-falsification-audit-corrects-stop-to-routing_identifiability_stop)); Task C-D001S confirmed `ROUTING_IDENTIFIABILITY_STOP` as a quantifier-complete impossibility theorem across all permitted observables ([ADR-0152](docs/DECISIONS_PHASE_C.md#adr-0152-c-d001s-quantifier-complete-task-sidesupport-identifiability-boundary-audit-confirms-routing_identifiability_stop-across-permitted-observable-space)); Task C-D001T proved lawful compositional descriptors with tie-break semantics separate adversarial worlds (Theorem 3), retracted and restricted ADR-0152's quantifier-complete claim, and reclassified stop to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0153](docs/DECISIONS_PHASE_C.md#adr-0153-c-d001t-semantic-descriptor-boundary--impossibility-proof-repair-audit-qualifies-stop-to-routing_identifiability_qualified_stop-adr-0152-quantifier-retracted--restricted)); Task C-D001U proved separating descriptors are $z$-computable and candidate-selection procedures are oracle-equivalent supervision under Charter H-C1, rejected Contract v1.1, and reconfirmed STOP ([ADR-0154](docs/DECISIONS_PHASE_C.md#adr-0154-c-d001u-semantic-descriptor-oracle-equivalence--minimality-audit-reconfirms-routing_identifiability_stop)); Task C-D001V pre-registered 5-dimensional non-circular oracle criteria calibrated against positive/negative controls, proved `FIRST/LAST`, `LEFTMOST/RIGHTMOST`, and procedural composition are strictly non-oracle, proved they separate World A/B on collisions, retracted ADR-0154's universal STOP, and reclassified stoppage to `ROUTING_IDENTIFIABILITY_QUALIFIED_STOP` ([ADR-0155](docs/DECISIONS_PHASE_C.md#adr-0155-c-d001v-non-circular-oracle-equivalence-falsification-experiment-retracts-adr-0154-universal-stop-to-routing_identifiability_qualified_stop)); Task C-D001W validated the 5-dimensional criterion against adversarial mixed controls, confirmed representation invariance, monotonicity, and leave-one-out necessity, upheld the 0/5 determination, and derived Training Information Contract v1.1 & deterministic baseline requirements ([ADR-0156](docs/DECISIONS_PHASE_C.md#adr-0156-c-d001w-adversarial-mixed-control-validation-confirms-adr-0155-05-non-oracle-determination-and-derives-training-information-contract-v11--deterministic-baseline)); Task C-D001X audited Contract v1.1 and the deterministic baseline ($B_{\text{det}}$), deconstructed H-C1's estimand, proved $B_{\text{det}}$ dominates 100% of routing/execution floors with 0 learning and $H(Z|X,D)=0.0$, determined Contract v1.1 trivializes H-C1, and recommended Charter retraction or restriction to continuous neural grounding ([ADR-0157](docs/DECISIONS_PHASE_C.md#adr-0157-c-d001x-contract-v11-hypothesis-preservation--deterministic-baseline-dominance-audit-finds-h-c1-estimand-trivialized-by-unlearned-deterministic-reduction)); Task C-D001Y defined Embedding-Aware Deterministic Baseline ($B_{\text{det\_emb}}$), proved it dominates invertible continuous representations with 100% ceiling and 0 learning across Controls 1, 2, 5, proved lossy failures (Controls 3, 4) are an identifiability limit rather than learnability limit, and recommended further Charter retraction or restriction to blind manifold discovery ([ADR-0158](docs/DECISIONS_PHASE_C.md#adr-0158-c-d001y-embedding-aware-deterministic-baseline-closure-audit-finds-continuous-neural-grounding-dominated-by-b_det_emb-under-invertible-representations)); Task C-D001Z formalized permutation ($S_V$) and orthogonal ($O(d)$) group actions, proved blind grounding is an identifiability impossibility ($H(Z|\text{obs}) > 0$) via indistinguishable symmetric worlds ($H_A=H_B, D_A=D_B, y_A=y_B, z^*_A \ne z^*_B$), proved unanchored controls fail floors while complete anchors trigger deterministic baseline dominance ($B_{\text{det\_emb}} = 1.000$), proved via the Dilemma Theorem that the non-trivial residual learning space is empty, and formally retracted residual Charter candidate `H-C1-Residual` ([ADR-0159](docs/DECISIONS_PHASE_C.md#adr-0159-c-d001z-blind-codebook-identifiability--symmetry-breaking-audit-proves-blind-grounding-impossible-and-retracts-residual-charter-candidate)); Task C-D001AA audited ADR-0150-0159 for falsification sufficiency, confirmed no in-charter conclusion-changing task remains, separated "this charter is falsified" from "APC in general is impossible," and formally terminated the charter (`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`, [ADR-0160](docs/DECISIONS_PHASE_C.md#adr-0160-c-d001aa-phase-c-falsification-sufficiency--charter-termination-audit-closes-the-current-charter-phase_c_current_charter_falsification_sufficient)). No architecture was ever selected and no first experiment was ever defined or authorized.
- **NRQ-001 (`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`)**: independent cross-phase [next-research-question review](docs/research/NEXT_RESEARCH_QUESTION_REVIEW_NRQ001.md) (not a Phase C task). Generalizes ADR-0159's Impossibility-Dominance Dilemma to a representation-agnostic dichotomy, surveys six candidate estimands, and finds none satisfy all of: non-trivial vs. a deterministic baseline, information-theoretically identifiable, oracle-free, relation-transfer compatible, and actually testing APC's core separation. Re-confirms the relation-inventory deficit (1/2 validation, 1/2 sealed) and confirms `PROGRAM_LINE_CLOSURE_CONFIRMED` for the oracle-free task/relation-inference research line (Phase B -> Phase C) — not a claim of APC's general impossibility, and not a reversal of Phase A/A.1/A.2's positive core-separation evidence ([ADR-0161](docs/DECISIONS_PHASE_C.md#adr-0161-nrq-001-next-research-question-review-finds-no-non-trivial-identifiable-estimand-no_nontrivial_estimand_identified)). No task is queued.
- **NRQ-002 (`NO_COUNTEREXAMPLE_CONSTRUCTED`)**: [constructive falsification experiment](docs/research/CONSTRUCTIVE_FALSIFICATION_NRQ002.md) (not a Phase C task) that actively tries to construct a counterexample to NRQ-001's dichotomy, targeting the "one honest loophole" (tractability of the disambiguating function under finite composition) NRQ-001 left open. Four construction attempts are each closed — one empirically, via Task A1-B004's already-measured 99.62%-EM oracle-free beam-search baseline on the exact construction attempted; two via this project's own measured bank/argument-domain scale; one on the same scope/inventory grounds as NRQ-001's candidate N2 — and a Bounded-Resource Corollary generalizes the closure to any finite composition of the current registry at in-scope scale. Reinforces, without reopening, `PROGRAM_LINE_CLOSURE_CONFIRMED` ([ADR-0162](docs/DECISIONS_PHASE_C.md#adr-0162-nrq-002-constructive-falsification-experiment-finds-no-counterexample-to-the-lawful-disambiguation-dichotomy-no_counterexample_constructed)). No task is queued.





## Development environment

The package requires Python 3.12. PyTorch is constrained to the tested minor-version
window `>=2.12,<2.14` so a new install cannot silently cross into an unverified release.

Create and install an editable development environment with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots]"
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and
activate it with `.venv\\Scripts\\Activate.ps1`.

## Phase A success question

Can a small model learn a sequence containing both novel compositions and genuinely novel operations, expand only for the latter, consolidate the learned computation into persistent low-rank primitives, free the temporary capacity, and retain previous capabilities?

## Planned package layout

```text
src/apc/
├── core/              # Stable core and working state
├── primitives/        # Primitive representation, bank, sparse router
├── plastic/           # Temporary capacity and allocation
├── consolidation/     # Distill, merge, prune, shadow validation
├── meta/              # Novelty estimator and finite-state controller
├── environments/      # Synthetic compositional tasks
├── evaluation/        # Retention, compute, reuse and growth metrics
└── utils/

tests/
configs/
scripts/
runs/                  # Experiment outputs; gitignored
```

## Initial implementation order

1. Repository skeleton, config system, deterministic synthetic data.
2. Stable core and supervised baseline.
3. Primitive representation and top-k router.
4. Primitive bank with parameter/accounting metrics.
5. Task families for known primitives, unseen compositions, and novel operations.
6. Novelty estimator and finite-state controller.
7. Expandable plastic workspace.
8. Consolidation into persistent primitives.
9. Shadow validation and resource release.
10. Sequential continual-learning benchmark and baselines.

Do not start with RL meta-control or pretrained LMs.

## Verification commands

These commands are the intended interface once the scaffold is implemented:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
python scripts/smoke_train.py --config configs/phase_a_smoke.yaml
python scripts/composition_benchmark.py --run-dir runs/phase_a_smoke
python scripts/novel_operation_benchmark.py --run-dir runs/phase_a_smoke
python scripts/sequential_benchmark.py --config configs/phase_a_sequential.yaml --run-dir runs/phase_a_sequential
python scripts/baseline_benchmark.py --config configs/phase_a_sequential.yaml --run-dir runs/phase_a_baselines
python scripts/stable_core_generalization_gate.py --config configs/phase_a1_stable_core_gate.yaml --run-dir runs/phase_a1_stable_core_gate
python scripts/stable_core_generalization_gate.py --config configs/phase_a1_stable_core_gate_permuted_copy_only.yaml --run-dir runs/phase_a1_stable_core_gate_permuted_copy_only
python scripts/shared_core_generalization_gate.py --config configs/phase_a1_shared_core_gate.yaml --run-dir runs/phase_a1_shared_core_gate
python scripts/task_content_probes_gate.py --config configs/phase_a1_task_content_probes.yaml --run-dir runs/phase_a1_task_content_probes
```

`composition_benchmark.py` evaluates a trained checkpoint on known-operation
(`K`) and held-out novel-composition (`C`) examples and writes
`composition_benchmark.json` into the run directory (Task 006). It requires a
run directory already produced by `smoke_train.py` or an equivalent training
script.

`novel_operation_benchmark.py` evaluates the same kind of checkpoint on
held-out novel-composition (`C`) and genuinely novel-operation (`N`, e.g.
`SORT`) examples and writes `novel_operation_benchmark.json` into the run
directory (Task 009). The checkpoint's training data never included the
novel operation, so `novelty_gap` (`C` minus `N` exact match) measures
whether a static composition baseline fails on `N` materially more than on
`C`.

`sequential_benchmark.py` runs the full closed loop end to end (pretraining
included) over the K/C/N/R task stream from `docs/exec-plans/active/
PHASE_A.md` Milestone A9 -- at least two learn/consolidate/release cycles,
one per novel operation (`SORT`, `REVERSE`) -- and writes `report.json` plus
`plots/` (requires the `plots` optional dependency group, `pip install
-e .[plots]`; pass `--no-plots` to skip it) into the run directory (Task
012). See `docs/DECISIONS_PHASE_A.md` ADR-0006 through ADR-0009 for load-bearing
design choices and measured limitations behind this benchmark's defaults,
in particular that the current dense core does not generalize known-op
execution to unseen token content, so novelty/PLASTIC/shadow are all
evaluated against the same fixed per-event example set rather than a
held-out split.

`baseline_benchmark.py` runs `docs/EXPERIMENT_PLAN.md` section 5's B0-B4
baselines (fixed dense, fixed sparse, grow-only, grow-plus-replay, and the
full APC loop) under one shared config -- same seed, task stream,
per-event data, model size, and PLASTIC-equivalent training budget for
every baseline -- writing each one's `report.json` under its own
subdirectory of the run directory, a combined `summary.json`, and (unless
`--no-plots`) cross-baseline comparison plots in `plots/` (Task 013). See
`docs/DECISIONS_PHASE_A.md` ADR-0010 through ADR-0012 for how the B1 fixed bank is
populated, why B2/B3 bypass the primitive bank/router entirely, and why B3
has its own replay-weight config key instead of reusing consolidation's.

`stable_core_generalization_gate.py` runs Phase A.1 Task A1-006, the Stable
Core systematic-generalization gate (H1 in `docs/EXPERIMENT_PLAN_PHASE_A1.md`,
a **STOP GATE**): trains a fresh Stable Core per `(operation, seed)` pair on
online-generated examples of one deterministic known operation at a time --
never a mixed pool, see `docs/DECISIONS_PHASE_A1.md` ADR-0020 -- with no primitive
bank, router, plastic workspace, or consolidation involved, then evaluates
exact match on a large held-out batch of unseen content. Writes one
`<operation>/seed_<n>/metrics.jsonl` per pair plus a combined
`report.json`/`summary.json` (Task A1-006's acceptance: mean unseen-content
exact match >= 0.95 across >= 5 seeds, all seeds reported, aggregated over
the whole operation x seed grid). Two config variants are shipped:
`configs/phase_a1_stable_core_gate.yaml` (primary: all four deterministic
operations -- `apc.environments.operations.DETERMINISTIC_OPERATION_NAMES`,
i.e. `COPY`/`NEGATE`/`COMPARE`/`ACCUMULATE`; see ADR-0017 for why
`SELECT`/`COUNT`/`SHIFT`/`BIND` are excluded -- with symbol permutation
off) and `configs/phase_a1_stable_core_gate_permuted_copy_only.yaml`
(secondary: `COPY` only, with symbol permutation on). See ADR-0019 for why
symbol permutation is opt-in here rather than the default: it makes
value/order-dependent operations (`NEGATE`/`COMPARE`/`ACCUMULATE`) provably
unrecoverable on held-out content regardless of training duration, so only
the primary (unpermuted) variant is a meaningful test of H1 across the full
operation set.

`shared_core_generalization_gate.py` runs Phase A.1 Correction Task A1-C004,
the Shared-Core systematic-generalization gate (H1b in `docs/
EXPERIMENT_PLAN_PHASE_A1.md`/`docs/exec-plans/active/PHASE_A1_CORRECTION.md`
A1-CM3, a **STOP GATE**): trains **one shared** Stable Core per seed on
`apc.environments.generator.build_mixed_operation_generator`'s online mixed
stream over all eight `KNOWN_OPERATION_NAMES` -- including
`SELECT`/`COUNT`/`SHIFT`/`BIND`, whose previously hidden per-instance
parameter (ADR-0017) is now part of the model-visible task-specification
segment (Task A1-C003) -- with no primitive bank, router, plastic workspace,
or consolidation involved, then evaluates overall and per-operation exact
match on a large held-out batch of unseen content. Also runs a
negative-control variant (`include_task_spec=False`) over the identical
mixed stream and model architecture, with the task segment stripped from the
model input, to verify any gap is attributable to that segment specifically
rather than to "one shared model, mixed data, online generation" alone (see
ADR-0020's ~0.25-0.35 pooled-operation ceiling with no task signal). Writes
one `{explicit,negative_control}/seed_<n>/metrics.jsonl` per variant/seed
plus a combined `report.json`/`summary.json` (Task A1-C004's acceptance:
overall mean unseen exact match >= 0.95, every operation's mean >= 0.90,
across >= 5 seeds, with the negative control materially underperforming the
explicit-task model). `configs/phase_a1_shared_core_gate.yaml` is the sole
shipped config; symbol permutation is left at its default (off), for the
same ADR-0019 reason as `configs/phase_a1_stable_core_gate.yaml`.

`task_content_probes_gate.py` runs Phase A.1 Correction Task A1-C005, the
Task/Content representation-probe gate (H1c in `docs/
EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md`, a **STOP GATE**): freezes one
shared Stable Core per seed -- trained exactly like Task A1-C004's
explicit-task variant (`apc.evaluation.shared_core_generalization.
train_shared_core`, refactored out of that gate so both share one training
implementation) -- and fits small linear probes directly on its
`DecoderOnlyTransformer.encode_split` output: operation identity from
`z_task` (`task_state`) at the `[TASK_END]` position, operation arguments
from `z_task` where applicable (SHIFT/COUNT/BIND as single-value
classification, SELECT's `indices` as a masked multi-hot/set-membership
probe), and per-position content-token identity from `h_content`
(`content_state`) at each content position. Also fits two optional,
non-gating leakage probes (operation from `content_state` at `[TASK_END]`;
content from `task_state` at content positions) -- see `apc.evaluation.
task_content_probes`'s module docstring for why `[TASK_END]` is causally
content-blind by construction (a structural, not merely empirical,
task/content read point) and why both leakage probes are expected to
succeed without contradicting a useful factorization. Writes one
`seed_<n>/metrics.jsonl` (the frozen core's own training progress) per seed
plus a combined `report.json`/`summary.json` (Task A1-C005's acceptance:
operation identity >= 0.95, every applicable-argument probe >= 0.90, content
probe >= 0.98 predeclared per-position token accuracy -- see the module for
the predeclaration rationale). `configs/phase_a1_task_content_probes.yaml`
is the sole shipped config, using 3 seeds (A1-C005 states no minimum seed
count, unlike A1-C004/A1-006's >=5).

If a tool is not yet configured, add it as part of the repository-bootstrap task rather than silently skipping verification.

## Experiment outputs

Every run should write a self-contained directory under `runs/` containing at least:

```text
config.yaml
metrics.jsonl
summary.json
system.json
checkpoint/
```

`system.json` should capture device name, PyTorch/CUDA versions, seed, peak VRAM, and git commit when available.

## Documentation

- `AGENTS.md` — operating instructions.
- `docs/design-docs/ARCHITECTURE.md` — architecture source of truth.
- `docs/exec-plans/active/PHASE_A.md` — actionable development plan.
- `docs/EXPERIMENT_PLAN.md` — experimental design and baselines.
- `docs/HARDWARE_ENVIRONMENT.md` — target machine constraints.
- `docs/TASKS.md` — issue-sized task queue.
- `docs/DECISIONS.md` — architecture decision log index (split by research phase into `docs/DECISIONS_PHASE_A.md`, `docs/DECISIONS_PHASE_A1.md`, `docs/DECISIONS_PHASE_A1_CORRECTION.md`, `docs/DECISIONS_PHASE_A1_POST_CORRECTION.md`, `docs/DECISIONS_A1_R005E_DIAGNOSTIC.md`; ADR numbers are one global sequence across all of them).
- `docs/research/REFERENCES.md` — existing research relevant to APC.
