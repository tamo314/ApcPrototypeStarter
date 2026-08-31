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

Phase A is complete with a negative scientific verdict and remains preserved as
historical evidence in `docs/exec-plans/completed/PHASE_A_RESULT.md`.

Current milestone: **Phase A.1 — Hypothesis Isolation and Oracle Ladder**.

Read `docs/exec-plans/active/PHASE_A1.md` and the linked Phase A.1 addendum,
architecture delta, experiment plan, and task queue before implementing features.

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
012). See `docs/DECISIONS.md` ADR-0006 through ADR-0009 for load-bearing
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
`docs/DECISIONS.md` ADR-0010 through ADR-0012 for how the B1 fixed bank is
populated, why B2/B3 bypass the primitive bank/router entirely, and why B3
has its own replay-weight config key instead of reusing consolidation's.

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
- `docs/DECISIONS.md` — architecture decision log.
- `docs/research/REFERENCES.md` — existing research relevant to APC.
