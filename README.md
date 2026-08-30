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

Current milestone: **Phase A — Synthetic closed-loop proof of concept**.

Read `docs/exec-plans/active/PHASE_A.md` before implementing features.

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
