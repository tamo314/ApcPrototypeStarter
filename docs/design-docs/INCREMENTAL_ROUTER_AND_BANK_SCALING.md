# Incremental Router and Bank Scaling

## 1. Problem

B008 showed ~99.99% routing with 10 known/consolidated operations.

A lifelong bank will grow. The key question becomes:

> Can routing remain accurate when new primitive classes are added without full historical retraining?

## 2. Incremental update conditions

For each new primitive installation, evaluate three conditions.

### R0 — Full retrain upper bound

Retrain router using all historical routing data. Diagnostic upper bound only.

### R1 — Naive new-class update

Update using new-class data only. Expected forgetting baseline.

### R2 — Bounded incremental update

Primary condition.

Use new-class examples plus a bounded replay memory or prototypes, with no unconstrained historical dataset.

Suggested replay budget:

- <=32 examples per old class, or
- <=512 total historical examples,

whichever is smaller for the active experiment.

## 3. Incremental acceptance

After each class addition:

- new-class top-1 >=95%
- old-class mean top-1 drop <=2pp
- worst old-class drop <=5pp
- overall top-1 >=95%
- unselected primitive forward calls = 0

Report the entire accuracy trajectory after every insertion.

## 4. Semantic bank growth vs routing-only scaling

Use two separate experiments.

### Semantic growth

Actually consolidate several new executable operations.

Preferred target:

`10 -> 12 -> 14 -> 16`

This tests real class-incremental routing.

### Routing/compute scaling

For larger resident sizes:

`16, 32, 64, 128`

matched-size frozen distractor bank entries may be used **only for scaling/competition tests**.

Do not describe distractor scaling as semantic continual-learning success.

## 5. Bank-size metrics

At each size report:

- router top-1/top-k accuracy,
- distractor false-selection rate,
- old-class accuracy,
- recurrence accuracy,
- resident params,
- active params,
- router FLOPs,
- primitive FLOPs,
- latency,
- memory.

## 6. Scaling success

For semantic bank size 16:

- top-1 >=95%
- old-class drop <=2pp
- recurrence reuse >=90%

For routing-only N=128:

- known-task top-1 >=95%
- distractor false selection <=5%
- active primitive forward count remains exactly one for top-1
- non-selected calls remain zero

## 7. Top-k

Top-k may be measured diagnostically.

Primary sparse inference remains top-1 unless a task explicitly authorizes top-k>1.

Do not hide poor top-1 behavior behind top-k metrics.
