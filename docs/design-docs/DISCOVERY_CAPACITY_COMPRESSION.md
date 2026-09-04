# Discovery Capacity vs Persistent Representation Capacity

## Core hypothesis

Operationalize APC's hypothesis as:

`C_discover > C_represent`

Strong practical evidence requires both:

1. a larger temporary workspace discovers the function more reliably or efficiently than a compact workspace;
2. the function can then be consolidated into a compact persistent primitive with high retention.

## Metrics

### Functional retention

`R_func = EM_candidate / max(eps, EM_temp)`

Target: `R_func >= 0.95`.

### Parameter compression

`R_param = P_candidate / P_temp`

Strong compression target: `R_param <= 0.25`.

### Discovery efficiency

Report:
- steps to 90% EM,
- steps to 95% EM,
- examples to threshold,
- seed success rate,
- learning-curve AUC,
- wall-clock.

Operational discovery advantage exists if under matched conditions either:

- large temp passes while compact discovery fails materially, or
- large temp reaches 95% using <=50% of compact median steps/examples with no worse seed reliability.

This is empirical support, not theoretical proof of necessity.

## Capacity ladder

| Tier | Approx params | Role |
|---|---:|---|
| T0 | 17k–25k | compact control |
| T1 | ~64k | medium |
| T2 | ~128k | overcomplete |
| T3 | ~256k | optional |

Select the smallest robustly successful overcomplete tier.

## Novel-task validity

A task is valid only if:
- existing bank fails,
- permitted composition search fails under declared depth/beam,
- oracle metadata does not leak the solution.

Use >=2 novel tasks; 3+ preferred.

## Compact direct-learning control

For each selected task, train the final <=25k candidate architecture directly from labels under the same discovery data budget.

This distinguishes:
- genuine large-temp discovery advantage,
from
- a task that was already easy for the compact candidate.

## Distillation

1. freeze temporary teacher;
2. generate separate distillation inputs;
3. train compact candidate on teacher function;
4. evaluate on held-out inputs unseen by discovery/distillation training.

Do not initialize candidate by copying/truncating teacher weights unless explicitly testing that ablation.

## Shadow safety

Before promotion:
- canonical operation forgetting <=2pp per operation,
- representative composition forgetting <=2pp,
- no Stable Core change,
- no existing primitive change.

## Persistent-only recurrence

After promotion:
1. release temp,
2. save persistent state,
3. start fresh runtime,
4. load persistent Core + Bank only,
5. rerun recurring task,
6. require zero adaptation.
