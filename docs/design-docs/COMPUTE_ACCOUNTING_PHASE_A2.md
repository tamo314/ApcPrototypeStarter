# Phase A.2 Compute Accounting

## 1. Why

B008 demonstrated 90.11% reduction in active primitive parameters relative to resident primitive parameters.

That is not identical to 90.11% end-to-end compute savings.

Phase A.2 must measure both structural sparsity and real execution cost.

## 2. Required accounting

For every scaling benchmark report:

### Parameters

- Stable Core params
- router params
- resident primitive params
- active primitive params
- temporary params

### FLOPs / operation estimates

- task encoder/router
- content Stable Core
- selected primitive
- decoder/readout
- total sparse path
- dense-all-primitives baseline

### Runtime

- median latency
- p95 latency
- throughput
- peak GPU memory
- optional CPU overhead

Use warmup and repeated measurements.

## 3. Dense baseline

Dense primitive baseline must actually execute all resident primitive modules.

Do not estimate dense latency by multiplying one primitive latency if an executable dense baseline is feasible.

## 4. Sparse invariant

For top-1 routing:

`selected primitive calls == batch/task calls`

`unselected primitive calls == 0`

## 5. Scaling interpretation

Expected structural behavior:

- resident params ~O(N)
- active primitive params ~O(1)
- primitive forward FLOPs ~O(1)
- router scoring cost may grow with N

Therefore total system latency may not be perfectly constant. Report the router-growth cost explicitly.

## 6. Performance target vs scientific gate

Hard scientific gates:

- sparse execution correctness,
- zero unselected calls,
- accuracy retention.

Latency/FLOPs ratios are primarily measured outcomes.

A practical target at N=128 is:

`sparse total latency <=30% of dense-all-primitives latency`

but missing this target is not by itself evidence that APC's functional claims fail.
