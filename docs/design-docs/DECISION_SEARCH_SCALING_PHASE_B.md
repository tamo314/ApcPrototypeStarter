# Design — Decision and Composition Search Scaling for Phase B

## Purpose

Phase A.2 established sparse final primitive execution and measured end-to-end sparse-vs-dense-all-bank behavior, but Phase B must account for the cost of finding and verifying the computation.

---

## 1. Cost decomposition

Track:

```text
C_decision
  = C_task_side
  + C_proposal
  + C_direct_verify
  + C_compose_search
  + C_controller
```

and separately:

```text
C_final_execution
```

End-to-end:

```text
C_total = C_decision + C_final_execution
```

Do not infer one from another.

---

## 2. Exhaustive reference

The current exhaustive or broad search procedure is the reference for:

- best direct candidate;
- best functional recipe;
- controller action.

It is not necessarily the production target.

---

## 3. Bounded search

Use the smallest mechanism that can preserve exhaustive decisions.

Preferred order:

1. router/retrieval top-k proposal;
2. bounded direct verification;
3. bounded beam composition;
4. recipe cache for previously verified recurrence/composition;
5. learned proposal only if the simpler proposal is empirically insufficient.

Avoid introducing a large learned search model before proving the smaller mechanism fails.

---

## 4. Candidate budgets

Budgets must be explicit config fields.

Recommended initial Phase B gate values:

```text
K_direct: 8
beam_width: 8
max_depth: 3
max_recipe_evaluations: 64
```

A run must log actual counts.

---

## 5. Composition beam

A beam state should contain enough information to reproduce/evaluate the candidate:

- ordered primitive calls;
- arguments;
- accumulated score;
- functional score on verification examples;
- depth;
- cached intermediate state only if it does not violate task/content invariants.

Do not rerun a task-conditioned Stable Core between primitive steps.

---

## 6. Cache semantics

A cached recipe is reusable only when:

- its task conditions are identifiable;
- its functional adequacy is revalidated when required;
- it does not bypass current arguments;
- it does not become an implicit operation-ID oracle.

Cache hits must be logged.

---

## 7. Resource accounting

For every decision:

- proposed candidate count;
- executed direct candidates;
- generated recipe count;
- executed recipes;
- depth reached;
- primitive forward calls;
- support-example count;
- measured latency;
- FLOPs;
- peak VRAM.

A final top-1 primitive execution is not "sparse" if 128 primitives were executed during verification.
