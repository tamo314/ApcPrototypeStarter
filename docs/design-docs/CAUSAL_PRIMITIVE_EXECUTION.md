# Causal Primitive Execution Architecture

## 1. Problem

The shared Stable Core already solves the eight known operations at near-perfect accuracy when task specification is visible.

Therefore this architecture is scientifically ambiguous:

```text
task + content
    -> Stable Core
    -> primitive
    -> output
```

The answer may already be encoded before the primitive executes.

## 2. Required factorization

```text
task specification
       |
       v
 Task Encoder
       |
     z_task
       |
 router / oracle
       |
 PrimitiveCall
       |
       +--------------------+
                            |
content                     v
  |                   selected primitive
  v                         |
Content Encoder             |
  |                         |
h_content ------------------+
            |
            v
     transformed state
            |
            v
         decoder
```

The key invariant is:

`h_content = f(content)`

not:

`h_content = f(task, content)`.

## 3. Weight sharing

A minimal implementation may use one shared Transformer module twice:

```text
Shared weights(task-only input)    -> z_task
Shared weights(content-only input) -> h_content
```

Separate weights are not required initially.

## 4. Counterfactual invariance

For fixed content `x` and different tasks `t1`, `t2`, ...:

`h_content(x, t1) == h_content(x, t2)`

up to predeclared numerical tolerance.

This is stronger than a leakage probe and is a required gate.

## 5. Primitive execution

### Parameter-free

`P_i(h)` for:
- COPY
- NEGATE
- COMPARE
- ACCUMULATE

### Parameterized

`P_i(h, a)` for:
- SHIFT(amount)
- SELECT(indices)
- COUNT(target)
- BIND(query_key)

## 6. Minimal argument conditioning

Preferred first implementation:

```text
argument -> embedding e_a
u = A_i h
c = C_i e_a
delta = B_i phi(u + c)
out = h + delta
```

A FiLM-style alternative is acceptable.

Do not start with a hypernetwork.

## 7. Decoder leakage control

The causal decoder must not receive task specification through a bypass that lets it solve the operation without primitives.

If formatting-only task information is required, isolate and document it.

## 8. Causal ablation matrix

Every primitive gate evaluates:

| Condition | Meaning |
|---|---|
| Correct | correct primitive and arguments |
| Wrong | incorrect primitive family |
| None | identity/no primitive |
| Wrong argument | correct family, incorrect argument |
| Optional shuffled | primitive call from another example |

A primitive is causally supported only if performance depends strongly on Correct.

## 9. Composition

```text
h0 = ContentEncoder(content)
h1 = P_a(h0, args_a)
h2 = P_b(h1, args_b)
...
```

Do not re-run task-conditioned Stable Core computation between primitive steps.

## 10. Residual Plastic Workspace

For a novel operation:

```text
existing_output = best_existing_recipe(h)
temporary learns only the residual needed beyond existing_output
```

## 11. Consolidation

Fit a compact primitive to temporary residual behavior over held-out probe states.

Target the function, not temporary weights.

## 12. Recurrence

On recurrence:
1. retrieve/select existing primitive,
2. execute without temporary allocation,
3. expand only if the current bank is demonstrably insufficient.

## 13. Learned router

Consumes `z_task`; no oracle metadata at inference.

May output primitive family, arguments/argument embedding, or recipe ID.

## 14. Learned novelty

Novelty represents lack of adequate computation in the bank.

Preferred signals:
- residual error after best recipe,
- retrieval confidence,
- optional gradient/subspace residual.

## 15. Core success pattern

```text
Correct primitive -> high accuracy
Wrong primitive   -> low accuracy
No primitive      -> low accuracy
```

Without this pattern, later composition/consolidation results are not interpretable as primitive computation.
