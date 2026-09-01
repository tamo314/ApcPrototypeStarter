# Parameterized Primitive Retry Design

## 1. Failure being investigated

The failed A1-R005 run showed:

```text
Correct         0.308
Wrong argument  0.255
Wrong family    0.028
None            0.143
```

The current architecture can distinguish primitive families, but argument-conditioned behavior is weak.

## 2. Candidate failure modes

### F1 — Wrong-argument control is partially non-causal

A changed argument can sometimes leave the target unchanged.

### F2 — SELECT encoder destroys order

Mean pooling of index embeddings is permutation invariant and therefore incompatible with ordered index sequences.

### F3 — Additive conditioning is too weak

Current conditioned primitive:

`B(phi(A(h) + C(a)))`

may learn family-average behavior while largely ignoring `C(a)`.

### F4 — Training does not force argument use

If each content is usually seen with one argument, the learner can exploit content statistics without learning counterfactual argument sensitivity.

### F5 — Multi-token exact-match penalty

SHIFT and SELECT may look worse in sequence exact match than token-level behavior suggests.

### F6 — Primitive compute type mismatch

SHIFT / SELECT / COUNT / BIND may require argument-conditioned data routing rather than simple feature transformation.

## 3. Revised argument encoding

### Integer arguments

Use explicit legal domains. Avoid silent modulo aliasing in scientific runs unless modulo semantics are intended.

### SELECT indices

Use an order-preserving encoder:
- index embeddings,
- argument-position embeddings,
- small masked sequence encoder,
- fixed-size output.

## 4. Counterfactual argument batches

Construct groups:

```text
same content x
  + arg a1 -> y1
  + arg a2 -> y2
  + arg a3 -> y3
```

where outputs are deliberately different.

Use this first for COUNT.

## 5. Conditioning variants

Only compare after identifiable training is established.

### V0 — Current additive
`B(phi(A(h) + C(a)))`

### V1 — Multiplicative / FiLM
`B(phi(A(h) * gamma(a) + beta(a)))`

### V2 — Low-rank basis modulation
`B diag(g(a)) A h`

### V3 — Tiny cross-attention primitive

Argument representation acts as query/control; content state is key/value.

Relevant especially to SELECT, SHIFT, COUNT, BIND.

Do not implement V3 before simpler diagnostics justify it.

## 6. Per-operation order

1. COUNT
2. BIND
3. SHIFT
4. SELECT

COUNT/BIND isolate single-token parameterized behavior. SHIFT/SELECT test sequence routing.

## 7. Retry gate

Final retry must show:

- correct family + correct argument high,
- effectful wrong argument low,
- wrong family low,
- none low,
- persistent family count independent of argument values.
