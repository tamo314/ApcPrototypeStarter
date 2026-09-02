# Representation / Operator Isolation

## 1. Motivation

Parameterized primitive retry failed across COUNT, BIND, SHIFT, and SELECT.

The current primitive family is approximately pointwise:

`h'_j = f(h_j, argument)`

while the failed operations require argument-conditioned cross-position computation:
- COUNT: query-conditioned aggregation,
- BIND: keyed retrieval,
- SHIFT: position permutation,
- SELECT: indexed gather.

The unresolved question is whether the bottleneck is the operator class or the frozen content representation supplied to it.

## 2. Information-path decomposition

```text
content
   |
   v
Frozen task-blind Content Encoder
   |
h_content
   |
   +--------------------------+
                              |
argument                      v
   |                  diagnostic operator
   +--------------------------+
                              |
                              v
                            output
```

The encoder stays frozen for the first diagnostics.

## 3. Representation sufficiency levels

### R0 — Raw token recoverability
Can each input token be reconstructed from its corresponding hidden state?

### R1 — Position recoverability
Can absolute/relative position be reconstructed?

### R2 — Structured-role recoverability
For BIND-like content, can a probe recover key/value role and pair/adjacency information?

### R3 — Task upper bound
Can a sufficiently expressive operator solve the parameterized task from frozen `h_content + argument`?

R3 is the most important test.

## 4. Reconstruction is not enough

High token/position probe accuracy proves information retention, not accessibility to a compact primitive.

Always follow probe diagnostics with a strong upper-bound operator.

## 5. High-capacity upper-bound operator

Use an intentionally expressive module:
- 2–4 Transformer/cross-attention blocks,
- sufficient width to remove obvious operator-capacity concerns.

Stable Core remains frozen.

Input:
- full frozen content sequence,
- explicit argument.

Forbidden:
- target tokens,
- oracle output,
- K/C/N/R metadata.

The module may be much larger than a legal APC primitive.

## 6. Oracle latent operators

Use deterministic addressing to test whether hidden states/decoder support the target.

### SHIFT
Permute hidden-state positions by oracle `amount`, then decode.

### SELECT
Gather hidden-state positions by oracle ordered `indices`, then decode.

### BIND
Use raw content symbols only to locate queried key; gather/read associated value hidden state and decode.

### COUNT
Use raw content symbols only to construct oracle target-match mask; aggregate matched positions with a documented deterministic or small diagnostic count readout.

These are diagnostics, not learned primitive results.

## 7. Compact operator probe

Only after a strong frozen upper bound succeeds, test a small cross-position operator:

```text
argument -> query/control
h_content -> keys/values
query-conditioned attention/routing
-> small output transform
```

This is a diagnostic prototype, not a full Primitive Bank redesign.

## 8. Queryable representation control

If frozen upper bound fails, jointly train:
- task-blind content encoder,
- strong argument-conditioned operator.

Task/argument enters only the operator.

If this succeeds while frozen upper bound fails, representation learning is the bottleneck.

If this also fails, revisit causal factorization/decoder/state layout.

## 9. Possible conclusions

- Operator branch: representation sufficient; design heterogeneous/cross-position primitives.
- Representation branch: frozen representation insufficient; design queryable task-blind representation learning.
- Interface branch: even joint task-blind strong model fails; revisit causal architecture.
- Mixed branch: different operations require different diagnoses.
