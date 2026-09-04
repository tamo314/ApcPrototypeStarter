# AGENTS Addendum — A1-B007X Discovery-to-Compact Consolidation

## Purpose

Test the APC central hypothesis more directly:

> computation discovery can benefit from substantially larger temporary capacity than the capacity needed to store and execute the discovered computation.

## Separate the claims

### C1 — Functional consolidation
Temporary solution -> persistent primitive.

Already supported by B006.

### C2 — Compressibility
A much larger temporary solution can be distilled into a much smaller persistent primitive.

Tested here.

### C3 — Discovery-capacity advantage
The larger temporary system succeeds more reliably or with materially lower learning cost than a compact system under matched conditions.

Tested separately here.

Passing C2 does not automatically prove C3.

## No result inflation

If a 128k temporary operator compresses to 18k but an 18k operator learns the task equally well under the same budget, report:

> compression demonstrated; discovery-capacity advantage not demonstrated.

## Frozen invariants

During discovery and consolidation:
- Stable Core frozen.
- Existing PrimitiveBank frozen.
- Composition search runs before plastic allocation.
- No persistent bank growth before shadow-validation pass.
- Temporary parameters isolated.

## Capacity ladder

Preferred, subject to hardware:
- compact: ~17k–25k
- medium: ~64k
- large: ~128k
- optional extra-large: ~256k

At least one temporary tier must be >=4x the persistent candidate budget.

Keep topology comparable where practical; document topology changes as confounds.

## Persistent candidate

Target `<=25,000 parameters`.

## Shadow safety

Report historical before/after metrics per canonical operation and representative composition, not only one aggregate.

## Persistent-only recurrence

After promotion:
- release workspace,
- serialize persistent state,
- create fresh runtime,
- load Core + Bank only,
- verify recurrence with zero adaptation.

## Final verdict categories

1. Strong capacity-gap support
2. Compressibility only
3. No compression support
4. Inconclusive
