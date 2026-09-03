# AGENTS Addendum — Shared Queryable Representation Gate

## Current evidence

E-006A showed that jointly training a task-blind encoder with the unchanged compact operator dramatically improves all four parameterized operations.

However, E-006A trained a separate encoder per operation.

This gate tests whether APC can use **one shared Stable Core** across operation families.

## Hard invariant

There must be exactly **one shared content encoder**.

The encoder sees content only.

It must never receive:
- operation ID,
- task tokens,
- argument tokens,
- oracle operation metadata.

## Forbidden hidden specialization

Do not use:
- operation-specific encoder adapters,
- operation-specific encoder LayerNorms,
- operation-switched encoder weights,
- task-specific learned content prefixes,
- operation-conditioned masks/dropout.

## Allowed specialization

After the shared encoder, operation-specific components are allowed:
- compact operator,
- argument encoder,
- readout head.

Oracle operation selection is allowed in this diagnostic.

## Mixed-operation training

The shared encoder must receive gradients from all four operations.

Use balanced or explicitly weighted sampling and report the realized proportions.

Track per-operation loss and, if practical, encoder gradient norm by operation.

## Historical controls

Compare against:
1. E-005 Frozen + Compact,
2. E-004 Frozen + High-capacity,
3. E-006A Per-operation Joint + Compact.

The question is whether sharing preserves most of E-006A's gain.

## Branch interpretation

- broad success -> Branch B strongly supported
- broad collapse -> operation-specific representation specialization remains important
- success except SHIFT -> Branch B + heterogeneous SHIFT operator

## Blocking rule

A1-R006 remains blocked.
