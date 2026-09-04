# A1-B007X Discovery-to-Compact Consolidation Gate — Patch Guide

## Placement

Insert this gate **after A1-B007 (Recurrence and Bank Reuse) passes** and **before learned retrieval/router, learned novelty detection, large-bank scaling, or the full K/C/N/R closed loop**.

B006 established functional consolidation, but temporary and persistent capacity were approximately equal (~17k -> ~17k), so it did not establish parameter compression or a discovery-capacity gap.

The new gate directly tests:

`P_temp >> P_persistent`

while preserving the discovered function.

## Add

- `docs/exec-plans/active/A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/CODEX_TASKS_A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/EXPERIMENT_PLAN_A1_B007X_DISCOVERY_COMPRESSION.md`
- `docs/design-docs/DISCOVERY_CAPACITY_COMPRESSION.md`
- `docs/AGENTS_A1_B007X_DISCOVERY_COMPRESSION_ADDENDUM.md`

## Existing roadmap edit

Insert:

```md
A1-B007 -> A1-B007X -> downstream learned retrieval/router/novelty work
```

Do not renumber historical tasks.

## Recommended Codex prompt

```text
Read AGENTS.md and all active Branch B documents.
Then read:
- docs/exec-plans/active/A1_B007X_DISCOVERY_COMPRESSION.md
- docs/CODEX_TASKS_A1_B007X_DISCOVERY_COMPRESSION.md
- docs/EXPERIMENT_PLAN_A1_B007X_DISCOVERY_COMPRESSION.md
- docs/design-docs/DISCOVERY_CAPACITY_COMPRESSION.md
- docs/AGENTS_A1_B007X_DISCOVERY_COMPRESSION_ADDENDUM.md

Implement only Task A1-B007X-001.

Prerequisite: A1-B007 must already have passed.
Do not begin learned retrieval/router/novelty tasks.
Preserve B005/B006/B007 artifacts unchanged.
Treat B006 as functional consolidation, not parameter compression.
```
