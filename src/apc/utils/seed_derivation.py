"""Deterministic, version-tagged seed derivation for benchmark example generation.

Fixes the nondeterminism documented in ADR-0080 (Task B-C005D2-005): the
original per-example RNG seed for benchmark generation included
``hash(operation) % 10000``. Python randomizes string hashing per process by
default (no repository run script pinned ``PYTHONHASHSEED``), so the exact
same config/seed produced *different* examples from a fresh process. This
module replaces that with a single canonical-JSON + SHA256 derivation that is
stable across processes, ``PYTHONHASHSEED`` values, call/iteration order, and
worker partitioning, because each example's seed depends only on named,
serialized fields -- never on Python object identity, string-hash
randomization, set-iteration order, or wall-clock time.

Task B-C005R3-001 (`docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`) scope:
this module only changes seed derivation. It must not be imported by, or
imply any change to, router/ArgumentScorer/verifier/controller/primitive
code or model weights.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

# `generator_version` values recognized by `derive_seed`/`generate_benchmark_examples`.
# Any new derivation scheme must add a new version string here rather than
# silently changing what an existing version string produces (design doc
# ``B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md`` S2.3).
GENERATOR_VERSION_V1_LEGACY: Final[str] = "v1_legacy_hash"
GENERATOR_VERSION_V2: Final[str] = "v2_sha256_indexed"
DEFAULT_GENERATOR_VERSION: Final[str] = GENERATOR_VERSION_V2
KNOWN_GENERATOR_VERSIONS: Final[tuple[str, ...]] = (
    GENERATOR_VERSION_V1_LEGACY,
    GENERATOR_VERSION_V2,
)

# Version of the task/operation *semantics* consumed by seed derivation
# (independent of code-version/commit). Bump only if the meaning of
# `task_key` (operation name) -> generation recipe changes in a way that
# should invalidate seed-derived comparability with prior v2 runs.
TASK_SEMANTICS_VERSION: Final[str] = "v1"

# SHA256 digest -> first 8 bytes -> unsigned 64-bit int -> mask to 63 bits.
# 63 bits (not 64) keeps the value a plain positive Python int well within
# `random.Random`'s accepted range on every supported platform, and matches
# the design doc's ``first_63_bits(SHA256(...))`` contract exactly.
_SEED_MASK_63_BIT: Final[int] = (1 << 63) - 1


def derive_seed(
    *,
    master_seed: int,
    stream_namespace: str,
    task_key: str,
    sample_index: int,
    generator_version: str = DEFAULT_GENERATOR_VERSION,
    task_semantics_version: str = TASK_SEMANTICS_VERSION,
) -> int:
    """Derive one deterministic per-example RNG seed from a canonical payload.

    The returned seed is a pure function of
    ``(generator_version, master_seed, stream_namespace, task_semantics_version,
    task_key, sample_index)``. In particular it does not depend on:

    - Python's per-process string-hash randomization (``PYTHONHASHSEED``),
    - the order in which examples/cells are generated or iterated,
    - where generation is resumed from,
    - how many workers split the work,
    - object identity or wall-clock time.

    Args:
        master_seed: The caller's top-level data seed (kept separate from any
            model-initialization seed by convention -- this function has no
            way to enforce that separation, callers must not reuse a model
            seed here).
        stream_namespace: Logical stream this example belongs to (e.g. the
            existing ``split`` values ``"train"``/``"test"``/``"adapt"``, or a
            future dedicated namespace such as ``"reference"``/``"probe"``).
            Two different namespaces with all other fields equal never
            collide by construction (the namespace string is part of the
            hashed payload).
        task_key: Generator-internal task identifier (the operation name).
            Never pass this into any model input.
        sample_index: The example's stable position in its
            (master_seed, stream_namespace, task_key) stream. Two calls with
            the same `sample_index` and other fields always produce the same
            seed, regardless of what other indices have or have not been
            generated yet -- this is what makes resumption, reversed
            iteration, and worker-count changes reproduce identical
            per-index examples.
        generator_version: Must be `GENERATOR_VERSION_V2`; other values raise
            `ValueError` (use `derive_seed_v1_legacy` explicitly for the old
            formula instead of overloading this function).
        task_semantics_version: See `TASK_SEMANTICS_VERSION`.

    Returns:
        A non-negative integer in `[0, 2**63)` suitable as a `random.Random` seed.
    """
    if generator_version != GENERATOR_VERSION_V2:
        raise ValueError(
            f"derive_seed implements only {GENERATOR_VERSION_V2!r}; got "
            f"generator_version={generator_version!r}. For the pre-fix "
            "formula, call derive_seed_v1_legacy explicitly -- it is "
            "diagnostic-only and not guaranteed stable across processes."
        )
    canonical_payload = {
        "generator_version": generator_version,
        "master_seed": int(master_seed),
        "sample_index": int(sample_index),
        "stream_namespace": str(stream_namespace),
        "task_key": str(task_key),
        "task_semantics_version": str(task_semantics_version),
    }
    canonical_bytes = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).digest()
    as_uint64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return as_uint64 & _SEED_MASK_63_BIT


def legacy_split_salt(split: str) -> int:
    """Return the pre-existing per-split salt used by the v1 formula.

    Kept as a named function (rather than inlined) so both the legacy path
    and any future namespace-derivation code reference one definition.
    """
    if split == "test":
        return 2000
    if split == "train":
        return 1000
    return 500


def derive_seed_v1_legacy(*, master_seed: int, salt: int, task_key: str) -> int:
    """Reproduce the exact pre-ADR-0080-fix seed formula, diagnostic-only.

    This is ``master_seed * 7919 + salt + (hash(task_key) % 10000)``, byte
    for byte what `recurrence_benchmark.py`/`consolidation_benchmark.py`
    computed before this task. Because Python's `hash()` on `str` is
    randomized per process unless `PYTHONHASHSEED` is pinned identically in
    both processes, **this function's return value is not guaranteed equal
    across processes**. It exists only so a historical run can be
    diagnostically reconstructed under a pinned `PYTHONHASHSEED` (as ADR-0080
    itself did); no new benchmark generation may default to it, and any run
    that uses it must record `generator_version=GENERATOR_VERSION_V1_LEGACY`
    plus the `PYTHONHASHSEED` value used.
    """
    return master_seed * 7919 + salt + (hash(task_key) % 10000)
