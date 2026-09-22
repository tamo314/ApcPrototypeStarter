"""Strict JSON bundles: separate build/evaluate, no pickle or training fallback."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from apc.embodied.control import (
    LinearMotionPrimitive,
    MotionBank,
    MotionCompositionLibrary,
    MotionRecipe,
)
from apc.embodied.world import SeedSplit, WorldConfig

BASE_COMMIT = "319f3af8b16accf7347574247e4eae6ab30ad402"
SCHEMA = "apc.embodied.linear-body-frame.v1"


def write_json(path: Path, data: Any) -> None:
    """Exclusive creation protects prior measurements and bundles."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parse_splits(data: Mapping[str, Any]) -> SeedSplit:
    if set(data) != {"train", "validation", "test"}:
        raise ValueError("Splits require train, validation and test")
    return SeedSplit(*(tuple(data[key]) for key in ("train", "validation", "test")))


def load_config(path: Path) -> tuple[WorldConfig, SeedSplit]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"version", "world", "splits"} or data["version"] != 1:
        raise ValueError("Unknown experiment configuration schema")
    return WorldConfig(**data["world"]), parse_splits(data["splits"])


def save_bundle(path: Path, config: WorldConfig, splits: SeedSplit,
                bank: MotionBank, library: MotionCompositionLibrary,
                exposure_hashes: list[str]) -> None:
    write_json(path, {"schema": SCHEMA, "base_commit": BASE_COMMIT,
                      "world": config.to_dict(),
                      "splits": {"train": list(splits.train),
                                 "validation": list(splits.validation), "test": list(splits.test)},
                      "primitives": bank.records(), "recipes": library.records(),
                      "exposure_hashes": sorted(set(exposure_hashes))})


def load_bundle(path: Path, *, backend: str = "apc"
                ) -> tuple[
                    WorldConfig, SeedSplit, MotionBank, MotionCompositionLibrary, set[str]
                ]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema", "base_commit", "world", "splits", "primitives", "recipes", "exposure_hashes"}
    if set(data) != fields or data["schema"] != SCHEMA:
        raise ValueError("Unsupported or malformed embodied bundle")
    config = WorldConfig(**data["world"])
    splits = parse_splits(data["splits"])
    bank = MotionBank(backend)
    for record in data["primitives"]:
        if set(record) != {"name", "weights", "sha256"}:
            raise ValueError("Malformed primitive record")
        policy = LinearMotionPrimitive(record["name"], record["weights"])
        if policy.fingerprint() != record["sha256"]:
            raise ValueError("Primitive weight checksum mismatch")
        bank.add(policy)
    if not bank.names():
        raise ValueError("Bundle has no primitives")
    library = MotionCompositionLibrary()
    for record in data["recipes"]:
        if set(record) != {"name", "operations"}:
            raise ValueError("Malformed recipe record")
        recipe = MotionRecipe(record["name"], tuple(record["operations"]))
        if any(operation not in bank.names() for operation in recipe.operations):
            raise ValueError("Recipe references an unknown primitive")
        library.add(recipe)
    if not library.recipes():
        raise ValueError("Bundle has no recipes")
    hashes = data["exposure_hashes"]
    if (not isinstance(hashes, list) or not hashes
            or any(not isinstance(h, str) or len(h) != 64
                   or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
        raise ValueError("Malformed or missing exposure hashes")
    return config, splits, bank, library, set(hashes)
